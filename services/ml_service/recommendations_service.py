''' Прототип сервиса рекомендательной системы '''
import logging,requests, os, s3fs
import pandas         as pd
import numpy          as np
import common_utils   as cu
from   rec_store  import Recommendations
from   itertools  import zip_longest
from   fastapi    import FastAPI
from   contextlib import asynccontextmanager
from   dotenv     import find_dotenv, load_dotenv
load_dotenv(find_dotenv("env.services")), load_dotenv()

PROJECT_ROOT = ""  if len(find_dotenv()) <= 5  else  f"{find_dotenv()[0:-5]}"
S3_DIR       = f"{os.environ['S3_BUCKET_NAME']}/Diplom/recommendations"

s3 = s3fs.core.S3FileSystem(
    endpoint_url=os.environ['AWS_ENDPOINT_URL'],
    key=os.environ['AWS_ACCESS_KEY_ID'],
    secret=os.environ['AWS_SECRET_ACCESS_KEY'], cache_regions=True
)

if  'TEST_MODE' in os.environ:
    logging.basicConfig(filename=f"{PROJECT_ROOT}/logs/test_service.log", filemode='w', level=logging.INFO)
    logger = logging.getLogger("rec_service")
    logger.warning("Service is executed in TEST_MODE!")
else:
    logger = logging.getLogger("uvicorn.error")
    logger.setLevel(int(os.environ['LOG_LEVEL']))

features_host = os.environ['FEATURES_DOCKER_HOST' if 'UNDER_DOCKER' in os.environ else 'APP_HOST']
events_host   = os.environ['EVENTS_DOCKER_HOST'   if 'UNDER_DOCKER' in os.environ else 'APP_HOST']
features_store_url = f"http://{features_host}:{os.environ['FEATURES_STORE_PORT']}"
events_store_url   = f"http://{events_host}:{os.environ['EVENTS_STORE_PORT']}"
logger.info(f"Locating features_store service at: {features_store_url}")
logger.info(f"Locating events_store service at: {events_store_url}")

# Инстанциируем класс формирования оффлайн-рекомендаций
rec_store = Recommendations(
    f"{S3_DIR}/final_recommendations.parquet",
    f"{S3_DIR}/top_popular.parquet",
    logger_name=logger.name
)

# создаём приложение FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    if  rec_store.load(s3,"default") and rec_store.load(s3,"personal"):
        logger.info("Service is ready to receive requests")
        yield
        rec_store.stats()
        logger.info("Service stopped")
    return
app = FastAPI(title="recommendations", lifespan=lifespan)
#app.handler = rec_store

# ---------------------------------------------------------------------------------------
#             Инициализируем интерфейс с Prometheus
# ---------------------------------------------------------------------------------------
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from prometheus_client import Counter, Gauge, Histogram
import psutil
psutil.cpu_percent()    # Ignore the first call result

# определяем метрики для мониторинга ошибок
app_exceptions_recstore_total = Counter(
    "app_error_exceptions_recstore_total",
    "Number of rec_store requests terminated with exception",
)
app_exceptions_eventstore_total = Counter(
    "app_error_exceptions_eventstore_total",
    "Number of event_store requests terminated with exception",
)
app_exceptions_featstore_total = Counter(
    "app_error_exceptions_featstore_total",
    "Number of feature_store requests terminated with exception",
)
app_exceptions_recservice_total = Counter(
    "app_error_exceptions_recservice_total",
    "Number of recommendations requests terminated with exception",
)
# определяем метрики для мониторинга рекомендаций
app_online_events_total = Counter(
    "app_online_events_total",
    "Number of online events received from events store",
)
app_similar_items_total = Counter(
    "app_similar_items_total",
    "Number of similar items received from features store",
)
app_als_items_total = Counter(
    "app_als_items_total",
    "Number of personalized recommendations received from recommendations store",
)
app_popular_items_total = Counter(
    "app_popular_items_total",
    "Number of non-personalized recommendations received from recommendations store",
)
app_offline_recs_percent = Histogram(
    "app_offline_recs_percent",
    "Rate of offline recommendations in the blended recommendations set",
    buckets=[ i/10 for i in range(1, 11) ]
)

# Определяем метрики для мониторинга нагрузки
cpu_usage    = Gauge('app_cpu_usage_percent', 'Current CPU usage in percent')
memory_usage = Gauge('app_memory_usage_percent', 'Current RAM usage in percent')

def mw_metrics(info: metrics.Info):
    cpu_usage.set(psutil.cpu_percent())
    memory_usage.set(psutil.virtual_memory().percent)
    return None

# инициализируем и запускаем экпортёр метрик
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)
instrumentator.add(mw_metrics).add(metrics.default())
# ---------------------------------------------------------------------------------------



@app.post("/recommendations_offline")
async def recommendations_offline(user_id: int, k: int = 100):
    """
    Возвращает список офлайн-рекомендаций длиной k для пользователя user_id
    """
    try:
        prev_stat  = rec_store.stats(log=False)            # запоминаем счетчики рекомендаций
        recs       = rec_store.get(user_id, k)             # получаем рекомендации
        stat       = rec_store.stats(log=False)            # новые значения счетчиков
        n_personal = stat["request_personal_count"] - prev_stat["request_personal_count"]
        n_default  = stat["request_default_count"]  - prev_stat["request_default_count"]
        logger.info(
            f"User: {user_id}, Offline recommendations ({n_personal} personal, {n_default} default): {recs}"
        )
        app_als_items_total.inc(n_personal)
        app_popular_items_total.inc(n_default)
    except:
        recs = []
        app_exceptions_recstore_total.inc()
    return {"recs": recs}


def dedup_ids(ids):
    """
    Дедублицирует список идентификаторов, оставляя только первое вхождение
    """
    seen = {None}
    ids  = [id for id in ids if not (id in seen or seen.add(id))]
    return ids


@app.post("/recommendations_online")
async def recommendations_online(user_id: int, k: int = 100, n_last: int = 3):
    """
    Возвращает список онлайн-рекомендаций длиной k для пользователя user_id
    """
    headers = {"Content-type": "application/json", "Accept": "text/plain"}

    # получаем список последних событий пользователя
    try:
        params = {"user_id": user_id, "k": n_last}
        resp   = requests.post(events_store_url + "/get", headers=headers, params=params).json()
        events = resp["events"]
        logger.info(f"User: {user_id}, Online events: {events}")
        app_online_events_total.inc(len(events))
    except Exception as err:
        events = []
        logger.error(f"Error {err} getting online events for user: {user_id}")
        app_exceptions_eventstore_total.inc()

    # получаем список айтемов, похожих на последние три, с которыми взаимодействовал пользователь
    items  = []
    scores = []
    for item_id in events:
        # для каждого item_id получаем список похожих в item_similar_items
        params = {"item_id": item_id, "k": k}
        try:
            resp   = requests.post(features_store_url+"/similar_items", headers=headers,params=params).json()
            items += resp["sim_itemid"]
            scores+= resp["sim_score"]
        except Exception as err:
            logger.error(f"Error {err} getting similar items for item: {item_id}")
            app_exceptions_featstore_total.inc()
    
    # сортируем похожие объекты по scores в убывающем порядке
    combined = list(zip(items, scores))
    combined = sorted(combined, key=lambda x: x[1], reverse=True)
    combined = [item for item, _ in combined]

    # удаляем дубликаты, чтобы не выдавать одинаковые рекомендации
    recs = dedup_ids(combined)[:k]

    logger.info(f"User: {user_id}, Online recommendations: {recs}")
    app_similar_items_total.inc(len(recs))
    return {"recs": recs}


@app.post("/recommendations")
async def recommendations(user_id: int, k: int = 100):
    """
    Возвращает список рекомендаций длиной k для пользователя user_id
    """
    recs_blended = []
    try:
        recs_offline = await recommendations_offline(user_id, k)
        recs_online  = await recommendations_online(user_id, k)

        for odd, even in zip_longest(recs_online["recs"], recs_offline["recs"]):
            recs_blended +=  [odd,even]

        # удаляем дубликаты и оставляем только первые k рекомендаций
        recs_blended = dedup_ids(recs_blended)[:k]
        logger.info(f"User: {user_id}, Blended recommendations: {recs_blended}")
        num_offline = len(set(recs_blended) - set(recs_online["recs"]))
        app_offline_recs_percent.observe(num_offline / len(recs_blended))
    except Exception as err:
        logger.error(f"Error {err} getting recommendations for user: {user_id}")
        app_exceptions_recservice_total.inc()
    return {"recs": recs_blended}


@app.post("/reload")
async def reload():
    """
    Перезагружает файлы с данными
    """
    return {"result": "OK" if rec_store.load(s3,"default") and rec_store.load(s3,"personal") else "ERROR"}


@app.post("/stats")
async def stats():
    """
    Возвращает статистику работы сервиса
    """
    return {"stat": rec_store.stats()}


@app.post("/stats_reset")
async def stats_reset():
    """
    Обнуляет статистику работы сервиса
    """
    return {"result": "OK" if rec_store.stats_reset() else "ERROR"}

