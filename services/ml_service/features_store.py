''' Прототип микросервиса извлечения объектов, подобных заданному '''
import logging, os, s3fs
import pandas         as pd
import numpy          as np
import common_utils   as cu
from   fastapi    import FastAPI
from   contextlib import asynccontextmanager
from   dotenv     import find_dotenv, load_dotenv
load_dotenv(find_dotenv("env.services")), load_dotenv()

PROJECT_ROOT = ""  if len(find_dotenv()) <= 5  else  f"{find_dotenv()[0:-5]}"
S3_DIR       = f"{os.environ['S3_BUCKET_NAME']}/Diplom/recommendations"
logger       = logging.getLogger("uvicorn.error")
logger.setLevel(int(os.environ['LOG_LEVEL']))


s3 = s3fs.core.S3FileSystem(
    endpoint_url=os.environ['AWS_ENDPOINT_URL'],
    key=os.environ['AWS_ACCESS_KEY_ID'],
    secret=os.environ['AWS_SECRET_ACCESS_KEY'], cache_regions=True
)



class SimilarItems:

    def __init__(self, path):
        self._similar_items = None
        self._path = path

    
    def load(self):
        """
        Загружаем и индексируем таблицу подобных объектов
        """
        if  (tmp := cu.load_parquet_file(s3, self._path, logger=logger)) is not None:
            self._similar_items = tmp.copy().set_index("itemid")
            logger.info (f"File: {self._path} loaded")
        return (tmp is not None)
        

    def get(self, item_id: int, k: int = 10):
        """
        Возвращает список похожих объектов
        """
        try:
            return self._similar_items.loc[item_id].head(k)[['sim_itemid','sim_score']].to_dict(orient='list')
        except KeyError:
            logger.warning(f"No similar items found for item {item_id}")
            return {"sim_itemid": [], "sim_score": []}
        except Exception as err:
            logger.error(f"ERROR {err} getting similar items for item {item_id}")
        return {"result": "ERROR"}



# Инициализируем класс выдачи рекомендаций на основе подобия товаров
sim_items_store = SimilarItems(f"{S3_DIR}/similar_items.parquet")

# создаём приложение FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    if  not sim_items_store.load(): return   # вызываем ошибку инициализации
    yield
    logger.info("Service stopped")
app = FastAPI(title="features", lifespan=lifespan)


@app.post("/similar_items")
async def similar_items(item_id: int, k: int = 10):
    """
    Возвращает список похожих объектов длиной k для item_id
    """
    return  sim_items_store.get(item_id, k)


@app.post("/reload")
async def reload():
    """
    Перезагружает файл с данными
    """
    return {"result": "OK" if sim_items_store.load() else "ERROR"}
