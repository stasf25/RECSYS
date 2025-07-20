''' Прототип микросервиса извлечения онлайн-взаимодействий пользователя '''
import logging, os, s3fs
import pandas         as pd
import numpy          as np
import common_utils   as cu
from   fastapi    import FastAPI
from   contextlib import asynccontextmanager
from   dotenv     import find_dotenv, load_dotenv
load_dotenv(find_dotenv("env.services")), load_dotenv()

PROJECT_ROOT = ""  if len(find_dotenv()) <= 5  else  f"{find_dotenv()[0:-5]}"
S3_DIR       = f"{os.environ['S3_BUCKET_NAME']}/Diplom/infer_data"
logger       = logging.getLogger("uvicorn.error")
logger.setLevel(int(os.environ['LOG_LEVEL']))


s3 = s3fs.core.S3FileSystem(
    endpoint_url=os.environ['AWS_ENDPOINT_URL'],
    key=os.environ['AWS_ACCESS_KEY_ID'],
    secret=os.environ['AWS_SECRET_ACCESS_KEY'], cache_regions=True
)


class EventStore:

    def __init__(self, events_path, max_events_per_item=3, max_events_per_user=10, exclude_users: list=[]):

        self.max_events_per_item = max_events_per_item
        self.max_events_per_user = max_events_per_user
        self.exclude_users       = exclude_users
        self._path               = events_path
        self.events              = None
    

    def load(self):
        """
        Загружаем и индексируем DataFrame с тестовыми событиями
        """
        if  (tmp := cu.load_parquet_file(s3, self._path, logger=logger)) is not None:
            self.events = tmp.copy().sort_values(by='timestamp',ascending=False)                    \
                             .groupby(['visitorid','itemid']).head(self.max_events_per_item) \
                             .groupby('visitorid').head(self.max_events_per_user)            \
                             .groupby('visitorid')['itemid'].agg(list)
            logger.info (f"File: {self._path} loaded")
        return (tmp is not None)



    def get(self, user_id, k):
        """
        Возвращает события для пользователя
        """
        try:
            user_events = [] if user_id in self.exclude_users  \
                             else self.events.at[user_id]
        except:
            user_events = []

        logger.info(f"Found {len(user_events)} events for user {user_id}")
        return user_events if k is None else user_events[:k]

    
    def put(self, user_id, item_id):
        """
        Сохраняет событие
        """
        # получаем полный список событий для user_id
        user_events = self.get(user_id, None)
        try:
            # добавляем новое событие в начало списка
            self.events.at[user_id] = list([item_id] + user_events)[:self.max_events_per_user]
            logger.info(f"Registered a new event with item {item_id} for user {user_id}")
            return True
        except:
            logger.error(f"ERROR registering a new event with item {item_id} for user {user_id}")
        return False


# Инициализируем класс и задаем ids "холодных пользователей" для тестов
events_store = EventStore(f"{S3_DIR}/last_events.parquet", exclude_users=[48,83]) 

# создаём приложение FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    if  not events_store.load(): return   # вызываем ошибку инициализации
    yield
    logger.info("Service stopped")
app = FastAPI(title="events", lifespan=lifespan)



@app.post("/put")
async def put(user_id: int, item_id: int):
    """
    Сохраняет событие для user_id, item_id
    """
    return {"result": "OK" if events_store.put(user_id, item_id) else "ERROR"}


@app.post("/get")
async def get(user_id: int, k: int = 10):
    """
    Возвращает список последних k событий для пользователя user_id
    """
    return {"events": events_store.get(user_id, k)}


@app.post("/reload")
async def reload():
    """
    Перезагружает файл с данными
    """
    return {"result": "OK" if events_store.load() else "ERROR"}
