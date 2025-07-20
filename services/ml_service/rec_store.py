import logging
import pandas as pd
import common_utils   as cu

DEBUG_MODE = False

class Recommendations:

    def __init__(self, path_personal, path_popular, logger_name=None):

        self.logger = logging.getLogger(logger_name)
        self._recs  = {"personal": None,           "default": None}
        self._path  = {"personal": path_personal,  "default": path_popular}
        self._stats = {"request_personal_count": 0,"request_default_count": 0}

    
    def load(self, s3, type):
        """
        Загружает рекомендации из файла
        """
        if  (tmp := cu.load_parquet_file(s3, self._path[type], logger=self.logger)) is not None:
            self._recs[type] = tmp
            if type == "personal":
                self._recs[type].set_index('visitorid', inplace=True)
            self.logger.info (f"File: {self._path[type]} loaded")
        return (tmp is not None)

    
    def get(self, user_id: int, k: int=10):
        """
        Возвращает список офлайн-рекомендаций для пользователя
        """
        if DEBUG_MODE:
            nrecs= self._recs["personal"].reset_index().query("visitorid==@user_id").shape[0]
            logging.info(f"Found {nrecs} personal recommendations for user {user_id}")

        try:                 # Отбираем персональные рекомендации
            recs = self._recs["personal"].loc[user_id,'itemid'].to_list()[:k]
            self._stats["request_personal_count"] += len(recs)
        except:
            recs = []

        if len(recs) < k:    # Если персональных мало - "добиваем" популярными
            def_recs = self._recs["default"]
            delta    = min(k-len(recs), def_recs.shape[0])
            recs    += def_recs.sample(delta, random_state=None)["itemid"].to_list()
            self._stats["request_default_count"] += delta
        return recs

    
    def stats(self, log=True):
        """
        Возвращает накопленную статистику офлайн-рекомендаций
        """
        if log:
            self.logger.info("Statistic counters for recommendations")
            for name, value in self._stats.items():
                self.logger.info(f"{name:<30} {value} ")
        return  dict(self._stats)  # a NEW dict to avoid access by reference to self._stats

    
    def stats_reset(self):
        """
        Обнуляет статистику офлайн-рекомендаций
        """
        for name, _ in self._stats.items():
            self._stats[name] = 0
        self.logger.warning("Statistic counters for recommendations RESET!")
        return  True


    def test(self, user_id: int, k: int=10):
        self.logger.info(f"TEST user_id: {user_id}, response: {self.get(user_id, k)}")


if __name__ == "__main__":

    # задаем уровень логирования
    logging.basicConfig(level=logging.INFO)
    
    import os, s3fs
    from   dotenv     import find_dotenv, load_dotenv
    load_dotenv()

    RANDOM_STATE = 0 #1
    PROJECT_ROOT = ""  if len(find_dotenv()) <= 5  else  f"{find_dotenv()[0:-5]}"
    S3_DIR       = f"{os.environ['S3_BUCKET_NAME']}/Diplom"

    s3 = s3fs.core.S3FileSystem(
        endpoint_url=os.environ['AWS_ENDPOINT_URL'],
        key=os.environ['AWS_ACCESS_KEY_ID'],
        secret=os.environ['AWS_SECRET_ACCESS_KEY'], cache_regions=True
    )

    # создаём обработчик запросов для API
    handler = Recommendations(
        f"{S3_DIR}/recommendations/final_recommendations.parquet", 
        f"{S3_DIR}/recommendations/top_popular.parquet"
    )

    # загружаем рекомендации
    handler.load(s3,"default")
    handler.load(s3,"personal")

    # загружаем тестовые взаимодействия
    last_events = cu.load_parquet_file(s3, f"{S3_DIR}/infer_data/last_events.parquet")

    # выделяем тестовых пользователей
    N_TEST_USERS= 10
    test_users  = last_events.sample(
        1000, random_state=RANDOM_STATE
    ).groupby('visitorid').sample(
        random_state=RANDOM_STATE
    ).reset_index().head(N_TEST_USERS)['visitorid'].unique()

    # делаем тестовые запросы
    K_RECS_PER_TEST_USER = 60
    for i in test_users:
        handler.test(i, k=K_RECS_PER_TEST_USER)
    handler.test(155, k=K_RECS_PER_TEST_USER)
    handler.test(717032, k=K_RECS_PER_TEST_USER)

    # вывод статистики в лог-файл 
    handler.stats()
