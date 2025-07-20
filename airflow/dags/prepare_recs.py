#dags/prepare_recs.py

# Регулярное переформирование оффлайн-рекомендаций на основании поступающих из внешнего источника
# обновленных исходных данных для рекомендательной системы
# --------------------------------------------------------------------------------------------------
# --- Время запуска синхронизировано (с лагом 30 минут) со временем загрузок item_properties.csv ---
# --------------------------------------------------------------------------------------------------

import pendulum
from airflow.decorators import dag, task
from steps.messages     import send_telegram_success_message, send_telegram_failure_message

@dag(
    dag_id='recsys_data',
    schedule='@weekly',
    start_date=pendulum.datetime(2015,9,13, 3,30,0, tz="UTC"),
    catchup=False,
    #on_success_callback=send_telegram_success_message,
    #on_failure_callback=send_telegram_failure_message,
    tags=["RECSYS"]
)
def prepare_recs():
    import data_handlers as dh
    import logging, os
    from   airflow.models.connection          import Connection
    from   airflow.utils.session              import create_session
    from   airflow.providers.amazon.aws.fs.s3 import get_fs
    from   dotenv     import find_dotenv, load_dotenv
    load_dotenv(find_dotenv("env.services"))

    S3_DIR          = f"{os.environ['S3_BUCKET_NAME']}/Diplom"
    AWS_ACCESS_KEY  = os.environ['AWS_ACCESS_KEY_ID']
    AWS_SECRET_KEY  = os.environ['AWS_SECRET_ACCESS_KEY']
    S3_ENDPOINT_URL = os.environ['AWS_ENDPOINT_URL']
    MY_HOST         = dh.host_ip()  if not "HOST_IP" in os.environ  else os.environ['HOST_IP']

    # параметры конфигурации DAG-а
    CONFIG      = {
        "EVENT_HISTORY_WEEKS": 26,  # сохранять events только за последние подгода
        "EVENT_POPULAR_WEEKS": 12,  # глубина истории (в неделях) для определения популярных товаров
        "EVENT_TARGET_WEEKS" : 2,   # количество недель для target-периода (в режиме переобучения моделей)
        "EVENT_TEST_WEEKS"   : 1,   # количество недель для test-периода (в режиме переобучения моделей)
        "EVENT_CUT_OFF_WEEKS": 4,   # количество недель для inference-периода (в режиме расчета рекомендаций)
        "ALS_RECS_PER_USER"  : 15,  # количество коллаборативных рекомендаций на пользователя
        "ALS_SIMS_PER_ITEM"  : 15,  # количество подобных товаров на основе коллаборативных рекомендаций
        "MAX_RECS_PER_USER"  : 100, # максимальное количество финальных рекомендаций на пользователя
        "EXPERIMENT_NAME"    : f"{os.environ['MLFLOW_EXPERIMENT_NAME']}",
        "MODEL_NAME"         : f"{os.environ['MLFLOW_MODEL_NAME']}",
        "MLFLOW_SERVER_URL"  : f"http://{MY_HOST}:{os.environ['MLFLOW_SERVER_PORT']}",
        "AUTONOMOUS_MODE"    : f"{os.environ['AUTONOMOUS_MODE']}"
    }

    # используемые ресурсы
    SRC_FILES = {
        "cats_src" : f"{S3_DIR}/source_data/category_tree.csv",
        "props_src":[f"{S3_DIR}/source_data/item_properties_part1.csv",
                     f"{S3_DIR}/source_data/item_properties_part2.csv"],
        "event_src": f"{S3_DIR}/source_data/events.csv"
    }
    INFER_FILES = {
        "cats_dst" : f"{S3_DIR}/infer_data/category_tree.parquet",
        "item_cat" : f"{S3_DIR}/infer_data/item_categories.parquet",
        "item_prop": f"{S3_DIR}/infer_data/item_properties.parquet",
        "available": f"{S3_DIR}/infer_data/item_availability.parquet",
        "event_dst": f"{S3_DIR}/infer_data/events.parquet",
        "eventlast": f"{S3_DIR}/infer_data/last_events.parquet",
    }
    PROD_FILES = {
        "top_pop"  : f"{S3_DIR}/recommendations/top_popular.parquet",
        "similar"  : f"{S3_DIR}/recommendations/similar_items.parquet",
        "ranked"   : f"{S3_DIR}/recommendations/ranked_candidades.parquet",
        "final"    : f"{S3_DIR}/recommendations/final_recommendations.parquet"
    }
    MODEL_FILES = {
        "als_parms": f"{S3_DIR}/model/als_params.pkl",
        "cb_parms" : f"{S3_DIR}/model/cb_params.pkl",
        "cb_model" : f"{S3_DIR}/model/cb_model.pkl"
    }
    MODEL_RETRAIN = {
        "als_parms": f"{S3_DIR}/model_retrained/als_params.pkl",
        "cb_parms" : f"{S3_DIR}/model_retrained/cb_params.pkl",
        "cb_model" : f"{S3_DIR}/model_retrained/cb_model.pkl"
    }
    REC_SERVICES= {
        "rec_serv" : f"http://{MY_HOST}:{os.environ['RECOMMENDATIONS_PORT']}",
        "features" : f"http://{MY_HOST}:{os.environ['FEATURES_STORE_PORT']}",
        "events"   : f"http://{MY_HOST}:{os.environ['EVENTS_STORE_PORT']}",
    }

    @task()
    def make_s3_connection(**kwargs):
        import json

        CONN_ID = "aws_cloud"
        while True:
            try:
                s3 = get_fs(CONN_ID)
                if s3.exists(S3_DIR): break
            except:
                aws_connection = Connection(conn_id=CONN_ID, conn_type="aws")
                aws_connection.extra = json.dumps({
                    "aws_access_key_id":     f"{AWS_ACCESS_KEY}",
                    "aws_secret_access_key": f"{AWS_SECRET_KEY}",
                    "endpoint_url":          f"{S3_ENDPOINT_URL}"
                })
                with create_session() as session:
                    session.add(aws_connection)
                    session.commit()
        logging.info(f"AWS connection '{CONN_ID}' created and tested")    
        return  CONN_ID
    

    @task()
    def prepare_data(conn_id, **kwargs):
        return dh.prepare_infer_data(get_fs(conn_id), SRC_FILES, INFER_FILES, CONFIG)
    

    @task()
    def pre_retrain_models(conn_id, **kwargs):
        return True  if  CONFIG['AUTONOMOUS_MODE'] != 'ON'  else \
               dh.calc_recs(get_fs(conn_id), MODEL_FILES, INFER_FILES, PROD_FILES, CONFIG, retrain=True)


    @task()
    def calc_recommendations(conn_id, **kwargs):
        return dh.calc_recs(get_fs(conn_id), MODEL_FILES, INFER_FILES, PROD_FILES, CONFIG)


    @task()
    def reload_data(conn_id, **kwargs):
        return dh.reload_data_files(get_fs(conn_id), REC_SERVICES)


    @task()
    def retrain_models(conn_id, **kwargs):
        return True  if  CONFIG['AUTONOMOUS_MODE'] == 'ON'  else \
               dh.calc_recs(get_fs(conn_id), MODEL_RETRAIN, INFER_FILES, PROD_FILES, CONFIG, retrain=True)


    @task()
    def clear_s3_connection(conn_id, **kwargs):
        try:
            with create_session() as session:
                session.delete(Connection.get_connection_from_secrets(conn_id))
                session.commit()
            logging.info(f"AWS connection '{conn_id}' deleted")
        except:
            logging.error(f"Error on deletion AWS connection '{conn_id}'")
        return


    conn_id = make_s3_connection()
    prepare_data(conn_id) >> pre_retrain_models(conn_id) >> calc_recommendations(conn_id) >> \
    reload_data(conn_id)  >> retrain_models(conn_id)     >> clear_s3_connection(conn_id)

prepare_recs()
