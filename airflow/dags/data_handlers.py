#dags/data_handlers.py
# обработчики входных данных для рекомендательной системы
import logging
import joblib,subprocess,requests,io
import pandas as pd
import numpy  as np
from   s3fs   import S3FileSystem

# -------------------------------------------------
#    системные функции
# -------------------------------------------------

def host_ip():
    ''' Возвращает ip-адрес текущего хоста '''
    return subprocess.run(['curl', 'ifconfig.co/'], capture_output=True, text=True).stdout.strip()

def post_request(url, params=None):
    ''' Формирует POST-запрос к заданному сервису '''
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

    resp = requests.post(url, headers=headers, params=params)

    if resp.status_code == 200:
        recs = resp.json()
    else:
        recs = {"result": "ERROR", "code": f"{resp.status_code}"}
        logging.error(f"Error status code: {resp.status_code} received from {url} ({params})")
    return recs

def pd_info(df: pd.DataFrame):
    ''' Отписывает результат pd.DataFrame.info в строку (для logging) '''
    with io.StringIO() as output:
        df.info(show_counts=True, buf=output)
        return output.getvalue()


 


# -------------------------------------------------
#    функции работы с файлами в хранилище S3
# -------------------------------------------------

def load_csv_files(s3: S3FileSystem, path_list: list):
    try:
        df = pd.DataFrame()
        for path in path_list:
            with s3.open(path, mode='r') as fd:
                df = pd.concat([df, pd.read_csv(fd)], axis=0, ignore_index=True)
        return df
    except Exception as e:
        logging.warning(f"Error {e} loading file: {path}")
    return None

def load_parquet_file(s3: S3FileSystem, path: str):
    try:
        df = pd.DataFrame()
        with s3.open(path, mode='rb') as fd:
            df = pd.read_parquet(fd)
        return df
    except Exception as e:
        logging.error(f"Error {e} loading file: {path}")
    return None

def load_pkl_file(s3: S3FileSystem, path: str):
    try:
        with s3.open(path, mode='rb') as fd:
            obj = joblib.load(fd)
        return obj
    except Exception as e:
        logging.error(f"Error {e} loading file: {path}")
    return None

def save_to_parquet(df, s3: S3FileSystem, path: str, verbose=True):
    try:
        with s3.open(path, mode='wb') as fd:
            df.to_parquet(fd)
        if verbose:  logging.info(f"save_to_parquet:\n{path}\n{pd_info(df)}")
    except Exception as e:
        logging.error(f"Error {e} writing to file: {path}")
    return

def save_to_pkl(obj, s3: S3FileSystem, path: str, verbose=True):
    try:
        with s3.open(path, mode='wb') as fd:
            joblib.dump (obj, fd)
        if verbose:  logging.info(f"save_to_pkl: {path}")
    except Exception as e:
        logging.error(f"Error {e} writing to file: {path}")
    return

def delete_s3_files(s3: S3FileSystem, files: dict):
    for k,path in files.items():
        try:
            s3.rm(path)
        except Exception as e:
            logging.error(f"Error {e} deleting file: {path}")
    return



# -------------------------------------------------
#    функции для извлечения item properties
# -------------------------------------------------

def get_registered_items(timestamp, items_ctgr):
    ''' Возвращает набор itemid, для которых определена корректная категория на заданный момент времени '''
    return  set(items_ctgr.query("timestamp <= @timestamp")['itemid'])

def get_unavailable_items(timestamp, items_avail):
    ''' Возвращает набор itemid, для которых установлен признак available==0 на заданный момент времени '''
    tmp = items_avail.query("timestamp <= @timestamp").drop_duplicates(subset=['itemid'], keep='first')
    return  set(tmp.query("value == '0'")['itemid'])

def get_available_items(timestamp, items_ctgr, items_avail):
    ''' Возвращает набор itemid, доступных на заданный момент времени '''
    return  get_registered_items(timestamp, items_ctgr) - get_unavailable_items(timestamp, items_avail)

def get_item_availability(timestamp, items_ctgr, items_avail):
    ''' Возвращает признак доступности товаров, актуальный на заданный момент времени '''
    lst = get_registered_items(timestamp, items_ctgr)
    tmp = items_avail.query("timestamp <= @timestamp and itemid in @lst") \
                     .drop_duplicates(subset=['itemid'], keep='first').reset_index(drop=True)
    return  tmp[['itemid','value']]

def get_item_category(timestamp, items_ctgr):
    ''' Возвращает категории товаров, актуальные на заданный момент времени '''
    tmp = items_ctgr.query("timestamp <= @timestamp") \
                    .drop_duplicates(subset=['itemid'], keep='first').reset_index(drop=True)
    return  tmp[['itemid','categoryid']]

def get_item_properties(timestamp, items):
    ''' Возвращает свойства товаров/товара, актуальные на заданный момент времени '''
    tmp = items.query("timestamp <= @timestamp") \
               .drop_duplicates(subset=['itemid','property'], keep='first').reset_index(drop=True)
    return  tmp[['itemid','property','value_code']]



# -----------------------------------------------------------------------
#                 процедура загрузки и очистки данных
# -----------------------------------------------------------------------

def prepare_infer_data(s3: S3FileSystem, SRC_FILES: dict, INFER_FILES: dict, CONFIG: dict):

    # Удаляем результаты предыдущих загрузок
    delete_s3_files(s3, INFER_FILES)

    # Загружаем исходные файлы
    if ((ctree := load_csv_files(s3, [SRC_FILES['cats_src' ]])) is None or
        (items := load_csv_files(s3,  SRC_FILES['props_src'] )) is None or
        (events:= load_csv_files(s3, [SRC_FILES['event_src']])) is None
    ):  return False

    # ---------------------------------------
    # Обрабатываем файл с категориями товаров
    # ---------------------------------------

    # преобразуем parentid у корневых категорий
    ctree['parentid'] = ctree['parentid'].fillna(-1).astype('int')

    # удаляем записи с некорректными parentid
    tmp = ctree.shape[0]
    ctree.query("parentid in @ctree['categoryid'].unique()  or  parentid == -1", inplace=True)
    if  tmp > ctree.shape[0]:
        logging.warning(f"{tmp-ctree.shape[0]} records with incorrect \'parentid\' "
                        f"deleted from {SRC_FILES['cats_src']}")

    # убеждаемся, что деревья категорий не пересекаются и не дублируются (ровно 1 родитель у каждого узла)
    if  ctree.groupby('categoryid').agg(parents=('parentid','count'))['parents'].max() != 1:
        logging.error(f"Inconsistent data in {SRC_FILES['cats_src']}")
        return False

    # определяем корневую категорию и глубину вложенности для каждого узла
    def get_ctree_root(id):
        nodelist = [id]
        while True:
            nodelist += [ ctree.at[nodelist[-1], 'parentid'] ]
            if nodelist[-1] == -1: break
        return nodelist[-2], len(nodelist) -1
    vec_ctree_root = np.vectorize(get_ctree_root)

    ctree.set_index('categoryid', inplace=True)
    ctree['root'], ctree['depth'] = vec_ctree_root(ctree.index.values)
    ctree.reset_index(inplace=True)
    

    # ---------------------------------------
    # Обрабатываем файл со свойствами товаров
    # ---------------------------------------

    # сдвигаем начало "операционного дня" к началу календарных суток
    TIMELAG = 3*3600*1000
    items['timestamp'] = items['timestamp'] - TIMELAG

    # обеспечиваем уникальность properties каждого объекта внутри каждой загрузки
    tmp = items.shape[0]
    items.drop_duplicates(subset=['timestamp','itemid','property'], keep='last', inplace=True)
    if tmp > items.shape[0]:
        logging.warning(f"{tmp-items.shape[0]} duplicated records deleted from items properties")

    # определяем itemid товаров, категории которых отсутствуют в ctree
    bad_itemids = set(
        items.query("property == 'categoryid'").sort_values(by='timestamp').merge(
            ctree[['categoryid']].astype(str), how='left', left_on='value', right_on='categoryid'
        ).query("categoryid.isna()")['itemid']
    )
    # удаляем записи о товарах с некорректными categoryid
    tmp = items.shape[0]
    items.query("itemid not in @bad_itemids", inplace=True)
    if tmp > items.shape[0]:
        logging.warning(f"{tmp-items.shape[0]} records with incorrect \'categoryid\' deleted from items properties")

    # выделяем записи categoryid и available в отдельнуые таблицы
    items_ctgr = items.query("property == 'categoryid'").sort_values(
        ['timestamp','itemid'], ascending=[False,True], ignore_index=True
    ).drop(columns='property').rename(columns={'value':'categoryid'})
    items_ctgr['categoryid'] = items_ctgr['categoryid'].astype(int)

    items_avail= items.query("property == 'available'").sort_values(
        ['timestamp','itemid'], ascending=[False,True], ignore_index=True
    ).drop(columns='property')

    # удаляем записи categoryid и available из item_properties
    items.query("property != 'available'  and  property != 'categoryid'", inplace=True)
    
    # заменяем уникальные значения property-value числовым кодом
    items['value_code'] = items.sort_values(by=['property','value']).groupby('property')['value'] \
                               .transform('rank', method='dense').astype(int) -1

    # преобразуем содержимое таблицы в целые числа
    items = items.drop(columns=['value']).astype(int).sort_values(
        ['timestamp','itemid','property'], ascending=[False,True,True], ignore_index=True
    )


    # ---------------------------------------
    # Обрабатываем файл событий
    # ---------------------------------------

    # сдвигаем начало "операционного дня" к началу календарных суток
    events['timestamp'] = events['timestamp'] - TIMELAG

    # кодируем значения в колонке event
    events['event'] = events['event'].replace({
        'view':'0', 'addtocart':'1', 'transaction':'2'
    }).astype(np.int8)

    # убираем события с объектами, категории которых не зарегистрированы в ctree
    tmp  = events.shape[0]
    events.query("itemid in @items_ctgr['itemid'].unique()", inplace=True)
    if  tmp > events.shape[0]:
        logging.warning(f"{tmp-events.shape[0]} event records with incorrect item ids removed")

    # удаляем события, произошедшие до регистрации категорий соответствующих items и, заодно,
    # определяем актуальную категорию товара для каждого события (на момент его совершения)
    events.sort_values(by='timestamp', inplace=True)
    evtmp = events.query("timestamp < @events['timestamp'].min()")  # инициализируем временный dataframe
    prev_date = 0
    del_count = 0       # счетчик удаляемых событий

    # цикл по датам загрузок item properties
    for load_date in sorted(items['timestamp'].unique().tolist() + [events['timestamp'].max()+1]):
        valid_items = get_registered_items(load_date-1, items_ctgr)

        # удаление событий с товарами, не зарегистрированными на момент совершения события
        tmp  = events.shape[0]
        events.query("timestamp >= @load_date or itemid in @valid_items", inplace=True)
        del_count += tmp - events.shape[0]

        # получение актуальной категории и доступности товара
        evtmp = pd.concat([
            evtmp,
            events.query("timestamp < @load_date and timestamp >= @prev_date").merge(
                get_item_category(load_date-1, items_ctgr), how='left', on='itemid'
            ).merge(
                get_item_availability(load_date-1, items_ctgr, items_avail), how='left', on='itemid' 
            )
        ], axis=0, ignore_index=True)
        
        prev_date = load_date

    if  del_count:  logging.warning(f"{del_count} event records with unregistered items removed")

    # Если к моменту определения категории товара его доступность явно не определена, считаем товар доступным
    evtmp['value'] = evtmp['value'].fillna('1').astype(np.int8)

    # добавляем в таблицу событий информацию о root-категории товаров, актуальной на момент взаимодействия
    events = evtmp.astype({
        'event': np.int8, 'categoryid': int
    }).rename(columns={'value':'available'}).merge(
        ctree[['categoryid','root']], how='left', on='categoryid'
    ).sort_values(by='timestamp', ignore_index=True)

    # разделяем файл событий по дате последней загрузки items
    inference_date = items['timestamp'].max()
    last_events    = events.query("timestamp >= @inference_date")
    logging.info(f"EVENTS file split by inference date: {pd.to_datetime(inference_date, unit='ms')}")

    # отсекаем слишком древние события, ввиду их неактуальности для целей расчета/ранжирования рекомендаций
    MS_PER_DAY     = 24*60*60*1000
    hist_start_date= inference_date - 7*CONFIG['EVENT_HISTORY_WEEKS'] * MS_PER_DAY
    events         = events.query("timestamp < @inference_date and timestamp >= @hist_start_date")
    logging.info(f"EVENTS older {pd.to_datetime(hist_start_date, unit='ms')} were pruned")

    # сохраняем очищенные данные
    save_to_parquet(ctree,      s3, INFER_FILES['cats_dst'])
    save_to_parquet(items_ctgr, s3, INFER_FILES['item_cat'])
    save_to_parquet(items,      s3, INFER_FILES['item_prop'])
    save_to_parquet(items_avail,s3, INFER_FILES['available'])
    save_to_parquet(events,     s3, INFER_FILES['event_dst'])
    save_to_parquet(last_events,s3, INFER_FILES['eventlast'])

    logging.info("Inference data saved successfully")
    return True



# -----------------------------------------------------------------------
#      процедура расчета оффлайн-рекомендаций | дообучения моделей
# -----------------------------------------------------------------------

def calc_recs(s3: S3FileSystem, MODEL_FILES: dict, INFER_FILES: dict, PROD_FILES: dict, CONFIG: dict,
              retrain=False):
    import scipy
    from   sklearn.preprocessing   import MinMaxScaler
    from   implicit.als            import AlternatingLeastSquares
    from   implicit.evaluation     import mean_average_precision_at_k
    from   sklearn.model_selection import ParameterGrid
    from   catboost                import CatBoostClassifier, Pool
    from   catboost.utils          import eval_metric
    from   threadpoolctl           import threadpool_limits
    threadpool_limits(1, "blas")
    RANDOM_STATE = 42

    def calc_item_rating(df):
        ''' Формирует вектор рейтинга взаимодействий: наличие просмотров + добавления в корзину + покупки*2 '''
        return ((df[0] > 0) + df[1] + df[2]*2).astype(np.int16)

    def user_item_matrix(events_set, users, items):
        ''' Формирует и возвращает матрицу взаимодействий в dense и sparse формате '''
        user_item = events_set.query("visitorid in @users  and  itemid in @items")     \
                              .groupby(['visitorid','itemid'])['event'].value_counts() \
                              .unstack(fill_value=0).reset_index()
        # формируем рейтинг взаимодействий
        user_item['rating'] = 0  if user_item.shape[0]==0  else calc_item_rating(user_item)
        return user_item, scipy.sparse.csr_matrix(
            (user_item['rating'], (user_item['visitorid'], user_item['itemid'])), 
             shape=(users.max()+1, items.max()+1)
        )

    def als_validate(csr_train, csr_val, k=10, **kwargs):
        ''' Обучает и валидирует ALS с заданными гиперпараметрами '''
        model = AlternatingLeastSquares(random_state=RANDOM_STATE, **kwargs)
        model.fit(csr_train, show_progress=False)
        map_k = 0 if csr_val is None  \
                else mean_average_precision_at_k(model, csr_train, csr_val, k, show_progress=False)
        return  model, map_k

    def als_train(csr_train, csr_val, **hyper_params):
        ''' Обучает ALS с возможным подбором гиперпараметров '''

        if hyper_params:
            model,_ = als_validate(csr_train, None, **hyper_params)
            return  model, hyper_params
        
        # подбор гиперпараметров
        grid = {
            'alpha':[30.0,50.0,70.0,100.0], 'regularization':[0.005,0.01,0.05], 
            'iterations':[10,15,20,25],     'factors':[50,100,150,200]
        }
        test_grid   = {'alpha':[100.0], 'factors':[100], 'iterations':[15], 'regularization':[0.005]}
        best_metric = 0
        best_params = {}
        best_model  = None

        for params in list(ParameterGrid(test_grid)):     # в prom-e ЗАМЕНИТЬ test_grid НА grid !!!
            model, metric = als_validate(csr_train, csr_val, **params)
            if metric > best_metric:
                best_metric = metric
                best_params = params
                best_model  = model
        return  best_model, best_params

    def cb_train(train_pool):
        ''' Обучает Catboost с подбором гиперпараметров '''

        model = CatBoostClassifier(random_state=RANDOM_STATE, verbose=False, 
            auto_class_weights='Balanced', loss_function='Logloss',  eval_metric='Recall'
        )

        grid = {
            'learning_rate':[0.18, 0.16, 0.14, 0.12, 0.09, 0.03],
            'l2_leaf_reg':  [1, 3, 5],
            'depth'        :[4, 6, 8, 10],
            'iterations'   :[10,100,500,1000],
        }
        test_grid = {
            'l2_leaf_reg':  [   1,    3],
            'learning_rate':[0.14, 0.12, 0.18],
            'iterations'   :[  10,  100],
            'depth'        :[   6,    4,   10],
        }
        
        result = model.grid_search(test_grid, X=train_pool, stratified=True, refit=True, 
            partition_random_seed=RANDOM_STATE, plot=False, verbose=False 
        )
        return  model, result['params']

    def reg_model(model, X, y_pred, metrics, params, config, artifacts=None, runid=None, desc="Scheduled model retrain"):
        import mlflow 
        MODEL_NAME = config["MODEL_NAME"]

        mlflow.set_tracking_uri(config['MLFLOW_SERVER_URL'])
        logging.info(f"MLFlow tracking URL set to: {mlflow.get_tracking_uri()}")
        experiment_id = mlflow.set_experiment(config['EXPERIMENT_NAME']).experiment_id
        logging.info(f"MLFlow experiment name/id: {config['EXPERIMENT_NAME']}/{experiment_id}")

        with mlflow.start_run(run_id=runid, run_name=f"{MODEL_NAME}_run", experiment_id=experiment_id, description=desc) as run:
            run_id = run.info.run_id
            if  model is not None:
                input_example = X[:10]
                signature     = mlflow.models.infer_signature(X, y_pred)
                model_info    = mlflow.catboost.log_model(
                    cb_model             = model,
                    registered_model_name= MODEL_NAME,
                    input_example        = input_example,
                    signature            = signature,
                    await_registration_for=0,
                    artifact_path        = f'model'
                )
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                mlflow.set_tag('estimator', 'CatBoostClassifier')
                logging.info(f"Model <{MODEL_NAME}> registered successfully (run_id:{run_id})")

            if artifacts is not None:
                for key,val in artifacts.items():
                    mlflow.log_dict(val, f"{key}.json")
                    logging.info(f"MLFlow artifact {key}.json logged (run_id:{run_id})")

        logging.info(f"MLFlow run {run_id} closed")
        return run_id



    logging.info(f"Running with config: {CONFIG}")
    # ---------------------------------------
    # Загружаем исходные файлы
    # ---------------------------------------
    if ((items      := load_parquet_file(s3, INFER_FILES['item_prop'])) is None or
        (items_ctgr := load_parquet_file(s3, INFER_FILES['item_cat' ])) is None or
        (items_avail:= load_parquet_file(s3, INFER_FILES['available'])) is None or
        (events     := load_parquet_file(s3, INFER_FILES['event_dst'])) is None
    ):  return False

    if (not retrain and (
        (als_params := load_pkl_file    (s3, MODEL_FILES['als_parms'])) is None or
        (cb_model   := load_pkl_file    (s3, MODEL_FILES['cb_model' ])) is None
    )):  return False

    # ---------------------------------------
    # Определяем опорные даты для расчетов
    # ---------------------------------------
    MS_PER_DAY     = 24*60*60*1000
    max_event_time = events['timestamp'].max()
    infer_time     = ((max_event_time-1) // MS_PER_DAY +1) * MS_PER_DAY
    infer_date     = pd.to_datetime(infer_time, unit='ms')
    logging.info(f"Calculated inference date: {infer_date}")

    # для ограничения размера матрицы взаимодействий, в режиме расчета рекомендаций возьмем только 
    # пользователей, проявивших активность за последние недели (~60 тыс уникальных пользователей в неделю)
    cut_off_time   = 0 if retrain  else infer_time - 7*CONFIG['EVENT_CUT_OFF_WEEKS']*MS_PER_DAY
    if not retrain:  logging.info(f"Calculated cut_off time: {pd.to_datetime(cut_off_time, unit='ms')}")

    # в режиме переобучения моделей необходимо определить точки для разделения выборки на train/target/test
    test_time      = infer_time if not retrain  else (
                     infer_time - 7*CONFIG['EVENT_TEST_WEEKS']*MS_PER_DAY
    )
    if  retrain:  logging.info(f"Calculated test_time: {pd.to_datetime(test_time, unit='ms')}")

    target_time    = infer_time if not retrain  else (
                     test_time - 7*CONFIG['EVENT_TARGET_WEEKS']*MS_PER_DAY
    )
    if  retrain:  logging.info(f"Calculated target_time: {pd.to_datetime(target_time, unit='ms')}")

    # рассчитываем время отсечения событий для определения АКТУАЛЬНЫХ топ-100 товаров
    top_pop_time   = infer_time - 7*CONFIG['EVENT_POPULAR_WEEKS']*MS_PER_DAY
    logging.info(f"Calculated top_pop_time: {pd.to_datetime(top_pop_time, unit='ms')}")

    # ---------------------------------------
    # Определяем топ-100 популярных товаров
    # ---------------------------------------
    top_popular = events.query("timestamp >= @top_pop_time")                               \
                        .groupby(['itemid'])['event'].value_counts().unstack(fill_value=0) \
                        .sort_values(by=2,ascending=False).head(100)

    # считаем и масштабируем рейтинг популярности
    top_popular['rating']    = calc_item_rating(top_popular)
    top_popular['pop_score'] = MinMaxScaler().fit_transform(top_popular['rating'].to_frame())

    # сортируем по скорингу
    top_popular = top_popular[['rating','pop_score']].sort_values(by='rating',ascending=False).reset_index()
    logging.info(f"Calculated top_popular: {pd_info(top_popular)}")


    # ---------------------------------------
    # Расчет персональных рекомендаций
    # ---------------------------------------
    
    # пользователи, на которых будем обучать als-модель
    als_users = events.query("timestamp < @target_time")['visitorid'].unique()
    logging.info(f"als_users: {als_users.shape[0]}")

    # для рекомендаций (но не для обучения!) будем использовать только доступные товары
    if  not retrain:
        av_items = np.array(list(get_available_items (infer_time, items_ctgr, items_avail)))
    else:
        av_items = np.array(list(get_registered_items(target_time, items_ctgr)))
    logging.info(f"available items: {av_items.shape[0]}")

    # строим матрицу взаимодействий для обучения:  als_users x av_items
    user_item, user_item_sparse = user_item_matrix(
        events.query("timestamp <  @target_time"), als_users, av_items
    )
    # валидационная матрица взаимодействий (в режиме расчета рекомендаций - ПУСТАЯ)
    user_item_val, user_item_val_sparse = user_item_matrix(
        events.query("timestamp >= @target_time"), als_users, av_items
    )

    # обучаем ALS-модель
    if  not retrain:
        als_model,_ = als_train(user_item_sparse, None, **als_params)
    else:
        als_model,als_params = als_train(user_item_sparse, user_item_val_sparse)
    logging.info(f"Retrained als_params: {als_params}")


    # пользователи, которым будем давать персональные рекомендации
    hot_users = events.query("visitorid in @user_item['visitorid'].unique()")['visitorid'].unique()
    logging.info(f"user_item users: {hot_users.shape[0]}")
    if  not retrain:
        hot_users = events.query(
            "visitorid in @hot_users  and  timestamp >= @cut_off_time"
        )['visitorid'].unique()
    else:
        hot_users = events.query(
            "visitorid in @hot_users  and  visitorid in @user_item_val['visitorid'].unique()"
        )['visitorid'].unique()
    logging.info(f"hot_users: {hot_users.shape[0]}")

    # вычисляем коллаборативные рекомендации
    RECS_PER_USER = CONFIG['ALS_RECS_PER_USER']
    als_recommendations = als_model.recommend(
        hot_users, 
        user_item_sparse[hot_users], 
        filter_already_liked_items=True, N=RECS_PER_USER
    )
    personal_als  = pd.DataFrame({
        'itemid'   : als_recommendations[0].ravel(),
        'als_score': als_recommendations[1].ravel()
    }, index=pd.MultiIndex.from_product(
        [hot_users, range(RECS_PER_USER)], names=['visitorid', 'als_rank'])
    ).reset_index()
    del als_recommendations, user_item_sparse, user_item_val_sparse
    logging.info(f"personal_als calculated:\n{pd_info(personal_als)}")

    
    # получаем подобные для всех товаров, известных модели
    SIMS_PER_ITEM = CONFIG['ALS_SIMS_PER_ITEM']
    sim_items = als_model.similar_items(user_item['itemid'].unique(), N=SIMS_PER_ITEM)

    similar_items = pd.DataFrame({
        'sim_itemid': sim_items[0].ravel(),
        'sim_score' : sim_items[1].ravel()
    }, index=pd.MultiIndex.from_product(
        [user_item['itemid'].unique(), range(SIMS_PER_ITEM)], names=['itemid', 'sim_rank']
    )).reset_index()
    del sim_items, user_item, user_item_val, als_model
    logging.info(f"similar_items calculated:\n{pd_info(similar_items)}")


    # объединяем информацию из personal_als, similar_items и top_popular в привязке к visitorid
    candidades = personal_als[['visitorid','itemid']].merge(
        similar_items, how='left', on='itemid'
    ).groupby(['visitorid','sim_itemid']).agg(
        sim_score=('sim_score','max')
    ).reset_index().rename(
        columns={'sim_itemid': 'itemid'} 
    ).merge(
        personal_als[['visitorid','itemid','als_score']], how='outer', on=['visitorid','itemid']
    ).merge(
        top_popular[['itemid','pop_score']], how='left', on='itemid'
    )
    del personal_als

    # формируем таргет для модели ранжирования (добавления в корзину и покупки)
    if  retrain:
        events_target = events.query("timestamp >= @target_time and timestamp < @test_time").copy()
        events_target['target'] = (events_target['event'] > 0).astype(np.int8)

        # расширяем состав candidades за счет положительных сэмплов из events_target
        candidades = candidades.merge(
            events_target.groupby(['visitorid','itemid']).agg(target=('target','max')).reset_index().query(
                "visitorid in @hot_users and target > 0"
            ), 
            how='outer', on=['visitorid','itemid']
        )
        candidades["target"] = candidades["target"].fillna(0).astype(np.int8)
    logging.info(f"candidades:\n{pd_info(candidades)}")

    # получаем свойства товаров, актуальные на target_time
    item_props = get_item_properties(target_time-1, items)

    # отбираем категориальные свойства (с количеством значений до 10-и)
    prop_vals = item_props.groupby('property').agg(nvalues=('value_code','max')).reset_index()
    categorical_props = prop_vals.query("nvalues <= 5")['property'].unique()

    # выясняем предпочтения пользователей по категориальным свойствам товаров
    user_item_prop = candidades[['visitorid','itemid']].merge(
        item_props.query("itemid in @candidades['itemid'].unique() and property in @categorical_props"),
        how='left', on='itemid'
    )
    user_prop_score = user_item_prop.groupby(['visitorid','property']).agg(
        prop_score=('itemid','nunique')
    ).reset_index()
    user_item_prop_score = user_item_prop.merge(
        user_prop_score, how='left', on=['visitorid','property']
    ).groupby(['visitorid','itemid']).agg(
        prop_score=('prop_score','mean')
    ).fillna(0).reset_index()
    del user_item_prop, user_prop_score
    logging.info(f"user_item_prop_score:\n{pd_info(user_item_prop_score)}")

    # нормализуем user_item_prop_score по каждому пользователю
    def normalize_col_by_col(df: pd.DataFrame, col: str, by_col: str):
        from sklearn.preprocessing import normalize
        tmp      = df.sort_values(by=by_col)
        tmp[col] = tmp.groupby(by_col)[col].apply(
            lambda x: normalize(x.values.reshape(-1,1), norm='l1', axis=0, copy=True, return_norm=True)[0].ravel()
        ).explode(col).values.astype('float32')
        df[col]  = tmp[col]
    normalize_col_by_col(user_item_prop_score, 'prop_score', 'visitorid')

    # считаем hit_score
    user_item_hit_score = events.query("visitorid in @hot_users and timestamp < @target_time").groupby(
        ['visitorid','itemid']
    ).agg(
        hit_score=('event','nunique')
    ).reset_index().merge(
        candidades[['visitorid','itemid']], how='right', on=['visitorid','itemid']
    ).fillna(0)
    logging.info(f"user_item_hit_score:\n{pd_info(user_item_hit_score)}")

    # добавляем признаки пользователя - "стаж" и активность
    user_features = events.query("visitorid in @hot_users and timestamp < @target_time").groupby(
        "visitorid"
    ).agg(
        stage  =('timestamp', lambda x: (infer_date - pd.to_datetime(x.min(),unit='ms')).days +1),
        nclicks=('timestamp', 'count'),
        nbuys  =('transactionid', 'count')
    ).reset_index()
    user_features["click_per_day"] = user_features["nclicks"] / user_features["stage"]
    user_features["buy_per_click"] = user_features["nbuys"]   / user_features["nclicks"]
    logging.info(f"user_features:\n{pd_info(user_features)}")

    # вносим сформированные дополнительные признаки в candidades
    candidades = candidades.merge(
        user_item_prop_score, how='left', on=['visitorid','itemid']
    ).merge(
        user_item_hit_score,  how='left', on=['visitorid','itemid']
    ).merge(
        user_features[['visitorid','stage','click_per_day','buy_per_click']], how='left', on=['visitorid']
    )
    # объединяем als-скоринги
    candidades['sim_score'] = candidades['als_score'].fillna(1) * candidades['sim_score'].fillna(0)
    logging.info(f"final candidades:\n{pd_info(candidades)}")

    # фиксируем список признаков
    feature_cols   = ['sim_score','pop_score','prop_score','hit_score',
                      'stage','click_per_day','buy_per_click']

    if  retrain:    # --------------------------------------- #
                    # -   Переобучаем модель ранжирования   - #
                    # --------------------------------------- #

        # в кандидатах оставляем только тех пользователей, у которых есть хотя бы один положительный таргет
        candidades_for_train = candidades.groupby("visitorid").filter(lambda x: x["target"].sum() > 0)

        # убираем неинформативные дубликаты (гарантированно оставляя положительный таргет)
        candidades_for_train = candidades_for_train.sort_values(by='target',ascending=False).drop_duplicates(
            subset=['visitorid','itemid']+feature_cols, keep='first'
        )
        logging.info(f"candidades for catboost train:\n{pd_info(candidades_for_train)}")

        # Обучаем ранжирующую модель с подбором гиперпараметров
        train_data = Pool(data=candidades_for_train[feature_cols], label=candidades_for_train['target'])
        cb_model, cb_params = cb_train (train_data)
        logging.info(f"catboost parameters:\n{cb_params}")

        # получаем оценку важности признаков
        feature_importance = pd.DataFrame(cb_model.get_feature_importance(), index=feature_cols, columns=["fi"])
        feature_importance = feature_importance.sort_values(by="fi", ascending=False)
        logging.info(f"feature_importance:\n{feature_importance}")

        # сохраняем данные
        save_to_pkl(als_params, s3, MODEL_FILES['als_parms'])
        save_to_pkl(cb_params,  s3, MODEL_FILES['cb_parms'])
        save_to_pkl(cb_model,   s3, MODEL_FILES['cb_model'])


    # получаем скоринг рекомендаций (в режиме retrain - уже с новой моделью)
    inference_data = Pool(data=candidades[feature_cols])
    predictions    = cb_model.predict_proba(inference_data)

    # сортируем в соответствии с feature importance и проставим rank, начиная с 1
    candidades["cb_score"] = predictions[:,1]
    candidades.sort_values(['visitorid', 'cb_score', 'sim_score', 'hit_score', 'prop_score', 'pop_score'], 
                        ascending=[True, False, False, False, False, False], inplace=True
    )
    candidades["rank"] = candidades.groupby("visitorid")["cb_score"].cumcount() +1


    if retrain:         # вычисляем Recall и сохраняем новую версию модели
        candidades_for_eval = events.query("timestamp >= @test_time and visitorid in @hot_users and event > 0") \
                                    .groupby(['visitorid','itemid']).agg(label=('event','max'))    \
                                    .merge(candidades, 
                                     how='left', left_index=True, right_on=['visitorid','itemid'])
        candidades_for_eval['tgt'] = candidades_for_eval['cb_score'].notna().astype(int)
        candidades_for_eval['lbl'] =(candidades_for_eval['label'] > 0).astype(int)
        metrics = eval_metric(
            candidades_for_eval['lbl'].tolist(), candidades_for_eval['tgt'].tolist(), metric='Recall'
        )
        # регистрируем в mlflow и сохраняем feature_importance и als_params в артефактах
        reg_model(cb_model, candidades_for_train[feature_cols], candidades_for_train['target'], 
                  {'Recall':metrics[0]}, cb_params, CONFIG,
                  artifacts={'feature_importance': feature_importance.to_dict(), 'als_params': als_params})
    
    else:
        # формируем финальные рекомендации с минимально необходимым для prod набором полей
        max_recommendations_per_user = CONFIG['MAX_RECS_PER_USER']
        final_recommendations = candidades.query(
            "rank <= @max_recommendations_per_user"
        )[['visitorid','itemid','rank']] #,'cb_score','als_score']+feature_cols]
        logging.info(f"final_recommendations calculated:\n{pd_info(final_recommendations)}")

        # сохраняем рекомендации
        save_to_parquet(top_popular,           s3, PROD_FILES['top_pop'])
        save_to_parquet(similar_items,         s3, PROD_FILES['similar'])
        save_to_parquet(final_recommendations, s3, PROD_FILES['final'  ])

        # на всякий случай сохраняем candidades с полным набором полей
        save_to_parquet(candidades,            s3, PROD_FILES['ranked'])
    return True



# -------------------------------------------------
#      процедура перезагрузки рекомендаций
# -------------------------------------------------

def reload_data_files(s3: S3FileSystem, REC_SERVICES: dict):
    ret = True
    try:
        for _,url in REC_SERVICES.items():
            logging.info(f"Reloading data file(s) for service {url}")
            resp = post_request(url + "/reload")
            ret &= ('result' in resp)  and  (resp['result'] == "OK")
            if  not ret: break
    except Exception as e:
            logging.error(f"Exception {e} occured - service unavailable!")
    return ret


