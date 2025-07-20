import requests, time, os, s3fs 
import pandas as pd
import numpy  as np
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv("env.services")), load_dotenv()

# Параметры  тестирования  --------------------------------------------------------------
OVERALL_TEST_TIME_SEC = 900     # время, в которое, по-возможности, должен уложиться тест
MAX_TEST_SAMPLES = 20000        # -1 - без ограничений (~41тыс. запросов)
DELAYED_REQUESTS_RATE = 0.005   # доля запросов с 10-и кратной задержкой
# ---------------------------------------------------------------------------------------

PROJECT_ROOT = f"{find_dotenv()[0:-5]}"
RANDOM_STATE = 42
START        = time.time()
S3_DIR       = f"{os.environ['S3_BUCKET_NAME']}/Diplom/infer_data"
url          = f"http://127.0.0.1:{os.environ['RECOMMENDATIONS_PORT']}/recommendations"
headers      = {'Content-type': 'application/json', 'Accept': 'text/plain'}

s3 = s3fs.core.S3FileSystem(
    endpoint_url=os.environ['AWS_ENDPOINT_URL'],
    key=os.environ['AWS_ACCESS_KEY_ID'],
    secret=os.environ['AWS_SECRET_ACCESS_KEY'], cache_regions=True
)

def load_parquet_file(s3, path: str):
    try:
        df = pd.DataFrame()
        with s3.open(path, mode='rb') as fd:
            df = pd.read_parquet(fd)
        return df
    except:
        #logging.error(f"Error loading file: {path}")
        return None


np.random.seed (RANDOM_STATE)
rng = np.random.default_rng()
df  = load_parquet_file(s3, f"{S3_DIR}/last_events.parquet")
# ---------------------------------------------------------------------------------------

num_samples = MAX_TEST_SAMPLES  if MAX_TEST_SAMPLES>0  else df.shape[0]
for i in range(num_samples):
    rnd = rng.random()

    params = {"user_id": df.sample(axis=0)['visitorid'], "k": 100}

    response = requests.post(url, headers=headers, params=params)

    elapsed = time.time() - START
    delay   = max(0, (OVERALL_TEST_TIME_SEC - elapsed)) / (num_samples - i) *rnd*2
    if  rnd > (1 - DELAYED_REQUESTS_RATE):  delay *= 10
    time.sleep(delay)

print (f"Elapsed time: {time.time() - START}")