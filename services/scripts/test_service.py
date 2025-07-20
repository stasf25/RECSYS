import os, requests
from dotenv     import find_dotenv, load_dotenv
load_dotenv(find_dotenv("env.services")), load_dotenv()

recommendations_url= f"http://127.0.0.1:{os.environ['RECOMMENDATIONS_PORT']}"
features_store_url = f"http://127.0.0.1:{os.environ['FEATURES_STORE_PORT']}"
events_store_url   = f"http://127.0.0.1:{os.environ['EVENTS_STORE_PORT']}"


def post_request(url, params):
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

    resp = requests.post(url, headers=headers, params=params)

    if resp.status_code == 200:
        recs = resp.json()
    else:
        recs = []
        print(f"status code: {resp.status_code}")
    return recs    
 
# обычный пользователь
post_request(recommendations_url+"/recommendations_offline",{"user_id": 108978, 'k': 10})
post_request(recommendations_url+"/recommendations_online", {"user_id": 108978, 'k': 10})
post_request(recommendations_url+"/recommendations",        {"user_id": 108978, 'k': 100})

# обычный пользователь без онлайн-взаимодействий
post_request(recommendations_url+"/recommendations",        {"user_id": 717032, 'k': 100})
# добавляем взаимодействие
post_request(events_store_url+"/put",                       {"user_id": 717032, 'item_id': 145730})
post_request(recommendations_url+"/recommendations",        {"user_id": 717032, 'k': 100})

# холодный пользователь с онлайн-взаимодействием
post_request(recommendations_url+"/recommendations",        {"user_id":     86, 'k': 100})

# холодный пользователь без онлайн-взаимодействий
post_request(recommendations_url+"/recommendations",        {"user_id":12345678,'k': 100})

print("DONE - see test_service.log")

