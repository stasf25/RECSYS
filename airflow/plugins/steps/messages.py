from airflow.providers.telegram.hooks.telegram import TelegramHook

def send_telegram_success_message(context: dict):
    hook = TelegramHook(token='', chat_id='')
    dag = context['dag']
    run_id = context['run_id']
    
    hook.send_message({
        'chat_id': '-4661822511',
        #'text': f'Success: {list(context)}'
        'text': f'Исполнение DAG {dag} с id={run_id} прошло успешно!'
    })


def send_telegram_failure_message(context: dict):
    hook = TelegramHook(token='', chat_id='')
    dag = context['dag']
    run_id = context['run_id']
    task = context['task_instance_key_str']
    hook.send_message({
        'chat_id': '-4661822511',
        #'text': f'Failed: {list(context)}'
        'text': f'Исполнение DAG {dag} с id={run_id} (task: {task}) завершилось неудачно'
    })
