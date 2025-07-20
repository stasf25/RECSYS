# Dockerfile
# пример запуска: docker run --publish 8090:8090
FROM python:3.10-slim
LABEL author="mle_20250228_1d75c84a52"

# Копируем в контейнер необходимые файлы
RUN mkdir -p /services/ml_service
COPY ./ml_service/common_utils.py /services/ml_service
COPY ./ml_service/rec_store.py /services/ml_service
COPY ./ml_service/recommendations_service.py /services/ml_service
COPY requirements.txt /services
RUN touch /services/.env
WORKDIR /services

# Устанавливаем окружение внутри контейнера
ENV VIRTUAL_ENV=/services/.venv
RUN python3.10 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install -U pip  &&  pip install -r requirements.txt >pip.log

# порт со стороны контейнера
EXPOSE $RECOMMENDATIONS_PORT

# том со стороны контейнера
#VOLUME /services/models

# команда запуска микросервиса в контейнере
WORKDIR /services/ml_service
CMD uvicorn recommendations_service:app --reload --host $APP_HOST --port $RECOMMENDATIONS_PORT
