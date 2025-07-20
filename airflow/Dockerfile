FROM apache/airflow:2.7.3-python3.10
# Устанавливаем libgomp1 внутри контейнера (нужен для implicit) и git для mlflow
USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         libgomp1 \
         git \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*
USER airflow
# Устанавливаем окружение
ADD requirements.txt /tmp/requirements.txt
RUN <<EOF
pip install -U pip
pip install -r /tmp/requirements.txt >/tmp/pip.log
EOF
