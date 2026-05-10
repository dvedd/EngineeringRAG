FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    AIRFLOW_HOME=/opt/airflow \
    PATH="/usr/local/bin:/usr/bin:/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3-pip \
    curl \
    libpq-dev \
    build-essential \
    gosu \
  && rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED \
  && rm -rf /var/lib/apt/lists/* \
  && gosu nobody true

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
  && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

ARG AIRFLOW_VERSION=3.2.0
ARG PYTHON_VERSION=3.12
ARG AIRFLOW_UID=1000
ARG AIRFLOW_GID=1000

# Install Airflow core
RUN pip install --no-cache-dir uv \
  && uv pip install --no-cache --system \
      "apache-airflow[celery,postgres,redis,fab]==${AIRFLOW_VERSION}" \
      --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt" \
  && pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true \
  && uv pip install --no-cache --system onnxruntime-gpu \
      --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
  && pip install --no-cache-dir FlagEmbedding

COPY requirements.txt /requirements.txt
RUN uv pip install --no-cache --system -r /requirements.txt 2>/dev/null || true

RUN groupadd -g ${AIRFLOW_GID} airflow 2>/dev/null || true \
  && getent passwd ${AIRFLOW_UID} > /dev/null 2>&1 || useradd -m -u ${AIRFLOW_UID} -g ${AIRFLOW_GID} -s /bin/bash airflow \
  && mkdir -p /opt/airflow/{dags,logs,plugins,config} \
  && chown -R ${AIRFLOW_UID}:${AIRFLOW_GID} /opt/airflow \
  && chmod -R g+rwX /opt/airflow

COPY airflow/dags /opt/airflow/dags
COPY airflow/plugins /opt/airflow/plugins
COPY airflow/config /opt/airflow/config

RUN mkdir -p /opt/airflow/logs

RUN chown -R ${AIRFLOW_UID}:${AIRFLOW_GID} /opt/airflow/dags /opt/airflow/plugins /opt/airflow/config /opt/airflow/logs \
  && chmod -R g+rwX /opt/airflow/dags /opt/airflow/plugins /opt/airflow/config /opt/airflow/logs

COPY docker/AirFlow/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /opt/airflow
ENTRYPOINT ["/entrypoint.sh"]
CMD ["airflow", "--help"]
