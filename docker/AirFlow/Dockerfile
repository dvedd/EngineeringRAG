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
  && rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED \
  && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
  && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

ARG AIRFLOW_VERSION=3.2.0
ARG PYTHON_VERSION=3.12

COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir uv \
  && uv pip install --no-cache --system \
      "apache-airflow[celery,postgres,redis,fab]==${AIRFLOW_VERSION}" \
      --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt" \
  && uv pip install --no-cache --system -r /requirements.txt \
  && pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true \
  && uv pip install --no-cache --system onnxruntime-gpu \
      --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

ARG AIRFLOW_UID=50000
RUN useradd -m -u ${AIRFLOW_UID} -s /bin/bash airflow \
  && mkdir -p /opt/airflow/{dags,logs,plugins,config} \
  && chown -R airflow:0 /opt/airflow \
  && chmod -R 775 /opt/airflow

USER airflow
WORKDIR /opt/airflow

# Переопределение nnidea entrypoint
ENTRYPOINT ["airflow"]
CMD ["--help"]
