#!/bin/bash
set -e

mkdir -p /opt/airflow/logs /opt/airflow/dags /opt/airflow/plugins /opt/airflow/config

chmod g+rwX /opt/airflow/logs 2>/dev/null || true
chmod g+rwX /opt/airflow/cache 2>/dev/null || true

exec airflow "$@"
