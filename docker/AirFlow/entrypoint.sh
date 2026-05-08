#!/bin/bash
set -e

# Validate directories exist (they should be baked into the image)
[ -d /opt/airflow ] || mkdir -p /opt/airflow
[ -d /opt/airflow/logs ] || mkdir -p /opt/airflow/logs
[ -d /opt/airflow/dags ] || mkdir -p /opt/airflow/dags
[ -d /opt/airflow/plugins ] || mkdir -p /opt/airflow/plugins
[ -d /opt/airflow/config ] || mkdir -p /opt/airflow/config

# Set permissions for runtime-generated content (logs, cache)
chmod g+rwX /opt/airflow/logs 2>/dev/null || true
chmod g+rwX /opt/airflow/cache 2>/dev/null || true

exec gosu airflow airflow "$@"
