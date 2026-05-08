#!/bin/bash
set -e

AIRFLOW_UID=${AIRFLOW_UID:-50000}

echo "Fixing permissions for ./airflow/ with UID=$AIRFLOW_UID"

sudo chown -R $AIRFLOW_UID:0 ./airflow/dags
sudo chmod -R g+rwX ./airflow/dags

sudo chown -R $AIRFLOW_UID:0 ./airflow/plugins
sudo chmod -R g+rwX ./airflow/plugins

sudo chown -R $AIRFLOW_UID:0 ./airflow/config
sudo chmod -R g+rwX ./airflow/config

sudo chown -R $AIRFLOW_UID:0 ./airflow/logs
sudo chmod -R g+rwX ./airflow/logs

find ./airflow -type d -exec sudo chmod g+s {} \;

echo "Done! Permissions fixed."
echo "You can now run: docker compose up -d"
