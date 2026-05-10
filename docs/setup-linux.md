# Подготовка Linux для EngineeringRAG

Этот документ описывает процесс подготовки Linux-системы для запуска EngineeringRAG.

## Требования

### Минимальные системные требования

| Компонент | Минимум | Рекомендовано |
|-----------|---------|---------------|
| RAM | 16 GB | 32+ GB |
| GPU | any CUDA-capable | NVIDIA RTX 3060+ (12GB+ VRAM) |
| CPU | 4 cores | 8+ cores |
| Storage | 50 GB | 100+ GB (SSD) |

### Поддерживаемые ОС

- Ubuntu 22.04 LTS, 24.04 LTS
- Debian 11+, 12+
- CentOS 8+
- RHEL 8+

## Установка через скрипт

### Быстрый старт

```bash
# Скачивание и запуск скрипта
curl -fsSL https://raw.githubusercontent.com/your-org/EngineeringRAG/main/scripts/setup-linux.sh -o setup-linux.sh
chmod +x setup-linux.sh
sudo ./setup-linux.sh
```

### Что устанавливает скрипт

| Компонент | Версия | Описание |
|-----------|--------|----------|
| Docker | latest | Контейнеризация приложения |
| NVIDIA Container Toolkit | latest | GPU поддержка в Docker |
| CUDA | 12.9 | Библиотеки для GPU ускорения |
| uv | latest | Менеджер Python зависимостей |
| Tesseract OCR | latest | OCR для PDF обработки |

## Ручная установка

### 1. Установка Docker

```bash
# Установка через официальный скрипт
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Добавление пользователя в группу docker
sudo usermod -aG docker $USER

# Запуск Docker сервиса
sudo systemctl enable docker
sudo systemctl start docker
```

### 2. Установка NVIDIA Docker Toolkit

```bash
# Добавление репозитория NVIDIA
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Установка
apt-get update
apt-get install -y nvidia-container-toolkit

# Перезапуск Docker
systemctl restart docker
```

### 3. Установка CUDA 12.9

```bash
# Скачивание
wget https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda_12.9.0_560.35.03_linux.run
chmod +x cuda_12.9.0_560.35.03_linux.run

# Установка (без драйвера, если уже установлен)
sudo ./cuda_12.9.0_560.35.03_linux.run

# Очистка
rm cuda_12.9.0_560.35.03_linux.run

# Настройка PATH (добавить в ~/.bashrc)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### 4. Установка uv

```bash
# Установка через официальный скрипт
curl -LsSf https://astral.sh/uv/install.sh | sh

# Перезагрузка PATH
source "$HOME/.cargo/env"
```

## Проверка установки

### Проверка Docker и GPU

```bash
# Проверка Docker
docker --version
docker-compose --version

# Проверка GPU доступа в Docker
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi
```

Ожидаемый вывод:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 560.35.03    Driver Version: 560.35.03    CUDA Version: 12.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap| Memory-Usage     Allocatable P2P |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0 Off |                  N/A |
|  0%   30C    P8    10W / 250W |   1024MiB / 24576MiB |      Not Supported   |
+-------------------------------+----------------------+----------------------+
```

### Проверка uv

```bash
uv --version
# uv 0.1.x (...)
```

## Структура проекта после установки

```
EngineeringRAG/
├── data/
│   ├── minio/              # MinIO хранилище
│   ├── docling_models/     # Модели Docling
│   ├── fastembed_cache/    # Кэш FastEmbed
│   ├── huggingface_cache/  # Кэш HuggingFace
│   └── cache_airflow/      # Кэш Airflow
├── client-postgres-init/   # Инициализация PostgreSQL
├── scripts/
│   └── setup-linux.sh      # Скрипт установки
├── docker/
│   ├── AirFlow/
│   ├── docling/
│   └── mineru/
├── airflow/
│   ├── dags/
│   ├── plugins/
│   └── config/
├── docs/
├── llm-service/
└── retriever_service/
```

## Устранение проблем

### Docker не видит GPU

1. Проверьте установку драйверов NVIDIA:
```bash
nvidia-smi
```

2. Перезапустите Docker:
```bash
sudo systemctl restart docker
```

3. Проверьте конфигурацию:
```bash
cat /etc/docker/daemon.json
```

Ожидаемое содержимое:
```json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```

### OOM при работе с моделями

Увеличьте лимиты в `compose.yaml` для сервисов с GPU:
```yaml
deploy:
    resources:
        reservations:
            devices:
                - driver: nvidia
                  count: all
                  capabilities: [gpu]
```

### CUDA out of memory

Уменьшите количество параллельных задач в DAG:
```python
# airflow/dags/batch_pipline.py
BATCH_SIZE = 2  # Уменьшено с 4
QDRANT_ENCODE_BATCH = 32  # Уменьшено с 64
```

## Дополнительные настройки

### Настройка переменных окружения

Создайте `.env` файл в корне проекта:

```bash
# Airflow
AIRFLOW_UID=1000
AIRFLOW_GID=1000

# MinIO
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin

# Superset
SUPERSET_SECRET_KEY=your-secret-key-here

# Warehouse PostgreSQL
WAREHOUSE_PG_USER=postgres
WAREHOUSE_PG_PASSWORD=postgres
WAREHOUSE_PG_DB=warehouse

# Client PostgreSQL
CLIENT_PG_USER=postgres
CLIENT_PG_PASSWORD=postgres
CLIENT_PG_DB=postgres
```

### Добавление в PATH

Добавьте в `~/.bashrc`:
```bash
# uv package manager
export PATH="$HOME/.cargo/bin:$PATH"

# CUDA
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
