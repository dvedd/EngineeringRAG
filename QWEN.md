# EngineeringRAG - Система RAG для строительных норм

## Обзор проекта

Это **RAG-система (Retrieval-Augmented Generation)** для семантического поиска по нормативно-техническим документам (СП, СНиП, ГОСТ). Система состоит из трёх основных компонентов:

| Компонент | Описание | Технологии |
|---|---|---|
| **Data Pipeline** | PDF → MinerU (OCR) → Docling (чанкинг) → Qdrant | Apache Airflow 3.2, MinerU, Docling |
| **Retriever** | Гибридный поиск (dense + sparse + ColBERT) | Qdrant, BAAI/bge-m3, FastEmbed |
| **LLM Service** | RAG-ответы на вопросы по нормам | vllm-light, OpenAI-compatible API |

---

## Архитектура pipeline обработки документов

```
MinIO (PDF)
    │
    ▼
MinerU API          ← OCR, распознавание формул и таблиц
    │
    ▼
Docling Serve       ← иерархический чанкинг по структуре документа
    │
    ▼
Airflow Worker      ← энкодинг: dense + sparse + ColBERT
    │
    ▼
Qdrant              ← гибридный векторный поиск
```

### Основные компоненты

| Компонент | Докер-образ | Порт | Назначение |
|---|---|---|---|
| `docling-serve` | `quay.io/docling-project/docling-serve-cu130:main` | 5001 | Чанкинг Markdown |
| `mineru-api` | `dvmed/mineru:v3.0.9` | 8000 | OCR парсинг PDF |
| `qdrant` | `qdrant/qdrant:latest` | 6333 | Векторная БД |
| `minio` | `minio/minio:latest` | 9000 | Хранилище файлов (S3) |
| `vllm-light` | `vllm/vllm-openai:v0.21.0` | 8020 | Query rewriter (LLM) |
| `airflow-*` | custom `airflow-3.2_cuda-12.9_python-3.12:latest` | 8080/8793 | Оркестрация DAG |

### Дополнительные сервисы

| Компонент | Докер-образ | Порт | Назначение |
|---|---|---|---|
| `postgres` | `postgres:16` | 5432 | Airflow metadata |
| `redis` | `redis:7.2-bookworm` | 6379 | Celery broker |
| `superset` | `apache/superset:6.1.0rc2` | 5054 | Бизнес-аналитика |
| `warehouse-postgres` | `postgres:16` | 5052 | Данные клиента |
| `client-postgres` | `postgres:16` | 5051 | Данные клиента |
| `pgadmin` | `dpage/pgadmin4` | 5050 | Управление БД |

### Конфигурация сервисов

- **MinIO Console:** `http://localhost:9001`
- **Airflow Web UI:** `http://localhost:8080` (login: `admin` / `admin`)
- **Superset:** `http://localhost:5054`
- **Query Rewriter (vllm-light):** `http://localhost:8020`
- **Retriever UI:** `http://localhost:8501`

---

## Building and Running

### Старт через Docker Compose

```bash
# Запуск всех сервисов
docker compose up -d

# Остановка
docker compose down

# Перезапуск конкретного сервиса
docker compose restart docling-serve
docker compose restart mineru-api

# Логи Airflow
docker compose logs -f airflow-scheduler
docker compose logs -f airflow-worker
```

### Airflow

- **Web UI:** `http://localhost:8080` (login: `admin` / `admin`)
- **DAG:** `batch_pipline` — запускается вручную (schedule=None)
- **Connections:** `minio`, `mineru`, `docling`, `qdrant`

### Ручной запуск pipeline

DAG `batch_pipline` обрабатывает PDF-файлы из MinIO бакета `ragfiles`:

1. **Discovery** — список файлов в `pdf/`
2. **MinerU OCR** — конвертация в Markdown
3. **Docling chunking** — иерархический чанкинг
4. **Qdrant upsert** — энкодинг и сохранение векторов

Запуск через Airflow UI или:

```bash
docker compose exec airflow-scheduler airflow dags trigger batch_pipline
```

---

## Retriever Service (Streamlit UI)

Локальная интерфейсная служба для поиска по нормативной базе:

```bash
# Запуск retriever service
cd retriever_service
streamlit run app.py

# Доступ: http://localhost:8501
```

**Режимы поиска:**
- `hybrid` — dense + sparse Prefetch → ColBERT rerank (рекомендуется)
- `dense` — только multilingual-mpnet-base-v2 (ANN)
- `sparse` — только BM25 (ключевые слова)

---

## Технологии эмбеддингов

В Qdrant коллекции используются три векторных сигнала:

| Имя вектора | Модель | Dim | Назначение |
|---|---|---|---|
| `dense` | `BAAI/bge-m3` | 1024 | Семантический поиск |
| `sparse` | `Qdrant/bm25` | — | Лексический поиск (BM25) |
| `colbert` | `BAAI/bge-m3` (late-interaction) | 1024×N | Реранкинг через MaxSim |

Для энкодинга используется `FlagEmbedding.BGEM3FlagModel` с GPU-ускорением.

---

## Структура Qdrant коллекции

**Collection name:** `construction_docs`

### Payload fields

| Поле | Тип | Описание |
|---|---|---|
| `text` | str | Исходный текст чанка |
| `chunk_index` | int | Порядковый номер чанка |
| `headings` | list[str] | Иерархия заголовков |
| `filename` | str | Имя исходного PDF |
| `page_numbers` | list[int] | Номера страниц |
| `is_table` | bool | Флаг таблицы |
| `refs` | list[str] | Нормативные ссылки (СП, СНиП, ГОСТ) |

### Payload индексы

`filename` (KEYWORD), `headings` (KEYWORD), `is_table` (BOOL), `refs` (KEYWORD)

---

## Development Conventions

### Python код

- **Airflow DAG:** Python 3.12, Airflow SDK (v3.2)
- **Retriever service:** Python 3.12, Streamlit, Qdrant client
- **Все библиотеки:** см. `requirements.txt`
- **GPU:** CUDA 12.9, PyTorch с GPU-ускорением
- **Кэши:** `HF_HOME` и `FEDEDEMBED_CACHE_PATH` монтируются в Airflow worker

### Именование

- Переменные Snake case: `qdrant_url`, `batch_size`
- Функции Snake case: `save_mineru_results`, `load_md_to_minio`
- Классы Pascal case: `QdrantRetriever`
- Константы UPPER_SNAKE_CASE: `QDRANT_URL`, `BATCH_SIZE`

### Документация

- All comments в коде — на **английском**
- Документация в `docs/` — на **русском**
- Markdown файлы для MinIO хранятся в `dev_data/mineru_md/`
- JSON чанки — в `dev_data/docling_jsons/`

---

## Конфигурационные переменные окружения

`.env` файл (не в репозитории):

```bash
# Airflow
AIRFLOW_UID=50000
AIRFLOW__CORE__EXECUTOR=CeleryExecutor
AIRFLOW__CORE__AUTH_MANAGER=airflow.providers.fab.auth_manager.fab_auth_manager.FabAuthManager

# MinIO
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin

# Superset
SUPERSET_SECRET_KEY=your-secret-key

# Warehouse PostgreSQL
WAREHOUSE_PG_USER=postgres
WAREHOUSE_PG_PASSWORD=postgres
WAREHOUSE_PG_DB=warehouse

# Client PostgreSQL
CLIENT_PG_USER=postgres
CLIENT_PG_PASSWORD=postgres
CLIENT_PG_DB=postgres
```

---

## Известные ограничения

| Проблема | Описание | Решение |
|---|---|---|
| MinerU VRAM | Не освобождает память между задачами | Перезапуск контейнера |
| Qdrant timeout | При больших ColBERT матрицах | `QDRANT_UPSERT_BATCH=32` |
| Граф связей | Не реализован | Требуется PostgreSQL/Neo4j |
| Терминология | Не покрывает все расхождения | Требуется словарь домена |

---

## Планируемые улучшения

- [ ] Metadata database для отслеживания файлов в MinIO (PostgreSQL)
- [ ] Граф обязательных связей между нормативными документами
- [ ] Добавление терминологического словаря по строительным нормам
- [ ] Автоматическое обновление векторной базы при изменении документов (S3 event trigger)
- [ ] Оптимизация docker images
- [ ] Автоматическое создание Airflow connections
- [ ] Вынос Qdrant в отдельный сервис

---

## Глоссарий

| Термин | Расшифровка |
|---|---|
| RAG | Retrieval-Augmented Generation |
| OCR | Optical Character Recognition |
| PDF | Portable Document Format |
| DAG | Directed Acyclic Graph (Airflow) |
| ANN | Approximate Nearest Neighbor |
| BM25 | Best Match 25 (ranking algorithm) |
| ColBERT | Contextualized Late Interaction Transformer |
