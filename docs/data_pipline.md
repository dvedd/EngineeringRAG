# RAG Document Pipeline

Система автоматической обработки нормативно-технических документов (PDF) с загрузкой в векторную базу данных для последующего семантического поиска (RAG).

---

## Архитектура

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

---

## Стек

| Компонент | Технология | Назначение |
|---|---|---|
| Оркестрация | Apache Airflow 3.2 (Airflow SDK) | DAG, dynamic task mapping |
| Хранилище файлов | MinIO (S3-compatible) | PDF, Markdown, JSON чанки |
| Парсинг PDF | MinerU API (`pipeline` backend) | Markdown из PDF, формулы, таблицы |
| Чанкинг | Docling Serve (`/v1/chunk/hierarchical`) | Иерархические чанки по структуре документа |
| Dense эмбеддинги | `sentence-transformers/all-MiniLM-L6-v2` | Семантический поиск, dim=384 |
| Sparse эмбеддинги | FastEmbed `Qdrant/bm25` | Лексический поиск (BM25) |
| Late-interaction | FastEmbed `colbert-ir/colbertv2.0` | Реранкинг через MaxSim, dim=128 |
| Векторная БД | Qdrant | Гибридный поиск: dense + sparse + ColBERT |

---

## Структура DAG

DAG `batch_pipline` запускается вручную (`schedule=None`) и состоит из 12 задач:

```
get_buckets_data
      │
list_files_to_process
      │
ckeck_mineru_health ──────────────────────┐
      │                                   │
create_file_batches              create_qdrant_collection
      │                                   │
batch_mineru_submit [dynamic]             │
      │                                   │
wait_mineru_batch [sensor]                │
      │                                   │
save_mineru_results [dynamic]             │
      │                                   │
load_md_to_minio [dynamic]                │
      │                                   │
docling_chunk_submit [dynamic]            │
      │                                   │
wait_docling_batch [sensor]               │
      │                                   │
save_docling_results [dynamic]            │
      │                                   │
      └──────────► save_to_qdrant ◄───────┘
                   [dynamic]
```

### Описание задач

**1. `get_buckets_data`**
Подключается к MinIO, получает список бакетов и их префиксы (виртуальные папки).

**2. `list_files_to_process`**
Собирает ключи объектов в бакете по поддерживаемым форматам (`pdf/`). Если включён `skip_process=True` — пропускает файлы для которых уже есть `.md` в MinIO.

**3. `ckeck_mineru_health`**
`GET /health` к MinerU API. Возвращает HTTP-статус, который блокирует сабмит при недоступности сервиса.

**4. `create_qdrant_collection`**
Создаёт коллекцию Qdrant с конфигурацией векторов (dense + colbert + sparse) и payload-индексами. Запускается один раз до всех `save_to_qdrant`, исключая race condition при параллельном маппинге.

**5. `create_file_batches`**
Делит список файлов на батчи размером `BATCH_SIZE` (по умолчанию 4). Каждый батч — отдельный dynamic task instance для изоляции ошибок.

**6. `batch_mineru_submit` [dynamic]**
Скачивает файлы из MinIO в `/tmp`, отправляет POST `/tasks` к MinerU с параметрами:
- `backend: pipeline`
- `lang_list: east_slavic`
- `formula_enable: true`, `table_enable: true`
- `parse_method: ocr`

**7. `wait_mineru_batch` [deferrable sensor]**
Асинхронно опрашивает MinerU до получения статуса `completed` по всем task_id батча. Освобождает воркер Airflow во время ожидания.

**8. `save_mineru_results` [dynamic]**
Забирает Markdown из `GET /tasks/{id}/result`, сохраняет `.md` файлы в `/tmp`.

**9. `load_md_to_minio` [dynamic]**
Загружает `.md` файлы в MinIO под префикс `dev_data/mineru_md/`.

**10. `docling_chunk_submit` [dynamic]**
Скачивает `.md` из MinIO, отправляет `POST /v1/chunk/hierarchical/file/async` к Docling Serve. Возвращает `task_id` для сенсора.

**11. `wait_docling_batch` [deferrable sensor]**
Асинхронно ждёт завершения чанкинга в Docling.

**12. `save_docling_results` [dynamic]**
Забирает JSON чанки из Docling, обогащает их:
- **Таблицы** — выделяются как отдельные чанки с флагом `is_table=True` и префиксом `[ТАБЛИЦА]`
- **Нормативные ссылки** — извлекаются regex-паттернами в поле `refs` (СП, СНиП, ГОСТ, п., табл., рис., прил.)

Сохраняет обогащённые JSON в `/tmp` и MinIO (`dev_data/docking_jsons/`).

**13. `save_to_qdrant` [dynamic]**
Кодирует тексты тремя моделями и пишет в Qdrant батчами:
- Двухуровневое батчирование: `QDRANT_ENCODE_BATCH=64` (RAM) × `QDRANT_UPSERT_BATCH=32` (HTTP)
- Детерминированный UUID из `filename:chunk_index` — повторный запуск перезаписывает те же точки
- Retry с экспоненциальным backoff на timeout и 5xx ошибки
- `retries=2` на уровне Airflow таски

---

## Структура Qdrant коллекции

**Коллекция:** `construction_docs`

### Векторы

| Имя | Модель | Dim | Тип | Назначение |
|---|---|---|---|---|
| `dense` | all-MiniLM-L6-v2 | 384 | `float` | Семантический candidate retrieval |
| `sparse` | Qdrant/bm25 | — | sparse IDF | Лексический retrieval |
| `colbert` | colbert-ir/colbertv2.0 | 128×N | multivector MaxSim | Late-interaction реранкинг |

### Payload

| Поле | Тип | Описание |
|---|---|---|
| `text` | str | Исходный текст чанка (оригинал, без очистки) |
| `chunk_index` | int | Порядковый номер чанка в документе |
| `headings` | list[str] | Иерархия заголовков (раздел → подраздел) |
| `doc_items` | list[str] | Ссылки на элементы Docling (`#/texts/812`) |
| `filename` | str | Имя исходного файла |
| `page_numbers` | list[int] | Номера страниц чанка |
| `is_table` | bool | `True` если чанк является таблицей |
| `refs` | list[str] | Нормативные ссылки внутри текста |

### Payload индексы

`filename` (KEYWORD), `headings` (KEYWORD), `is_table` (BOOL), `refs` (KEYWORD)

---

## Конфигурация

```python
# MinIO
RAG_DATA_BUCKET     = "ragfiles"
DEV_DATA_MINERU_MD  = "dev_data/mineru_md"
DEV_DATA_DOCLING_JSON = "dev_data/docking_jsons"

# Батчинг
BATCH_SIZE          = 4     # файлов на MinerU батч

# Qdrant
QDRANT_URL          = "http://qdrant:6333"
QDRANT_COLLECTION   = "construction_docs"
QDRANT_DENSE_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
QDRANT_COLBERT_MODEL = "colbert-ir/colbertv2.0"
QDRANT_VECTOR_SIZE  = 384
QDRANT_COLBERT_SIZE = 128
QDRANT_ENCODE_BATCH = 64    # чанков за один encode вызов
QDRANT_UPSERT_BATCH = 32    # точек за один upsert запрос
```

---

## Airflow Connections

| conn_id | Тип | Описание |
|---|---|---|
| `minio` | Amazon Web Services | MinIO endpoint, access/secret key |
| `mineru` | HTTP | `http://mineru-api:8000` |
| `docling` | HTTP | Docling Serve endpoint |


---

## Известные ограничения

- **MinerU pipeline** не освобождает VRAM между задачами. Ограничение через `MINERU_VIRTUAL_VRAM_SIZE`. Для полной очистки требуется `docker compose stop && rm && up`.
- **Qdrant upsert timeout** при больших ColBERT матрицах — решается `QDRANT_UPSERT_BATCH=32` и `timeout=120` на клиенте.
- **Graф связей между нормами** не реализован — ссылки извлекаются в `refs` payload, но граф обхода отсутствует.
- **Терминологический словарь** (`TERM_NORMALIZATION`) применяется только на стороне запроса, не покрывает все расхождения.

---

## Планируемые улучшения

- [ ] Metadata database для отслеживания файлов в MinIO (PostgreSQL)
- [ ] Перенастроить Mineru на хранение методанных в Redis, так как наблюдается поретя данных в случае OMKill
- [ ] Граф обязательных связей между нормативными документами (PostgreSQL / Neo4j)
- [ ] Добавление терминологического словаря по домену строительных норм
- [ ] Автоматическое обновление вектороной базы при изменении документов в MinIO (S3 event → trigger DAG)
- [ ] Очистка памити системы - Mineru оставляет держит в памяти модели и нет поинта для очистки
- [ ] Разгрузить save_docling_results
- [ ] Оптимизация docker images.
- [ ] Автоматическое создание коннектов Airflow
