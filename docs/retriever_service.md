# Retriever Service

Система семантического поиска по нормативно-техническим документам (СП, СНиП, ГОСТ) с гибридным поиском и rerank-моделями.

---

## Архитектура

```
Streamlit UI
    │
    ▼
Query Rewriter      ← переформулирование вопроса (vllm-light)
    │
    ▼
Qdrant Retriever    ← гибридный поиск: dense + sparse + ColBERT
    │
    ▼
Qdrant              ← векторная БД с мультивекторным индексом
```

---

## Стек

| Компонент | Технология | Назначение |
|---|---|---|
| UI | Streamlit | Веб-интерфейс поиска с параметрами |
| Query Rewriter | vllm-light (OpenAI-compatible) | Переформулирование запросов, dim=1024 |
| Dense embeddings | `BAAI/bge-m3` | Семантический поиск, dim=1024 |
| Sparse embeddings | `BAAI/bge-m3` (BM25) | Лексический поиск (BM25) |
| Late-interaction | `BAAI/bge-m3` (ColBERT) | Реранкинг через MaxSim, dim=1024×N |
| Векторная БД | Qdrant | Гибридный поиск: dense + sparse + ColBERT |

---

## Структура app.py

### QueryRewriter

Переформулирует пользовательский вопрос в поисковый запрос для векторной базы.

**Промпт:**
- Вернуть ТОЛЬКО переформулированный запрос без пояснений
- Убрать разговорные обороты («расскажи мне», «хочу узнать»)
- Сохранить технические термины, номера стандартов, классы материалов
- Добавить релевантные синонимы и уточнения области применения
- Длина — не более двух предложений

**Ошибка полдключения:** при недоступности vllm-light возвращает исходный запрос.

**Конфигурация:**
- `REWRITER_BASE_URL = "http://vllm-light:8020/v1"`
- `REWRITER_MODEL = "query-rewriter"`
- `temperature=0.2`, `max_tokens=256`, `timeout=5.0`

---

### SidebarParams

Боковая панель с параметрами поиска:

| Параметр | Значения | Описание |
|---|---|---|
| **mode** | `hybrid`, `dense`, `sparse` | Режим поиска |
| **top_k** | 1–20 | Финальное количество результатов |
| **prefetch_k** | ≥top_k | Кандидатов для ColBERT rerank (только hybrid) |
| **only_tables** | Все / Текст / Таблицы | Фильтр по типу чанка |
| **filename_filter** | строка | Фильтр по имени файла (filename) |
| **use_rewriter** | toggle | Переформулировать запрос через LLM |

**Режим hybrid:**
- `prefetch_k` по умолчанию = `top_k × 4`, максимум 100
- Dense Prefetch (prefetch_k/2) + Sparse Prefetch (prefetch_k) → ColBERT rerank (top_k)

**Режимы dense/sparse:**
- Прямой ANN/точный поиск без rerank
- `prefetch_k = top_k × 4`

---

### SearchRunner

Оркестрирует поиск:

1. **Maybe rewrite** — вызывает `QueryRewriter.rewrite()` при включённом `use_rewriter`
2. **Fetch results** — вызывает `QdrantRetriever.search()` с параметрами из UI
3. **Validate and store** — проверяет порог `SCORE_THRESHOLD = 4.0` (hybrid) или 0.02 (dense/sparse)

**Поведение при пустом результате:**
- Если `results` пуст ИЛИ `results[0].score < SCORE_THRESHOLD`:
  - Показать warning: «Документ по этой теме отсутствует в базе»
  - Отобразить лучший скор и порог
  - Вызвать `st.stop()` для прекращения выполнения

---

### ResultsView

Отображает метрики и список найденных чанков.

**Метрики (строка):**
- Найдено — количество результатов
- Режим — режим поиска (HYBRID/DENSE/SPARSE)
- prefetch_k — только для hybrid
- Таблиц — количество таблиц в результатах
- Макс. score — максимальный балл

**Отображение чанка:**
- Collapsible expander с заголовком:
  - 🟢/🟡/🔴 — иконка по score (threshold-based)
  - Номер, score, тип (Текст/Таблица), filename, chunk_index
- **Headings** — иерархия разделов через ` › `
- **Text** — синий фон для текста, monospace для таблиц
- **Refs** — синие бейджи нормативных ссылок
- **Metadata** — JSON в pop-over (id, score, filename, headings, refs и т.д.)

---

## Методы поиска

### hybrid (рекомендуется)

```
dense Prefetch (prefetch_k/2)  ─┐
                                ├► ColBERT MaxSim rerank → top_k
sparse Prefetch (prefetch_k)   ─┘
```

**Преимущества:**
- Широкий recall от двух векторных сигналов (dense + sparse)
- Точный rerank на токен-уровне через ColBERT (late interaction)
- Редуцирование шумовых чанков

**Пороги:**
- Score ≥ 15 — 🟢 (высокая релевантность)
- Score 8–15 — 🟡 (средняя)
- Score < 8 — 🔴 (низкая)

---

### dense

Прямой ANN-поиск по dense-вектору через HNSW-индекс.

**Модель:** `BAAI/bge-m3` (multilingual), dim=1024

**Пороги:**
- Score ≥ 0.02 — 🟢
- Score 0.01–0.02 — 🟡
- Score < 0.01 — 🔴

---

### sparse

Точный лексический поиск по BM25-вектору (инвертированный индекс).

**Модель:** `BAAI/bge-m3` (sparse tokenizer), BM25 weight vector

**Пороги:** аналогично dense

---

**ONNX Runtime:**
- Автоматически выбирает `CUDAExecutionProvider` при наличии GPU
- Fallback на `CPUExecutionProvider`

**Кэширование модели:**
- `@lru_cache(maxsize=1)` на `get_bge_m3()` — один инстанс модели на сессию

---

## Структура Qdrant коллекции

**Коллекция:** `construction_docs`

### Векторы

| Имя | Модель | Dim | Тип | Назначение |
|---|---|---|---|---|
| `dense` | BAAI/bge-m3 | 1024 | `float` | Семантический candidate retrieval |
| `sparse` | BAAI/bge-m3 (BM25) | — | sparse IDF | Лексический retrieval |
| `colbert` | BAAI/bge-m3 (late-interaction) | 1024×N | multivector MaxSim | Late-interaction реранкинг |

### Payload (из data_pipline.md)

| Поле | Тип | Описание |
|---|---|---|
| `text` | str | Исходный текст чанка |
| `chunk_index` | int | Порядковый номер чанка |
| `headings` | list[str] | Иерархия заголовков |
| `filename` | str | Имя исходного файла |
| `page_numbers` | list[int] | Номера страниц |
| `is_table` | bool | Флаг таблицы |
| `refs` | list[str] | Нормативные ссылки (СП, СНиП, ГОСТ) |

### Payload индексы

`filename` (KEYWORD), `headings` (KEYWORD), `is_table` (BOOL), `refs` (KEYWORD)

---

## Известные ограничения

| Проблема | Описание | Статус |
|---|---|---|
| **Отсутствие Ref Expansion** | Граф связей между нормативными документами не построен — `refs` извлекаются, но обход отсутствует | 📋 Backlog |
| **Нет Topics в payload** | В payload отсутствует поле `topics` для pre-filter фильтрации по тематикам | 📋 Backlog |
| **Терминология** | Не покрывает все расхождения терминов — требуется словарь по строительным нормам | 📋 Backlog |

---

## Планируемые улучшения

| ID | Название | Файл | Описание |
|---|---|---|---|
| **1.1** | Раздельные пороги | `app.py` | Меньше ложных срабатываний: пороги для `hybrid`, `dense`, `sparse` разные |
| **1.2** | Sufficiency Check | `app.py` | Честный ответ при пустой базе: проверка `SCORE_THRESHOLD` перед показом результатов |
| **1.3** | Нормализация refs | `pipeline` | Разблокирует Ref Expansion: стандартизация форматов (СП 256.1325800.2016 → СП 256.1325800.2016) |
| **1.4** | Query Router | `app.py` | Точный поиск по номерам норм: direct lookup для `СП XX.XXX.XXX.XXX` → bypass semantic search |
| **2.1** | Query Decomposition | `app.py` | Покрывает размытые запросы: разбиение на подзапросы и объединение результатов |
| **2.2** | Ref Expansion | `retriever-2.py` | Ответы из связанных норм: обход графа `refs` в Qdrant |
| **2.3** | Relevance Judge | `app.py` | Фильтрует шумовые чанки: классификатор релевантности (LLM or small cross-encoder) |
| **3.1** | Refs mandatory/cross | `pipeline` | Умный обход ссылок: обязательные кросс-ссылки vs рекомендательные |
| **3.2** | Topics в payload | `pipeline` | Точная pre-filter фильтрация: добавление `topics` как indexed payload field |
| **3.3** | Cross-encoder rerank | `retriever-2.py` | Качество dense/sparse: BERT-based reranker вместо ColBERT для максимального качества |

| ID | Название | Файл | Описание |
|---|---|---|---|
| **1.1** | Refs mandatory Table | `pipeline/cleaner.py` | Обязательные таблицы, должны попадать в конеткс и выводиться в ответе |
| **1.2** | Refs mandatory/cross | `pipeline` | Умный обход ссылок: по обязательным будет вестьсь поиск на 1 уровень vs рекомендательные пока не будут использоваться|
| **1.2** | Refs mandatory/cross | `pipeline` | Умный обход ссылок: по обязательным будет вестьсь поиск на 1 уровень vs рекомендательные пока не будут использоваться|
---
## Заметки по выполнению
### Список слов, которые заставляют ссылать на норму в тексте 
- в соответствии с
- по
