"""
Batch Document Ingestion Pipeline
==================================
End-to-end pipeline that discovers PDF files in MinIO, converts them to
Markdown via MinerU (OCR), chunks the Markdown hierarchically via Docling,
and upserts dense / sparse / ColBERT vectors into a Qdrant collection.

Pipeline stages
---------------
1. **Discovery** – list buckets, filter supported formats, skip already-processed files.
2. **Batching** – split file keys into fixed-size batches for parallel processing.
3. **MinerU OCR** – submit batches, wait (deferrable sensor), retrieve ``.md`` results.
4. **MinIO upload** – persist ``.md`` files under ``dev_data/mineru_md/``.
5. **Docling chunking** – hierarchical Markdown chunking via async Docling API.
6. **Enrichment** – extract normative references, tag tables, clean OCR artefacts.
7. **Qdrant upsert** – encode with dense / sparse / ColBERT models and upsert points.

Connections required
--------------------
minio, mineru, docling
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import requests
import torch
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.http.hooks.http import HttpHook
from airflow.sdk import dag, task
from common.sensors.docling_sensor import DoclingBatchStatusSensor
from common.sensors.mineru_sensor import MineruBatchStatusSensor
from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding, TextEmbedding
from pendulum import datetime

RAG_DATA_BUCKET = "ragfiles"
DEV_DATA = "dev_data"
DEV_DATA_MINERU_MD = f"{DEV_DATA}/mineru_md"
DEV_DATA_DOCLING_JSON = f"{DEV_DATA}/docling_jsons"
SUP_FORMATS = ["pdf/"]
BATCH_SIZE = 16

QDRANT_URL = "http://qdrant:6333"
QDRANT_COLLECTION = "construction_docs"
QDRANT_DENSE_MODEL = "intfloat/multilingual-e5-large"
# QDRANT_DENSE_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
QDRANT_COLBERT_MODEL = "colbert-ir/colbertv2.0"
QDRANT_VECTOR_SIZE = 1024
QDRANT_ENCODE_BATCH = 16
QDRANT_UPSERT_BATCH = 32
QDRANT_COLBERT_SIZE = 128


# ============================================================
# Helpers
# ============================================================


def del_file(file: str | Path) -> None:
    """Delete a local file if it exists"""
    path = Path(file)
    if path.exists():
        path.unlink()
        logging.info(f">>> Removed existing: {path}")


def load_to_s3(
    hook: S3Hook,
    filepath: str | list[str],
    bucket_name: str = RAG_DATA_BUCKET,
    prefix: str = DEV_DATA_MINERU_MD,
) -> list[str] | None:
    """
    Upload one or more local files to MinIO (S3-compatible storage).

    Calls ``S3Hook.load_file`` with ``replace=True`` for each path, so
    existing objects are overwritten. Errors are logged without re-raising.

    Parameters
    ----------
    hook : S3Hook
        Initialised Airflow ``S3Hook`` connected to MinIO.
    filepath : str or list of str
        Local file path(s) to upload.
    bucket_name : str, optional
        Target MinIO bucket. Default is ``RAG_DATA_BUCKET``.
    prefix : str, optional
        Key prefix (folder path) inside the bucket.
        Default is ``DEV_DATA_MINERU_MD``.

    Returns
    -------
    str or None
        Newline-separated string of uploaded paths on success,
        ``None`` if an exception occurred.
    """
    filepath = filepath if isinstance(filepath, list) else [filepath]
    out = ""
    s3_keys = []
    try:
        for patch in filepath:
            s3_key = f"{DEV_DATA_MINERU_MD}/{Path(patch).name}"
            hook.load_file(
                filename=patch,
                bucket_name=bucket_name,
                key=f"{prefix}/{patch.split('/')[-1]}",
                replace=True,
            )
            out += f"\n{patch}"
            s3_keys.append(s3_key)
        logging.info(f">> File loaded to minio: {out}")
        return s3_keys
    except Exception as ex:
        logging.error(f">> Unknown error loading data to minio: {ex}")


def batch_list(items: list, batch_size: int = BATCH_SIZE) -> list[list]:
    """Split a flat list into fixed-size chunks"""
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def clean_chunk_text(text: str) -> str:
    """
    Remove common OCR artefacts from chunk text before vectorisation.

    Normalises excess whitespace, collapses repeated newlines, fixes
    hyphenated number ranges, and strips repeated pipe characters.

    Parameters
    ----------
    text : str
        Raw OCR text from a document chunk.

    Returns
    -------
    str
        Cleaned text, or the original value if it was empty / falsy.
    """
    if not text:
        return text
    # spaces and hyphenation
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Spaces around the hyphen in numbers: "4 4 4" cannot be fixed, but "44 -45" → "44-45"
    text = re.sub(r"(\d)\s*[-–]\s*(\d)", r"\1–\2", text)
    # OCR special characters
    text = re.sub(r"[|]{2,}", "", text)
    return text.strip()


def execute_refs(text: str) -> list[str]:
    """
    Extract normative document references from chunk text.

    Searches for Russian building-code patterns: СП, СНиП, ГОСТ,
    paragraph numbers, table / figure references, and appendix labels.

    Parameters
    ----------
    text : str
        Plain text of a document chunk.

    Returns
    -------
    list of str
        Deduplicated list of matched reference strings.
        Returns an empty list when no patterns are found.
    """
    patterns = [
        r"[Сс][Пп]\s*\d+[.\d]*"  # СП 63.13330
        r"[СсГг][НнОо][ИиСс][ПпТт]\s*[\d.\-]+"  # СНиП, ГОСТ
        r"[Пп]\.?\s*\d+[\.\d]*"  # п. 3.45
        r"[Тт]абл(?:ица|\.)\s*\d+",  # таблица 7 / табл. 7
        r"[Рр]ис(?:унок|\.)\s*\d+",  # рисунок 3
        r"[Пп]риложени[еяй]\s*[А-ЯA-Z\d]+",  # приложение А
    ]
    refs = []
    for pat in patterns:
        refs.extend(re.findall(pat, text))
    return list(set(refs))


PROVIDERS = (
    ["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
)


@lru_cache(maxsize=1)
def get_dense_model() -> TextEmbedding:
    return TextEmbedding(QDRANT_DENSE_MODEL, providers=PROVIDERS)


@lru_cache(maxsize=1)
def get_sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding("Qdrant/bm25")  # BM25 — CPU-only, памяти мало


@lru_cache(maxsize=1)
def get_colbert_model() -> LateInteractionTextEmbedding:
    return LateInteractionTextEmbedding(QDRANT_COLBERT_MODEL, providers=PROVIDERS)


# ============================================================
# DAG
# ============================================================


@dag(
    dag_id="batch_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["data_pipeline"],
    doc_md=__doc__,
)
def batch_pipeline():
    """Batch document processing pipeline — see module docstring for full description."""

    # ------------------------------------------------------------------------------------------------
    # 1. Discovery
    # ------------------------------------------------------------------------------------------------

    @task
    def get_buckets_data(
        bucket_name_filter: str | list[str] | None = None,
    ) -> dict[str, list[str]]:
        """
        List all accessible MinIO buckets and their top-level prefixes.

        Connects via the ``minio`` Airflow connection, retrieves all buckets,
        optionally filters by name, then lists first-level prefixes within
        each bucket.

        Parameters
        ----------
        bucket_name_filter : str or list of str or None, optional
            If provided, only buckets whose names appear in this filter are
            included. Pass ``None`` to include all buckets.

        Returns
        -------
        dict of {str : list of str}
            Mapping ``bucket_name -> [prefix, ...]``.
            Example: ``{"ragfiles": ["pdf/", "images/"]}``.

        Notes
        -----
        Exceptions are caught via ``logging.exception``; the task returns
        an empty list implicitly, allowing upstream retry logic to handle
        the failure.
        """
        hook = S3Hook(aws_conn_id="minio")
        try:
            buckets_data = hook.get_conn().list_buckets()
            buckets = buckets_data.get("Buckets", [])

            if bucket_name_filter:
                """
                "Buckets": [
                    {
                        "Name": "...",
                        "CreationDate": "..."
                    }, ...
                ],
                """
                target = (
                    [bucket_name_filter]
                    if isinstance(bucket_name_filter, str)
                    else bucket_name_filter
                )
                names = [b["Name"] for b in buckets if b["Name"] in target]
            else:
                names = [b["Name"] for b in buckets]

            logging.info(f"Found buckets: {names}")
            data = defaultdict(list)

            for name in names:
                prefs = hook.list_prefixes(bucket_name=name, delimiter="/")
                data[name].extend(prefs)

            logging.info(f"Found prefixes {data}")
            return data
        except Exception as ex:
            logging.exception(ex)
            return []

    @task
    def list_files_to_process(
        bucket: str,
        prefix: str | None = None,
        skip_process: bool = True,
    ) -> list[str]:
        """
        Collect all S3 object keys eligible for processing from a bucket.

        Parameters
        ----------
        bucket : str
            MinIO bucket name to scan.
        prefix : str or None, optional
            Restrict the scan to a specific prefix. If ``None``, only
            prefixes listed in ``SUP_FORMATS`` are scanned.
        skip_process : bool, optional
            If ``True`` (default), skips files for which a corresponding
            ``.md`` already exists under ``DEV_DATA_MINERU_MD``.
            Set to ``False`` to reprocess all files unconditionally.

        Returns
        -------
        list of str
            S3 object keys not yet processed, e.g. ``["pdf/report.pdf"]``.
        """
        hook = S3Hook(aws_conn_id="minio")

        if prefix:
            candidates = hook.list_keys(bucket_name=bucket, prefix=prefix) or []
        else:
            candidates = []
            for pref in hook.list_prefixes(bucket_name=bucket, delimiter="/"):
                if pref in SUP_FORMATS:
                    candidates.extend(
                        hook.list_keys(bucket_name=bucket, prefix=pref) or []
                    )

        if not skip_process:
            return candidates

        to_process = []
        for key in candidates:
            stem = Path(key).stem
            md_key = f"{DEV_DATA_MINERU_MD}/{stem}.md"

            if hook.check_for_key(key=md_key, bucket_name=bucket):
                logging.info(f">>> Skipping already processed: {key} (found {md_key})")
            else:
                to_process.append(key)

        logging.info(f">>> {len(candidates)} total, {len(to_process)} to process")
        return to_process

    @task
    def check_mineru_health() -> int:
        """Verify that the MinerU API is reachable and healthy"""
        hook_mineru = HttpHook(method="GET", http_conn_id="mineru")
        req = hook_mineru.run(endpoint="/health")
        return req.status_code

    # ------------------------------------------------------------------------------------------------
    # 2. Batching
    # ------------------------------------------------------------------------------------------------

    @task
    def create_file_batches(
        files: list[str],
        batch_size: int = BATCH_SIZE,
    ) -> list[list[str]]:
        """
        Partition a flat list of file keys into fixed-size batches.

        Parameters
        ----------
        files : list of str
            S3 object keys to partition.
        batch_size : int, optional
            Maximum number of files per batch. Default is ``BATCH_SIZE``.

        Returns
        -------
        list of list of str
            Nested list where each inner list is one batch,
            e.g. ``[["f1","f2","f3","f4"], ["f5","f6"]]``.
        """
        batches = batch_list(files, batch_size)
        logging.info(f">>> {len(files)} files split into {len(batches)} batches")

        for i, batch in enumerate(batches):
            logging.info(f"    Batch {i}: {len(batch)} files -> {batch}")
        return batches

    # ------------------------------------------------------------------------------------------------
    # 3. Submit batch to MinerU
    # ------------------------------------------------------------------------------------------------

    @task()
    def batch_mineru_submit(
        health_status: int,
        qerant_status: str,
        file_keys: list[str],
        bucket_name: str,
    ) -> list[str]:
        """
        Download files from MinIO and submit them to the MinerU OCR API.

        Guards against an unhealthy MinerU instance before any I/O.
        Each file is downloaded to ``/tmp``, posted to ``POST /tasks``,
        and cleaned up afterwards.

        Parameters
        ----------
        health_status : int
            HTTP status code from ``ckeck_mineru_health``.
            Must be ``200``; otherwise ``ValueError`` is raised.
        file_keys : list of str
            S3 object keys to process in this batch.
        bucket_name : str
            Source MinIO bucket.

        Returns
        -------
        list of str
            MinerU ``task_id`` values, one per submitted file.

        Raises
        ------
        ValueError
            If ``health_status != 200``.
        """
        if health_status != 200:
            raise ValueError(f"MinerU unhealthy: {health_status}")

        if not qerant_status:
            raise ValueError("Qdrant unhealthy")

        hook_minio = S3Hook(aws_conn_id="minio")
        task_ids: list[str] = []

        for file_key in file_keys:
            # ----- download -----
            file_name = Path("/tmp") / file_key.split("/")[-1]
            del_file(file_name)

            file_path = hook_minio.download_file(
                key=file_key,
                bucket_name=bucket_name,
                local_path="/tmp",
                preserve_file_name=True,
                use_autogenerated_subdir=False,
            )
            logging.info(f">>> Downloaded: {file_path}")

            # ----- submit -----
            with open(file_path, "rb") as f:
                resp = requests.post(
                    "http://mineru-api:8000/tasks",
                    files={"files": f},
                    data={
                        "lang_list": "east_slavic",
                        "backend": "pipeline",
                        "parse_method": "ocr",
                        "formula_enable": "true",
                        "table_enable": "true",
                        "return_md": "true",
                        "return_middle_json": "false",
                        "return_model_output": "false",
                        "return_content_list": "false",
                        "return_images": "false",
                    },
                )
            resp.raise_for_status()
            tid = resp.json()["task_id"]
            task_ids.append(tid)
            logging.info(f">>> Submitted {file_key} -> task_id: {tid}")

            del_file(file_path)

        logging.info(f">>> Batch submitted: {task_ids}")
        return task_ids

    # ------------------------------------------------------------------------------------------------
    # 4. Save MinerU results (runs AFTER sensor confirms completion)
    # ------------------------------------------------------------------------------------------------

    @task()
    def save_mineru_results(mineru_task_ids: list[str]) -> list[str]:
        """
        Fetch completed MinerU results and write Markdown files to ``/tmp``.

        Calls ``GET /tasks/{task_id}/result`` for every task ID and writes
        the returned ``md_content`` to ``/tmp/<name>.md``.

        Parameters
        ----------
        mineru_task_ids : list of str
            Task IDs produced by ``batch_mineru_submit`` for one batch.

        Returns
        -------
        list of str
            Absolute paths of the written ``.md`` files.
        """
        hook_mineru = HttpHook(method="GET", http_conn_id="mineru")
        all_paths: list[str] = []

        for task_id in mineru_task_ids:
            req = hook_mineru.run(endpoint=f"/tasks/{task_id}/result")
            req.raise_for_status()
            results = req.json().get("results", {})

            for name, payload in results.items():
                file_path = f"/tmp/{name}.md"
                del_file(file_path)

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(payload["md_content"])
                all_paths.append(file_path)

        logging.info(f">>> Saved {len(all_paths)} .md files from batch")
        return all_paths

    # ------------------------------------------------------------------------------------------------
    # 5. Upload .md to MinIO
    # ------------------------------------------------------------------------------------------------

    @task
    def load_md_to_minio(mineru_result: list[str]) -> list[str]:
        """
        Upload a batch of local ``.md`` files to MinIO and pass paths downstream.

        Parameters
        ----------
        mineru_result : list of str
            Local ``/tmp/*.md`` paths returned by ``save_mineru_results``.

        Returns
        -------
        list of str
            The same paths passed through unchanged so downstream tasks
            can consume the local files without a second download.
        """
        hook = S3Hook(aws_conn_id="minio")
        s3_keys = load_to_s3(hook, mineru_result, prefix=DEV_DATA_MINERU_MD)
        return s3_keys

    # ------------------------------------------------------------------------------------------------
    # 6. Chanking
    # ------------------------------------------------------------------------------------------------
    @task()
    def single_docling():
        """
        List all ``.md`` files in MinIO and batch them for Docling.

        Reads directly from ``DEV_DATA_MINERU_MD`` in ``RAG_DATA_BUCKET``,
        bypassing the MinerU pipeline. Intended for standalone Docling
        tests or reprocessing runs.

        Returns
        -------
        list of list of str
            S3 keys grouped into batches of ``BATCH_SIZE``.
        """
        hook_minio = S3Hook(aws_conn_id="minio")

        keys = (
            hook_minio.list_keys(bucket_name=RAG_DATA_BUCKET, prefix=DEV_DATA_MINERU_MD)
            or []
        )
        return [keys[i : i + BATCH_SIZE] for i in range(0, len(keys), BATCH_SIZE)]

    @task()
    def docling_chunk_submit(
        mineru_result: list[str],
        bucket_name: str,
    ) -> list[str]:
        """
        Download ``.md`` files from MinIO and submit them to the Docling chunking API.

        Posts each file to ``POST /v1/chunk/hierarchical/file/async`` with
        hierarchical chunking settings; Markdown tables are preserved as
        structured chunks.

        Parameters
        ----------
        mineru_result : list of str
            S3 keys of ``.md`` files to chunk.
        bucket_name : str
            Source MinIO bucket used for downloads.

        Returns
        -------
        list of str
            Docling ``task_id`` values, one per submitted file.
        """
        chunk_task_ids: list[str] = []
        hook_docling = HttpHook(method="POST", http_conn_id="docling")
        hook_minio = S3Hook(aws_conn_id="minio")

        for filepath in mineru_result:
            file_name = Path("/tmp") / filepath.split("/")[-1]
            del_file(file_name)

            file_path = hook_minio.download_file(
                key=filepath,
                bucket_name=bucket_name,
                local_path="/tmp",
                preserve_file_name=True,
                use_autogenerated_subdir=False,
            )
            logging.info(f">>> Downloaded: {file_path}")

            path = Path(file_path)
            with open(file_path, "rb") as f:
                resp = hook_docling.run(
                    endpoint="/v1/chunk/hierarchical/file/async",
                    files={
                        "files": (
                            path.name,
                            f,
                            "text/markdown",
                        )
                    },
                    data={
                        "convert_from_formats": "json_docling",
                        "convert_do_ocr": "false",
                        "convert_do_table_structure": "false",
                        "convert_include_images": "false",
                        "convert_do_formula_enrichment": "false",
                        "target_type": "inbody",
                        "include_converted_doc": "false",
                        "chunking_use_markdown_tables": "true",
                        "chunking_include_raw_text": "false",
                    },
                )
                resp.raise_for_status()
                task_id = resp.json()["task_id"]
                chunk_task_ids.append(task_id)
                logging.info(f">>> Chunk submit: {path.name} -> task_id: {task_id}")

        logging.info(f">>> Chunk batch submitted: {chunk_task_ids}")

        return chunk_task_ids

    @task
    def save_docling_results(docling_task_ids: list[str]) -> list[str]:
        """
        Retrieve Docling chunking results, enrich chunks, and persist to MinIO.

        Fetches each completed task, separates text chunks from table chunks,
        extracts normative references via ``execute_refs``, and serialises
        the enriched payload to ``/tmp/<stem>.json`` before uploading to
        ``DEV_DATA_DOCLING_JSON``.

        Parameters
        ----------
        docling_task_ids : list of str
            Task IDs produced by ``docling_chunk_submit`` for one batch.

        Returns
        -------
        list of str
            Absolute paths of the written ``.json`` files.
        """
        hook_docling = HttpHook(method="GET", http_conn_id="docling")
        hook_minio = S3Hook(aws_conn_id="minio")
        out_paths: list[str] = []

        for task_id in docling_task_ids:
            resp = hook_docling.run(endpoint=f"/v1/result/{task_id}")
            resp.raise_for_status()
            result = resp.json()

            all_chunks = result.get("chunks", [])
            documents = result.get("documents", [])

            if not documents:
                logging.warning(f">>> No documents for task_id={task_id}. Skip")
                continue

            for doc in documents:
                file_name = doc.get("content", {}).get("filename")
                stem = Path(file_name).stem if "." in file_name else file_name

                doc_chunks = [
                    chunk
                    for chunk in all_chunks
                    if chunk.get("meta", {}).get("origin", {}).get("filename")
                    == file_name
                ]
                if not doc_chunks and len(documents) == 1:
                    doc_chunks = all_chunks

                if not doc_chunks:
                    logging.warning(f">>> No chunks for {file_name}. Skip")
                    continue

                """
                Разделить текстовые чанки и таблицы
                Таблицы хранятся как отдельные точки в Qdrant - это позволяет
                искать по ним отдельно (фильтр is_table=True) и не терять
                нормативные значения внутри усредненного текстового вектора.
                """
                text_chunks = [c for c in doc_chunks if c.get("type") != "table"]
                table_chunks = [c for c in doc_chunks if c.get("type") == "table"]

                for tc in table_chunks:
                    tc["is_table"] = True
                    # Префикс помогает модели понять контекст при энкодинге
                    tc["text"] = f"[ТАБЛИЦА] {tc.get('text', '')}"
                """
                Добавить нормативные ссылки в каждый чанк
                Ссылки извлекаются из исходного текста и сохраняются в payload,
                что позволяет впоследствии строить граф связей между нормами.
                """
                all_enriched: list[dict] = []
                for chunk in text_chunks + table_chunks:
                    chunk["is_table"] = chunk.get("is_table", False)
                    chunk["refs"] = execute_refs(chunk.get("text", ""))
                    all_enriched.append(chunk)

                json_path = f"/tmp/{stem}.json"
                del_file(json_path)
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump(all_enriched, jf, ensure_ascii=False, indent=2)

                load_to_s3(
                    hook=hook_minio, filepath=json_path, prefix=DEV_DATA_DOCLING_JSON
                )
                out_paths.append(json_path)
                logging.info(f">>> Saved {len(doc_chunks)} chunks -> {json_path}")

        return out_paths

    @task()
    def create_qdrant_collection() -> str:
        """
        Ensure the Qdrant collection exists with the required vector configuration.

        Creates ``QDRANT_COLLECTION`` with three named vector spaces:
        ``dense`` (384-dim cosine), ``colbert`` (128-dim multi-vector MaxSim),
        and ``sparse`` (BM25 IDF), plus payload indices for fast filtering.
        Skips creation silently if the collection already exists.

        Returns
        -------
        str
            Name of the collection (``QDRANT_COLLECTION``).
        """
        from qdrant_client import QdrantClient
        from qdrant_client.models import (
            Distance,
            Modifier,
            MultiVectorComparator,
            MultiVectorConfig,
            PayloadSchemaType,
            SparseVectorParams,
            VectorParams,
        )

        client = QdrantClient(url=QDRANT_URL, timeout=20)

        existing = {c.name for c in client.get_collections().collections}
        if QDRANT_COLLECTION in existing:
            logging.info(f">>> Collection '{QDRANT_COLLECTION}' already exists — skip")
            return QDRANT_COLLECTION

        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config={
                "dense": VectorParams(
                    size=QDRANT_VECTOR_SIZE, distance=Distance.COSINE
                ),
                "colbert": VectorParams(
                    size=QDRANT_COLBERT_SIZE,
                    distance=Distance.COSINE,
                    multivector_config=MultiVectorConfig(
                        comparator=MultiVectorComparator.MAX_SIM,
                    ),
                ),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(modifier=Modifier.IDF),
            },
        )
        for field, schema in [
            ("filename", PayloadSchemaType.KEYWORD),
            ("headings", PayloadSchemaType.KEYWORD),
            ("is_table", PayloadSchemaType.BOOL),
            ("refs", PayloadSchemaType.KEYWORD),
        ]:
            client.create_payload_index(QDRANT_COLLECTION, field, schema)

        logging.info(f">>> Collection '{QDRANT_COLLECTION}' created")
        return QDRANT_COLLECTION

    @task()
    def save_to_qdrant(docling_json_paths: list[str]) -> int:
        """
        Encode enriched chunks with three models and upsert them into Qdrant.

        Loads each ``.json`` file, encodes text in windows of
        ``QDRANT_ENCODE_BATCH`` using ``all-MiniLM-L6-v2`` (dense),
        ``Qdrant/bm25`` (sparse), and ``colbertv2.0`` (late-interaction),
        then upserts points in sub-batches of ``QDRANT_UPSERT_BATCH``.

        Point IDs are deterministic UUIDs derived from
        ``filename + chunk_index``, so re-runs overwrite existing points
        rather than creating duplicates.

        Parameters
        ----------
        docling_json_paths : list of str
            Local paths to enriched chunk JSON files produced by
            ``save_docling_results``.

        Returns
        -------
        int
            Total number of points upserted across all files.
        """
        import gc

        import torch
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct, SparseVector

        # Диагностика GPU внутри воркера
        logging.info(f">>> CUDA available: {torch.cuda.is_available()}")
        logging.info(f">>> PROVIDERS: {PROVIDERS}")
        if torch.cuda.is_available():
            logging.info(f">>> GPU: {torch.cuda.get_device_name(0)}")

        client = QdrantClient(
            url=QDRANT_URL,
            timeout=120,
        )

        dense_model = get_dense_model()
        sparse_model = get_sparse_model()
        colbert_model = get_colbert_model()
        total_upserted = 0

        for json_path in docling_json_paths:
            path = Path(json_path)
            with open(json_path, encoding="utf-8") as f:
                chunks = json.load(f)

            if not chunks:
                logging.warning(f">>> Empty chunk list in {json_path}, skip")
                continue

            logging.info(f">>> Processing {len(chunks)} chunks from {path.name}")

            # -----Iterate in QDRANT_ENCODE_BATCH sized windows -----
            # Цикл 1 (ENCODE_BATCH=64):  контролирует RAM модели при энкодинге
            # Цикл 2 (UPSERT_BATCH=256): контролирует размер HTTP-запроса к Qdrant
            for enc_start in range(0, len(chunks), QDRANT_ENCODE_BATCH):
                # ----- 1) Batching and Cheaning text into chanks -----
                enc_batch = chunks[enc_start : enc_start + QDRANT_ENCODE_BATCH]
                texts = [
                    "passage: " + clean_chunk_text(c.get("text", "")) for c in enc_batch
                ]

                # ----- 2) Vectors -----
                dense_vecs = list(dense_model.embed(texts))  # list[ndarray(384,)]
                sparse_vecs = list(sparse_model.embed(texts))  # list[SparseEmbedding]
                colbert_vecs = list(
                    colbert_model.embed(texts)
                )  # list[ndarray(n_tok, 128)]
                # list[ndarray(n_tokens, 128)] — рахмер матрицы разный для каждого чанка

                # ----- 3) Build PointStruct list -----
                points: list[PointStruct] = []
                for i, chunk in enumerate(enc_batch):
                    # Deterministic UUID from filename + chunk_index so re-runs
                    # overwrite the same point instead of creating duplicates.
                    point_id = str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"{path.stem}:{chunk.get('chunk_index', enc_start + i)}",
                        )
                    )

                    sv = sparse_vecs[i]
                    cv = colbert_vecs[i]
                    points.append(
                        PointStruct(
                            id=point_id,
                            vector={
                                "dense": dense_vecs[i].tolist(),
                                "sparse": SparseVector(
                                    indices=sv.indices.tolist(),
                                    values=sv.values.tolist(),
                                ),
                                # ColBERT: матрица (n_tokens × 128) → list[list[float]]
                                "colbert": cv.tolist(),
                            },
                            payload={
                                "text": chunk.get(
                                    "text", ""
                                ),  # оригинал для отображения
                                "chunk_index": chunk.get("chunk_index"),
                                "headings": chunk.get(
                                    "headings", []
                                ),  # оригинал для отображения
                                "doc_items": chunk.get("doc_items", []),
                                "filename": chunk.get(
                                    "filename", path.stem
                                ),  # для фильтрации по файлу
                                "is_table": chunk.get("is_table", False),
                                "refs": chunk.get("refs", []),
                            },
                        )
                    )

                # ----- Upsert in sub-batches to control request size -----
                for ups_start in range(0, len(points), QDRANT_UPSERT_BATCH):
                    sub = points[ups_start : ups_start + QDRANT_UPSERT_BATCH]
                    client.upsert(collection_name=QDRANT_COLLECTION, points=sub)
                    total_upserted += len(sub)
                    logging.info(
                        f">>> Upserted {len(sub)} points(file={path.name},"
                        f"enc_offset={enc_start},"
                        f"ups_offset={ups_start})"
                    )
                del dense_vecs, sparse_vecs, colbert_vecs, points, texts, enc_batch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            logging.info(f">>> Done: {path.name}, running total={total_upserted}")

        return total_upserted

    # ==========================================================
    # Graph
    # ==========================================================
    # ----- 0) Docling single precess -----
    md_loaded = single_docling()
    docling_task_ids = docling_chunk_submit.partial(bucket_name=RAG_DATA_BUCKET).expand(
        mineru_result=md_loaded
    )

    docling_wait = DoclingBatchStatusSensor.partial(
        task_id="wait_docling_batch",
        docling_conn_id="docling",
        poll_interval=10,
    ).expand(external_task_ids=docling_task_ids)
    docling_chunks = save_docling_results.expand(docling_task_ids=docling_wait.output)
    create_qdrant_collection() >> save_to_qdrant.expand(
        docling_json_paths=docling_chunks
    )
    # ------------------------------------------------------------
    # ----- 1) Discovery -----
    # get_buckets_data()
    # mineru_health = check_mineru_health()
    # qdrant_health = create_qdrant_collection()
    # files_task = list_files_to_process(bucket=RAG_DATA_BUCKET)
    # # ----- 2) Split into batches of BATCH_SIZE -----
    # file_batches = create_file_batches(files_task)
    # # [["f1","f2",...,"f8"], ["f9","f10",...], ...]

    # # ----- 3) Submit each batch to MinerU (one dynamic task per batch) -----
    # mineru_task_ids = batch_mineru_submit.partial(
    #     bucket_name=RAG_DATA_BUCKET,
    #     health_status=mineru_health,
    #     qerant_status=qdrant_health,
    # ).expand(file_keys=file_batches)
    # # [["tid1","tid2",...], ["tid9",...], ...]

    # # ----- 4) Deferrable sensor - async wait for each batch (frees workers) -----
    # mineru_wait = MineruBatchStatusSensor.partial(
    #     task_id="wait_mineru_batch",
    #     mineru_conn_id="mineru",
    #     poll_interval=60,
    # ).expand(external_task_ids=mineru_task_ids)
    # # output: [["tid1","tid2",...], ["tid9",...], ...]

    # # ----- 5) Save .md files per batch -----
    # md_files = save_mineru_results.expand(mineru_task_ids=mineru_wait.output)

    # # ----- 6) Upload .md to MinIO per batch -----
    # md_loaded = load_md_to_minio.expand(mineru_result=md_files)

    # # ----- 7) Chanking -----
    # docling_task_ids = docling_chunk_submit.partial(bucket_name=RAG_DATA_BUCKET).expand(
    #     mineru_result=md_loaded
    # )

    # # ----- 8) Deferrable sensor - async wait for each batch (frees workers) -----
    # docling_wait = DoclingBatchStatusSensor.partial(
    #     task_id="wait_docling_batch",
    #     docling_conn_id="docling",
    #     poll_interval=30,
    # ).expand(external_task_ids=docling_task_ids)

    # # ----- 6) Upload .jsom to MinIO per batch -----
    # docling_chunks = save_docling_results.expand(docling_task_ids=docling_wait.output)
    # # ----- 8) Qdrant  -----
    # save_to_qdrant.expand(docling_json_paths=docling_chunks)


batch_pipeline()
