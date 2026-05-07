from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import onnxruntime as ort
import torch
from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Prefetch,
    SparseVector,
)

QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "construction_docs"

DENSE_MODEL = "intfloat/multilingual-e5-large"
SPARSE_MODEL = "Qdrant/bm25"
COLBERT_MODEL = "colbert-ir/colbertv2.0"

DENSE_SIZE = 1024


PROVIDERS = (
    ["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
)
SearchMode = Literal["hybrid", "dense", "sparse"]

print(ort.get_available_providers())
# ==================================
# Singleton-models
# ==================================


@lru_cache(maxsize=1)
def get_dense_model() -> TextEmbedding:
    return TextEmbedding(DENSE_MODEL, providers=PROVIDERS)


@lru_cache(maxsize=1)
def get_sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(SPARSE_MODEL)


@lru_cache(maxsize=1)
def get_colbert_model() -> LateInteractionTextEmbedding:
    return LateInteractionTextEmbedding(COLBERT_MODEL, providers=PROVIDERS)


@dataclass
class RetrievalResult:
    id: str
    score: float
    text: str
    filename: str
    headings: list[str]
    is_table: bool
    refs: list[str]
    chunk_index: int | None = None


# ==================================
# Retriever
# ==================================


class QdrantRetriever:
    """
    Retriever для коллекции construction_docs.

    Режимы поиска
    -------------
    hybrid (рекомендуется)
        dense Prefetch + sparse Prefetch → ColBERT rerank (MaxSim).
        Широкий recall от двух сигналов + точный rerank на токен-уровне.

    dense
        Только all-MiniLM-L6-v2 (ANN по HNSW).

    sparse
        Только BM25 (точное вхождение терминов).

    Важно
    -----
    ColBERT-вектор ("colbert") должен быть создан с hnsw_config.m=0 —
    он не используется для ANN-поиска, только для rerank через MaxSim.
    """

    def __init__(
        self,
        url: str = QDRANT_URL,
        collection: str = QDRANT_COLLECTION,
        timeout: int = 30,
    ):
        self.client = QdrantClient(url=url, timeout=timeout)
        self.collection = collection

    def search(
        self,
        query: str,
        top_k: int = 10,
        prefetch_k: int = 40,
        mode: SearchMode = "hybrid",
        only_tables: bool | None = None,
        filename_filter: str | None = None,
    ) -> list[RetrievalResult]:

        qdrant_filter = self._build_filter(only_tables, filename_filter)

        if mode == "dense":
            return self._search_dense(query, top_k, qdrant_filter)
        elif mode == "sparse":
            return self._search_sparse(query, top_k, qdrant_filter)
        else:
            return self._search_hybrid_rerank(query, top_k, prefetch_k, qdrant_filter)

    @staticmethod
    def _build_filter(only_tables, filename_filter):
        conditions = []
        if only_tables is not None:
            conditions.append(
                FieldCondition(key="is_table", match=MatchValue(value=only_tables))
            )
        if filename_filter:
            conditions.append(
                FieldCondition(key="filename", match=MatchValue(value=filename_filter))
            )
        return Filter(must=conditions) if conditions else None

    @staticmethod
    def _hit_to_result(hit) -> RetrievalResult:
        p = hit.payload or {}
        return RetrievalResult(
            id=str(hit.id),
            score=hit.score,
            text=p.get("text", ""),
            filename=p.get("filename", ""),
            headings=p.get("headings", []),
            is_table=p.get("is_table", False),
            refs=p.get("refs", []),
            chunk_index=p.get("chunk_index"),
        )

    def _search_hybrid_rerank(
        self,
        query: str,
        top_k: int,
        prefetch_k: int,
        qdrant_filter,
    ) -> list[RetrievalResult]:
        """
        Hybrid search with ColBERT rerank.

        Prefetch dense (prefetch_k)
                                    -> ColBERT MaxSim rerank → top_k
        Prefetch sparse (prefetch_k)
        """
        dense_vec = list(get_dense_model().embed(["query: " + query]))[0].tolist()
        sparse_emb = list(get_sparse_model().embed(["query: " + query]))[0]
        colbert_vec = list(get_colbert_model().embed(["query: " + query]))[0].tolist()
        hits = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                # semantic recall
                Prefetch(
                    query=dense_vec,
                    using="dense",
                    limit=prefetch_k // 2,
                    filter=qdrant_filter,
                ),
                # keyword recall
                Prefetch(
                    query=SparseVector(
                        indices=sparse_emb.indices.tolist(),
                        values=sparse_emb.values.tolist(),
                    ),
                    using="sparse",
                    limit=prefetch_k,
                    filter=qdrant_filter,
                ),
            ],
            # ColBERT rerank
            query=colbert_vec,
            using="colbert",
            limit=top_k,
            with_payload=True,
        ).points

        return [self._hit_to_result(h) for h in hits]

    def _search_dense(self, query, top_k, qdrant_filter) -> list[RetrievalResult]:
        vec = list(get_dense_model().embed(["query: " + query]))[0].tolist()
        result = self.client.query_points(
            collection_name=self.collection,
            query=vec,
            using="dense",
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        return [self._hit_to_result(h) for h in result.points]

    def _search_sparse(self, query, top_k, qdrant_filter) -> list[RetrievalResult]:

        sv = list(get_sparse_model().embed([query]))[0]
        result = self.client.query_points(
            collection_name=self.collection,
            query=SparseVector(
                indices=sv.indices.tolist(),
                values=sv.values.tolist(),
            ),
            using="sparse",
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        return [self._hit_to_result(h) for h in result.points]
