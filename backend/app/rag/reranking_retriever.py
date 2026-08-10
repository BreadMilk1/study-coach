"""Reranking retriever: post-processes any base retriever's candidates with
a cross-encoder reranker.

Per 2026 hybrid RAG best practice (BM25 + Dense → RRF → cross-encoder rerank),
cross-encoder reranking lifts the right chunk from rank 6-10 to top-5 when the
candidate set has been over-broad due to RRF tie-breaking.
"""
from __future__ import annotations

from typing import Protocol


class _BaseRetrieverLike(Protocol):
    def search(self, query: str, top_k: int) -> list[dict]: ...
    def add_chunks(self, chunks: list[dict]) -> None: ...


class RerankerLike(Protocol):
    def score(self, query: str, documents: list[str]) -> list[float]: ...


class RerankingRetriever:
    def __init__(
        self,
        base: _BaseRetrieverLike,
        reranker: RerankerLike,
        retrieval_depth: int = 20,
    ) -> None:
        self.base = base
        self.reranker = reranker
        self.retrieval_depth = retrieval_depth

    def add_chunks(self, chunks: list[dict]) -> None:
        self.base.add_chunks(chunks)

    def get_chunk(self, chunk_id: str) -> dict | None:
        getter = getattr(self.base, "get_chunk", None)
        if callable(getter):
            return getter(chunk_id)
        return None

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        candidates = self.base.search(query, top_k=self.retrieval_depth)
        if not candidates:
            return []
        scores = self.reranker.score(
            query=query,
            documents=[c["content"] for c in candidates],
        )
        ranked = sorted(
            zip(candidates, scores), key=lambda x: x[1], reverse=True
        )[:top_k]
        return [{**c, "score": float(s)} for c, s in ranked]


class FastembedReranker:
    """Thin adapter wrapping fastembed.TextCrossEncoder.

    Default model `jinaai/jina-reranker-v2-base-multilingual` is multilingual
    (handles Chinese queries against English documents), 1.11 GB, in fastembed's
    default supported list (no custom_add required).
    """

    def __init__(
        self, model_name: str = "jinaai/jina-reranker-v2-base-multilingual"
    ) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=model_name)

    def score(self, query: str, documents: list[str]) -> list[float]:
        return list(self._model.rerank(query, documents))
