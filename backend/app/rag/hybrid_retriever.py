"""Hybrid retriever combining BM25 (sparse/lexical) and Dense (Chroma) via
Reciprocal Rank Fusion (RRF).

Per 2026 hybrid RAG best practice, BM25 catches keyword / proper-noun queries
that dense embeddings miss (e.g., "ReAct", "Cycle of Quality"); dense catches
semantic / paraphrase queries. RRF with k_smoothing=60 follows Cormack et al.
2009.
"""
from __future__ import annotations

import re
from typing import Protocol

from rank_bm25 import BM25Okapi


class _DenseLike(Protocol):
    def search(self, query: str, top_k: int) -> list[dict]: ...
    def add_chunks(self, chunks: list[dict]) -> None: ...


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k_smoothing: int = 60,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k_smoothing + rank)
    return scores


class BM25Index:
    def __init__(self) -> None:
        self._chunks: list[dict] = []
        self._tokenized: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def add_chunks(self, chunks: list[dict]) -> None:
        new_tokens = [_tokenize(c["content"]) for c in chunks]
        self._chunks.extend(chunks)
        self._tokenized.extend(new_tokens)
        if self._tokenized:
            self._bm25 = BM25Okapi(self._tokenized)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        if self._bm25 is None:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            ((i, float(s)) for i, s in enumerate(scores) if s > 0),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]
        return [{**self._chunks[i], "score": s} for i, s in ranked]


class HybridRetriever:
    def __init__(self, dense: _DenseLike, bm25: BM25Index | None = None) -> None:
        self.dense = dense
        self.bm25 = bm25 or BM25Index()

    def add_chunks(self, chunks: list[dict]) -> None:
        self.dense.add_chunks(chunks)
        self.bm25.add_chunks(chunks)

    def search(
        self, query: str, top_k: int = 5, retrieval_depth: int = 20
    ) -> list[dict]:
        dense_results = self.dense.search(query, top_k=retrieval_depth)
        bm25_results = self.bm25.search(query, top_k=retrieval_depth)

        chunk_by_id: dict[str, dict] = {}
        for c in dense_results + bm25_results:
            chunk_by_id[c["chunk_id"]] = c

        fused = reciprocal_rank_fusion([
            [c["chunk_id"] for c in dense_results],
            [c["chunk_id"] for c in bm25_results],
        ])
        ranked_ids = sorted(fused.keys(), key=lambda i: fused[i], reverse=True)[:top_k]
        return [{**chunk_by_id[cid], "score": fused[cid]} for cid in ranked_ids]
