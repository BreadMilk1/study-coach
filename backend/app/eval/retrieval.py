"""Retrieval evaluation harness.

Pure functions to measure Hit Rate@K and MRR over a ground-truth query set,
plus an aggregator that runs a retriever across all queries.

Ground-truth shape per query: list of {"source": <pdf_filename>, "pages": [int, ...]}.
A returned chunk matches when its source+page falls inside any expected entry.
"""
from dataclasses import dataclass, field
from typing import Protocol


class RetrieverLike(Protocol):
    def search(self, query: str, top_k: int) -> list[dict]: ...


@dataclass
class EvalQuery:
    query: str
    expected: list[dict] = field(default_factory=list)
    classification: str = ""


@dataclass
class EvalReport:
    per_query: list[dict] = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)


def _is_match(chunk: dict, expected: list[dict]) -> bool:
    for e in expected:
        if chunk.get("source") == e["source"] and chunk.get("page") in e["pages"]:
            return True
    return False


def _hit_rank(returned: list[dict], expected: list[dict]) -> int | None:
    for i, c in enumerate(returned, start=1):
        if _is_match(c, expected):
            return i
    return None


def hit_rate_at_k(returned: list[dict], expected: list[dict], k: int) -> float:
    return 1.0 if any(_is_match(c, expected) for c in returned[:k]) else 0.0


def mrr(returned: list[dict], expected: list[dict]) -> float:
    rank = _hit_rank(returned, expected)
    return 1.0 / rank if rank is not None else 0.0


def evaluate(
    retriever: RetrieverLike,
    queries: list[EvalQuery],
    top_k: int = 10,
) -> EvalReport:
    per_query: list[dict] = []
    hit_5_sum = 0.0
    mrr_sum = 0.0
    for q in queries:
        returned = retriever.search(q.query, top_k=top_k)
        h5 = hit_rate_at_k(returned, q.expected, k=5)
        m = mrr(returned, q.expected)
        rank = _hit_rank(returned, q.expected)
        per_query.append({
            "query": q.query,
            "classification": q.classification,
            "returned_count": len(returned),
            "hit_rank": rank,
            "hit_rate@5": h5,
            "mrr": m,
        })
        hit_5_sum += h5
        mrr_sum += m
    n = len(queries) or 1
    return EvalReport(
        per_query=per_query,
        aggregate={
            "hit_rate@5": hit_5_sum / n,
            "mrr": mrr_sum / n,
        },
    )
