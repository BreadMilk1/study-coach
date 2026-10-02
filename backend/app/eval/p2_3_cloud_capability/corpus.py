from __future__ import annotations

from typing import Any, Mapping

from app.eval.learning_run.contracts import canonical_hash


def snapshot_hash(chunks: list[Mapping[str, Any]]) -> str:
    payload = [
        {
            "chunk_id": chunk["chunk_id"],
            "content": chunk["content"],
            "source": chunk["source"],
            "page": chunk["page"],
        }
        for chunk in sorted(chunks, key=lambda item: str(item["chunk_id"]))
    ]
    return canonical_hash(payload)


def retriever_config_hash(cfg: Mapping[str, Any]) -> str:
    return canonical_hash(dict(cfg))


def chunks_from_collection(collection) -> list[dict[str, Any]]:
    data = collection.get(include=["documents", "metadatas"])
    chunks: list[dict[str, Any]] = []
    for i, chunk_id in enumerate(data["ids"]):
        meta = data["metadatas"][i] or {}
        chunks.append(
            {
                "chunk_id": chunk_id,
                "content": data["documents"][i] or "",
                "source": meta.get("source", ""),
                "page": meta.get("page", -1),
            }
        )
    return chunks


class RecordingRetriever:
    def __init__(self, inner):
        self.inner = inner
        self.captures: list[dict[str, Any]] = []

    def search(self, query: str, top_k: int = 5):
        try:
            evidence = self.inner.search(query, top_k=top_k) or []
        except Exception:
            self.captures.append(
                {
                    "query": query,
                    "top_k": top_k,
                    "evidence": None,
                    "capture_status": "missing",
                }
            )
            raise
        self.captures.append(
            {
                "query": query,
                "top_k": top_k,
                "evidence": evidence,
                "capture_status": "ok",
            }
        )
        return evidence
