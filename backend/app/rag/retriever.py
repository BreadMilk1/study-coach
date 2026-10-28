from .embedder import Embedder
from .scope import RetrievalScope

_METADATA_KEYS = ("owner", "document_id", "hash")


def _metadata_for(chunk: dict) -> dict:
    metadata = {"source": chunk["source"], "page": chunk["page"]}
    for key in _METADATA_KEYS:
        if key in chunk:
            metadata[key] = chunk[key]
    return metadata


def _with_optional_metadata(meta: dict, chunk: dict) -> dict:
    for key in _METADATA_KEYS:
        if key in meta:
            chunk[key] = meta[key]
    return chunk


class Retriever:
    def __init__(self, collection, embedder: Embedder):
        self.collection = collection
        self.embedder = embedder
        self._closed = False

    def add_chunks(self, chunks: list[dict]) -> None:
        if not chunks:
            return
        embeddings = self.embedder.embed([c["content"] for c in chunks])
        # Caller-supplied source is authoritative so a renamed partial retry after
        # index-before-SQL failure can realign Chroma with the new SQL filename.
        # owner/document_id/hash ride along so scoped retrieval and authorized
        # readback keep working after a restart rehydrates from the collection.
        self.collection.upsert(
            ids=[c["chunk_id"] for c in chunks],
            documents=[c["content"] for c in chunks],
            metadatas=[_metadata_for(c) for c in chunks],
            embeddings=embeddings,
        )

    def get_chunk(self, chunk_id: str) -> dict | None:
        result = self.collection.get(
            ids=[chunk_id],
            include=["documents", "metadatas"],
        )
        if not result["ids"]:
            return None
        meta = result["metadatas"][0] or {}
        return _with_optional_metadata(meta, {
            "chunk_id": result["ids"][0],
            "content": result["documents"][0] or "",
            "source": meta.get("source", ""),
            "page": meta.get("page", -1),
        })

    def search(self, query: str, top_k: int = 5, scope: RetrievalScope | None = None) -> list[dict]:
        if scope is not None and not scope.documents:
            # No SQL-owned document: fail closed without touching the index.
            return []
        query_emb = self.embedder.embed([query])[0]
        kwargs: dict = {"query_embeddings": [query_emb], "n_results": top_k}
        if scope is not None:
            # Positive filter on owner AND the caller's SQL-authorized document
            # ids, so a same-owner chunk that is unbound or carries a different
            # document id (an index-before-SQL orphan) can never crowd the
            # legitimate top-k. A chunk missing either key is not returned.
            kwargs["where"] = {
                "$and": [
                    {"owner": {"$eq": scope.owner}},
                    {"document_id": {"$in": sorted(set(scope.documents.values()))}},
                ]
            }
        result = self.collection.query(**kwargs)
        ids = result["ids"][0]
        chunks = [
            _with_optional_metadata(result["metadatas"][0][i] or {}, {
                "chunk_id": ids[i],
                "content": result["documents"][0][i],
                "source": (result["metadatas"][0][i] or {}).get("source", ""),
                "page": (result["metadatas"][0][i] or {}).get("page", -1),
                "score": 1.0 - float(result["distances"][0][i]),
            })
            for i in range(len(ids))
        ]
        if scope is not None:
            # Post-validate the full id/metadata/document consistency as well.
            chunks = [chunk for chunk in chunks if scope.allows(chunk)]
        return chunks

    def search_scoped(
        self, query: str, top_k: int, scope: RetrievalScope
    ) -> list[dict]:
        return self.search(query, top_k=top_k, scope=scope)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self.embedder, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
