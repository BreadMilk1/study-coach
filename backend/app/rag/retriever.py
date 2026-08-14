from .embedder import Embedder


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
        self.collection.upsert(
            ids=[c["chunk_id"] for c in chunks],
            documents=[c["content"] for c in chunks],
            metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks],
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
        return {
            "chunk_id": result["ids"][0],
            "content": result["documents"][0] or "",
            "source": meta.get("source", ""),
            "page": meta.get("page", -1),
        }

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_emb = self.embedder.embed([query])[0]
        result = self.collection.query(query_embeddings=[query_emb], n_results=top_k)
        ids = result["ids"][0]
        return [
            {
                "chunk_id": ids[i],
                "content": result["documents"][0][i],
                "source": result["metadatas"][0][i].get("source", ""),
                "page": result["metadatas"][0][i].get("page", -1),
                "score": 1.0 - float(result["distances"][0][i]),
            }
            for i in range(len(ids))
        ]

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
