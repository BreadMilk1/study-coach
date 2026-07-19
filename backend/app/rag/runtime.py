import os
from pathlib import Path
from typing import Callable

import chromadb
from chromadb.errors import NotFoundError

from app.rag.embedder import OllamaEmbedder
from app.rag.hybrid_retriever import BM25Index, HybridRetriever
from app.rag.reranking_retriever import FastembedReranker, RerankingRetriever
from app.rag.retriever import Retriever


class RetrieverRuntime:
    def __init__(
        self,
        *,
        client,
        collection_name: str,
        builder: Callable[[object], RerankingRetriever],
    ):
        self.client = client
        self.collection_name = collection_name
        self.builder = builder
        self.collection = client.get_or_create_collection(collection_name)
        self.retriever = builder(self.collection)

    def vector_count(self) -> int:
        return self.collection.count()

    def reset_empty(self) -> RerankingRetriever:
        try:
            self.client.delete_collection(self.collection_name)
        except NotFoundError:
            pass
        collection = self.client.get_or_create_collection(self.collection_name)
        retriever = self.builder(collection)
        self.collection = collection
        self.retriever = retriever
        return retriever


def _build_retriever(
    collection,
    *,
    embed_model: str,
    embed_host: str | None,
) -> RerankingRetriever:
    embedder = OllamaEmbedder(model=embed_model, base_url=embed_host)
    dense = Retriever(collection=collection, embedder=embedder)
    bm25 = BM25Index()
    if collection.count() > 0:
        data = collection.get(include=["documents", "metadatas"])
        chunks = [
            {
                "chunk_id": data["ids"][i],
                "content": data["documents"][i],
                "source": data["metadatas"][i].get("source", ""),
                "page": data["metadatas"][i].get("page", -1),
            }
            for i in range(len(data["ids"]))
        ]
        bm25.add_chunks(chunks)

    hybrid = HybridRetriever(dense=dense, bm25=bm25)
    reranker = FastembedReranker()
    return RerankingRetriever(base=hybrid, reranker=reranker, retrieval_depth=20)


def build_default_runtime() -> RetrieverRuntime:
    chroma_path = os.environ.get("CHROMA_PATH", str(Path("./chroma_data").resolve()))
    client = chromadb.PersistentClient(path=chroma_path)
    embed_model = os.environ.get("EMBED_MODEL", "nomic-embed-text")
    embed_host = os.environ.get("OLLAMA_HOST") or None

    def builder(collection) -> RerankingRetriever:
        return _build_retriever(
            collection,
            embed_model=embed_model,
            embed_host=embed_host,
        )

    return RetrieverRuntime(
        client=client,
        collection_name="study_coach_chunks",
        builder=builder,
    )
