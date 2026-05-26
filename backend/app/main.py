import os
from pathlib import Path

import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from app.api.routes import router
from app.rag.document_processor import DocumentProcessor
from app.rag.embedder import OllamaEmbedder
from app.rag.hybrid_retriever import BM25Index, HybridRetriever
from app.rag.reranking_retriever import FastembedReranker, RerankingRetriever
from app.rag.retriever import Retriever


def _build_default_retriever() -> RerankingRetriever:
    chroma_path = os.environ.get("CHROMA_PATH", str(Path("./chroma_data").resolve()))
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection("study_coach_chunks")

    embed_model = os.environ.get("EMBED_MODEL", "nomic-embed-text")
    embed_host = os.environ.get("OLLAMA_HOST") or None
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


def create_app() -> FastAPI:
    app = FastAPI(title="Study Coach", version="0.0.1")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Alembic is now the single source of truth for schema; apply pending
    # migrations on every boot (idempotent). Runs even under STUDY_COACH_TEST_MODE
    # because tests monkeypatch DATABASE_URL to a tmp file before calling create_app.
    from app.db.session import migrate_to_head
    migrate_to_head()

    # App-singleton InMemorySaver — persists LangGraph state across SSE
    # requests within this process (lost on restart). P3 upgrade target:
    # SqliteSaver for cross-restart durability; see project memory.
    app.state.checkpointer = InMemorySaver()

    app.state.document_processor = DocumentProcessor()
    if os.environ.get("STUDY_COACH_TEST_MODE") != "1":
        app.state.retriever = _build_default_retriever()
    app.include_router(router)
    from app.api.auth_routes import auth_router
    app.include_router(auth_router)
    return app


app = None if os.environ.get("STUDY_COACH_TEST_MODE") == "1" else create_app()
