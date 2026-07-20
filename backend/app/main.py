import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from app.api.routes import router
from app.data_lifecycle import DataLifecycleGate
from app.rag.document_processor import DocumentProcessor
from app.rag.reranking_retriever import RerankingRetriever
from app.rag.runtime import build_default_runtime


def _build_default_retriever() -> RerankingRetriever:
    return build_default_runtime().retriever


def create_app() -> FastAPI:
    app = FastAPI(title="Study Coach", version="0.0.1")
    app.state.data_lifecycle_gate = DataLifecycleGate()

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
        runtime = build_default_runtime()
        app.state.retriever_runtime = runtime
        app.state.retriever = runtime.retriever
    app.include_router(router)
    from app.api.auth_routes import auth_router
    app.include_router(auth_router)
    return app


app = None if os.environ.get("STUDY_COACH_TEST_MODE") == "1" else create_app()
