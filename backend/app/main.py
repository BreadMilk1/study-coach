import os
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from app.api.routes import router
from app.data_lifecycle import DataLifecycleGate
from app.rag.document_processor import DocumentProcessor
from app.rag.reranking_retriever import RerankingRetriever
from app.rag.runtime import build_default_runtime
from app.eval.learning_run.corpus import CorpusMaterializerController, CorpusSnapshotLoader
from app.eval.learning_run.registry import TaskRegistry
from app.eval.learning_run.runner import TutorRunner
from app.agent.tutor_attempt import TutorAttemptEngine


def _build_default_retriever() -> RerankingRetriever:
    return build_default_runtime().retriever


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        try:
            yield
        finally:
            controller = getattr(application.state, "eval_materializer_controller", None)
            if controller is not None:
                controller.shutdown(wait=False)

    app = FastAPI(title="Study Coach", version="0.0.1", lifespan=lifespan)
    app.state.data_lifecycle_gate = DataLifecycleGate()

    # Innermost → outermost (last add_middleware is outermost):
    # 1) Lifecycle lease: before FastAPI multipart parse; held through SSE.
    # 2) Upload body limit: before multipart spool; Content-Length / stream 413
    #    without taking a lease when the declared size already exceeds the cap.
    # 3) CORS: wraps middleware-generated 409/413 JSON with Access-Control-*.
    from app.api.lifecycle_middleware import DataLifecycleLeaseMiddleware
    from app.api.upload_body_limit_middleware import UploadBodyLimitMiddleware

    app.add_middleware(DataLifecycleLeaseMiddleware)
    app.add_middleware(UploadBodyLimitMiddleware)
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
    started_before = datetime.utcnow()
    migrate_to_head()
    # Restart repair runs before routers become serviceable.  The cutoff is
    # captured explicitly so rows created after boot cannot be mistaken for
    # leftovers from the interrupted process.
    from app.db.session import session_scope
    from app.eval.learning_run.repositories import EvalExecutionControlRepository
    from sqlalchemy import inspect

    with session_scope() as eval_session:
        # A few legacy app-factory tests intentionally stub migrations; in
        # that isolated case the eval tables do not exist yet.  A real boot
        # always reaches this branch after Alembic has created both tables.
        tables = set(inspect(eval_session.bind).get_table_names())
        if {"eval_runs", "eval_score_sets"}.issubset(tables):
            EvalExecutionControlRepository(eval_session).reconcile(
                started_before=started_before
            )

    # App-singleton InMemorySaver — persists LangGraph state across SSE
    # requests within this process (lost on restart). P3 upgrade target:
    # SqliteSaver for cross-restart durability; see project memory.
    app.state.checkpointer = InMemorySaver()

    # Evaluation definitions and its isolated materializer are app-scoped,
    # while model/DB services remain request-scoped.  Constructing these
    # objects does not create Chroma collections or model clients.
    app.state.eval_registry = TaskRegistry.load_default()
    app.state.eval_corpus_loader = CorpusSnapshotLoader()
    app.state.eval_materializer_controller = CorpusMaterializerController(
        app.state.eval_corpus_loader
    )
    app.state.eval_tutor_runner = TutorRunner(
        corpus_loader=app.state.eval_corpus_loader,
        attempt_engine=TutorAttemptEngine(),
        materializer_controller=app.state.eval_materializer_controller,
    )
    # Tests and local adapters may replace these request-scoped factories
    # without constructing provider clients or touching Chroma.
    app.state.eval_service_factory = None
    app.state.eval_connection_factory = None

    app.state.document_processor = DocumentProcessor()
    if os.environ.get("STUDY_COACH_TEST_MODE") != "1":
        runtime = build_default_runtime()
        app.state.retriever_runtime = runtime
        app.state.retriever = runtime.retriever
    app.include_router(router)
    from app.api.auth_routes import auth_router
    app.include_router(auth_router)
    from app.api.data_routes import data_router
    app.include_router(data_router)
    from app.api.eval_routes import eval_router
    app.include_router(eval_router)
    return app


app = None if os.environ.get("STUDY_COACH_TEST_MODE") == "1" else create_app()
