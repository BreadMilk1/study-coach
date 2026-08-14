import os
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

def get_database_url() -> str:
    return os.environ.get("DATABASE_URL", "sqlite:///./study_coach.db")


def _is_sqlite_url(url: str) -> bool:
    return make_url(url).get_backend_name() == "sqlite"


def make_engine(url: str | None = None):
    url = url or get_database_url()
    connect_args = {"check_same_thread": False} if _is_sqlite_url(url) else {}
    engine = create_engine(url, connect_args=connect_args, echo=False)
    if engine.dialect.name == "sqlite":
        @event.listens_for(engine, "connect")
        def set_sqlite_foreign_keys(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    return engine


_engine = None
_SessionLocal = None


def get_engine():
    """Lazy-init the engine + sessionmaker.

    Schema setup is the responsibility of `migrate_to_head()` (called from app
    startup) — engine init no longer auto-creates tables. This makes Alembic the
    single source of truth for schema evolution.
    """
    global _engine, _SessionLocal
    if _engine is None:
        _engine = make_engine()
        _SessionLocal = sessionmaker(_engine, expire_on_commit=False)
    return _engine


def migrate_to_head() -> None:
    """Apply Alembic migrations up to head against the configured DATABASE_URL.

    Called from `app.main.create_app()` so every app boot reconciles schema.
    Idempotent: a DB already at head is a no-op (alembic_version table tracks state).
    """
    from alembic import command
    from alembic.config import Config

    backend_root = Path(__file__).resolve().parents[2]
    cfg = Config(str(backend_root / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", get_database_url())
    command.upgrade(cfg, "head")


@contextmanager
def session_scope() -> Session:
    get_engine()
    with _SessionLocal() as s:
        yield s


def get_session() -> Session:
    """FastAPI dependency."""
    get_engine()
    s = _SessionLocal()
    try:
        yield s
    finally:
        s.close()


def get_eval_session() -> Session:
    """FastAPI dependency for an isolated evaluation session.

    Evaluation claims must never reuse the authentication/learning session:
    the claim repository starts an explicit SQLite ``BEGIN IMMEDIATE`` on this
    clean session before it performs any read.
    """
    get_engine()
    s = _SessionLocal()
    try:
        yield s
    finally:
        s.close()
