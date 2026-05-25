import os
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base


def get_database_url() -> str:
    return os.environ.get("DATABASE_URL", "sqlite:///./study_coach.db")


def make_engine(url: str | None = None):
    url = url or get_database_url()
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args, echo=False)


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
