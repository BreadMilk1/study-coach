"""Cut ① — Alembic baseline migration test.

Verifies that `alembic upgrade head` against an empty SQLite DB produces the
P1 baseline schema (users + documents tables + alembic_version bookkeeping).
"""
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect


BACKEND_ROOT = Path(__file__).resolve().parents[2]


def _alembic_config(db_url: str) -> Config:
    cfg = Config(str(BACKEND_ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", db_url)
    return cfg


def test_alembic_upgrade_head_creates_baseline_tables(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'baseline.db'}"
    cfg = _alembic_config(db_url)

    command.upgrade(cfg, "head")

    engine = create_engine(db_url)
    tables = set(inspect(engine).get_table_names())
    assert {"users", "documents", "alembic_version"} <= tables


def test_alembic_baseline_users_columns_match_orm(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'columns.db'}"
    cfg = _alembic_config(db_url)
    command.upgrade(cfg, "head")

    engine = create_engine(db_url)
    cols = {c["name"] for c in inspect(engine).get_columns("users")}
    assert cols == {"id", "fingerprint", "created_at"}


def test_alembic_baseline_documents_columns_match_orm(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'docs.db'}"
    cfg = _alembic_config(db_url)
    command.upgrade(cfg, "head")

    engine = create_engine(db_url)
    cols = {c["name"] for c in inspect(engine).get_columns("documents")}
    assert cols == {"id", "user_id", "filename", "hash", "chunks_count", "created_at"}


def test_alembic_upgrade_head_creates_p2_1_3_memory_tables(tmp_path):
    """Cut ②: after p2_1_3_memory_schema migration runs, all 9 new tables exist."""
    db_url = f"sqlite:///{tmp_path / 'p2.db'}"
    cfg = _alembic_config(db_url)
    command.upgrade(cfg, "head")

    engine = create_engine(db_url)
    tables = set(inspect(engine).get_table_names())
    expected = {
        "goals",
        "topics",
        "plans",
        "questions",
        "mastery",
        "mistakes",
        "sessions",
        "messages",
        "citations",
    }
    missing = expected - tables
    assert not missing, f"missing tables: {missing}"


def test_migrate_to_head_reads_database_url_env(tmp_path, monkeypatch):
    """Follow-up #2: session.migrate_to_head() honors DATABASE_URL env override."""
    db_path = tmp_path / "fresh.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    from app.db.session import migrate_to_head

    migrate_to_head()

    engine = create_engine(f"sqlite:///{db_path}")
    tables = set(inspect(engine).get_table_names())
    assert "alembic_version" in tables
    assert "users" in tables       # baseline
    assert "goals" in tables       # p2_1_3 head


def test_migrate_to_head_is_idempotent(tmp_path, monkeypatch):
    """Calling twice in a row must not error (app reload / multi-worker safety)."""
    db_path = tmp_path / "idem.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    from app.db.session import migrate_to_head

    migrate_to_head()
    migrate_to_head()  # should be a no-op, not raise

    engine = create_engine(f"sqlite:///{db_path}")
    assert "goals" in set(inspect(engine).get_table_names())
