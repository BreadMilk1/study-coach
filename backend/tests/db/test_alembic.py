"""Cut ① — Alembic baseline migration test.

Verifies that `alembic upgrade head` against an empty SQLite DB produces the
P1 baseline schema (users + documents tables + alembic_version bookkeeping).
"""
import json
import logging
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.exc import IntegrityError


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
    assert cols == {"id", "fingerprint", "google_id", "email", "created_at"}


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


def test_alembic_upgrade_head_creates_plan_milestone_progression_tables(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'p4_plan.db'}"
    cfg = _alembic_config(db_url)
    command.upgrade(cfg, "head")

    engine = create_engine(db_url)
    tables = set(inspect(engine).get_table_names())
    assert {"plan_milestones", "plan_events"} <= tables

    milestone_cols = {c["name"] for c in inspect(engine).get_columns("plan_milestones")}
    assert {
        "id",
        "plan_id",
        "topic_id",
        "topic_name",
        "title",
        "due_at",
        "done",
        "completed_at",
        "sort_order",
        "source",
        "created_at",
        "updated_at",
    } <= milestone_cols

    event_cols = {c["name"] for c in inspect(engine).get_columns("plan_events")}
    assert {
        "id",
        "plan_id",
        "milestone_id",
        "actor",
        "action",
        "before_json",
        "after_json",
        "reason",
        "created_at",
    } <= event_cols


def test_alembic_upgrade_head_creates_only_bounded_learning_run_eval_tables(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'learning_run_eval.db'}"
    cfg = _alembic_config(db_url)

    command.upgrade(cfg, "head")

    engine = create_engine(db_url)
    tables = set(inspect(engine).get_table_names())
    assert {"eval_runs", "eval_score_sets", "eval_scorer_executions"} <= tables
    assert "eval_suite_executions" not in tables


def test_learning_run_eval_migration_downgrade_and_reupgrade_preserves_single_head(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'learning_run_round_trip.db'}"
    cfg = _alembic_config(db_url)

    command.upgrade(cfg, "head")
    engine = create_engine(db_url)
    assert set(inspect(engine).get_table_names()) >= {
        "eval_runs",
        "eval_score_sets",
        "eval_scorer_executions",
    }

    command.downgrade(cfg, "7a52fe598fd1")
    downgraded_tables = set(inspect(engine).get_table_names())
    assert not {
        "eval_runs",
        "eval_score_sets",
        "eval_scorer_executions",
    } & downgraded_tables

    command.upgrade(cfg, "head")
    upgraded_tables = set(inspect(engine).get_table_names())
    assert {"eval_runs", "eval_score_sets", "eval_scorer_executions"} <= upgraded_tables


def test_learning_run_eval_schema_has_foreign_keys_checks_and_unique_scorer_identity(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'learning_run_schema.db'}"
    cfg = _alembic_config(db_url)
    command.upgrade(cfg, "head")

    engine = create_engine(db_url)
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    assert {"eval_runs", "eval_score_sets", "eval_scorer_executions"} <= tables
    run_columns = {column["name"] for column in inspector.get_columns("eval_runs")}
    score_columns = {column["name"] for column in inspector.get_columns("eval_score_sets")}
    execution_columns = {
        column["name"] for column in inspector.get_columns("eval_scorer_executions")
    }
    assert "user_id" not in run_columns | score_columns | execution_columns
    assert {
        "id",
        "run_id",
        "scorer_id",
        "scorer_version",
        "scorer_snapshot_json",
        "scorer_definition_hash",
        "artifact_input_hash",
    } <= score_columns
    assert {"id", "score_set_id", "scorer_id", "scorer_version", "input_hash"} <= execution_columns

    score_fks = inspector.get_foreign_keys("eval_score_sets")
    execution_fks = inspector.get_foreign_keys("eval_scorer_executions")
    assert {fk["referred_table"] for fk in score_fks} == {"eval_runs"}
    assert {fk["referred_table"] for fk in execution_fks} == {"eval_score_sets"}

    unique_constraints = inspector.get_unique_constraints("eval_scorer_executions")
    assert any(
        set(constraint["column_names"])
        == {"score_set_id", "scorer_id", "scorer_version"}
        for constraint in unique_constraints
    )
    assert inspector.get_check_constraints("eval_runs")
    assert inspector.get_check_constraints("eval_score_sets")
    assert inspector.get_check_constraints("eval_scorer_executions")


def test_migrated_eval_foreign_keys_require_child_first_deletion(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'learning_run_fk.db'}"
    cfg = _alembic_config(db_url)
    command.upgrade(cfg, "head")

    engine = create_engine(db_url)

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    with engine.begin() as conn:
        assert conn.execute(text("PRAGMA foreign_keys")).scalar_one() == 1
        conn.execute(
            text(
                "INSERT INTO eval_runs "
                "(id, experiment_id, task_case_id, task_case_version, variant_id, "
                "run_profile, lifecycle, manifest_json, manifest_hash) "
                "VALUES ('fk-run', 'experiment', 'case', '1', 'tutor-v2', "
                "'evaluation', 'queued', '{\"manifest\": true}', 'manifest-hash')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO eval_score_sets "
                "(id, run_id, scorer_id, scorer_version, scorer_snapshot_json, "
                "scorer_definition_hash, artifact_input_hash, "
                "status, quality_verdict) VALUES "
                "('fk-score-set', 'fk-run', 'hybrid', 'v1', '{\"scorer_id\": \"hybrid\"}', "
                "'hash', 'artifact-hash', "
                "'pending', 'not_evaluated')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO eval_scorer_executions "
                "(id, score_set_id, scorer_id, scorer_version, status, input_hash) "
                "VALUES ('fk-execution', 'fk-score-set', 'citations', 'v1', "
                "'skipped', 'artifact-hash')"
            )
        )

    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM eval_runs WHERE id = 'fk-run'"))

    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM eval_scorer_executions WHERE id = 'fk-execution'")
        )
        conn.execute(text("DELETE FROM eval_score_sets WHERE id = 'fk-score-set'"))
        conn.execute(text("DELETE FROM eval_runs WHERE id = 'fk-run'"))
        assert conn.execute(text("PRAGMA foreign_key_check")).all() == []


def test_alembic_plan_milestone_backfill_skips_malformed_rows_and_preserves_plan_json(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'p4_backfill.db'}"
    cfg = _alembic_config(db_url)
    command.upgrade(cfg, "cae9687d6295")

    original_milestones = [
        {
            "id": "milestone-undone",
            "topic": "Algebra",
            "title": "Review linear equations",
            "due_at": "2026-06-01T09:00:00",
            "done": "false",
        },
        {
            "id": "milestone-done",
            "topic_name": "Geometry",
            "title": "Finish triangles",
            "done": True,
        },
        "bad-row",
    ]
    original_json = json.dumps(original_milestones)

    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "INSERT INTO users (id, fingerprint, created_at) VALUES (?, ?, ?)",
            ("user-1", "fingerprint-1", "2026-05-26 00:00:00"),
        )
        conn.exec_driver_sql(
            "INSERT INTO goals (id, user_id, title, exam_date, status) VALUES (?, ?, ?, ?, ?)",
            ("goal-1", "user-1", "Study goal", None, "active"),
        )
        conn.exec_driver_sql(
            "INSERT INTO plans (id, goal_id, milestones_json, updated_at) VALUES (?, ?, ?, ?)",
            ("plan-1", "goal-1", original_json, "2026-05-26 01:00:00"),
        )

    command.upgrade(cfg, "head")

    with engine.begin() as conn:
        rows = conn.exec_driver_sql(
            """
            SELECT id, topic_name, title, done, completed_at, sort_order, source
            FROM plan_milestones
            ORDER BY sort_order
            """
        ).mappings().all()
        preserved_json = conn.exec_driver_sql(
            "SELECT milestones_json FROM plans WHERE id = ?",
            ("plan-1",),
        ).scalar_one()
        conn.exec_driver_sql(
            """
            INSERT INTO plan_milestones (
                id, plan_id, title, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            ("manual-milestone", "plan-1", "Manual milestone", "2026-05-26 02:00:00", "2026-05-26 02:00:00"),
        )
        default_source = conn.exec_driver_sql(
            "SELECT source FROM plan_milestones WHERE id = ?",
            ("manual-milestone",),
        ).scalar_one()

    assert len(rows) == 2
    assert [row["id"] for row in rows] == ["milestone-undone", "milestone-done"]
    assert [row["sort_order"] for row in rows] == [0, 1]
    assert rows[0]["topic_name"] == "Algebra"
    assert rows[0]["done"] in (False, 0)
    assert rows[0]["completed_at"] is None
    assert rows[0]["source"] == "migrated"
    assert rows[1]["topic_name"] == "Geometry"
    assert rows[1]["done"] in (True, 1)
    assert rows[1]["completed_at"] is not None
    assert rows[1]["source"] == "migrated"
    assert json.loads(preserved_json) == original_milestones
    assert default_source == "ai"


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


def test_migrate_to_head_keeps_existing_application_loggers_enabled(tmp_path, monkeypatch):
    db_path = tmp_path / "logging.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    application_logger = logging.getLogger("uvicorn.error")
    application_logger.disabled = False

    from app.db.session import migrate_to_head

    migrate_to_head()

    assert application_logger.disabled is False
