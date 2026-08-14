"""Atomic suite import rollback tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import Session

from app.db.models import Base, EvalRun, EvalScoreSet
from app.eval.learning_run.contracts import canonical_hash, hash_without_field
from app.eval.learning_run.registry import TaskRegistry
from app.eval.learning_run.repositories import (
    ChecksumMismatchError,
    EvalSuiteImportRepository,
)


REGISTRY = TaskRegistry.load_default()


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    with Session(engine) as db_session:
        yield db_session


def _record(run_id: str = "import-run-1", *, artifact: str = "answer", hash_suffix: str = "") -> dict:
    snapshot = dict(REGISTRY.scorer_document("hybrid-v1"))
    definition_hash = snapshot["definition_hash"]
    artifact_payload = {"answer": artifact, "citations": [], "exact_evidence": [], "formatted_context": "", "usage": "unavailable", "trace": [], "budget": {}}
    manifest = {
        "experiment_id": REGISTRY.experiment.experiment_id,
        "task_case_id": "tgqa-001",
        "variant_id": "tutor-v2",
        "prompt_version": "tutor-v2",
    }
    if hash_suffix:
        manifest["tamper"] = hash_suffix
    return {
        "run": {
            "id": run_id,
            "experiment_id": REGISTRY.experiment.experiment_id,
            "task_case_id": "tgqa-001",
            "task_case_version": REGISTRY.task_cases["tgqa-001"].task_case_version,
            "variant_id": "tutor-v2",
            "run_profile": "evaluation",
            "lifecycle": "finished",
            "outcome": "success",
            "suite_execution_id": "suite-demo",
            "manifest": manifest,
            "manifest_hash": canonical_hash(manifest),
            "candidate_artifact": artifact_payload,
            "artifact_hash": canonical_hash(artifact_payload),
        },
        "score_sets": [
            {
                "id": f"score-{run_id}",
                "scorer_id": "hybrid",
                "scorer_version": "hybrid-v1",
                "scorer_snapshot": snapshot,
                "scorer_definition_hash": definition_hash,
                "artifact_input_hash": canonical_hash(artifact_payload),
                "status": "completed",
                "quality_verdict": "pass",
            }
        ],
        "executions": [],
    }


def test_valid_fixture_imports_atomically(session):
    imported = EvalSuiteImportRepository(session, registry=REGISTRY).import_records(
        [_record("import-a"), _record("import-b")]
    )
    assert imported == 2
    assert session.scalar(select(func.count()).select_from(EvalRun)) == 2
    assert session.scalar(select(func.count()).select_from(EvalScoreSet)) == 2


@pytest.mark.parametrize("kind", ["missing", "duplicate", "unknown", "hash"])
def test_invalid_import_writes_zero_rows(session, kind):
    good = _record("good-run")
    if kind == "missing":
        bad = {"score_sets": []}
    elif kind == "duplicate":
        bad = _record("good-run")
    elif kind == "unknown":
        bad = _record("bad-run")
        bad["run"]["task_case_id"] = "missing-case"
    else:
        bad = _record("bad-run")
        bad["run"]["manifest_hash"] = "0" * 64
    repo = EvalSuiteImportRepository(session, registry=REGISTRY)
    with pytest.raises((ValueError, ChecksumMismatchError)):
        repo.import_records([good, bad])
    assert session.scalar(select(func.count()).select_from(EvalRun)) == 0
    assert session.scalar(select(func.count()).select_from(EvalScoreSet)) == 0


def test_suite_execution_id_groups_without_new_table(session):
    EvalSuiteImportRepository(session, registry=REGISTRY).import_records(
        [_record("one"), _record("two")]
    )
    ids = session.scalars(select(EvalRun.suite_execution_id)).all()
    assert ids == ["suite-demo", "suite-demo"]
    assert "eval_suite_executions" not in session.bind.dialect.default_schema_name or True
    from sqlalchemy import inspect

    assert "eval_suite_executions" not in inspect(session.bind).get_table_names()
