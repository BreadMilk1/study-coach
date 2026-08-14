"""Atomic suite import rollback tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event, func, inspect, select
from sqlalchemy.orm import Session

from app.db.models import Base, EvalRun, EvalScoreSet
from app.eval.learning_run.contracts import canonical_hash
from app.eval.learning_run.registry import TaskRegistry
from app.eval.learning_run.repositories import (
    ChecksumMismatchError,
    EvalSuiteImportRepository,
)


def _load_import_script():
    path = Path(__file__).resolve().parents[2] / "scripts" / "import_learning_run_suite.py"
    spec = importlib.util.spec_from_file_location("import_learning_run_suite", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def _record(
    run_id: str,
    *,
    case_id: str = "tgqa-001",
    variant_id: str = "tutor-v2",
    scorer_version: str = "hybrid-v1",
    artifact: str = "answer",
) -> dict:
    snapshot = dict(REGISTRY.scorer_document(scorer_version))
    definition_hash = snapshot["definition_hash"]
    artifact_payload = {
        "answer": artifact,
        "citations": [],
        "exact_evidence": [],
        "formatted_context": "",
        "usage": "unavailable",
        "trace": [],
        "budget": {},
    }
    artifact_hash = canonical_hash(artifact_payload)
    case = REGISTRY.task_cases[case_id]
    manifest = {
        "experiment_id": REGISTRY.experiment.experiment_id,
        "task_case_id": case_id,
        "variant_id": variant_id,
        "prompt_version": variant_id,
    }
    return {
        "run": {
            "id": run_id,
            "experiment_id": REGISTRY.experiment.experiment_id,
            "task_case_id": case_id,
            "task_case_version": case.task_case_version,
            "variant_id": variant_id,
            "run_profile": "evaluation",
            "lifecycle": "finished",
            "outcome": "success",
            "suite_execution_id": "suite-demo",
            "manifest": manifest,
            "manifest_hash": canonical_hash(manifest),
            "candidate_artifact": artifact_payload,
            "artifact_hash": artifact_hash,
        },
        "score_sets": [
            {
                "id": f"score-{run_id}",
                "scorer_id": "hybrid",
                "scorer_version": scorer_version,
                "scorer_snapshot": snapshot,
                "scorer_definition_hash": definition_hash,
                "artifact_input_hash": artifact_hash,
                "status": "completed",
                "quality_verdict": "pass",
            }
        ],
        "executions": [],
    }


def _complete_suite() -> list[dict]:
    return [
        _record(f"{case_id}-{variant_id}", case_id=case_id, variant_id=variant_id)
        for case_id in REGISTRY.task_cases
        for variant_id in REGISTRY.experiment.variants
    ]


def test_valid_complete_suite_imports_atomically(session):
    records = _complete_suite()
    imported = EvalSuiteImportRepository(session, registry=REGISTRY).import_records(records)
    expected = len(REGISTRY.task_cases) * len(REGISTRY.experiment.variants)
    assert imported == expected
    assert session.scalar(select(func.count()).select_from(EvalRun)) == expected
    assert session.scalar(select(func.count()).select_from(EvalScoreSet)) == expected


@pytest.mark.parametrize("kind", ["missing", "duplicate", "unknown_case", "unknown_scorer", "hash", "incomplete", "artifact_align", "case_version"])
def test_invalid_import_writes_zero_rows(session, kind):
    records = _complete_suite()
    if kind == "missing":
        records[0] = {"score_sets": []}
    elif kind == "duplicate":
        records.append(_record("dup", case_id="tgqa-001", variant_id="tutor-v2"))
    elif kind == "unknown_case":
        records[0]["run"]["task_case_id"] = "missing-case"
    elif kind == "unknown_scorer":
        records[0]["score_sets"][0]["scorer_version"] = "hybrid-v9"
    elif kind == "hash":
        records[0]["run"]["manifest_hash"] = "0" * 64
    elif kind == "incomplete":
        records = records[:-1]
    elif kind == "artifact_align":
        records[0]["score_sets"][0]["artifact_input_hash"] = "1" * 64
    else:
        records[0]["run"]["task_case_version"] = "not-the-registry-version"
    repo = EvalSuiteImportRepository(session, registry=REGISTRY)
    with pytest.raises((ValueError, ChecksumMismatchError)):
        repo.import_records(records)
    assert session.scalar(select(func.count()).select_from(EvalRun)) == 0
    assert session.scalar(select(func.count()).select_from(EvalScoreSet)) == 0


def test_suite_execution_id_groups_without_new_table(session):
    EvalSuiteImportRepository(session, registry=REGISTRY).import_records(_complete_suite())
    ids = set(session.scalars(select(EvalRun.suite_execution_id)).all())
    assert ids == {"suite-demo"}
    assert "eval_suite_executions" not in inspect(session.bind).get_table_names()


def test_parse_jsonl_rejects_malformed_lines(tmp_path: Path):
    path = tmp_path / "suite.jsonl"
    path.write_text("{}\nnot-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSONL"):
        _load_import_script().parse_jsonl(path)


def test_parse_jsonl_skips_blank_lines(tmp_path: Path):
    path = tmp_path / "suite.jsonl"
    path.write_text('\n{"run": {"id": "a"}}\n\n', encoding="utf-8")
    assert _load_import_script().parse_jsonl(path) == [{"run": {"id": "a"}}]
