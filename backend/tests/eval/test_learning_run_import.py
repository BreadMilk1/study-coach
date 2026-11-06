"""Atomic suite import rollback tests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event, func, inspect, select
from sqlalchemy.orm import Session

from app.db.models import Base, EvalRun, EvalScoreSet
from app.eval.learning_run.contracts import canonical_hash
from app.eval.learning_run.registry import TaskRegistry
from app.eval.learning_run.repositories import (
    ChecksumMismatchError,
    EvalRunRepository,
    EvalScoreSetRepository,
    EvalScorerExecutionRepository,
    EvalSuiteImportRepository,
)


CURATED_FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "app"
    / "eval"
    / "learning_run"
    / "fixtures"
    / "tutor-prompt-regression-v1.jsonl"
)


def _stdlib_manifest_hash(payload: dict) -> str:
    """Canonical hash computed independently, with the standard library only."""
    import hashlib

    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


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
        "task_case_version": case.task_case_version,
        "variant_id": variant_id,
        "run_profile": "evaluation",
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


def test_committed_24_record_fixture_imports_with_original_hashes(session):
    """The committed suite still imports and keeps its original manifest hashes.

    The records predate the optional ``usage_contract`` field, so none of them
    carries it and every stored manifest hash must equal the committed one.
    """
    records = [
        json.loads(line)
        for line in CURATED_FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 24

    imported = EvalSuiteImportRepository(session, registry=REGISTRY).import_records(records)

    assert imported == 24
    assert session.scalar(select(func.count()).select_from(EvalRun)) == 24
    run_repo = EvalRunRepository(session)
    score_repo = EvalScoreSetRepository(session)
    for record in records:
        stored = run_repo.get_verified(record["run"]["id"])
        assert stored.manifest_hash == record["run"]["manifest_hash"]
        assert "usage_contract" not in stored.manifest_json
        for raw_score in record["score_sets"]:
            # Historical score-set fields are re-read exactly as committed: the
            # nine-key projection and its definition hash must not be re-signed.
            stored_score = score_repo.get_verified(raw_score["id"])
            assert stored_score.scorer_snapshot_json == raw_score["scorer_snapshot"]
            assert (
                stored_score.scorer_definition_hash
                == raw_score["scorer_definition_hash"]
            )


# The committed hybrid-v1 nine-key projection hash, computed independently with
# the standard library. It is deliberately different from the document hash.
HYBRID_V1_PROJECTION_HASH = (
    "80ab0249b27ff284601d5eb2da43d6442d935015ce0d9d0c062765a74fa46a60"
)


def test_import_accepts_both_self_consistent_scorer_snapshot_shapes(session):
    """Historical reads accept the nine-key projection and the full document."""
    records = _complete_suite()
    document = dict(REGISTRY.scorer_document("hybrid-v1"))
    projection = {
        key: value
        for key, value in document.items()
        if key not in ("schema_version", "definition_hash")
    }
    # Independent stdlib expectation: the projection hash is not the document
    # hash, and both shapes are stored without re-signing.
    assert _stdlib_manifest_hash(projection) == HYBRID_V1_PROJECTION_HASH
    assert document["definition_hash"] != HYBRID_V1_PROJECTION_HASH
    # Default records carry the full document; make the first carry the
    # historical nine-key projection instead.
    records[0]["score_sets"][0]["scorer_snapshot"] = projection
    records[0]["score_sets"][0]["scorer_definition_hash"] = HYBRID_V1_PROJECTION_HASH

    imported = EvalSuiteImportRepository(session, registry=REGISTRY).import_records(records)
    assert imported == 24

    score_repo = EvalScoreSetRepository(session)
    stored_projection = score_repo.get_verified(records[0]["score_sets"][0]["id"])
    assert stored_projection.scorer_snapshot_json == projection
    assert stored_projection.scorer_definition_hash == HYBRID_V1_PROJECTION_HASH
    # The full-document shape is preserved exactly as well.
    stored_document = score_repo.get_verified(records[1]["score_sets"][0]["id"])
    assert stored_document.scorer_snapshot_json == document
    assert stored_document.scorer_definition_hash == document["definition_hash"]


def test_import_preserves_a_synthetic_usage_contract_manifest(session):
    """A record that declares the new field is stored and verified as written."""
    records = _complete_suite()
    manifest = dict(records[0]["run"]["manifest"])
    manifest["usage_contract"] = "tutor-chunk-sum-v1"
    records[0]["run"]["manifest"] = manifest
    records[0]["run"]["manifest_hash"] = _stdlib_manifest_hash(manifest)

    imported = EvalSuiteImportRepository(session, registry=REGISTRY).import_records(records)

    assert imported == len(records)
    stored = EvalRunRepository(session).get_verified(records[0]["run"]["id"])
    assert stored.manifest_json["usage_contract"] == "tutor-chunk-sum-v1"
    assert stored.manifest_hash == records[0]["run"]["manifest_hash"]


# ---------------------------------------------------------------------------
# D5d batch A — explicit Run identity, terminal lifecycle, artifact/relation
# and component integrity on the suite-import boundary.
# ---------------------------------------------------------------------------


def _fixture_records() -> list[dict]:
    return [
        json.loads(line)
        for line in CURATED_FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _assert_zero_rows(session) -> None:
    assert session.scalar(select(func.count()).select_from(EvalRun)) == 0
    assert session.scalar(select(func.count()).select_from(EvalScoreSet)) == 0


def test_committed_fixture_every_record_is_verified_read_after_import(session):
    """Every frozen row is verified-read with its original snapshot/hash bytes.

    The committed fixture predates the optional usage_contract field, so the
    import must neither supplement nor re-sign any manifest, artifact or scorer
    snapshot.  Each record carries two ScoreSets and eight ScorerExecutions.
    """
    records = _fixture_records()
    assert len(records) == 24

    imported = EvalSuiteImportRepository(session, registry=REGISTRY).import_records(records)
    assert imported == 24

    run_repo = EvalRunRepository(session)
    score_repo = EvalScoreSetRepository(session)
    execution_repo = EvalScorerExecutionRepository(session)
    for record in records:
        stored_run = run_repo.get_verified(record["run"]["id"])
        assert stored_run.manifest_json == record["run"]["manifest"]
        assert stored_run.manifest_hash == record["run"]["manifest_hash"]
        assert "usage_contract" not in stored_run.manifest_json
        assert stored_run.candidate_artifact_json == record["run"]["candidate_artifact"]
        assert stored_run.artifact_hash == record["run"]["artifact_hash"]

        assert len(record["score_sets"]) == 2
        assert len(record["executions"]) == 8
        assert len(score_repo.list_verified(record["run"]["id"])) == 2
        for raw_score in record["score_sets"]:
            stored_score = score_repo.get_verified(raw_score["id"])
            assert stored_score.scorer_snapshot_json == raw_score["scorer_snapshot"]
            assert stored_score.scorer_definition_hash == raw_score["scorer_definition_hash"]
            assert stored_score.artifact_input_hash == raw_score["artifact_input_hash"]
        for raw_execution in record["executions"]:
            stored_execution = execution_repo.get_verified(raw_execution["id"])
            assert stored_execution.input_hash == raw_execution["input_hash"]
            assert stored_execution.scorer_id == raw_execution["scorer_id"]
            assert stored_execution.scorer_version == raw_execution["scorer_version"]


def test_import_rejects_manifest_secrets_and_urls_atomically(session):
    for mutation in ({"api_key": "secret-value"}, {"note": "https://example.com/v1"}):
        records = _complete_suite()
        manifest = dict(records[0]["run"]["manifest"])
        manifest.update(mutation)
        records[0]["run"]["manifest"] = manifest
        records[0]["run"]["manifest_hash"] = _stdlib_manifest_hash(manifest)

        with pytest.raises(ValueError):
            EvalSuiteImportRepository(session, registry=REGISTRY).import_records(records)
        _assert_zero_rows(session)


def test_import_requires_explicit_run_identity_fields(session):
    for field in (
        "experiment_id",
        "task_case_id",
        "task_case_version",
        "variant_id",
        "run_profile",
    ):
        records = _complete_suite()
        del records[0]["run"][field]
        with pytest.raises(ValueError):
            EvalSuiteImportRepository(session, registry=REGISTRY).import_records(records)
        _assert_zero_rows(session)


def test_import_requires_every_identity_key_in_the_manifest(session):
    for field in ("task_case_version", "run_profile"):
        records = _complete_suite()
        manifest = dict(records[0]["run"]["manifest"])
        del manifest[field]
        records[0]["run"]["manifest"] = manifest
        records[0]["run"]["manifest_hash"] = _stdlib_manifest_hash(manifest)

        with pytest.raises(ValueError):
            EvalSuiteImportRepository(session, registry=REGISTRY).import_records(records)
        _assert_zero_rows(session)


def test_import_rejects_conflicting_run_and_manifest_identity(session):
    records = _complete_suite()
    manifest = dict(records[0]["run"]["manifest"])
    manifest["variant_id"] = "tutor-v3"
    records[0]["run"]["manifest"] = manifest
    records[0]["run"]["manifest_hash"] = _stdlib_manifest_hash(manifest)

    with pytest.raises(ValueError):
        EvalSuiteImportRepository(session, registry=REGISTRY).import_records(records)
    _assert_zero_rows(session)


def test_import_rejects_score_set_without_a_frozen_artifact(session):
    records = _complete_suite()
    records[0]["run"].pop("candidate_artifact", None)
    records[0]["run"].pop("artifact_hash", None)

    with pytest.raises((ValueError, ChecksumMismatchError)):
        EvalSuiteImportRepository(session, registry=REGISTRY).import_records(records)
    _assert_zero_rows(session)


def test_import_rejects_score_set_without_explicit_artifact_input_hash(session):
    records = _complete_suite()
    del records[0]["score_sets"][0]["artifact_input_hash"]

    with pytest.raises((ValueError, ChecksumMismatchError)):
        EvalSuiteImportRepository(session, registry=REGISTRY).import_records(records)
    _assert_zero_rows(session)


def test_import_rejects_nonterminal_score_set_status(session):
    for status in ("pending", "running", "unknown"):
        records = _complete_suite()
        records[0]["score_sets"][0]["status"] = status
        with pytest.raises(ValueError):
            EvalSuiteImportRepository(session, registry=REGISTRY).import_records(records)
        _assert_zero_rows(session)


def test_import_rejects_execution_with_a_foreign_parent(session):
    record = _fixture_records()[0]
    record["executions"][0]["score_set_id"] = "00000000-0000-0000-0000-000000000000"

    with pytest.raises(ValueError):
        EvalSuiteImportRepository(session, registry=REGISTRY).import_records([record])
    _assert_zero_rows(session)


def test_import_rejects_execution_with_a_wrong_input_hash(session):
    record = _fixture_records()[0]
    record["executions"][0]["input_hash"] = "0" * 64

    with pytest.raises(ChecksumMismatchError):
        EvalSuiteImportRepository(session, registry=REGISTRY).import_records([record])
    _assert_zero_rows(session)


def test_import_rejects_execution_with_an_undeclared_component_version(session):
    record = _fixture_records()[0]
    record["executions"][0]["scorer_version"] = "v9"

    with pytest.raises(ValueError):
        EvalSuiteImportRepository(session, registry=REGISTRY).import_records([record])
    _assert_zero_rows(session)


def test_import_rejects_execution_with_an_invalid_status(session):
    record = _fixture_records()[0]
    record["executions"][0]["status"] = "running"

    with pytest.raises(ValueError):
        EvalSuiteImportRepository(session, registry=REGISTRY).import_records([record])
    _assert_zero_rows(session)


def test_import_rejects_duplicate_execution_identity(session):
    records = _fixture_records()
    donor = records[0]["executions"][0]
    clone = dict(donor)
    clone["id"] = "duplicate-execution-id"
    # Same parent ScoreSet and same scorer id/version as the donor: a duplicate
    # execution identity that the whole-batch check must reject before commit.
    records[0]["executions"].append(clone)

    with pytest.raises(ValueError):
        EvalSuiteImportRepository(session, registry=REGISTRY).import_records(records)
    _assert_zero_rows(session)


def test_import_rejects_nonterminal_run_lifecycle(session):
    for lifecycle in ("queued", "running", ""):
        record = _fixture_records()[0]
        record["run"]["lifecycle"] = lifecycle
        with pytest.raises(ValueError):
            EvalSuiteImportRepository(session, registry=REGISTRY).import_records([record])
        _assert_zero_rows(session)


def test_import_rejects_invalid_or_missing_finished_outcome(session):
    for outcome in ("exploded", None, ""):
        record = _fixture_records()[0]
        record["run"]["lifecycle"] = "finished"
        record["run"]["outcome"] = outcome
        with pytest.raises(ValueError):
            EvalSuiteImportRepository(session, registry=REGISTRY).import_records([record])
        _assert_zero_rows(session)


def test_import_rejects_cancelled_run_carrying_an_outcome(session):
    record = _fixture_records()[0]
    record["run"]["lifecycle"] = "cancelled"
    record["run"]["outcome"] = "success"

    with pytest.raises(ValueError):
        EvalSuiteImportRepository(session, registry=REGISTRY).import_records([record])
    _assert_zero_rows(session)


def test_import_accepts_legitimate_terminal_failure_and_cancelled_rows(session):
    records = _complete_suite()

    # finished failure without a frozen artifact or ScoreSet.
    failure = records[0]
    failure["run"]["lifecycle"] = "finished"
    failure["run"]["outcome"] = "system_failed"
    failure["run"].pop("candidate_artifact", None)
    failure["run"].pop("artifact_hash", None)
    failure["score_sets"] = []
    failure["executions"] = []

    # cancelled Run without a frozen artifact or ScoreSet.
    cancelled = records[1]
    cancelled["run"]["lifecycle"] = "cancelled"
    cancelled["run"]["outcome"] = None
    cancelled["run"].pop("candidate_artifact", None)
    cancelled["run"].pop("artifact_hash", None)
    cancelled["score_sets"] = []
    cancelled["executions"] = []

    # frozen artifact with a cancelled ScoreSet.
    cancelled_score = records[2]
    cancelled_score["score_sets"][0]["status"] = "cancelled"

    imported = EvalSuiteImportRepository(session, registry=REGISTRY).import_records(records)
    assert imported == 24

    run_repo = EvalRunRepository(session)
    score_repo = EvalScoreSetRepository(session)
    stored_failure = run_repo.get_verified(failure["run"]["id"])
    assert stored_failure.outcome == "system_failed"
    assert stored_failure.artifact_hash is None
    stored_cancelled = run_repo.get_verified(cancelled["run"]["id"])
    assert stored_cancelled.lifecycle == "cancelled"
    assert stored_cancelled.outcome is None
    stored_cancelled_score = score_repo.get_verified(cancelled_score["score_sets"][0]["id"])
    assert stored_cancelled_score.status == "cancelled"


def test_import_historical_cannot_bypass_verified_read_or_privacy(session):
    record = _fixture_records()[0]
    raw = record["run"]

    def _row(**overrides):
        fields = {
            "id": "historical-raw",
            "experiment_id": raw["experiment_id"],
            "suite_execution_id": raw.get("suite_execution_id"),
            "task_case_id": raw["task_case_id"],
            "task_case_version": raw["task_case_version"],
            "variant_id": raw["variant_id"],
            "run_profile": raw["run_profile"],
            "lifecycle": "finished",
            "outcome": "success",
            "operational_error_json": None,
            "manifest_json": raw["manifest"],
            "manifest_hash": raw["manifest_hash"],
            "candidate_artifact_json": raw["candidate_artifact"],
            "artifact_hash": raw["artifact_hash"],
        }
        fields.update(overrides)
        return EvalRun(**fields)

    run_repo = EvalRunRepository(session)
    with pytest.raises(ChecksumMismatchError):
        run_repo.import_historical(row=_row(variant_id="mismatch-variant"))
    _assert_zero_rows(session)

    secret_manifest = dict(raw["manifest"])
    secret_manifest["api_key"] = "secret-value"
    with pytest.raises(ValueError):
        run_repo.import_historical(
            row=_row(
                manifest_json=secret_manifest,
                manifest_hash=canonical_hash(secret_manifest),
            )
        )
    _assert_zero_rows(session)


# ---------------------------------------------------------------------------
# D5d batch A — container alias selection is by key presence, not truthiness.
# ---------------------------------------------------------------------------


def test_import_preserves_empty_primary_container_over_nonempty_alias(session):
    """An explicit empty primary value must win over a non-empty alias."""
    records = _fixture_records()
    score = records[0]["score_sets"][0]
    score["aggregate_scores"] = {}
    score["aggregate_scores_json"] = {"groundedness": 4}
    score["findings"] = []
    score["findings_json"] = [{"code": "alias"}]
    execution = records[0]["executions"][0]
    execution["output"] = {}
    execution["output_json"] = {"result": {"alias": True}}
    execution["usage"] = {}
    execution["usage_json"] = {"input_tokens": 4}

    imported = EvalSuiteImportRepository(session, registry=REGISTRY).import_records(records)
    assert imported == 24

    stored_score = EvalScoreSetRepository(session).get_verified(score["id"])
    assert stored_score.aggregate_scores_json == {}
    assert stored_score.findings_json == []
    stored_execution = EvalScorerExecutionRepository(session).get_verified(execution["id"])
    assert stored_execution.output_json == {}
    assert stored_execution.usage_json == {}


def test_import_reads_alias_only_container_fields(session):
    """Alias-only container fields remain readable when the primary is absent."""
    records = _fixture_records()
    score = records[0]["score_sets"][0]
    score.pop("aggregate_scores", None)
    score["aggregate_scores_json"] = {"groundedness": 3}
    score.pop("findings", None)
    score["findings_json"] = [{"code": "alias-only"}]
    execution = records[0]["executions"][0]
    execution.pop("output", None)
    execution["output_json"] = {"result": 3}
    execution.pop("usage", None)
    execution["usage_json"] = {"output_tokens": 2}

    imported = EvalSuiteImportRepository(session, registry=REGISTRY).import_records(records)
    assert imported == 24

    stored_score = EvalScoreSetRepository(session).get_verified(score["id"])
    assert stored_score.aggregate_scores_json == {"groundedness": 3}
    assert stored_score.findings_json == [{"code": "alias-only"}]
    stored_execution = EvalScorerExecutionRepository(session).get_verified(execution["id"])
    assert stored_execution.output_json == {"result": 3}
    assert stored_execution.usage_json == {"output_tokens": 2}


def test_import_does_not_fabricate_absent_container_fields(session):
    """Neither key present stores an explicit None, never a new value."""
    records = _fixture_records()
    score = records[0]["score_sets"][0]
    score.pop("aggregate_scores", None)
    score.pop("aggregate_scores_json", None)
    score.pop("findings", None)
    score.pop("findings_json", None)
    execution = records[0]["executions"][0]
    execution.pop("output", None)
    execution.pop("output_json", None)
    execution.pop("usage", None)
    execution.pop("usage_json", None)

    imported = EvalSuiteImportRepository(session, registry=REGISTRY).import_records(records)
    assert imported == 24

    stored_score = EvalScoreSetRepository(session).get_verified(score["id"])
    assert stored_score.aggregate_scores_json is None
    assert stored_score.findings_json is None
    stored_execution = EvalScorerExecutionRepository(session).get_verified(execution["id"])
    assert stored_execution.output_json is None
    assert stored_execution.usage_json is None
