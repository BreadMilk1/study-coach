"""Contract tests for the bounded Learning Run persistence repositories.

These tests intentionally use a real SQLite engine and SQLAlchemy session.  The
repositories promise application-level append-only and checksum-verified
semantics; the tests therefore exercise both normal calls and direct SQL
tampering at the persistence boundary.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import models
from app.db.models import Base
from app.eval.learning_run.contracts import CandidateArtifact, RunManifest, ScorerExecutionDraft, canonical_hash
from app.eval.learning_run.registry import TaskRegistry


REGISTRY = TaskRegistry.load_default()


CURATED_FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "app"
    / "eval"
    / "learning_run"
    / "fixtures"
    / "tutor-prompt-regression-v1.jsonl"
)

# The first committed manifest record's hash, read from the curated fixture
# before the optional usage_contract field was added. It must not move.
LEGACY_FIXTURE_MANIFEST_HASH = (
    "d71b9f7892a74b1c122ef3280e7bff61990b7454211082e993a462c33bfff3b3"
)


def _first_curated_fixture_record() -> dict:
    with CURATED_FIXTURE_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
    raise AssertionError("curated fixture is empty")


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    with Session(engine) as db_session:
        yield db_session


def _repository_api():
    """Turn a missing Task 3 surface into an explicit feature RED."""

    assert hasattr(models, "EvalRun"), "EvalRun model is missing"
    assert hasattr(models, "EvalScoreSet"), "EvalScoreSet model is missing"
    assert hasattr(models, "EvalScorerExecution"), "EvalScorerExecution model is missing"
    try:
        from app.eval.learning_run.repositories import (
            ChecksumMismatchError,
            EvalRunRepository,
            EvalScoreSetRepository,
            EvalScorerExecutionRepository,
            RepositoryConflictError,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover - intentional RED guard
        pytest.fail(f"learning-run repository module is missing: {exc}", pytrace=False)
    return (
        ChecksumMismatchError,
        EvalRunRepository,
        EvalScoreSetRepository,
        EvalScorerExecutionRepository,
        RepositoryConflictError,
    )


def _manifest(label: str = "manifest") -> RunManifest:
    # The nested task snapshot identity must match the run's own task identity so
    # the execution-derived finalization can cross-check it against the row.
    task_snapshot = dict(REGISTRY.task_cases["tgqa-001"].to_dict())
    task_snapshot["id"] = f"case-{label}"
    task_snapshot["version"] = "1"
    return RunManifest(
        experiment_id="experiment-test",
        task_case_id=f"case-{label}",
        task_case_version="1",
        variant_id="tutor-v2",
        run_profile="evaluation",
        task_snapshot=task_snapshot,
        prompt_text="frozen prompt",
        corpus_snapshot={"snapshot_id": "corpus-test", "version": "1", "aggregate_hash": "c" * 64},
        scorer_snapshot={"id": "hybrid", "version": "v1", "hash": "s" * 64},
        connection_fingerprint="d" * 64,
        corpus_snapshot_id="corpus-test",
        corpus_snapshot_version="1",
        corpus_snapshot_hash="c" * 64,
        prompt_version="tutor-v2",
        prompt_hash="p" * 64,
        scorer_bundle_version="hybrid-v1",
        scorer_bundle_hash="s" * 64,
        provider="test-provider",
        model="test-model",
        model_parameters={"temperature": 0, "max_tokens": 32},
        retrieval_config={"top_k": 5},
        reranker_config={"version": "test-reranker-v1"},
        chunking_config_version="chunking-v1",
        embedding_config_version="embedding-v1",
        budget={"total_seconds": 1},
        runtime_judge=False,
        runner_version="runner-v1",
        schema_version="learning-run-test-v1",
        code_revision="test-revision",
    )


def _artifact(label: str = "artifact", *, usage: object = "unavailable") -> CandidateArtifact:
    return CandidateArtifact(
        answer=f"answer-{label}",
        citations=({"chunk_id": "chunk-1", "page": 1},),
        exact_evidence=({"chunk_id": "chunk-1", "text": "evidence"},),
        formatted_context="evidence",
        usage=usage,
        trace=({"stage": "tutor", "event": "complete"},),
        budget={"total_seconds": 1},
    )


def _create_running_run(session, label: str = "run"):
    _, RunRepository, _, _, _ = _repository_api()
    repo = RunRepository(session)
    manifest = _manifest(label)
    run = repo.create(
        id=f"run-{label}",
        experiment_id="experiment-test",
        task_case_id=f"case-{label}",
        task_case_version="1",
        variant_id="tutor-v2",
        run_profile="evaluation",
        manifest=manifest,
        manifest_hash=manifest.compute_hash(),
    )
    running = repo.claim_running(run.id)
    return repo, running


def _finish_run(session, label: str = "run"):
    repo, running = _create_running_run(session, label)
    artifact = _artifact(label)
    finished = repo.finalize_candidate(
        running.id,
        expected_lifecycle="running",
        candidate_artifact=artifact,
        artifact_hash=artifact.compute_hash(),
    )
    return repo, finished


def _create_score_set(session, finished, *, scorer_version: str = "hybrid-v1"):
    """Create a pending ScoreSet frozen against the full registry document.

    New execution appends and derived finalization require the parent's own
    frozen components, so the synthetic helpers must carry the complete
    document rather than a two-key legacy snapshot.
    """

    _, _, ScoreSetRepository, _, _ = _repository_api()
    bundle = REGISTRY.scorer_for(scorer_version)
    document = dict(REGISTRY.scorer_document(scorer_version))
    score_set = ScoreSetRepository(session).create(
        run_id=finished.id,
        scorer_bundle=bundle,
        scorer_snapshot=document,
        artifact_input_hash=finished.artifact_hash,
    )
    return score_set, bundle


def _score_set_with_document(session, finished, *, scorer_version: str = "hybrid-v1"):
    score_set, bundle = _create_score_set(session, finished, scorer_version=scorer_version)
    _, _, ScoreSetRepository, _, _ = _repository_api()
    claimed = ScoreSetRepository(session).claim_running(score_set.id)
    return claimed, bundle


def _append_success_components(session, score_set, bundle, *, groundedness: int = 4):
    _, _, _, ExecutionRepository, _ = _repository_api()
    executions = ExecutionRepository(session)
    for component in bundle.components:
        if component.kind == "llm":
            output = {
                "groundedness": groundedness,
                "citation_entailment": groundedness,
                "coverage": groundedness,
                "reasoning": "frozen synthetic result",
                "findings": [],
            }
        else:
            output = {"ok": True}
        executions.append_draft(
            score_set.id,
            ScorerExecutionDraft(
                component_id=component.component_id,
                component_version=component.version,
                scorer_id=component.component_id,
                scorer_version=component.version,
                status="success",
                input_hash=score_set.artifact_input_hash,
                output=output,
                findings=(),
            ),
        )


def _finalize_from_derivation(session, score_set_id):
    _, _, ScoreSetRepository, _, _ = _repository_api()
    repo = ScoreSetRepository(session)
    derived = repo.derive_from_executions(score_set_id)
    return repo.finalize_once(
        score_set_id,
        status=derived.status,
        quality_verdict=derived.verdict,
        aggregate_scores=derived.aggregate_scores,
        findings=derived.findings,
        error_code=derived.error_code,
        sanitized_message=derived.error_message,
    )


def test_eval_models_expose_exact_local_instance_global_schema(session):
    _repository_api()
    tables = set(inspect(session.bind).get_table_names())
    assert {"eval_runs", "eval_score_sets", "eval_scorer_executions"} <= tables
    assert "eval_suite_executions" not in tables
    assert not any(
        column["name"] == "user_id"
        for table_name in ("eval_runs", "eval_score_sets", "eval_scorer_executions")
        for column in inspect(session.bind).get_columns(table_name)
    )


def test_run_lifecycle_compare_and_set_and_one_time_candidate_finalize(session):
    _, RunRepository, _, _, Conflict = _repository_api()
    repo = RunRepository(session)
    manifest = _manifest("lifecycle")
    run = repo.create(
        id="run-lifecycle",
        experiment_id="experiment-test",
        task_case_id="case-lifecycle",
        task_case_version="1",
        variant_id="tutor-v2",
        run_profile="evaluation",
        manifest=manifest,
        manifest_hash=manifest.compute_hash(),
    )

    running = repo.claim_running(run.id)
    assert running.lifecycle == "running"
    with pytest.raises(Conflict):
        repo.claim_running(run.id)

    artifact = _artifact("lifecycle")
    finished = repo.finalize_candidate(
        run.id,
        expected_lifecycle="running",
        candidate_artifact=artifact,
        artifact_hash=artifact.compute_hash(),
    )
    assert finished.lifecycle == "finished"
    assert finished.outcome == "success"
    assert finished.candidate_artifact_json == artifact.to_dict()
    assert finished.finished_at is not None

    with pytest.raises(Conflict):
        repo.finalize_candidate(
            run.id,
            expected_lifecycle="running",
            candidate_artifact=_artifact("replacement"),
            artifact_hash=_artifact("replacement").compute_hash(),
        )
    persisted = repo.get_verified(run.id)
    assert persisted.candidate_artifact_json == artifact.to_dict()


def test_freeze_candidate_then_finalize_success_is_two_phase_and_single_use(session):
    _, RunRepository, _, _, Conflict = _repository_api()
    repo, running = _create_running_run(session, "two-phase")
    artifact = _artifact("two-phase")

    frozen = repo.freeze_candidate(
        running.id,
        artifact,
        artifact_hash=artifact.compute_hash(),
    )
    assert frozen.lifecycle == "running"
    assert frozen.outcome is None
    assert frozen.artifact_hash == artifact.compute_hash()

    finished = repo.finalize_success(frozen.id)
    assert finished.lifecycle == "finished"
    assert finished.outcome == "success"
    with pytest.raises(Conflict):
        repo.finalize_success(finished.id)


def test_failure_finalization_preserves_a_frozen_candidate_and_typed_budget(session):
    _, RunRepository, _, _, Conflict = _repository_api()
    repo, running = _create_running_run(session, "failure-with-artifact")
    artifact = _artifact("failure-with-artifact")
    repo.freeze_candidate(running.id, artifact, artifact_hash=artifact.compute_hash())

    failed = repo.finalize_failure(
        running.id,
        outcome="system_failed",
        error_code="harness_internal_error",
        sanitized_message="database callback failed",
        stage="scoring",
        retryable=False,
        spent_budget={"elapsed_seconds": 1.25, "total_limit_seconds": 90},
    )
    assert failed.outcome == "system_failed"
    assert failed.artifact_hash == artifact.compute_hash()
    assert failed.candidate_artifact_json == artifact.to_dict()
    assert failed.operational_error_json == {
        "code": "harness_internal_error",
        "message": "database callback failed",
        "stage": "scoring",
        "retryable": False,
        "spent_budget": {"elapsed_seconds": 1.25, "total_limit_seconds": 90},
    }
    with pytest.raises(Conflict):
        repo.finalize_failure(
            running.id,
            outcome="system_failed",
            error_code="harness_internal_error",
        )


def test_run_terminal_finalization_is_running_only_and_rejects_queued_without_mutation(session):
    _, RunRepository, _, _, Conflict = _repository_api()
    repo = RunRepository(session)

    candidate_run = repo.create(
        id="run-queued-candidate",
        experiment_id="experiment-test",
        task_case_id="case-queued-candidate",
        task_case_version="1",
        variant_id="tutor-v2",
        run_profile="evaluation",
        manifest=_manifest("queued-candidate"),
        manifest_hash=_manifest("queued-candidate").compute_hash(),
    )
    candidate = _artifact("queued-candidate")
    with pytest.raises(Conflict):
        repo.finalize_candidate(
            candidate_run.id,
            expected_lifecycle="queued",
            candidate_artifact=candidate,
            artifact_hash=candidate.compute_hash(),
        )
    persisted_candidate = repo.get_verified(candidate_run.id)
    assert persisted_candidate.lifecycle == "queued"
    assert persisted_candidate.outcome is None
    assert persisted_candidate.candidate_artifact_json is None

    failure_run = repo.create(
        id="run-queued-failure",
        experiment_id="experiment-test",
        task_case_id="case-queued-failure",
        task_case_version="1",
        variant_id="tutor-v2",
        run_profile="evaluation",
        manifest=_manifest("queued-failure"),
        manifest_hash=_manifest("queued-failure").compute_hash(),
    )
    with pytest.raises(Conflict):
        repo.finalize_failure(
            failure_run.id,
            expected_lifecycle="queued",
            outcome="system_failed",
            error_code="harness_internal_error",
            sanitized_message="not started",
        )
    persisted_failure = repo.get_verified(failure_run.id)
    assert persisted_failure.lifecycle == "queued"
    assert persisted_failure.outcome is None
    assert persisted_failure.operational_error_json is None


def test_run_repository_requires_typed_manifest_and_candidate_artifact(session):
    _, RunRepository, _, _, _ = _repository_api()
    repo = RunRepository(session)
    manifest = _manifest("typed-boundary")

    with pytest.raises(TypeError):
        repo.create(
            id="run-mapping-manifest",
            experiment_id="experiment-test",
            task_case_id="case-typed-boundary",
            task_case_version="1",
            variant_id="tutor-v2",
            run_profile="evaluation",
            manifest=manifest.payload(),
            manifest_hash=manifest.compute_hash(),
        )

    run = repo.create(
        id="run-mapping-artifact",
        experiment_id="experiment-test",
        task_case_id="case-typed-boundary",
        task_case_version="1",
        variant_id="tutor-v2",
        run_profile="evaluation",
        manifest=manifest,
        manifest_hash=manifest.compute_hash(),
    )
    repo.claim_running(run.id)
    artifact = _artifact("mapping-boundary")
    with pytest.raises(TypeError):
        repo.finalize_candidate(
            run.id,
            candidate_artifact=artifact.to_dict(),
            artifact_hash=artifact.compute_hash(),
        )
    assert repo.get_verified(run.id).lifecycle == "running"


@pytest.mark.parametrize("usage", [None, 0, "fabricated"])
def test_candidate_artifact_usage_must_be_available_mapping_or_unavailable(session, usage):
    _, RunRepository, _, _, _ = _repository_api()
    repo, running = _create_running_run(session, f"invalid-usage-{usage}")
    artifact = _artifact(f"invalid-usage-{usage}", usage=usage)

    with pytest.raises((TypeError, ValueError)):
        repo.finalize_candidate(
            running.id,
            candidate_artifact=artifact,
            artifact_hash=artifact.compute_hash(),
        )
    assert repo.get_verified(running.id).lifecycle == "running"


def test_run_failure_and_cancel_transitions_are_single_terminal_writes(session):
    _, RunRepository, _, _, Conflict = _repository_api()
    repo = RunRepository(session)

    failed_run = repo.create(
        id="run-timeout",
        experiment_id="experiment-test",
        task_case_id="case-timeout",
        task_case_version="1",
        variant_id="tutor-v2",
        run_profile="evaluation",
        manifest=_manifest("timeout"),
        manifest_hash=_manifest("timeout").compute_hash(),
    )
    repo.claim_running(failed_run.id)
    failed = repo.finalize_failure(
        failed_run.id,
        expected_lifecycle="running",
        outcome="timed_out",
        error_code="generation_timeout",
        sanitized_message="model deadline exceeded",
    )
    assert failed.lifecycle == "finished"
    assert failed.outcome == "timed_out"
    assert failed.operational_error_json == {
        "code": "generation_timeout",
        "message": "model deadline exceeded",
    }
    with pytest.raises(Conflict):
        repo.finalize_failure(
            failed_run.id,
            expected_lifecycle="running",
            outcome="system_failed",
            error_code="harness_internal_error",
            sanitized_message="second terminal write",
        )

    cancelled_run = repo.create(
        id="run-cancelled",
        experiment_id="experiment-test",
        task_case_id="case-cancelled",
        task_case_version="1",
        variant_id="tutor-v2",
        run_profile="evaluation",
        manifest=_manifest("cancelled"),
        manifest_hash=_manifest("cancelled").compute_hash(),
    )
    repo.claim_running(cancelled_run.id)
    cancelled = repo.cancel_once(
        cancelled_run.id,
        expected_lifecycle="running",
        error_code="cancelled",
        sanitized_message="user requested cancellation",
    )
    assert cancelled.lifecycle == "cancelled"
    with pytest.raises(Conflict):
        repo.cancel_once(cancelled_run.id, expected_lifecycle="running")


def test_run_cancellation_cannot_skip_queued_to_running_transition(session):
    _, RunRepository, _, _, Conflict = _repository_api()
    repo = RunRepository(session)
    manifest = _manifest("queued-cancel")
    run = repo.create(
        id="run-queued-cancel",
        experiment_id="experiment-test",
        task_case_id="case-queued-cancel",
        task_case_version="1",
        variant_id="tutor-v2",
        run_profile="evaluation",
        manifest=manifest,
        manifest_hash=manifest.compute_hash(),
    )

    with pytest.raises(Conflict):
        repo.cancel_once(run.id, expected_lifecycle="queued")
    assert repo.get_verified(run.id).lifecycle == "queued"


def test_missing_transition_ids_are_not_reported_as_state_conflicts(session):
    _, RunRepository, _, _, _ = _repository_api()
    from app.eval.learning_run.repositories import RepositoryNotFoundError

    repo = RunRepository(session)

    with pytest.raises(RepositoryNotFoundError):
        repo.claim_running("run-does-not-exist")


def test_operational_error_messages_remove_tracebacks_and_secrets(session):
    _, RunRepository, _, _, _ = _repository_api()
    repo = RunRepository(session)
    manifest = _manifest("sanitized")
    run = repo.create(
        id="run-sanitized",
        experiment_id="experiment-test",
        task_case_id="case-sanitized",
        task_case_version="1",
        variant_id="tutor-v2",
        run_profile="evaluation",
        manifest=manifest,
        manifest_hash=manifest.compute_hash(),
    )
    repo.claim_running(run.id)

    failed = repo.finalize_failure(
        run.id,
        outcome="system_failed",
        error_code="model_unavailable",
        sanitized_message=(
            "provider failed with api_key=secret-value\n"
            "Traceback (most recent call last): internal stack details"
        ),
    )
    assert failed.operational_error_json == {
        "code": "model_unavailable",
        "message": "provider failed with [redacted]",
    }


def test_error_codes_are_fixed_allowlist_and_rejected_before_any_write(session):
    _, RunRepository, ScoreSetRepository, ExecutionRepository, _ = _repository_api()

    run_repo = RunRepository(session)
    manifest = _manifest("invalid-error-code")
    run = run_repo.create(
        id="run-invalid-error-code",
        experiment_id="experiment-test",
        task_case_id="case-invalid-error-code",
        task_case_version="1",
        variant_id="tutor-v2",
        run_profile="evaluation",
        manifest=manifest,
        manifest_hash=manifest.compute_hash(),
    )
    run_repo.claim_running(run.id)

    with pytest.raises(ValueError):
        run_repo.finalize_failure(
            run.id,
            outcome="system_failed",
            error_code="api_key=secret-value",
            sanitized_message="must not persist",
        )
    assert run_repo.get_verified(run.id).lifecycle == "running"

    artifact = _artifact("invalid-error-code")
    run_repo.finalize_candidate(
        run.id,
        candidate_artifact=artifact,
        artifact_hash=artifact.compute_hash(),
    )
    score_repo = ScoreSetRepository(session)
    score_set = score_repo.create(
        run_id=run.id,
        scorer_id="hybrid",
        scorer_version="v1",
        artifact_input_hash=artifact.compute_hash(),
    )
    score_repo.claim_running(score_set.id)
    with pytest.raises(ValueError):
        score_repo.finalize_once(
            score_set.id,
            status="failed",
            quality_verdict="inconclusive",
            error_code="not-an-approved-error",
            sanitized_message="must not persist",
        )
    assert score_repo.get_verified(score_set.id).status == "running"

    executions = ExecutionRepository(session)
    with pytest.raises(ValueError):
        executions.append(
            score_set_id=score_set.id,
            scorer_id="hybrid-deterministic",
            scorer_version="v1",
            status="failed",
            input_hash=artifact.compute_hash(),
            error_code="secret-token",
            sanitized_message="must not persist",
        )
    assert executions.list_verified(score_set.id) == []


def test_manifest_and_artifact_hashes_are_verified_before_reads(session):
    _, finished = _finish_run(session, "tamper")
    ChecksumMismatch, RunRepository, _, _, _ = _repository_api()
    repo = RunRepository(session)

    session.execute(
        text("UPDATE eval_runs SET manifest_json = :payload WHERE id = :run_id"),
        {"payload": '{"tampered": true}', "run_id": finished.id},
    )
    session.commit()
    with pytest.raises(ChecksumMismatch):
        repo.get_verified(finished.id)


def test_candidate_artifact_tamper_blocks_compare_and_rescore_access(session):
    _, finished = _finish_run(session, "artifact-tamper")
    ChecksumMismatch, RunRepository, ScoreSetRepository, _, _ = _repository_api()
    run_repo = RunRepository(session)
    score_repo = ScoreSetRepository(session)
    score_set = score_repo.create(
        run_id=finished.id,
        scorer_id="hybrid",
        scorer_version="v1",
        artifact_input_hash=finished.artifact_hash,
    )

    session.execute(
        text("UPDATE eval_runs SET candidate_artifact_json = :payload WHERE id = :run_id"),
        {
            "payload": '{"answer": "tampered", "citations": []}',
            "run_id": finished.id,
        },
    )
    session.commit()
    with pytest.raises(ChecksumMismatch):
        run_repo.get_verified(finished.id)
    with pytest.raises(ChecksumMismatch):
        score_repo.get_for_compare(score_set.id)
    with pytest.raises(ChecksumMismatch):
        score_repo.get_for_rescore(score_set.id)


def test_score_set_history_is_append_only_and_each_row_terminal_once(session):
    _, finished = _finish_run(session, "scores")
    _, _, ScoreSetRepository, _, Conflict = _repository_api()
    score_repo = ScoreSetRepository(session)
    artifact_hash = finished.artifact_hash

    first, bundle = _score_set_with_document(session, finished)
    second = score_repo.create(
        run_id=finished.id,
        scorer_id="hybrid",
        scorer_version="v1",
        artifact_input_hash=artifact_hash,
    )
    assert first.id != second.id
    assert second.status == "pending"
    assert first.scorer_snapshot_json
    assert first.scorer_definition_hash

    assert first.status == "running"
    _append_success_components(session, first, bundle)
    completed = _finalize_from_derivation(session, first.id)
    assert completed.status == "completed"
    assert completed.quality_verdict == "pass"
    assert completed.aggregate_scores_json == {
        "groundedness": 4,
        "citation_entailment": 4,
        "coverage": 4,
    }
    with pytest.raises(Conflict):
        score_repo.finalize_once(
            first.id,
            status="failed",
            quality_verdict="inconclusive",
            error_code="scorer_timeout",
            sanitized_message="late result",
        )
    cancelled = score_repo.cancel_once(second.id, error_code="cancelled")
    assert cancelled.status == "cancelled"
    assert score_repo.get_verified(first.id).status == "completed"


def _stdlib_hash_without_field(payload: dict, field: str) -> str:
    """Independent definition hash, computed with the standard library only."""
    import hashlib

    body = {key: value for key, value in payload.items() if key != field}
    return hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def test_create_rejects_an_invalid_scorer_document_before_insert(session):
    from app.eval.learning_run.registry import TaskRegistry

    ChecksumMismatch, _, ScoreSetRepository, _, _ = _repository_api()
    _, finished = _finish_run(session, "scorer-document")
    registry = TaskRegistry.load_default()
    bundle = registry.scorer_for("hybrid-v1")
    document = dict(registry.scorer_document("hybrid-v1"))
    score_repo = ScoreSetRepository(session)

    def _create(snapshot, *, scorer_bundle=bundle, expected=None):
        return score_repo.create(
            run_id=finished.id,
            artifact_input_hash=finished.artifact_hash,
            scorer_bundle=scorer_bundle,
            scorer_snapshot=snapshot,
            scorer_definition_hash=expected,
        )

    bad_hash = {**document, "definition_hash": "0" * 64}
    missing_hash = {key: value for key, value in document.items() if key != "definition_hash"}
    missing_schema = {key: value for key, value in document.items() if key != "schema_version"}
    empty_schema = {**document, "schema_version": ""}
    illegal_schema = {**document, "schema_version": 7}
    # A wrong non-empty schema still changes the document hash, so a
    # self-consistent document can never match the executing bundle's hash.
    wrong_schema = {**document, "schema_version": "scorer-bundle-v2"}
    wrong_schema["definition_hash"] = _stdlib_hash_without_field(
        wrong_schema, "definition_hash"
    )
    # Self-consistent (stdlib-hashed) but a different definition than the
    # executing bundle declares.
    other_definition = {**document, "parser_version": "json-rubric-v1-other"}
    other_definition["definition_hash"] = _stdlib_hash_without_field(
        other_definition, "definition_hash"
    )

    rejected = {
        "hash": (bad_hash, bundle, None),
        "missing_hash": (missing_hash, bundle, None),
        "missing_schema": (missing_schema, bundle, None),
        "empty_schema": (empty_schema, bundle, None),
        "illegal_schema": (illegal_schema, bundle, None),
        "wrong_schema": (wrong_schema, bundle, None),
        "bundle_hash_mismatch": (other_definition, bundle, None),
        "bundle_missing_hash": (document, replace(bundle, definition_hash=""), None),
        "explicit_hash_mismatch": (document, bundle, "1" * 64),
        "bundle_id_mismatch": (document, replace(bundle, scorer_id="other"), None),
        "bundle_version_mismatch": (document, replace(bundle, version="hybrid-v9"), None),
        "bundle_projection_mismatch": (
            document,
            replace(bundle, parser_version="json-rubric-v1-tampered"),
            None,
        ),
    }
    for name, (snapshot, scorer_bundle, expected) in rejected.items():
        with pytest.raises(ChecksumMismatch):
            _create(snapshot, scorer_bundle=scorer_bundle, expected=expected)
        assert session.query(models.EvalScoreSet).count() == 0, name

    created = _create(document)
    assert created.scorer_snapshot_json == document
    assert created.scorer_definition_hash == document["definition_hash"]


def test_score_set_finalize_requires_running_and_rejects_pending_without_mutation(session):
    _, finished = _finish_run(session, "pending-finalize")
    _, _, ScoreSetRepository, _, Conflict = _repository_api()
    score_repo = ScoreSetRepository(session)
    score_set = score_repo.create(
        run_id=finished.id,
        scorer_id="hybrid",
        scorer_version="v1",
        artifact_input_hash=finished.artifact_hash,
    )

    with pytest.raises(Conflict):
        score_repo.finalize_once(
            score_set.id,
            status="completed",
            quality_verdict="pass",
            aggregate_scores={"groundedness": 4},
            findings=[],
        )
    assert score_repo.get_verified(score_set.id).status == "pending"


def test_scorer_execution_append_requires_running_score_set_and_rejects_terminal_parent(session):
    _, finished = _finish_run(session, "execution-lifecycle")
    _, _, ScoreSetRepository, ExecutionRepository, Conflict = _repository_api()
    score_repo = ScoreSetRepository(session)
    executions = ExecutionRepository(session)
    score_set, bundle = _create_score_set(session, finished)

    with pytest.raises(Conflict):
        executions.append(
            score_set_id=score_set.id,
            scorer_id=bundle.components[0].component_id,
            scorer_version=bundle.components[0].version,
            status="success",
            input_hash=finished.artifact_hash,
            output={"citation_entailment": 4},
        )
    assert executions.list_verified(score_set.id) == []

    score_repo.claim_running(score_set.id)
    _append_success_components(session, score_set, bundle)
    _finalize_from_derivation(session, score_set.id)
    with pytest.raises(Conflict):
        executions.append(
            score_set_id=score_set.id,
            scorer_id="late-scorer",
            scorer_version="v1",
            status="success",
            input_hash=finished.artifact_hash,
            output={"late": True},
        )


def test_score_set_and_scorer_input_checksums_block_compare_and_rescore(session):
    _, finished = _finish_run(session, "score-tamper")
    ChecksumMismatch, _, ScoreSetRepository, _, _ = _repository_api()
    score_repo = ScoreSetRepository(session)
    score_set = score_repo.create(
        run_id=finished.id,
        scorer_id="hybrid",
        scorer_version="v1",
        artifact_input_hash=finished.artifact_hash,
    )

    session.execute(
        text(
            "UPDATE eval_score_sets SET artifact_input_hash = :hash "
            "WHERE id = :score_set_id"
        ),
        {"hash": "0" * 64, "score_set_id": score_set.id},
    )
    session.commit()
    with pytest.raises(ChecksumMismatch):
        score_repo.get_verified(score_set.id)
    with pytest.raises(ChecksumMismatch):
        score_repo.get_for_compare(score_set.id)
    with pytest.raises(ChecksumMismatch):
        score_repo.get_for_rescore(score_set.id)


def test_scorer_executions_append_only_verified_and_duplicate_identity_rejected(session):
    _, finished = _finish_run(session, "executions")
    ChecksumMismatch, _, ScoreSetRepository, ExecutionRepository, Conflict = _repository_api()
    score_repo = ScoreSetRepository(session)
    score_set, bundle = _score_set_with_document(session, finished)
    component = bundle.components[0]
    executions = ExecutionRepository(session)
    execution = executions.append(
        score_set_id=score_set.id,
        scorer_id=component.component_id,
        scorer_version=component.version,
        status="success",
        input_hash=finished.artifact_hash,
        output={"citation_entailment": 4},
        usage=None,
    )
    assert execution.output_json == {"citation_entailment": 4}
    assert execution.usage_json is None
    with pytest.raises(Conflict):
        executions.append(
            score_set_id=score_set.id,
            scorer_id=component.component_id,
            scorer_version=component.version,
            status="success",
            input_hash=finished.artifact_hash,
            output={"citation_entailment": 5},
        )
    assert not hasattr(executions, "update")
    assert not hasattr(executions, "delete")
    assert executions.get_verified(execution.id).output_json == {
        "citation_entailment": 4
    }

    session.execute(
        text(
            "UPDATE eval_scorer_executions SET input_hash = :hash "
            "WHERE id = :execution_id"
        ),
        {"hash": "1" * 64, "execution_id": execution.id},
    )
    session.commit()
    with pytest.raises(ChecksumMismatch):
        executions.get_verified(execution.id)
    with pytest.raises(ChecksumMismatch):
        executions.list_verified(score_set.id)


@pytest.mark.parametrize("usage_field", ["usage", "usage_json"])
@pytest.mark.parametrize(
    "invalid_usage",
    [0, -1, 1, True, False, "unavailable"],
)
def test_scorer_execution_rejects_fabricated_scalar_usage(
    session,
    usage_field,
    invalid_usage,
):
    _, finished = _finish_run(
        session,
        f"invalid-scorer-usage-{usage_field}-{str(invalid_usage).lower()}",
    )
    _, _, ScoreSetRepository, ExecutionRepository, Conflict = _repository_api()
    score_repo = ScoreSetRepository(session)
    score_set, bundle = _score_set_with_document(session, finished)
    component = bundle.components[0]
    executions = ExecutionRepository(session)

    with pytest.raises((TypeError, ValueError)):
        executions.append(
            score_set_id=score_set.id,
            scorer_id=component.component_id,
            scorer_version=component.version,
            status="success",
            input_hash=finished.artifact_hash,
            output={"groundedness": 4},
            **{usage_field: invalid_usage},
        )
    assert executions.list_verified(score_set.id) == []


@pytest.mark.parametrize("usage_field", ["usage", "usage_json"])
@pytest.mark.parametrize("usage_value", [{"input_tokens": 4}, None])
def test_scorer_execution_accepts_mapping_or_null_usage(
    session,
    usage_field,
    usage_value,
):
    _, finished = _finish_run(
        session,
        f"valid-scorer-usage-{usage_field}-{usage_value is None}",
    )
    _, _, ScoreSetRepository, ExecutionRepository, _ = _repository_api()
    score_set, bundle = _score_set_with_document(session, finished)
    component = bundle.components[0]
    execution = ExecutionRepository(session).append(
        score_set_id=score_set.id,
        scorer_id=component.component_id,
        scorer_version=component.version,
        status="success",
        input_hash=finished.artifact_hash,
        output={"groundedness": 4},
        **{usage_field: usage_value},
    )

    assert execution.usage_json == usage_value
    persisted_usage = session.execute(
        text(
            "SELECT usage_json FROM eval_scorer_executions WHERE id = :execution_id"
        ),
        {"execution_id": execution.id},
    ).scalar_one()
    if usage_value is None:
        assert persisted_usage is None
    else:
        assert persisted_usage == '{"input_tokens": 4}'


def test_scorer_execution_append_rechecks_running_parent_atomically_under_interleaving(
    tmp_path,
    monkeypatch,
):
    """A parent terminal transition between eligibility observation and INSERT must win."""

    db_url = f"sqlite:///{tmp_path / 'scorer_append_race.db'}"
    engine = create_engine(db_url, connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    session_a = Session(engine)
    session_b = Session(engine)
    try:
        run_repo_a = _repository_api()[1](session_a)
        manifest = _manifest("scorer-append-race")
        run = run_repo_a.create(
            id="run-scorer-append-race",
            experiment_id="experiment-test",
            task_case_id="case-scorer-append-race",
            task_case_version="1",
            variant_id="tutor-v2",
            run_profile="evaluation",
            manifest=manifest,
            manifest_hash=manifest.compute_hash(),
        )
        run_repo_a.claim_running(run.id)
        artifact = _artifact("scorer-append-race")
        finished = run_repo_a.finalize_candidate(
            run.id,
            candidate_artifact=artifact,
            artifact_hash=artifact.compute_hash(),
        )
        score_repo_a = _repository_api()[2](session_a)
        bundle = REGISTRY.scorer_for("hybrid-v1")
        document = dict(REGISTRY.scorer_document("hybrid-v1"))
        score_set = score_repo_a.create(
            run_id=finished.id,
            scorer_bundle=bundle,
            scorer_snapshot=document,
            artifact_input_hash=finished.artifact_hash,
        )
        score_repo_a.claim_running(score_set.id)
        score_repo_b = _repository_api()[2](session_b)
        executions = _repository_api()[3](session_a)
        observed = {"running": False, "terminalized": False}
        original_get_verified = executions.score_sets.get_verified

        def observe_running(score_set_id):
            row = original_get_verified(score_set_id)
            if not observed["running"]:
                observed["running"] = True
                stale_snapshot = SimpleNamespace(
                    status=row.status,
                    artifact_input_hash=row.artifact_input_hash,
                    scorer_snapshot_json=row.scorer_snapshot_json,
                )
                assert stale_snapshot.status == "running"
                session_a.rollback()
                score_repo_b.cancel_once(score_set_id, error_code="cancelled")
                assert score_repo_b.get_verified(score_set_id).status == "cancelled"
                observed["terminalized"] = True
                return stale_snapshot
            return row

        monkeypatch.setattr(executions.score_sets, "get_verified", observe_running)

        with pytest.raises(_repository_api()[4]):
            executions.append(
                score_set_id=score_set.id,
                scorer_id=bundle.components[0].component_id,
                scorer_version=bundle.components[0].version,
                status="success",
                input_hash=run_repo_a.get_verified(run.id).artifact_hash,
                output={"groundedness": 4},
            )

        assert observed == {"running": True, "terminalized": True}
        assert executions.list_verified(score_set.id) == []
        assert score_repo_b.get_verified(score_set.id).status == "cancelled"
    finally:
        session_a.close()
        session_b.close()
        engine.dispose()


@pytest.mark.parametrize(
    "message",
    [
        "Authorization: Bearer super-secret-token",
        "authorization: bearer lower-secret-token",
        "Authorization super-secret-token",
        "{'Authorization': 'Bearer repr-secret-token'}",
    ],
)
def test_operational_error_sanitizer_redacts_full_authorization_values(session, message):
    _, RunRepository, _, _, _ = _repository_api()
    repo = RunRepository(session)
    label = message.split("-")[-1].replace("'", "")
    manifest = _manifest(f"authorization-{label}")
    run = repo.create(
        id=f"run-authorization-{label}",
        experiment_id="experiment-test",
        task_case_id=f"case-authorization-{label}",
        task_case_version="1",
        variant_id="tutor-v2",
        run_profile="evaluation",
        manifest=manifest,
        manifest_hash=manifest.compute_hash(),
    )
    repo.claim_running(run.id)
    failed = repo.finalize_failure(
        run.id,
        outcome="system_failed",
        error_code="model_unavailable",
        sanitized_message=message,
    )
    persisted = failed.operational_error_json["message"]
    assert "authorization" not in persisted.lower()
    assert "bearer" not in persisted.lower()
    assert "secret-token" not in persisted.lower()


@pytest.mark.parametrize(
    "message, forbidden_value, forbidden_field",
    [
        (
            "Authorization: Basic dXNlcjpwYXNz",
            "dXNlcjpwYXNz",
            "authorization",
        ),
        (
            "{'Authorization': 'Basic dXNlcjpwYXNz'}",
            "dXNlcjpwYXNz",
            "authorization",
        ),
        ("password=super-secret", "super-secret", "password="),
        ("accessToken=secret", "secret", "accesstoken="),
        ("client_secret: secret", "secret", "client_secret:"),
        ("cookie=session-secret", "session-secret", "cookie="),
    ],
)
def test_operational_error_sanitizer_redacts_common_credential_assignments(
    session,
    message,
    forbidden_value,
    forbidden_field,
):
    _, RunRepository, _, _, _ = _repository_api()
    repo = RunRepository(session)
    manifest = _manifest("credential-assignments")
    run = repo.create(
        experiment_id=manifest.experiment_id,
        task_case_id=manifest.task_case_id,
        task_case_version=manifest.task_case_version,
        variant_id=manifest.variant_id,
        run_profile=manifest.run_profile,
        manifest=manifest,
        manifest_hash=manifest.compute_hash(),
    )
    repo.claim_running(run.id)
    failed = repo.finalize_failure(
        run.id,
        outcome="system_failed",
        error_code="model_unavailable",
        sanitized_message=message,
    )
    persisted = failed.operational_error_json["message"]
    assert forbidden_value not in persisted
    assert forbidden_field not in persisted.lower()


@pytest.mark.parametrize(
    "manifest_field, sensitive_value",
    [
        ("model_parameters", {"api_key": "secret"}),
        ("model_parameters", {"nested": {"authorization": "Bearer secret"}}),
        ("retrieval_config", {"nested": {"token": "secret"}}),
        ("reranker_config", {"password": "secret"}),
        ("model_parameters", {"secret": "secret"}),
        ("retrieval_config", {"endpoint": "https://example.com/v1"}),
        ("reranker_config", {"base_url": "http://localhost:8000"}),
        ("model_parameters", {"prompt": "call https://example.com/v1"}),
        ("model_parameters", {"authorizationHeader": "Bearer secret"}),
        ("model_parameters", {"accessToken": "secret"}),
        ("retrieval_config", {"refreshToken": "secret"}),
        ("reranker_config", {"clientSecret": "secret"}),
        ("model_parameters", {"apiKey": "secret"}),
        ("retrieval_config", {"sessionToken": "secret"}),
    ],
)
def test_manifest_privacy_boundary_rejects_credentials_and_endpoints_before_insert(
    session,
    manifest_field,
    sensitive_value,
):
    _, RunRepository, _, _, _ = _repository_api()
    repo = RunRepository(session)
    manifest = replace(_manifest(f"privacy-{manifest_field}"), **{manifest_field: sensitive_value})

    with pytest.raises(ValueError):
        repo.create(
            id=f"run-privacy-{manifest_field}-{len(session.new)}",
            experiment_id=manifest.experiment_id,
            task_case_id=manifest.task_case_id,
            task_case_version=manifest.task_case_version,
            variant_id=manifest.variant_id,
            run_profile=manifest.run_profile,
            manifest=manifest,
            manifest_hash=manifest.compute_hash(),
        )
    assert session.execute(text("SELECT COUNT(*) FROM eval_runs")).scalar_one() == 0


@pytest.mark.parametrize(
    "identity_field",
    ["experiment_id", "task_case_id", "task_case_version", "variant_id", "run_profile"],
)
def test_create_rejects_identity_mismatch_with_typed_manifest_before_insert(
    session,
    identity_field,
):
    _, RunRepository, _, _, _ = _repository_api()
    repo = RunRepository(session)
    manifest = _manifest(f"identity-{identity_field}")
    args = {
        "experiment_id": manifest.experiment_id,
        "task_case_id": manifest.task_case_id,
        "task_case_version": manifest.task_case_version,
        "variant_id": manifest.variant_id,
        "run_profile": manifest.run_profile,
    }
    args[identity_field] = f"mismatch-{identity_field}"

    with pytest.raises(ValueError):
        repo.create(
            id=f"run-identity-{identity_field}",
            manifest=manifest,
            manifest_hash=manifest.compute_hash(),
            **args,
        )
    assert session.execute(text("SELECT COUNT(*) FROM eval_runs")).scalar_one() == 0


def test_generated_repository_ids_are_canonical_uuid_strings(session):
    _, RunRepository, ScoreSetRepository, ExecutionRepository, _ = _repository_api()
    run_repo = RunRepository(session)
    manifest = _manifest("uuid-ids")
    run = run_repo.create(
        experiment_id=manifest.experiment_id,
        task_case_id=manifest.task_case_id,
        task_case_version=manifest.task_case_version,
        variant_id=manifest.variant_id,
        run_profile=manifest.run_profile,
        manifest=manifest,
        manifest_hash=manifest.compute_hash(),
    )
    run_repo.claim_running(run.id)
    artifact = _artifact("uuid-ids")
    finished = run_repo.finalize_candidate(
        run.id,
        candidate_artifact=artifact,
        artifact_hash=artifact.compute_hash(),
    )
    score_repo = ScoreSetRepository(session)
    score_set, bundle = _score_set_with_document(session, finished)
    execution = ExecutionRepository(session).append(
        score_set_id=score_set.id,
        scorer_id=bundle.components[0].component_id,
        scorer_version=bundle.components[0].version,
        status="success",
        input_hash=finished.artifact_hash,
        output={"groundedness": 4},
    )

    for row_id in (run.id, score_set.id, execution.id):
        assert len(row_id) == 36
        assert str(UUID(row_id)) == row_id


def test_manifest_hash_field_and_explicit_hash_must_both_match_content(session):
    ChecksumMismatch, RunRepository, _, _, _ = _repository_api()
    repo = RunRepository(session)
    manifest = replace(_manifest("hash-sources"), manifest_hash="0" * 64)

    with pytest.raises(ChecksumMismatch):
        repo.create(
            id="run-hash-sources",
            experiment_id=manifest.experiment_id,
            task_case_id=manifest.task_case_id,
            task_case_version=manifest.task_case_version,
            variant_id=manifest.variant_id,
            run_profile=manifest.run_profile,
            manifest=manifest,
            manifest_hash=manifest.compute_hash(),
        )
    assert session.execute(text("SELECT COUNT(*) FROM eval_runs")).scalar_one() == 0


def test_checksum_mismatch_is_direct_sibling_not_cas_conflict():
    from app.eval.learning_run.repositories import (
        ChecksumMismatchError,
        LearningRunRepositoryError,
        RepositoryConflictError,
    )

    assert ChecksumMismatchError.__bases__ == (LearningRunRepositoryError,)
    assert issubclass(ChecksumMismatchError, LearningRunRepositoryError)
    assert not issubclass(ChecksumMismatchError, RepositoryConflictError)


def test_foreign_keys_require_child_first_deletion_and_leave_no_violations(session):
    _, finished = _finish_run(session, "delete-order")
    _, _, ScoreSetRepository, ExecutionRepository, _ = _repository_api()
    score_set, bundle = _score_set_with_document(session, finished)
    execution = ExecutionRepository(session).append(
        score_set_id=score_set.id,
        scorer_id=bundle.components[0].component_id,
        scorer_version=bundle.components[0].version,
        status="skipped",
        input_hash=finished.artifact_hash,
    )

    with pytest.raises(IntegrityError):
        session.delete(finished)
        session.commit()
    session.rollback()

    session.delete(execution)
    session.commit()
    session.delete(score_set)
    session.commit()
    session.delete(finished)
    session.commit()
    violations = session.execute(text("PRAGMA foreign_key_check")).all()
    assert violations == []


def test_append_draft_persists_stable_output_envelope_and_rejects_untyped_values(session):
    _, finished = _finish_run(session, "draft-envelope")
    _, _, ScoreSetRepository, ExecutionRepository, _ = _repository_api()
    score_set, bundle = _score_set_with_document(session, finished)
    executions = ExecutionRepository(session)
    first_component = bundle.components[0]
    second_component = bundle.components[1]
    success = executions.append_draft(
        score_set.id,
        ScorerExecutionDraft(
            component_id=first_component.component_id,
            component_version=first_component.version,
            scorer_id=first_component.component_id,
            scorer_version=first_component.version,
            status="success",
            input_hash=finished.artifact_hash,
            output={"score": 4},
            findings=({"code": "incomplete_answer", "severity": "noncritical"},),
        ),
    )
    assert success.output_json == {
        "result": {"score": 4},
        "findings": [{"code": "incomplete_answer", "severity": "noncritical"}],
    }
    skipped = executions.append_draft(
        score_set.id,
        ScorerExecutionDraft(
            component_id=second_component.component_id,
            component_version=second_component.version,
            scorer_id=second_component.component_id,
            scorer_version=second_component.version,
            status="skipped",
            input_hash=finished.artifact_hash,
            output={"applicable": False},
        ),
    )
    assert skipped.output_json == {"result": {"applicable": False}, "findings": []}
    with pytest.raises(TypeError):
        executions.append_draft(score_set.id, {"status": "success"})


# ---------------------------------------------------------------------------
# D5c-2 — the optional manifest ``usage_contract`` field.
# ---------------------------------------------------------------------------


def test_legacy_run_manifest_payload_and_hash_are_byte_identical():
    """The optional usage_contract field must not move a legacy manifest.

    The golden payload and hash are the committed curated fixture's first
    record, read before the field existed; the empty default must keep the
    payload and hash byte-for-byte identical.
    """
    record = _first_curated_fixture_record()
    frozen_manifest = record["run"]["manifest"]
    assert record["run"]["manifest_hash"] == LEGACY_FIXTURE_MANIFEST_HASH

    manifest = RunManifest(**frozen_manifest)

    assert manifest.payload() == frozen_manifest
    assert "usage_contract" not in manifest.payload()
    assert manifest.compute_hash() == LEGACY_FIXTURE_MANIFEST_HASH


def test_new_manifest_usage_contract_field_is_saved_and_verified(session):
    _, RunRepository, _, _, _ = _repository_api()
    repo = RunRepository(session)
    legacy = _manifest("usage-contract")
    manifest = replace(legacy, usage_contract="tutor-chunk-sum-v1")
    assert manifest.compute_hash() != legacy.compute_hash()

    run = repo.create(
        id="run-usage-contract",
        experiment_id=manifest.experiment_id,
        task_case_id=manifest.task_case_id,
        task_case_version=manifest.task_case_version,
        variant_id=manifest.variant_id,
        run_profile=manifest.run_profile,
        manifest=manifest,
        manifest_hash=manifest.compute_hash(),
    )

    stored = repo.get_verified(run.id)
    assert stored.manifest_json["usage_contract"] == "tutor-chunk-sum-v1"
    assert stored.manifest_hash == manifest.compute_hash()


def test_legacy_manifest_persists_and_verifies_without_usage_contract(session):
    _, finished = _finish_run(session, "legacy-verify")
    _, RunRepository, _, _, _ = _repository_api()

    stored = RunRepository(session).get_verified(finished.id)
    assert "usage_contract" not in stored.manifest_json
    assert stored.manifest_hash == _manifest("legacy-verify").compute_hash()


# ---------------------------------------------------------------------------
# D5d batch A — row identity cross-checks on the verified read path.
# ---------------------------------------------------------------------------


def test_run_identity_column_tamper_is_rejected_without_manifest_change(session):
    """A structured identity column must match the frozen manifest on read.

    The manifest JSON and its hash are untouched, so only the new cross-check
    between the row column and the frozen manifest can reject the tamper.
    """
    ChecksumMismatch, RunRepository, _, _, _ = _repository_api()
    _, finished = _finish_run(session, "identity-column-tamper")
    repo = RunRepository(session)

    session.execute(
        text("UPDATE eval_runs SET variant_id = :variant WHERE id = :run_id"),
        {"variant": "tampered-variant", "run_id": finished.id},
    )
    session.commit()

    with pytest.raises(ChecksumMismatch):
        repo.get_verified(finished.id)


def test_run_identity_empty_column_is_rejected_not_passed_by_empty_manifest(session):
    """A missing/empty identity value must fail even when the other side is empty.

    Clearing a structured identity column cannot self-validate as ``"" == ""``.
    """
    ChecksumMismatch, RunRepository, _, _, _ = _repository_api()
    _, finished = _finish_run(session, "identity-empty-column")
    repo = RunRepository(session)

    session.execute(
        text("UPDATE eval_runs SET run_profile = '' WHERE id = :run_id"),
        {"run_id": finished.id},
    )
    session.commit()

    with pytest.raises(ChecksumMismatch):
        repo.get_verified(finished.id)


def test_score_set_scorer_identity_column_tamper_is_rejected(session):
    """A ScoreSet scorer column must stay consistent with its frozen snapshot."""
    ChecksumMismatch, _, ScoreSetRepository, _, _ = _repository_api()
    _, finished = _finish_run(session, "scorer-column-tamper")
    score_repo = ScoreSetRepository(session)
    score_set = score_repo.create(
        run_id=finished.id,
        scorer_id="hybrid",
        scorer_version="v1",
        artifact_input_hash=finished.artifact_hash,
    )

    session.execute(
        text(
            "UPDATE eval_score_sets SET scorer_version = :version "
            "WHERE id = :score_set_id"
        ),
        {"version": "v9", "score_set_id": score_set.id},
    )
    session.commit()

    with pytest.raises(ChecksumMismatch):
        score_repo.get_verified(score_set.id)


def test_create_rejects_empty_manifest_identity_before_insert(session):
    """A live write cannot create a row its own verified read would reject."""
    _, RunRepository, _, _, _ = _repository_api()
    repo = RunRepository(session)
    manifest = replace(_manifest("empty-identity"), experiment_id="")

    with pytest.raises(ValueError):
        repo.create(
            id="run-empty-identity",
            experiment_id="",
            task_case_id=manifest.task_case_id,
            task_case_version=manifest.task_case_version,
            variant_id=manifest.variant_id,
            run_profile=manifest.run_profile,
            manifest=manifest,
            manifest_hash=manifest.compute_hash(),
        )
    assert session.execute(text("SELECT COUNT(*) FROM eval_runs")).scalar_one() == 0


# ---------------------------------------------------------------------------
# D5d batch A — the raw historical insert shares the suite-import terminal rule.
# ---------------------------------------------------------------------------


def _historical_row(manifest: RunManifest, *, row_id: str, lifecycle: str, outcome):
    return models.EvalRun(
        id=row_id,
        experiment_id=manifest.experiment_id,
        suite_execution_id=None,
        task_case_id=manifest.task_case_id,
        task_case_version=manifest.task_case_version,
        variant_id=manifest.variant_id,
        run_profile=manifest.run_profile,
        lifecycle=lifecycle,
        outcome=outcome,
        operational_error_json=None,
        manifest_json=manifest.payload(),
        manifest_hash=manifest.compute_hash(),
        candidate_artifact_json=None,
        artifact_hash=None,
    )


def test_import_historical_rejects_nonterminal_lifecycle_without_insert(session):
    _, RunRepository, _, _, _ = _repository_api()
    repo = RunRepository(session)
    manifest = _manifest("historical-nonterminal")

    for lifecycle in ("queued", "running"):
        row = _historical_row(
            manifest, row_id=f"historical-{lifecycle}", lifecycle=lifecycle, outcome=None
        )
        with pytest.raises(ValueError):
            repo.import_historical(row=row)
    assert session.execute(text("SELECT COUNT(*) FROM eval_runs")).scalar_one() == 0


def test_import_historical_rejects_invalid_finished_outcome_without_insert(session):
    _, RunRepository, _, _, _ = _repository_api()
    repo = RunRepository(session)
    manifest = _manifest("historical-outcome")

    for outcome in ("exploded", None, ""):
        row = _historical_row(
            manifest,
            row_id="historical-invalid-outcome",
            lifecycle="finished",
            outcome=outcome,
        )
        with pytest.raises(ValueError):
            repo.import_historical(row=row)
    assert session.execute(text("SELECT COUNT(*) FROM eval_runs")).scalar_one() == 0


def test_import_historical_accepts_legal_failure_and_cancelled(session):
    _, RunRepository, _, _, _ = _repository_api()
    repo = RunRepository(session)

    failed = _historical_row(
        _manifest("historical-failed"),
        row_id="historical-failed",
        lifecycle="finished",
        outcome="system_failed",
    )
    stored_failed = repo.import_historical(row=failed)
    assert stored_failed.lifecycle == "finished"
    assert stored_failed.outcome == "system_failed"

    cancelled = _historical_row(
        _manifest("historical-cancelled"),
        row_id="historical-cancelled",
        lifecycle="cancelled",
        outcome=None,
    )
    stored_cancelled = repo.import_historical(row=cancelled)
    assert stored_cancelled.lifecycle == "cancelled"
    assert stored_cancelled.outcome is None

    assert session.execute(text("SELECT COUNT(*) FROM eval_runs")).scalar_one() == 2


# ---------------------------------------------------------------------------
# D5d batch B — execution-derived ScoreSet finalization.
# ---------------------------------------------------------------------------


def test_finalize_completed_without_executions_is_rejected_and_row_stays_running(session):
    ChecksumMismatch, _, ScoreSetRepository, _, _ = _repository_api()
    _, finished = _finish_run(session, "finalize-no-executions")
    score_set, _bundle = _score_set_with_document(session, finished)
    repo = ScoreSetRepository(session)

    with pytest.raises(ChecksumMismatch):
        repo.finalize_once(
            score_set.id,
            status="completed",
            quality_verdict="pass",
            aggregate_scores={"groundedness": 4},
            findings=[],
        )
    assert repo.get_verified(score_set.id).status == "running"


def test_finalize_with_wrong_verdict_or_scores_is_rejected_and_row_stays_running(session):
    ChecksumMismatch, _, ScoreSetRepository, _, _ = _repository_api()
    _, finished = _finish_run(session, "finalize-mismatch")
    score_set, bundle = _score_set_with_document(session, finished)
    _append_success_components(session, score_set, bundle)
    repo = ScoreSetRepository(session)

    # A passing-looking summary that disagrees with the stored executions.
    with pytest.raises(ChecksumMismatch):
        repo.finalize_once(
            score_set.id,
            status="completed",
            quality_verdict="fail",
            aggregate_scores={"groundedness": 4, "citation_entailment": 4, "coverage": 4},
            findings=[],
        )
    with pytest.raises(ChecksumMismatch):
        repo.finalize_once(
            score_set.id,
            status="completed",
            quality_verdict="pass",
            aggregate_scores={"groundedness": 5, "citation_entailment": 4, "coverage": 4},
            findings=[],
        )
    assert repo.get_verified(score_set.id).status == "running"


def test_finalize_rejects_out_of_range_scores_even_with_matching_executions(session):
    _, _, ScoreSetRepository, _, _ = _repository_api()
    _, finished = _finish_run(session, "finalize-out-of-range")
    score_set, bundle = _score_set_with_document(session, finished)
    _append_success_components(session, score_set, bundle)
    repo = ScoreSetRepository(session)

    for bad_value in (99, 0, True, 4.0):
        with pytest.raises(ValueError):
            repo.finalize_once(
                score_set.id,
                status="completed",
                quality_verdict="pass",
                aggregate_scores={"groundedness": bad_value},
                findings=[],
            )
    assert repo.get_verified(score_set.id).status == "running"


def test_finalize_persists_the_derived_values_read_back_from_db(session):
    _, _, ScoreSetRepository, _, _ = _repository_api()
    _, finished = _finish_run(session, "finalize-derived")
    score_set, bundle = _score_set_with_document(session, finished)
    _append_success_components(session, score_set, bundle)
    repo = ScoreSetRepository(session)
    derived = repo.derive_from_executions(score_set.id)

    finalized = repo.finalize_once(
        score_set.id,
        status=derived.status,
        quality_verdict=derived.verdict,
        aggregate_scores=derived.aggregate_scores,
        findings=derived.findings,
        error_code=derived.error_code,
        sanitized_message=derived.error_message,
    )
    assert finalized.status == "completed"
    assert finalized.quality_verdict == "pass"
    reread = repo.get_verified(score_set.id)
    assert reread.aggregate_scores_json == dict(derived.aggregate_scores)


def test_derive_from_executions_reflects_persisted_rows_not_memory(session):
    _, _, ScoreSetRepository, ExecutionRepository, _ = _repository_api()
    _, finished = _finish_run(session, "derive-from-db")
    score_set, bundle = _score_set_with_document(session, finished)
    _append_success_components(session, score_set, bundle)
    repo = ScoreSetRepository(session)
    assert repo.derive_from_executions(score_set.id).verdict == "pass"

    llm_id = next(
        row.id
        for row in ExecutionRepository(session).list_verified(score_set.id)
        if row.scorer_id == "grounded-quality-rubric"
    )
    session.execute(
        text(
            "UPDATE eval_scorer_executions SET output_json = :payload WHERE id = :id"
        ),
        {
            "payload": json.dumps(
                {
                    "result": {
                        "groundedness": 2,
                        "citation_entailment": 2,
                        "coverage": 2,
                        "reasoning": "tampered",
                        "findings": [],
                    },
                    "findings": [],
                }
            ),
            "id": llm_id,
        },
    )
    session.commit()

    derived = repo.derive_from_executions(score_set.id)
    assert derived.verdict == "fail"
    assert derived.aggregate_scores["groundedness"] == 2


def test_append_rejects_a_component_not_in_the_frozen_snapshot(session):
    _, _, _, ExecutionRepository, _ = _repository_api()
    _, finished = _finish_run(session, "append-unknown-component")
    score_set, _bundle = _score_set_with_document(session, finished)

    with pytest.raises(ValueError):
        ExecutionRepository(session).append(
            score_set_id=score_set.id,
            scorer_id="not-a-frozen-component",
            scorer_version="v1",
            status="success",
            input_hash=finished.artifact_hash,
            output={"groundedness": 4},
        )
    assert ExecutionRepository(session).list_verified(score_set.id) == []


def test_execution_read_rejects_a_tampered_component_identity(session):
    ChecksumMismatch, _, _, ExecutionRepository, _ = _repository_api()
    _, finished = _finish_run(session, "execution-component-tamper")
    score_set, bundle = _score_set_with_document(session, finished)
    component = bundle.components[0]
    execution = ExecutionRepository(session).append(
        score_set_id=score_set.id,
        scorer_id=component.component_id,
        scorer_version=component.version,
        status="success",
        input_hash=finished.artifact_hash,
        output={"groundedness": 4},
    )

    session.execute(
        text(
            "UPDATE eval_scorer_executions SET scorer_version = :version WHERE id = :id"
        ),
        {"version": "v9", "id": execution.id},
    )
    session.commit()

    with pytest.raises(ChecksumMismatch):
        ExecutionRepository(session).get_verified(execution.id)


# ---------------------------------------------------------------------------
# D5d batch B follow-up -- terminal score-domain gate and nested task identity.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["failed", "partial"])
@pytest.mark.parametrize(
    "bad_scores",
    [
        [99],
        "bad",
        1,
        {"groundedness": 99},
        {"groundedness": True},
        {"groundedness": 4.0},
    ],
)
def test_finalize_rejects_invalid_scores_on_the_failure_fallback(session, status, bad_scores):
    _, _, ScoreSetRepository, _, _ = _repository_api()
    _, finished = _finish_run(session, "fallback-domain")
    score_set, _bundle = _score_set_with_document(session, finished)
    repo = ScoreSetRepository(session)

    with pytest.raises(ValueError):
        repo.finalize_once(
            score_set.id,
            status=status,
            quality_verdict="inconclusive",
            aggregate_scores=bad_scores,
            findings=[],
        )
    assert repo.get_verified(score_set.id).status == "running"


@pytest.mark.parametrize("empty_scores", [None, {}])
def test_finalize_keeps_the_scoreless_failure_fallback(session, empty_scores):
    _, _, ScoreSetRepository, _, _ = _repository_api()
    _, finished = _finish_run(session, "scoreless-fallback")
    score_set, _bundle = _score_set_with_document(session, finished)
    repo = ScoreSetRepository(session)

    finalized = repo.finalize_once(
        score_set.id,
        status="failed",
        quality_verdict="inconclusive",
        aggregate_scores=empty_scores,
        findings=[],
    )
    assert finalized.status == "failed"
    assert finalized.quality_verdict == "inconclusive"


def test_derive_and_finalize_reject_a_mismatched_nested_task_identity(session):
    ChecksumMismatch, RunRepository, ScoreSetRepository, _, _ = _repository_api()
    _, finished = _finish_run(session, "nested-task-identity")
    score_set, bundle = _score_set_with_document(session, finished)
    _append_success_components(session, score_set, bundle)
    repo = ScoreSetRepository(session)

    # Rewrite only the nested task snapshot identity and re-sign the manifest, so
    # the outer five-field identity check still passes and the nested check is
    # the one that must fire.
    run = RunRepository(session).get_verified(finished.id)
    manifest = dict(run.manifest_json)
    snapshot = dict(manifest["task_snapshot"])
    snapshot["id"] = "different-task"
    snapshot["version"] = "9"
    manifest["task_snapshot"] = snapshot
    run.manifest_json = manifest
    run.manifest_hash = canonical_hash(manifest)
    session.add(run)
    session.commit()

    with pytest.raises(ChecksumMismatch):
        repo.derive_from_executions(score_set.id)

    with pytest.raises(ChecksumMismatch):
        repo.finalize_once(
            score_set.id,
            status="completed",
            quality_verdict="pass",
            aggregate_scores={
                "groundedness": 4,
                "citation_entailment": 4,
                "coverage": 4,
            },
            findings=[],
        )
    assert repo.get_verified(score_set.id).status == "running"
