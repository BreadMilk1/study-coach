"""Contract tests for the bounded Learning Run persistence repositories.

These tests intentionally use a real SQLite engine and SQLAlchemy session.  The
repositories promise application-level append-only and checksum-verified
semantics; the tests therefore exercise both normal calls and direct SQL
tampering at the persistence boundary.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from uuid import UUID

import pytest
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import models
from app.db.models import Base
from app.eval.learning_run.contracts import CandidateArtifact, RunManifest, ScorerExecutionDraft


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
    return RunManifest(
        experiment_id="experiment-test",
        task_case_id=f"case-{label}",
        task_case_version="1",
        variant_id="tutor-v2",
        run_profile="evaluation",
        task_snapshot={"id": f"case-{label}", "version": "1", "question": "question"},
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

    first = score_repo.create(
        run_id=finished.id,
        scorer_id="hybrid",
        scorer_version="v1",
        artifact_input_hash=artifact_hash,
    )
    second = score_repo.create(
        run_id=finished.id,
        scorer_id="hybrid",
        scorer_version="v1",
        artifact_input_hash=artifact_hash,
    )
    assert first.id != second.id
    assert first.status == second.status == "pending"

    running = score_repo.claim_running(first.id)
    assert running.status == "running"
    completed = score_repo.finalize_once(
        first.id,
        status="completed",
        quality_verdict="pass",
        aggregate_scores={"groundedness": 4},
        findings=[],
    )
    assert completed.status == "completed"
    assert completed.aggregate_scores_json == {"groundedness": 4}
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
    score_set = score_repo.create(
        run_id=finished.id,
        scorer_id="hybrid",
        scorer_version="v1",
        artifact_input_hash=finished.artifact_hash,
    )

    with pytest.raises(Conflict):
        executions.append(
            score_set_id=score_set.id,
            scorer_id="deterministic-citations",
            scorer_version="v1",
            status="success",
            input_hash=finished.artifact_hash,
            output={"citation_entailment": 4},
        )
    assert executions.list_verified(score_set.id) == []

    score_repo.claim_running(score_set.id)
    executions.append(
        score_set_id=score_set.id,
        scorer_id="deterministic-citations",
        scorer_version="v1",
        status="success",
        input_hash=finished.artifact_hash,
        output={"citation_entailment": 4},
    )
    score_repo.finalize_once(
        score_set.id,
        status="completed",
        quality_verdict="pass",
        aggregate_scores={"citation_entailment": 4},
        findings=[],
    )
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
    score_set = score_repo.create(
        run_id=finished.id,
        scorer_id="hybrid",
        scorer_version="v1",
        artifact_input_hash=finished.artifact_hash,
    )
    score_repo.claim_running(score_set.id)
    executions = ExecutionRepository(session)
    execution = executions.append(
        score_set_id=score_set.id,
        scorer_id="deterministic-citations",
        scorer_version="v1",
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
            scorer_id="deterministic-citations",
            scorer_version="v1",
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
    score_set = score_repo.create(
        run_id=finished.id,
        scorer_id="hybrid",
        scorer_version="v1",
        artifact_input_hash=finished.artifact_hash,
    )
    score_repo.claim_running(score_set.id)
    executions = ExecutionRepository(session)

    with pytest.raises((TypeError, ValueError)):
        executions.append(
            score_set_id=score_set.id,
            scorer_id="scalar-usage-scorer",
            scorer_version="v1",
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
    score_repo = ScoreSetRepository(session)
    score_set = score_repo.create(
        run_id=finished.id,
        scorer_id="hybrid",
        scorer_version="v1",
        artifact_input_hash=finished.artifact_hash,
    )
    score_repo.claim_running(score_set.id)
    execution = ExecutionRepository(session).append(
        score_set_id=score_set.id,
        scorer_id="mapping-usage-scorer",
        scorer_version="v1",
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
        score_set = score_repo_a.create(
            run_id=finished.id,
            scorer_id="hybrid",
            scorer_version="v1",
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
                scorer_id="race-scorer",
                scorer_version="v1",
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
    score_set = score_repo.create(
        run_id=finished.id,
        scorer_id="hybrid",
        scorer_version="v1",
        artifact_input_hash=finished.artifact_hash,
    )
    score_repo.claim_running(score_set.id)
    execution = ExecutionRepository(session).append(
        score_set_id=score_set.id,
        scorer_id="uuid-scorer",
        scorer_version="v1",
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
    score_repo = ScoreSetRepository(session)
    score_set = score_repo.create(
        run_id=finished.id,
        scorer_id="hybrid",
        scorer_version="v1",
        artifact_input_hash=finished.artifact_hash,
    )
    score_repo.claim_running(score_set.id)
    execution = ExecutionRepository(session).append(
        score_set_id=score_set.id,
        scorer_id="deterministic-citations",
        scorer_version="v1",
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
    score_repo = ScoreSetRepository(session)
    score_set = score_repo.create(
        run_id=finished.id,
        scorer_id="hybrid",
        scorer_version="v1",
        artifact_input_hash=finished.artifact_hash,
    )
    score_repo.claim_running(score_set.id)
    executions = ExecutionRepository(session)
    success = executions.append_draft(
        score_set.id,
        ScorerExecutionDraft(
            component_id="component-a",
            component_version="v1",
            scorer_id="component-a",
            scorer_version="v1",
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
            component_id="component-b",
            component_version="v1",
            scorer_id="component-b",
            scorer_version="v1",
            status="skipped",
            input_hash=finished.artifact_hash,
            output={"applicable": False},
        ),
    )
    assert skipped.output_json == {"result": {"applicable": False}, "findings": []}
    with pytest.raises(TypeError):
        executions.append_draft(score_set.id, {"status": "success"})
