"""Task 6 DB-backed single-flight race tests."""

from __future__ import annotations

import threading
import asyncio
from datetime import datetime, timedelta
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models import Base, Document, EvalRun, EvalScoreSet, EvalScorerExecution, User
from app.db.repositories import DataLifecycleRepository
from app.eval.learning_run.contracts import (
    CandidateArtifact,
    RunManifest,
    ScorerExecutionDraft,
    canonical_hash,
)
from app.agent.tutor_attempt import TutorCandidate
from app.eval.learning_run.registry import TaskRegistry
from app.eval.learning_run.repositories import (
    ChecksumMismatchError,
    EvalExecutionClaimRepository,
    EvalExecutionControlRepository,
    EvalRunRepository,
    EvalScoreSetRepository,
    EvalScorerExecutionRepository,
)
from app.eval.learning_run.service import EvalModelConnection, RunService


def _manifest(label: str) -> RunManifest:
    return RunManifest(
        experiment_id="experiment-test",
        task_case_id=f"case-{label}",
        task_case_version="1",
        variant_id="tutor-v3",
        run_profile="evaluation",
        task_snapshot={"id": f"case-{label}", "version": "1", "question": "question"},
        prompt_text="frozen prompt",
        corpus_snapshot={"snapshot_id": "corpus-test", "version": "1", "aggregate_hash": "c" * 64},
        scorer_snapshot={"scorer_id": "hybrid-v1", "version": "v1"},
        connection_fingerprint="d" * 64,
        corpus_snapshot_id="corpus-test",
        corpus_snapshot_version="1",
        corpus_snapshot_hash="c" * 64,
        prompt_version="tutor-v3",
        prompt_hash="p" * 64,
        scorer_bundle_version="hybrid-v1",
        scorer_bundle_hash="s" * 64,
        provider="test-provider",
        model="test-model",
        model_parameters={"temperature": 0, "top_p": 1},
        retrieval_config={"version": "test-retrieval-v1", "top_k": 5},
        reranker_config={"version": "test-reranker-v1"},
        chunking_config_version="chunking-v1",
        embedding_config_version="embedding-v1",
        budget={"total_seconds": 90},
        runtime_judge=False,
        runner_version="runner-v1",
        schema_version="learning-run-test-v1",
        code_revision="test-revision",
    )


def _claim_api():
    try:
        from app.eval.learning_run.repositories import (
            EvalExecutionClaimRepository,
            EvaluationBusyError,
        )
    except ImportError as exc:  # intentional Task 6 RED guard
        pytest.fail(f"atomic eval claim repository is missing: {exc}", pytrace=False)
    return EvalExecutionClaimRepository, EvaluationBusyError


def _unavailable_api():
    try:
        from app.eval.learning_run.repositories import EvaluationUnavailableError
    except ImportError as exc:  # intentional Task 6 RED guard
        pytest.fail(f"evaluation unavailable error is missing: {exc}", pytrace=False)
    return EvaluationUnavailableError


def _task7_control_api():
    try:
        from app.eval.learning_run.repositories import (
            EvalExecutionControlRepository,
            EvaluationBusyError,
            RepositoryConflictError,
        )
    except ImportError as exc:  # intentional Task 7 RED guard
        pytest.fail(f"Task 7 control repository is missing: {exc}", pytrace=False)
    return EvalExecutionControlRepository, EvaluationBusyError, RepositoryConflictError


def _engine(path: Path, *, timeout: float = 30):
    engine = create_engine(
        f"sqlite:///{path}",
        connect_args={"check_same_thread": False, "timeout": timeout},
    )

    @event.listens_for(engine, "connect")
    def _foreign_keys(dbapi_connection, _connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    return engine


def _seed_frozen_run(engine, *, run_id: str = "parent-run", finished: bool = False):
    manifest = _manifest("parent")
    artifact = CandidateArtifact(
        answer="frozen answer",
        citations=(),
        exact_evidence=(),
        formatted_context="evidence",
        usage="unavailable",
        trace=(),
        budget={"total_seconds": 90},
    )
    with Session(engine) as session:
        repo = EvalRunRepository(session)
        run = repo.create(
            experiment_id=manifest.experiment_id,
            task_case_id=manifest.task_case_id,
            task_case_version=manifest.task_case_version,
            variant_id=manifest.variant_id,
            run_profile=manifest.run_profile,
            manifest=manifest,
            manifest_hash=manifest.compute_hash(),
            id=run_id,
        )
        repo.claim_running(run.id)
        run = repo.freeze_candidate(run.id, artifact, artifact_hash=artifact.compute_hash())
        if finished:
            run = repo.finalize_success(run.id)
        return run


def _seed_stale_active_rows(engine, *, run_id: str = "stale-run", score_set_id: str = "stale-score"):
    stale_at = datetime(2026, 1, 1)
    manifest = _manifest("stale")
    artifact = CandidateArtifact(
        answer="stale answer",
        citations=(),
        exact_evidence=(),
        formatted_context="evidence",
        usage="unavailable",
        trace=(),
        budget={"total_seconds": 90},
    )
    with Session(engine) as session:
        run_repo = EvalRunRepository(session)
        run = run_repo.create(
            experiment_id=manifest.experiment_id,
            task_case_id=manifest.task_case_id,
            task_case_version=manifest.task_case_version,
            variant_id=manifest.variant_id,
            run_profile=manifest.run_profile,
            manifest=manifest,
            manifest_hash=manifest.compute_hash(),
            id=run_id,
            created_at=stale_at,
        )
        run_repo.claim_running(run.id)
        run_repo.freeze_candidate(run.id, artifact, artifact_hash=artifact.compute_hash())
        score_repo = EvalScoreSetRepository(session)
        score = score_repo.create(
            run_id=run.id,
            scorer_id="hybrid-v1",
            scorer_version="v1",
            artifact_input_hash=artifact.compute_hash(),
            id=score_set_id,
            created_at=stale_at,
        )
        score_repo.claim_running(score.id)
        session.execute(
            EvalRun.__table__.update()
            .where(EvalRun.id == run_id)
            .values(started_at=stale_at)
        )
        session.execute(
            EvalScoreSet.__table__.update()
            .where(EvalScoreSet.id == score_set_id)
            .values(started_at=stale_at)
        )
        session.commit()


def test_two_independent_connections_allow_exactly_one_run_claim(tmp_path):
    EvalExecutionClaimRepository, EvaluationBusyError = _claim_api()
    path = tmp_path / "single-flight.db"
    engine = _engine(path)
    barrier = threading.Barrier(2)
    results = []

    def claim(label):
        with Session(engine) as session:
            repo = EvalExecutionClaimRepository(session)
            try:
                barrier.wait()
                row = repo.claim_run(manifest=_manifest(label))
                results.append(("claimed", row.id))
            except EvaluationBusyError as exc:
                results.append(("busy", exc.active_entity_id, exc.active_kind))

    threads = [threading.Thread(target=claim, args=(label,)) for label in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()

    assert sorted(result[0] for result in results) == ["busy", "claimed"]
    with Session(engine) as session:
        active = session.execute(
            select(EvalRun).where(EvalRun.lifecycle.in_(("queued", "running")))
        ).scalars().all()
        assert len(active) == 1


def test_active_score_set_blocks_new_run_and_terminal_rows_do_not(tmp_path):
    EvalExecutionClaimRepository, EvaluationBusyError = _claim_api()
    path = tmp_path / "score-set-flight.db"
    engine = _engine(path)
    with Session(engine) as session:
        repo = EvalExecutionClaimRepository(session)
        # A legitimate terminal parent/score-set row is needed for SQLite FK
        # enforcement; an active ScoreSet is always a child of an EvalRun.
        session.add(
            EvalRun(
                id="run-existing",
                experiment_id="experiment-test",
                task_case_id="case-existing",
                task_case_version="1",
                variant_id="tutor-v3",
                run_profile="evaluation",
                lifecycle="finished",
                outcome="success",
                manifest_json={"id": "existing"},
                manifest_hash=canonical_hash({"id": "existing"}),
                candidate_artifact_json={"answer": "existing"},
                artifact_hash=canonical_hash({"answer": "existing"}),
            )
        )
        session.commit()
        from app.eval.learning_run.repositories import EvalScoreSetRepository

        score_set = EvalScoreSetRepository(session).create(
            run_id="run-existing",
            scorer_id="hybrid-v1",
            scorer_version="v1",
            artifact_input_hash=canonical_hash({"answer": "existing"}),
            id="score-set-active",
        )
        score_set = EvalScoreSetRepository(session).claim_running(score_set.id)
        with pytest.raises(EvaluationBusyError) as caught:
            repo.claim_run(manifest=_manifest("blocked"))
        assert caught.value.active_entity_id == "score-set-active"
        assert caught.value.active_kind == "score_set"

        session.execute(
            EvalScoreSet.__table__.update()
            .where(EvalScoreSet.id == "score-set-active")
            .values(status="completed", quality_verdict="pass")
        )
        session.commit()
        claimed = repo.claim_run(manifest=_manifest("after-terminal"))
        assert claimed.id


def test_failed_claim_transaction_rolls_back_and_next_claim_can_succeed(tmp_path):
    EvalExecutionClaimRepository, _ = _claim_api()
    engine = _engine(tmp_path / "failed-claim-recovery.db")
    with Session(engine) as session:
        repo = EvalExecutionClaimRepository(session)
        with pytest.raises((ValueError, IntegrityError, TypeError)):
            repo.claim_run(manifest=None)
        recovered = repo.claim_run(manifest=_manifest("recovered"))
        assert recovered.id


def test_sqlite_write_lock_is_unavailable_not_busy_and_same_session_recovers(tmp_path):
    EvalExecutionClaimRepository, EvaluationBusyError = _claim_api()
    EvaluationUnavailableError = _unavailable_api()
    path = tmp_path / "write-lock.db"
    claimant_engine = _engine(path, timeout=0)
    locker_engine = _engine(path, timeout=0)

    with Session(claimant_engine) as session:
        repo = EvalExecutionClaimRepository(session)
        with locker_engine.connect() as lock_connection:
            lock_connection.exec_driver_sql("BEGIN IMMEDIATE")
            lock_held = threading.Event()
            lock_held.set()
            assert lock_held.is_set()
            with pytest.raises(EvaluationUnavailableError) as caught:
                repo.claim_run(manifest=_manifest("locked"))
            assert not isinstance(caught.value, EvaluationBusyError)
            lock_connection.rollback()
        claimed = repo.claim_run(manifest=_manifest("after-lock"))
        assert claimed.lifecycle == "running"


def test_claim_score_set_checksum_failure_rolls_back_and_same_session_recovers(tmp_path):
    path = tmp_path / "score-claim-rollback.db"
    claimant_engine = _engine(path, timeout=0)
    locker_engine = _engine(path, timeout=0)
    parent = _seed_frozen_run(claimant_engine, run_id="score-claim-parent", finished=True)
    artifact_hash = _artifact_hash(claimant_engine, parent.id)
    manifest = _manifest("parent")

    with Session(claimant_engine) as session:
        session.execute(
            EvalRun.__table__.update()
            .where(EvalRun.id == parent.id)
            .values(manifest_json={"tampered": True})
        )
        session.commit()
        repository = EvalExecutionClaimRepository(session)
        with pytest.raises(ChecksumMismatchError):
            repository.claim_score_set(
                run_id=parent.id,
                artifact_input_hash=artifact_hash,
                scorer_id="hybrid-v1",
                scorer_version="v1",
            )

        # A failed typed claim must release BEGIN IMMEDIATE before another
        # connection can acquire the write gate.
        with locker_engine.connect() as connection:
            connection.exec_driver_sql("BEGIN IMMEDIATE")
            connection.exec_driver_sql(
                "UPDATE eval_runs SET outcome = outcome WHERE id = 'score-claim-parent'"
            )
            connection.rollback()

        session.execute(
            EvalRun.__table__.update()
            .where(EvalRun.id == parent.id)
            .values(manifest_json=manifest.payload())
        )
        session.commit()
        recovered = repository.claim_score_set(
            run_id=parent.id,
            artifact_input_hash=artifact_hash,
            scorer_id="hybrid-v1",
            scorer_version="v1",
            id="score-claim-recovered",
        )
        assert recovered.status == "running"


@pytest.mark.parametrize("tampered", ["run", "score_set"])
def test_reconcile_validates_all_cutoff_rows_before_any_bulk_mutation(tmp_path, tampered):
    from app.eval.learning_run.repositories import ChecksumMismatchError

    path = tmp_path / f"reconcile-tampered-{tampered}.db"
    engine = _engine(path, timeout=0)
    locker_engine = _engine(path, timeout=0)
    _seed_stale_active_rows(engine)
    with Session(engine) as session:
        if tampered == "run":
            session.execute(
                EvalRun.__table__.update()
                .where(EvalRun.id == "stale-run")
                .values(manifest_json={"tampered": True})
            )
        else:
            session.execute(
                EvalScoreSet.__table__.update()
                .where(EvalScoreSet.id == "stale-score")
                .values(artifact_input_hash="0" * 64)
            )
        session.commit()

        before_run = session.execute(
            select(
                EvalRun.lifecycle,
                EvalRun.outcome,
                EvalRun.finished_at,
                EvalRun.operational_error_json,
            ).where(EvalRun.id == "stale-run")
        ).one()
        before_score = session.execute(
            select(
                EvalScoreSet.status,
                EvalScoreSet.finished_at,
                EvalScoreSet.operational_error_code,
            ).where(EvalScoreSet.id == "stale-score")
        ).one()
        with pytest.raises(ChecksumMismatchError):
            EvalExecutionControlRepository(session).reconcile(
                started_before=datetime(2026, 1, 2)
            )

        after_run = session.execute(
            select(
                EvalRun.lifecycle,
                EvalRun.outcome,
                EvalRun.finished_at,
                EvalRun.operational_error_json,
            ).where(EvalRun.id == "stale-run")
        ).one()
        after_score = session.execute(
            select(
                EvalScoreSet.status,
                EvalScoreSet.finished_at,
                EvalScoreSet.operational_error_code,
            ).where(EvalScoreSet.id == "stale-score")
        ).one()
        assert after_run == before_run
        assert after_score == before_score

        # The reconciliation failure must roll back its write gate so startup
        # can still make progress on the same database.
        with locker_engine.connect() as connection:
            connection.exec_driver_sql("BEGIN IMMEDIATE")
            connection.rollback()


@pytest.mark.asyncio
async def test_projection_read_failure_after_success_cas_keeps_durable_success_terminal():
    from app.eval.learning_run.contracts import ScoreSetResultDraft

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    registry = TaskRegistry.load_default()
    candidate = TutorCandidate(
        answer="answer",
        citations=[],
        evidence=[],
        formatted_context="",
        usage="unavailable",
        trace=[],
    )

    class Runner:
        async def run(self, **_kwargs):
            return candidate

    class SuccessfulScoring:
        async def score(self, *, candidate, on_execution, scorer_bundle, **_kwargs):
            drafts = tuple(
                ScorerExecutionDraft(
                    component_id=component.component_id,
                    component_version=component.version,
                    scorer_id=component.component_id,
                    scorer_version=component.version,
                    status="failed",
                    input_hash=candidate.compute_hash(),
                    error_code="scorer_parse_error",
                    error_message="scorer output could not be parsed",
                )
                for component in scorer_bundle.components
            )
            for draft in drafts:
                on_execution(draft)
            return ScoreSetResultDraft(
                status="partial",
                verdict="inconclusive",
                executions=drafts,
                input_hash=candidate.compute_hash(),
            )

    with Session(engine) as session:
        service = RunService(
            registry=registry,
            tutor_runner=Runner(),
            runs=EvalRunRepository(session),
            score_sets=EvalScoreSetRepository(session),
            scorer_executions=EvalScorerExecutionRepository(session),
            scoring_service_factory=lambda *_args, **_kwargs: SuccessfulScoring(),
        )
        controls = registry.experiment.variants["tutor-v3"]
        scorer_config = dict(registry.scorer.model_config)
        connection = EvalModelConnection(
            tutor_provider=controls["provider"],
            tutor_model=controls["model"],
            tutor_parameters=dict(controls["parameters"]),
            tutor_llm=object(),
            scorer_provider=scorer_config["provider"],
            scorer_model=scorer_config["model"],
            scorer_parameters={
                key: value
                for key, value in scorer_config.items()
                if key not in {"provider", "model"}
            },
            scorer_llm=object(),
            connection_fingerprint="a" * 64,
        )
        events: list[dict] = []
        service.scorer_executions.list_verified = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("projection unavailable")
        )
        result = await service.run(
            experiment_id=registry.experiment.experiment_id,
            task_case_id="tgqa-001",
            variant_id="tutor-v3",
            run_profile="evaluation",
            connection=connection,
            events=events,
        )
        assert result.run.lifecycle == "finished"
        assert result.run.outcome == "success"
        assert [
            event for event in events
            if event.get("type") == "run_finished"
        ] == [{
            "type": "run_finished",
            "run_id": result.run.id,
            "lifecycle": "finished",
            "outcome": "success",
        }]
        persisted = EvalRunRepository(session).get_verified(result.run.id)
        assert persisted.lifecycle == "finished"
        assert persisted.outcome == "success"


@pytest.mark.asyncio
async def test_scoring_timeout_emits_sanitized_failed_event_for_each_missing_component():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    registry = TaskRegistry.load_default()
    candidate = TutorCandidate(
        answer="answer",
        citations=[],
        evidence=[],
        formatted_context="",
        usage="unavailable",
        trace=[],
    )

    class Runner:
        async def run(self, **_kwargs):
            return candidate

    class BlockingScoring:
        async def score(self, **_kwargs):
            await asyncio.Event().wait()

    events = []
    budget = dict(registry.experiment.budget)
    budget["hybrid_scoring_seconds"] = 0.001
    original_resolve = registry.resolve_run
    registry.resolve_run = lambda **kwargs: replace(
        original_resolve(**kwargs), budget=budget
    )
    try:
        with Session(engine) as session:
            service = RunService(
                registry=registry,
                tutor_runner=Runner(),
                runs=EvalRunRepository(session),
                score_sets=EvalScoreSetRepository(session),
                scorer_executions=EvalScorerExecutionRepository(session),
                scoring_service_factory=lambda *_args, **_kwargs: BlockingScoring(),
            )
            controls = registry.experiment.variants["tutor-v3"]
            scorer_config = dict(registry.scorer.model_config)
            connection = EvalModelConnection(
                tutor_provider=controls["provider"],
                tutor_model=controls["model"],
                tutor_parameters=dict(controls["parameters"]),
                tutor_llm=object(),
                scorer_provider=scorer_config["provider"],
                scorer_model=scorer_config["model"],
                scorer_parameters={
                    key: value
                    for key, value in scorer_config.items()
                    if key not in {"provider", "model"}
                },
                scorer_llm=object(),
                connection_fingerprint="a" * 64,
            )
            result = await service.run(
                experiment_id=registry.experiment.experiment_id,
                task_case_id="tgqa-001",
                variant_id="tutor-v3",
                run_profile="evaluation",
                connection=connection,
                events=events,
            )
            assert result.score_set is not None
            expected_failed = {
                component.component_id
                for component in registry.scorer.components
            }
            failed_events = {
                event["scorer_id"]
                for event in events
                if event.get("type") == "scorer_failed"
            }
            assert failed_events == expected_failed
            assert all(event.get("error_code") == "scorer_timeout" for event in events if event.get("type") == "scorer_failed")
    finally:
        registry.resolve_run = original_resolve


@pytest.mark.parametrize("winner", ["cancel", "finalize"])
def test_cancel_finalize_race_has_one_terminal_winner_and_preserves_candidate(tmp_path, winner):
    EvalExecutionControlRepository, _, RepositoryConflictError = _task7_control_api()
    engine = _engine(tmp_path / f"cancel-finalize-{winner}.db")
    run = _seed_frozen_run(engine, run_id="cancel-finalize-run")

    with Session(engine) as cancel_session, Session(engine) as finalize_session:
        control = EvalExecutionControlRepository(cancel_session)
        finalizer = EvalRunRepository(finalize_session)
        if winner == "cancel":
            cancelled = control.cancel_run(run.id)
            assert cancelled.lifecycle == "cancelled"
            with pytest.raises(RepositoryConflictError):
                finalizer.finalize_success(run.id)
        else:
            finished = finalizer.finalize_success(run.id)
            assert finished.lifecycle == "finished"
            cancelled = control.cancel_run(run.id)
            assert cancelled.lifecycle == "finished"

    with Session(engine) as session:
        persisted = EvalRunRepository(session).get_verified(run.id)
        assert persisted.lifecycle in {"finished", "cancelled"}
        assert persisted.outcome in {"success", None}
        assert persisted.candidate_artifact_json is not None
        assert sum(
            value == persisted.lifecycle
            for value in ("finished", "cancelled")
        ) == 1


@pytest.mark.parametrize("reconcile_first", [True, False])
def test_startup_reconciliation_cutoff_does_not_kill_new_run(tmp_path, reconcile_first):
    EvalExecutionControlRepository, _, _ = _task7_control_api()
    engine = _engine(tmp_path / f"reconcile-new-{reconcile_first}.db")
    _seed_stale_active_rows(engine)
    cutoff = datetime(2026, 1, 2)
    new_manifest = _manifest("new-after-restart")

    with Session(engine) as session:
        control = EvalExecutionControlRepository(session)
        if reconcile_first:
            result = control.reconcile(started_before=cutoff)
            assert result.runs_reconciled == 1
            assert result.score_sets_reconciled == 1
            new_run = EvalExecutionClaimRepository(session).claim_run(
                manifest=new_manifest,
                created_at=cutoff + timedelta(days=1),
            )
        else:
            # Model the concurrent writer having committed its fresh active
            # row before the reconciliation transaction acquires the gate.
            new_run = EvalRunRepository(session).create(
                experiment_id=new_manifest.experiment_id,
                task_case_id=new_manifest.task_case_id,
                task_case_version=new_manifest.task_case_version,
                variant_id=new_manifest.variant_id,
                run_profile=new_manifest.run_profile,
                manifest=new_manifest,
                manifest_hash=new_manifest.compute_hash(),
                created_at=cutoff + timedelta(days=1),
            )
            new_run = EvalRunRepository(session).claim_running(new_run.id)
            new_run_id = new_run.id
            result = control.reconcile(started_before=cutoff)
        if reconcile_first:
            new_run_id = new_run.id
            assert result.runs_reconciled == 1
            assert result.score_sets_reconciled == 1

    with Session(engine) as session:
        stale = EvalRunRepository(session).get_verified("stale-run")
        fresh = EvalRunRepository(session).get_verified(new_run_id)
        score = EvalScoreSetRepository(session).get_verified("stale-score")
        assert stale.lifecycle == "finished"
        assert stale.outcome == "system_failed"
        assert stale.operational_error_json["code"] == "process_interrupted"
        assert score.status == "failed"
        assert score.quality_verdict == "inconclusive"
        assert score.operational_error_code == "process_interrupted"
        assert fresh.lifecycle == "running"


def test_reconciliation_lock_failure_is_not_silent(tmp_path):
    from app.eval.learning_run.repositories import EvaluationUnavailableError

    EvalExecutionControlRepository, _, _ = _task7_control_api()
    path = tmp_path / "reconcile-lock.db"
    engine = _engine(path, timeout=0)
    locker = _engine(path, timeout=0)
    with locker.connect() as connection:
        connection.exec_driver_sql("BEGIN IMMEDIATE")
        with Session(engine) as session:
            with pytest.raises(EvaluationUnavailableError):
                EvalExecutionControlRepository(session).reconcile(
                    started_before=datetime(2026, 1, 2)
                )
        connection.rollback()


def test_cancel_tampered_active_run_rolls_back_without_touching_terminal_fields(tmp_path):
    EvalExecutionControlRepository, _, _ = _task7_control_api()
    engine = _engine(tmp_path / "cancel-tampered-run.db")
    run = _seed_frozen_run(engine, run_id="tampered-active-run")
    with Session(engine) as session:
        session.execute(
            EvalRun.__table__.update()
            .where(EvalRun.id == run.id)
            .values(manifest_json={"tampered": True})
        )
        session.commit()
        with pytest.raises(ChecksumMismatchError):
            EvalExecutionControlRepository(session).cancel_run(run.id)

    with Session(engine) as session:
        raw = session.execute(
            select(EvalRun).where(EvalRun.id == run.id)
        ).scalar_one()
        assert raw.lifecycle == "running"
        assert raw.outcome is None
        assert raw.finished_at is None
        assert raw.artifact_hash == run.artifact_hash


def test_cancel_tampered_active_score_set_rolls_back_run_and_score_set(tmp_path):
    EvalExecutionControlRepository, _, _ = _task7_control_api()
    engine = _engine(tmp_path / "cancel-tampered-score-set.db")
    run = _seed_frozen_run(engine, run_id="tampered-score-parent")
    with Session(engine) as session:
        score_repo = EvalScoreSetRepository(session)
        score = score_repo.create(
            run_id=run.id,
            scorer_id="hybrid-v1",
            scorer_version="v1",
            artifact_input_hash=run.artifact_hash,
            id="tampered-score-set",
        )
        score_repo.claim_running(score.id)
        session.execute(
            EvalScoreSet.__table__.update()
            .where(EvalScoreSet.id == score.id)
            .values(artifact_input_hash="0" * 64)
        )
        session.commit()
        with pytest.raises(ChecksumMismatchError):
            EvalExecutionControlRepository(session).cancel_run(run.id)

    with Session(engine) as session:
        raw_run = session.execute(
            select(EvalRun).where(EvalRun.id == run.id)
        ).scalar_one()
        raw_score = session.execute(
            select(EvalScoreSet).where(EvalScoreSet.id == "tampered-score-set")
        ).scalar_one()
        assert raw_run.lifecycle == "running"
        assert raw_run.finished_at is None
        assert raw_score.status == "running"
        assert raw_score.finished_at is None


def test_finished_run_cancel_does_not_touch_active_historical_rescore(tmp_path):
    EvalExecutionControlRepository, _, _ = _task7_control_api()
    engine = _engine(tmp_path / "finished-rescore-noop.db")
    run = _seed_frozen_run(engine, run_id="finished-parent", finished=True)
    with Session(engine) as session:
        score = EvalExecutionClaimRepository(session).claim_score_set(
            run_id=run.id,
            artifact_input_hash=run.artifact_hash,
            scorer_id="hybrid-v1",
            scorer_version="v1",
            id="active-historical-rescore",
        )
        before_finished = score.finished_at
        cancelled = EvalExecutionControlRepository(session).cancel_run(run.id)
        assert cancelled.lifecycle == "finished"
        assert cancelled.outcome == "success"
        after = EvalScoreSetRepository(session).get_verified(score.id)
        assert after.status == "running"
        assert after.finished_at == before_finished


def test_reconciliation_lock_failure_reports_typed_unavailable(tmp_path):
    from app.eval.learning_run.repositories import EvaluationUnavailableError

    EvalExecutionControlRepository, _, _ = _task7_control_api()
    path = tmp_path / "reconcile-typed-lock.db"
    engine = _engine(path, timeout=0)
    locker = _engine(path, timeout=0)
    with locker.connect() as connection:
        connection.exec_driver_sql("BEGIN IMMEDIATE")
        with Session(engine) as session:
            with pytest.raises(EvaluationUnavailableError):
                EvalExecutionControlRepository(session).reconcile(
                    started_before=datetime(2026, 1, 2)
                )
        connection.rollback()


@pytest.mark.asyncio
async def test_service_task_cancellation_waits_for_runner_release_and_propagates_cancel(tmp_path):
    from app.eval.learning_run.service import EvalModelConnection, RunService

    engine = _engine(tmp_path / "service-cancel-propagation.db")
    registry = TaskRegistry.load_default()
    released = asyncio.Event()
    entered = asyncio.Event()

    class Runner:
        async def run(self, **_kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                released.set()

    controls = registry.experiment.variants["tutor-v3"]
    scorer_config = dict(registry.scorer.model_config)
    connection = EvalModelConnection(
        tutor_provider=controls["provider"],
        tutor_model=controls["model"],
        tutor_parameters=dict(controls["parameters"]),
        tutor_llm=object(),
        scorer_provider=scorer_config["provider"],
        scorer_model=scorer_config["model"],
        scorer_parameters={
            key: value for key, value in scorer_config.items() if key not in {"provider", "model"}
        },
        scorer_llm=object(),
        connection_fingerprint="a" * 64,
    )
    with Session(engine) as session:
        service = RunService(
            registry=registry,
            tutor_runner=Runner(),
            runs=EvalRunRepository(session),
            score_sets=EvalScoreSetRepository(session),
            scorer_executions=EvalScorerExecutionRepository(session),
        )
        prepared = service.prepare(
            experiment_id=registry.experiment.experiment_id,
            task_case_id="tgqa-001",
            variant_id="tutor-v3",
            run_profile="evaluation",
            connection=connection,
        )
        run_id = prepared.run.id
        task = asyncio.create_task(service.execute_prepared(prepared, events=[]))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert released.is_set()

    with Session(engine) as session:
        persisted = EvalRunRepository(session).get_verified(run_id)
        assert persisted.lifecycle == "cancelled"
        assert persisted.outcome is None


@pytest.mark.asyncio
async def test_parent_cancel_closes_runner_gate_before_bounded_handoff(tmp_path):
    """A cancellation-resistant Tutor child cannot outlive the attached task."""

    from app.eval.learning_run.corpus import CorpusMaterializerBusyError, CorpusMaterializerController

    engine = _engine(tmp_path / "bounded-runner-cancel.db")
    registry = TaskRegistry.load_default()
    entered = asyncio.Event()
    caught_cancel = asyncio.Event()
    release = asyncio.Event()
    cleanup_done = threading.Event()

    class Resource:
        def close(self):
            cleanup_done.set()

    class Loader:
        def load(self, *, snapshot, stop=None, deadline=None):
            del snapshot, stop, deadline
            return Resource()

    controller = CorpusMaterializerController(Loader())

    class CancellationResistantRunner:
        def __init__(self):
            self.late_events: list[dict] = []
            self.lease = None

        async def run(self, *, definition, events, **_kwargs):
            self.lease = await controller.acquire(
                snapshot=definition.corpus,
                deadline=asyncio.get_running_loop().time() + 10,
            )
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                caught_cancel.set()
                try:
                    events({"type": "token", "text": "late"})
                except BaseException as exc:
                    self.late_events.append({"error": type(exc).__name__})
                await release.wait()
                raise
            finally:
                self.lease.release()

    runner = CancellationResistantRunner()
    events: list[dict] = []
    controls = registry.experiment.variants["tutor-v3"]
    scorer_config = dict(registry.scorer.model_config)
    connection = EvalModelConnection(
        tutor_provider=controls["provider"],
        tutor_model=controls["model"],
        tutor_parameters=dict(controls["parameters"]),
        tutor_llm=object(),
        scorer_provider=scorer_config["provider"],
        scorer_model=scorer_config["model"],
        scorer_parameters={
            key: value for key, value in scorer_config.items() if key not in {"provider", "model"}
        },
        scorer_llm=object(),
        connection_fingerprint="a" * 64,
    )
    try:
        with Session(engine) as session:
            service = RunService(
                registry=registry,
                tutor_runner=runner,
                runs=EvalRunRepository(session),
                score_sets=EvalScoreSetRepository(session),
                scorer_executions=EvalScorerExecutionRepository(session),
            )
            prepared = service.prepare(
                experiment_id=registry.experiment.experiment_id,
                task_case_id="tgqa-001",
                variant_id="tutor-v3",
                run_profile="evaluation",
                connection=connection,
            )
            task = asyncio.create_task(service.execute_prepared(prepared, events=events))
            await entered.wait()
            task.cancel()
            await caught_cancel.wait()
            try:
                await controller.acquire(
                    snapshot=prepared.definition.corpus,
                    deadline=asyncio.get_running_loop().time() + 10,
                )
            except CorpusMaterializerBusyError:
                pass
            else:
                pytest.fail("materializer admitted a new lease before cancellation release")
            done, _ = await asyncio.wait({task}, timeout=0.1)
            assert task in done
            assert prepared.cancellation.cancelled
            assert runner.late_events == [{"error": "_CooperativeCancellation"}]
            assert not any(event.get("text") == "late" for event in events)
            release.set()
            try:
                await task
            except asyncio.CancelledError:
                pass
            assert await asyncio.to_thread(cleanup_done.wait, 1)
    finally:
        release.set()
        controller.shutdown(wait=False)


@pytest.mark.asyncio
async def test_attached_scoring_cancel_closes_callback_and_rejects_late_append(tmp_path):
    from app.eval.learning_run.contracts import ScorerExecutionDraft
    from app.eval.learning_run.service import EvalModelConnection, RunService

    engine = _engine(tmp_path / "service-scoring-cancel.db")
    registry = TaskRegistry.load_default()
    entered = asyncio.Event()
    candidate = TutorCandidate(
        answer="answer",
        citations=[],
        evidence=[],
        formatted_context="",
        usage="unavailable",
        trace=[],
    )

    class Runner:
        async def run(self, **_kwargs):
            return candidate

    class BlockingScoring:
        def __init__(self):
            self.callback = None

        async def score(self, *, candidate, on_execution, **_kwargs):
            self.callback = on_execution
            on_execution(
                ScorerExecutionDraft(
                    component_id="retrieval-integrity",
                    component_version="v1",
                    scorer_id="retrieval-integrity",
                    scorer_version="v1",
                    status="success",
                    input_hash=candidate.compute_hash(),
                    output={"evidence_count": 0},
                    findings=(),
                )
            )
            entered.set()
            await asyncio.Event().wait()

    scoring = BlockingScoring()
    controls = registry.experiment.variants["tutor-v3"]
    scorer_config = dict(registry.scorer.model_config)
    connection = EvalModelConnection(
        tutor_provider=controls["provider"],
        tutor_model=controls["model"],
        tutor_parameters=dict(controls["parameters"]),
        tutor_llm=object(),
        scorer_provider=scorer_config["provider"],
        scorer_model=scorer_config["model"],
        scorer_parameters={
            key: value for key, value in scorer_config.items() if key not in {"provider", "model"}
        },
        scorer_llm=object(),
        connection_fingerprint="a" * 64,
    )
    with Session(engine) as session:
        service = RunService(
            registry=registry,
            tutor_runner=Runner(),
            runs=EvalRunRepository(session),
            score_sets=EvalScoreSetRepository(session),
            scorer_executions=EvalScorerExecutionRepository(session),
            scoring_service_factory=lambda *_args, **_kwargs: scoring,
        )
        prepared = service.prepare(
            experiment_id=registry.experiment.experiment_id,
            task_case_id="tgqa-001",
            variant_id="tutor-v3",
            run_profile="evaluation",
            connection=connection,
        )
        run_id = prepared.run.id
        task = asyncio.create_task(service.execute_prepared(prepared, events=[]))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert scoring.callback is not None

    with Session(engine) as session:
        run = EvalRunRepository(session).get_verified(run_id)
        assert run.lifecycle == "cancelled"
        score_sets = EvalScoreSetRepository(session).list_verified(run_id)
        assert len(score_sets) == 1
        score_set = score_sets[0]
        assert score_set.status == "cancelled"
        before = EvalScorerExecutionRepository(session).list_verified(score_set.id)
        late = ScorerExecutionDraft(
            component_id="citation-integrity",
            component_version="v1",
            scorer_id="citation-integrity",
            scorer_version="v1",
            status="success",
            input_hash=run.artifact_hash,
            output={"late": True},
            findings=(),
        )
        scoring.callback(late)
        after = EvalScorerExecutionRepository(session).list_verified(score_set.id)
        assert len(after) == len(before)


@pytest.mark.asyncio
async def test_parent_cancel_closes_scoring_gate_before_child_late_callback(tmp_path):
    """Scoring callbacks after child cancellation must not append executions."""

    from app.eval.learning_run.contracts import ScorerExecutionDraft

    engine = _engine(tmp_path / "bounded-scoring-cancel.db")
    registry = TaskRegistry.load_default()
    entered = asyncio.Event()
    caught_cancel = asyncio.Event()
    release = asyncio.Event()
    candidate = TutorCandidate(
        answer="answer",
        citations=[],
        evidence=[],
        formatted_context="",
        usage="unavailable",
        trace=[],
    )

    class Runner:
        async def run(self, **_kwargs):
            return candidate

    class CancellationResistantScoring:
        async def score(self, *, candidate, on_execution, **_kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                caught_cancel.set()
                on_execution(
                    ScorerExecutionDraft(
                        component_id="citation-integrity",
                        component_version="v1",
                        scorer_id="citation-integrity",
                        scorer_version="v1",
                        status="success",
                        input_hash=candidate.compute_hash(),
                        output={"late": True},
                        findings=(),
                    )
                )
                await release.wait()
                raise

    scoring = CancellationResistantScoring()
    controls = registry.experiment.variants["tutor-v3"]
    scorer_config = dict(registry.scorer.model_config)
    connection = EvalModelConnection(
        tutor_provider=controls["provider"],
        tutor_model=controls["model"],
        tutor_parameters=dict(controls["parameters"]),
        tutor_llm=object(),
        scorer_provider=scorer_config["provider"],
        scorer_model=scorer_config["model"],
        scorer_parameters={
            key: value for key, value in scorer_config.items() if key not in {"provider", "model"}
        },
        scorer_llm=object(),
        connection_fingerprint="a" * 64,
    )
    with Session(engine) as session:
        service = RunService(
            registry=registry,
            tutor_runner=Runner(),
            runs=EvalRunRepository(session),
            score_sets=EvalScoreSetRepository(session),
            scorer_executions=EvalScorerExecutionRepository(session),
            scoring_service_factory=lambda *_args, **_kwargs: scoring,
        )
        prepared = service.prepare(
            experiment_id=registry.experiment.experiment_id,
            task_case_id="tgqa-001",
            variant_id="tutor-v3",
            run_profile="evaluation",
            connection=connection,
        )
        task = asyncio.create_task(service.execute_prepared(prepared, events=[]))
        await entered.wait()
        task.cancel()
        await caught_cancel.wait()
        try:
            assert prepared.cancellation.cancelled
        finally:
            release.set()
            try:
                await task
            except asyncio.CancelledError:
                pass
        run = EvalRunRepository(session).get_verified(prepared.run.id)
        assert run.lifecycle == "cancelled"
        score_set = EvalScoreSetRepository(session).list_verified(prepared.run.id)[0]
        assert score_set.status == "cancelled"
        executions = EvalScorerExecutionRepository(session).list_verified(score_set.id)
        assert executions == []


@pytest.mark.asyncio
async def test_real_tutor_runner_cooperative_cancel_terminalizes_cancelled(tmp_path):
    """A token boundary through TutorAttemptEngine/Runner is a normal cancel."""

    from app.agent.tutor_attempt import TutorAttemptEngine
    from app.eval.learning_run.corpus import CorpusMaterializerController
    from app.eval.learning_run.runner import TutorRunner

    engine = _engine(tmp_path / "cooperative-runner-cancel.db")
    registry = TaskRegistry.load_default()

    class Retriever:
        def search(self, _query, top_k):
            assert top_k == 5
            return []

        def close(self):
            return None

    class Loader:
        def load(self, *, snapshot, stop=None, deadline=None):
            del snapshot, stop, deadline
            return Retriever()

    class Chunk:
        def __init__(self, content):
            self.content = content

    class LLM:
        async def astream(self, _messages):
            yield Chunk("first")
            await asyncio.sleep(0)
            yield Chunk("second")

    loader = Loader()
    controller = CorpusMaterializerController(loader)
    runner = TutorRunner(
        corpus_loader=loader,
        attempt_engine=TutorAttemptEngine(),
        materializer_controller=controller,
    )
    controls = registry.experiment.variants["tutor-v3"]
    scorer_config = dict(registry.scorer.model_config)
    connection = EvalModelConnection(
        tutor_provider=controls["provider"],
        tutor_model=controls["model"],
        tutor_parameters=dict(controls["parameters"]),
        tutor_llm=LLM(),
        scorer_provider=scorer_config["provider"],
        scorer_model=scorer_config["model"],
        scorer_parameters={
            key: value for key, value in scorer_config.items() if key not in {"provider", "model"}
        },
        scorer_llm=object(),
        connection_fingerprint="a" * 64,
    )
    events: list[dict] = []
    with Session(engine) as session:
        service = RunService(
            registry=registry,
            tutor_runner=runner,
            runs=EvalRunRepository(session),
            score_sets=EvalScoreSetRepository(session),
            scorer_executions=EvalScorerExecutionRepository(session),
        )
        prepared = service.prepare(
            experiment_id=registry.experiment.experiment_id,
            task_case_id="tgqa-001",
            variant_id="tutor-v3",
            run_profile="evaluation",
            connection=connection,
        )

        def cancel_on_first_token(payload):
            events.append(payload)
            if payload.get("type") == "token":
                prepared.cancellation.cancel()

        try:
            result = await service.execute_prepared(prepared, cancel_on_first_token)
            assert result.run.lifecycle == "cancelled"
            assert result.run.outcome is None
        finally:
            controller.shutdown(wait=False)


@pytest.mark.asyncio
async def test_cooperative_cancel_durable_failure_propagates_typed_error(tmp_path, monkeypatch):
    from app.eval.learning_run.repositories import EvaluationUnavailableError

    engine = _engine(tmp_path / "cooperative-cancel-durable-failure.db")
    registry = TaskRegistry.load_default()
    candidate = TutorCandidate(
        answer="answer",
        citations=[],
        evidence=[],
        formatted_context="",
        usage="unavailable",
        trace=[],
    )

    class Runner:
        def __init__(self):
            self.token = None

        async def run(self, **_kwargs):
            self.token.cancel()
            return candidate

    runner = Runner()
    controls = registry.experiment.variants["tutor-v3"]
    scorer_config = dict(registry.scorer.model_config)
    connection = EvalModelConnection(
        tutor_provider=controls["provider"],
        tutor_model=controls["model"],
        tutor_parameters=dict(controls["parameters"]),
        tutor_llm=object(),
        scorer_provider=scorer_config["provider"],
        scorer_model=scorer_config["model"],
        scorer_parameters={
            key: value for key, value in scorer_config.items() if key not in {"provider", "model"}
        },
        scorer_llm=object(),
        connection_fingerprint="a" * 64,
    )

    def unavailable_cancel(self, _run_id, **_kwargs):
        raise EvaluationUnavailableError()

    monkeypatch.setattr(
        "app.eval.learning_run.repositories.EvalExecutionControlRepository.cancel_run",
        unavailable_cancel,
    )
    with Session(engine) as session:
        service = RunService(
            registry=registry,
            tutor_runner=runner,
            runs=EvalRunRepository(session),
            score_sets=EvalScoreSetRepository(session),
            scorer_executions=EvalScorerExecutionRepository(session),
        )
        prepared = service.prepare(
            experiment_id=registry.experiment.experiment_id,
            task_case_id="tgqa-001",
            variant_id="tutor-v3",
            run_profile="evaluation",
            connection=connection,
        )
        runner.token = prepared.cancellation
        with pytest.raises(EvaluationUnavailableError):
            await service.execute_prepared(prepared, events=[])


@pytest.mark.asyncio
async def test_service_token_boundaries_do_not_probe_durable_state_per_event(tmp_path, monkeypatch):
    from app.eval.learning_run.service import EvalModelConnection, RunService

    engine = _engine(tmp_path / "service-token-probes.db")
    registry = TaskRegistry.load_default()
    EvalExecutionControlRepository, _, _ = _task7_control_api()
    probe_calls: list[str] = []
    original_probe = EvalExecutionControlRepository.is_cancelled

    def spy_probe(self, run_id):
        probe_calls.append(run_id)
        return original_probe(self, run_id)

    monkeypatch.setattr(EvalExecutionControlRepository, "is_cancelled", spy_probe)
    candidate = TutorCandidate(
        answer="answer",
        citations=[],
        evidence=[],
        formatted_context="",
        usage="unavailable",
        trace=[],
    )

    class Runner:
        def __init__(self):
            self.token = None

        async def run(self, **kwargs):
            for index in range(100):
                kwargs["events"]({"type": "token", "text": str(index)})
            self.token.cancel()
            return candidate

    runner = Runner()
    controls = registry.experiment.variants["tutor-v3"]
    scorer_config = dict(registry.scorer.model_config)
    connection = EvalModelConnection(
        tutor_provider=controls["provider"],
        tutor_model=controls["model"],
        tutor_parameters=dict(controls["parameters"]),
        tutor_llm=object(),
        scorer_provider=scorer_config["provider"],
        scorer_model=scorer_config["model"],
        scorer_parameters={
            key: value for key, value in scorer_config.items() if key not in {"provider", "model"}
        },
        scorer_llm=object(),
        connection_fingerprint="a" * 64,
    )
    with Session(engine) as session:
        service = RunService(
            registry=registry,
            tutor_runner=runner,
            runs=EvalRunRepository(session),
            score_sets=EvalScoreSetRepository(session),
            scorer_executions=EvalScorerExecutionRepository(session),
        )
        prepared = service.prepare(
            experiment_id=registry.experiment.experiment_id,
            task_case_id="tgqa-001",
            variant_id="tutor-v3",
            run_profile="evaluation",
            connection=connection,
        )
        runner.token = prepared.cancellation
        result = await service.execute_prepared(prepared, events=[])
        assert result.run.lifecycle == "cancelled"
    assert len(probe_calls) <= 3


def _artifact_hash(engine, run_id: str) -> str:
    with Session(engine) as session:
        return session.execute(
            select(EvalRun.artifact_hash).where(EvalRun.id == run_id)
        ).scalar_one()


def _join_controlled_threads(threads: list[threading.Thread]) -> None:
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive(), "controlled race worker did not finish"


@pytest.mark.parametrize("winner", ["a", "b"])
def test_controlled_run_run_commit_orders_release_gate_after_winner_terminal(tmp_path, winner):
    """Two real sessions use typed claims; winner order is selected by Event."""

    _, EvaluationBusyError, _ = _task7_control_api()
    engine = _engine(tmp_path / f"controlled-run-run-{winner}.db")
    barrier = threading.Barrier(2)
    winner_claimed = threading.Event()
    claims: list[tuple[str, str, str | None]] = []

    def claim(label: str) -> None:
        manifest = _manifest(f"controlled-{label}")
        with Session(engine) as session:
            barrier.wait()
            if label != winner:
                winner_claimed.wait()
            try:
                row = EvalExecutionClaimRepository(session).claim_run(manifest=manifest)
                claims.append((label, "claimed", row.id))
            except EvaluationBusyError as exc:
                claims.append((label, "busy", exc.active_entity_id))
            finally:
                if label == winner:
                    winner_claimed.set()

    _join_controlled_threads([
        threading.Thread(target=claim, args=("a",)),
        threading.Thread(target=claim, args=("b",)),
    ])
    assert sorted(item[1] for item in claims) == ["busy", "claimed"]
    winner_id = next(item[2] for item in claims if item[1] == "claimed")
    with Session(engine) as session:
        terminal = EvalExecutionControlRepository(session).cancel_run(winner_id)
        assert terminal.lifecycle == "cancelled"
        fresh = EvalExecutionClaimRepository(session).claim_run(
            manifest=_manifest("controlled-run-run-after")
        )
        assert fresh.lifecycle == "running"


@pytest.mark.parametrize("winner", ["run", "score_set"])
def test_controlled_run_claim_score_set_orders_are_typed_and_release_gate(tmp_path, winner):
    _, EvaluationBusyError, _ = _task7_control_api()
    engine = _engine(tmp_path / f"controlled-run-score-{winner}.db")
    parent = _seed_frozen_run(engine, run_id="controlled-rescore-parent", finished=True)
    parent_hash = _artifact_hash(engine, parent.id)
    barrier = threading.Barrier(2)
    winner_claimed = threading.Event()
    outcomes: list[tuple[str, str, str | None]] = []

    def claim_run() -> None:
        with Session(engine) as session:
            barrier.wait()
            if winner != "run":
                winner_claimed.wait()
            try:
                row = EvalExecutionClaimRepository(session).claim_run(
                    manifest=_manifest("controlled-new-run")
                )
                outcomes.append(("run", "claimed", row.id))
            except EvaluationBusyError as exc:
                outcomes.append(("run", "busy", exc.active_entity_id))
            finally:
                if winner == "run":
                    winner_claimed.set()

    def claim_score_set() -> None:
        with Session(engine) as session:
            barrier.wait()
            if winner != "score_set":
                winner_claimed.wait()
            try:
                row = EvalExecutionClaimRepository(session).claim_score_set(
                    run_id=parent.id,
                    artifact_input_hash=parent_hash,
                    scorer_id="hybrid-v1",
                    scorer_version="v1",
                    id=f"controlled-score-{winner}",
                )
                outcomes.append(("score_set", "claimed", row.id))
            except EvaluationBusyError as exc:
                outcomes.append(("score_set", "busy", exc.active_entity_id))
            finally:
                if winner == "score_set":
                    winner_claimed.set()

    _join_controlled_threads([
        threading.Thread(target=claim_run),
        threading.Thread(target=claim_score_set),
    ])
    assert sorted(item[1] for item in outcomes) == ["busy", "claimed"]
    claimed_kind, _, claimed_id = next(item for item in outcomes if item[1] == "claimed")

    with Session(engine) as session:
        if claimed_kind == "run":
            terminal = EvalExecutionControlRepository(session).cancel_run(claimed_id)
            assert terminal.lifecycle == "cancelled"
        else:
            terminal = EvalScoreSetRepository(session).cancel_once(claimed_id)
            assert terminal.status == "cancelled"
        fresh = EvalExecutionClaimRepository(session).claim_run(
            manifest=_manifest("controlled-run-score-after")
        )
        assert fresh.lifecycle == "running"


@pytest.mark.parametrize("winner", ["cancel", "finalize"])
def test_controlled_cancel_finalize_orders_preserve_candidate_and_reject_late_append(tmp_path, winner):
    control_cls, _, conflict_cls = _task7_control_api()
    engine = _engine(tmp_path / f"controlled-cancel-finalize-{winner}.db")
    run = _seed_frozen_run(engine, run_id="controlled-cancel-finalize")
    artifact_hash = _artifact_hash(engine, run.id)
    with Session(engine) as session:
        score = EvalScoreSetRepository(session).create(
            run_id=run.id,
            scorer_id="hybrid-v1",
            scorer_version="v1",
            artifact_input_hash=artifact_hash,
            id="controlled-cancel-score",
        )
        EvalScoreSetRepository(session).claim_running(score.id)

    barrier = threading.Barrier(2)
    winner_done = threading.Event()
    outcomes: list[tuple[str, str, str | None]] = []

    def cancel_worker() -> None:
        with Session(engine) as session:
            barrier.wait()
            if winner != "cancel":
                winner_done.wait()
            try:
                row = control_cls(session).cancel_run(run.id)
                outcomes.append(("cancel", row.lifecycle, row.outcome))
            except BaseException as exc:
                outcomes.append(("cancel", type(exc).__name__, None))
            finally:
                if winner == "cancel":
                    winner_done.set()

    def finalize_worker() -> None:
        with Session(engine) as session:
            barrier.wait()
            if winner != "finalize":
                winner_done.wait()
            try:
                row = EvalRunRepository(session).finalize_success(run.id)
                outcomes.append(("finalize", row.lifecycle, row.outcome))
            except BaseException as exc:
                outcomes.append(("finalize", type(exc).__name__, None))
            finally:
                if winner == "finalize":
                    winner_done.set()

    _join_controlled_threads([
        threading.Thread(target=cancel_worker),
        threading.Thread(target=finalize_worker),
    ])
    with Session(engine) as session:
        persisted = EvalRunRepository(session).get_verified(run.id)
        assert persisted.candidate_artifact_json is not None
        assert persisted.artifact_hash == artifact_hash
        score = EvalScoreSetRepository(session).get_verified("controlled-cancel-score")
        if score.status == "running":
            EvalScoreSetRepository(session).finalize_once(
                score.id,
                status="failed",
                quality_verdict="inconclusive",
                aggregate_scores={},
                findings=[],
            )
        with pytest.raises(conflict_cls):
            EvalScorerExecutionRepository(session).append(
                score_set_id=score.id,
                scorer_id="late-scorer",
                scorer_version="v1",
                status="success",
                input_hash=artifact_hash,
                output={"late": True},
            )
        fresh = EvalExecutionClaimRepository(session).claim_run(
            manifest=_manifest("controlled-cancel-finalize-after")
        )
        assert fresh.lifecycle == "running"


@pytest.mark.parametrize("reconcile_first", [True, False])
def test_controlled_reconcile_new_claim_orders_use_typed_claim_and_cutoff(tmp_path, reconcile_first):
    control_cls, EvaluationBusyError, _ = _task7_control_api()
    engine = _engine(tmp_path / f"controlled-reconcile-{reconcile_first}.db")
    _seed_stale_active_rows(engine)
    cutoff = datetime(2026, 1, 2)
    barrier = threading.Barrier(2)
    reconcile_done = threading.Event()
    claim_attempted = threading.Event()
    outcomes: list[tuple[str, str, str | None]] = []

    def reconcile_worker() -> None:
        with Session(engine) as session:
            barrier.wait()
            if not reconcile_first:
                claim_attempted.wait()
            result = control_cls(session).reconcile(started_before=cutoff)
            outcomes.append(("reconcile", "done", str(result.runs_reconciled)))
            reconcile_done.set()

    def claim_worker() -> None:
        with Session(engine) as session:
            barrier.wait()
            if reconcile_first:
                reconcile_done.wait()
            try:
                row = EvalExecutionClaimRepository(session).claim_run(
                    manifest=_manifest("controlled-reconcile-fresh"),
                    created_at=cutoff + timedelta(days=1),
                )
                outcomes.append(("claim", "claimed", row.id))
            except EvaluationBusyError as exc:
                outcomes.append(("claim", "busy", exc.active_entity_id))
                claim_attempted.set()
                reconcile_done.wait()
                row = EvalExecutionClaimRepository(session).claim_run(
                    manifest=_manifest("controlled-reconcile-fresh-retry"),
                    created_at=cutoff + timedelta(days=1),
                )
                outcomes.append(("claim", "claimed", row.id))
            else:
                claim_attempted.set()

    _join_controlled_threads([
        threading.Thread(target=reconcile_worker),
        threading.Thread(target=claim_worker),
    ])
    assert any(item[0] == "reconcile" and item[2] == "1" for item in outcomes)
    fresh_id = next(item[2] for item in outcomes if item[0] == "claim" and item[1] == "claimed")
    with Session(engine) as session:
        stale = EvalRunRepository(session).get_verified("stale-run")
        fresh = EvalRunRepository(session).get_verified(fresh_id)
        score = EvalScoreSetRepository(session).get_verified("stale-score")
        assert stale.outcome == "system_failed"
        assert score.status == "failed"
        assert fresh.lifecycle == "running"
        EvalExecutionControlRepository(session).cancel_run(fresh.id)
        another = EvalExecutionClaimRepository(session).claim_run(
            manifest=_manifest("controlled-reconcile-after")
        )
        assert another.lifecycle == "running"


def test_learning_reset_during_and_after_preserves_eval_on_second_connection(tmp_path):
    engine = _engine(tmp_path / "learning-reset-preserves-eval.db")
    with Session(engine) as session:
        session.add(User(id="user-1", fingerprint="learning-owner"))
        session.flush()
        session.add(
            Document(
                id="document-1",
                user_id="user-1",
                filename="notes.pdf",
                hash="hash-1",
                chunks_count=2,
            )
        )
        session.add(
            EvalRun(
                id="preserved-run",
                experiment_id="tutor-prompt-regression-v1",
                task_case_id="tgqa-001",
                task_case_version="1",
                variant_id="tutor-v2",
                run_profile="evaluation",
                lifecycle="finished",
                outcome="success",
                manifest_json={"task_case_id": "tgqa-001"},
                manifest_hash="a" * 64,
                candidate_artifact_json={"answer": "x"},
                artifact_hash="b" * 64,
            )
        )
        session.add(
            EvalScoreSet(
                id="preserved-score",
                run_id="preserved-run",
                scorer_id="hybrid-v1",
                scorer_version="v1",
                scorer_snapshot_json={"scorer_id": "hybrid", "version": "v1"},
                scorer_definition_hash="c" * 64,
                artifact_input_hash="b" * 64,
                status="completed",
                quality_verdict="pass",
            )
        )
        session.add(
            EvalScorerExecution(
                id="preserved-exec",
                score_set_id="preserved-score",
                scorer_id="retrieval-integrity",
                scorer_version="v1",
                status="success",
                input_hash="b" * 64,
                output_json={"result": True},
            )
        )
        session.commit()

    started = threading.Event()
    release = threading.Event()

    def learning_delete() -> None:
        with Session(engine) as session:
            started.set()
            release.wait(timeout=10)
            DataLifecycleRepository(session).delete_learning_data(include_users=False)
            session.commit()

    worker = threading.Thread(target=learning_delete)
    worker.start()
    try:
        assert started.wait(timeout=5)
        with Session(engine) as session:
            during = DataLifecycleRepository(session).count_eval()
        assert during["runs"] == 1
        assert during["score_sets"] == 1
        assert during["scorer_executions"] == 1
    finally:
        release.set()
        worker.join(timeout=10)
        assert not worker.is_alive()

    with Session(engine) as session:
        repo = DataLifecycleRepository(session)
        after = repo.count_eval()
        assert after["runs"] == 1
        assert after["score_sets"] == 1
        assert after["scorer_executions"] == 1
        assert repo.count_all()["documents"] == 0
