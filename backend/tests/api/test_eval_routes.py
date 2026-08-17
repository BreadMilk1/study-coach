"""Task 6 local-only authenticated evaluation API tests."""

from __future__ import annotations

import json
import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.auth import issue_token
from app.db.models import User
from app.db.models import EvalRun, EvalScoreSet
from app.db.session import session_scope
from app.eval.learning_run.contracts import CandidateArtifact, RunManifest
from app.eval.learning_run.repositories import (
    EvalRunRepository,
    EvalScoreSetRepository,
    EvalScorerExecutionRepository,
)


class _FakeConnectionFactory:
    def __init__(self):
        self.calls = 0
        self.variants = []

    def __call__(self, *, provider, model, variant_id, api_key=None, base_url=None, **_kwargs):
        self.calls += 1
        self.variants.append(variant_id)
        assert api_key == "secret-value" or api_key is None
        assert base_url == "https://secret.example/v1" or base_url is None
        from app.eval.learning_run.service import EvalModelConnection

        return EvalModelConnection(
            tutor_provider=provider or "ollama",
            tutor_model=model or "llama3.2",
            tutor_parameters={"temperature": 0, "top_p": 1},
            tutor_llm=object(),
            scorer_provider="ollama",
            scorer_model="llama3.2",
            scorer_parameters={"temperature": 0, "max_tokens": 512},
            scorer_llm=object(),
            connection_fingerprint="a" * 64,
        )


RUN_REQUEST = {
    "experiment_id": "tutor-prompt-regression-v1",
    "task_case_id": "tgqa-004",
    "variant_id": "tutor-v3",
    "run_profile": "evaluation",
}


def _read_sse(response):
    events = []
    for line in response.text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


class _FakePrepared:
    run = SimpleNamespace(
        id="fake-run-001",
        lifecycle="running",
        outcome=None,
    )


class _BlockingEvalService:
    """A stream service whose execution remains attached until cancelled."""

    def __init__(self, run_id: str):
        self.prepared = SimpleNamespace(
            run=SimpleNamespace(id=run_id, lifecycle="running", outcome=None)
        )
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False
        self.done = asyncio.Event()

    def prepare(self, **_kwargs):
        return self.prepared

    async def execute_prepared(self, prepared, events):
        events({"type": "run_created", "run_id": prepared.run.id})
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        finally:
            self.done.set()
        return SimpleNamespace(
            run=SimpleNamespace(id=prepared.run.id, lifecycle="finished", outcome="success"),
            score_set=None,
            executions=(),
        )


class _FakeEvalService:
    def __init__(self):
        self.prepare_calls = 0
        self.execute_calls = 0

    def prepare(self, *, connection, **_kwargs):
        self.prepare_calls += 1
        if connection.tutor_provider != "ollama" or connection.tutor_model != "llama3.2":
            from app.eval.learning_run.service import RunRequestError

            raise RunRequestError(
                "evaluation_config_mismatch",
                "evaluation configuration does not match",
                fields=("provider", "model"),
            )
        return _FakePrepared()

    def prepare_rescore(self, **_kwargs):
        self.prepare_calls += 1
        return SimpleNamespace(
            score_set=SimpleNamespace(id="fake-score-set-001"),
            cancellation=SimpleNamespace(cancel=lambda: None),
        )

    async def execute_rescore(self, prepared, events):
        self.execute_calls += 1
        events({"type": "score_set_created", "score_set_id": prepared.score_set.id})
        events({
            "type": "score_set_finished",
            "score_set_id": prepared.score_set.id,
            "status": "completed",
            "quality_verdict": "pass",
        })
        return prepared.score_set

    async def execute_prepared(self, prepared, events):
        self.execute_calls += 1
        score_set_id = "fake-score-set-001"
        events({"type": "run_created", "run_id": prepared.run.id})
        events({"type": "stage_started", "run_id": prepared.run.id, "stage": "tutor"})
        events({"type": "stage_completed", "run_id": prepared.run.id, "stage": "tutor"})
        events({"type": "score_set_created", "run_id": prepared.run.id, "score_set_id": score_set_id})
        events({
            "type": "scorer_completed",
            "run_id": prepared.run.id,
            "score_set_id": score_set_id,
            "scorer_id": "fake-scorer",
            "status": "success",
        })
        events({
            "type": "score_set_finished",
            "run_id": prepared.run.id,
            "score_set_id": score_set_id,
            "status": "completed",
            "quality_verdict": "pass",
        })
        events({"type": "run_finished", "run_id": prepared.run.id, "lifecycle": "finished", "outcome": "success"})
        return SimpleNamespace(
            run=SimpleNamespace(
                id=prepared.run.id,
                lifecycle="finished",
                outcome="success",
                experiment_id="tutor-prompt-regression-v1",
                task_case_id="tgqa-004",
                variant_id="tutor-v3",
                run_profile="evaluation",
            ),
            score_set=None,
            executions=(),
        )


class _BusyEvalService(_FakeEvalService):
    def prepare(self, **_kwargs):
        from app.eval.learning_run.repositories import EvaluationBusyError

        self.prepare_calls += 1
        raise EvaluationBusyError("active-run-001", "run")

    def prepare_rescore(self, **_kwargs):
        from app.eval.learning_run.repositories import EvaluationBusyError

        self.prepare_calls += 1
        raise EvaluationBusyError("active-score-set-001", "score_set")


class _FailingEvalService(_FakeEvalService):
    async def execute_prepared(self, *_args, **_kwargs):
        self.execute_calls += 1
        raise RuntimeError("api_key=secret-value")


class _FailedEventEvalService(_FakeEvalService):
    async def execute_prepared(self, prepared, events):
        self.execute_calls += 1
        score_set_id = "fake-score-set-failed-001"
        events({"type": "run_created", "run_id": prepared.run.id})
        events({"type": "stage_started", "run_id": prepared.run.id, "stage": "tutor"})
        events({"type": "stage_completed", "run_id": prepared.run.id, "stage": "tutor"})
        events({"type": "score_set_created", "run_id": prepared.run.id, "score_set_id": score_set_id})
        events({
            "type": "scorer_failed",
            "run_id": prepared.run.id,
            "score_set_id": score_set_id,
            "scorer_id": "fake-scorer",
            "status": "failed",
            "error_code": "scorer_parse_error",
        })
        events({
            "type": "score_set_finished",
            "run_id": prepared.run.id,
            "score_set_id": score_set_id,
            "status": "failed",
            "quality_verdict": "inconclusive",
        })
        events({
            "type": "run_finished",
            "run_id": prepared.run.id,
            "lifecycle": "finished",
            "outcome": "system_failed",
            "error_code": "scorer_parse_error",
        })


class _DurableSuccessProjectionFailureService(_FakeEvalService):
    def __init__(self, run_id: str):
        super().__init__()
        self._prepared = SimpleNamespace(
            run=SimpleNamespace(id=run_id, lifecycle="running", outcome=None)
        )

    def prepare(self, **_kwargs):
        self.prepare_calls += 1
        return self._prepared

    async def execute_prepared(self, prepared, events):
        self.execute_calls += 1
        events({"type": "run_created", "run_id": prepared.run.id})
        raise RuntimeError("scorer projection unavailable")


class _ExplodingConnectionFactory:
    def __call__(self, **_kwargs):
        raise RuntimeError(
            "api_key=SECRET_API_KEY_MARKER "
            "authorization=Bearer SECRET_AUTH_MARKER "
            "https://SECRET_FULL_URL_MARKER.example/v1"
        )


class _UnavailableEvalService(_FakeEvalService):
    def prepare(self, **_kwargs):
        from app.eval.learning_run.repositories import EvaluationUnavailableError

        self.prepare_calls += 1
        raise EvaluationUnavailableError()


def _seed_historical_run() -> str:
    manifest = RunManifest(
        experiment_id="tutor-prompt-regression-v1",
        task_case_id="tgqa-004",
        task_case_version="1",
        variant_id="tutor-v3",
        run_profile="evaluation",
        task_snapshot={"id": "tgqa-004", "version": "1", "question": "question"},
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
        provider="ollama",
        model="llama3.2",
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
    artifact = CandidateArtifact(
        answer="frozen answer",
        citations=(),
        exact_evidence=(),
        formatted_context="evidence",
        usage="unavailable",
        trace=(),
        budget={"total_seconds": 90},
    )
    with session_scope() as session:
        run_repo = EvalRunRepository(session)
        run = run_repo.create(
            experiment_id=manifest.experiment_id,
            task_case_id=manifest.task_case_id,
            task_case_version=manifest.task_case_version,
            variant_id=manifest.variant_id,
            run_profile=manifest.run_profile,
            manifest=manifest,
            manifest_hash=manifest.compute_hash(),
            id="historical-run-001",
        )
        run_repo.claim_running(run.id)
        run_repo.freeze_candidate(run.id, artifact, artifact_hash=artifact.compute_hash())
        run = run_repo.finalize_success(run.id)
        score_repo = EvalScoreSetRepository(session)
        score_set = score_repo.create(
            run_id=run.id,
            scorer_id="hybrid-v1",
            scorer_version="v1",
            artifact_input_hash=run.artifact_hash,
            id="historical-score-set-001",
        )
        score_repo.claim_running(score_set.id)
        EvalScorerExecutionRepository(session).append(
            score_set_id=score_set.id,
            scorer_id="retrieval-integrity",
            scorer_version="v1",
            status="success",
            input_hash=run.artifact_hash,
            output={"result": {"evidence_count": 0}, "findings": []},
        )
        score_repo.finalize_once(
            score_set.id,
            status="completed",
            quality_verdict="pass",
            aggregate_scores={"groundedness": 4},
            findings=[],
        )
    return run.id


def _seed_running_run(*, run_id: str, frozen: bool = False, score_set_id: str | None = None) -> str:
    manifest = RunManifest(
        experiment_id="tutor-prompt-regression-v1",
        task_case_id="tgqa-004",
        task_case_version="1",
        variant_id="tutor-v3",
        run_profile="evaluation",
        task_snapshot={"id": "tgqa-004", "version": "1", "question": "question"},
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
        provider="ollama",
        model="llama3.2",
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
    artifact = CandidateArtifact(
        answer="frozen answer",
        citations=(),
        exact_evidence=(),
        formatted_context="evidence",
        usage="unavailable",
        trace=(),
        budget={"total_seconds": 90},
    )
    with session_scope() as session:
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
        )
        run = run_repo.claim_running(run.id)
        if frozen:
            run = run_repo.freeze_candidate(run.id, artifact, artifact_hash=artifact.compute_hash())
        if score_set_id is not None:
            score_repo = EvalScoreSetRepository(session)
            score = score_repo.create(
                run_id=run.id,
                scorer_id="hybrid-v1",
                scorer_version="v1",
                artifact_input_hash=run.artifact_hash or "0" * 64,
                id=score_set_id,
            )
            score_repo.claim_running(score.id)
    return run_id


@pytest.fixture
def eval_client(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDY_COACH_TEST_MODE", "1")
    monkeypatch.setenv("STUDY_COACH_LOCAL_MODE", "1")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/eval-api.db")
    from app.db import session as session_module
    from app.main import create_app

    session_module._engine = None
    session_module._SessionLocal = None
    application = create_app()
    fake_service = _FakeEvalService()
    fake_connection_factory = _FakeConnectionFactory()
    application.state.eval_service_factory = lambda **_kwargs: fake_service
    application.state.eval_connection_factory = fake_connection_factory
    with session_scope() as session:
        session.add(User(id="eval-user", fingerprint="eval-fingerprint"))
        session.commit()
    token = issue_token("eval-user", "guest")
    with TestClient(application, raise_server_exceptions=False) as client:
        yield client, {"Authorization": f"Bearer {token}"}, application, fake_service, fake_connection_factory
    if session_module._engine is not None:
        session_module._engine.dispose()
    session_module._engine = None
    session_module._SessionLocal = None


def test_eval_routes_are_authenticated_and_local_only(eval_client, monkeypatch):
    client, headers, _, _, _ = eval_client
    assert client.get("/api/eval/experiments").status_code == 401

    monkeypatch.setenv("STUDY_COACH_LOCAL_MODE", "0")
    response = client.get("/api/eval/experiments", headers=headers)
    assert response.status_code == 403
    assert response.json()["detail"] == {
        "code": "evaluation_disabled",
        "message": "local evaluation mode is disabled",
    }


def test_create_app_reconciles_stale_eval_rows_before_serving_routes(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDY_COACH_TEST_MODE", "1")
    monkeypatch.setenv("STUDY_COACH_LOCAL_MODE", "1")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/startup-reconcile.db")
    from app.db import session as session_module
    from app.main import create_app

    session_module._engine = None
    session_module._SessionLocal = None
    apps = []
    try:
        apps.append(create_app())
        run_id = _seed_running_run(
            run_id="startup-reconcile-run",
            frozen=True,
            score_set_id="startup-reconcile-score",
        )
        stale_at = datetime.utcnow() - timedelta(days=1)
        with session_scope() as session:
            session.execute(
                EvalRun.__table__.update()
                .where(EvalRun.id == run_id)
                .values(started_at=stale_at)
            )
            session.execute(
                EvalScoreSet.__table__.update()
                .where(EvalScoreSet.id == "startup-reconcile-score")
                .values(started_at=stale_at)
            )
            session.commit()

        # A second boot is the restart boundary: the stale rows are repaired
        # before the new app exposes its routers.
        apps.append(create_app())
        with session_scope() as session:
            run = EvalRunRepository(session).get_verified(run_id)
            score = EvalScoreSetRepository(session).get_verified("startup-reconcile-score")
            assert run.lifecycle == "finished"
            assert run.outcome == "system_failed"
            assert run.operational_error_json["code"] == "process_interrupted"
            assert score.status == "failed"
            assert score.operational_error_code == "process_interrupted"
    finally:
        for application in apps:
            controller = getattr(application.state, "eval_materializer_controller", None)
            if controller is not None:
                controller.shutdown(wait=False)
        if session_module._engine is not None:
            session_module._engine.dispose()
        session_module._engine = None
        session_module._SessionLocal = None


def test_eval_stream_rejects_extra_payload_fields(eval_client):
    client, headers, _, _, _ = eval_client
    response = client.post(
        "/api/eval/runs/stream",
        headers=headers,
        json={**RUN_REQUEST, "prompt": "injected", "corpus_path": "/tmp/x"},
    )
    assert response.status_code == 422


def test_eval_config_mismatch_is_409_without_values(eval_client):
    client, headers, _, _, _ = eval_client
    response = client.post(
        "/api/eval/runs/stream",
        headers={**headers, "x-provider": "wrong-provider", "x-model": "wrong-model"},
        json=RUN_REQUEST,
    )
    assert response.status_code == 409
    body = response.json()["detail"]
    assert body["code"] == "evaluation_config_mismatch"
    assert set(body["fields"]) >= {"provider", "model"}
    assert "wrong-provider" not in response.text
    assert "wrong-model" not in response.text


def test_eval_stream_emits_one_sanitized_terminal_event(eval_client):
    client, headers, _, fake_service, fake_connection_factory = eval_client
    response = client.post(
        "/api/eval/runs/stream",
        headers={
            **headers,
            "x-provider": "ollama",
            "x-model": "llama3.2",
            "x-api-key": "secret-value",
            "x-base-url": "https://secret.example/v1",
        },
        json=RUN_REQUEST,
    )
    assert response.status_code == 200
    events = _read_sse(response)
    assert events[0]["type"] == "run_created"
    assert events[0]["run_id"]
    assert sum(event["type"] == "run_finished" for event in events) == 1
    assert all(event["schema_version"] == "eval-api-v1" for event in events)
    assert "secret-value" not in response.text
    assert "https://secret.example" not in response.text
    assert fake_service.execute_calls == 1
    assert fake_connection_factory.calls == 1


def test_eval_stream_success_has_complete_public_event_sequence_and_validates_union(eval_client):
    from pydantic import TypeAdapter

    from app.api.eval_schemas import EvalEvent

    client, headers, _, _, _ = eval_client
    response = client.post(
        "/api/eval/runs/stream",
        headers={**headers, "x-provider": "ollama", "x-model": "llama3.2"},
        json=RUN_REQUEST,
    )
    assert response.status_code == 200
    events = _read_sse(response)
    assert [event["type"] for event in events] == [
        "run_created",
        "stage_started",
        "stage_completed",
        "score_set_created",
        "scorer_completed",
        "score_set_finished",
        "run_finished",
    ]
    adapter = TypeAdapter(EvalEvent)
    assert all(adapter.validate_python(event) for event in events)


def test_eval_stream_failed_has_complete_sanitized_event_sequence(eval_client):
    from pydantic import TypeAdapter

    from app.api.eval_schemas import EvalEvent

    client, headers, application, _, _ = eval_client
    failed = _FailedEventEvalService()
    application.state.eval_service_factory = lambda **_kwargs: failed
    response = client.post(
        "/api/eval/runs/stream",
        headers={**headers, "x-provider": "ollama", "x-model": "llama3.2"},
        json=RUN_REQUEST,
    )
    assert response.status_code == 200
    events = _read_sse(response)
    assert [event["type"] for event in events] == [
        "run_created",
        "stage_started",
        "stage_completed",
        "score_set_created",
        "scorer_failed",
        "score_set_finished",
        "run_finished",
    ]
    adapter = TypeAdapter(EvalEvent)
    assert all(adapter.validate_python(event) for event in events)


def test_connection_factory_failure_is_sanitized_without_db_or_log_secrets(eval_client, caplog):
    client, headers, application, _, _ = eval_client
    application.state.eval_connection_factory = _ExplodingConnectionFactory()
    caplog.set_level("ERROR")
    response = client.post(
        "/api/eval/runs/stream",
        headers={
            **headers,
            "x-provider": "ollama",
            "x-model": "llama3.2",
            "x-api-key": "SECRET_API_KEY_MARKER",
            "x-base-url": "https://SECRET_FULL_URL_MARKER.example/v1",
        },
        json=RUN_REQUEST,
    )
    assert response.status_code == 503
    assert response.json()["detail"] == {
        "code": "model_unavailable",
        "message": "evaluation model is unavailable",
    }
    combined_logs = "\n".join(record.getMessage() for record in caplog.records)
    for marker in ("SECRET_API_KEY_MARKER", "SECRET_AUTH_MARKER", "SECRET_FULL_URL_MARKER"):
        assert marker not in response.text
        assert marker not in combined_logs
    with session_scope() as session:
        assert session.query(EvalRun).count() == 0


def test_projection_failure_after_durable_success_uses_success_terminal(eval_client):
    client, headers, application, _, _ = eval_client
    run_id = _seed_historical_run()
    failed_projection = _DurableSuccessProjectionFailureService(run_id)
    application.state.eval_service_factory = lambda **_kwargs: failed_projection
    response = client.post(
        "/api/eval/runs/stream",
        headers={**headers, "x-provider": "ollama", "x-model": "llama3.2"},
        json=RUN_REQUEST,
    )
    assert response.status_code == 200
    events = _read_sse(response)
    assert sum(event["type"] == "run_finished" for event in events) == 1
    assert events[-1]["outcome"] == "success"
    assert events[-1].get("error_code") is None


def test_sql_tamper_has_same_sanitized_integrity_error_for_list_and_detail(eval_client):
    client, headers, _, _, _ = eval_client
    list_run_id = _seed_historical_run()
    with session_scope() as session:
        session.execute(
            EvalRun.__table__.update()
            .where(EvalRun.id == list_run_id)
            .values(manifest_json={"tampered": "SECRET_TAMPER_MARKER"})
        )
        session.commit()
    listed = client.get("/api/eval/runs", headers=headers)
    assert listed.status_code == 500
    assert listed.json()["detail"] == {
        "code": "evaluation_integrity_error",
        "message": "evaluation artifact integrity could not be verified",
    }
    assert "SECRET_TAMPER_MARKER" not in listed.text

    detailed = client.get(f"/api/eval/runs/{list_run_id}", headers=headers)
    assert detailed.status_code == 500
    assert detailed.json()["detail"] == {
        "code": "evaluation_integrity_error",
        "message": "evaluation artifact integrity could not be verified",
    }
    assert "SECRET_TAMPER_MARKER" not in detailed.text


def test_sqlite_lock_maps_to_evaluation_unavailable_before_stream(eval_client):
    client, headers, application, _, _ = eval_client
    unavailable = _UnavailableEvalService()
    application.state.eval_service_factory = lambda **_kwargs: unavailable
    response = client.post(
        "/api/eval/runs/stream",
        headers={**headers, "x-provider": "ollama", "x-model": "llama3.2"},
        json=RUN_REQUEST,
    )
    assert response.status_code == 503
    assert response.json()["detail"] == {
        "code": "evaluation_unavailable",
        "message": "evaluation storage is unavailable",
    }


def test_eval_stream_passes_tutor_v2_variant_to_connection_factory(eval_client):
    client, headers, _, fake_service, fake_connection_factory = eval_client
    response = client.post(
        "/api/eval/runs/stream",
        headers={**headers, "x-provider": "ollama", "x-model": "llama3.2"},
        json={**RUN_REQUEST, "variant_id": "tutor-v2"},
    )
    assert response.status_code == 200
    assert fake_connection_factory.variants == ["tutor-v2"]
    assert fake_service.execute_calls == 1


def test_eval_get_lists_and_detail_are_read_only_and_checksum_verified(eval_client):
    client, headers, _, _, _ = eval_client
    run_id = _seed_historical_run()
    experiments = client.get("/api/eval/experiments", headers=headers)
    assert experiments.status_code == 200
    assert experiments.json()[0]["experiment_id"] == "tutor-prompt-regression-v1"
    runs = client.get("/api/eval/runs", headers=headers)
    assert runs.status_code == 200
    assert runs.json()[0]["run_id"] == run_id
    detail = client.get(f"/api/eval/runs/{run_id}", headers=headers)
    assert detail.status_code == 200
    assert detail.json()["manifest"]["task_case_id"] == "tgqa-004"
    assert len(detail.json()["score_sets"]) == 1
    assert detail.json()["score_sets"][0]["scorer_snapshot"]
    assert detail.json()["score_sets"][0]["scorer_definition_hash"]
    assert len(detail.json()["scorer_executions"]) == 1
    missing = client.get("/api/eval/runs/not-found", headers=headers)
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "evaluation_not_found"
    assert experiments.json()[0]["regression_count"] == 0


def test_imported_fixture_compare_is_controlled_and_counts_refusal_regressions(eval_client):
    from pathlib import Path

    from app.db.session import session_scope
    from app.eval.learning_run.registry import TaskRegistry
    from app.eval.learning_run.repositories import EvalSuiteImportRepository

    client, headers, _, _, _ = eval_client
    fixture = (
        Path(__file__).resolve().parents[2]
        / "app"
        / "eval"
        / "learning_run"
        / "fixtures"
        / "tutor-prompt-regression-v1.jsonl"
    )
    records = [
        json.loads(line)
        for line in fixture.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with session_scope() as session:
        imported = EvalSuiteImportRepository(
            session, registry=TaskRegistry.load_default()
        ).import_records(records)
    assert imported == 24

    experiments = client.get("/api/eval/experiments", headers=headers)
    assert experiments.status_code == 200
    assert experiments.json()[0]["regression_count"] == 2

    runs = client.get("/api/eval/runs", headers=headers)
    assert runs.status_code == 200
    left = next(
        row
        for row in runs.json()
        if row["task_case_id"] == "tgqa-008" and row["variant_id"] == "tutor-v2"
    )
    right = next(
        row
        for row in runs.json()
        if row["task_case_id"] == "tgqa-008" and row["variant_id"] == "tutor-v3"
    )
    compare = client.get(
        f"/api/eval/compare?left={left['run_id']}&right={right['run_id']}",
        headers=headers,
    )
    assert compare.status_code == 200
    payload = compare.json()
    assert payload["compatibility"] == "controlled"
    assert payload["reasons"] == []


def test_rescore_busy_is_http_409_before_stream(eval_client):
    client, headers, application, _, _ = eval_client
    run_id = _seed_historical_run()
    busy = _BusyEvalService()
    application.state.eval_service_factory = lambda **_kwargs: busy
    response = client.post(
        f"/api/eval/runs/{run_id}/rescore/stream",
        headers={**headers, "x-provider": "ollama", "x-model": "llama3.2"},
        json={"scorer_version": "hybrid-v2", "run_profile": "evaluation"},
    )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "evaluation_busy"
    assert response.json()["detail"]["active_kind"] == "score_set"
    assert busy.execute_calls == 0


def test_rescore_stream_emits_score_set_id_before_finished(eval_client):
    client, headers, application, fake_service, _ = eval_client
    run_id = _seed_historical_run()
    application.state.eval_service_factory = lambda **_kwargs: fake_service
    response = client.post(
        f"/api/eval/runs/{run_id}/rescore/stream",
        headers={**headers, "x-provider": "ollama", "x-model": "llama3.2"},
        json={"scorer_version": "hybrid-v2", "run_profile": "evaluation"},
    )
    assert response.status_code == 200
    events = _read_sse(response)
    assert events[0]["type"] == "score_set_created"
    assert events[0]["score_set_id"]
    assert any(event["type"] == "score_set_finished" for event in events)


def test_busy_is_returned_before_streaming_response_and_never_executes(eval_client):
    client, headers, application, _, _ = eval_client
    busy = _BusyEvalService()
    application.state.eval_service_factory = lambda **_kwargs: busy
    response = client.post(
        "/api/eval/runs/stream",
        headers={**headers, "x-provider": "ollama", "x-model": "llama3.2"},
        json=RUN_REQUEST,
    )
    assert response.status_code == 409
    assert response.json()["detail"] == {
        "code": "evaluation_busy",
        "message": "another evaluation is already running",
        "active_entity_id": "active-run-001",
        "active_kind": "run",
    }
    assert busy.execute_calls == 0


def test_stream_adapter_adds_one_sanitized_terminal_for_execution_failure(eval_client):
    client, headers, application, _, _ = eval_client
    failed = _FailingEvalService()
    application.state.eval_service_factory = lambda **_kwargs: failed
    response = client.post(
        "/api/eval/runs/stream",
        headers={**headers, "x-provider": "ollama", "x-model": "llama3.2"},
        json=RUN_REQUEST,
    )
    assert response.status_code == 200
    events = _read_sse(response)
    assert [event["type"] for event in events] == ["run_created", "run_finished"]
    assert events[-1]["error_code"] == "harness_internal_error"
    assert "secret-value" not in response.text


def test_cancel_endpoint_is_idempotent_and_cancels_active_score_set(eval_client):
    client, headers, _, _, _ = eval_client
    run_id = _seed_running_run(
        run_id="cancel-endpoint-run",
        frozen=True,
        score_set_id="cancel-endpoint-score-set",
    )

    first = client.post(f"/api/eval/runs/{run_id}/cancel", headers=headers)
    second = client.post(f"/api/eval/runs/{run_id}/cancel", headers=headers)

    assert first.status_code == 200
    assert second.status_code == 200
    first_body = first.json()
    second_body = second.json()
    assert (first_body["lifecycle"], first_body["outcome"]) == ("cancelled", None)
    assert (second_body["lifecycle"], second_body["outcome"]) == ("cancelled", None)
    assert first_body["latest_score_set"]["status"] == "cancelled"
    assert second_body["latest_score_set"]["status"] == "cancelled"
    with session_scope() as session:
        run = EvalRunRepository(session).get_verified(run_id)
        assert run.candidate_artifact_json is not None
        score_set = EvalScoreSetRepository(session).get_verified("cancel-endpoint-score-set")
        assert score_set.status == "cancelled"
        assert score_set.quality_verdict == "not_evaluated"
        assert score_set.operational_error_code == "cancelled"


def test_cancel_endpoint_finished_run_is_a_noop(eval_client):
    client, headers, _, _, _ = eval_client
    run_id = _seed_historical_run()
    before = client.get(f"/api/eval/runs/{run_id}", headers=headers).json()

    response = client.post(f"/api/eval/runs/{run_id}/cancel", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["lifecycle"] == "finished"
    assert body["outcome"] == "success"
    assert body["latest_score_set"]["status"] == before["score_sets"][0]["status"]


def test_score_set_cancel_is_idempotent_and_does_not_rewrite_finished(eval_client):
    client, headers, _, _, _ = eval_client
    run_id = _seed_historical_run()
    before = client.get(f"/api/eval/runs/{run_id}", headers=headers).json()
    finished_id = before["score_sets"][0]["score_set_id"]

    finished = client.post(f"/api/eval/score-sets/{finished_id}/cancel", headers=headers)
    assert finished.status_code == 200
    assert finished.json()["status"] == before["score_sets"][0]["status"]
    assert finished.json()["quality_verdict"] == before["score_sets"][0]["quality_verdict"]

    _seed_running_run(
        run_id="score-set-cancel-run",
        frozen=True,
        score_set_id="score-set-cancel-set",
    )
    first = client.post("/api/eval/score-sets/score-set-cancel-set/cancel", headers=headers)
    second = client.post("/api/eval/score-sets/score-set-cancel-set/cancel", headers=headers)
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["status"] == "cancelled"
    assert second.json()["status"] == "cancelled"
    assert first.json()["quality_verdict"] == "not_evaluated"


def test_cancel_endpoint_rejects_tampered_active_run_without_mutation(eval_client):
    client, headers, _, _, _ = eval_client
    run_id = _seed_running_run(run_id="api-tampered-active-run", frozen=True)
    with session_scope() as session:
        session.execute(
            EvalRun.__table__.update()
            .where(EvalRun.id == run_id)
            .values(manifest_json={"tampered": "api"})
        )
        session.commit()

    response = client.post(f"/api/eval/runs/{run_id}/cancel", headers=headers)

    assert response.status_code == 500
    assert response.json()["detail"] == {
        "code": "evaluation_integrity_error",
        "message": "evaluation artifact integrity could not be verified",
    }
    with session_scope() as session:
        raw = session.execute(select(EvalRun).where(EvalRun.id == run_id)).scalar_one()
        assert raw.lifecycle == "running"
        assert raw.finished_at is None


def test_cancel_endpoint_maps_control_storage_failure_to_sanitized_503(eval_client, monkeypatch):
    client, headers, application, _, _ = eval_client
    run_id = _seed_running_run(run_id="cancel-storage-failure", frozen=False)

    from app.eval.learning_run.repositories import EvaluationUnavailableError

    def unavailable(*_args, **_kwargs):
        raise EvaluationUnavailableError()

    monkeypatch.setattr(
        "app.api.eval_routes.EvalExecutionControlRepository.cancel_run",
        unavailable,
    )
    response = client.post(f"/api/eval/runs/{run_id}/cancel", headers=headers)

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "code": "evaluation_unavailable",
        "message": "evaluation storage is unavailable",
    }


@pytest.mark.asyncio
async def test_stream_body_iterator_aclose_durably_cancels_attached_run(eval_client, monkeypatch):
    from starlette.requests import Request

    from app.api.eval_routes import stream_run
    from app.api.eval_schemas import RunStreamRequest

    _, _, application, _, _ = eval_client
    run_id = _seed_running_run(run_id="generator-close-run", frozen=False)
    blocking = _BlockingEvalService(run_id)
    from app.eval.learning_run.service import EvalModelConnection

    monkeypatch.setattr(
        "app.api.eval_routes._build_connection",
        lambda *_args, **_kwargs: EvalModelConnection(
            tutor_provider="ollama",
            tutor_model="llama3.2",
            tutor_parameters={"temperature": 0, "top_p": 1},
            tutor_llm=object(),
            scorer_provider="ollama",
            scorer_model="llama3.2",
            scorer_parameters={"temperature": 0, "max_tokens": 512},
            scorer_llm=object(),
            connection_fingerprint="a" * 64,
        ),
    )
    application.state.eval_service_factory = lambda **_kwargs: blocking
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/eval/runs/stream",
            "query_string": b"",
            "headers": [],
            "app": application,
        }
    )
    with session_scope() as session:
        response = await stream_run(
            request,
            RunStreamRequest.model_validate(RUN_REQUEST),
            session,
            x_provider="ollama",
            x_model="llama3.2",
        )
        iterator = response.body_iterator
        first = await iterator.__anext__()
        assert '"type":"run_created"' in first
        await blocking.started.wait()
        await iterator.aclose()
        await blocking.done.wait()

    assert blocking.cancelled is True
    with session_scope() as session:
        run = EvalRunRepository(session).get_verified(run_id)
        assert run.lifecycle == "cancelled"
        assert run.outcome is None


@pytest.mark.asyncio
async def test_stream_body_consumer_cancelled_error_is_distinct_from_aclose(eval_client, monkeypatch):
    from starlette.requests import Request

    from app.api.eval_routes import stream_run
    from app.api.eval_schemas import RunStreamRequest

    _, _, application, _, _ = eval_client
    run_id = _seed_running_run(run_id="consumer-cancel-run", frozen=False)
    blocking = _BlockingEvalService(run_id)
    from app.eval.learning_run.service import EvalModelConnection

    monkeypatch.setattr(
        "app.api.eval_routes._build_connection",
        lambda *_args, **_kwargs: EvalModelConnection(
            tutor_provider="ollama",
            tutor_model="llama3.2",
            tutor_parameters={"temperature": 0, "top_p": 1},
            tutor_llm=object(),
            scorer_provider="ollama",
            scorer_model="llama3.2",
            scorer_parameters={"temperature": 0, "max_tokens": 512},
            scorer_llm=object(),
            connection_fingerprint="a" * 64,
        ),
    )
    application.state.eval_service_factory = lambda **_kwargs: blocking
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/eval/runs/stream",
            "query_string": b"",
            "headers": [],
            "app": application,
        }
    )
    with session_scope() as session:
        response = await stream_run(
            request,
            RunStreamRequest.model_validate(RUN_REQUEST),
            session,
            x_provider="ollama",
            x_model="llama3.2",
        )
        iterator = response.body_iterator
        await iterator.__anext__()
        await blocking.started.wait()
        consumer = asyncio.create_task(iterator.__anext__())
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer
        await blocking.done.wait()

    assert blocking.cancelled is True
    with session_scope() as session:
        assert EvalRunRepository(session).get_verified(run_id).lifecycle == "cancelled"


def test_exclusive_reset_rejects_eval_stream_before_any_row(eval_client):
    client, headers, application, fake_service, _ = eval_client
    with session_scope() as session:
        before = session.execute(select(EvalRun.id)).scalars().all()

    with application.state.data_lifecycle_gate.exclusive_reset("factory"):
        response = client.post(
            "/api/eval/runs/stream",
            headers={**headers, "x-provider": "ollama", "x-model": "llama3.2"},
            json=RUN_REQUEST,
        )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "reset_in_progress"
    assert fake_service.prepare_calls == 0
    with session_scope() as session:
        after = session.execute(select(EvalRun.id)).scalars().all()
    assert after == before
