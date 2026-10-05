"""TDD tracer-bullet tests for the isolated one-attempt RunService."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessageChunk
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session

from app.agent.tutor_attempt import TutorCandidate
from app.db.models import Base, EvalRun
from app.eval.learning_run.contracts import ScorerExecutionDraft, canonical_hash
from app.eval.learning_run.registry import TaskRegistry
from app.eval.learning_run.repositories import (
    EvalRunRepository,
    EvalScoreSetRepository,
    EvalScorerExecutionRepository,
)
from app.eval.learning_run.runner import RunnerOperationalError


def _service_api():
    try:
        from app.eval.learning_run.service import (
            EvalModelConnection,
            RunRequestError,
            RunService,
            RunServiceResult,
            SystemClock,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover - intentional RED guard
        pytest.fail(f"learning-run service module is missing: {exc}", pytrace=False)
    return EvalModelConnection, RunRequestError, RunService, RunServiceResult, SystemClock


REGISTRY = TaskRegistry.load_default()
ANSWERABLE = REGISTRY.task_cases["tgqa-001"]
SCORER = REGISTRY.scorer


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeTutorLLM:
    async def astream(self, _messages):
        if False:  # pragma: no cover - protocol-only fake
            yield None


class FakeScorerLLM:
    def __init__(self, *, malformed: bool = False, delay: float = 0.0):
        self.malformed = malformed
        self.delay = delay
        self.calls = 0

    async def ainvoke(self, _messages):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.malformed:
            return SimpleNamespace(content="not-json")
        payload = {
            "groundedness": 4,
            "citation_entailment": 4,
            "coverage": 4,
            "reasoning": "frozen fake scorer result",
            "findings": [],
        }
        return SimpleNamespace(content=json.dumps(payload))


class RecordingTutorRunner:
    def __init__(self, candidate: TutorCandidate, *, error: BaseException | None = None, delay: float = 0.0):
        self.candidate = candidate
        self.error = error
        self.delay = delay
        self.calls = 0

    async def run(self, **_kwargs):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.candidate


class ForbiddenDependency:
    def __getattr__(self, name):
        raise AssertionError(f"forbidden dependency called: {name}")


class GuardedRegistry:
    """Expose only registry resolution; orchestration dependencies explode."""

    def __init__(self, registry):
        self._registry = registry
        self.router = ForbiddenDependency()
        self.runtime_judge = ForbiddenDependency()
        self.memory = ForbiddenDependency()
        self.chat_repository = ForbiddenDependency()
        self.global_retriever = ForbiddenDependency()

    def resolve_run(self, **kwargs):
        return self._registry.resolve_run(**kwargs)

    def scorer_for(self, version):
        return self._registry.scorer_for(version)

    def scorer_document(self, version):
        return self._registry.scorer_document(version)

    @property
    def task_cases(self):
        return self._registry.task_cases

    @property
    def experiment(self):
        return self._registry.experiment

    @property
    def scorers(self):
        return self._registry.scorers


def _candidate(*, empty: bool = False) -> TutorCandidate:
    evidence = [] if empty else [
        {
            "chunk_id": "tgqa-c01-rrf",
            "content": "Reciprocal rank fusion combines ranked lists.",
            "source": "learning-run-notes.md",
            "page": 1,
        }
    ]
    return TutorCandidate(
        answer="I don't know; the sources do not contain that fact." if empty else "Reciprocal rank fusion combines ranked lists [1].",
        citations=[] if empty else [{
            "chunk_id": evidence[0]["chunk_id"],
            "source": evidence[0]["source"],
            "page": evidence[0]["page"],
            "span_start": 0,
            "span_end": len(evidence[0]["content"]),
        }],
        evidence=evidence,
        formatted_context="" if empty else "[1] learning-run-notes.md p.1: Reciprocal rank fusion combines ranked lists.",
        usage={"input_tokens": 4, "output_tokens": 8, "total_tokens": 12},
        trace=[{"stage": "tutor", "event": "complete"}],
    )


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    with Session(engine) as db_session:
        yield db_session


def _connection(
    *,
    scorer_llm=None,
    fingerprint: str = "a" * 64,
    tutor_provider: str | None = None,
    tutor_llm=None,
):
    EvalModelConnection, _, _, _, _ = _service_api()
    controls = REGISTRY.experiment.variants["tutor-v3"]
    scorer_config = dict(SCORER.model_config)
    scorer_parameters = {
        key: value for key, value in scorer_config.items() if key not in {"provider", "model"}
    }
    return EvalModelConnection(
        tutor_provider=tutor_provider or controls["provider"],
        tutor_model=controls["model"],
        tutor_parameters=dict(controls["parameters"]),
        tutor_llm=tutor_llm or FakeTutorLLM(),
        scorer_provider=scorer_config["provider"],
        scorer_model=scorer_config["model"],
        scorer_parameters=scorer_parameters,
        scorer_llm=scorer_llm or FakeScorerLLM(),
        connection_fingerprint=fingerprint,
    )


def _service(session, runner, *, scorer_llm=None, clock=None, scoring_factory=None):
    _, _, RunService, _, _ = _service_api()
    return RunService(
        registry=REGISTRY,
        tutor_runner=runner,
        runs=EvalRunRepository(session),
        score_sets=EvalScoreSetRepository(session),
        scorer_executions=EvalScorerExecutionRepository(session),
        clock=clock or FakeClock(),
        code_revision="test-revision",
        scoring_service_factory=scoring_factory,
    )


@pytest.mark.asyncio
async def test_run_service_freezes_candidate_then_scores_four_auditable_executions(session):
    runner = RecordingTutorRunner(_candidate())
    scorer_llm = FakeScorerLLM()
    service = _service(session, runner, scorer_llm=scorer_llm)

    result = await service.run(
        experiment_id="tutor-prompt-regression-v1",
        task_case_id="tgqa-001",
        variant_id="tutor-v3",
        run_profile="evaluation",
        connection=_connection(scorer_llm=scorer_llm),
        events=[],
    )

    _, _, _, RunServiceResult, _ = _service_api()
    assert isinstance(result, RunServiceResult)
    assert runner.calls == 1
    assert result.run.lifecycle == "finished"
    assert result.run.outcome == "success"
    assert result.run.artifact_hash == canonical_hash(result.run.candidate_artifact_json)
    assert result.score_set is not None
    assert result.score_set.artifact_input_hash == result.run.artifact_hash
    executions = EvalScorerExecutionRepository(session).list_verified(result.score_set.id)
    assert len(executions) == 4
    assert [row.scorer_id for row in executions] == [
        "retrieval-integrity",
        "citation-integrity",
        "expected-refusal-observation",
        "grounded-quality-rubric",
    ]
    assert all(row.input_hash == result.run.artifact_hash for row in executions)
    assert executions[0].output_json["result"] is not None
    assert "findings" in executions[0].output_json
    assert executions[3].output_json["result"]["groundedness"] == 4
    assert result.score_set.aggregate_scores_json == {"groundedness": 4, "citation_entailment": 4, "coverage": 4}


class ExplodingDependency:
    def __getattr__(self, name):
        raise AssertionError("Tutor must not run")


@pytest.mark.asyncio
async def test_rescore_reads_frozen_artifact_and_never_calls_tutor(session):
    runner = RecordingTutorRunner(_candidate())
    scorer_llm = FakeScorerLLM()
    service = _service(session, runner, scorer_llm=scorer_llm)
    first = await service.run(
        experiment_id="tutor-prompt-regression-v1",
        task_case_id="tgqa-001",
        variant_id="tutor-v3",
        run_profile="evaluation",
        connection=_connection(scorer_llm=scorer_llm),
        events=[],
    )
    service.tutor_runner = ExplodingDependency()
    service.attempt_engine = ExplodingDependency()
    events = []
    new_score_set = await service.rescore(
        run_id=first.run.id,
        scorer_version="hybrid-v2",
        connection=_connection(scorer_llm=FakeScorerLLM()),
        events=events,
    )
    assert new_score_set.run_id == first.run.id
    assert new_score_set.artifact_input_hash == first.run.artifact_hash
    assert new_score_set.scorer_version == "hybrid-v2"
    assert new_score_set.scorer_definition_hash == REGISTRY.scorer_for("hybrid-v2").definition_hash
    assert len(service.score_sets.list_for_run(first.run.id)) == 2
    assert events[0]["type"] == "score_set_created"
    assert events[0]["score_set_id"] == new_score_set.id
    assert runner.calls == 1


@pytest.mark.asyncio
async def test_prepare_rescore_prefers_frozen_task_snapshot(session):
    runner = RecordingTutorRunner(_candidate())
    scorer_llm = FakeScorerLLM()
    service = _service(session, runner, scorer_llm=scorer_llm)
    first = await service.run(
        experiment_id="tutor-prompt-regression-v1",
        task_case_id="tgqa-001",
        variant_id="tutor-v3",
        run_profile="evaluation",
        connection=_connection(scorer_llm=scorer_llm),
        events=[],
    )
    manifest = dict(first.run.manifest_json)
    snapshot = dict(manifest["task_snapshot"])
    snapshot["question"] = "frozen historical question"
    manifest["task_snapshot"] = snapshot
    first.run.manifest_json = manifest
    first.run.manifest_hash = canonical_hash(manifest)
    session.add(first.run)
    session.commit()
    prepared = service.prepare_rescore(
        run_id=first.run.id,
        scorer_version="hybrid-v2",
        connection=_connection(scorer_llm=FakeScorerLLM()),
    )
    assert prepared.task.question == "frozen historical question"


@pytest.mark.asyncio
async def test_rescore_failure_terminalizes_score_set(session):
    runner = RecordingTutorRunner(_candidate())
    scorer_llm = FakeScorerLLM()
    service = _service(session, runner, scorer_llm=scorer_llm)
    first = await service.run(
        experiment_id="tutor-prompt-regression-v1",
        task_case_id="tgqa-001",
        variant_id="tutor-v3",
        run_profile="evaluation",
        connection=_connection(scorer_llm=scorer_llm),
        events=[],
    )

    def explode(*_args, **_kwargs):
        raise RuntimeError("scorer exploded")

    service.scoring_service_factory = explode
    with pytest.raises(RuntimeError, match="scorer exploded"):
        await service.rescore(
            run_id=first.run.id,
            scorer_version="hybrid-v2",
            connection=_connection(scorer_llm=FakeScorerLLM()),
        )
    rows = service.score_sets.list_for_run(first.run.id)
    assert rows[-1].status == "failed"
    assert rows[-1].quality_verdict == "inconclusive"


@pytest.mark.asyncio
async def test_service_uses_only_injected_eval_collaborators_and_never_business_dependencies(session):
    runner = RecordingTutorRunner(_candidate())
    service = _service(session, runner)
    service.registry = GuardedRegistry(REGISTRY)

    result = await service.run(
        experiment_id="tutor-prompt-regression-v1",
        task_case_id="tgqa-001",
        variant_id="tutor-v3",
        run_profile="evaluation",
        connection=_connection(),
    )

    assert result.run.outcome == "success"
    assert runner.calls == 1


@pytest.mark.asyncio
async def test_service_event_sink_failure_terminalizes_run_without_scoring(session):
    runner = RecordingTutorRunner(_candidate())
    service = _service(session, runner)

    def rejecting_events(_event):
        raise RuntimeError("api_key=secret-value")

    result = await service.run(
        experiment_id="tutor-prompt-regression-v1",
        task_case_id="tgqa-001",
        variant_id="tutor-v3",
        run_profile="evaluation",
        connection=_connection(),
        events=rejecting_events,
    )

    assert result.run.outcome == "system_failed"
    assert result.run.operational_error_json["code"] == "harness_internal_error"
    assert result.run.operational_error_json["stage"] == "events"
    assert result.score_set is None
    assert "secret-value" not in json.dumps(result.run.operational_error_json)


@pytest.mark.asyncio
async def test_service_event_sink_failure_after_success_cas_does_not_rewrite_run(session):
    runner = RecordingTutorRunner(_candidate())
    service = _service(session, runner)

    def rejecting_events(event):
        if event.get("type") == "score_set_finished":
            raise RuntimeError("api_key=secret-value")

    result = await service.run(
        experiment_id="tutor-prompt-regression-v1",
        task_case_id="tgqa-001",
        variant_id="tutor-v3",
        run_profile="evaluation",
        connection=_connection(),
        events=rejecting_events,
    )

    assert result.run.lifecycle == "finished"
    assert result.run.outcome == "success"
    assert result.run.artifact_hash
    assert "secret-value" not in json.dumps(result.run.operational_error_json)


@pytest.mark.asyncio
async def test_success_cas_failure_does_not_emit_success_terminal_event(session):
    runner = RecordingTutorRunner(_candidate())
    service = _service(session, runner)
    events: list[dict[str, Any]] = []

    def fail_success_cas(*_args, **_kwargs):
        raise RuntimeError("run success CAS unavailable")

    original_finalize_success = service.runs.finalize_success
    service.runs.finalize_success = fail_success_cas
    try:
        result = await service.run(
            experiment_id="tutor-prompt-regression-v1",
            task_case_id="tgqa-001",
            variant_id="tutor-v3",
            run_profile="evaluation",
            connection=_connection(),
            events=events,
        )
    finally:
        service.runs.finalize_success = original_finalize_success

    assert result.run.lifecycle == "finished"
    assert result.run.outcome != "success"
    assert not any(
        event.get("type") == "run_finished" and event.get("outcome") == "success"
        for event in events
    )


@pytest.mark.asyncio
async def test_terminal_event_sink_failure_after_success_cas_does_not_rewrite_run(session):
    runner = RecordingTutorRunner(_candidate())
    service = _service(session, runner)

    def rejecting_terminal_event(event):
        if event.get("type") == "run_finished":
            raise RuntimeError("terminal event sink unavailable")

    result = await service.run(
        experiment_id="tutor-prompt-regression-v1",
        task_case_id="tgqa-001",
        variant_id="tutor-v3",
        run_profile="evaluation",
        connection=_connection(),
        events=rejecting_terminal_event,
    )

    assert result.run.lifecycle == "finished"
    assert result.run.outcome == "success"
    assert "terminal event sink unavailable" not in json.dumps(result.run.operational_error_json)


def test_service_module_has_no_production_orchestration_imports():
    from pathlib import Path

    source = (Path(__file__).parents[2] / "app/eval/learning_run/service.py").read_text()
    forbidden = (
        "agent.graph",
        "agent.judge",
        "judge_response",
        "Router",
        "Memory",
        "Chat",
        "FastAPI",
        "global_retriever",
        "get_chat_model",
    )
    assert not [token for token in forbidden if token in source]


@pytest.mark.asyncio
async def test_manifest_is_complete_frozen_and_privacy_safe_without_connection_secrets(session):
    runner = RecordingTutorRunner(_candidate())
    service = _service(session, runner)
    result = await service.run(
        experiment_id="tutor-prompt-regression-v1",
        task_case_id="tgqa-001",
        variant_id="tutor-v3",
        run_profile="evaluation",
        connection=_connection(),
    )

    manifest = result.run.manifest_json
    assert manifest["task_snapshot"]["id"] == "tgqa-001"
    assert manifest["prompt_text"] == REGISTRY.prompts["tutor-v3"].text
    assert manifest["corpus_snapshot"]["aggregate_hash"] == REGISTRY.corpus.aggregate_hash
    assert manifest["scorer_snapshot"]["version"] == "hybrid-v1"
    assert manifest["connection_fingerprint"] == "a" * 64
    dumped = json.dumps(manifest, sort_keys=True)
    assert "api_key" not in dumped.lower()
    assert "authorization" not in dumped.lower()
    assert "http://" not in dumped.lower()
    assert "https://" not in dumped.lower()


@pytest.mark.asyncio
async def test_manifest_scorer_snapshot_is_exact_frozen_bundle_payload_plus_hash(session):
    service = _service(session, RecordingTutorRunner(_candidate()))
    result = await service.run(
        experiment_id="tutor-prompt-regression-v1",
        task_case_id="tgqa-001",
        variant_id="tutor-v3",
        run_profile="evaluation",
        connection=_connection(),
    )

    expected = SCORER.payload()
    expected["definition_hash"] = SCORER.definition_hash
    snapshot = result.run.manifest_json["scorer_snapshot"]
    assert snapshot == expected
    assert snapshot["required_dimensions_by_case_type"] == {
        key: list(value)
        for key, value in SCORER.required_dimensions_by_case_type.items()
    }
    assert snapshot["calibration_hash"] == SCORER.calibration_hash


@pytest.mark.asyncio
async def test_connection_mismatch_is_request_error_before_any_run_row(session):
    _, RunRequestError, _, _, _ = _service_api()
    service = _service(session, RecordingTutorRunner(_candidate()))

    with pytest.raises(RunRequestError) as caught:
        await service.run(
            experiment_id="tutor-prompt-regression-v1",
            task_case_id="tgqa-001",
            variant_id="tutor-v3",
            run_profile="evaluation",
            connection=_connection(tutor_provider="wrong-provider"),
        )

    assert caught.value.code == "evaluation_config_mismatch"
    assert session.execute(select(EvalRun)).scalars().all() == []


@pytest.mark.asyncio
async def test_runner_operational_failure_persists_typed_budget_without_raw_exception(session):
    runner = RecordingTutorRunner(
        _candidate(),
        error=RunnerOperationalError(
            stage="generation",
            code="model_unavailable",
            sanitized_message="tutor model unavailable (RuntimeError)",
            retryable=True,
        ),
    )
    service = _service(session, runner)
    result = await service.run(
        experiment_id="tutor-prompt-regression-v1",
        task_case_id="tgqa-001",
        variant_id="tutor-v3",
        run_profile="evaluation",
        connection=_connection(),
    )

    assert result.run.outcome == "system_failed"
    assert result.run.operational_error_json["stage"] == "generation"
    assert result.run.operational_error_json["retryable"] is True
    assert "spent_budget" in result.run.operational_error_json
    assert "RuntimeError" in result.run.operational_error_json["message"]
    assert "api_key" not in json.dumps(result.run.operational_error_json).lower()


@pytest.mark.asyncio
async def test_malformed_scorer_is_partial_inconclusive_but_run_still_succeeds(session):
    runner = RecordingTutorRunner(_candidate())
    malformed = FakeScorerLLM(malformed=True)
    service = _service(session, runner)
    result = await service.run(
        experiment_id="tutor-prompt-regression-v1",
        task_case_id="tgqa-001",
        variant_id="tutor-v3",
        run_profile="evaluation",
        connection=_connection(scorer_llm=malformed),
    )

    assert runner.calls == 1
    assert result.run.outcome == "success"
    assert result.score_set.status == "partial"
    assert result.score_set.quality_verdict == "inconclusive"
    rows = EvalScorerExecutionRepository(session).list_verified(result.score_set.id)
    assert len(rows) == 4
    assert rows[-1].status == "failed"
    assert rows[-1].operational_error_code == "scorer_parse_error"


@pytest.mark.asyncio
async def test_scoring_timeout_fills_missing_components_and_derives_deterministic_findings(session):
    runner = RecordingTutorRunner(_candidate())
    entered = asyncio.Event()

    class PartialThenBlockingScoring:
        async def score(self, *, on_execution, candidate, **_kwargs):
            entered.set()
            on_execution(
                ScorerExecutionDraft(
                    component_id="retrieval-integrity",
                    component_version="v1",
                    scorer_id="retrieval-integrity",
                    scorer_version="v1",
                    status="success",
                    input_hash=canonical_hash(candidate.to_dict()),
                    output={"evidence_count": 1},
                    findings=({
                        "code": "retrieval_empty",
                        "severity": "noncritical",
                        "message": "no exact evidence was retrieved",
                    },),
                )
            )
            await asyncio.Event().wait()

    definition_budget = dict(REGISTRY.experiment.budget)
    definition_budget["hybrid_scoring_seconds"] = 0.01

    def factory(_llm, *, timeout_seconds):
        assert timeout_seconds == 0.01
        return PartialThenBlockingScoring()

    service = _service(session, runner, scoring_factory=factory)
    original_resolve = service.registry.resolve_run
    service.registry.resolve_run = lambda **kwargs: replace(
        original_resolve(**kwargs), budget=definition_budget
    )
    try:
        result = await service.run(
            experiment_id="tutor-prompt-regression-v1",
            task_case_id="tgqa-001",
            variant_id="tutor-v3",
            run_profile="evaluation",
            connection=_connection(),
        )
    finally:
        service.registry.resolve_run = original_resolve

    assert entered.is_set()
    assert result.run.outcome == "success"
    assert result.score_set.status == "partial"
    assert result.score_set.quality_verdict == "inconclusive"
    assert any(item["code"] == "retrieval_empty" for item in result.score_set.findings_json)
    rows = EvalScorerExecutionRepository(session).list_verified(result.score_set.id)
    assert [row.scorer_id for row in rows] == [component.component_id for component in SCORER.components]
    assert rows[0].status == "success"
    assert all(row.status == "failed" for row in rows[1:])
    assert all(row.operational_error_code == "scorer_timeout" for row in rows[1:])
    assert all(row.latency_ms is None for row in rows[1:])
    assert all(row.output_json is None and row.usage_json is None for row in rows[1:])


@pytest.mark.asyncio
async def test_scoring_timeout_with_no_emissions_is_failed_not_count_based_partial(session):
    runner = RecordingTutorRunner(_candidate())

    class BlockingScoring:
        async def score(self, **_kwargs):
            await asyncio.Event().wait()

    definition_budget = dict(REGISTRY.experiment.budget)
    definition_budget["hybrid_scoring_seconds"] = 0.01

    service = _service(
        session,
        runner,
        scoring_factory=lambda _llm, **_kwargs: BlockingScoring(),
    )
    original_resolve = service.registry.resolve_run
    service.registry.resolve_run = lambda **kwargs: replace(
        original_resolve(**kwargs), budget=definition_budget
    )
    try:
        result = await service.run(
            experiment_id="tutor-prompt-regression-v1",
            task_case_id="tgqa-001",
            variant_id="tutor-v3",
            run_profile="evaluation",
            connection=_connection(),
        )
    finally:
        service.registry.resolve_run = original_resolve

    assert result.run.outcome == "success"
    assert result.score_set.status == "failed"
    assert result.score_set.quality_verdict == "inconclusive"
    rows = EvalScorerExecutionRepository(session).list_verified(result.score_set.id)
    assert len(rows) == len(SCORER.components)
    assert all(row.status == "failed" for row in rows)


@pytest.mark.asyncio
async def test_callback_persistence_failure_preserves_candidate_and_rows_and_fails_run(session):
    runner = RecordingTutorRunner(_candidate())
    service = _service(session, runner)
    original_append = service.scorer_executions.append_draft
    calls = 0

    def fail_on_second(score_set_id, draft):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("database callback failure")
        return original_append(score_set_id, draft)

    service.scorer_executions.append_draft = fail_on_second
    result = await service.run(
        experiment_id="tutor-prompt-regression-v1",
        task_case_id="tgqa-001",
        variant_id="tutor-v3",
        run_profile="evaluation",
        connection=_connection(),
    )

    assert runner.calls == 1
    assert result.run.outcome == "system_failed"
    assert result.run.artifact_hash
    assert result.score_set.status in {"partial", "failed"}
    assert result.score_set.quality_verdict != "pass"
    rows = EvalScorerExecutionRepository(session).list_verified(result.score_set.id)
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_total_deadline_before_artifact_fails_budget_and_does_not_call_scoring(session):
    runner = RecordingTutorRunner(_candidate(), delay=0.05)
    definition_budget = dict(REGISTRY.experiment.budget)
    definition_budget["total_seconds"] = 0.001
    service = _service(session, runner)
    original_resolve = service.registry.resolve_run
    service.registry.resolve_run = lambda **kwargs: replace(
        original_resolve(**kwargs), budget=definition_budget
    )

    try:
        result = await service.run(
            experiment_id="tutor-prompt-regression-v1",
            task_case_id="tgqa-001",
            variant_id="tutor-v3",
            run_profile="evaluation",
            connection=_connection(),
        )
    finally:
        service.registry.resolve_run = original_resolve

    assert result.run.outcome == "budget_exceeded"
    assert result.run.candidate_artifact_json is None
    assert result.score_set is None


@pytest.mark.asyncio
async def test_blocking_loader_cannot_block_total_deadline_before_candidate_freeze(session):
    from app.eval.learning_run.runner import TutorRunner

    class BlockingLoader:
        def __init__(self):
            self.calls = 0

        def load(self, *, snapshot):
            self.calls += 1
            time.sleep(0.15)
            return object()

    class NeverCalledEngine:
        def __init__(self):
            self.calls = 0

        async def answer(self, **_kwargs):
            self.calls += 1
            return _candidate()

    loader = BlockingLoader()
    engine = NeverCalledEngine()
    runner = TutorRunner(corpus_loader=loader, attempt_engine=engine)
    definition_budget = dict(REGISTRY.experiment.budget)
    definition_budget["total_seconds"] = 0.01
    service = _service(session, runner)
    original_resolve = service.registry.resolve_run
    service.registry.resolve_run = lambda **kwargs: replace(
        original_resolve(**kwargs), budget=definition_budget
    )
    started = time.monotonic()
    try:
        result = await service.run(
            experiment_id="tutor-prompt-regression-v1",
            task_case_id="tgqa-001",
            variant_id="tutor-v3",
            run_profile="evaluation",
            connection=_connection(),
        )
        elapsed = time.monotonic() - started
        await asyncio.sleep(0.17)
    finally:
        service.registry.resolve_run = original_resolve

    assert elapsed < 0.10
    assert result.run.outcome == "budget_exceeded"
    assert result.run.candidate_artifact_json is None
    assert result.score_set is None
    assert loader.calls == 1
    assert engine.calls == 0


@pytest.mark.asyncio
async def test_total_deadline_closes_runner_event_gate_for_late_child(session):
    late_events: list[dict[str, Any]] = []

    class LateRunner:
        calls = 0

        async def run(self, *, events, **_kwargs):
            self.calls += 1
            try:
                await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                events({"type": "token", "text": "late"})
                raise

    runner = LateRunner()
    definition_budget = dict(REGISTRY.experiment.budget)
    definition_budget["total_seconds"] = 0.001
    service = _service(session, runner)
    original_resolve = service.registry.resolve_run
    service.registry.resolve_run = lambda **kwargs: replace(
        original_resolve(**kwargs), budget=definition_budget
    )
    try:
        result = await service.run(
            experiment_id="tutor-prompt-regression-v1",
            task_case_id="tgqa-001",
            variant_id="tutor-v3",
            run_profile="evaluation",
            connection=_connection(),
            events=late_events,
        )
    finally:
        service.registry.resolve_run = original_resolve

    assert result.run.outcome == "budget_exceeded"
    assert {event["type"] for event in late_events} <= {"run_created", "stage_started"}


@pytest.mark.asyncio
async def test_total_deadline_after_artifact_keeps_artifact_and_blocks_late_callback(session):
    runner = RecordingTutorRunner(_candidate())

    class SlowScoring:
        async def score(self, *, on_execution, **_kwargs):
            await asyncio.sleep(0.05)
            return None

    definition_budget = dict(REGISTRY.experiment.budget)
    definition_budget["total_seconds"] = 0.001

    def factory(_llm, *, timeout_seconds):
        assert timeout_seconds == 25
        return SlowScoring()

    service = _service(session, runner, scoring_factory=factory)
    original_resolve = service.registry.resolve_run
    service.registry.resolve_run = lambda **kwargs: replace(
        original_resolve(**kwargs), budget=definition_budget
    )

    try:
        result = await service.run(
            experiment_id="tutor-prompt-regression-v1",
            task_case_id="tgqa-001",
            variant_id="tutor-v3",
            run_profile="evaluation",
            connection=_connection(),
        )
    finally:
        service.registry.resolve_run = original_resolve

    assert result.run.outcome == "budget_exceeded"
    assert result.run.artifact_hash
    assert result.score_set is not None
    assert result.score_set.quality_verdict == "inconclusive"
    rows = EvalScorerExecutionRepository(session).list_verified(result.score_set.id)
    assert len(rows) == len(SCORER.components)
    assert all(row.status == "failed" for row in rows)


@pytest.mark.asyncio
async def test_hybrid_stage_deadline_is_hard_and_persists_failed_semantic_execution(session):
    runner = RecordingTutorRunner(_candidate())
    semantic_component = next(
        component for component in SCORER.components if component.kind == "llm"
    )

    class CancellationResistantScoring:
        async def score(self, *, on_execution, candidate, **_kwargs):
            try:
                await asyncio.sleep(0.20)
            except asyncio.CancelledError:
                await asyncio.sleep(0.05)
                on_execution(
                    ScorerExecutionDraft(
                        component_id=semantic_component.component_id,
                        component_version=semantic_component.version,
                        scorer_id=semantic_component.component_id,
                        scorer_version=semantic_component.version,
                        status="success",
                        input_hash=canonical_hash(candidate.to_dict()),
                        output={"groundedness": 5},
                    )
                )
                return None
            return None

    definition_budget = dict(REGISTRY.experiment.budget)
    definition_budget["hybrid_scoring_seconds"] = 0.01

    def factory(_llm, *, timeout_seconds):
        assert timeout_seconds == 0.01
        return CancellationResistantScoring()

    service = _service(session, runner, scoring_factory=factory)
    original_resolve = service.registry.resolve_run
    service.registry.resolve_run = lambda **kwargs: replace(
        original_resolve(**kwargs), budget=definition_budget
    )
    started = time.monotonic()
    try:
        result = await service.run(
            experiment_id="tutor-prompt-regression-v1",
            task_case_id="tgqa-001",
            variant_id="tutor-v3",
            run_profile="evaluation",
            connection=_connection(),
        )
        elapsed = time.monotonic() - started
        await asyncio.sleep(0.07)
    finally:
        service.registry.resolve_run = original_resolve

    assert elapsed < 0.12
    assert result.run.outcome == "success"
    assert result.score_set.status in {"partial", "failed"}
    assert result.score_set.quality_verdict == "inconclusive"
    assert result.score_set.operational_error_code == "scorer_timeout"
    rows = EvalScorerExecutionRepository(session).list_verified(result.score_set.id)
    assert len(rows) == len(SCORER.components)
    assert [row.scorer_id for row in rows] == [component.component_id for component in SCORER.components]
    assert all(row.input_hash == result.run.artifact_hash for row in rows)
    assert all(row.status == "failed" for row in rows)
    assert all(row.operational_error_code == "scorer_timeout" for row in rows)
    assert all(row.output_json is None for row in rows)
    assert all(row.usage_json is None for row in rows)
    assert all(row.latency_ms is None for row in rows)
    assert all(row.operational_error_message == "scorer timed out" for row in rows)


@pytest.mark.asyncio
async def test_cancellation_resistant_scorer_cannot_append_after_hard_deadline(session):
    runner = RecordingTutorRunner(_candidate())

    class CancellationResistantScoring:
        async def score(self, *, on_execution, **kwargs):
            try:
                await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                await asyncio.sleep(0)
                on_execution(
                    ScorerExecutionDraft(
                        component_id="late-component",
                        component_version="v1",
                        scorer_id="late-component",
                        scorer_version="v1",
                        status="success",
                        input_hash=canonical_hash(kwargs["candidate"].to_dict()),
                        output={"late": True},
                    )
                )
                return None
            return None

    definition_budget = dict(REGISTRY.experiment.budget)
    definition_budget["total_seconds"] = 0.001

    def factory(_llm, *, timeout_seconds):
        return CancellationResistantScoring()

    service = _service(session, runner, scoring_factory=factory)
    original_resolve = service.registry.resolve_run
    service.registry.resolve_run = lambda **kwargs: replace(
        original_resolve(**kwargs), budget=definition_budget
    )
    try:
        result = await service.run(
            experiment_id="tutor-prompt-regression-v1",
            task_case_id="tgqa-001",
            variant_id="tutor-v3",
            run_profile="evaluation",
            connection=_connection(),
        )
        await asyncio.sleep(0.01)
    finally:
        service.registry.resolve_run = original_resolve

    assert result.run.outcome == "budget_exceeded"
    rows = EvalScorerExecutionRepository(session).list_verified(result.score_set.id)
    assert len(rows) == len(SCORER.components)
    assert all(row.status == "failed" for row in rows)
    assert all(row.operational_error_code == "budget_exceeded" for row in rows)


@pytest.mark.asyncio
async def test_score_set_terminalization_failure_raises_but_keeps_frozen_candidate(session):
    runner = RecordingTutorRunner(_candidate())
    service = _service(session, runner)
    real_finalize = service.score_sets.finalize_once

    def fail_terminalization(*_args, **_kwargs):
        raise RuntimeError("score set storage unavailable")

    service.score_sets.finalize_once = fail_terminalization
    with pytest.raises(RuntimeError, match="score set storage unavailable"):
        await service.run(
            experiment_id="tutor-prompt-regression-v1",
            task_case_id="tgqa-001",
            variant_id="tutor-v3",
            run_profile="evaluation",
            connection=_connection(),
        )

    rows = session.execute(select(EvalRun)).scalars().all()
    assert len(rows) == 1
    assert rows[0].lifecycle == "finished"
    assert rows[0].outcome == "system_failed"
    assert rows[0].operational_error_json["code"] == "harness_internal_error"
    assert rows[0].artifact_hash
    assert rows[0].candidate_artifact_json is not None
    service.score_sets.finalize_once = real_finalize


@pytest.mark.asyncio
async def test_zero_retrieval_is_successful_candidate_with_quality_finding_not_operational_failure(session):
    runner = RecordingTutorRunner(_candidate(empty=True))
    service = _service(session, runner)
    result = await service.run(
        experiment_id="tutor-prompt-regression-v1",
        task_case_id="tgqa-001",
        variant_id="tutor-v3",
        run_profile="evaluation",
        connection=_connection(),
    )

    assert result.run.outcome == "success"
    assert result.score_set.quality_verdict == "fail"
    assert any(item["code"] == "retrieval_empty" for item in result.score_set.findings_json)


# ---------------------------------------------------------------------------
# Batch C — a real TutorRunner + real TutorAttemptEngine behind the RunService,
# with a tmp SQLite database and a fake scorer.
# ---------------------------------------------------------------------------

_EMPTY_CONTENT_MESSAGE = "LLM response text has no non-whitespace body"
_MALFORMED_CONTENT_MESSAGE = "LLM response content is not a supported text shape"
_SENTINEL = "SENTINEL-DO-NOT-LEAK"

_CORPUS_CHUNK = {
    "chunk_id": "tgqa-c01-rrf",
    "content": "Reciprocal rank fusion combines ranked lists.",
    "source": "learning-run-notes.md",
    "page": 1,
    "score": 0.9,
}
_CANDIDATE_ANSWER = "Reciprocal rank fusion combines ranked lists [1]."


class _RealPathRetriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = list(chunks)
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        self.queries.append((query, top_k))
        return list(self.chunks[:top_k])

    def close(self) -> None:
        return None


class RealPathLoader:
    """Fake isolated loader; the Attempt itself is the real engine."""

    def __init__(self, chunks: list[dict] | None = None):
        self.snapshots: list = []
        self.retriever = _RealPathRetriever(chunks or [dict(_CORPUS_CHUNK)])

    def load(self, *, snapshot, stop=None, deadline=None):
        del stop, deadline
        self.snapshots.append(snapshot)
        return self.retriever


class BlockChunkTutorLLM:
    """One streaming chunk per element; content may be blocks or a bare string."""

    def __init__(self, contents: list, *, usage: dict | None = None):
        self.contents = list(contents)
        self.usage = usage
        self.astream_calls = 0

    async def astream(self, messages, **_kwargs):
        self.astream_calls += 1
        for index, content in enumerate(self.contents):
            chunk = AIMessageChunk(content=content)
            if self.usage is not None and index == len(self.contents) - 1:
                chunk.usage_metadata = dict(self.usage)
            yield chunk


def _real_runner(loader):
    from app.agent.tutor_attempt import TutorAttemptEngine
    from app.eval.learning_run.corpus import CorpusMaterializerController
    from app.eval.learning_run.runner import TutorRunner

    controller = CorpusMaterializerController(loader)
    runner = TutorRunner(
        corpus_loader=loader,
        attempt_engine=TutorAttemptEngine(),
        materializer_controller=controller,
    )
    return runner, controller


@pytest.mark.asyncio
async def test_real_attempt_service_freezes_verifiable_candidate_and_scores_once(session):
    loader = RealPathLoader()
    runner, controller = _real_runner(loader)
    scorer_llm = FakeScorerLLM()
    service = _service(session, runner, scorer_llm=scorer_llm)
    llm = BlockChunkTutorLLM(
        [
            [{"type": "thinking", "thinking": _SENTINEL}],
            [{"type": "text", "text": "Reciprocal rank fusion "}, "combines ranked lists [1]."],
        ],
        usage={"input_tokens": 4, "output_tokens": 8, "total_tokens": 12},
    )
    events: list[dict] = []

    try:
        result = await service.run(
            experiment_id="tutor-prompt-regression-v1",
            task_case_id="tgqa-001",
            variant_id="tutor-v3",
            run_profile="evaluation",
            connection=_connection(scorer_llm=scorer_llm, tutor_llm=llm),
            events=events,
        )
    finally:
        controller.shutdown(wait=False)

    assert len(loader.snapshots) == 1
    assert llm.astream_calls == 1, "evaluation profile must run exactly one Attempt"
    assert loader.retriever.queries == [(ANSWERABLE.question, 5)]

    assert result.run.lifecycle == "finished"
    assert result.run.outcome == "success"
    assert result.run.artifact_hash == canonical_hash(result.run.candidate_artifact_json)

    artifact_payload = result.run.candidate_artifact_json
    assert artifact_payload["answer"] == _CANDIDATE_ANSWER
    assert artifact_payload["citations"] == [
        {
            "chunk_id": "tgqa-c01-rrf",
            "source": "learning-run-notes.md",
            "page": 1,
            "span_start": 0,
            "span_end": len(_CORPUS_CHUNK["content"]),
        }
    ]
    assert [chunk["chunk_id"] for chunk in artifact_payload["exact_evidence"]] == [
        "tgqa-c01-rrf"
    ]
    assert artifact_payload["usage"] == {
        "input_tokens": 4,
        "output_tokens": 8,
        "total_tokens": 12,
    }
    assert artifact_payload["budget"]
    assert artifact_payload["trace"]
    assert _SENTINEL not in json.dumps(artifact_payload)

    assert result.score_set is not None
    assert result.score_set.artifact_input_hash == result.run.artifact_hash
    executions = EvalScorerExecutionRepository(session).list_verified(result.score_set.id)
    assert len(executions) == 4
    assert scorer_llm.calls >= 1

    # Every streamed token is answer text, never a raw block payload.
    token_events = [event for event in events if event.get("type") == "token"]
    assert [event["text"] for event in token_events] == [_CANDIDATE_ANSWER]
    assert all(isinstance(event["text"], str) for event in token_events)


@pytest.mark.asyncio
async def test_real_attempt_without_usable_body_fails_run_without_candidate_or_scoring(session):
    loader = RealPathLoader()
    runner, controller = _real_runner(loader)
    scorer_llm = FakeScorerLLM()
    service = _service(session, runner, scorer_llm=scorer_llm)
    llm = BlockChunkTutorLLM([[{"type": "thinking", "thinking": _SENTINEL}]])
    events: list[dict] = []

    try:
        result = await service.run(
            experiment_id="tutor-prompt-regression-v1",
            task_case_id="tgqa-001",
            variant_id="tutor-v3",
            run_profile="evaluation",
            connection=_connection(scorer_llm=scorer_llm, tutor_llm=llm),
            events=events,
        )
    finally:
        controller.shutdown(wait=False)

    final = EvalRunRepository(session).get(result.run.id)
    assert final.lifecycle == "finished"
    assert final.outcome == "system_failed"
    assert final.operational_error_json is not None
    operational = final.operational_error_json
    assert operational["code"] == "model_unavailable"
    assert operational["stage"] == "generation"
    assert operational["retryable"] is True
    assert _SENTINEL not in final.operational_error_json

    # No frozen Candidate, no hash, no ScoreSet, and the scorer was never called.
    assert final.artifact_hash is None
    assert final.candidate_artifact_json is None
    assert result.score_set is None
    assert result.executions == ()
    assert scorer_llm.calls == 0
    assert session.scalars(select(EvalRun)).all() == [final]

    # One retrieval and one Attempt happened before the failure.
    assert len(loader.snapshots) == 1
    assert llm.astream_calls == 1
    # No raw block payload reaches the run event stream as token text.
    assert all(
        isinstance(event.get("text"), str)
        for event in events
        if event.get("type") == "token"
    )
    assert _SENTINEL not in json.dumps(events, default=str)


@pytest.mark.asyncio
async def test_real_attempt_malformed_block_fails_run_without_scoring(session):
    loader = RealPathLoader()
    runner, controller = _real_runner(loader)
    scorer_llm = FakeScorerLLM()
    service = _service(session, runner, scorer_llm=scorer_llm)
    llm = BlockChunkTutorLLM(
        [
            [{"type": "text", "text": "partial "}],
            [{"type": "non_standard", "value": _SENTINEL}],
        ]
    )
    events: list[dict] = []

    try:
        result = await service.run(
            experiment_id="tutor-prompt-regression-v1",
            task_case_id="tgqa-001",
            variant_id="tutor-v3",
            run_profile="evaluation",
            connection=_connection(scorer_llm=scorer_llm, tutor_llm=llm),
            events=events,
        )
    finally:
        controller.shutdown(wait=False)

    final = EvalRunRepository(session).get(result.run.id)
    assert final.outcome == "system_failed"
    operational = final.operational_error_json
    assert operational["code"] == "model_unavailable"
    assert operational["stage"] == "generation"
    assert _SENTINEL not in final.operational_error_json
    assert final.artifact_hash is None
    assert final.candidate_artifact_json is None
    assert result.score_set is None
    assert scorer_llm.calls == 0
    assert all(
        isinstance(event.get("text"), str)
        for event in events
        if event.get("type") == "token"
    )
    assert _SENTINEL not in json.dumps(events, default=str)
