"""TDD contract tests for the isolated Learning Run TutorRunner boundary."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import threading
import time
from typing import Any

import pytest

from app.agent.tutor_attempt import TutorCandidate
from app.eval.learning_run.registry import TaskRegistry


def _runner_api():
    try:
        from app.eval.learning_run.runner import (
            RunnerOperationalError,
            TutorRunner,
            resolved_retrieval_config,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover - intentional RED guard
        pytest.fail(f"learning-run runner module is missing: {exc}", pytrace=False)
    return TutorRunner, RunnerOperationalError, resolved_retrieval_config


REGISTRY = TaskRegistry.load_default()
DEFINITION = REGISTRY.resolve_run(
    experiment_id="tutor-prompt-regression-v1",
    task_case_id="tgqa-001",
    variant_id="tutor-v3",
    run_profile="evaluation",
)


def _candidate(*, evidence: list[dict[str, Any]] | None = None) -> TutorCandidate:
    evidence = evidence if evidence is not None else [
        {
            "chunk_id": "tgqa-c01-rrf",
            "content": "Reciprocal rank fusion combines ranked lists.",
            "source": "learning-run-notes.md",
            "page": 1,
        }
    ]
    return TutorCandidate(
        answer="Reciprocal rank fusion combines ranked lists [1].",
        citations=[
            {
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "page": chunk["page"],
                "span_start": 0,
                "span_end": len(chunk["content"]),
            }
            for chunk in evidence
        ],
        evidence=evidence,
        formatted_context="[1] learning-run-notes.md p.1: Reciprocal rank fusion combines ranked lists.",
        usage={"input_tokens": 4, "output_tokens": 8, "total_tokens": 12},
        trace=[{"stage": "tutor", "event": "complete"}],
    )


def _wait_for_controller_state(
    controller: Any, expected: str, *, timeout: float = 5.0
) -> bool:
    """Wait for a background state transition against a deadline.

    `time.sleep` is what makes this correct rather than a slower spin: it
    releases the GIL so the materializer's cleanup thread can actually be
    scheduled. A tight pure-Python loop starves that thread on a
    few-core runner while claiming to wait for it.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if controller.state == expected:
            return True
        time.sleep(0.005)
    return controller.state == expected


def test_wait_for_controller_state_waits_on_a_deadline_not_a_spin_count():
    """The helper must outlast a background thread that is slower than N reads.

    A fixed spin count is not a synchronisation primitive: on a constrained
    runner the materializer's cleanup thread is not scheduled within the spin
    budget, and a tight pure-Python loop holds the GIL against the very thread
    it is waiting for. This controller becomes idle on the wall clock, so it is
    unreachable by spinning faster.
    """

    class LateController:
        def __init__(self, ready_after_seconds: float) -> None:
            self._ready_at = time.monotonic() + ready_after_seconds

        @property
        def state(self) -> str:
            return "idle" if time.monotonic() >= self._ready_at else "draining"

    assert _wait_for_controller_state(LateController(0.05), "idle")


class RecordingLoader:
    def __init__(self, *, error: BaseException | None = None):
        self.calls: list[Any] = []
        self.error = error
        self.closed_event = threading.Event()

        class ClosableRetriever:
            def __init__(self, closed_event: threading.Event) -> None:
                self.close_calls = 0
                self._closed_event = closed_event

            def close(self) -> None:
                self.close_calls += 1
                self._closed_event.set()

        self.retriever = ClosableRetriever(self.closed_event)

    def load(self, *, snapshot):
        self.calls.append(snapshot)
        if self.error is not None:
            raise self.error
        return self.retriever


class RecordingEngine:
    def __init__(self, candidate: TutorCandidate | None = None, error: BaseException | None = None, error_step: str = "retrieval"):
        self.calls: list[dict[str, Any]] = []
        self.candidate = candidate or _candidate()
        self.error = error
        self.error_step = error_step

    async def answer(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            event_sink = kwargs.get("event_sink")
            if callable(event_sink):
                event_sink({"type": "trace", "step": self.error_step, "status": "failed"})
            raise self.error
        event_sink = kwargs.get("event_sink")
        if callable(event_sink):
            event_sink({"type": "budget", "stage": "retrieval"})
        return self.candidate


class FakeLLM:
    async def astream(self, _messages):
        if False:  # pragma: no cover - protocol-only fake
            yield None


@pytest.mark.asyncio
async def test_runner_loads_isolated_corpus_and_attempt_once_with_frozen_v3_prompt():
    TutorRunner, _, resolved_retrieval_config = _runner_api()
    loader = RecordingLoader()
    engine = RecordingEngine()
    events: list[dict[str, Any]] = []
    runner = TutorRunner(corpus_loader=loader, attempt_engine=engine)

    candidate = await runner.run(definition=DEFINITION, llm=FakeLLM(), events=events.append)

    assert candidate == engine.candidate
    assert len(loader.calls) == 1
    assert len(engine.calls) == 1
    call = engine.calls[0]
    assert call["retriever"] is loader.retriever
    assert call["question"] == DEFINITION.task.question
    assert call["prompt_template"].version == "tutor-v3"
    assert call["prompt_template"].system_instruction.encode("utf-8") == DEFINITION.prompt.text.encode("utf-8")
    assert call["prompt_template"].system_instruction != "production_v2"
    assert call["attempt_config"].top_k == 5
    assert 0 < call["attempt_config"].retrieval_seconds <= 5
    assert call["attempt_config"].generation_seconds == 55
    assert resolved_retrieval_config(DEFINITION) == {"version": "hybrid-rrf-v1", "top_k": 5}


@pytest.mark.asyncio
async def test_runner_preflight_hash_mismatch_is_typed_and_does_not_load():
    _, RunnerOperationalError, _ = _runner_api()
    loader = RecordingLoader()
    engine = RecordingEngine()
    broken = replace(DEFINITION, corpus=replace(DEFINITION.corpus, aggregate_hash="0" * 64))
    runner = _runner_api()[0](corpus_loader=loader, attempt_engine=engine)

    with pytest.raises(RunnerOperationalError) as caught:
        await runner.run(definition=broken, llm=FakeLLM(), events=[])

    assert caught.value.code == "corpus_mismatch"
    assert caught.value.stage == "corpus"
    assert caught.value.retryable is False
    assert loader.calls == []
    assert engine.calls == []
    assert "0" * 64 not in caught.value.sanitized_message


@pytest.mark.asyncio
async def test_runner_loader_failure_maps_to_corpus_unavailable():
    TutorRunner, RunnerOperationalError, _ = _runner_api()
    loader = RecordingLoader(error=RuntimeError("provider URL and api_key=secret"))
    runner = TutorRunner(corpus_loader=loader, attempt_engine=RecordingEngine())

    with pytest.raises(RunnerOperationalError) as caught:
        await runner.run(definition=DEFINITION, llm=FakeLLM(), events=[])

    assert caught.value.code == "corpus_unavailable"
    assert caught.value.stage == "corpus"
    assert caught.value.retryable is True
    assert "secret" not in caught.value.sanitized_message
    assert "http" not in caught.value.sanitized_message.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("step", "raised", "code"),
    [
        ("retrieval", TimeoutError("retriever deadline"), "retriever_error"),
        ("retrieval", RuntimeError("retriever failed"), "retriever_error"),
        ("generation", TimeoutError("generation deadline"), "generation_timeout"),
        ("generation", RuntimeError("model failed"), "model_unavailable"),
    ],
)
async def test_runner_maps_attempt_failure_by_forwarded_stage(step, raised, code):
    TutorRunner, RunnerOperationalError, _ = _runner_api()

    def events(event):
        if event.get("type") == "trace" and event.get("step") == step:
            event["status"] = "failed"

    loader = RecordingLoader()
    runner = TutorRunner(corpus_loader=loader, attempt_engine=RecordingEngine(error=raised, error_step=step))

    with pytest.raises(RunnerOperationalError) as caught:
        await runner.run(definition=DEFINITION, llm=FakeLLM(), events=events)

    assert caught.value.code == code
    assert caught.value.stage == step
    assert caught.value.sanitized_message


@pytest.mark.asyncio
async def test_runner_propagates_parent_cancellation_without_wrapping():
    TutorRunner, _, _ = _runner_api()
    loader = RecordingLoader()
    runner = TutorRunner(
        corpus_loader=loader,
        attempt_engine=RecordingEngine(error=asyncio.CancelledError()),
    )

    with pytest.raises(asyncio.CancelledError):
        await runner.run(definition=DEFINITION, llm=FakeLLM(), events=[])


@pytest.mark.asyncio
async def test_runner_event_forwarding_failure_is_harness_internal_error():
    TutorRunner, RunnerOperationalError, _ = _runner_api()
    runner = TutorRunner(corpus_loader=RecordingLoader(), attempt_engine=RecordingEngine())

    def rejecting_events(_event):
        raise RuntimeError("secret api_key=do-not-store")

    with pytest.raises(RunnerOperationalError) as caught:
        await runner.run(definition=DEFINITION, llm=FakeLLM(), events=rejecting_events)

    assert caught.value.code == "harness_internal_error"
    assert caught.value.stage == "events"
    assert "do-not-store" not in caught.value.sanitized_message


def test_runner_module_has_no_production_orchestration_imports():
    from pathlib import Path

    source = (Path(__file__).parents[2] / "app/eval/learning_run/runner.py").read_text()
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


def test_runner_retrieval_profile_is_explicit_and_unknown_profile_is_typed():
    TutorRunner, RunnerOperationalError, resolved_retrieval_config = _runner_api()
    unknown = replace(
        DEFINITION,
        variant_controls={**DEFINITION.variant_controls, "retrieval_config_version": "unknown"},
    )

    with pytest.raises(RunnerOperationalError) as caught:
        resolved_retrieval_config(unknown)

    assert caught.value.code == "manifest_invalid"
    assert caught.value.stage == "manifest"
    assert TutorRunner


def test_runner_rejects_mutable_retrieval_profile_injection_to_prevent_manifest_drift():
    TutorRunner, _, _ = _runner_api()

    with pytest.raises(TypeError, match="retrieval_profiles|resolver"):
        TutorRunner(
            corpus_loader=RecordingLoader(),
            attempt_engine=RecordingEngine(),
            retrieval_profiles={"hybrid-rrf-v1": {"version": "hybrid-rrf-v1", "top_k": 3}},
        )


@pytest.mark.asyncio
async def test_runner_moves_blocking_loader_off_event_loop_and_returns_before_late_load():
    TutorRunner, _, _ = _runner_api()

    class BlockingLoader(RecordingLoader):
        def load(self, *, snapshot):
            self.calls.append(snapshot)
            time.sleep(0.15)
            return self.retriever

    loader = BlockingLoader()
    engine = RecordingEngine()
    runner = TutorRunner(corpus_loader=loader, attempt_engine=engine)
    events: list[dict[str, Any]] = []

    started = time.monotonic()
    task = asyncio.create_task(runner.run(definition=DEFINITION, llm=FakeLLM(), events=events))
    await asyncio.sleep(0.005)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    elapsed = time.monotonic() - started

    assert elapsed < 0.10
    assert len(loader.calls) == 1
    await asyncio.sleep(0.17)
    assert engine.calls == []
    assert events == []
    assert loader.retriever.close_calls == 1


@pytest.mark.asyncio
async def test_runner_releases_materialized_retriever_once_on_success_and_attempt_failure():
    TutorRunner, RunnerOperationalError, _ = _runner_api()
    loader = RecordingLoader()
    runner = TutorRunner(corpus_loader=loader, attempt_engine=RecordingEngine())

    await runner.run(definition=DEFINITION, llm=FakeLLM(), events=[])
    assert await asyncio.to_thread(loader.closed_event.wait, 1)
    assert await asyncio.to_thread(
        lambda: _wait_for_controller_state(runner.materializer_controller, "idle")
    )
    assert loader.retriever.close_calls == 1

    failing_loader = RecordingLoader()
    failing_runner = TutorRunner(
        corpus_loader=failing_loader,
        attempt_engine=RecordingEngine(error=RuntimeError("attempt failed")),
    )
    with pytest.raises(RunnerOperationalError):
        await failing_runner.run(definition=DEFINITION, llm=FakeLLM(), events=[])
    assert await asyncio.to_thread(failing_loader.closed_event.wait, 1)
    assert await asyncio.to_thread(
        lambda: _wait_for_controller_state(failing_runner.materializer_controller, "idle")
    )
    assert failing_loader.retriever.close_calls == 1


@pytest.mark.asyncio
async def test_runner_releases_materialized_retriever_on_parent_cancellation():
    TutorRunner, _, _ = _runner_api()
    loader = RecordingLoader()
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingEngine:
        async def answer(self, **_kwargs):
            started.set()
            await release.wait()
            return _candidate()

    runner = TutorRunner(corpus_loader=loader, attempt_engine=BlockingEngine())
    task = asyncio.create_task(runner.run(definition=DEFINITION, llm=FakeLLM(), events=[]))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await asyncio.to_thread(
        lambda: _wait_for_controller_state(runner.materializer_controller, "idle")
    )
    assert loader.retriever.close_calls == 1


@pytest.mark.asyncio
async def test_runner_preflight_deadline_gates_blocking_loader_before_attempt():
    TutorRunner, RunnerOperationalError, _ = _runner_api()

    class SlowLoader(RecordingLoader):
        def load(self, *, snapshot):
            self.calls.append(snapshot)
            time.sleep(0.05)
            return self.retriever

    budget = {**DEFINITION.budget, "retrieval_preflight_seconds": 0.001}
    definition = replace(DEFINITION, budget=budget)
    loader = SlowLoader()
    engine = RecordingEngine()
    events: list[dict[str, Any]] = []
    runner = TutorRunner(corpus_loader=loader, attempt_engine=engine)

    started = time.monotonic()
    with pytest.raises(RunnerOperationalError) as caught:
        await runner.run(definition=definition, llm=FakeLLM(), events=events)
    elapsed = time.monotonic() - started

    assert elapsed < 0.04
    assert caught.value.stage == "retrieval"
    assert caught.value.code == "retriever_error"
    assert caught.value.retryable is True
    assert "deadline" in caught.value.sanitized_message
    assert len(loader.calls) == 1
    assert engine.calls == []
    await asyncio.sleep(0.06)
    assert engine.calls == []
    assert events == []


@pytest.mark.asyncio
async def test_runner_passes_remaining_positive_retrieval_budget_to_attempt():
    TutorRunner, _, _ = _runner_api()

    class PartlySlowLoader(RecordingLoader):
        def load(self, *, snapshot):
            self.calls.append(snapshot)
            time.sleep(0.01)
            return self.retriever

    budget = {**DEFINITION.budget, "retrieval_preflight_seconds": 0.05}
    definition = replace(DEFINITION, budget=budget)
    loader = PartlySlowLoader()
    engine = RecordingEngine()
    runner = TutorRunner(corpus_loader=loader, attempt_engine=engine)

    await runner.run(definition=definition, llm=FakeLLM(), events=[])

    assert len(engine.calls) == 1
    retrieval_seconds = engine.calls[0]["attempt_config"].retrieval_seconds
    assert 0 < retrieval_seconds < 0.05


class _ThreadBlockingRawRetriever:
    def __init__(self) -> None:
        self.search_started = threading.Event()
        self.release_search = threading.Event()
        self.closed = threading.Event()
        self.close_calls = 0
        self.close_thread_names: list[str] = []

    def search(self, _query: str, top_k: int = 5) -> list[dict[str, Any]]:
        del top_k
        self.search_started.set()
        self.release_search.wait(timeout=5)
        return []

    def close(self) -> None:
        self.close_calls += 1
        self.close_thread_names.append(threading.current_thread().name)
        self.closed.set()


class _ThreadBlockingLoader:
    def __init__(self, raw: _ThreadBlockingRawRetriever) -> None:
        self.raw = raw
        self.calls = 0

    def load(self, *, snapshot, stop=None, deadline=None):
        del snapshot, stop, deadline
        self.calls += 1
        from app.eval.learning_run.corpus import IsolatedCorpusRetriever

        return IsolatedCorpusRetriever(
            client=_ClosableClient(),
            collection=_ClosableCollection("runner-blocking"),
            retriever=self.raw,
        )


class _ClosableCollection:
    def __init__(self, name: str) -> None:
        self.name = name


class _ClosableClient:
    def __init__(self) -> None:
        self.delete_calls = 0
        self.close_calls = 0

    def delete_collection(self, _name: str) -> None:
        self.delete_calls += 1

    def close(self) -> None:
        self.close_calls += 1


class _SearchTimeoutEngine:
    async def answer(self, **kwargs):
        config = kwargs["attempt_config"]
        await asyncio.wait_for(
            asyncio.to_thread(kwargs["retriever"].search, "blocked"),
            timeout=config.retrieval_seconds,
        )
        return _candidate()


@pytest.mark.asyncio
async def test_runner_timeout_handoffs_cleanup_without_waiting_for_search_lock():
    TutorRunner, RunnerOperationalError, _ = _runner_api()
    raw = _ThreadBlockingRawRetriever()
    loader = _ThreadBlockingLoader(raw)
    budget = {**DEFINITION.budget, "retrieval_preflight_seconds": 0.05}
    definition = replace(DEFINITION, budget=budget)
    from app.eval.learning_run.corpus import CorpusMaterializerBusyError, CorpusMaterializerController

    controller = CorpusMaterializerController(loader)
    runner = TutorRunner(
        corpus_loader=loader,
        attempt_engine=_SearchTimeoutEngine(),
        materializer_controller=controller,
    )
    task = asyncio.create_task(runner.run(definition=definition, llm=FakeLLM(), events=[]))
    assert await asyncio.to_thread(raw.search_started.wait, 1)

    with pytest.raises(RunnerOperationalError) as caught:
        await asyncio.wait_for(task, timeout=0.2)
    assert caught.value.code == "retriever_error"
    assert controller.state == "draining"
    assert controller.outstanding_count == 1
    assert not raw.closed.is_set()
    with pytest.raises(CorpusMaterializerBusyError):
        await controller.acquire(snapshot=DEFINITION.corpus, deadline=asyncio.get_running_loop().time() + 1)
    assert loader.calls == 1

    raw.release_search.set()
    assert await asyncio.to_thread(raw.closed.wait, 1)
    assert raw.close_calls == 1
    assert raw.close_thread_names[0].startswith("learning-eval-corpus")
    assert await asyncio.to_thread(lambda: controller.state == "idle")
    assert controller.outstanding_count == 0
    controller.shutdown(wait=False)


@pytest.mark.asyncio
async def test_runner_parent_cancel_handoffs_cleanup_without_waiting_for_search_lock():
    TutorRunner, _, _ = _runner_api()
    raw = _ThreadBlockingRawRetriever()
    loader = _ThreadBlockingLoader(raw)
    from app.eval.learning_run.corpus import CorpusMaterializerController

    controller = CorpusMaterializerController(loader)
    runner = TutorRunner(
        corpus_loader=loader,
        attempt_engine=_SearchTimeoutEngine(),
        materializer_controller=controller,
    )
    task = asyncio.create_task(runner.run(definition=DEFINITION, llm=FakeLLM(), events=[]))
    assert await asyncio.to_thread(raw.search_started.wait, 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=0.2)
    assert controller.state == "draining"
    assert controller.outstanding_count == 1
    assert not raw.closed.is_set()
    raw.release_search.set()
    assert await asyncio.to_thread(raw.closed.wait, 1)
    assert raw.close_calls == 1
    assert await asyncio.to_thread(lambda: controller.state == "idle")
    controller.shutdown(wait=False)
