"""D5c-3: tutor usage consumption across four real locked provider SDK chains.

Four closed loops, one per provider wire shape. Each drives a fixed synthetic
wire (built by the accepted D5a transport builders) through the real locked
SDK / adapter, into the real ``TutorAttemptEngine`` (real candidate + completed
trace), through the real ``RunService``, into a file-backed temporary SQLite
database, and finally a fresh-Session ``get_verified`` re-read. Only the socket
is replaced (``httpx.MockTransport``); the scorer is the accepted fake and
retrieval the isolated in-process ``RealPathLoader`` corpus.

This is usage-consumption evidence, NOT a production provider/model identity
check. The registry's ``tutor-v3`` variant is labelled ``provider="ollama"`` /
``model="llama3.2"``; that label is only the manifest seam the RunService
validates. Injecting a real OpenAI, Anthropic or Ollama SDK behind ``tutor_llm``
proves the consumer consumes that SDK's projected usage -- it does not claim the
run's production identity is that provider. No cloud provider is passed, so the
run never trips an ``evaluation_config_mismatch``.

Every expected value is a fixed literal derived from the synthetic wire plus the
accepted projections (the locked adapter's usage projection and the consumer's
per-key chunk sum); none is computed from a run's output. Each case observes the
raw SDK chunks and the ``on_llm_end`` callback separately from the real consumer
read out of the frozen artifact -- the callback never stands in for the consumer.

The four nodes are:
1. OpenAI: a four-frame SSE stream (po/ng/finish/usage-only) whose single tail
   usage frame projects to the consumer (3,2,5).
2. Anthropic: ``d5a_stream_with_usage`` opens with usage the adapter withholds,
   so only the terminal frame projects; consumer (0,5,5).
3. Anthropic: ``d5a2_stream`` with a usage-less ``message_delta``; the terminal
   frame derives an all-zero usage, so the consumer sees (0,0,0), not
   "unavailable".
4. Ollama: ``d5a_counted_stream`` puts both counters on the top-level done
   frame; the adapter derives the total, consumer (7,3,10).

Selected cancellation, body-failure and hash cases already carry accepted
evidence in the D5c-1 contract and are not repeated here. Retries remain out of
scope for this batch and are deliberately NOT claimed as covered.
"""

from __future__ import annotations

import json
import socket
from typing import Any

import pytest
from langchain_core.callbacks import AsyncCallbackHandler
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from tests.llm.sdk_transport_support import (  # noqa: F401  (autouse scope)
    deny_sockets,
    offline_env,
    offline_openai,
    sse_event,
)
from tests.llm.test_async_transport_contract import (
    d5a_add_chunks as add_sdk_chunks,
    stream_chunk,
)
from tests.llm.test_async_anthropic_transport_contract import (  # noqa: F401
    anthropic_env,
    d5a2_message_delta,
    d5a2_stream,
    d5a_stream_with_usage,
    offline_anthropic,
)
from tests.llm.test_async_ollama_transport_contract import (  # noqa: F401
    d5a_counted_stream,
    ollama_env,
    offline_ollama,
)
from tests.eval.test_learning_run_service import (
    FakeClock,
    FakeScorerLLM,
    RealPathLoader,
    _connection,
    _real_runner,
    _service,
)

from app.agent.tutor_attempt import TutorAttemptEngine, TutorCandidate
from app.db.models import Base
from app.eval.learning_run.repositories import EvalRunRepository

_EXPERIMENT_ID = "tutor-prompt-regression-v1"
_TASK_CASE_ID = "tgqa-001"
_VARIANT_ID = "tutor-v3"
_RUN_PROFILE = "evaluation"

_OFFLINE_FIXTURES = ("offline_env", "deny_sockets", "anthropic_env", "ollama_env")
_USAGE_COUNT_KEYS = frozenset({"input_tokens", "output_tokens", "total_tokens"})

# ---------------------------------------------------------------------------
# Independent expectations (fixed literals, never computed from the code)
# ---------------------------------------------------------------------------

# Node 1 -- OpenAI tail usage frame projects input 3 / output 2 / total 5.
_OPENAI_SDK_USAGE = {
    "input_tokens": 3,
    "output_tokens": 2,
    "total_tokens": 5,
    "input_token_details": {},
    "output_token_details": {},
}
_OPENAI_CONSUMER = {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}

# Node 2 -- Anthropic opening usage is withheld; terminal projects output 5.
_ANTHROPIC_SDK_USAGE = {
    "input_tokens": 0,
    "output_tokens": 5,
    "total_tokens": 5,
    "input_token_details": {},
}
_ANTHROPIC_CONSUMER = {"input_tokens": 0, "output_tokens": 5, "total_tokens": 5}

# Node 3 -- usage-less message_delta; the adapter derives an all-zero usage.
_ANTHROPIC_ZERO_SDK_USAGE = {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
    "input_token_details": {},
}
_ANTHROPIC_ZERO_CONSUMER = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

# Node 4 -- Ollama done frame carries the two counters; total is adapter-derived.
_OLLAMA_SDK_USAGE = {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
_OLLAMA_CONSUMER = {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}


# ---------------------------------------------------------------------------
# Local observers and thin consumer helpers
# ---------------------------------------------------------------------------


class _EndCapture(AsyncCallbackHandler):
    """Local observer of the real aggregated SDK message; it changes nothing."""

    def __init__(self) -> None:
        self.messages: list[Any] = []

    async def on_llm_end(self, response, **kwargs):
        self.messages.append(response.generations[0][0].message)


class PassThroughChunkObserver:
    """Thin pass-through that records raw SDK chunks and alters nothing."""

    def __init__(self, model) -> None:
        self._model = model
        self.chunks: list[Any] = []

    def astream(self, messages, **kwargs):
        return self._record(messages, **kwargs)

    async def _record(self, messages, **kwargs):
        async for chunk in self._model.astream(messages, **kwargs):
            self.chunks.append(chunk)
            yield chunk


class RecordingTutorAttemptEngine(TutorAttemptEngine):
    """The real engine plus a record of the real consumer's Candidate.

    It only awaits ``super().answer`` and stores the outcome; it never changes
    the answer path.
    """

    def __init__(self) -> None:
        super().__init__()
        self.candidates: list[TutorCandidate] = []

    async def answer(self, **kwargs):
        candidate = await super().answer(**kwargs)
        self.candidates.append(candidate)
        return candidate


def _generation_completed(trace) -> dict:
    matches = [
        event
        for event in trace
        if event.get("type") == "trace"
        and event.get("step") == "generation"
        and event.get("status") == "completed"
    ]
    assert len(matches) == 1, "exactly one generation-completed trace is expected"
    return matches[0]


def _verified_run(session, run_id: str):
    """Re-read a persisted run through a brand-new Session on the same file DB."""
    with Session(session.get_bind()) as fresh:
        return EvalRunRepository(fresh).get_verified(run_id)


def _sse_payload(frame: bytes) -> dict:
    """Decode the single ``data:`` JSON line of one raw SSE frame."""
    for line in frame.splitlines():
        if line.startswith(b"data: "):
            return json.loads(line[len(b"data: "):])
    raise AssertionError(f"SSE frame carries no data line: {frame!r}")


def _ndjson_payload(line: bytes) -> dict:
    """Decode one raw NDJSON line."""
    return json.loads(line.decode())


def _assert_offline_fixtures(request) -> None:
    """The explicitly imported autouse fixtures are live for this case."""
    active = set(request.fixturenames)
    for name in _OFFLINE_FIXTURES:
        assert name in active, f"{name} must be autouse for this module"


def _assert_candidate(result, recorder, expected_usage, session):
    """Assert the real consumer, artifact, trace and stored value all agree."""
    assert result.run.lifecycle == "finished"
    assert result.run.outcome == "success"
    assert len(recorder.candidates) == 1
    candidate = recorder.candidates[0]
    assert candidate.answer == "pong"
    assert candidate.usage == expected_usage
    # The consumer keeps only the three canonical counts; SDK details never
    # cross this boundary.
    assert set(candidate.usage) == _USAGE_COUNT_KEYS

    artifact = result.run.candidate_artifact_json
    assert artifact["usage"] == expected_usage
    completed = _generation_completed(artifact["trace"])
    assert completed["usage"] == expected_usage
    assert completed["answer_length"] == len("pong")
    assert candidate.trace == artifact["trace"]
    assert result.score_set is not None

    stored = _verified_run(session, result.run.id)
    assert stored.candidate_artifact_json["usage"] == expected_usage
    assert stored.candidate_artifact_json["trace"] == artifact["trace"]
    assert stored.artifact_hash == result.run.artifact_hash
    return candidate


@pytest.fixture
def eval_session(tmp_path):
    """Independent file-backed SQLite (create_all, FK ON, dispose)."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'd5c3-eval.sqlite'}",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    try:
        with Session(engine) as db_session:
            yield db_session
    finally:
        engine.dispose()


async def _execute_run(session, *, respond, responder, tutor_llm, scorer_llm, clock=None):
    """Run one real Attempt behind the real RunService into ``session``."""
    respond(responder)
    loader = RealPathLoader()
    runner, controller = _real_runner(loader)
    recorder = RecordingTutorAttemptEngine()
    runner.attempt_engine = recorder
    service = _service(session, runner, scorer_llm=scorer_llm, clock=clock or FakeClock())
    try:
        result = await service.run(
            experiment_id=_EXPERIMENT_ID,
            task_case_id=_TASK_CASE_ID,
            variant_id=_VARIANT_ID,
            run_profile=_RUN_PROFILE,
            connection=_connection(scorer_llm=scorer_llm, tutor_llm=tutor_llm),
            events=None,
        )
    finally:
        controller.shutdown(wait=False)
    return result, recorder, loader


# ---------------------------------------------------------------------------
# Node 1/4: OpenAI -- complete four-frame SSE, tail usage projection
# ---------------------------------------------------------------------------


async def test_d5c3_openai_tail_usage_projection(request, eval_session, offline_openai):
    """A full OpenAI SSE stream projects only its tail usage frame.

    The autouse fixtures are asserted live here and the socket guard is proven
    to block a real DNS/connect attempt. The four adapter frames plus the
    langchain-core sentinel are observed raw, the SDK aggregation and the real
    ``on_llm_end`` callback are compared, and the consumer is read from the
    frozen artifact -- never from the callback.
    """
    _assert_offline_fixtures(request)
    with pytest.raises(AssertionError, match="real network access"):
        socket.getaddrinfo("api.openai.com", 443)
    with pytest.raises(AssertionError, match="real network access"):
        socket.create_connection(("api.openai.com", 443))

    capture = _EndCapture()
    events = [
        sse_event(stream_chunk("po")),
        sse_event(stream_chunk("ng")),
        sse_event(stream_chunk(finish_reason="stop")),
        sse_event(
            stream_chunk(
                choices=[],
                usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            )
        ),
    ]
    frames = [_sse_payload(frame) for frame in events]
    assert frames[2]["choices"][0]["finish_reason"] == "stop"
    assert frames[3]["choices"] == []
    assert frames[3]["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }
    body = offline_openai.sse_body(events)
    model = offline_openai.model(callbacks=[capture])
    observer = PassThroughChunkObserver(model)
    scorer_llm = FakeScorerLLM()

    result, recorder, loader = await _execute_run(
        eval_session, respond=offline_openai.respond,
        responder=lambda request: offline_openai.sse_response(body),
        tutor_llm=observer, scorer_llm=scorer_llm,
    )

    assert len(offline_openai.requests) == 1
    assert len(loader.snapshots) == 1, "the evaluation profile runs exactly one Attempt"

    # Usage sits on the usage-only adapter frame (index 3); the sentinel (4)
    # carries none.
    assert [chunk.content for chunk in observer.chunks] == ["po", "ng", "", "", ""]
    assert [chunk.usage_metadata for chunk in observer.chunks] == [
        None,
        None,
        None,
        _OPENAI_SDK_USAGE,
        None,
    ]

    # SDK aggregation and the real callback agree, and both keep the detail
    # dicts the consumer must drop.
    assert add_sdk_chunks(observer.chunks).usage_metadata == _OPENAI_SDK_USAGE
    assert len(capture.messages) == 1
    assert capture.messages[0].usage_metadata == _OPENAI_SDK_USAGE

    candidate = _assert_candidate(result, recorder, _OPENAI_CONSUMER, eval_session)
    assert "input_token_details" not in candidate.usage


# ---------------------------------------------------------------------------
# Node 2/4: Anthropic -- opening usage withheld, terminal projects
# ---------------------------------------------------------------------------


async def test_d5c3_anthropic_opening_withheld_terminal_projection(
    request, eval_session, offline_anthropic
):
    """The opening frame reports usage but the adapter withholds it.

    ``d5a_stream_with_usage`` opens with input 3 / output 1 and closes with
    output 5. The adapter projects neither as a running total: the opening frame
    carries no ``usage_metadata`` and only the terminal frame projects, landing
    the consumer on (0,5,5).
    """
    _assert_offline_fixtures(request)

    capture = _EndCapture()
    events = d5a_stream_with_usage(
        open_usage={"input_tokens": 3, "output_tokens": 1},
        final_output=5,
    )
    start_payload = _sse_payload(events[0])
    delta_payload = _sse_payload(events[5])
    assert start_payload["message"]["usage"] == {"input_tokens": 3, "output_tokens": 1}
    assert delta_payload["usage"] == {"output_tokens": 5}
    body = offline_anthropic.sse_body(events)
    model = offline_anthropic.model(callbacks=[capture])
    observer = PassThroughChunkObserver(model)
    scorer_llm = FakeScorerLLM()

    result, recorder, loader = await _execute_run(
        eval_session, respond=offline_anthropic.respond,
        responder=lambda request: offline_anthropic.sse_response(body),
        tutor_llm=observer, scorer_llm=scorer_llm,
    )

    assert len(offline_anthropic.requests) == 1
    assert len(loader.snapshots) == 1

    assert [chunk.content for chunk in observer.chunks] == ["", "po", "ng", ""]
    assert [chunk.usage_metadata for chunk in observer.chunks] == [
        None,
        None,
        None,
        _ANTHROPIC_SDK_USAGE,
    ]
    # The opening frame reported input 3 on the wire yet projects nothing.
    assert observer.chunks[0].usage_metadata is None

    assert add_sdk_chunks(observer.chunks).usage_metadata == _ANTHROPIC_SDK_USAGE
    assert len(capture.messages) == 1
    assert capture.messages[0].usage_metadata == _ANTHROPIC_SDK_USAGE

    candidate = _assert_candidate(result, recorder, _ANTHROPIC_CONSUMER, eval_session)
    assert "input_token_details" not in candidate.usage


# ---------------------------------------------------------------------------
# Node 3/4: Anthropic -- terminal derives an all-zero usage, not unavailable
# ---------------------------------------------------------------------------


async def test_d5c3_anthropic_terminal_derives_zero_not_unavailable(
    request, eval_session, offline_anthropic
):
    """A usage-less terminal frame derives zeros rather than "unavailable".

    The opening frame still carries usage on the wire, but ``message_delta``
    omits usage. The SDK builds the absent field as its default and the adapter
    derives an all-zero usage, so the consumer sees (0,0,0) -- a real dict, kept
    distinct from the "unavailable" sentinel used when no usage is extracted.
    """
    _assert_offline_fixtures(request)

    capture = _EndCapture()
    events = d5a2_stream(message_delta=d5a2_message_delta(omit_usage=True))
    start_payload = _sse_payload(events[0])
    delta_payload = _sse_payload(events[5])
    assert "usage" in start_payload["message"]
    assert start_payload["message"]["usage"] == {"input_tokens": 3, "output_tokens": 0}
    assert "usage" not in delta_payload
    body = offline_anthropic.sse_body(events)
    model = offline_anthropic.model(callbacks=[capture])
    observer = PassThroughChunkObserver(model)
    scorer_llm = FakeScorerLLM()

    result, recorder, loader = await _execute_run(
        eval_session, respond=offline_anthropic.respond,
        responder=lambda request: offline_anthropic.sse_response(body),
        tutor_llm=observer, scorer_llm=scorer_llm,
    )

    assert len(offline_anthropic.requests) == 1
    assert len(loader.snapshots) == 1

    assert [chunk.content for chunk in observer.chunks] == ["", "po", "ng", ""]
    assert [chunk.usage_metadata for chunk in observer.chunks] == [
        None,
        None,
        None,
        _ANTHROPIC_ZERO_SDK_USAGE,
    ]

    assert add_sdk_chunks(observer.chunks).usage_metadata == _ANTHROPIC_ZERO_SDK_USAGE
    assert len(capture.messages) == 1
    assert capture.messages[0].usage_metadata == _ANTHROPIC_ZERO_SDK_USAGE

    candidate = _assert_candidate(
        result, recorder, _ANTHROPIC_ZERO_CONSUMER, eval_session
    )
    assert candidate.usage != "unavailable"
    assert isinstance(candidate.usage, dict)


# ---------------------------------------------------------------------------
# Node 4/4: Ollama -- done frame carries both counters, total derived
# ---------------------------------------------------------------------------


async def test_d5c3_ollama_done_frame_derived_total(request, eval_session, offline_ollama):
    """The Ollama done frame ships both counters and no total.

    ``d5a_counted_stream`` puts ``prompt_eval_count`` and ``eval_count`` on the
    top-level done frame; the wire carries no total, so the adapter derives it
    (7 + 3) and the consumer lands on (7,3,10).
    """
    _assert_offline_fixtures(request)

    capture = _EndCapture()
    lines = d5a_counted_stream(prompt=7, output=3)
    done_payload = _ndjson_payload(lines[-1])
    assert done_payload["done"] is True
    assert done_payload["prompt_eval_count"] == 7
    assert done_payload["eval_count"] == 3
    assert "total_tokens" not in done_payload
    body = offline_ollama.ndjson_body(lines)
    model = offline_ollama.model(callbacks=[capture])
    observer = PassThroughChunkObserver(model)
    scorer_llm = FakeScorerLLM()

    result, recorder, loader = await _execute_run(
        eval_session, respond=offline_ollama.respond,
        responder=lambda request: offline_ollama.ndjson_response(body),
        tutor_llm=observer, scorer_llm=scorer_llm,
    )

    assert len(offline_ollama.requests) == 1
    assert len(loader.snapshots) == 1

    assert [chunk.content for chunk in observer.chunks] == ["po", "ng", "", ""]
    assert [chunk.usage_metadata for chunk in observer.chunks] == [
        None,
        None,
        _OLLAMA_SDK_USAGE,
        None,
    ]
    # The done frame carries the two counters and no total; the adapter derives
    # the total, so it is not a provider-reported field.
    done_meta = observer.chunks[2].response_metadata
    assert done_meta["prompt_eval_count"] == 7
    assert done_meta["eval_count"] == 3
    assert "total_tokens" not in done_meta

    assert add_sdk_chunks(observer.chunks).usage_metadata == _OLLAMA_SDK_USAGE
    assert len(capture.messages) == 1
    assert capture.messages[0].usage_metadata == _OLLAMA_SDK_USAGE

    _assert_candidate(result, recorder, _OLLAMA_CONSUMER, eval_session)
