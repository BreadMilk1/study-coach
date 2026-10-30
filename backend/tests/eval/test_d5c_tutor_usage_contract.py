"""D5c-1: Gemini usage across the real locked chain, into a real Learning Run.

Scope: one closed loop -- synthetic Gemini SSE wire -> the real locked
``langchain-google-genai`` adapter and Google SDK parser -> the real
``TutorAttemptEngine`` -> a real Tutor candidate / completed trace -> the real
``RunService`` -> a file-backed temporary SQLite database -> a fresh-Session
``get_verified`` re-read. Only the socket is replaced, through the same
``client_args`` transport seam the accepted D5a contract already validated; the
scorer stays the accepted fake and retrieval stays an isolated in-process fake
corpus (``RealPathLoader``), so no Chroma and no provider are involved.

This module pins the current usage contract: the Tutor now sums each extracted
field across chunks (``TUTOR_USAGE_CONTRACT = "tutor-chunk-sum-v1"``), so the
expected values below describe that summed behaviour. The earlier
"last non-empty" characterization is not erased -- it is preserved in PR26 at
commit ``1702b1d`` (``1702b1d965500c6399bc4a492426542fd02a5959``). A number here
says what the locked code does today; it is NOT a billing figure, a
provider-completion guarantee, or a product-correctness claim. Every expected
value below is derived from the fixed synthetic wire plus the accepted
projections (the locked adapter's usage projection and the consumer's per-key
chunk sum), never from a run's own output. These expectations are fixed before
execution; disagreements must be reported, not rewritten to make tests pass.

The multi-frame and hash cases were migrated from the pre-sum expectations
``(0,2,2)`` / ``(0,1,1)`` to the summed ``(4,7,11)`` / ``(10,5,15)`` when the
engine adopted ``TUTOR_USAGE_CONTRACT = "tutor-chunk-sum-v1"``. The earlier bytes
and their acceptance live in PR26 at that commit and are preserved, not erased.
The wire frames are unchanged -- only the consumer projection moved from "last
non-empty" to the per-key sum.

The ``_connection`` helper reuses the accepted evaluation registry, whose
``tutor-v3`` variant is labelled ``provider="ollama"``. That label is only the
manifest seam identity: it proves Gemini usage crosses this test boundary, not
that production Gemini is the run's identity. Likewise the ``offline_gemini``
harness is a MockTransport, so no real Google credential, DNS lookup or packet
is ever used.

Cancellation here is the service's own *cooperative* cancellation intent
(``CancellationToken``), signalled from an event callback -- it is not an
``asyncio`` task cancellation and must not be described as one.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from typing import Any

import httpx
import pytest
from langchain_core.callbacks import AsyncCallbackHandler
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from tests.llm.sdk_transport_support import (  # noqa: F401  (autouse scope)
    deny_sockets,
    fail_with,
    offline_env,
    send,
)
from tests.llm.test_async_gemini_transport_contract import (  # noqa: F401
    d5a_add_chunks,
    d5a_frame,
    gemini_env,
    offline_gemini,
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

# Distinguishes "no usageMetadata key at all" from an explicit JSON null.
_MISSING = object()


# ---------------------------------------------------------------------------
# Independent expectations (never computed from the code under test)
# ---------------------------------------------------------------------------

_USAGE_COUNT_KEYS = ("input_tokens", "output_tokens", "total_tokens")


def _chunk_usage(input_tokens: int, output_tokens: int, total_tokens: int) -> dict:
    """The locked adapter's projected per-chunk usage, including cache_read 0."""
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_token_details": {"cache_read": 0},
    }


def _candidate_usage(input_tokens: int, output_tokens: int, total_tokens: int) -> dict:
    """The consumer's normalized usage -- no cache_read detail crosses over."""
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _independent_hash(payload: dict) -> str:
    """Recompute the canonical SHA-256 the way the contracts define it."""
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


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


def _frames_responder(gemini, frames):
    return lambda request: gemini.sse_response(gemini.frame_body(frames))


def _single_usage_frame(wire_usage) -> bytes:
    """One Gemini frame with ``pong``/STOP and a controllable usageMetadata.

    ``d5a_frame`` cannot express ``usageMetadata: null`` (passing ``usage=None``
    omits the key), so this parses that existing frame, edits only the local
    payload, and re-serialises it. The imported fixture is never modified.
    """
    frame = d5a_frame("pong", finish_reason="STOP")
    payload = json.loads(frame[len(b"data: ") :])
    if wire_usage is _MISSING:
        payload.pop("usageMetadata", None)
    else:
        payload["usageMetadata"] = wire_usage
    return (f"data: {json.dumps(payload)}\n\n").encode()


def _verified_run(session, run_id: str):
    """Re-read a persisted run through a brand-new Session on the same file DB."""
    with Session(session.get_bind()) as fresh:
        return EvalRunRepository(fresh).get_verified(run_id)


# ---------------------------------------------------------------------------
# Real consumer helpers (thin, behaviour-preserving)
# ---------------------------------------------------------------------------


class _EndCapture(AsyncCallbackHandler):
    """Local observer of the real aggregated SDK message; it changes nothing."""

    def __init__(self) -> None:
        self.messages: list[Any] = []

    async def on_llm_end(self, response, **kwargs):
        self.messages.append(response.generations[0][0].message)


class PassThroughChunkObserver:
    """Thin pass-through that records the raw SDK chunks and alters nothing.

    It exists only so a single run can be observed at both the raw chunk seam
    and the real consumer seam. It never substitutes for the consumer.
    """

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
    """The real engine plus a record of what the consumer returned (or did not).

    It only awaits ``super().answer`` and stores the outcome; it never changes
    the answer path, so one run can be observed at the consumer boundary as well
    as through the frozen artifact.
    """

    def __init__(self) -> None:
        super().__init__()
        self.candidates: list[TutorCandidate] = []
        self.raised: list[BaseException] = []

    async def answer(self, **kwargs):
        try:
            candidate = await super().answer(**kwargs)
        except BaseException as exc:
            self.raised.append(exc)
            raise
        self.candidates.append(candidate)
        return candidate


# ---------------------------------------------------------------------------
# Session fixture: independent file-backed SQLite (create_all, FK ON, dispose)
# ---------------------------------------------------------------------------


@pytest.fixture
def eval_session(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'd5c-eval.sqlite'}",
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


# ---------------------------------------------------------------------------
# Full-chain driver
# ---------------------------------------------------------------------------


async def _execute_run(
    session,
    gemini,
    *,
    responder,
    tutor_llm,
    scorer_llm,
    events=None,
    clock=None,
):
    """Run one real Attempt behind the real RunService into ``session``."""
    gemini.respond(responder)
    loader = RealPathLoader()
    runner, controller = _real_runner(loader)
    recorder = RecordingTutorAttemptEngine()
    runner.attempt_engine = recorder
    service = _service(session, runner, scorer_llm=scorer_llm, clock=clock)
    try:
        result = await service.run(
            experiment_id=_EXPERIMENT_ID,
            task_case_id=_TASK_CASE_ID,
            variant_id=_VARIANT_ID,
            run_profile=_RUN_PROFILE,
            connection=_connection(scorer_llm=scorer_llm, tutor_llm=tutor_llm),
            events=events,
        )
    finally:
        controller.shutdown(wait=False)
    return result, recorder, loader


# ---------------------------------------------------------------------------
# Node 1/12: guard -- the offline scope really is active, and it is asserted
# ---------------------------------------------------------------------------


def test_d5c_guard_offline_scope_and_socket_deny(request):
    """The explicitly imported autouse fixtures are live, and DNS is denied.

    No packet is sent: every socket call is denied before any address is
    dialled. The fixture-name assertion is what proves the explicit import in
    this module actually took effect rather than being assumed.
    """
    import socket

    active = set(request.fixturenames)
    assert "deny_sockets" in active, "deny_sockets must be autouse for this module"
    assert "offline_env" in active, "offline_env must be autouse for this module"
    assert "gemini_env" in active, "gemini_env must be autouse for this module"

    with pytest.raises(AssertionError, match="real network access"):
        socket.getaddrinfo("generativelanguage.googleapis.com", 443)
    with pytest.raises(AssertionError, match="real network access"):
        socket.create_connection(("generativelanguage.googleapis.com", 443))


# ---------------------------------------------------------------------------
# Node 2/12: multi-frame wire -- adapter deltas, callback aggregation, consumer
# ---------------------------------------------------------------------------


async def test_d5c_multiframe_usage_through_real_chain(eval_session, offline_gemini):
    """Cumulative wire -> incremental deltas -> summed consumer value.

    The wire repeats accumulated usage, so the adapter projects per-frame
    deltas ``(4,2,6) (0,3,3) (0,2,2)`` plus a no-usage core sentinel. The
    consumer now sums those deltas per key, so both the real ``on_llm_end``
    callback aggregation and the real consumer land on the cumulative
    ``(4,7,11)``. The callback still only observes the SDK aggregation; it never
    stands in for the consumer, which is read from the frozen artifact.
    """
    capture = _EndCapture()
    frames = [
        d5a_frame(
            "po",
            usage={"promptTokenCount": 4, "candidatesTokenCount": 2, "totalTokenCount": 6},
        ),
        d5a_frame(
            "ng",
            usage={"promptTokenCount": 4, "candidatesTokenCount": 5, "totalTokenCount": 9},
        ),
        d5a_frame(
            "",
            finish_reason="STOP",
            usage={"promptTokenCount": 4, "candidatesTokenCount": 7, "totalTokenCount": 11},
        ),
    ]
    model = offline_gemini.model(callbacks=[capture])
    observer = PassThroughChunkObserver(model)
    scorer_llm = FakeScorerLLM()
    events: list[dict] = []

    result, recorder, loader = await _execute_run(
        eval_session,
        offline_gemini,
        responder=_frames_responder(offline_gemini, frames),
        tutor_llm=observer,
        scorer_llm=scorer_llm,
        events=events,
    )

    assert len(offline_gemini.requests) == 1
    assert len(loader.snapshots) == 1, "the evaluation profile runs exactly one Attempt"

    # Raw per-chunk deltas, from the pass-through observer.
    assert [chunk.content for chunk in observer.chunks] == ["po", "ng", "", ""]
    assert [chunk.usage_metadata for chunk in observer.chunks] == [
        _chunk_usage(4, 2, 6),
        _chunk_usage(0, 3, 3),
        _chunk_usage(0, 2, 2),
        None,
    ]

    # The real SDK callback aggregation is the cumulative (4,7,11).
    assert len(capture.messages) == 1
    aggregated = d5a_add_chunks(observer.chunks)
    assert aggregated.content == "pong"
    assert aggregated.usage_metadata == _chunk_usage(4, 7, 11)
    assert capture.messages[0].usage_metadata == _chunk_usage(4, 7, 11)

    # The real consumer sums the deltas per key, landing on (4,7,11).
    assert len(recorder.candidates) == 1
    candidate = recorder.candidates[0]
    assert candidate.answer == "pong"
    assert candidate.usage == _candidate_usage(4, 7, 11)

    assert result.run.lifecycle == "finished"
    assert result.run.outcome == "success"
    artifact = result.run.candidate_artifact_json
    assert artifact["usage"] == _candidate_usage(4, 7, 11)
    completed = _generation_completed(artifact["trace"])
    assert completed["usage"] == _candidate_usage(4, 7, 11)
    assert completed["answer_length"] == len("pong")
    assert result.score_set is not None

    # The consumer's three counts equal the callback aggregation's counts, but
    # only the SDK aggregation keeps the cache_read detail, so the two dicts are
    # not byte-equal and must not be compared as raw dicts.
    callback_usage = capture.messages[0].usage_metadata
    assert {key: callback_usage[key] for key in _USAGE_COUNT_KEYS} == candidate.usage
    assert "input_token_details" not in candidate.usage
    assert callback_usage["input_token_details"] == {"cache_read": 0}

    # The candidate trace is the frozen artifact trace, and the same completed
    # generation event is visible in the forwarded event stream.
    assert candidate.trace == artifact["trace"]
    event_completed = _generation_completed(events)
    assert event_completed == completed
    assert event_completed["usage"] == candidate.usage

    # A fresh Session re-reads the same usage, trace and hash, and the stored
    # hash recomputes independently from the stored payload.
    stored = _verified_run(eval_session, result.run.id)
    assert stored.candidate_artifact_json["usage"] == candidate.usage
    assert stored.candidate_artifact_json["trace"] == artifact["trace"]
    assert stored.artifact_hash == result.run.artifact_hash
    assert _independent_hash(stored.candidate_artifact_json) == stored.artifact_hash


# ---------------------------------------------------------------------------
# Node 3-9/12: single-frame usage shapes (7 parametrized cases)
# ---------------------------------------------------------------------------

_USAGE_CASES = [
    pytest.param(_MISSING, "unavailable", id="missing"),
    pytest.param(None, "unavailable", id="explicit_null"),
    pytest.param(
        {"promptTokenCount": 0, "candidatesTokenCount": 0, "totalTokenCount": 0},
        "unavailable",
        id="explicit_zero",
    ),
    pytest.param(
        {"promptTokenCount": None, "candidatesTokenCount": None, "totalTokenCount": None},
        "unavailable",
        id="null_fields",
    ),
    pytest.param(
        {"promptTokenCount": 4},
        _candidate_usage(4, 0, 0),
        id="partial_prompt_only",
    ),
    pytest.param(
        {"promptTokenCount": 4, "candidatesTokenCount": 2},
        _candidate_usage(4, 2, 0),
        id="partial_prompt_cand",
    ),
    pytest.param(
        {"totalTokenCount": 11},
        _candidate_usage(0, 0, 11),
        id="partial_total_only",
    ),
]


@pytest.mark.parametrize("wire_usage,expected", _USAGE_CASES)
async def test_d5c_single_frame_usage_projection(
    eval_session, offline_gemini, wire_usage, expected
):
    """One fixed wire shape per case, carried through to the frozen artifact.

    A missing key, an explicit null, an explicit all-zero object and three
    null-valued fields each project to ``None`` and surface as ``"unavailable"``
    -- they are kept apart on the wire so a projected ``None`` is never reported
    as "the provider said zero". The three partial objects are projected as
    written, with the adapter supplying ``total_tokens: 0`` so the consumer does
    not fall back to summing input + output.
    """
    frame = _single_usage_frame(wire_usage)

    # The raw wire really carries this case's distinct shape.
    payload = json.loads(frame[len(b"data: ") :])
    assert payload["candidates"][0]["content"]["parts"][0]["text"] == "pong"
    assert payload["candidates"][0]["finishReason"] == "STOP"
    if wire_usage is _MISSING:
        assert "usageMetadata" not in payload, "the key is absent, not null"
    elif wire_usage is None:
        assert "usageMetadata" in payload
        assert payload["usageMetadata"] is None, "an explicit null, not an absence"
    else:
        assert payload["usageMetadata"] == wire_usage

    model = offline_gemini.model()
    observer = PassThroughChunkObserver(model)
    scorer_llm = FakeScorerLLM()
    events: list[dict] = []

    result, recorder, loader = await _execute_run(
        eval_session,
        offline_gemini,
        responder=_frames_responder(offline_gemini, [frame]),
        tutor_llm=observer,
        scorer_llm=scorer_llm,
        events=events,
    )

    assert len(offline_gemini.requests) == 1
    assert len(loader.snapshots) == 1

    # The projected chunk usage: none for the four "absent-like" shapes, the
    # explicit projection (with cache_read 0) for the three partial shapes.
    if expected == "unavailable":
        assert [chunk.usage_metadata for chunk in observer.chunks] == [None, None]
    else:
        assert observer.chunks[0].usage_metadata == _chunk_usage(
            expected["input_tokens"], expected["output_tokens"], expected["total_tokens"]
        )
        assert observer.chunks[-1].usage_metadata is None, "the core sentinel has no usage"

    assert len(recorder.candidates) == 1
    candidate = recorder.candidates[0]
    assert candidate.answer == "pong"
    assert candidate.usage == expected

    assert result.run.lifecycle == "finished"
    assert result.run.outcome == "success"
    artifact = result.run.candidate_artifact_json
    assert artifact["usage"] == expected
    assert _generation_completed(artifact["trace"])["usage"] == expected

    # The saved value re-reads identically through a brand-new Session.
    stored = _verified_run(eval_session, result.run.id)
    assert stored.candidate_artifact_json["usage"] == expected


# ---------------------------------------------------------------------------
# Node 10/12: usage already consumed, then a real httpx read failure
# ---------------------------------------------------------------------------


async def test_d5c_read_error_after_usage_fails_generation_without_artifact(
    eval_session, offline_gemini
):
    """A body that sends a usage frame and then fails is a generation failure.

    The failure surfaces as ``model_unavailable`` at the ``generation`` stage:
    no candidate is returned, no completed trace is emitted, and no artifact,
    hash, ScoreSet or scorer call is produced.
    """
    injected = httpx.ReadError("synthetic d5c read error")
    body = offline_gemini.body(
        [
            send(
                d5a_frame(
                    "po",
                    usage={
                        "promptTokenCount": 4,
                        "candidatesTokenCount": 2,
                        "totalTokenCount": 6,
                    },
                )
            ),
            fail_with(injected),
        ]
    )
    model = offline_gemini.model()
    scorer_llm = FakeScorerLLM()
    events: list[dict] = []

    result, recorder, loader = await _execute_run(
        eval_session,
        offline_gemini,
        responder=lambda request: offline_gemini.sse_response(body),
        tutor_llm=model,
        scorer_llm=scorer_llm,
        events=events,
    )

    assert len(offline_gemini.requests) == 1
    assert len(loader.snapshots) == 1, "one retrieval and one Attempt preceded the failure"
    assert body.close_count >= 1, "the failed response body was closed"

    assert result.run.lifecycle == "finished"
    assert result.run.outcome == "system_failed"
    error = result.run.operational_error_json
    assert error["code"] == "model_unavailable"
    assert error["stage"] == "generation"
    assert error["retryable"] is True
    assert "synthetic d5c read error" not in json.dumps(error), "no payload leak"

    assert recorder.candidates == [], "no candidate is returned"
    assert len(recorder.raised) == 1
    assert isinstance(recorder.raised[0], httpx.ReadError)

    assert result.run.candidate_artifact_json is None
    assert result.run.artifact_hash is None
    assert result.score_set is None
    assert result.executions == ()
    assert scorer_llm.calls == 0

    # The first token was delivered before the failure, and the failure is
    # visible as a generation failed trace carrying the original error class.
    assert [event["text"] for event in events if event.get("type") == "token"] == ["po"]
    failed_traces = [
        event
        for event in events
        if event.get("type") == "trace"
        and event.get("step") == "generation"
        and event.get("status") == "failed"
    ]
    assert len(failed_traces) == 1
    assert failed_traces[0]["error"] == "ReadError"

    assert not any(
        event.get("type") == "trace"
        and event.get("step") == "generation"
        and event.get("status") == "completed"
        for event in events
    ), "no generation-completed trace is emitted"


# ---------------------------------------------------------------------------
# Node 11/12: cooperative cancellation after the first chunk's usage
# ---------------------------------------------------------------------------


async def test_d5c_cooperative_cancellation_after_first_chunk_usage(
    eval_session, offline_gemini
):
    """Cooperative cancellation, signalled on the second token, not a task cancel.

    The events callback requests the service's own cancellation intent on the
    second token. Because the engine emits a token before it extracts usage, the
    first chunk's usage has already been consumed by then; the next emitted
    token then raises the service's internal BaseException signal, so the Attempt
    never returns a candidate. This is the cooperative ``CancellationToken``
    path, deliberately NOT an ``asyncio`` task cancellation.
    """
    frames = [
        d5a_frame(
            "po",
            usage={"promptTokenCount": 4, "candidatesTokenCount": 2, "totalTokenCount": 6},
        ),
        d5a_frame(
            "ng",
            usage={"promptTokenCount": 4, "candidatesTokenCount": 5, "totalTokenCount": 9},
        ),
        d5a_frame(
            "st",
            finish_reason="STOP",
            usage={"promptTokenCount": 4, "candidatesTokenCount": 7, "totalTokenCount": 11},
        ),
    ]
    offline_gemini.respond(_frames_responder(offline_gemini, frames))
    model = offline_gemini.model()
    observer = PassThroughChunkObserver(model)
    scorer_llm = FakeScorerLLM()

    loader = RealPathLoader()
    runner, controller = _real_runner(loader)
    recorder = RecordingTutorAttemptEngine()
    runner.attempt_engine = recorder
    service = _service(eval_session, runner, scorer_llm=scorer_llm)

    events: list[dict] = []
    tokens: list[str] = []
    first_token_seen = asyncio.Event()

    prepared = service.prepare(
        experiment_id=_EXPERIMENT_ID,
        task_case_id=_TASK_CASE_ID,
        variant_id=_VARIANT_ID,
        run_profile=_RUN_PROFILE,
        connection=_connection(scorer_llm=scorer_llm, tutor_llm=observer),
    )

    def handler(event: dict) -> None:
        events.append(event)
        if event.get("type") != "token":
            return
        tokens.append(event["text"])
        if len(tokens) == 1:
            # Deterministic ordering marker (set, never awaited): the consume
            # loop emits the first token before the second, so this records that
            # the first chunk's usage extraction preceded the cancel request.
            first_token_seen.set()
        elif len(tokens) == 2:
            prepared.cancellation.cancel()

    try:
        result = await service.execute_prepared(prepared, events=handler)
    finally:
        controller.shutdown(wait=False)

    # The marker is set: the consume loop's per-chunk order (emit token, then
    # extract usage) ensures the first chunk's usage was consumed before the
    # second token requested the cancel. This is a deterministic ordering
    # observation, not an asynchronous barrier.
    assert first_token_seen.is_set()
    assert tokens == ["po", "ng"], "the third token is refused after cancellation"
    assert observer.chunks[0].usage_metadata == _chunk_usage(4, 2, 6)
    assert observer.chunks[1].usage_metadata == _chunk_usage(0, 3, 3)
    assert len(offline_gemini.requests) == 1

    assert result.run.lifecycle == "cancelled"
    assert result.run.outcome is None
    assert result.run.operational_error_json["code"] == "cancelled"

    assert result.run.candidate_artifact_json is None
    assert result.run.artifact_hash is None
    assert result.score_set is None
    assert result.executions == ()
    assert scorer_llm.calls == 0

    assert recorder.candidates == [], "no candidate is returned"
    assert len(recorder.raised) == 1
    # The cooperative signal is a BaseException, so a plain ``except Exception``
    # in the engine would not catch it. Only the visible boundary is asserted.
    assert not isinstance(recorder.raised[0], Exception)

    assert not any(
        event.get("type") == "trace"
        and event.get("step") == "generation"
        and event.get("status") == "completed"
        for event in events
    ), "no generation-completed trace is emitted"


# ---------------------------------------------------------------------------
# Node 12/12: usage changes the artifact hash, and only the known positions
# ---------------------------------------------------------------------------


async def test_d5c_usage_changes_artifact_hash_only(eval_session, offline_gemini):
    """Two strictly serial runs differ only in usage, so only the artifact hash moves.

    Same text, same corpus and the same ``FakeClock`` for both runs. Removing
    exactly the two known usage positions -- the top-level ``usage`` and the
    generation-completed trace's ``usage`` -- leaves the payloads deep-equal, so
    the hash change is attributed to usage alone. Both hashes are recomputed
    independently and re-read through a fresh Session.
    """
    clock = FakeClock()
    scorer_llm = FakeScorerLLM()

    frames_a = [
        d5a_frame(
            "po",
            usage={"promptTokenCount": 4, "candidatesTokenCount": 2, "totalTokenCount": 6},
        ),
        d5a_frame(
            "ng",
            usage={"promptTokenCount": 4, "candidatesTokenCount": 5, "totalTokenCount": 9},
        ),
        d5a_frame(
            "",
            finish_reason="STOP",
            usage={"promptTokenCount": 4, "candidatesTokenCount": 7, "totalTokenCount": 11},
        ),
    ]
    frames_b = [
        d5a_frame(
            "po",
            usage={"promptTokenCount": 10, "candidatesTokenCount": 3, "totalTokenCount": 13},
        ),
        d5a_frame(
            "ng",
            usage={"promptTokenCount": 10, "candidatesTokenCount": 4, "totalTokenCount": 14},
        ),
        d5a_frame(
            "",
            finish_reason="STOP",
            usage={"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
        ),
    ]

    model_a = offline_gemini.model()
    result_a, recorder_a, _ = await _execute_run(
        eval_session,
        offline_gemini,
        responder=_frames_responder(offline_gemini, frames_a),
        tutor_llm=model_a,
        scorer_llm=scorer_llm,
        clock=clock,
    )
    model_b = offline_gemini.model()
    result_b, recorder_b, _ = await _execute_run(
        eval_session,
        offline_gemini,
        responder=_frames_responder(offline_gemini, frames_b),
        tutor_llm=model_b,
        scorer_llm=scorer_llm,
        clock=clock,
    )

    assert len(recorder_a.candidates) == 1 and len(recorder_b.candidates) == 1
    assert recorder_a.candidates[0].answer == recorder_b.candidates[0].answer == "pong"

    # Same manifest, different artifact hash.
    assert result_a.run.manifest_hash == result_b.run.manifest_hash
    assert result_a.run.artifact_hash != result_b.run.artifact_hash

    a = result_a.run.candidate_artifact_json
    b = result_b.run.candidate_artifact_json
    assert a["usage"] == _candidate_usage(4, 7, 11)
    assert b["usage"] == _candidate_usage(10, 5, 15)

    # The ONLY differing positions are the two known usage spots.
    assert set(a.keys()) == set(b.keys())
    for key in a:
        if key == "usage":
            continue
        if key == "trace":
            assert len(a["trace"]) == len(b["trace"])
            for event_a, event_b in zip(a["trace"], b["trace"]):
                is_completed_generation = (
                    event_a.get("step") == "generation"
                    and event_a.get("status") == "completed"
                )
                if is_completed_generation:
                    assert {
                        k: v for k, v in event_a.items() if k != "usage"
                    } == {k: v for k, v in event_b.items() if k != "usage"}
                    assert event_a["usage"] != event_b["usage"]
                else:
                    assert event_a == event_b
        else:
            assert a[key] == b[key]

    def _strip_known_usage(payload: dict) -> dict:
        stripped = copy.deepcopy(payload)
        stripped["usage"] = None
        for trace_event in stripped["trace"]:
            if (
                trace_event.get("step") == "generation"
                and trace_event.get("status") == "completed"
            ):
                trace_event["usage"] = None
        return stripped

    assert _strip_known_usage(a) == _strip_known_usage(b)

    # Both hashes recomputed independently.
    assert _independent_hash(a) == result_a.run.artifact_hash
    assert _independent_hash(b) == result_b.run.artifact_hash

    # The saved values re-read through a fresh Session prove persistence.
    stored_a = _verified_run(eval_session, result_a.run.id)
    stored_b = _verified_run(eval_session, result_b.run.id)
    assert stored_a.artifact_hash == result_a.run.artifact_hash
    assert stored_b.artifact_hash == result_b.run.artifact_hash
    assert stored_a.manifest_hash == stored_b.manifest_hash
    assert stored_a.candidate_artifact_json["usage"] == _candidate_usage(4, 7, 11)
    assert stored_b.candidate_artifact_json["usage"] == _candidate_usage(10, 5, 15)
