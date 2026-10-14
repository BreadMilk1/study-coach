"""D4b-1: real locked SDK -> real consumer contract evidence.

Scope: the three consumers that sit directly behind an LLM call --
``TutorAttemptEngine.answer``, ``production_judge.judge_response`` and the
production Graph. Everything between the socket and the consumer is real:
``get_chat_model`` builds a real adapter, the real SDK resource and parser run,
and the real consumer code makes the decision. Only the socket is replaced,
through the same seams D4a already validated.

The non-LLM seams are synthetic on purpose and are the only things faked here:
a synthetic ``retriever.search`` and an in-memory ``memory_writer`` spy. The
Judge, its classifier, the Graph router and the production node are never
replaced -- ``build_graph(judge_policy="production_v1")`` is the real thing, and
``judge_response`` is always ``app.agent.production_judge.judge_response``, never
the legacy ``app.agent.judge`` entry.

Every evidence chunk, payload and marker in this file is synthetic. No real
document, provider, key, database or model is involved, and a valid score here
means "the payload was legal", never "the provider really scored it".
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import sysconfig
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from tests.llm.sdk_transport_support import (  # noqa: F401  (frozen, read-only)
    DONE_EVENT,
    ScriptedByteStream,
    assert_module_from_backend,
    deny_sockets,
    fail_with,
    offline_openai,
    send,
    sse_event,
    wait_for,
)
from tests.llm.sdk_transport_support import offline_env as offline_env  # noqa: F401
from tests.llm.test_async_gemini_transport_contract import (  # noqa: F401
    gemini_env,
    offline_gemini,
)

from app.agent.graph import build_graph
from app.agent.judge import load_tutor_rubric
from app.agent.production_judge import (
    TUTOR_DIMENSIONS,
    judge_response as production_judge_response,
)
from app.agent.prompt import TutorPromptTemplate, format_context
from app.agent.tutor_attempt import TutorAttemptConfig, TutorAttemptEngine
from app.llm.content import EMPTY_CONTENT_MESSAGE


# ---------------------------------------------------------------------------
# Synthetic inputs. None of this is a real document, score or credential.
# ---------------------------------------------------------------------------

SYNTHETIC_CHUNK_A = {
    "chunk_id": "synthetic:rrf:1:0",
    "content": "Synthetic note about reciprocal rank fusion.",
    "source": "synthetic-retrieval.pdf",
    "page": 1,
    "score": 0.95,
}
SYNTHETIC_CHUNK_B = {
    "chunk_id": "synthetic:rrf:2:0",
    "content": "Synthetic note about weighted scores.",
    "source": "synthetic-retrieval.pdf",
    "page": 2,
    "score": 0.82,
}
SYNTHETIC_QUESTION = "What does the synthetic retrieval note say?"

# A distinctive marker: the production contract forbids the Judge's free
# reasoning from reaching state, trace or messages. If this string ever shows
# up in an observable output, the contract is broken.
SYNTHETIC_REASONING_MARKER = "SYNTHETIC-JUDGE-REASONING-MARKER-d4b1"

OPENAI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"


class SyntheticRetriever:
    """Allowed non-LLM seam. Records the exact retrieval input."""

    def __init__(self, chunks: list[dict]):
        self.chunks = list(chunks)
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        self.calls.append((query, top_k))
        return list(self.chunks[:top_k])


class MemoryWriterSpy:
    """Allowed non-LLM seam: counts calls, returns the no-op update."""

    def __init__(self):
        self.calls = 0
        self.states: list[dict] = []

    def __call__(self, state) -> dict:
        self.calls += 1
        self.states.append(dict(state))
        return {}


# ---------------------------------------------------------------------------
# Wire builders (small, provider-specific, no shared framework)
# ---------------------------------------------------------------------------


def openai_text_chunk(text: str, *, finish_reason: str | None = None) -> str:
    return json.dumps(
        {
            "id": "chatcmpl-synthetic",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": OPENAI_MODEL,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": text} if text != "" else {},
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
        }
    )


def openai_usage_chunk(
    *, prompt: int = 3, completion: int = 2, total: int = 5
) -> str:
    return json.dumps(
        {
            "id": "chatcmpl-synthetic",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": OPENAI_MODEL,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": total,
            },
        }
    )


def openai_sse(*payloads: str) -> list[bytes]:
    """Return one SSE body as a list of byte chunks for a scripted stream.

    No terminator is appended here: the frozen ``sse_body`` helper owns the
    single ``[DONE]`` terminator, so a response carries it exactly once no
    matter how many frames this builder produced.
    """
    return [(f"data: {p}\n\n").encode() for p in payloads]


def openai_message_body(text: str | None) -> dict:
    message: dict = {"role": "assistant", "refusal": None}
    message["content"] = text
    return {
        "id": "chatcmpl-synthetic",
        "object": "chat.completion",
        "created": 0,
        "model": OPENAI_MODEL,
        "choices": [
            {
                "index": 0,
                "message": message,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def judge_payload(
    *, score: int = 4, reasoning: str = SYNTHETIC_REASONING_MARKER
) -> str:
    """A legal strict rubric payload across all six TUTOR_DIMENSIONS."""
    dims = {dim: score for dim in TUTOR_DIMENSIONS}
    return json.dumps({**dims, "reasoning": reasoning})


def json_responder(payload: dict, *, status: int = 200, headers: dict | None = None):
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=dict(payload), headers=headers or {})

    return responder


def openai_status_responder(status: int):
    """A fresh Google/OpenAI-style error body per attempt (never a reused one)."""
    body = {
        "error": {
            "code": status,
            "message": "synthetic failure",
            "type": "synthetic",
        }
    }

    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=dict(body))

    return responder


def request_json(request: httpx.Request) -> dict:
    return json.loads(request.content)


# ---------------------------------------------------------------------------
# Expected prompt bodies.
#
# These are rendered by the same frozen production builders the consumers call,
# from this file's synthetic inputs. Nothing here compares an observed prompt
# with itself: the expectation is built before the request is sent, and a
# mismatch names the whole rendered prompt.
# ---------------------------------------------------------------------------


def expected_tutor_prompt(chunks: list[dict]) -> str:
    """The exact user content the Tutor must put on the wire for ``chunks``."""
    return TutorPromptTemplate.production_v2().render(SYNTHETIC_QUESTION, chunks)


def expected_judge_prompt(*, answer: str, context: str) -> str:
    """The exact user content the production Judge must put on the wire."""
    return load_tutor_rubric().format(
        question=SYNTHETIC_QUESTION, answer=answer, context=context
    )


# The Judge's inputs are fixed by ``_run_judge`` below, so the expected rubric
# prompt is fixed too and every J case (including every retry) checks against it.
JUDGE_ANSWER = "Synthetic answer body."
JUDGE_CONTEXT = format_context([SYNTHETIC_CHUNK_A])
EXPECTED_JUDGE_PROMPT = expected_judge_prompt(
    answer=JUDGE_ANSWER, context=JUDGE_CONTEXT
)


def assert_openai_chat_request(
    request: httpx.Request, *, streaming: bool, expected_prompt: str
) -> dict:
    """Real request shape and the full expected prompt, not just a count.

    ``expected_prompt`` is required: a call that omitted it would check the
    method/host/path but never the input the consumer actually sent.
    """
    assert request.method == "POST"
    assert request.url.scheme == "https"
    assert request.url.host == "api.openai.com"
    assert request.url.path == "/v1/chat/completions"
    assert "api-key" not in request.headers and "x-goog-api-key" not in request.headers
    body = request_json(request)
    assert body["model"] == OPENAI_MODEL
    assert body["messages"][-1]["role"] == "user"
    if streaming:
        assert body.get("stream") is True
    else:
        assert body.get("stream") is not True
    # The whole user content, not a fragment: one system/user pair only.
    assert body["messages"] == [{"role": "user", "content": expected_prompt}]
    return body


def assert_gemini_generate_content_request(
    request: httpx.Request, *, expected_prompt: str
) -> dict:
    """The Gemini non-stream request shape and its full expected user text."""
    assert request.method == "POST"
    assert request.url.scheme == "https"
    assert request.url.host == "generativelanguage.googleapis.com"
    assert request.url.path == f"/v1beta/models/{GEMINI_MODEL}:generateContent"
    assert ":streamGenerateContent" not in request.url.path
    assert "alt" not in dict(request.url.params)
    body = request_json(request)
    assert body["contents"] == [
        {"role": "user", "parts": [{"text": expected_prompt}]}
    ]
    assert "tools" not in body
    return body


def sse_frame_json(frame: bytes) -> dict:
    """Parse the JSON payload of one SSE ``data:`` frame this file produced.

    The framing is stripped explicitly, so a parse failure can only mean the
    payload itself was not the legal JSON this scenario intends to send.
    """
    assert frame.startswith(b"data: "), f"not a data frame: {frame[:32]!r}"
    return json.loads(frame[len(b"data: ") :])


def assert_single_done(stream: ScriptedByteStream, frames: list[bytes]) -> None:
    """The response that was actually delivered carries one [DONE], once.

    ``openai_sse`` no longer terminates, and the frozen ``sse_body`` adds
    exactly one terminator, so the delivered byte count must equal the frames
    plus exactly one ``DONE_EVENT``.
    """
    assert DONE_EVENT not in b"".join(frames), "frames must not pre-terminate"
    assert stream.bytes_sent == sum(len(frame) for frame in frames) + len(DONE_EVENT)


class RecordingEventSink:
    """The real ``event_sink`` callable TutorAttemptEngine.answer receives.

    ``token_event`` is optional: when given, it is set by the real token event
    itself, so a test can wait for a token that genuinely arrived instead of
    polling a sleep loop.
    """

    def __init__(self, token_event: asyncio.Event | None = None):
        self.events: list[dict] = []
        self.token_event = token_event

    def __call__(self, event: dict) -> None:
        self.events.append(dict(event))
        if self.token_event is not None and event.get("type") == "token":
            self.token_event.set()

    def tokens(self) -> list[str]:
        return [e["text"] for e in self.events if e.get("type") == "token"]

    def traces(self) -> list[dict]:
        return [e for e in self.events if e.get("type") == "trace"]


async def reclaim_own_task(task: asyncio.Task, *, timeout: float = 5.0) -> None:
    """Cancel and bounded-await *this test's own* task, without hiding anything.

    Only the ``CancelledError`` this cleanup itself caused is consumed. Any
    other exception the task raised is re-raised untouched, and a task that does
    not finish within ``timeout`` raises ``TimeoutError`` instead of being left
    dangling for pytest teardown to deal with.
    """
    if not task.done():
        task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=timeout)


class CloseObservingBody(ScriptedByteStream):
    """Local body subclass: proves an actual ``aclose`` happened, via an event.

    Used only where the contract asks for a bounded wait on a real close; it
    observes the SDK's own close, it does not cause one.
    """

    def __init__(self, steps, *, name: str = "tutor-body"):
        super().__init__(steps, name=name)
        self.closed_event = asyncio.Event()

    async def aclose(self) -> None:
        await super().aclose()
        self.closed_event.set()


async def run_tutor(offline, model, *, retriever, sink, **kwargs):
    """Call the real TutorAttemptEngine.answer with production defaults."""
    return await TutorAttemptEngine().answer(
        question=kwargs.pop("question", SYNTHETIC_QUESTION),
        retriever=retriever,
        llm=model,
        prompt_template=TutorPromptTemplate.production_v2(),
        event_sink=sink,
        attempt_config=TutorAttemptConfig.production_default(),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Guard / source selfchecks (not part of the behavioural scenarios)
# ---------------------------------------------------------------------------


def test_socket_deny_is_active_inside_pytest():
    """All four socket entry points are called and denied right now.

    Each one is really invoked, and every socket created here is closed again
    with ``finally``; the deny raises before any connect is attempted, so no
    address below is ever dialled. TEST-NET-1 is used precisely because it is
    reserved for documentation.
    """
    import socket

    with pytest.raises(AssertionError, match="real network access"):
        socket.getaddrinfo("api.openai.com", 443)
    with pytest.raises(AssertionError, match="real network access"):
        socket.create_connection(("api.openai.com", 443))

    connect_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(AssertionError, match="real network access"):
            connect_sock.connect(("192.0.2.1", 443))
    finally:
        connect_sock.close()

    connect_ex_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(AssertionError, match="real network access"):
            connect_ex_sock.connect_ex(("192.0.2.1", 443))
    finally:
        connect_ex_sock.close()

    assert connect_sock.fileno() == -1, "the created socket was closed again"
    assert connect_ex_sock.fileno() == -1, "the created socket was closed again"


def test_openai_models_are_real_clients_on_mock_transport(offline_openai):
    """The seam really is a MockTransport with env proxies off."""
    model = offline_openai.model()
    async_client = model.root_async_client._client
    assert isinstance(async_client, httpx.AsyncClient)
    assert isinstance(async_client._transport, httpx.MockTransport)
    assert async_client.trust_env is False
    assert model.stream_usage is True


def test_gemini_model_is_a_real_client_on_mock_transport(offline_gemini):
    model = offline_gemini.model()
    httpx_client = model.client._api_client._async_httpx_client
    assert isinstance(httpx_client, httpx.AsyncClient)
    assert isinstance(httpx_client._transport, httpx.MockTransport)
    assert httpx_client.trust_env is False
    assert model.vertexai is False


def venv_site_packages() -> Path:
    """The install root of *this* interpreter's site-packages.

    Derived from the running interpreter's own installation info, never from
    the location of a module under test -- deriving the expectation from the
    observed ``openai`` path would make the check tautological. The expected
    ``purelib`` is also cross-checked against ``sys.prefix``, so a wrong
    interpreter is caught instead of silently redefining the target.
    """
    purelib = Path(sysconfig.get_path("purelib")).resolve()
    expected = Path(
        sys.prefix,
        "lib",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
        "site-packages",
    ).resolve()
    assert purelib == expected, (
        f"purelib {purelib} is not the site-packages of this interpreter's "
        f"prefix {sys.prefix} (expected {expected})"
    )
    assert purelib.is_dir(), f"{purelib} does not exist"
    return purelib


def test_source_identity_is_exact_for_this_checkout():
    """Exact resolved equality for project modules and the locked SDKs."""
    import app.agent.graph as graph_module
    import app.agent.production_judge as production_judge_module
    import app.agent.tutor_attempt as tutor_module
    import app.llm.provider as provider_module
    import langchain_google_genai.chat_models as gemini_adapter
    import langchain_openai.chat_models.base as openai_adapter
    import tests.llm.sdk_transport_support as helper_module
    from google.genai import _api_client as genai_sdk

    assert_module_from_backend(provider_module, "app/llm/provider.py")
    assert_module_from_backend(tutor_module, "app/agent/tutor_attempt.py")
    assert_module_from_backend(
        production_judge_module, "app/agent/production_judge.py"
    )
    assert_module_from_backend(graph_module, "app/agent/graph.py")
    assert_module_from_backend(helper_module, "tests/llm/sdk_transport_support.py")
    assert_module_from_backend(
        sys.modules[__name__], "tests/agent/test_sdk_consumer_contract.py"
    )

    # The two adapters and the Google SDK module, pinned to this venv's
    # site-packages -- exact equality, never a prefix match.
    sdk_root = venv_site_packages()
    assert Path(openai_adapter.__file__).resolve() == (
        sdk_root / "langchain_openai" / "chat_models" / "base.py"
    )
    assert Path(gemini_adapter.__file__).resolve() == (
        sdk_root / "langchain_google_genai" / "chat_models.py"
    )
    assert Path(genai_sdk.__file__).resolve() == (
        sdk_root / "google" / "genai" / "_api_client.py"
    )

    # The OpenAI SDK itself is pinned the same way, so the expectation for the
    # two adapters above cannot be satisfied by some other checkout's install.
    assert Path(openai.__file__).resolve() == sdk_root / "openai" / "__init__.py"
    import openai._base_client as openai_base_client

    assert Path(openai_base_client.__file__).resolve() == (
        sdk_root / "openai" / "_base_client.py"
    )


# ---------------------------------------------------------------------------
# T: TutorAttemptEngine (6 scenarios)
# ---------------------------------------------------------------------------


async def test_T1_openai_stream_yields_full_candidate(offline_openai):
    """T1: two non-empty body frames + usage + clean end."""
    frames = openai_sse(
        openai_text_chunk("Reciprocal "),
        openai_text_chunk("rank fusion."),
        openai_usage_chunk(),
    )
    offline_openai.respond(
        lambda request: offline_openai.sse_response(
            offline_openai.sse_body(frames)
        )
    )
    model = offline_openai.model()
    retriever = SyntheticRetriever([SYNTHETIC_CHUNK_A, SYNTHETIC_CHUNK_B])
    sink = RecordingEventSink()

    candidate = await run_tutor(offline_openai, model, retriever=retriever, sink=sink)

    assert len(offline_openai.requests) == 1
    # The real user content is the production_v2 render of *this* case's two
    # chunks -- matched in full, not by a fragment that could hide a rewrite.
    assert_openai_chat_request(
        offline_openai.requests[0],
        streaming=True,
        expected_prompt=expected_tutor_prompt([SYNTHETIC_CHUNK_A, SYNTHETIC_CHUNK_B]),
    )
    assert_single_done(offline_openai.streams[0], frames)

    assert retriever.calls == [(SYNTHETIC_QUESTION, 5)], "production top_k=5"
    assert candidate.answer == "Reciprocal rank fusion."
    assert candidate.citations == [
        {
            "chunk_id": SYNTHETIC_CHUNK_A["chunk_id"],
            "source": SYNTHETIC_CHUNK_A["source"],
            "page": SYNTHETIC_CHUNK_A["page"],
            "span_start": 0,
            "span_end": len(SYNTHETIC_CHUNK_A["content"]),
        },
        {
            "chunk_id": SYNTHETIC_CHUNK_B["chunk_id"],
            "source": SYNTHETIC_CHUNK_B["source"],
            "page": SYNTHETIC_CHUNK_B["page"],
            "span_start": 0,
            "span_end": len(SYNTHETIC_CHUNK_B["content"]),
        },
    ]
    assert candidate.evidence == [SYNTHETIC_CHUNK_A, SYNTHETIC_CHUNK_B]
    assert candidate.formatted_context == format_context(
        [SYNTHETIC_CHUNK_A, SYNTHETIC_CHUNK_B]
    )
    assert candidate.usage == {
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
    }
    assert sink.tokens() == ["Reciprocal ", "rank fusion."]
    traces = sink.traces()
    assert [t["step"] for t in traces] == [
        "retrieval",
        "retrieval",
        "generation",
        "generation",
    ]
    assert traces[1]["status"] == "completed" and traces[1]["chunks_count"] == 2
    assert traces[3]["status"] == "completed"
    assert traces[3]["answer_length"] == len(candidate.answer)
    assert traces[3]["usage"] == candidate.usage
    # The whole trace the candidate carries, not just its last entry.
    assert candidate.trace == sink.traces()
    assert not any(t.get("status") == "failed" for t in traces)
    assert [e for e in sink.events if e.get("type") == "citations"]


async def test_T2_empty_retrieval_with_non_empty_refusal(offline_openai):
    """T2: no sources retrieved, but the model still answers -- not an error."""
    frames = openai_sse(openai_text_chunk("I don't know from the sources."))
    offline_openai.respond(
        lambda request: offline_openai.sse_response(
            offline_openai.sse_body(frames)
        )
    )
    model = offline_openai.model()
    retriever = SyntheticRetriever([])
    sink = RecordingEventSink()

    candidate = await run_tutor(offline_openai, model, retriever=retriever, sink=sink)

    assert len(offline_openai.requests) == 1
    # This case's actual chunks are empty, so the prompt must be the empty
    # retrieval render -- checked in full.
    assert_openai_chat_request(
        offline_openai.requests[0],
        streaming=True,
        expected_prompt=expected_tutor_prompt([]),
    )
    assert_single_done(offline_openai.streams[0], frames)

    assert candidate.answer == "I don't know from the sources."
    assert candidate.citations == []
    assert candidate.evidence == []
    assert candidate.formatted_context == "(no relevant sources retrieved)"
    assert candidate.usage == "unavailable"
    assert not any(t.get("status") == "failed" for t in sink.traces())
    assert sink.tokens() == ["I don't know from the sources."]


async def test_T3_legal_empty_delta_is_a_failed_generation(offline_openai):
    """T3: legal empty delta + [DONE] -> exact ValueError, not a fake model fault."""
    frames = openai_sse(openai_text_chunk(""))
    offline_openai.respond(
        lambda request: offline_openai.sse_response(
            offline_openai.sse_body(frames)
        )
    )
    model = offline_openai.model()
    sink = RecordingEventSink()

    with pytest.raises(ValueError) as excinfo:
        await run_tutor(
            offline_openai,
            model,
            retriever=SyntheticRetriever([SYNTHETIC_CHUNK_A]),
            sink=sink,
        )

    assert type(excinfo.value) is ValueError
    assert str(excinfo.value) == EMPTY_CONTENT_MESSAGE
    assert len(offline_openai.requests) == 1
    assert_openai_chat_request(
        offline_openai.requests[0],
        streaming=True,
        expected_prompt=expected_tutor_prompt([SYNTHETIC_CHUNK_A]),
    )
    # Input precondition, parsed from the exact bytes that were sent: this was a
    # *legal* empty delta -- one real chunk frame with delta {} -- not empty
    # choices and not a malformed schema. ``sse_body`` adds the single [DONE].
    assert len(frames) == 1
    first_frame = sse_frame_json(frames[0])
    assert first_frame["object"] == "chat.completion.chunk"
    assert first_frame["model"] == OPENAI_MODEL
    assert len(first_frame["choices"]) == 1
    assert first_frame["choices"][0]["delta"] == {}
    assert_single_done(offline_openai.streams[0], frames)
    assert sink.tokens() == []
    traces = sink.traces()
    assert traces[-1]["step"] == "generation"
    assert traces[-1]["status"] == "failed"
    assert traces[-1]["error"] == "ValueError"
    assert not any(
        t.get("status") == "completed" and t.get("step") == "generation" for t in traces
    )


@pytest.mark.parametrize(
    "exc_cls,label",
    [
        pytest.param(httpx.ReadError, "read-error", id="readerror"),
        pytest.param(httpx.ReadTimeout, "read-timeout", id="readtimeout"),
    ],
)
async def test_T4_T5_prefix_then_body_failure(offline_openai, exc_cls, label):
    """T4/T5: a delivered prefix survives an original HTTPX body failure."""
    injected = exc_cls(f"synthetic {label} mid body")
    body = offline_openai.body(
        [
            send(sse_event(openai_text_chunk("Partial "))),
            fail_with(injected),
        ]
    )
    offline_openai.respond(lambda request: offline_openai.sse_response(body))
    model = offline_openai.model()
    sink = RecordingEventSink()

    with pytest.raises(exc_cls) as excinfo:
        await run_tutor(
            offline_openai,
            model,
            retriever=SyntheticRetriever([SYNTHETIC_CHUNK_A]),
            sink=sink,
        )

    assert type(excinfo.value) is exc_cls
    assert excinfo.value is injected, "the captured object is the injected one"
    assert len(offline_openai.requests) == 1, "a body failure is not replayed"
    assert_openai_chat_request(
        offline_openai.requests[0],
        streaming=True,
        expected_prompt=expected_tutor_prompt([SYNTHETIC_CHUNK_A]),
    )
    assert sink.tokens() == ["Partial "], "prefix tokens already emitted remain"
    traces = sink.traces()
    assert traces[-1]["status"] == "failed"
    assert traces[-1]["step"] == "generation"
    assert traces[-1]["error"] == exc_cls.__name__
    assert not any(
        t.get("status") == "completed" and t.get("step") == "generation" for t in traces
    )
    # Observed before the fixture actively closes the SDK clients.
    assert body.close_count >= 1


async def test_T6_cancel_after_prefix_keeps_prefix_and_closes(offline_openai):
    """T6: cancel the answer task after a prefix event; wait for a real close.

    Production cancels the inner stream task without awaiting its cleanup, so
    the close is observed with a bounded wait on an event set by the body's own
    ``aclose`` -- never by releasing the gate to fake a cancellation.

    Every wait is event-driven and bounded, and the ``finally`` only reclaims
    this test's own task; it is not, and never becomes, close evidence.
    """
    gate = asyncio.Event()  # never set: the body stays open until cancellation
    body = CloseObservingBody(
        [
            send(sse_event(openai_text_chunk("Prefix "))),
            wait_for(gate),
            send(sse_event(openai_text_chunk("late"))),
        ]
    )
    offline_openai.respond(lambda request: offline_openai.sse_response(body))
    model = offline_openai.model()
    prefix_seen = asyncio.Event()
    sink = RecordingEventSink(prefix_seen)

    task = asyncio.create_task(
        run_tutor(
            offline_openai,
            model,
            retriever=SyntheticRetriever([SYNTHETIC_CHUNK_A]),
            sink=sink,
        )
    )
    try:
        # Gate on the real emitted prefix event, not on a polling loop.
        await asyncio.wait_for(prefix_seen.wait(), timeout=5)
        assert sink.tokens() == ["Prefix "]

        task.cancel()
        # This wait carries its own deadline: if the cancellation never returns,
        # the bounded wait raises TimeoutError and the case fails, rather than
        # hanging until the 180s process limit. ``shield`` keeps that local
        # timeout from cancelling the consumer task a second time; the explicit
        # cancel above is the only cancellation this test performs.
        with pytest.raises(asyncio.CancelledError) as excinfo:
            await asyncio.wait_for(asyncio.shield(task), timeout=5)
        assert type(excinfo.value) is asyncio.CancelledError
        assert task.cancelled()

        # The inner stream task's cleanup may finish shortly after answer()
        # returns; the event is set by the body's own aclose, never by the sink
        # or by this cleanup.
        await asyncio.wait_for(body.closed_event.wait(), timeout=5)
        assert body.close_count >= 1, "a local aclose happened before the finally"

        assert len(offline_openai.requests) == 1
        assert_openai_chat_request(
            offline_openai.requests[0],
            streaming=True,
            expected_prompt=expected_tutor_prompt([SYNTHETIC_CHUNK_A]),
        )
        assert sink.tokens() == ["Prefix "], "no fabricated token after cancellation"
        traces = sink.traces()
        assert not any(t.get("status") == "failed" for t in traces), (
            "no fabricated failed trace"
        )
        assert not any(
            t.get("status") == "completed" and t.get("step") == "generation"
            for t in traces
        ), "no generation completed"
    finally:
        await reclaim_own_task(task)


# ---------------------------------------------------------------------------
# J: production Judge (8 scenarios)
# ---------------------------------------------------------------------------


async def _run_judge(judge_llm) -> dict:
    return await production_judge_response(
        question=SYNTHETIC_QUESTION,
        answer=JUDGE_ANSWER,
        context=JUDGE_CONTEXT,
        rubric=load_tutor_rubric(),
        judge_llm=judge_llm,
    )


def assert_judge_requests(
    requests, *, streaming: bool, expected_prompt: str = EXPECTED_JUDGE_PROMPT
) -> None:
    """Every single attempt carried the exact expected rubric prompt.

    Retries are not skipped: each recorded request is checked, so a prompt that
    only the first attempt got right still fails here.
    """
    assert requests, "no Judge request was recorded"
    for request in requests:
        if streaming:
            assert_openai_chat_request(
                request, streaming=False, expected_prompt=expected_prompt
            )
        else:
            assert_gemini_generate_content_request(
                request, expected_prompt=expected_prompt
            )


async def test_J1_openai_non_stream_valid_rubric(offline_openai):
    offline_openai.respond(json_responder(openai_message_body(judge_payload())))
    model = offline_openai.model()

    result = await _run_judge(model)

    assert len(offline_openai.requests) == 1
    assert_judge_requests(offline_openai.requests, streaming=True)

    assert result["status"] == "valid"
    assert result["score"] == 0.8
    assert result["verdict"] == "pass"
    assert result["weak_dims"] == []
    # Calling the function directly legitimately exposes the reasoning; the
    # public boundary is what the Graph scenarios check.
    assert result["reasoning"] == SYNTHETIC_REASONING_MARKER


async def test_J2_openai_503_is_unavailable_after_real_sdk_retries(offline_openai):
    offline_openai.respond(openai_status_responder(503))
    model = offline_openai.model()

    result = await _run_judge(model)

    assert len(offline_openai.requests) == 3, "the real SDK retried 5xx three times"
    assert_judge_requests(offline_openai.requests, streaming=True)
    assert result["status"] == "unavailable"
    assert result["score"] is None
    assert result["verdict"] is None
    assert SYNTHETIC_REASONING_MARKER not in result["reason"]
    assert "synthetic failure" not in result["reason"], "safe reason, no payload leak"


async def test_J3_gemini_401_is_unavailable(offline_gemini):
    offline_gemini.respond(json_responder({"error": {"code": 401, "message": "synthetic", "status": "UNAUTHENTICATED"}}, status=401))
    model = offline_gemini.model()

    result = await _run_judge(model)

    assert len(offline_gemini.requests) == 1, "401 is not retried by the real SDK"
    assert_judge_requests(offline_gemini.requests, streaming=False)
    assert result["status"] == "unavailable"
    assert result["score"] is None
    assert result["verdict"] is None
    assert "synthetic" not in result["reason"]


async def test_J4_openai_empty_content_is_unavailable(offline_openai):
    """A legal ``message.content=''`` body is the fixed invalid-payload result."""
    offline_openai.respond(json_responder(openai_message_body("")))
    model = offline_openai.model()

    result = await _run_judge(model)

    assert len(offline_openai.requests) == 1
    assert_judge_requests(offline_openai.requests, streaming=True)
    assert result["status"] == "unavailable"
    assert result["score"] is None
    assert result["verdict"] is None
    assert result["reason"] == "Judge response was not a valid scoring payload."
    assert "reasoning" not in result, "an unavailable result carries no reasoning"


@pytest.mark.parametrize(
    "exc_cls,label",
    [
        pytest.param(httpx.ReadTimeout, "read-timeout", id="readtimeout"),
        pytest.param(httpx.ConnectError, "connect-error", id="connecterror"),
    ],
)
async def test_J5_J6_gemini_send_failures_propagate_raw(offline_gemini, exc_cls, label):
    """J5/J6: the Judge's approved class list does not cover these HTTPX types.

    This is the source-derived expectation, asserted as observed: they are not
    converted to unavailable.
    """
    seen: list[BaseException] = []

    def responder(request: httpx.Request) -> httpx.Response:
        err = exc_cls(f"synthetic {label}")
        seen.append(err)
        raise err

    offline_gemini.respond(responder)
    model = offline_gemini.model()

    with pytest.raises(exc_cls) as excinfo:
        await _run_judge(model)

    assert type(excinfo.value) is exc_cls
    assert excinfo.value is seen[-1]
    assert len(offline_gemini.requests) == 6, "the real SDK retried to exhaustion"
    assert len(seen) == 6, "each attempt builds its own exception"
    # All six attempts, not just the last one, carried the rubric prompt.
    assert_judge_requests(offline_gemini.requests, streaming=False)


async def test_J7_openai_null_message_propagates_attribute_error(offline_openai):
    """A real ``message: null`` body still reaches the consumer's AttributeError."""
    payload = openai_message_body("x")
    payload["choices"][0]["message"] = None
    offline_openai.respond(json_responder(payload))
    model = offline_openai.model()

    with pytest.raises(AttributeError) as excinfo:
        await _run_judge(model)

    assert type(excinfo.value) is AttributeError
    assert len(offline_openai.requests) == 1
    assert_judge_requests(offline_openai.requests, streaming=True)


async def test_J8_openai_cancelled_body_read_propagates(offline_openai):
    """J8: explicit task cancellation on a gated real HTTPX body.

    Close timing for this path is deliberately not asserted. ``close_count`` is
    only *recorded*, after the cancellation returned and before this test's own
    cleanup or the fixture's client close, as one fixed synthetic diagnostic
    line. It proves nothing about immediacy, connection release or billing, and
    it is not promoted to a product rule.
    """
    gate = asyncio.Event()  # never set: the body stays open until cancellation
    body = offline_openai.body([wait_for(gate), send(b"never-reached")])
    offline_openai.respond(
        lambda request: httpx.Response(
            200, json={}, headers={"content-type": "application/json"}, stream=body
        )
    )
    model = offline_openai.model()

    task = asyncio.create_task(_run_judge(model))
    try:
        await asyncio.wait_for(body.started.wait(), timeout=5)
        task.cancel()
        # This wait carries its own deadline: if the cancellation never returns,
        # the bounded wait raises TimeoutError and the case fails, rather than
        # hanging until the 180s process limit. ``shield`` keeps that local
        # timeout from cancelling the consumer task a second time; the explicit
        # cancel above is the only cancellation this test performs.
        with pytest.raises(asyncio.CancelledError) as excinfo:
            await asyncio.wait_for(asyncio.shield(task), timeout=5)
        assert type(excinfo.value) is asyncio.CancelledError
        assert task.cancelled()

        assert len(offline_openai.requests) == 1
        assert_judge_requests(offline_openai.requests, streaming=True)

        # Observed here: before this test's cleanup and before the fixture
        # closes the SDK clients. Recorded, never asserted.
        pre_close_count = body.close_count
        assert isinstance(pre_close_count, int)
        print(f"D4b J8 fixture_pre_close_count={pre_close_count}")
    finally:
        await reclaim_own_task(task)


# ---------------------------------------------------------------------------
# G: production Graph (2 scenarios)
# ---------------------------------------------------------------------------


def _split_openai_tutor_and_judge(offline, tutor_response, judge_response):
    """One harness, two real models; stream flag selects the responder.

    Building both models from the same harness keeps a single set of patched
    factories, so the two consumers can never overwrite each other's seam.
    """

    def responder(request: httpx.Request) -> httpx.Response:
        if request_json(request).get("stream"):
            return tutor_response(request)
        return judge_response(request)

    offline.respond(responder)
    return offline.model(), offline.model()


async def _drive_graph(graph, *, thread_id, configurable=None, **initial):
    """Observe the real Tutor candidate, then the final state.

    ``values`` is used rather than ``updates`` for the candidate: the raw node
    update carries an ``id=None`` AIMessage, because ``add_messages`` assigns the
    real id when it merges into state. The first state view per message id is
    therefore the genuine, observed Tutor candidate -- before any Judge
    replacement rewrites its content.

    ``configurable`` is merged into the run config, never into the input state:
    state fields are checkpointed, and the callables are not serialisable.
    """
    config = {
        "configurable": {"thread_id": thread_id, **(configurable or {})}
    }
    observed: dict[str, Any] = {"candidates": [], "custom": [], "seen_ids": set()}
    async for mode, chunk in graph.astream(
        initial,
        config=config,
        stream_mode=["values", "custom"],
    ):
        if mode == "values":
            for message in chunk.get("messages", []):
                if not isinstance(message, AIMessage):
                    continue
                if message.id in observed["seen_ids"]:
                    continue
                observed["seen_ids"].add(message.id)
                observed["candidates"].append(
                    {
                        "id": message.id,
                        "content": message.content,
                        "response_metadata": dict(
                            getattr(message, "response_metadata", {}) or {}
                        ),
                        "additional_kwargs": dict(
                            getattr(message, "additional_kwargs", {}) or {}
                        ),
                    }
                )
        else:
            observed["custom"].append(chunk)
    return observed, await graph.aget_state(config)


async def test_G1_production_graph_pass_path(offline_openai):
    """G1: Tutor streams, Judge scores 4/4 -> valid pass, one memory write."""
    tutor_frames = openai_sse(
        openai_text_chunk("Reciprocal "), openai_text_chunk("rank fusion.")
    )
    tutor_llm, judge_llm = _split_openai_tutor_and_judge(
        offline_openai,
        lambda request: offline_openai.sse_response(
            offline_openai.sse_body(tutor_frames)
        ),
        json_responder(openai_message_body(judge_payload())),
    )
    retriever = SyntheticRetriever([SYNTHETIC_CHUNK_A])
    writer = MemoryWriterSpy()
    graph = build_graph(
        retriever=retriever,
        llm=tutor_llm,
        checkpointer=InMemorySaver(),
        judge_policy="production_v1",
    )

    observed, final_state = await _drive_graph(
        graph,
        thread_id="d4b1-g1",
        messages=[HumanMessage(content=SYNTHETIC_QUESTION)],
        configurable={"judge_llm": judge_llm, "memory_writer": writer},
    )

    assert len(offline_openai.requests) == 2, "exactly one Tutor + one Judge call"
    assert request_json(offline_openai.requests[0]).get("stream") is True
    assert request_json(offline_openai.requests[1]).get("stream") is not True

    candidates = observed["candidates"]
    assert len(candidates) == 1, "no quality retry happened"
    candidate = candidates[0]
    assert candidate["content"] == "Reciprocal rank fusion."

    # The Tutor request: the plain production_v2 render, single chunk, no retry
    # suffix -- and exactly one [DONE] in the response that was delivered.
    assert_openai_chat_request(
        offline_openai.requests[0],
        streaming=True,
        expected_prompt=expected_tutor_prompt([SYNTHETIC_CHUNK_A]),
    )
    assert_single_done(offline_openai.streams[0], tutor_frames)
    # The Judge request: the rubric prompt for *this* Tutor answer and this
    # retrieval context, not for some other case's inputs.
    assert_openai_chat_request(
        offline_openai.requests[1],
        streaming=False,
        expected_prompt=expected_judge_prompt(
            answer=candidate["content"], context=format_context([SYNTHETIC_CHUNK_A])
        ),
    )
    assert retriever.calls == [(SYNTHETIC_QUESTION, 5)]

    ai_msgs = [m for m in final_state.values["messages"] if isinstance(m, AIMessage)]
    assert len(ai_msgs) == 1
    assert ai_msgs[0].content == candidate["content"]
    assert ai_msgs[0].id == candidate["id"]
    assert ai_msgs[0].response_metadata == candidate["response_metadata"]
    assert ai_msgs[0].additional_kwargs == candidate["additional_kwargs"]

    assert final_state.values["judge_status"] == "valid"
    assert final_state.values["judge_score"] == 0.8
    assert final_state.values["retry_count"] == 0
    assert writer.calls == 1, "pass routes through the memory writer once"

    # The Judge's free reasoning never reaches any observable boundary.
    assert SYNTHETIC_REASONING_MARKER not in json.dumps(
        final_state.values, default=str
    )
    assert all(
        SYNTHETIC_REASONING_MARKER not in json.dumps(event, default=str)
        for event in observed["custom"]
    )


async def test_G2_production_graph_unavailable_appends_suffix(offline_openai):
    """G2: Tutor streams, Judge 503 -> kept content + fixed warning, no write."""
    tutor_frames = openai_sse(openai_text_chunk("Reciprocal rank fusion."))
    tutor_llm, judge_llm = _split_openai_tutor_and_judge(
        offline_openai,
        lambda request: offline_openai.sse_response(
            offline_openai.sse_body(tutor_frames)
        ),
        openai_status_responder(503),
    )
    retriever = SyntheticRetriever([SYNTHETIC_CHUNK_A])
    writer = MemoryWriterSpy()
    graph = build_graph(
        retriever=retriever,
        llm=tutor_llm,
        checkpointer=InMemorySaver(),
        judge_policy="production_v1",
    )

    observed, final_state = await _drive_graph(
        graph,
        thread_id="d4b1-g2",
        messages=[HumanMessage(content=SYNTHETIC_QUESTION)],
        # Synthetic non-empty pending values: the clearing assertions below are
        # only meaningful if something was actually there to clear.
        pending_mastery_delta={"synthetic-topic": 0.5},
        pending_mistake={"question_id": "synthetic-question"},
        configurable={"judge_llm": judge_llm, "memory_writer": writer},
    )

    tutor_requests = [
        r for r in offline_openai.requests if request_json(r).get("stream") is True
    ]
    judge_requests = [
        r for r in offline_openai.requests if request_json(r).get("stream") is not True
    ]
    assert len(tutor_requests) == 1, "no quality regeneration"
    assert len(judge_requests) == 3, "the real SDK retried the 503 three times"

    candidates = observed["candidates"]
    assert len(candidates) == 1
    candidate = candidates[0]
    body = candidate["content"]

    assert_openai_chat_request(
        tutor_requests[0],
        streaming=True,
        expected_prompt=expected_tutor_prompt([SYNTHETIC_CHUNK_A]),
    )
    assert_single_done(offline_openai.streams[0], tutor_frames)
    # All three Judge attempts carried the rubric prompt for this Tutor answer.
    assert_judge_requests(
        judge_requests,
        streaming=True,
        expected_prompt=expected_judge_prompt(
            answer=body, context=format_context([SYNTHETIC_CHUNK_A])
        ),
    )

    ai_msgs = [m for m in final_state.values["messages"] if isinstance(m, AIMessage)]
    assert len(ai_msgs) == 1
    final_msg = ai_msgs[0]
    assert final_msg.id == candidate["id"]
    assert final_msg.response_metadata == candidate["response_metadata"]
    assert final_msg.additional_kwargs == candidate["additional_kwargs"]
    assert body in final_msg.content, "kept content is still present"
    assert final_msg.content == body + graph_unavailable_suffix()

    assert final_state.values["judge_status"] == "unavailable"
    assert final_state.values["judge_score"] is None
    assert final_state.values["retry_count"] == 0
    assert final_state.values["pending_mastery_delta"] == {}
    assert final_state.values["pending_mistake"] is None
    assert writer.calls == 0, "unavailable skips the memory writer"

    blob = json.dumps(final_state.values, default=str) + json.dumps(
        observed["custom"], default=str
    )
    assert "synthetic failure" not in blob, "provider detail never surfaces"
    assert SYNTHETIC_REASONING_MARKER not in blob


def graph_unavailable_suffix() -> str:
    """The exact fixed suffix production appends (read from the source module)."""
    from app.agent.graph import _UNAVAILABLE_SUFFIX

    return _UNAVAILABLE_SUFFIX
