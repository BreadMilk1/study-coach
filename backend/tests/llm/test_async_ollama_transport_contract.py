"""D4a-3: offline evidence for the real Ollama SDK's async NDJSON transport.

Scope is one provider only. The chain that runs for real is ``get_chat_model``
-> ``ChatOllama`` -> a real ``ollama.AsyncClient`` -> the real ``chat`` resource
-> the SDK's real NDJSON line parsing -> the adapter's own aggregation /
``astream`` conversion.

Unlike D4a-1 and D4a-2, **nothing is patched here**. The mock transport reaches
the SDK through the adapter's existing transport configuration entry
(``async_client_kwargs`` / ``sync_client_kwargs``), which is a documented
product-level knob. So the factory, both SDK clients, ``chat``, the NDJSON
parser and the aggregation all stay real.

Ollama has no SDK retry of its own, so this file pins that absence rather than
inventing a retry policy: an HTTP failure is exactly one request. The SDK's
``stream=True`` branch also does not translate ``httpx`` transport errors the
way its non-stream branch does, so the narrow original exception is recorded as
it actually surfaces.

Nothing here is normalised into a product-level "unavailable" result, and no
test swallows a surprising error to make itself pass.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from typing import Any, Callable

import httpx
import ollama
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from pydantic import ValidationError

from tests.llm.sdk_transport_support import (  # noqa: F401  (shared, read-only)
    ScriptedByteStream,
    assert_module_from_backend,
    deny_sockets,
    fail_with,
    observe_explicit_cancellation,
    reclaim_own_task,
    send,
    wait_for,
)

REPRESENTATIVE_MODEL = "gemma3:4b"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"

OLLAMA_STEERING_ENV_VARS = (
    "OLLAMA_HOST",
    "OLLAMA_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_AUTH_TOKEN",
)


# ---------------------------------------------------------------------------
# NDJSON wire builders
#
# One JSON object per line. Content frames and the final done frame are built
# separately so "empty content", "no done frame" and "no lines at all" stay
# clearly separate shapes rather than one vague "empty response".
# ---------------------------------------------------------------------------


def content_line(text: str | None = "", *, done: bool = False) -> str:
    return json.dumps(
        {
            "model": REPRESENTATIVE_MODEL,
            "created_at": "2026-10-13T00:00:00Z",
            "message": {"role": "assistant", "content": text},
            "done": done,
        }
    )


def done_line(
    text: str | None = "",
    *,
    done_reason: str = "stop",
    raw: dict | None = None,
) -> str:
    if raw is not None:
        return json.dumps(raw)
    return json.dumps(
        {
            "model": REPRESENTATIVE_MODEL,
            "created_at": "2026-10-13T00:00:00Z",
            "message": {"role": "assistant", "content": text},
            "done": True,
            "done_reason": done_reason,
            "total_duration": 1000,
            "eval_count": 2,
        }
    )


def ndjson(*lines: str) -> list[bytes]:
    """Each line must end with a newline; httpx splits the body on newlines."""
    return [(line + "\n").encode() for line in lines]


def normal_stream(*texts: str) -> list[bytes]:
    lines = [content_line(t, done=False) for t in texts]
    lines.append(done_line(""))
    return ndjson(*lines)


def json_response(payload: dict, *, status: int = 200, headers: dict | None = None):
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload, headers=headers or {})

    return responder


def raise_this(exc: BaseException):
    """Raise one specific pre-built instance.

    Used by the transport matrices so a test can prove the captured object is
    that very instance, not merely something ``raises(exc_cls)`` would accept.
    """

    def responder(request: httpx.Request) -> httpx.Response:
        raise exc

    return responder


def request_json(request: httpx.Request) -> dict:
    return json.loads(request.content)


# ---------------------------------------------------------------------------
# Harness (nothing patched; transport supplied via the adapter's own kwargs)
# ---------------------------------------------------------------------------


class OfflineOllama:
    """Records the NDJSON exchange while both SDK clients talk to a mock."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.streams: list[ScriptedByteStream] = []
        self.models: list[Any] = []
        self._responder: Callable[[httpx.Request], httpx.Response] | None = None

    def respond(self, responder) -> "OfflineOllama":
        self._responder = responder
        return self

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self._responder is None:
            raise AssertionError("offline transport has no responder installed")
        return self._responder(request)

    def body(self, steps, *, name: str = "ndjson") -> ScriptedByteStream:
        stream = ScriptedByteStream(steps, name=name)
        self.streams.append(stream)
        return stream

    def ndjson_response(
        self,
        stream: ScriptedByteStream,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        return httpx.Response(
            status,
            headers={"content-type": "application/x-ndjson", **(headers or {})},
            stream=stream,
        )

    def ndjson_body(self, lines, *, name: str = "ndjson") -> ScriptedByteStream:
        return self.body([send(line) for line in lines], name=name)

    def model(self, **kwargs):
        """Build through the product factory, using the adapter's own transport
        configuration entry. No patching of factory, client, chat or parser."""
        from app.llm.provider import LLMConfig, get_chat_model

        config = LLMConfig(provider="ollama", model=REPRESENTATIVE_MODEL)
        built = get_chat_model(
            config,
            sync_client_kwargs={
                "transport": httpx.MockTransport(self.handle),
                "trust_env": False,
            },
            async_client_kwargs={
                "transport": httpx.MockTransport(self.handle),
                "trust_env": False,
            },
            **kwargs,
        )
        self.models.append(built)
        return built

    async def aclose(self) -> None:
        """Close both real SDK clients the adapter created."""
        for model in self.models:
            client = getattr(model, "_client", None)
            if client is not None:
                client.close()
            async_client = getattr(model, "_async_client", None)
            if async_client is not None:
                await async_client.close()
        self.models.clear()


@pytest.fixture(autouse=True)
def ollama_env(monkeypatch):
    """Ollama's own steering variables, cleaned and restored per test.

    The URL is localhost, but that does not relax the network guard: no real
    Ollama is contacted and no model is started or downloaded.
    """
    for name in OLLAMA_STEERING_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("STUDY_COACH_TEST_MODE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    yield


@pytest.fixture
async def offline_ollama() -> AsyncIterator[OfflineOllama]:
    harness = OfflineOllama()
    try:
        yield harness
    finally:
        await harness.aclose()


# ---------------------------------------------------------------------------
# Harness self-checks: isolation, default config, source identity
# ---------------------------------------------------------------------------


def test_offline_harness_rejects_dns_and_connect_attempts():
    """The socket deny is real, not a printed claim."""
    import socket

    with pytest.raises(AssertionError, match="real network access"):
        socket.getaddrinfo("127.0.0.1", 11434)
    with pytest.raises(AssertionError, match="real network access"):
        socket.create_connection(("127.0.0.1", 11434))


def test_offline_harness_pins_default_client_config(offline_ollama):
    """Both real SDK clients exist, on a mock transport, with SDK defaults."""
    model = offline_ollama.model()

    assert type(model).__name__ == "ChatOllama"
    assert model.model == REPRESENTATIVE_MODEL
    assert model.base_url is None, "no base_url was supplied"
    # Model probing on init would be a real network call; it must be off.
    assert model.validate_model_on_init is False

    for sdk_client in (model._client, model._async_client):
        httpx_client = sdk_client._client
        assert isinstance(httpx_client, httpx.Client if sdk_client is model._client else httpx.AsyncClient)
        assert isinstance(httpx_client._transport, httpx.MockTransport)
        assert httpx_client.trust_env is False
        assert httpx_client.follow_redirects is True, "the Ollama SDK sets this True"
        assert httpx_client.timeout.connect is None
        assert httpx_client.timeout.read is None
        assert httpx_client.timeout.write is None
        assert httpx_client.timeout.pool is None
        assert str(httpx_client.base_url).rstrip("/") == DEFAULT_OLLAMA_HOST


async def test_offline_harness_request_has_no_auth_header(offline_ollama):
    """Ollama needs no key; nothing should manufacture an auth header."""
    offline_ollama.respond(
        lambda request: offline_ollama.ndjson_response(
            offline_ollama.ndjson_body(normal_stream("ok"))
        )
    )
    model = offline_ollama.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_ollama.requests) == 1
    headers = offline_ollama.requests[0].headers
    assert "authorization" not in headers
    assert "x-api-key" not in headers
    assert "".join(c.content for c in chunks if isinstance(c.content, str)) == "ok"


def test_source_identity_is_portable_and_exact():
    """app code and helpers come from this checkout, resolved from this file."""
    import app.llm.provider as provider_module
    import tests.llm.sdk_transport_support as support_module

    from tests.llm.sdk_transport_support import backend_root

    root = backend_root()
    assert root.name == "backend"
    assert (root / "pyproject.toml").is_file()

    assert_module_from_backend(provider_module, "app/llm/provider.py")
    assert_module_from_backend(support_module, "tests/llm/sdk_transport_support.py")
    assert_module_from_backend(
        sys.modules[__name__], "tests/llm/test_async_ollama_transport_contract.py"
    )


# ---------------------------------------------------------------------------
# 1. Happy path for both entries
# ---------------------------------------------------------------------------


async def test_ainvoke_aggregates_ndjson_content(offline_ollama):
    offline_ollama.respond(
        lambda request: offline_ollama.ndjson_response(
            offline_ollama.ndjson_body(normal_stream("po", "ng"))
        )
    )
    model = offline_ollama.model()

    result = await model.ainvoke("ping")

    assert len(offline_ollama.requests) == 1
    request = offline_ollama.requests[0]
    assert request.method == "POST"
    assert str(request.url) == f"{DEFAULT_OLLAMA_HOST}/api/chat"
    assert "authorization" not in request.headers

    body = request_json(request)
    assert body["model"] == REPRESENTATIVE_MODEL
    assert body["messages"] == [{"role": "user", "content": "ping"}]
    assert body["stream"] is True, "ainvoke still takes the NDJSON streaming branch"

    assert isinstance(result, AIMessage)
    assert result.content == "pong"


async def test_astream_delivers_content_chunks(offline_ollama):
    offline_ollama.respond(
        lambda request: offline_ollama.ndjson_response(
            offline_ollama.ndjson_body(normal_stream("po", "ng"))
        )
    )
    model = offline_ollama.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_ollama.requests) == 1
    # MockTransport accepts any path, so the real streaming request shape is
    # pinned here too, not only in the ainvoke case.
    request = offline_ollama.requests[0]
    assert request.method == "POST"
    assert str(request.url) == f"{DEFAULT_OLLAMA_HOST}/api/chat"
    assert "authorization" not in request.headers
    assert "x-api-key" not in request.headers

    body = request_json(request)
    assert body["model"] == REPRESENTATIVE_MODEL
    assert body["messages"] == [{"role": "user", "content": "ping"}]
    assert body["stream"] is True

    assert all(isinstance(c, AIMessageChunk) for c in chunks)
    assert "".join(c.content for c in chunks if isinstance(c.content, str)) == "pong"


# ---------------------------------------------------------------------------
# 2. Empty content vs zero-line EOF (kept distinct)
# ---------------------------------------------------------------------------


async def test_ainvoke_legal_empty_content_frame(offline_ollama):
    """content:"" with a normal done_reason is a legal, empty answer."""
    offline_ollama.respond(
        lambda request: offline_ollama.ndjson_response(
            offline_ollama.ndjson_body(ndjson(content_line(""), done_line("")))
        )
    )
    model = offline_ollama.model()

    result = await model.ainvoke("ping")

    assert len(offline_ollama.requests) == 1
    assert isinstance(result, AIMessage)
    assert result.content == ""


async def test_astream_legal_empty_content_frame(offline_ollama):
    offline_ollama.respond(
        lambda request: offline_ollama.ndjson_response(
            offline_ollama.ndjson_body(ndjson(content_line(""), done_line("")))
        )
    )
    model = offline_ollama.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert "".join(c.content for c in chunks if isinstance(c.content, str)) == ""


async def test_ainvoke_zero_line_eof_raises_no_data_received(offline_ollama):
    """A body with no NDJSON line at all is not an empty answer."""
    offline_ollama.respond(
        lambda request: offline_ollama.ndjson_response(
            offline_ollama.ndjson_body([])
        )
    )
    model = offline_ollama.model()

    with pytest.raises(ValueError, match="No data received from Ollama stream."):
        await model.ainvoke("ping")

    assert len(offline_ollama.requests) == 1


async def test_astream_zero_line_eof_raises_actual_error(offline_ollama):
    """Ollama's own message differs from ainvoke's, and both are pinned."""
    offline_ollama.respond(
        lambda request: offline_ollama.ndjson_response(
            offline_ollama.ndjson_body([])
        )
    )
    model = offline_ollama.model()

    with pytest.raises(ValueError, match="No generation chunks were returned"):
        async for _ in model.astream("ping"):
            pass

    assert len(offline_ollama.requests) == 1


# ---------------------------------------------------------------------------
# 3. Bad wire
# ---------------------------------------------------------------------------


async def test_ainvoke_null_message_raises_narrow_error(offline_ollama):
    """Legal NDJSON, but one field changed to a bad shape."""
    line = json.dumps(
        {
            "model": REPRESENTATIVE_MODEL,
            "created_at": "2026-10-13T00:00:00Z",
            "message": None,
            "done": True,
            "done_reason": "stop",
        }
    )
    offline_ollama.respond(
        lambda request: offline_ollama.ndjson_response(
            offline_ollama.ndjson_body(ndjson(line))
        )
    )
    model = offline_ollama.model()

    # The SDK builds ``ChatResponse(**part)`` per NDJSON line, so a null message
    # fails the SDK's own pydantic model before the adapter ever sees it. Pinned
    # as observed -- not broadened to ``Exception``, not normalised.
    with pytest.raises(ValidationError) as excinfo:
        await model.ainvoke("ping")

    assert type(excinfo.value) is ValidationError
    assert "message" in str(excinfo.value)
    assert len(offline_ollama.requests) == 1


async def test_astream_bad_json_line_raises_decode_error(offline_ollama):
    body = offline_ollama.body([send(b"{not-json\n")])
    offline_ollama.respond(lambda request: offline_ollama.ndjson_response(body))
    model = offline_ollama.model()

    with pytest.raises(json.JSONDecodeError):
        async for _ in model.astream("ping"):
            pass

    assert len(offline_ollama.requests) == 1
    assert body.close_count >= 1


# ---------------------------------------------------------------------------
# 4. HTTP status for both entries -- no SDK retry, exactly one request
# ---------------------------------------------------------------------------


_HTTP_STATUS_MATRIX = [
    pytest.param(401, id="401"),
    pytest.param(429, id="429"),
    pytest.param(503, id="503"),
]


@pytest.mark.parametrize("status", _HTTP_STATUS_MATRIX)
async def test_ainvoke_http_status_raises_response_error(offline_ollama, status):
    offline_ollama.respond(json_response({"error": "nope"}, status=status))
    model = offline_ollama.model()

    with pytest.raises(ollama.ResponseError) as excinfo:
        await model.ainvoke("ping")

    assert type(excinfo.value) is ollama.ResponseError
    assert excinfo.value.status_code == status, "status_code must match the HTTP status"
    assert len(offline_ollama.requests) == 1, "the Ollama SDK has no retry"


@pytest.mark.parametrize("status", _HTTP_STATUS_MATRIX)
async def test_astream_http_status_raises_response_error(offline_ollama, status):
    offline_ollama.respond(json_response({"error": "nope"}, status=status))
    model = offline_ollama.model()

    with pytest.raises(ollama.ResponseError) as excinfo:
        async for _ in model.astream("ping"):
            pass

    assert type(excinfo.value) is ollama.ResponseError
    assert excinfo.value.status_code == status
    assert len(offline_ollama.requests) == 1


# ---------------------------------------------------------------------------
# 5. Send-phase failures for both entries
# ---------------------------------------------------------------------------


_SEND_FAILURE_MATRIX = [
    pytest.param(httpx.ReadTimeout, "simulated read timeout", id="readtimeout"),
    pytest.param(httpx.ConnectError, "simulated connect failure", id="connecterror"),
]


@pytest.mark.parametrize("exc_cls,message", _SEND_FAILURE_MATRIX)
async def test_ainvoke_send_failure_keeps_original_httpx_error(offline_ollama, exc_cls, message):
    """The SDK's stream=True branch does not translate transport errors."""
    injected = exc_cls(message)
    offline_ollama.respond(raise_this(injected))
    model = offline_ollama.model()

    with pytest.raises(exc_cls, match=message) as excinfo:
        await model.ainvoke("ping")

    assert type(excinfo.value) is exc_cls
    assert excinfo.value is injected, "the captured object must be the injected one"
    assert len(offline_ollama.requests) == 1


@pytest.mark.parametrize("exc_cls,message", _SEND_FAILURE_MATRIX)
async def test_astream_send_failure_keeps_original_httpx_error(offline_ollama, exc_cls, message):
    injected = exc_cls(message)
    offline_ollama.respond(raise_this(injected))
    model = offline_ollama.model()

    with pytest.raises(exc_cls, match=message) as excinfo:
        async for _ in model.astream("ping"):
            pass

    assert type(excinfo.value) is exc_cls
    assert excinfo.value is injected
    assert len(offline_ollama.requests) == 1


# ---------------------------------------------------------------------------
# 6. Body-read failures: 2 exception types x before/after first content
# ---------------------------------------------------------------------------

_BODY_FAILURE_MATRIX = [
    pytest.param(httpx.ReadTimeout, "stalled on timeout", id="readtimeout"),
    pytest.param(httpx.ReadError, "stalled on read", id="readerror"),
]


@pytest.mark.parametrize("exc_cls,message", _BODY_FAILURE_MATRIX)
async def test_astream_body_failure_before_first_content(offline_ollama, exc_cls, message):
    injected = exc_cls(message)
    body = offline_ollama.body([fail_with(injected)])
    offline_ollama.respond(lambda request: offline_ollama.ndjson_response(body))
    model = offline_ollama.model()

    received = []
    with pytest.raises(exc_cls, match=message) as excinfo:
        async for chunk in model.astream("ping"):
            received.append(chunk)

    assert type(excinfo.value) is exc_cls
    assert excinfo.value is injected
    assert len(offline_ollama.requests) == 1
    assert received == []
    # Observed here, before the fixture closes the SDK clients.
    assert body.close_count >= 1


@pytest.mark.parametrize("exc_cls,message", _BODY_FAILURE_MATRIX)
async def test_astream_body_failure_after_first_content(offline_ollama, exc_cls, message):
    injected = exc_cls(message)
    body = offline_ollama.body(
        [
            send((content_line("par", done=False) + "\n").encode()),
            fail_with(injected),
        ]
    )
    offline_ollama.respond(lambda request: offline_ollama.ndjson_response(body))
    model = offline_ollama.model()

    received = []
    with pytest.raises(exc_cls, match=message) as excinfo:
        async for chunk in model.astream("ping"):
            received.append(chunk)

    assert type(excinfo.value) is exc_cls
    assert excinfo.value is injected
    assert len(offline_ollama.requests) == 1
    assert "".join(c.content for c in received if isinstance(c.content, str)) == "par"
    assert body.close_count >= 1


# ---------------------------------------------------------------------------
# 7. HTTP 200 with an error line in the middle of the NDJSON body
# ---------------------------------------------------------------------------


async def test_mid_stream_error_line_raises_response_error(offline_ollama):
    """A 200 body whose later line carries {"error": ...} is an NDJSON error.

    This is not an HTTP 503 and must not be dressed up as one.
    """
    body = offline_ollama.body(
        [
            send((content_line("par", done=False) + "\n").encode()),
            send((json.dumps({"error": "synthetic failure"}) + "\n").encode()),
        ]
    )
    offline_ollama.respond(lambda request: offline_ollama.ndjson_response(body))
    model = offline_ollama.model()

    received = []
    with pytest.raises(ollama.ResponseError) as excinfo:
        async for chunk in model.astream("ping"):
            received.append(chunk)

    assert type(excinfo.value) is ollama.ResponseError
    assert excinfo.value.status_code == -1, "a body error carries no HTTP status"
    assert "synthetic failure" in str(excinfo.value)
    assert len(offline_ollama.requests) == 1
    assert "".join(c.content for c in received if isinstance(c.content, str)) == "par"
    assert body.close_count >= 1


# ---------------------------------------------------------------------------
# 8. Explicit task cancellation, gated on events
# ---------------------------------------------------------------------------


async def test_cancel_active_ainvoke_reading_body(offline_ollama):
    gate = asyncio.Event()
    body = offline_ollama.body([wait_for(gate), send((content_line("late") + "\n").encode())])
    offline_ollama.respond(lambda request: offline_ollama.ndjson_response(body))
    model = offline_ollama.model()

    task = asyncio.create_task(model.ainvoke("ping"))
    try:
        await asyncio.wait_for(body.started.wait(), timeout=5)
        task.cancel()
        await observe_explicit_cancellation(task, timeout=5)

        # Business assertions run while the gate is still unset and before
        # the fixture can close the client, so none of them is cleanup evidence.
        assert len(offline_ollama.requests) == 1
        assert body.close_count >= 1
    finally:
        await reclaim_own_task(task, gate=gate)



async def test_cancel_astream_before_first_content(offline_ollama):
    gate = asyncio.Event()
    body = offline_ollama.body(
        [wait_for(gate), send((content_line("late") + "\n").encode())]
    )
    offline_ollama.respond(lambda request: offline_ollama.ndjson_response(body))
    model = offline_ollama.model()

    received = []

    async def consume():
        async for chunk in model.astream("ping"):
            received.append(chunk)

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(body.started.wait(), timeout=5)
        task.cancel()
        await observe_explicit_cancellation(task, timeout=5)

        # Business assertions run while the gate is still unset and before
        # the fixture can close the client, so none of them is cleanup evidence.
        assert len(offline_ollama.requests) == 1
        assert received == []
        assert body.close_count >= 1
    finally:
        await reclaim_own_task(task, gate=gate)



async def test_cancel_astream_after_first_content_keeps_prefix(offline_ollama):
    gate = asyncio.Event()
    body = offline_ollama.body(
        [
            send((content_line("first", done=False) + "\n").encode()),
            wait_for(gate),
        ]
    )
    offline_ollama.respond(lambda request: offline_ollama.ndjson_response(body))
    model = offline_ollama.model()

    received = []
    first_content_seen = asyncio.Event()

    async def consume():
        async for chunk in model.astream("ping"):
            received.append(chunk)
            if chunk.content:
                first_content_seen.set()

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(first_content_seen.wait(), timeout=5)
        task.cancel()
        await observe_explicit_cancellation(task, timeout=5)

        # Business assertions run while the gate is still unset and before
        # the fixture can close the client, so none of them is cleanup evidence.
        assert len(offline_ollama.requests) == 1
        assert "".join(c.content for c in received if isinstance(c.content, str)) == "first"
        assert body.close_count >= 1
    finally:
        await reclaim_own_task(task, gate=gate)



# ---------------------------------------------------------------------------
# 9. The done_reason='load' special case
# ---------------------------------------------------------------------------


async def test_done_reason_load_with_empty_content_is_skipped(offline_ollama):
    """A model-load frame with no content is skipped, and is not a success.

    The adapter skips the frame, so nothing is left to aggregate.
    """
    load_line = json.dumps(
        {
            "model": REPRESENTATIVE_MODEL,
            "created_at": "2026-10-13T00:00:00Z",
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "load",
        }
    )
    offline_ollama.respond(
        lambda request: offline_ollama.ndjson_response(
            offline_ollama.ndjson_body(ndjson(load_line))
        )
    )
    model = offline_ollama.model()

    with pytest.raises(ValueError, match="No data received from Ollama stream."):
        await model.ainvoke("ping")

    assert len(offline_ollama.requests) == 1


async def test_done_reason_load_with_empty_content_yields_no_text(offline_ollama):
    load_line = json.dumps(
        {
            "model": REPRESENTATIVE_MODEL,
            "created_at": "2026-10-13T00:00:00Z",
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "load",
        }
    )
    offline_ollama.respond(
        lambda request: offline_ollama.ndjson_response(
            offline_ollama.ndjson_body(ndjson(load_line))
        )
    )
    model = offline_ollama.model()

    with pytest.raises(ValueError, match="No generation chunks were returned"):
        async for _ in model.astream("ping"):
            pass

    assert len(offline_ollama.requests) == 1


# ---------------------------------------------------------------------------
# 10. Transport EOF with no done frame (characterisation only)
# ---------------------------------------------------------------------------


async def test_stream_eof_without_done_frame_delivers_text(offline_ollama):
    """Characterisation only: text delivered before a raw EOF still surfaces.

    Says nothing about done/finish/usage completeness; that stays out of scope.
    """
    offline_ollama.respond(
        lambda request: offline_ollama.ndjson_response(
            offline_ollama.ndjson_body(ndjson(content_line("par", done=False)))
        )
    )
    model = offline_ollama.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    # Observed: one content chunk plus a trailing empty chunk; no done frame was
    # required for the text to surface.
    assert len(offline_ollama.requests) == 1
    assert [c.content for c in chunks] == ["par", ""]


# ===========================================================================
# D5A-1 R2: normal completion/usage projection + streamed usage position
#
# Appended at EOF only. Every byte above this marker is the D4 baseline and is
# unchanged. The usage numbers and chunk contents are carried over from the
# first run unchanged; only the sentinel and request assertions are added.
#
#   langchain_core/language_models/chat_models.py:935-950   terminal sentinel
#   langchain_ollama/chat_models.py:102-116    _get_usage_metadata_from_generation_info
#   langchain_ollama/chat_models.py:1152-1176  _achat_stream_with_aggregation
#   langchain_ollama/chat_models.py:1248-1282  done-frame generation_info + usage
#   langchain_ollama/chat_models.py:1381-1403  _agenerate final-message projection
#   langchain_core/messages/ai.py:676-715,721-740           chunk add sums usage
# ===========================================================================


def d5a_add_chunks(chunks: list) -> AIMessageChunk:
    """LangChain's own public chunk addition, applied left to right."""
    total = chunks[0]
    for chunk in chunks[1:]:
        total = total + chunk
    return total


def d5a_assert_core_sentinel(chunk: AIMessageChunk) -> None:
    """Pin the langchain-core terminal sentinel exactly.

    ``chat_models.py:935-950`` appends this chunk itself, after the adapter's
    stream ends, whenever the last adapter chunk carried no ``chunk_position``.
    The Ollama adapter never sets one, so the sentinel is always present here
    and must be asserted rather than filtered out.
    """
    assert chunk.content == ""
    assert chunk.usage_metadata is None
    assert chunk.response_metadata == {}
    assert chunk.chunk_position == "last"


def d5a_counted_done_line(*, prompt: int, output: int, done_reason: str = "stop") -> str:
    """A terminal done frame carrying *both* token counters.

    ``done_line`` above ships only ``eval_count``, and
    ``_get_usage_metadata_from_generation_info`` returns None unless
    ``prompt_eval_count`` and ``eval_count`` are both present, so that shape
    cannot express usage at all. This builder keeps both counters.
    """
    return json.dumps(
        {
            "model": REPRESENTATIVE_MODEL,
            "created_at": "2026-10-13T00:00:00Z",
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": done_reason,
            "total_duration": 1000,
            "prompt_eval_count": prompt,
            "eval_count": output,
        }
    )


def d5a_counted_stream(*, prompt: int, output: int) -> list[bytes]:
    """Two content frames plus a counted terminal done frame."""
    return ndjson(
        content_line("po"),
        content_line("ng"),
        d5a_counted_done_line(prompt=prompt, output=output),
    )


async def test_d5a_baseline_ainvoke(offline_ollama):
    """One normal non-stream answer pins the whole usage projection."""
    offline_ollama.respond(
        lambda request: offline_ollama.ndjson_response(
            offline_ollama.ndjson_body(d5a_counted_stream(prompt=7, output=3))
        )
    )
    model = offline_ollama.model()

    result = await model.ainvoke("ping")

    assert len(offline_ollama.requests) == 1
    assert request_json(offline_ollama.requests[0])["messages"] == [
        {"role": "user", "content": "ping"}
    ]

    assert result.content == "pong"
    # The wire has no total counter; the adapter adds the two it reads, and
    # emits no *_token_details keys at all (so none are present here).
    assert result.usage_metadata == {
        "input_tokens": 7,
        "output_tokens": 3,
        "total_tokens": 10,
    }
    # The done frame is republished as generation_info with the raw wire keys
    # intact, minus "message", plus the adapter's own two labels.
    assert result.response_metadata["done"] is True
    assert result.response_metadata["done_reason"] == "stop"
    assert result.response_metadata["prompt_eval_count"] == 7
    assert result.response_metadata["eval_count"] == 3
    assert result.response_metadata["model_name"] == REPRESENTATIVE_MODEL
    assert result.response_metadata["model_provider"] == "ollama"
    assert "message" not in result.response_metadata


async def test_d5a_baseline_astream(offline_ollama):
    """Streamed usage lands on the terminal done frame."""
    offline_ollama.respond(
        lambda request: offline_ollama.ndjson_response(
            offline_ollama.ndjson_body(d5a_counted_stream(prompt=7, output=3))
        )
    )
    model = offline_ollama.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_ollama.requests) == 1
    body = request_json(offline_ollama.requests[0])
    assert body["messages"] == [{"role": "user", "content": "ping"}]
    assert body["stream"] is True

    # Two content frames, the terminal done frame, and the langchain-core
    # sentinel the adapter never marks itself.
    assert len(chunks) == 4
    assert [c.content for c in chunks] == ["po", "ng", "", ""]
    # Content frames carry no counters, so only the done frame has usage, and
    # it is index 2 rather than the final chunk.
    assert [c.usage_metadata for c in chunks] == [
        None,
        None,
        {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
        None,
    ]
    assert chunks[2].response_metadata["done_reason"] == "stop"
    assert chunks[2].response_metadata["model_provider"] == "ollama"
    assert chunks[2].chunk_position is None
    d5a_assert_core_sentinel(chunks[3])

    aggregated = d5a_add_chunks(chunks)
    assert aggregated.content == "pong"
    assert aggregated.usage_metadata == {
        "input_tokens": 7,
        "output_tokens": 3,
        "total_tokens": 10,
    }
    assert aggregated.response_metadata["done_reason"] == "stop"


async def test_d5a_usage_lifecycle_astream(offline_ollama):
    """Exactly one usage report, and it is the terminal adapter frame.

    ``/api/chat`` reports ``prompt_eval_count`` and ``eval_count`` only on the
    terminal done frame; content frames carry no counters. A single request
    therefore cannot separate cumulative from incremental reporting. The test
    pins that real single-terminal shape rather than inventing mid-stream
    counters the protocol does not send.
    """
    offline_ollama.respond(
        lambda request: offline_ollama.ndjson_response(
            offline_ollama.ndjson_body(d5a_counted_stream(prompt=7, output=3))
        )
    )
    model = offline_ollama.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_ollama.requests) == 1
    body = request_json(offline_ollama.requests[0])
    assert body["messages"] == [{"role": "user", "content": "ping"}]
    assert body["stream"] is True

    assert len(chunks) == 4
    assert [c.content for c in chunks] == ["po", "ng", "", ""]
    assert [c.usage_metadata for c in chunks] == [
        None,
        None,
        {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
        None,
    ]
    # The no-usage range is exactly the two content frames; the terminal
    # adapter frame reports usage and the sentinel reports none.
    assert chunks[0].usage_metadata is None
    assert chunks[1].usage_metadata is None
    assert chunks[2].usage_metadata is not None
    assert chunks[2].chunk_position is None
    d5a_assert_core_sentinel(chunks[3])

    aggregated = d5a_add_chunks(chunks)
    assert aggregated.content == "pong"
    assert aggregated.usage_metadata == {
        "input_tokens": 7,
        "output_tokens": 3,
        "total_tokens": 10,
    }


# ===========================================================================
# D5A-2: completion boundaries and missing usage (7 nodes)
#
# Appended at EOF only. Every byte above this marker is the D4 baseline plus
# the accepted D5A-1 section, and is unchanged. Each fixed expectation was
# derived from the locked sources below, not from an observed run:
#
#   ollama/_types.py:230-246                 done_reason: Optional[str]
#   ollama/_client.py:785-788                ChatResponse(**part)
#   langchain_ollama/chat_models.py:102-116  needs BOTH counters
#   langchain_ollama/chat_models.py:1330-1342 dict(stream_resp) as generation_info
#   langchain_ollama/chat_models.py:1172-1174 raise only when zero frames
#   langchain_core/language_models/chat_models.py:935-950  core sentinel
#   langchain_core/messages/ai.py:676-681    no usage -> None
# ===========================================================================


def d5a2_done_line(*, payload: dict) -> str:
    """One done frame whose shape is supplied field by field.

    ``done_line`` above always ships done_reason and eval_count, and
    ``d5a_counted_done_line`` always ships both counters, so neither can express
    "reason absent", "reason null" or "counters absent" as separate wires.
    """
    frame: dict = {
        "model": REPRESENTATIVE_MODEL,
        "created_at": "2026-10-13T00:00:00Z",
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "total_duration": 1000,
    }
    frame.update(payload)
    return json.dumps(frame)


def d5a2_no_reason_done_line(*, prompt: int, output: int) -> str:
    """A counted done frame whose done_reason key is simply absent."""
    return d5a2_done_line(
        payload={"prompt_eval_count": prompt, "eval_count": output}
    )


def d5a2_null_reason_done_line(*, prompt: int, output: int) -> str:
    """A counted done frame carrying an explicit JSON null done_reason."""
    return d5a2_done_line(
        payload={
            "done_reason": None,
            "prompt_eval_count": prompt,
            "eval_count": output,
        }
    )


def d5a2_uncounted_done_line(*, done_reason: str = "stop") -> str:
    """A done frame with a reason but neither token counter."""
    return d5a2_done_line(payload={"done_reason": done_reason})


def d5a2_stream(done_line_text: str) -> list[bytes]:
    """Two content frames plus the supplied done frame."""
    return ndjson(content_line("po"), content_line("ng"), done_line_text)


async def test_d5a_completion_missing_astream(offline_ollama):
    """No done_reason key on the done frame, counters still present."""
    offline_ollama.respond(
        lambda request: offline_ollama.ndjson_response(
            offline_ollama.ndjson_body(
                d5a2_stream(d5a2_no_reason_done_line(prompt=7, output=3))
            )
        )
    )
    model = offline_ollama.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_ollama.requests) == 1
    body = request_json(offline_ollama.requests[0])
    assert body["messages"] == [{"role": "user", "content": "ping"}]
    assert body["stream"] is True

    assert len(chunks) == 4
    assert [c.content for c in chunks] == ["po", "ng", "", ""]
    assert [c.usage_metadata for c in chunks] == [
        None,
        None,
        {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
        None,
    ]
    # The done frame is republished verbatim, and pydantic materialises every
    # declared field, so done_reason is present as None rather than absent.
    assert chunks[2].response_metadata["done"] is True
    assert "done_reason" in chunks[2].response_metadata
    assert chunks[2].response_metadata["done_reason"] is None
    assert chunks[2].chunk_position is None
    d5a_assert_core_sentinel(chunks[3])

    aggregated = d5a_add_chunks(chunks)
    assert aggregated.content == "pong"
    assert aggregated.usage_metadata == {
        "input_tokens": 7,
        "output_tokens": 3,
        "total_tokens": 10,
    }
    assert aggregated.response_metadata["done_reason"] is None


async def test_d5a_completion_null_astream(offline_ollama):
    """An explicit JSON null done_reason, kept apart from the missing key.

    Both shapes decode to None on a declared field, so the projection is the
    same as test_d5a_completion_missing_astream. The wire really does differ.
    """
    offline_ollama.respond(
        lambda request: offline_ollama.ndjson_response(
            offline_ollama.ndjson_body(
                d5a2_stream(d5a2_null_reason_done_line(prompt=7, output=3))
            )
        )
    )
    model = offline_ollama.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_ollama.requests) == 1
    body = request_json(offline_ollama.requests[0])
    assert body["messages"] == [{"role": "user", "content": "ping"}]
    assert body["stream"] is True

    assert "done_reason" not in json.loads(d5a2_no_reason_done_line(prompt=7, output=3))
    assert json.loads(d5a2_null_reason_done_line(prompt=7, output=3))["done_reason"] is None

    assert len(chunks) == 4
    assert [c.content for c in chunks] == ["po", "ng", "", ""]
    assert [c.usage_metadata for c in chunks] == [
        None,
        None,
        {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
        None,
    ]
    assert "done_reason" in chunks[2].response_metadata
    assert chunks[2].response_metadata["done_reason"] is None
    d5a_assert_core_sentinel(chunks[3])

    aggregated = d5a_add_chunks(chunks)
    assert aggregated.content == "pong"
    assert aggregated.usage_metadata == {
        "input_tokens": 7,
        "output_tokens": 3,
        "total_tokens": 10,
    }
    assert aggregated.response_metadata["done_reason"] is None


async def test_d5a_usage_missing_astream(offline_ollama):
    """Completion signal normal, neither token counter sent.

    The adapter needs both counters to build any usage at all, so a frame with
    neither produces no usage rather than a zero one.
    """
    offline_ollama.respond(
        lambda request: offline_ollama.ndjson_response(
            offline_ollama.ndjson_body(d5a2_stream(d5a2_uncounted_done_line()))
        )
    )
    model = offline_ollama.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_ollama.requests) == 1
    body = request_json(offline_ollama.requests[0])
    assert body["messages"] == [{"role": "user", "content": "ping"}]
    assert body["stream"] is True

    assert len(chunks) == 4
    assert [c.content for c in chunks] == ["po", "ng", "", ""]
    assert [c.usage_metadata for c in chunks] == [None, None, None, None]
    assert chunks[2].response_metadata["done_reason"] == "stop"
    assert chunks[2].response_metadata["prompt_eval_count"] is None
    assert chunks[2].response_metadata["eval_count"] is None
    d5a_assert_core_sentinel(chunks[3])

    aggregated = d5a_add_chunks(chunks)
    assert aggregated.content == "pong"
    assert aggregated.usage_metadata is None
    assert aggregated.response_metadata["done_reason"] == "stop"


async def test_d5a_truncated_ainvoke(offline_ollama):
    """The protocol's own truncation reason, "length"."""
    offline_ollama.respond(
        lambda request: offline_ollama.ndjson_response(
            offline_ollama.ndjson_body(
                d5a2_stream(
                    d5a_counted_done_line(prompt=7, output=3, done_reason="length")
                )
            )
        )
    )
    model = offline_ollama.model()

    result = await model.ainvoke("ping")

    assert len(offline_ollama.requests) == 1
    body = request_json(offline_ollama.requests[0])
    assert body["messages"] == [{"role": "user", "content": "ping"}]

    assert result.content == "pong"
    assert result.usage_metadata == {
        "input_tokens": 7,
        "output_tokens": 3,
        "total_tokens": 10,
    }
    # done_reason is a plain Optional[str] here, never a Literal and never
    # branched on except for the "load" skip case.
    assert result.response_metadata["done_reason"] == "length"
    assert result.response_metadata["done"] is True
    assert result.response_metadata["prompt_eval_count"] == 7
    assert result.response_metadata["eval_count"] == 3
    assert result.response_metadata["model_provider"] == "ollama"
    assert "message" not in result.response_metadata


async def test_d5a_truncated_astream(offline_ollama):
    """The same "length" reason through the streaming entry."""
    offline_ollama.respond(
        lambda request: offline_ollama.ndjson_response(
            offline_ollama.ndjson_body(
                d5a2_stream(
                    d5a_counted_done_line(prompt=7, output=3, done_reason="length")
                )
            )
        )
    )
    model = offline_ollama.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_ollama.requests) == 1
    body = request_json(offline_ollama.requests[0])
    assert body["messages"] == [{"role": "user", "content": "ping"}]
    assert body["stream"] is True

    assert len(chunks) == 4
    assert [c.content for c in chunks] == ["po", "ng", "", ""]
    assert [c.usage_metadata for c in chunks] == [
        None,
        None,
        {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
        None,
    ]
    assert chunks[2].response_metadata["done_reason"] == "length"
    d5a_assert_core_sentinel(chunks[3])

    aggregated = d5a_add_chunks(chunks)
    assert aggregated.content == "pong"
    assert aggregated.usage_metadata == {
        "input_tokens": 7,
        "output_tokens": 3,
        "total_tokens": 10,
    }
    assert aggregated.response_metadata["done_reason"] == "length"


async def test_d5a_unknown_ainvoke(offline_ollama):
    """A synthetic unknown done_reason is accepted, not rejected."""
    offline_ollama.respond(
        lambda request: offline_ollama.ndjson_response(
            offline_ollama.ndjson_body(
                d5a2_stream(
                    d5a_counted_done_line(
                        prompt=7, output=3, done_reason="d5a_unknown"
                    )
                )
            )
        )
    )
    model = offline_ollama.model()

    result = await model.ainvoke("ping")

    assert len(offline_ollama.requests) == 1
    assert request_json(offline_ollama.requests[0])["messages"] == [
        {"role": "user", "content": "ping"}
    ]

    assert result.content == "pong"
    assert result.usage_metadata == {
        "input_tokens": 7,
        "output_tokens": 3,
        "total_tokens": 10,
    }
    # The adapter never normalises the reason, so the raw string survives.
    assert result.response_metadata["done_reason"] == "d5a_unknown"
    assert result.response_metadata["model_provider"] == "ollama"
    assert result.response_metadata["eval_count"] == 3


async def test_d5a_terminal_frame_missing_astream(offline_ollama):
    """A complete content prefix with no done frame at all.

    No data is lost in the sense of raising: the text still surfaces. What is
    missing is every completion and usage fact the done frame would have
    carried, so the aggregate metadata is empty and the aggregate usage is None.
    """
    offline_ollama.respond(
        lambda request: offline_ollama.ndjson_response(
            offline_ollama.ndjson_body(ndjson(content_line("po"), content_line("ng")))
        )
    )
    model = offline_ollama.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_ollama.requests) == 1
    body = request_json(offline_ollama.requests[0])
    assert body["messages"] == [{"role": "user", "content": "ping"}]
    assert body["stream"] is True

    assert len(chunks) == 3
    assert [c.content for c in chunks] == ["po", "ng", ""]
    assert [c.usage_metadata for c in chunks] == [None, None, None]
    # Content frames carry no generation_info, so nothing lands in metadata.
    assert chunks[0].response_metadata == {}
    assert chunks[1].response_metadata == {}
    d5a_assert_core_sentinel(chunks[2])

    aggregated = d5a_add_chunks(chunks)
    assert aggregated.content == "pong"
    assert aggregated.usage_metadata is None
    assert aggregated.response_metadata == {}
