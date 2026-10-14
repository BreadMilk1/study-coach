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
    await asyncio.wait_for(body.started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(offline_ollama.requests) == 1
    assert body.close_count >= 1


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
    await asyncio.wait_for(body.started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(offline_ollama.requests) == 1
    assert received == []
    assert body.close_count >= 1


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
    await asyncio.wait_for(first_content_seen.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(offline_ollama.requests) == 1
    assert "".join(c.content for c in received if isinstance(c.content, str)) == "first"
    assert body.close_count >= 1


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
