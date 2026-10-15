"""D4a-2: offline evidence for the real Anthropic SDK's async transport contract.

Scope is one provider only. The chain that runs for real is
``get_chat_model`` -> ``ChatAnthropic`` -> ``anthropic.AsyncClient`` -> the real
``messages`` resource and SSE parser -> the adapter's own event conversion.
Only the two HTTPX client-construction entries the adapter itself calls are
patched, plus the SDK's retry backoff (set to 0 so exhaustion does not sleep).

Everything Anthropic-shaped here is Anthropic wire: ``message_start`` /
``content_block_*`` / ``message_delta`` / ``message_stop``. OpenAI's
``choices``/``[DONE]`` framing is deliberately not used.

Failure surfaces stay separate, exactly as in D4a-1:

* **send retries** -- the SDK re-issues; visible as request count.
* **body not replayed** -- a mid-body failure cannot be retried.
* **task cancellation** -- ``asyncio.CancelledError`` propagates untouched.

Nothing here is normalised into a product-level "unavailable" result, and no
test swallows a surprising error to make itself pass.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from typing import Any, Callable

import anthropic
import httpx
import pytest
from langchain_anthropic import chat_models as anthropic_chat_models
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
    sse_event,
    wait_for,
)

SYNTHETIC_ANTHROPIC_KEY = "sk-ant-offline-d4a2"
REPRESENTATIVE_MODEL = "claude-haiku-4-5"
SDK_DEFAULT_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"

# The locked factory takes keyword-only arguments and distinguishes "timeout was
# not supplied at all" from "timeout=None". This sentinel preserves that.
_NOT_GIVEN: Any = object()

ANTHROPIC_STEERING_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_PROXY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "OPENAI_PROXY",
    "OPENAI_ORG_ID",
    "OPENAI_ORGANIZATION",
)


# ---------------------------------------------------------------------------
# Anthropic wire builders
# ---------------------------------------------------------------------------


def message_payload(text: str | None = "pong") -> dict:
    content: list = [] if text is None else [{"type": "text", "text": text}]
    return {
        "id": "msg_offline",
        "type": "message",
        "role": "assistant",
        "model": REPRESENTATIVE_MODEL,
        "content": content,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 3, "output_tokens": 1},
    }


def event_message_start() -> bytes:
    """A protocol-legal opening frame.

    The Anthropic streaming contract requires the opening ``message`` to carry
    an empty ``content`` list; body text only ever arrives later as
    ``content_block_delta``. Seeding the opening message with body text here
    would make every "legal SSE" fixture below quietly wrong.
    """
    body = message_payload()
    body["content"] = []
    body["stop_reason"] = None
    body["usage"] = {"input_tokens": 3, "output_tokens": 0}
    return sse_event(json.dumps({"type": "message_start", "message": body}), event="message_start")


def event_content_block_start() -> bytes:
    return sse_event(
        json.dumps(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }
        ),
        event="content_block_start",
    )


def event_content_block_delta(text: Any = "po") -> bytes:
    return sse_event(
        json.dumps(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            }
        ),
        event="content_block_delta",
    )


def event_content_block_stop() -> bytes:
    return sse_event(
        json.dumps({"type": "content_block_stop", "index": 0}),
        event="content_block_stop",
    )


def event_message_delta() -> bytes:
    return sse_event(
        json.dumps(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 2},
            }
        ),
        event="message_delta",
    )


def event_message_stop() -> bytes:
    return sse_event(json.dumps({"type": "message_stop"}), event="message_stop")


def full_text_stream(*texts: str) -> list[bytes]:
    """A complete, legal Anthropic SSE frame sequence carrying ``texts``."""
    events = [event_message_start(), event_content_block_start()]
    events.extend(event_content_block_delta(t) for t in texts)
    events.extend([event_content_block_stop(), event_message_delta(), event_message_stop()])
    return events


def json_response(payload: dict, *, status: int = 200, headers: dict | None = None):
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload, headers=headers or {})

    return responder


def raise_in_transport(exc_factory):
    """A responder that fails during send, before any response exists."""

    def responder(request: httpx.Request) -> httpx.Response:
        raise exc_factory()

    return responder


def request_json(request: httpx.Request) -> dict:
    return json.loads(request.content)


# ---------------------------------------------------------------------------
# Harness (Anthropic-local; the OpenAI helper is not reused as a fixture)
# ---------------------------------------------------------------------------


class OfflineAnthropic:
    """MockTransport-backed stand-in for the Anthropic socket."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.streams: list[ScriptedByteStream] = []
        self.async_clients: list[httpx.AsyncClient] = []
        self.sync_clients: list[httpx.Client] = []
        self._responder: Callable[[httpx.Request], httpx.Response] | None = None

    def respond(self, responder) -> "OfflineAnthropic":
        self._responder = responder
        return self

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self._responder is None:
            raise AssertionError("offline transport has no responder installed")
        return self._responder(request)

    def _base_kwargs(self, base_url, timeout, anthropic_proxy):
        # Mirrors the locked factory's own "explicit None vs not supplied" rule.
        kwargs: dict[str, Any] = {
            "base_url": base_url
            or SDK_DEFAULT_BASE_URL,
        }
        if timeout is not _NOT_GIVEN:
            kwargs["timeout"] = timeout
        if anthropic_proxy is not None:
            kwargs["proxy"] = anthropic_proxy
        return kwargs

    def async_client_factory(self, *, base_url, timeout=_NOT_GIVEN, anthropic_proxy=None):
        client = anthropic.DefaultAsyncHttpxClient(
            transport=httpx.MockTransport(self.handle),
            trust_env=False,
            **self._base_kwargs(base_url, timeout, anthropic_proxy),
        )
        self.async_clients.append(client)
        return client

    def sync_client_factory(self, *, base_url, timeout=_NOT_GIVEN, anthropic_proxy=None):
        client = anthropic.DefaultHttpxClient(
            transport=httpx.MockTransport(self.handle),
            trust_env=False,
            **self._base_kwargs(base_url, timeout, anthropic_proxy),
        )
        self.sync_clients.append(client)
        return client

    def body(self, steps, *, name: str = "body") -> ScriptedByteStream:
        stream = ScriptedByteStream(steps, name=name)
        self.streams.append(stream)
        return stream

    def sse_response(
        self,
        stream: ScriptedByteStream,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        return httpx.Response(
            status,
            headers={"content-type": "text/event-stream", **(headers or {})},
            stream=stream,
        )

    def sse_body(
        self,
        events,
        *,
        terminator: bytes | None = None,
        stall: bool = False,
        name: str = "sse",
    ) -> ScriptedByteStream:
        """Anthropic SSE body. There is no ``[DONE]`` marker in this protocol,
        so ``terminator`` is off by default; ``stall=True`` parks the body."""
        steps: list[tuple[str, Any]] = [send(e) for e in events]
        if terminator is not None:
            steps.append(send(terminator))
        if stall:
            steps.append(wait_for(asyncio.Event()))
        return self.body(steps, name=name)

    def model(self, **kwargs):
        from app.llm.provider import LLMConfig, get_chat_model

        config = LLMConfig(
            provider="anthropic",
            model=REPRESENTATIVE_MODEL,
            api_key=SYNTHETIC_ANTHROPIC_KEY,
        )
        return get_chat_model(config, **kwargs)

    async def aclose(self) -> None:
        for client in self.async_clients:
            await client.aclose()
        self.async_clients.clear()
        for client in self.sync_clients:
            client.close()
        self.sync_clients.clear()


@pytest.fixture(autouse=True)
def anthropic_env(monkeypatch):
    """Anthropic's own steering variables, cleaned and restored per test.

    The shared OpenAI fixture only clears OpenAI variables, so it cannot stand
    in for provider-wide isolation.
    """
    for name in ANTHROPIC_STEERING_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("STUDY_COACH_TEST_MODE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    yield


@pytest.fixture
async def offline_anthropic(monkeypatch) -> AsyncIterator[OfflineAnthropic]:
    harness = OfflineAnthropic()

    # Only the two HTTPX construction entries are patched. The adapter's
    # _client/_async_client properties, the messages resource, the SDK SSE
    # parser and the adapter event conversion all stay real.
    monkeypatch.setattr(
        anthropic_chat_models, "_get_default_async_httpx_client", harness.async_client_factory
    )
    monkeypatch.setattr(
        anthropic_chat_models, "_get_default_httpx_client", harness.sync_client_factory
    )

    # Remove only the real backoff wait; max_retries and should_retry are untouched.
    monkeypatch.setattr(
        anthropic._base_client.BaseClient,
        "_calculate_retry_timeout",
        lambda *args, **kwargs: 0.0,
    )

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
        socket.getaddrinfo("api.anthropic.com", 443)
    with pytest.raises(AssertionError, match="real network access"):
        socket.create_connection(("api.anthropic.com", 443))


def test_offline_harness_uses_sdk_default_httpx_clients(offline_anthropic):
    """The client keeps the SDK's own defaults, not bare-httpx defaults."""
    model = offline_anthropic.model()
    # Touch the cached properties explicitly: creating them is the point here.
    async_client = model._async_client._client

    assert type(async_client) is anthropic.DefaultAsyncHttpxClient
    assert async_client.follow_redirects is True, (
        "a bare httpx.AsyncClient would have shipped follow_redirects=False"
    )


def test_offline_harness_pins_sdk_default_config(offline_anthropic):
    """The adapter built its own clients the default way, on a mock socket."""
    model = offline_anthropic.model()

    assert type(model).__name__ == "ChatAnthropic"
    assert model.model == REPRESENTATIVE_MODEL  # ``model_name`` is only its alias
    # The adapter itself fills in the SDK default; nothing overrode it.
    assert model.anthropic_api_url == SDK_DEFAULT_BASE_URL
    assert model.max_retries == 2
    assert model.default_request_timeout is None
    assert model.stream_usage is True

    async_client = model._async_client._client
    assert isinstance(async_client._transport, httpx.MockTransport)
    assert async_client.trust_env is False
    # Unlike OpenAI's /v1 base, this one is a bare origin and httpx keeps it verbatim.
    assert str(async_client.base_url) == SDK_DEFAULT_BASE_URL


def test_offline_harness_timeouts_are_all_none(offline_anthropic):
    """Model, SDK and HTTPX report "no timeout"; HTTPX field by field.

    Worth pinning: the SDK ships ``DEFAULT_TIMEOUT`` with a 5s connect budget,
    but the adapter passes ``timeout=None`` explicitly, so the factory's
    "not supplied" branch never fires and that default is not in effect.
    """
    model = offline_anthropic.model()

    assert model.default_request_timeout is None
    assert model._async_client.timeout is None

    for httpx_client in offline_anthropic.async_clients + offline_anthropic.sync_clients:
        assert httpx_client.timeout.connect is None
        assert httpx_client.timeout.read is None
        assert httpx_client.timeout.write is None
        assert httpx_client.timeout.pool is None


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
        sys.modules[__name__], "tests/llm/test_async_anthropic_transport_contract.py"
    )


# ---------------------------------------------------------------------------
# 1. Happy path for both entries
# ---------------------------------------------------------------------------


async def test_ainvoke_sends_real_messages_request(offline_anthropic):
    offline_anthropic.respond(json_response(message_payload("pong")))
    model = offline_anthropic.model()

    result = await model.ainvoke("ping")

    assert len(offline_anthropic.requests) == 1
    request = offline_anthropic.requests[0]
    assert request.method == "POST"
    assert str(request.url) == f"{SDK_DEFAULT_BASE_URL}/v1/messages"
    assert request.url.host == "api.anthropic.com"
    assert request.headers["x-api-key"] == SYNTHETIC_ANTHROPIC_KEY
    assert request.headers["anthropic-version"] == ANTHROPIC_VERSION

    body = request_json(request)
    assert body["model"] == REPRESENTATIVE_MODEL
    assert body["messages"] == [{"role": "user", "content": "ping"}]
    assert body["max_tokens"] > 0
    assert body.get("stream") is not True

    assert isinstance(result, AIMessage)
    assert result.content == "pong"


async def test_astream_reads_real_anthropic_sse(offline_anthropic):
    offline_anthropic.respond(
        lambda request: offline_anthropic.sse_response(
            offline_anthropic.sse_body(full_text_stream("po", "ng"))
        )
    )
    model = offline_anthropic.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_anthropic.requests) == 1
    request = offline_anthropic.requests[0]
    # MockTransport would happily accept any path, so the real request shape is
    # pinned here too, not only in the ainvoke case.
    assert request.method == "POST"
    assert str(request.url) == f"{SDK_DEFAULT_BASE_URL}/v1/messages"
    assert request.url.host == "api.anthropic.com"
    assert request.headers["x-api-key"] == SYNTHETIC_ANTHROPIC_KEY
    assert request.headers["anthropic-version"] == ANTHROPIC_VERSION

    body = request_json(request)
    assert body["model"] == REPRESENTATIVE_MODEL
    assert body["messages"] == [{"role": "user", "content": "ping"}]
    assert body["max_tokens"] > 0
    assert body["stream"] is True

    assert all(isinstance(c, AIMessageChunk) for c in chunks)
    assert "".join(c.content for c in chunks if isinstance(c.content, str)) == "pong"


# ---------------------------------------------------------------------------
# 2. Empty / missing data
# ---------------------------------------------------------------------------


async def test_ainvoke_empty_text_block_is_a_real_empty_message(offline_anthropic):
    offline_anthropic.respond(json_response(message_payload("")))
    model = offline_anthropic.model()

    result = await model.ainvoke("ping")

    assert len(offline_anthropic.requests) == 1
    assert isinstance(result, AIMessage)
    assert result.content == ""


async def test_ainvoke_no_content_blocks_returns_empty_list_content(offline_anthropic):
    """``content: []`` is a legal message shape, and it is NOT the same as an
    empty text block: the adapter returns a list, where ``text: ""`` returns a
    string. Observed and pinned rather than normalised."""
    offline_anthropic.respond(json_response(message_payload(None)))
    model = offline_anthropic.model()

    result = await model.ainvoke("ping")

    assert len(offline_anthropic.requests) == 1
    assert isinstance(result, AIMessage)
    assert result.content == []


async def test_astream_empty_text_delta_yields_empty_chunk(offline_anthropic):
    """A legal ``text_delta`` carrying "" is a chunk, not an event."""
    offline_anthropic.respond(
        lambda request: offline_anthropic.sse_response(
            offline_anthropic.sse_body(full_text_stream(""))
        )
    )
    model = offline_anthropic.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(chunks) >= 1
    assert "".join(c.content for c in chunks if isinstance(c.content, str)) == ""


def metadata_only_complete_stream() -> list[bytes]:
    """A protocol-complete stream that carries metadata but no body text.

    The content block is opened and then properly closed, and the stream still
    carries ``message_delta`` before ``message_stop`` -- so this is a *normal*
    stream, not a truncated one.
    """
    return [
        event_message_start(),
        event_content_block_start(),
        event_content_block_stop(),
        event_message_delta(),
        event_message_stop(),
    ]


async def test_astream_with_metadata_frames_but_no_content(offline_anthropic):
    """A complete, legal stream with no text delta delivers no body text.

    ``message_start`` and ``message_delta`` each yield an empty-content chunk;
    those are metadata, not body text, so the delivered body text stays empty.
    """
    events = metadata_only_complete_stream()

    offline_anthropic.respond(
        lambda request: offline_anthropic.sse_response(
            offline_anthropic.sse_body(events)
        )
    )
    model = offline_anthropic.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_anthropic.requests) == 1
    assert len(chunks) >= 1
    assert "".join(c.content for c in chunks if isinstance(c.content, str)) == "", (
        "no text delta was sent, so nothing may be counted as body text"
    )


async def test_astream_without_any_event_raises_instead_of_yielding(offline_anthropic):
    """A stream that emits no event at all is a hard error."""
    offline_anthropic.respond(
        lambda request: offline_anthropic.sse_response(offline_anthropic.sse_body([]))
    )
    model = offline_anthropic.model()

    with pytest.raises(ValueError, match="No generation chunks were returned"):
        async for _ in model.astream("ping"):
            pass

    assert len(offline_anthropic.requests) == 1


# ---------------------------------------------------------------------------
# 3. Bad wire
# ---------------------------------------------------------------------------


async def test_ainvoke_null_content_raises_narrow_adapter_error(offline_anthropic):
    """``content: null`` reaches the adapter's formatter as None.

    The narrow failure is fixed as observed; deliberately not broadened to
    ``Exception`` and not normalised into an "unavailable" result.
    """
    payload = message_payload("pong")
    payload["content"] = None
    offline_anthropic.respond(json_response(payload))
    model = offline_anthropic.model()

    with pytest.raises(TypeError):
        await model.ainvoke("ping")

    assert len(offline_anthropic.requests) == 1


async def test_astream_invalid_json_raises_decode_error(offline_anthropic):
    """The SDK's SSE decoder does not parse JSON itself -- each event branch
    does. So the frame must carry a real event name, otherwise the malformed
    payload is simply never dispatched and nothing is parsed."""
    offline_anthropic.respond(
        lambda request: offline_anthropic.sse_response(
            offline_anthropic.sse_body(
                [sse_event("{not-json", event="content_block_delta")]
            )
        )
    )
    model = offline_anthropic.model()

    with pytest.raises(json.JSONDecodeError):
        async for _ in model.astream("ping"):
            pass

    assert len(offline_anthropic.requests) == 1


async def test_astream_malformed_frame_without_event_name_is_never_parsed(offline_anthropic):
    """Counterpart to the case above: with no event name the same bytes are
    dropped silently and the stream simply produces no chunks."""
    offline_anthropic.respond(
        lambda request: offline_anthropic.sse_response(
            offline_anthropic.sse_body([b"data: {not-json\n\n"])
        )
    )
    model = offline_anthropic.model()

    with pytest.raises(ValueError, match="No generation chunks were returned"):
        async for _ in model.astream("ping"):
            pass

    assert len(offline_anthropic.requests) == 1


async def test_astream_bad_text_delta_shape_raises_narrow_validation_error(offline_anthropic):
    """Legal stream first, then exactly one field changed to a bad shape.

    The SDK builds its event model leniently, so an integer ``text`` reaches
    the adapter intact; the narrow failure is then the pydantic validation of
    ``AIMessageChunk.content``. Pinned as observed -- not broadened to
    ``Exception`` and not normalised into an "unavailable" result.
    """
    offline_anthropic.respond(
        lambda request: offline_anthropic.sse_response(
            offline_anthropic.sse_body(full_text_stream("po"))
        )
    )
    model = offline_anthropic.model()
    good = [chunk async for chunk in model.astream("ping")]
    assert "".join(c.content for c in good if isinstance(c.content, str)) == "po"

    bad_events = [
        event_message_start(),
        event_content_block_start(),
        event_content_block_delta(123),
    ]
    offline_anthropic.respond(
        lambda request: offline_anthropic.sse_response(
            offline_anthropic.sse_body(bad_events)
        )
    )

    with pytest.raises(ValidationError) as excinfo:
        async for _ in model.astream("ping"):
            pass

    assert type(excinfo.value) is ValidationError
    assert "content" in str(excinfo.value)
    assert len(offline_anthropic.requests) == 2, "the bad wire was not replayed"


# ---------------------------------------------------------------------------
# 4. HTTP status for both entries
# ---------------------------------------------------------------------------


_HTTP_STATUS_MATRIX = [
    pytest.param(401, anthropic.AuthenticationError, 1, id="401-auth"),
    pytest.param(429, anthropic.RateLimitError, 3, id="429-ratelimit"),
    pytest.param(503, anthropic.InternalServerError, 3, id="503-server"),
]


def error_body(message: str = "nope") -> dict:
    return {"type": "error", "error": {"type": "api_error", "message": message}}


@pytest.mark.parametrize("status,expected,attempts", _HTTP_STATUS_MATRIX)
async def test_ainvoke_http_status_maps_to_real_sdk_error(
    offline_anthropic, status, expected, attempts
):
    """401 is not retryable; 429 and 5xx exhaust max_retries=2 (3 attempts)."""
    offline_anthropic.respond(json_response(error_body(), status=status))
    model = offline_anthropic.model()

    with pytest.raises(expected) as excinfo:
        await model.ainvoke("ping")

    assert type(excinfo.value) is expected, "exact class, not merely a subclass"
    assert len(offline_anthropic.requests) == attempts


@pytest.mark.parametrize("status,expected,attempts", _HTTP_STATUS_MATRIX)
async def test_astream_http_status_maps_to_real_sdk_error(
    offline_anthropic, status, expected, attempts
):
    """Streaming shares the send-time status policy, including retry counts."""
    offline_anthropic.respond(json_response(error_body(), status=status))
    model = offline_anthropic.model()

    with pytest.raises(expected) as excinfo:
        async for _ in model.astream("ping"):
            pass

    assert type(excinfo.value) is expected, "exact class, not merely a subclass"
    assert len(offline_anthropic.requests) == attempts


# ---------------------------------------------------------------------------
# 5. Send-phase failures for both entries
# ---------------------------------------------------------------------------


_SEND_FAILURE_MATRIX = [
    pytest.param(httpx.ReadTimeout, "simulated read timeout", anthropic.APITimeoutError, id="readtimeout"),
    pytest.param(httpx.ConnectError, "simulated connect failure", anthropic.APIConnectionError, id="connecterror"),
]


@pytest.mark.parametrize("exc_cls,message,expected", _SEND_FAILURE_MATRIX)
async def test_ainvoke_send_failure_exhausts_into_sdk_wrapper(
    offline_anthropic, exc_cls, message, expected
):
    offline_anthropic.respond(raise_in_transport(lambda: exc_cls(message)))
    model = offline_anthropic.model()

    with pytest.raises(expected) as excinfo:
        await model.ainvoke("ping")

    assert type(excinfo.value) is expected, "exact class, not merely a subclass"
    assert len(offline_anthropic.requests) == 3
    assert model.max_retries == 2, "the retry budget must stay at the SDK default"


@pytest.mark.parametrize("exc_cls,message,expected", _SEND_FAILURE_MATRIX)
async def test_astream_send_failure_exhausts_into_sdk_wrapper(
    offline_anthropic, exc_cls, message, expected
):
    offline_anthropic.respond(raise_in_transport(lambda: exc_cls(message)))
    model = offline_anthropic.model()

    with pytest.raises(expected) as excinfo:
        async for _ in model.astream("ping"):
            pass

    assert type(excinfo.value) is expected, "exact class, not merely a subclass"
    assert len(offline_anthropic.requests) == 3


# ---------------------------------------------------------------------------
# 6. Retry and SDK override (representative entries only)
# ---------------------------------------------------------------------------


async def test_retryable_then_success(offline_anthropic):
    responses = [
        lambda request: httpx.Response(429, json=error_body("slow down")),
        lambda request: offline_anthropic.sse_response(
            offline_anthropic.sse_body(full_text_stream("recovered"))
        ),
    ]
    calls = {"n": 0}

    def responder(request):
        index = calls["n"]
        calls["n"] += 1
        return responses[min(index, len(responses) - 1)](request)

    offline_anthropic.respond(responder)
    model = offline_anthropic.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_anthropic.requests) == 2
    assert "".join(c.content for c in chunks if isinstance(c.content, str)) == "recovered"


async def test_should_retry_true_on_401_retries_and_succeeds(offline_anthropic):
    responses = [
        lambda request: httpx.Response(
            401, json=error_body("nope"), headers={"x-should-retry": "true"}
        ),
        lambda request: offline_anthropic.sse_response(
            offline_anthropic.sse_body(full_text_stream("ok"))
        ),
    ]
    calls = {"n": 0}

    def responder(request):
        index = calls["n"]
        calls["n"] += 1
        return responses[min(index, len(responses) - 1)](request)

    offline_anthropic.respond(responder)
    model = offline_anthropic.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_anthropic.requests) == 2
    assert "".join(c.content for c in chunks if isinstance(c.content, str)) == "ok"


async def test_should_retry_false_on_503_fails_immediately(offline_anthropic):
    offline_anthropic.respond(
        json_response(
            error_body("hard stop"), status=503, headers={"x-should-retry": "false"}
        )
    )
    model = offline_anthropic.model()

    with pytest.raises(anthropic.InternalServerError):
        await model.ainvoke("ping")

    assert len(offline_anthropic.requests) == 1, (
        "x-should-retry:false must suppress the SDK's default 5xx retry"
    )


# ---------------------------------------------------------------------------
# 7. Body-read failures: 2 exception types x before/after first content
# ---------------------------------------------------------------------------

_BODY_FAILURE_MATRIX = [
    pytest.param(httpx.ReadTimeout, "stalled on timeout", id="readtimeout"),
    pytest.param(httpx.ReadError, "stalled on read", id="readerror"),
]


@pytest.mark.parametrize("exc_cls,message", _BODY_FAILURE_MATRIX)
async def test_astream_body_failure_before_first_content(offline_anthropic, exc_cls, message):
    injected = exc_cls(message)
    body = offline_anthropic.body(
        [send(event_message_start()), fail_with(injected)]
    )
    offline_anthropic.respond(lambda request: offline_anthropic.sse_response(body))
    model = offline_anthropic.model()

    received = []
    with pytest.raises(exc_cls, match=message) as excinfo:
        async for chunk in model.astream("ping"):
            received.append(chunk)

    # Exact class and exact identity: this is the very object the body raised,
    # not merely something a `raises(exc_cls)` would accept.
    assert type(excinfo.value) is exc_cls
    assert excinfo.value is injected

    assert len(offline_anthropic.requests) == 1, "a body failure cannot be replayed"
    assert "".join(c.content for c in received if isinstance(c.content, str)) == "", (
        "metadata frames are not body text"
    )
    # Observed here, before the fixture actively closes the client.
    assert body.close_count >= 1


@pytest.mark.parametrize("exc_cls,message", _BODY_FAILURE_MATRIX)
async def test_astream_body_failure_after_first_content(offline_anthropic, exc_cls, message):
    injected = exc_cls(message)
    body = offline_anthropic.body(
        [
            send(event_message_start()),
            send(event_content_block_start()),
            send(event_content_block_delta("par")),
            fail_with(injected),
        ]
    )
    offline_anthropic.respond(lambda request: offline_anthropic.sse_response(body))
    model = offline_anthropic.model()

    received = []
    with pytest.raises(exc_cls, match=message) as excinfo:
        async for chunk in model.astream("ping"):
            received.append(chunk)

    assert type(excinfo.value) is exc_cls
    assert excinfo.value is injected

    assert len(offline_anthropic.requests) == 1
    assert "".join(c.content for c in received if isinstance(c.content, str)) == "par"
    assert body.close_count >= 1


# ---------------------------------------------------------------------------
# 8. HTTP 200 SSE error event
# ---------------------------------------------------------------------------


async def test_http_200_sse_error_event_records_real_class(offline_anthropic):
    """A 200 carrying ``event: error`` is mapped by status, not by error name.

    The payload deliberately uses a ``rate_limit_error`` name; the SDK must not
    turn it into ``RateLimitError``, and must not retry on it.
    """
    error_event = sse_event(
        json.dumps(
            {
                "type": "error",
                "error": {"type": "rate_limit_error", "message": "upstream blew up"},
            }
        ),
        event="error",
    )
    offline_anthropic.respond(
        lambda request: offline_anthropic.sse_response(
            offline_anthropic.sse_body([error_event])
        )
    )
    model = offline_anthropic.model()

    with pytest.raises(anthropic.APIStatusError) as excinfo:
        async for _ in model.astream("ping"):
            pass

    assert type(excinfo.value) is anthropic.APIStatusError, (
        "status 200 falls through to the generic APIStatusError"
    )
    assert not isinstance(excinfo.value, anthropic.RateLimitError)
    assert excinfo.value.status_code == 200
    assert len(offline_anthropic.requests) == 1, "an SSE error event must not be retried"


# ---------------------------------------------------------------------------
# 9. Explicit task cancellation, gated on events
# ---------------------------------------------------------------------------


async def test_cancel_active_ainvoke_reading_body(offline_anthropic):
    gate = asyncio.Event()
    body = offline_anthropic.body([wait_for(gate), send(b"never-reached")])
    offline_anthropic.respond(lambda request: offline_anthropic.sse_response(body))
    model = offline_anthropic.model()

    task = asyncio.create_task(model.ainvoke("ping"))
    try:
        await asyncio.wait_for(body.started.wait(), timeout=5)
        task.cancel()
        await observe_explicit_cancellation(task, timeout=5)

        # Business assertions run while the gate is still unset and before
        # the fixture can close the client, so none of them is cleanup evidence.
        assert len(offline_anthropic.requests) == 1
        assert body.close_count >= 1, "response created before cancellation must close"
    finally:
        await reclaim_own_task(task, gate=gate)



async def test_cancel_astream_before_first_content(offline_anthropic):
    """Metadata frames are legal before the gate; no content text is delivered."""
    gate = asyncio.Event()
    body = offline_anthropic.body(
        [
            send(event_message_start()),
            send(event_content_block_start()),
            wait_for(gate),
            send(event_content_block_delta("late")),
        ]
    )
    offline_anthropic.respond(lambda request: offline_anthropic.sse_response(body))
    model = offline_anthropic.model()

    received = []
    first_chunk_seen = asyncio.Event()

    async def consume():
        async for chunk in model.astream("ping"):
            received.append(chunk)
            first_chunk_seen.set()

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(first_chunk_seen.wait(), timeout=5)
        task.cancel()
        await observe_explicit_cancellation(task, timeout=5)

        # Business assertions run while the gate is still unset and before
        # the fixture can close the client, so none of them is cleanup evidence.
        assert len(offline_anthropic.requests) == 1
        assert "".join(c.content for c in received if isinstance(c.content, str)) == "", (
            "only metadata frames were delivered before cancellation"
        )
        assert body.close_count >= 1
    finally:
        await reclaim_own_task(task, gate=gate)



async def test_cancel_astream_after_first_content_keeps_prefix(offline_anthropic):
    gate = asyncio.Event()
    body = offline_anthropic.body(
        [
            send(event_message_start()),
            send(event_content_block_start()),
            send(event_content_block_delta("first")),
            wait_for(gate),
        ]
    )
    offline_anthropic.respond(lambda request: offline_anthropic.sse_response(body))
    model = offline_anthropic.model()

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
        assert len(offline_anthropic.requests) == 1
        assert "".join(c.content for c in received if isinstance(c.content, str)) == "first"
        assert body.close_count >= 1
    finally:
        await reclaim_own_task(task, gate=gate)



# ---------------------------------------------------------------------------
# 10. Transport EOF without message_stop (characterisation only)
# ---------------------------------------------------------------------------


def sse_event_names(frames: list[bytes]) -> list[str]:
    """Parse the ``event:`` names out of raw SSE frames.

    Used as an input precondition so the EOF case proves what it claims from
    the bytes themselves, instead of trusting a helper name or a test id.
    """
    names: list[str] = []
    for frame in frames:
        for line in frame.decode().splitlines():
            if line.startswith("event: "):
                names.append(line[len("event: ") :])
    return names


@pytest.mark.parametrize(
    "events,label",
    [
        pytest.param(
            [
                event_message_start(),
                event_content_block_start(),
                event_content_block_delta("par"),
            ],
            "eof-before-content-block-stop",
            id="eof-before-content-block-stop",
        ),
        # Truncated at message_delta: full_text_stream appends message_stop, so
        # this slice deliberately drops the final frame.
        pytest.param(
            full_text_stream("par")[:-1],
            "eof-after-message-delta-no-stop",
            id="eof-after-message-delta-no-stop",
        ),
    ],
)
async def test_stream_transport_eof_without_message_stop(offline_anthropic, events, label):
    """Characterisation only: text already delivered survives a raw EOF.

    Says nothing about provider-side completion or usage integrity; those stay
    out of scope for D4a-2.
    """
    # Input precondition, derived from the bytes rather than from the label.
    names = sse_event_names(events)
    assert "message_stop" not in names, f"{label} must not contain message_stop: {names}"

    offline_anthropic.respond(
        lambda request: offline_anthropic.sse_response(offline_anthropic.sse_body(events))
    )
    model = offline_anthropic.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_anthropic.requests) == 1
    assert "".join(c.content for c in chunks if isinstance(c.content, str)) == "par"
