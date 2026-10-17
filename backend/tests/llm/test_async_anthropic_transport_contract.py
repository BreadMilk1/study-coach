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


# ===========================================================================
# D5A-1 R2: normal completion/usage projection + streamed usage position
#
# Appended at EOF only. Every byte above this marker is the D4 baseline and is
# unchanged. Each fixed expectation was derived from the locked sources below,
# not from an observed run. The usage numbers and chunk contents are carried
# over from the first run unchanged; only the sentinel and request assertions
# are added.
#
#   langchain_core/language_models/chat_models.py:935-950   terminal sentinel
#   langchain_anthropic/chat_models.py:1483-1494            message_start
#   langchain_anthropic/chat_models.py:1540-1548            text_delta
#   langchain_anthropic/chat_models.py:1602-1631            message_delta + last
#   langchain_anthropic/chat_models.py:1634-1689            _format_output
#   langchain_anthropic/chat_models.py:2255-2304            _create_usage_metadata
#   langchain_core/messages/ai.py:676-715,721-740           chunk add sums usage
# ===========================================================================


def d5a_add_chunks(chunks: list) -> AIMessageChunk:
    """LangChain's own public chunk addition, applied left to right."""
    total = chunks[0]
    for chunk in chunks[1:]:
        total = total + chunk
    return total


def d5a_stream_with_usage(open_usage: dict, final_output: int) -> list[bytes]:
    """A legal Anthropic stream whose two usage reports both increase.

    Anthropic reports ``input_tokens`` once, in ``message_start``, and a
    growing ``output_tokens`` in ``message_delta``. Both frames are built here
    so the lifecycle test can show where the adapter projects each one.
    """
    start = message_payload()
    start["content"] = []
    start["stop_reason"] = None
    start["usage"] = open_usage
    return [
        sse_event(
            json.dumps({"type": "message_start", "message": start}),
            event="message_start",
        ),
        event_content_block_start(),
        event_content_block_delta("po"),
        event_content_block_delta("ng"),
        event_content_block_stop(),
        sse_event(
            json.dumps(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": final_output},
                }
            ),
            event="message_delta",
        ),
        event_message_stop(),
    ]


async def test_d5a_baseline_ainvoke(offline_anthropic):
    """One normal non-stream message pins the whole usage projection."""
    offline_anthropic.respond(json_response(message_payload("pong")))
    model = offline_anthropic.model()

    result = await model.ainvoke("ping")

    assert len(offline_anthropic.requests) == 1
    assert request_json(offline_anthropic.requests[0])["messages"] == [
        {"role": "user", "content": "ping"}
    ]

    assert result.content == "pong"
    # _create_usage_metadata recomposes input as base + cache_read +
    # cache_creation and drops the cache keys when Anthropic reports them as
    # None, so total is derived (3 + 1) rather than read off the wire.
    assert result.usage_metadata == {
        "input_tokens": 3,
        "output_tokens": 1,
        "total_tokens": 4,
        "input_token_details": {},
    }
    assert result.response_metadata["stop_reason"] == "end_turn"
    assert result.response_metadata["model_name"] == REPRESENTATIVE_MODEL
    assert result.response_metadata["model_provider"] == "anthropic"
    # The untouched model dump is published alongside the renamed projection.
    assert result.response_metadata["usage"]["input_tokens"] == 3
    assert result.response_metadata["usage"]["output_tokens"] == 1


async def test_d5a_baseline_astream(offline_anthropic):
    """Streamed usage lands on the terminal frame, and its position is pinned.

    Anthropic is the one adapter that marks its own terminal frame with
    ``chunk_position="last"``, so langchain-core appends no sentinel here and
    the last chunk really is the adapter's.
    """
    offline_anthropic.respond(
        lambda request: offline_anthropic.sse_response(
            offline_anthropic.sse_body(full_text_stream("po", "ng"))
        )
    )
    model = offline_anthropic.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_anthropic.requests) == 1
    body = request_json(offline_anthropic.requests[0])
    assert body["messages"] == [{"role": "user", "content": "ping"}]
    assert body["stream"] is True

    # message_start, two text deltas, message_delta. The text content_block
    # frame and both stop frames are unhandled event types and yield nothing.
    assert len(chunks) == 4
    assert [c.content for c in chunks] == ["", "po", "ng", ""]
    # message_start deliberately withholds usage ("not needed ... reported in
    # message_delta"), and message_delta carries no input_tokens, so the
    # streamed input count is 0 even though the opening frame reported 3.
    # That 0 is the adapter's omission, not a provider-reported zero.
    assert [c.usage_metadata for c in chunks] == [
        None,
        None,
        None,
        {
            "input_tokens": 0,
            "output_tokens": 2,
            "total_tokens": 2,
            "input_token_details": {},
        },
    ]
    assert chunks[0].response_metadata["model_name"] == REPRESENTATIVE_MODEL
    assert chunks[-1].response_metadata["stop_reason"] == "end_turn"
    assert chunks[-1].response_metadata["stop_sequence"] is None
    assert all(c.response_metadata["model_provider"] == "anthropic" for c in chunks)
    # The adapter marks this frame itself, which is why core adds no sentinel.
    assert chunks[-1].chunk_position == "last"
    assert chunks[-2].chunk_position is None

    aggregated = d5a_add_chunks(chunks)
    assert aggregated.content == "pong"
    assert aggregated.usage_metadata == {
        "input_tokens": 0,
        "output_tokens": 2,
        "total_tokens": 2,
        "input_token_details": {},
    }
    assert aggregated.response_metadata["stop_reason"] == "end_turn"


async def test_d5a_usage_lifecycle_astream(offline_anthropic):
    """Two increasing cumulative wire reports; only the last one is projected.

    The wire raises output_tokens 1 -> 5 across the two usage frames. The
    adapter projects neither as a running total: message_start is withheld by
    design and message_delta is taken as-is. So the streamed lifecycle carries
    one terminal report, and the input count reported at open is not merged.
    """
    offline_anthropic.respond(
        lambda request: offline_anthropic.sse_response(
            offline_anthropic.sse_body(
                d5a_stream_with_usage(
                    open_usage={"input_tokens": 3, "output_tokens": 1},
                    final_output=5,
                )
            )
        )
    )
    model = offline_anthropic.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_anthropic.requests) == 1
    body = request_json(offline_anthropic.requests[0])
    assert body["messages"] == [{"role": "user", "content": "ping"}]
    assert body["stream"] is True

    # The opening frame reports input 3 / output 1, yet it carries no
    # usage_metadata at all.
    assert len(chunks) == 4
    assert chunks[0].usage_metadata is None
    assert chunks[0].response_metadata["model_name"] == REPRESENTATIVE_MODEL
    assert [c.usage_metadata for c in chunks] == [
        None,
        None,
        None,
        {
            "input_tokens": 0,
            "output_tokens": 5,
            "total_tokens": 5,
            "input_token_details": {},
        },
    ]
    # Only the terminal adapter frame carries usage, and nothing follows it.
    assert chunks[1].usage_metadata is None
    assert chunks[2].usage_metadata is None
    assert chunks[-1].chunk_position == "last"

    aggregated = d5a_add_chunks(chunks)
    assert aggregated.content == "pong"
    assert aggregated.usage_metadata == {
        "input_tokens": 0,
        "output_tokens": 5,
        "total_tokens": 5,
        "input_token_details": {},
    }


# ===========================================================================
# D5A-2: completion boundaries and missing usage (7 nodes)
#
# Appended at EOF only. Every byte above this marker is the D4 baseline plus
# the accepted D5A-1 section, and is unchanged. Each fixed expectation was
# derived from the locked sources below, not from an observed run:
#
#   anthropic/types/raw_message_delta_event.py:25,35  stop_reason/usage
#   anthropic/types/stop_reason.py                    StopReason Literal
#   anthropic/_client.py:573 + _base_client.py:669   lenient construct
#   langchain_anthropic/chat_models.py:1602-1631      message_delta + last
#   langchain_anthropic/chat_models.py:1625-1628      message_stop unhandled
#   langchain_anthropic/chat_models.py:2255-2304      _create_usage_metadata
#   langchain_core/language_models/chat_models.py:935-950   core sentinel
#   langchain_core/messages/ai.py:676-681             no usage -> None
# ===========================================================================


def d5a2_message_delta(
    *,
    stop_reason: str = "end_turn",
    omit_stop_reason: bool = False,
    omit_usage: bool = False,
) -> bytes:
    """One ``message_delta`` frame with individually removable parts.

    ``event_message_delta`` above always sends both ``delta.stop_reason`` and
    ``usage``, so it cannot express "reason absent" or "usage absent" as
    separate wire shapes.

    The missing-reason branch still carries ``stop_sequence``, so omitting the
    reason differs from sending an explicit null in exactly one key. Both
    fields default to None on the SDK's ``Delta``, so this changes nothing
    about the projected message.
    """
    delta: dict = {"stop_sequence": None}
    if not omit_stop_reason:
        delta["stop_reason"] = stop_reason
    body: dict = {"type": "message_delta", "delta": delta}
    if not omit_usage:
        body["usage"] = {"output_tokens": 2}
    return sse_event(json.dumps(body), event="message_delta")


def d5a2_stream(*, message_delta: bytes) -> list[bytes]:
    """A full Anthropic stream whose ``message_delta`` frame is supplied."""
    return [
        event_message_start(),
        event_content_block_start(),
        event_content_block_delta("po"),
        event_content_block_delta("ng"),
        event_content_block_stop(),
        message_delta,
        event_message_stop(),
    ]


def d5a_assert_anthropic_sentinel(chunk: AIMessageChunk) -> None:
    """Pin the langchain-core terminal sentinel.

    ``chat_models.py:935-950`` appends this chunk whenever the adapter's last
    chunk carries no ``chunk_position``. Anthropic normally marks its own final
    frame, so the sentinel is expected only in the boundary cases where the
    stop_reason that triggers that marking is absent.
    """
    assert chunk.content == ""
    assert chunk.usage_metadata is None
    assert chunk.response_metadata == {}
    assert chunk.chunk_position == "last"


async def test_d5a_completion_missing_astream(offline_anthropic):
    """No stop_reason key in message_delta, while body text and usage remain.

    Because the reason is falsy the adapter does not mark this frame as the
    last one, so langchain-core appends its sentinel afterwards.
    """
    offline_anthropic.respond(
        lambda request: offline_anthropic.sse_response(
            offline_anthropic.sse_body(
                d5a2_stream(message_delta=d5a2_message_delta(omit_stop_reason=True))
            )
        )
    )
    model = offline_anthropic.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_anthropic.requests) == 1
    body = request_json(offline_anthropic.requests[0])
    assert body["messages"] == [{"role": "user", "content": "ping"}]
    assert body["stream"] is True

    assert len(chunks) == 5
    assert [c.content for c in chunks] == ["", "po", "ng", "", ""]
    # message_delta carries no input_tokens, so the streamed input count is 0:
    # the adapter withholds message_start usage by design. That 0 is an
    # adapter omission, not a provider-reported zero.
    assert [c.usage_metadata for c in chunks] == [
        None,
        None,
        None,
        {
            "input_tokens": 0,
            "output_tokens": 2,
            "total_tokens": 2,
            "input_token_details": {},
        },
        None,
    ]
    assert chunks[3].response_metadata == {
        "stop_reason": None,
        "stop_sequence": None,
        "model_provider": "anthropic",
    }
    assert chunks[3].chunk_position is None
    d5a_assert_anthropic_sentinel(chunks[4])

    aggregated = d5a_add_chunks(chunks)
    assert aggregated.content == "pong"
    assert aggregated.usage_metadata == {
        "input_tokens": 0,
        "output_tokens": 2,
        "total_tokens": 2,
        "input_token_details": {},
    }
    assert aggregated.response_metadata["stop_reason"] is None
    assert aggregated.response_metadata["model_name"] == REPRESENTATIVE_MODEL


async def test_d5a_completion_null_astream(offline_anthropic):
    """An explicit JSON null stop_reason, kept apart from the missing key.

    Delta.stop_reason defaults to None, so an absent key and an explicit null
    decode identically and the projection is byte-for-byte the same as
    test_d5a_completion_missing_astream. The wire really does differ.
    """
    offline_anthropic.respond(
        lambda request: offline_anthropic.sse_response(
            offline_anthropic.sse_body(
                d5a2_stream(message_delta=d5a2_message_delta(stop_reason=None))
            )
        )
    )
    model = offline_anthropic.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_anthropic.requests) == 1
    body = request_json(offline_anthropic.requests[0])
    assert body["messages"] == [{"role": "user", "content": "ping"}]
    assert body["stream"] is True

    missing_delta = json.loads(
        d5a2_message_delta(omit_stop_reason=True).split(b"data: ", 1)[1]
    )["delta"]
    null_delta = json.loads(
        d5a2_message_delta(stop_reason=None).split(b"data: ", 1)[1]
    )["delta"]
    # The two deltas differ in exactly one key: stop_reason is absent on one
    # side and explicitly null on the other, with stop_sequence sent by both.
    assert missing_delta == {"stop_sequence": None}
    assert null_delta == {"stop_sequence": None, "stop_reason": None}
    assert set(null_delta) - set(missing_delta) == {"stop_reason"}

    assert len(chunks) == 5
    assert [c.content for c in chunks] == ["", "po", "ng", "", ""]
    assert [c.usage_metadata for c in chunks] == [
        None,
        None,
        None,
        {
            "input_tokens": 0,
            "output_tokens": 2,
            "total_tokens": 2,
            "input_token_details": {},
        },
        None,
    ]
    assert chunks[3].response_metadata == {
        "stop_reason": None,
        "stop_sequence": None,
        "model_provider": "anthropic",
    }
    assert chunks[3].chunk_position is None
    d5a_assert_anthropic_sentinel(chunks[4])

    aggregated = d5a_add_chunks(chunks)
    assert aggregated.content == "pong"
    assert aggregated.usage_metadata == {
        "input_tokens": 0,
        "output_tokens": 2,
        "total_tokens": 2,
        "input_token_details": {},
    }
    assert aggregated.response_metadata["stop_reason"] is None


async def test_d5a_usage_missing_astream(offline_anthropic):
    """Completion signal normal, usage simply not sent.

    RawMessageDeltaEvent.usage is a required field, but the SDK constructs
    responses leniently and fills the absent field with its default None, so
    _create_usage_metadata(None) derives an all-zero usage. Those zeros are
    SDK-derived: the provider reported no usage at all.
    """
    offline_anthropic.respond(
        lambda request: offline_anthropic.sse_response(
            offline_anthropic.sse_body(
                d5a2_stream(message_delta=d5a2_message_delta(omit_usage=True))
            )
        )
    )
    model = offline_anthropic.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_anthropic.requests) == 1
    body = request_json(offline_anthropic.requests[0])
    assert body["messages"] == [{"role": "user", "content": "ping"}]
    assert body["stream"] is True

    # stop_reason is still present, so this frame is marked last and core adds
    # no sentinel.
    assert len(chunks) == 4
    assert [c.content for c in chunks] == ["", "po", "ng", ""]
    assert [c.usage_metadata for c in chunks] == [
        None,
        None,
        None,
        {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "input_token_details": {},
        },
    ]
    assert chunks[3].response_metadata == {
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "model_provider": "anthropic",
    }
    assert chunks[3].chunk_position == "last"

    aggregated = d5a_add_chunks(chunks)
    assert aggregated.content == "pong"
    assert aggregated.usage_metadata == {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "input_token_details": {},
    }
    assert aggregated.response_metadata["stop_reason"] == "end_turn"


async def test_d5a_truncated_ainvoke(offline_anthropic):
    """The protocol's own truncation reason, "max_tokens"."""
    payload = message_payload("pong")
    payload["stop_reason"] = "max_tokens"
    offline_anthropic.respond(json_response(payload))
    model = offline_anthropic.model()

    result = await model.ainvoke("ping")

    assert len(offline_anthropic.requests) == 1
    assert request_json(offline_anthropic.requests[0])["messages"] == [
        {"role": "user", "content": "ping"}
    ]

    assert result.content == "pong"
    assert result.usage_metadata == {
        "input_tokens": 3,
        "output_tokens": 1,
        "total_tokens": 4,
        "input_token_details": {},
    }
    # "max_tokens" is a real StopReason literal; the adapter republishes the
    # raw dump and adds no truncation flag of its own.
    assert result.response_metadata["stop_reason"] == "max_tokens"
    assert result.response_metadata["model_name"] == REPRESENTATIVE_MODEL
    assert result.response_metadata["model_provider"] == "anthropic"
    assert result.response_metadata["usage"]["input_tokens"] == 3
    assert result.response_metadata["usage"]["output_tokens"] == 1


async def test_d5a_truncated_astream(offline_anthropic):
    """The same "max_tokens" reason through the streaming entry."""
    offline_anthropic.respond(
        lambda request: offline_anthropic.sse_response(
            offline_anthropic.sse_body(
                d5a2_stream(message_delta=d5a2_message_delta(stop_reason="max_tokens"))
            )
        )
    )
    model = offline_anthropic.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_anthropic.requests) == 1
    body = request_json(offline_anthropic.requests[0])
    assert body["messages"] == [{"role": "user", "content": "ping"}]
    assert body["stream"] is True

    assert len(chunks) == 4
    assert [c.content for c in chunks] == ["", "po", "ng", ""]
    assert [c.usage_metadata for c in chunks] == [
        None,
        None,
        None,
        {
            "input_tokens": 0,
            "output_tokens": 2,
            "total_tokens": 2,
            "input_token_details": {},
        },
    ]
    assert chunks[3].response_metadata == {
        "stop_reason": "max_tokens",
        "stop_sequence": None,
        "model_provider": "anthropic",
    }
    assert chunks[3].chunk_position == "last"

    aggregated = d5a_add_chunks(chunks)
    assert aggregated.content == "pong"
    assert aggregated.usage_metadata == {
        "input_tokens": 0,
        "output_tokens": 2,
        "total_tokens": 2,
        "input_token_details": {},
    }
    assert aggregated.response_metadata["stop_reason"] == "max_tokens"


async def test_d5a_unknown_ainvoke(offline_anthropic):
    """A synthetic unknown stop_reason is accepted, not rejected.

    The SDK builds responses leniently and does not validate Literal fields, so
    "d5a_unknown" reaches response_metadata verbatim.
    """
    payload = message_payload("pong")
    payload["stop_reason"] = "d5a_unknown"
    offline_anthropic.respond(json_response(payload))
    model = offline_anthropic.model()

    result = await model.ainvoke("ping")

    assert len(offline_anthropic.requests) == 1
    assert request_json(offline_anthropic.requests[0])["messages"] == [
        {"role": "user", "content": "ping"}
    ]

    assert result.content == "pong"
    assert result.usage_metadata == {
        "input_tokens": 3,
        "output_tokens": 1,
        "total_tokens": 4,
        "input_token_details": {},
    }
    assert result.response_metadata["stop_reason"] == "d5a_unknown"
    assert result.response_metadata["model_name"] == REPRESENTATIVE_MODEL
    assert result.response_metadata["model_provider"] == "anthropic"


async def test_d5a_terminal_frame_missing_astream(offline_anthropic):
    """A complete prefix of frames with no message_stop.

    message_stop reaches the SDK decoder but is an unhandled event type for
    the adapter, and the previous frame is already marked last, so removing it
    changes nothing observable above the adapter. This proves the prefix was
    delivered, not that the SDK noticed the missing terminator.
    """
    stream = d5a2_stream(message_delta=d5a2_message_delta())
    assert stream[-1] == event_message_stop()
    offline_anthropic.respond(
        lambda request: offline_anthropic.sse_response(
            offline_anthropic.sse_body(stream[:-1])
        )
    )
    model = offline_anthropic.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_anthropic.requests) == 1
    body = request_json(offline_anthropic.requests[0])
    assert body["messages"] == [{"role": "user", "content": "ping"}]
    assert body["stream"] is True

    assert len(chunks) == 4
    assert [c.content for c in chunks] == ["", "po", "ng", ""]
    assert [c.usage_metadata for c in chunks] == [
        None,
        None,
        None,
        {
            "input_tokens": 0,
            "output_tokens": 2,
            "total_tokens": 2,
            "input_token_details": {},
        },
    ]
    assert chunks[3].response_metadata["stop_reason"] == "end_turn"
    assert chunks[3].chunk_position == "last"

    aggregated = d5a_add_chunks(chunks)
    assert aggregated.content == "pong"
    assert aggregated.usage_metadata == {
        "input_tokens": 0,
        "output_tokens": 2,
        "total_tokens": 2,
        "input_token_details": {},
    }
