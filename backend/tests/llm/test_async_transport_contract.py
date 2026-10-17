"""D4a-1: offline evidence for the real OpenAI SDK's async transport contract.

Each test maps back to one numbered contract item in the D4a-1 task sheet.
Four failure surfaces are deliberately kept apart and never merged:

* **send retries** -- the SDK re-issues the request, visible as a growing
  ``len(offline.requests)``.
* **body not replayed** -- a failure part-way through the response body cannot
  be retried, so the request count stays at one.
* **adapter chunk guard** -- ``StreamChunkTimeoutError`` is raised by the
  adapter's wrapper around the SDK iterator, after the response already
  succeeded.
* **task cancellation** -- ``asyncio.CancelledError`` propagates untouched.

Every assertion describes what the locked stack actually does. Nothing here is
normalised into a product-level "unavailable" result, and no test swallows a
surprising error to make itself pass.
"""

from __future__ import annotations

import asyncio
import json
import sys

import httpx
import openai
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_openai import StreamChunkTimeoutError

from tests.llm.sdk_transport_support import (  # noqa: F401  (autouse fixtures)
    DONE_EVENT,
    REPRESENTATIVE_MODEL,
    SDK_DEFAULT_BASE_URL,
    SYNTHETIC_OPENAI_KEY,
    assert_module_from_backend,
    assert_module_is_test_module,
    backend_root,
    deny_sockets,
    fail_with,
    observe_explicit_cancellation,
    offline_env,
    offline_openai,
    reclaim_own_task,
    send,
    sse_event,
    wait_for,
)


# ---------------------------------------------------------------------------
# Wire builders
# ---------------------------------------------------------------------------


def completion_body(
    text: str | None = "pong",
    *,
    finish_reason: str | None = "stop",
) -> dict:
    """A normal Chat Completions payload, with knobs for the bad-wire cases."""
    message: dict = {"role": "assistant", "refusal": None}
    message["content"] = text
    return {
        "id": "chatcmpl-offline",
        "object": "chat.completion",
        "created": 0,
        "model": REPRESENTATIVE_MODEL,
        "choices": [
            {
                "index": 0,
                "message": message,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }


def stream_chunk(
    content: str | None = None,
    *,
    finish_reason: str | None = None,
    usage: dict | None = None,
    choices: list | None = None,
) -> str:
    """Build one SSE ``data:`` payload.

    ``choices`` overrides the whole choice list, which is how the tests express
    an explicit ``choices[0].delta == {}`` (legal, empty) separately from
    ``choices == []`` (no choice at all).
    """
    payload: dict = {
        "id": "chatcmpl-offline",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": REPRESENTATIVE_MODEL,
        "choices": choices if choices is not None else _default_choices(content, finish_reason),
    }
    if usage is not None:
        payload["usage"] = usage
    return json.dumps(payload)


def _default_choices(content: str | None, finish_reason: str | None) -> list:
    if content is None and finish_reason is None:
        return []
    delta: dict = {}
    if content is not None:
        delta["content"] = content
    return [
        {"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish_reason}
    ]


def empty_delta_chunk() -> str:
    """``choices[0].delta == {}`` -- a legal choice that carries no content."""
    return stream_chunk(choices=[{"index": 0, "delta": {}, "logprobs": None, "finish_reason": None}])


def no_choices_chunk() -> str:
    """``choices == []`` -- a legal frame that names no choice at all."""
    return stream_chunk(choices=[])


def null_delta_chunk() -> str:
    """``choices[0].delta is None`` -- the SDK's "no content here" marker."""
    return stream_chunk(choices=[{"index": 0, "delta": None, "finish_reason": None}])


def json_response(payload: dict, *, status: int = 200, headers: dict | None = None):
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload, headers=headers or {})

    return responder


def request_json(request: httpx.Request) -> dict:
    return json.loads(request.content)


def raise_in_transport(exc_factory):
    """A responder that fails during send, before any response exists."""

    def responder(request: httpx.Request) -> httpx.Response:
        raise exc_factory()

    return responder


# ---------------------------------------------------------------------------
# Harness self-checks (offline isolation, default branch, source identity)
# ---------------------------------------------------------------------------


def test_offline_harness_rejects_dns_and_connect_attempts():
    """The socket deny is real, not a printed claim."""
    import socket

    with pytest.raises(AssertionError, match="real network access"):
        socket.getaddrinfo("api.openai.com", 443)
    with pytest.raises(AssertionError, match="real network access"):
        socket.create_connection(("api.openai.com", 443))


def test_offline_harness_uses_sdk_default_httpx_clients(offline_openai):
    """The client keeps the SDK's own defaults, not bare-httpx defaults."""
    model = offline_openai.model()

    async_client = model.root_async_client._client
    sync_client = model.root_client._client

    assert type(async_client) is openai.DefaultAsyncHttpxClient
    assert type(sync_client) is openai.DefaultHttpxClient

    # A bare httpx.AsyncClient would have shipped follow_redirects=False.
    assert async_client.follow_redirects is True
    assert sync_client.follow_redirects is True


def test_offline_harness_pins_sdk_default_construction(offline_openai):
    """The adapter built its own client the default way, on a mock socket."""
    model = offline_openai.model()

    assert type(model).__name__ == "ChatOpenAI"
    assert model.model_name == REPRESENTATIVE_MODEL
    assert model.openai_api_base is None
    assert model.stream_usage is True, (
        "default branch must keep stream_usage on; passing an http_async_client "
        "would have silently turned this off"
    )

    async_client = model.root_async_client._client
    assert isinstance(async_client._transport, httpx.MockTransport)
    assert async_client.trust_env is False
    # httpx normalises the SDK default URL with a trailing slash.
    assert str(async_client.base_url) == SDK_DEFAULT_BASE_URL + "/"
    assert model.root_async_client.max_retries == 2

    # Nothing was injected into the model to reach the mock socket.
    assert model.http_async_client is None
    assert model.http_client is None
    assert model.openai_api_base is None
    # Both resources were built by the adapter itself, one per client.
    assert model.client is model.root_client.chat.completions
    assert model.async_client is model.root_async_client.chat.completions


def test_offline_harness_timeouts_are_all_none(offline_openai):
    """Model and SDK report "no timeout"; HTTPX reports it field by field.

    Worth pinning: the SDK ships ``DEFAULT_TIMEOUT`` with a 5s connect budget,
    but the adapter passes ``timeout=None`` explicitly, so ``setdefault`` never
    fires and the SDK default is not in effect on this path.
    """
    model = offline_openai.model()

    assert model.request_timeout is None
    assert model.root_async_client.timeout is None
    assert model.root_client.timeout is None

    for httpx_client in offline_openai.async_clients + offline_openai.sync_clients:
        assert httpx_client.timeout.connect is None
        assert httpx_client.timeout.read is None
        assert httpx_client.timeout.write is None
        assert httpx_client.timeout.pool is None


def test_source_identity_is_portable_and_exact():
    """app code and helpers come from this checkout, resolved from this file.

    The root is derived from the test file's own location, so the same
    assertions hold in any checkout; membership is exact resolved equality, so
    a different checkout that merely shares a path prefix is rejected.
    """
    import app.llm.provider as provider_module
    import tests.llm.sdk_transport_support as support_module

    root = backend_root()
    assert root.name == "backend"
    assert (root / "pyproject.toml").is_file()
    assert (root / "tests" / "llm" / "__init__.py").is_file()

    assert_module_from_backend(provider_module, "app/llm/provider.py")
    assert_module_from_backend(support_module, "tests/llm/sdk_transport_support.py")
    assert_module_is_test_module(sys.modules[__name__])


def test_source_identity_helper_rejects_a_lookalike_path():
    """The exactness check must not accept a same-prefix different path."""
    from tests.llm.sdk_transport_support import assert_module_from_backend as check

    class _Fake:
        __name__ = "lookalike_provider"
        __file__ = str(backend_root() / "..-other" / "backend" / "app" / "llm" / "provider.py")

    with pytest.raises(AssertionError, match="but this checkout expects"):
        check(_Fake(), "app/llm/provider.py")


# ---------------------------------------------------------------------------
# 1. Success: ainvoke body, astream multi-chunk SSE with [DONE]
# ---------------------------------------------------------------------------


async def test_ainvoke_sends_real_chat_completions_request(offline_openai):
    offline_openai.respond(json_response(completion_body("pong")))
    model = offline_openai.model()

    result = await model.ainvoke("ping")

    assert len(offline_openai.requests) == 1
    request = offline_openai.requests[0]
    assert request.method == "POST"
    assert str(request.url) == f"{SDK_DEFAULT_BASE_URL}/chat/completions"
    assert request.url.host == "api.openai.com"
    assert request.headers["authorization"] == f"Bearer {SYNTHETIC_OPENAI_KEY}"
    body = request_json(request)
    assert body["model"] == REPRESENTATIVE_MODEL
    assert body["messages"] == [{"role": "user", "content": "ping"}]
    assert body.get("stream") is not True
    assert isinstance(result, AIMessage)
    assert result.content == "pong"


async def test_astream_reads_real_sse_chunks(offline_openai):
    offline_openai.respond(
        lambda request: offline_openai.sse_response(
            offline_openai.sse_body(
                [
                    sse_event(stream_chunk("po")),
                    sse_event(stream_chunk("ng")),
                    sse_event(
                        stream_chunk(
                            usage={
                                "prompt_tokens": 3,
                                "completion_tokens": 2,
                                "total_tokens": 5,
                            }
                        )
                    ),
                ]
            )
        )
    )
    model = offline_openai.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_openai.requests) == 1
    body = request_json(offline_openai.requests[0])
    assert body.get("stream") is True
    assert body["stream_options"] == {"include_usage": True}, (
        "stream_usage defaults on the default branch, so the real request "
        "carries stream_options without the test passing it"
    )
    assert "".join(
        c.content for c in chunks if isinstance(c.content, str)
    ) == "pong"
    assert all(isinstance(c, AIMessageChunk) for c in chunks)


# ---------------------------------------------------------------------------
# 2. Empty: legal empty delta vs no choice vs no chunk at all
# ---------------------------------------------------------------------------


async def test_ainvoke_empty_content_is_a_real_empty_message(offline_openai):
    offline_openai.respond(json_response(completion_body("")))
    model = offline_openai.model()

    result = await model.ainvoke("ping")

    assert len(offline_openai.requests) == 1
    assert isinstance(result, AIMessage)
    assert result.content == ""


async def test_ainvoke_no_choices_is_distinct_from_empty_content(offline_openai):
    payload = completion_body("")
    payload["choices"] = []
    offline_openai.respond(json_response(payload))
    model = offline_openai.model()

    with pytest.raises(IndexError):
        await model.ainvoke("ping")

    assert len(offline_openai.requests) == 1


async def test_astream_explicit_empty_delta_yields_empty_chunk(offline_openai):
    """``choices[0].delta == {}`` is legal and yields a chunk with no content."""
    offline_openai.respond(
        lambda request: offline_openai.sse_response(
            offline_openai.sse_body([sse_event(empty_delta_chunk())])
        )
    )
    model = offline_openai.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(chunks) >= 1
    assert "".join(
        c.content for c in chunks if isinstance(c.content, str)
    ) == ""


async def test_astream_chunk_with_no_choices_still_yields_empty_chunk(offline_openai):
    """``choices == []`` is a different frame and is named as such."""
    offline_openai.respond(
        lambda request: offline_openai.sse_response(
            offline_openai.sse_body(
                [sse_event(no_choices_chunk()), sse_event(stream_chunk("ok"))]
            )
        )
    )
    model = offline_openai.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_openai.requests) == 1
    assert "".join(c.content for c in chunks) == "ok"


async def test_astream_without_any_chunk_raises_instead_of_yielding(offline_openai):
    """A stream that ends before any chunk is a hard error, not an empty stream.

    This is what separates "legal empty delta" (yields an empty chunk) from
    "no chunk at all" (raises).
    """
    offline_openai.respond(
        lambda request: offline_openai.sse_response(
            offline_openai.sse_body([])
        )
    )
    model = offline_openai.model()

    with pytest.raises(ValueError, match="No generation chunks were returned"):
        async for _ in model.astream("ping"):
            pass

    assert len(offline_openai.requests) == 1


# ---------------------------------------------------------------------------
# 3. Bad wire: real SDK/adapter exceptions, never silently normalised
# ---------------------------------------------------------------------------


async def test_ainvoke_missing_choices_lands_on_the_null_choices_error(offline_openai):
    """A *missing* choices key and a *null* choices key are indistinguishable.

    The openai SDK builds its response model leniently, so an absent
    ``choices`` is materialised as ``None`` before the adapter ever sees it.
    That makes the adapter's separate "Response missing 'choices'" KeyError
    branch unreachable through the real SDK on this path.
    """
    payload = completion_body("pong")
    del payload["choices"]
    offline_openai.respond(json_response(payload))
    model = offline_openai.model()

    with pytest.raises(TypeError, match="null value for 'choices'"):
        await model.ainvoke("ping")

    assert len(offline_openai.requests) == 1


async def test_ainvoke_null_choices_raises_narrow_error(offline_openai):
    payload = completion_body("pong")
    payload["choices"] = None
    offline_openai.respond(json_response(payload))
    model = offline_openai.model()

    with pytest.raises(TypeError, match="null value for 'choices'"):
        await model.ainvoke("ping")

    assert len(offline_openai.requests) == 1


async def test_ainvoke_null_message_raises_narrow_adapter_error(offline_openai):
    """``message: null`` reaches the adapter's message conversion as None.

    The narrow failure is fixed here as observed; it is deliberately not
    broadened to ``Exception`` and not normalised into an "unavailable" result.
    """
    payload = completion_body("pong")
    payload["choices"][0]["message"] = None
    offline_openai.respond(json_response(payload))
    model = offline_openai.model()

    with pytest.raises(AttributeError) as excinfo:
        await model.ainvoke("ping")

    assert "NoneType" in str(excinfo.value)
    assert len(offline_openai.requests) == 1


async def test_astream_invalid_json_raises_decode_error(offline_openai):
    offline_openai.respond(
        lambda request: offline_openai.sse_response(
            offline_openai.sse_body([b"data: {not-json\n\n"])
        )
    )
    model = offline_openai.model()

    with pytest.raises(json.JSONDecodeError):
        async for _ in model.astream("ping"):
            pass

    assert len(offline_openai.requests) == 1


async def test_astream_null_delta_is_dropped_not_an_exception(offline_openai):
    """A null delta is the SDK's 'no content here' signal; it yields nothing."""
    offline_openai.respond(
        lambda request: offline_openai.sse_response(
            offline_openai.sse_body(
                [
                    sse_event(null_delta_chunk()),
                    sse_event(stream_chunk("ok")),
                ]
            )
        )
    )
    model = offline_openai.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_openai.requests) == 1
    assert "".join(c.content for c in chunks) == "ok"


# ---------------------------------------------------------------------------
# 4. HTTP status for both entries: exact SDK type and exact attempt count
# ---------------------------------------------------------------------------


_HTTP_STATUS_MATRIX = [
    pytest.param(401, openai.AuthenticationError, 1, id="401-auth"),
    pytest.param(429, openai.RateLimitError, 3, id="429-ratelimit"),
    pytest.param(503, openai.InternalServerError, 3, id="503-server"),
]


@pytest.mark.parametrize("status,expected,attempts", _HTTP_STATUS_MATRIX)
async def test_ainvoke_http_status_maps_to_real_sdk_error(
    offline_openai, status, expected, attempts
):
    """401 is not retryable; 429 and 5xx exhaust max_retries=2 (3 attempts)."""
    offline_openai.respond(json_response({"error": {"message": "nope"}}, status=status))
    model = offline_openai.model()

    with pytest.raises(expected):
        await model.ainvoke("ping")

    assert len(offline_openai.requests) == attempts


@pytest.mark.parametrize("status,expected,attempts", _HTTP_STATUS_MATRIX)
async def test_astream_http_status_maps_to_real_sdk_error(
    offline_openai, status, expected, attempts
):
    """Streaming shares the send-time status policy, including retry counts."""
    offline_openai.respond(json_response({"error": {"message": "nope"}}, status=status))
    model = offline_openai.model()

    with pytest.raises(expected):
        async for _ in model.astream("ping"):
            pass

    assert len(offline_openai.requests) == attempts


async def test_retryable_then_success_returns_result_without_normalisation(offline_openai):
    """A 429 followed by a 200 succeeds on attempt 2; no product fallback."""
    responses = [
        lambda request: httpx.Response(429, json={"error": {"message": "slow down"}}),
        lambda request: offline_openai.sse_response(
            offline_openai.sse_body([sse_event(stream_chunk("recovered"))])
        ),
    ]
    calls = {"n": 0}

    def responder(request):
        index = calls["n"]
        calls["n"] += 1
        return responses[min(index, len(responses) - 1)](request)

    offline_openai.respond(responder)
    model = offline_openai.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_openai.requests) == 2
    assert "".join(c.content for c in chunks) == "recovered"


# ---------------------------------------------------------------------------
# 5. SDK override: x-should-retry header wins over status policy
# ---------------------------------------------------------------------------


async def test_should_retry_true_on_401_retries_and_succeeds(offline_openai):
    responses = [
        lambda request: httpx.Response(
            401, json={"error": {"message": "nope"}}, headers={"x-should-retry": "true"}
        ),
        lambda request: offline_openai.sse_response(
            offline_openai.sse_body([sse_event(stream_chunk("ok"))])
        ),
    ]
    calls = {"n": 0}

    def responder(request):
        index = calls["n"]
        calls["n"] += 1
        return responses[min(index, len(responses) - 1)](request)

    offline_openai.respond(responder)
    model = offline_openai.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_openai.requests) == 2
    assert "".join(c.content for c in chunks) == "ok"


async def test_should_retry_false_on_503_fails_immediately(offline_openai):
    offline_openai.respond(
        json_response(
            {"error": {"message": "hard stop"}},
            status=503,
            headers={"x-should-retry": "false"},
        )
    )
    model = offline_openai.model()

    with pytest.raises(openai.InternalServerError):
        await model.ainvoke("ping")

    assert len(offline_openai.requests) == 1, (
        "x-should-retry:false must suppress the SDK's default 5xx retry"
    )


# ---------------------------------------------------------------------------
# 6. Send-phase failures for both entries: HTTPX error -> real SDK wrapper
# ---------------------------------------------------------------------------


_SEND_FAILURE_MATRIX = [
    pytest.param(
        httpx.ReadTimeout, "simulated read timeout", openai.APITimeoutError, id="readtimeout"
    ),
    pytest.param(
        httpx.ConnectError, "simulated connect failure", openai.APIConnectionError, id="connecterror"
    ),
]


@pytest.mark.parametrize("exc_cls,message,expected", _SEND_FAILURE_MATRIX)
async def test_ainvoke_send_failure_exhausts_into_sdk_wrapper(
    offline_openai, exc_cls, message, expected
):
    """Send failures are retryable, so the default budget is 3 attempts."""
    offline_openai.respond(raise_in_transport(lambda: exc_cls(message)))
    model = offline_openai.model()

    with pytest.raises(expected):
        await model.ainvoke("ping")

    assert len(offline_openai.requests) == 3
    assert model.max_retries in (None, 2), "max_retries must stay at the SDK default"


@pytest.mark.parametrize("exc_cls,message,expected", _SEND_FAILURE_MATRIX)
async def test_astream_send_failure_exhausts_into_sdk_wrapper(
    offline_openai, exc_cls, message, expected
):
    """The same send-time failures on the streaming entry."""
    offline_openai.respond(raise_in_transport(lambda: exc_cls(message)))
    model = offline_openai.model()

    with pytest.raises(expected):
        async for _ in model.astream("ping"):
            pass

    assert len(offline_openai.requests) == 3


# ---------------------------------------------------------------------------
# 7. Body-read failures: 2 exception types x before/after the first content
# ---------------------------------------------------------------------------

_BODY_FAILURE_MATRIX = [
    pytest.param(httpx.ReadTimeout, "stalled on timeout", id="readtimeout"),
    pytest.param(httpx.ReadError, "stalled on read", id="readerror"),
]


@pytest.mark.parametrize("exc_cls,message", _BODY_FAILURE_MATRIX)
async def test_astream_body_failure_before_first_chunk(offline_openai, exc_cls, message):
    body = offline_openai.body([fail_with(exc_cls(message))])
    offline_openai.respond(lambda request: offline_openai.sse_response(body))
    model = offline_openai.model()

    received = []
    with pytest.raises(exc_cls, match=message):
        async for chunk in model.astream("ping"):
            received.append(chunk)

    assert len(offline_openai.requests) == 1, "a body failure cannot be replayed"
    assert received == [], "no content chunk was ever delivered"
    # Observed here, before the fixture actively closes the client.
    assert body.close_count >= 1


@pytest.mark.parametrize("exc_cls,message", _BODY_FAILURE_MATRIX)
async def test_astream_body_failure_after_first_chunk(offline_openai, exc_cls, message):
    body = offline_openai.body(
        [
            send(sse_event(stream_chunk("par"))),
            fail_with(exc_cls(message)),
        ]
    )
    offline_openai.respond(lambda request: offline_openai.sse_response(body))
    model = offline_openai.model()

    received = []
    with pytest.raises(exc_cls, match=message):
        async for chunk in model.astream("ping"):
            received.append(chunk)

    assert len(offline_openai.requests) == 1
    assert "".join(c.content for c in received) == "par", "prefix must survive"
    assert body.close_count >= 1


# ---------------------------------------------------------------------------
# 8. HTTP 200 SSE error event
# ---------------------------------------------------------------------------


async def test_http_200_sse_error_event_raises_api_error(offline_openai):
    error_event = sse_event(
        json.dumps({"error": {"message": "upstream blew up", "type": "server_error"}}),
        event="error",
    )
    offline_openai.respond(
        lambda request: offline_openai.sse_response(offline_openai.sse_body([error_event]))
    )
    model = offline_openai.model()

    with pytest.raises(openai.APIError) as excinfo:
        async for _ in model.astream("ping"):
            pass

    assert "upstream blew up" in str(excinfo.value)
    assert not isinstance(excinfo.value, openai.RateLimitError), (
        "an SSE error event is not an HTTP 429, even if the body looks like one"
    )
    assert len(offline_openai.requests) == 1


# ---------------------------------------------------------------------------
# 9. Explicit task cancellation, gated on events (never sleep timing)
# ---------------------------------------------------------------------------


async def test_cancel_active_ainvoke_propagates_without_replay(offline_openai):
    gate = asyncio.Event()
    body = offline_openai.body([wait_for(gate), send(b"never-reached")])
    offline_openai.respond(lambda request: offline_openai.sse_response(body))
    model = offline_openai.model()

    task = asyncio.create_task(model.ainvoke("ping"))
    try:
        # Wait for the body to actually start being read; no sleep-based guessing.
        await asyncio.wait_for(body.started.wait(), timeout=5)
        task.cancel()
        await observe_explicit_cancellation(task, timeout=5)

        # Business assertions run while the gate is still unset and before the
        # fixture can close the client, so nothing here is cleanup evidence.
        assert len(offline_openai.requests) == 1
        assert body.close_count >= 1, "response created before cancellation must close"
    finally:
        await reclaim_own_task(task, gate=gate)


async def test_cancel_astream_before_first_chunk(offline_openai):
    gate = asyncio.Event()
    body = offline_openai.body([wait_for(gate), send(sse_event(stream_chunk("late")))])
    offline_openai.respond(lambda request: offline_openai.sse_response(body))
    model = offline_openai.model()

    received = []

    async def consume():
        async for chunk in model.astream("ping"):
            received.append(chunk)

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(body.started.wait(), timeout=5)
        task.cancel()
        await observe_explicit_cancellation(task, timeout=5)

        assert len(offline_openai.requests) == 1
        assert received == []
        assert body.close_count >= 1
    finally:
        await reclaim_own_task(task, gate=gate)


async def test_cancel_astream_after_first_chunk_keeps_prefix(offline_openai):
    gate = asyncio.Event()
    body = offline_openai.body(
        [send(sse_event(stream_chunk("first"))), wait_for(gate)]
    )
    offline_openai.respond(lambda request: offline_openai.sse_response(body))
    model = offline_openai.model()

    received = []
    # `first_chunk_seen` is set from inside the consume loop, so by the time it
    # fires the chunk has been produced and the iterator is parked on the gate.
    first_chunk_seen = asyncio.Event()

    async def consume_and_signal():
        async for chunk in model.astream("ping"):
            received.append(chunk)
            first_chunk_seen.set()

    task = asyncio.create_task(consume_and_signal())
    try:
        await asyncio.wait_for(first_chunk_seen.wait(), timeout=5)
        task.cancel()
        await observe_explicit_cancellation(task, timeout=5)

        assert len(offline_openai.requests) == 1
        assert "".join(c.content for c in received) == "first"
        assert body.close_count >= 1
    finally:
        await reclaim_own_task(task, gate=gate)


# ---------------------------------------------------------------------------
# 10. Existing chunk guard: default 120s, env override, real stall error
# ---------------------------------------------------------------------------


def test_stream_chunk_timeout_default_is_120_seconds(offline_openai):
    model = offline_openai.model()
    assert model.stream_chunk_timeout == 120.0


def test_stream_chunk_timeout_env_override(offline_openai, monkeypatch):
    monkeypatch.setenv("LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S", "7.5")
    model = offline_openai.model()
    assert model.stream_chunk_timeout == 7.5


async def test_streaming_stall_raises_chunk_timeout_and_closes_response(offline_openai):
    """A dedicated stall case uses the existing constructor knob and restores env.

    This does not wait 120s and does not claim to be an HTTP timeout -- it is
    the adapter's own inter-chunk guard. The body close recorded here is the
    only close evidence claimed in this file; it is observed inside the test,
    before the fixture closes the client.
    """
    body = offline_openai.sse_body(
        [sse_event(stream_chunk("partial"))], terminator=None, stall=True
    )
    offline_openai.respond(lambda request: offline_openai.sse_response(body))
    model = offline_openai.model(stream_chunk_timeout=0.05)

    received = []
    with pytest.raises(StreamChunkTimeoutError) as excinfo:
        async for chunk in model.astream("ping"):
            received.append(chunk)

    assert len(offline_openai.requests) == 1, "the guard must not replay"
    assert "".join(c.content for c in received) == "partial"
    assert excinfo.value.chunks_received == 1
    assert excinfo.value.timeout_s == 0.05
    # Observed here, before the fixture actively closes the client.
    assert body.close_count >= 1


# ---------------------------------------------------------------------------
# 11. Transport EOF without finish metadata / without [DONE] (characterisation)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "terminator,label",
    [
        pytest.param(DONE_EVENT, "done-terminator", id="done-terminator"),
        pytest.param(None, "transport-eof", id="transport-eof"),
    ],
)
async def test_stream_without_finish_metadata_still_yields_text(
    offline_openai, terminator, label
):
    """Characterisation only: content survives a stream that sends neither a
    finish_reason nor (in the EOF variant) a ``[DONE]`` marker.

    This says nothing about provider-side completion or usage integrity; those
    remain out of scope for D4a-1.
    """
    offline_openai.respond(
        lambda request: offline_openai.sse_response(
            offline_openai.sse_body(
                [sse_event(stream_chunk("text"))], terminator=terminator
            )
        )
    )
    model = offline_openai.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_openai.requests) == 1
    assert "".join(c.content for c in chunks) == "text"


# ---------------------------------------------------------------------------
# 12. Pure-asyncio mechanism checks for the cancellation helpers.
#
# IMPORTANT: these two cases are **not** SDK evidence. They exercise the
# test-owned `observe_explicit_cancellation` / `reclaim_own_task` helpers with
# plain asyncio primitives only -- no SDK, no adapter, no transport, no socket.
# They exist so the bounded observation cannot silently regress into "wait
# forever" or "swallow the timeout", and they do not replace or weaken any of
# the twelve real SDK cancellation cases above.
# ---------------------------------------------------------------------------


async def test_meta_explicit_cancel_keeps_exact_cancelled_error_and_state():
    """A normal explicit cancel keeps the exact CancelledError + cancelled state."""
    started = asyncio.Event()
    parked_gate = asyncio.Event()

    async def victim():
        started.set()
        await parked_gate.wait()

    task = asyncio.create_task(victim())
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        exc = await observe_explicit_cancellation(task, timeout=5)
        assert type(exc) is asyncio.CancelledError
        assert task.cancelled() is True
    finally:
        await reclaim_own_task(task, gate=parked_gate, timeout=5)


async def test_meta_swallowed_cancel_times_out_without_second_cancel():
    """A task that swallows the first cancel: bounded TimeoutError, no 2nd cancel.

    The task catches ``CancelledError`` once and then parks on a gate that stays
    unset, which is exactly the failure shape the bounded observation exists to
    catch. The local deadline must raise ``TimeoutError`` rather than count as
    the expected cancellation, and ``shield`` must not have cancelled the target
    a second time -- proven with ``Task.cancelling()``, which stays at 1.
    """
    started = asyncio.Event()
    swallowed = asyncio.Event()
    parked_gate = asyncio.Event()

    async def stubborn():
        started.set()
        try:
            await parked_gate.wait()
        except asyncio.CancelledError:
            swallowed.set()  # deliberately ignore the first cancellation
            await parked_gate.wait()  # park again; this task never completes

    task = asyncio.create_task(stubborn())
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        await asyncio.wait_for(swallowed.wait(), timeout=5)
        assert task.cancelling() == 1, "exactly one cancellation was requested"

        with pytest.raises(asyncio.TimeoutError):
            # Short local deadline for the meta case only; the twelve real SDK
            # cases keep their own 5 second budget.
            await observe_explicit_cancellation(task, timeout=0.2)

        assert not task.done(), "the stubborn task must still be pending"
        assert task.cancelling() == 1, (
            "the bounded observation must not cancel the target a second time"
        )
    finally:
        # Releasing the gate lets the pending task finish, so the bounded
        # reclaim can complete instead of hanging the whole pytest process.
        await reclaim_own_task(task, gate=parked_gate, timeout=5)

    assert task.done(), "the reclaimed task must have settled"


# ===========================================================================
# D5A-1 R2: normal completion/usage projection + streamed usage position
#
# Appended at EOF only. Every byte above this marker is the D4 baseline and is
# unchanged. Each fixed expectation was derived from the locked sources below,
# not from an observed run.
#
#   langchain_core/language_models/chat_models.py:935-950   terminal sentinel
#   langchain_core/language_models/chat_models.py:1797,2144-2151
#                                                      llm_output vs combine
#   langchain_openai/chat_models/base.py:1733-1741,1776-1796
#                                                      _create_chat_result
#   langchain_openai/chat_models/base.py:3996-4043       _create_usage_metadata
#   langchain_openai/chat_models/base.py:1333-1391      chunk conversion
#   openai/types/chat/chat_completion.py:87              system_fingerprint=None
#   openai/types/completion_usage.py:44-60               *_details: Optional
#   langchain_core/messages/ai.py:676-715,721-740        chunk add sums usage
# ===========================================================================


def d5a_add_chunks(chunks: list) -> AIMessageChunk:
    """LangChain's own public chunk addition, applied left to right.

    ``add_ai_message_chunks`` sums ``usage_metadata`` and merges
    ``response_metadata``. Using the operator keeps the aggregated object the
    SDK itself would build, instead of a hand-rolled counter.
    """
    total = chunks[0]
    for chunk in chunks[1:]:
        total = total + chunk
    return total


def d5a_assert_core_sentinel(chunk: AIMessageChunk) -> None:
    """Pin the langchain-core terminal sentinel exactly.

    ``chat_models.py:935-950`` appends this chunk itself, after the adapter's
    stream ends, whenever the last adapter chunk carried no ``chunk_position``.
    It is not a provider wire frame: empty content, no usage, no metadata, and
    the only chunk marked ``last``.
    """
    assert chunk.content == ""
    assert chunk.usage_metadata is None
    assert chunk.response_metadata == {}
    assert chunk.chunk_position == "last"


async def test_d5a_baseline_ainvoke(offline_openai):
    """One normal non-stream completion pins the whole usage projection."""
    offline_openai.respond(json_response(completion_body("pong")))
    model = offline_openai.model()

    result = await model.ainvoke("ping")

    assert len(offline_openai.requests) == 1
    assert request_json(offline_openai.requests[0])["messages"] == [
        {"role": "user", "content": "ping"}
    ]

    assert result.content == "pong"
    # _create_usage_metadata renames the provider keys and drops every detail
    # key whose source value is None. This wire carries no *_tokens_details,
    # so the projection holds two *empty* detail dicts, not absent ones.
    assert result.usage_metadata == {
        "input_tokens": 3,
        "output_tokens": 1,
        "total_tokens": 4,
        "input_token_details": {},
        "output_token_details": {},
    }
    assert result.response_metadata["finish_reason"] == "stop"
    # The standardising projection above is a *different* path from the raw
    # publish below: _create_chat_result puts the SDK's own model_dump straight
    # into llm_output, so the None detail keys survive here. They are nulls the
    # SDK filled in, not nulls the wire sent.
    assert result.response_metadata["token_usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 1,
        "total_tokens": 4,
        "prompt_tokens_details": None,
        "completion_tokens_details": None,
    }
    # model_dump() is not exclude_unset, so ChatCompletion.system_fingerprint
    # is always present -- as None here, and the key is not dropped.
    assert "system_fingerprint" in result.response_metadata
    assert result.response_metadata["system_fingerprint"] is None
    assert result.response_metadata["id"] == "chatcmpl-offline"
    assert result.response_metadata["model_provider"] == "openai"
    assert result.response_metadata["model_name"] == REPRESENTATIVE_MODEL
    # The choice carried logprobs: None, and the adapter copies it verbatim.
    assert result.response_metadata["logprobs"] is None
    # service_tier is only added when truthy, and the SDK default is None.
    assert "service_tier" not in result.response_metadata


async def test_d5a_baseline_astream(offline_openai):
    """Streamed usage lands on one chunk, and its position is pinned."""
    offline_openai.respond(
        lambda request: offline_openai.sse_response(
            offline_openai.sse_body(
                [
                    sse_event(stream_chunk("po")),
                    sse_event(stream_chunk("ng")),
                    sse_event(stream_chunk(finish_reason="stop")),
                    sse_event(
                        stream_chunk(
                            choices=[],
                            usage={
                                "prompt_tokens": 3,
                                "completion_tokens": 2,
                                "total_tokens": 5,
                            },
                        )
                    ),
                ]
            )
        )
    )
    model = offline_openai.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_openai.requests) == 1
    body = request_json(offline_openai.requests[0])
    assert body["messages"] == [{"role": "user", "content": "ping"}]

    # Four adapter frames survive (two text deltas, the finish frame and the
    # usage-only frame; [DONE] terminates the SDK stream and yields nothing),
    # plus the langchain-core sentinel the adapter never marks itself.
    assert len(chunks) == 5
    assert [c.content for c in chunks] == ["po", "ng", "", "", ""]
    # usage_metadata is absent on every chunk except the usage-only frame,
    # which sits at index 3 -- the terminal *adapter* frame, not the last one.
    assert [c.usage_metadata for c in chunks] == [
        None,
        None,
        None,
        {
            "input_tokens": 3,
            "output_tokens": 2,
            "total_tokens": 5,
            "input_token_details": {},
            "output_token_details": {},
        },
        None,
    ]
    # The finish frame is still index 2: the sentinel is appended after it.
    assert chunks[2].response_metadata["finish_reason"] == "stop"
    assert chunks[2].response_metadata["model_name"] == REPRESENTATIVE_MODEL
    assert chunks[2].response_metadata["model_provider"] == "openai"
    assert chunks[2].chunk_position is None
    # The usage-only frame takes the len(choices)==0 branch, which builds a
    # fresh chunk and sets no response_metadata at all.
    assert chunks[3].response_metadata == {}
    assert chunks[3].chunk_position is None
    d5a_assert_core_sentinel(chunks[4])

    aggregated = d5a_add_chunks(chunks)
    assert aggregated.content == "pong"
    assert aggregated.usage_metadata == {
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
        "input_token_details": {},
        "output_token_details": {},
    }
    assert aggregated.response_metadata["finish_reason"] == "stop"
    assert aggregated.response_metadata["model_name"] == REPRESENTATIVE_MODEL


async def test_d5a_usage_lifecycle_astream(offline_openai):
    """Exactly one usage report, and it is the terminal adapter frame.

    ``include_usage`` lets OpenAI emit a single usage frame per call, so a
    single request cannot separate cumulative from incremental reporting. That
    is a property of the wire protocol, not of this fixture: the test pins the
    real single-terminal shape instead of inventing extra usage frames.
    """
    offline_openai.respond(
        lambda request: offline_openai.sse_response(
            offline_openai.sse_body(
                [
                    sse_event(stream_chunk("po")),
                    sse_event(stream_chunk("ng")),
                    sse_event(
                        stream_chunk(
                            choices=[],
                            usage={
                                "prompt_tokens": 5,
                                "completion_tokens": 2,
                                "total_tokens": 7,
                            },
                        )
                    ),
                ]
            )
        )
    )
    model = offline_openai.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_openai.requests) == 1
    assert request_json(offline_openai.requests[0])["messages"] == [
        {"role": "user", "content": "ping"}
    ]

    assert len(chunks) == 4
    assert [c.content for c in chunks] == ["po", "ng", "", ""]
    assert [c.usage_metadata for c in chunks] == [
        None,
        None,
        {
            "input_tokens": 5,
            "output_tokens": 2,
            "total_tokens": 7,
            "input_token_details": {},
            "output_token_details": {},
        },
        None,
    ]
    # Usage appears on the terminal *adapter* frame only, so the no-usage
    # range is exactly the two content frames -- not chunks[:-1], which now
    # includes the sentinel rather than the usage frame.
    assert chunks[0].usage_metadata is None
    assert chunks[1].usage_metadata is None
    assert chunks[2].usage_metadata is not None
    assert chunks[2].chunk_position is None
    d5a_assert_core_sentinel(chunks[3])

    aggregated = d5a_add_chunks(chunks)
    assert aggregated.content == "pong"
    assert aggregated.usage_metadata == {
        "input_tokens": 5,
        "output_tokens": 2,
        "total_tokens": 7,
        "input_token_details": {},
        "output_token_details": {},
    }


# ===========================================================================
# D5A-2: completion boundaries and missing usage (7 nodes)
#
# Appended at EOF only. Every byte above this marker is the D4 baseline plus
# the accepted D5A-1 section, and is unchanged. Each fixed expectation was
# derived from the locked sources below, not from an observed run:
#
#   openai/_client.py:655 + _base_client.py:673-676  strict validation is OFF
#   openai/_models.py:663-675,706                  construct() skips Literals
#   openai/types/chat/chat_completion.py:25        finish_reason Literal
#   langchain_openai/chat_models/base.py:1374,1390 falsy finish_reason
#   langchain_openai/chat_models/base.py:1778-1796 raw usage dump in llm_output
#   langchain_core/language_models/chat_models.py:935-950  core sentinel
#   langchain_core/messages/ai.py:676-681           no usage -> None, not zero
#   openai/_streaming.py:172-174                   EOF without [DONE] is fine
# ===========================================================================


async def test_d5a_completion_missing_astream(offline_openai):
    """No finish_reason key at all, while body text and usage stay valid.

    The usage-only frame is independent of the finish reason, so usage still
    arrives; only the completion signal is absent.
    """
    offline_openai.respond(
        lambda request: offline_openai.sse_response(
            offline_openai.sse_body(
                [
                    sse_event(
                        stream_chunk(
                            choices=[
                                {"index": 0, "delta": {"content": "po"}, "logprobs": None}
                            ]
                        )
                    ),
                    sse_event(
                        stream_chunk(
                            choices=[
                                {"index": 0, "delta": {"content": "ng"}, "logprobs": None}
                            ]
                        )
                    ),
                    sse_event(
                        stream_chunk(
                            choices=[],
                            usage={
                                "prompt_tokens": 3,
                                "completion_tokens": 2,
                                "total_tokens": 5,
                            },
                        )
                    ),
                ]
            )
        )
    )
    model = offline_openai.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_openai.requests) == 1
    assert request_json(offline_openai.requests[0])["messages"] == [
        {"role": "user", "content": "ping"}
    ]

    assert len(chunks) == 4
    assert [c.content for c in chunks] == ["po", "ng", "", ""]
    assert [c.usage_metadata for c in chunks] == [
        None,
        None,
        {
            "input_tokens": 3,
            "output_tokens": 2,
            "total_tokens": 5,
            "input_token_details": {},
            "output_token_details": {},
        },
        None,
    ]
    assert chunks[0].response_metadata == {"model_provider": "openai"}
    assert chunks[2].response_metadata == {}
    assert chunks[2].chunk_position is None
    # No chunk reports a completion reason: the key was absent, and
    # `if finish_reason := choice.get("finish_reason")` is false for None too.
    assert all("finish_reason" not in c.response_metadata for c in chunks)
    assert all("model_name" not in c.response_metadata for c in chunks)
    d5a_assert_core_sentinel(chunks[3])

    aggregated = d5a_add_chunks(chunks)
    assert aggregated.content == "pong"
    assert aggregated.usage_metadata == {
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
        "input_token_details": {},
        "output_token_details": {},
    }
    assert aggregated.response_metadata == {"model_provider": "openai"}
    assert "finish_reason" not in aggregated.response_metadata


async def test_d5a_completion_null_astream(offline_openai):
    """An explicit JSON null finish_reason, kept separate from the missing key.

    The SDK materialises an absent key and an explicit null to the same value
    before the adapter reads it, so this projection is byte-for-byte the same
    as test_d5a_completion_missing_astream. That collapse is the point being
    pinned; the wire really does differ.
    """
    offline_openai.respond(
        lambda request: offline_openai.sse_response(
            offline_openai.sse_body(
                [
                    sse_event(stream_chunk("po")),
                    sse_event(stream_chunk("ng")),
                    sse_event(
                        stream_chunk(
                            choices=[],
                            usage={
                                "prompt_tokens": 3,
                                "completion_tokens": 2,
                                "total_tokens": 5,
                            },
                        )
                    ),
                ]
            )
        )
    )
    model = offline_openai.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_openai.requests) == 1
    assert request_json(offline_openai.requests[0])["messages"] == [
        {"role": "user", "content": "ping"}
    ]

    # The two wires genuinely differ in that one key: stream_chunk(content)
    # emits an explicit "finish_reason": null, while an explicit choices list
    # with no finish_reason omits the key entirely.
    with_null = json.loads(stream_chunk("po"))["choices"][0]
    without_key = json.loads(
        stream_chunk(
            choices=[{"index": 0, "delta": {"content": "po"}, "logprobs": None}]
        )
    )["choices"][0]
    assert with_null["finish_reason"] is None
    assert "finish_reason" not in without_key

    assert len(chunks) == 4
    assert [c.content for c in chunks] == ["po", "ng", "", ""]
    assert [c.usage_metadata for c in chunks] == [
        None,
        None,
        {
            "input_tokens": 3,
            "output_tokens": 2,
            "total_tokens": 5,
            "input_token_details": {},
            "output_token_details": {},
        },
        None,
    ]
    assert chunks[0].response_metadata == {"model_provider": "openai"}
    assert chunks[2].response_metadata == {}
    assert all("finish_reason" not in c.response_metadata for c in chunks)
    d5a_assert_core_sentinel(chunks[3])

    aggregated = d5a_add_chunks(chunks)
    assert aggregated.content == "pong"
    assert aggregated.usage_metadata == {
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
        "input_token_details": {},
        "output_token_details": {},
    }
    assert aggregated.response_metadata == {"model_provider": "openai"}


async def test_d5a_usage_missing_astream(offline_openai):
    """Completion signal present, usage simply never sent.

    The request still asks for usage, so this is a wire that stopped honouring
    include_usage rather than a request that never asked.
    """
    offline_openai.respond(
        lambda request: offline_openai.sse_response(
            offline_openai.sse_body(
                [
                    sse_event(stream_chunk("po")),
                    sse_event(stream_chunk("ng")),
                    sse_event(stream_chunk(finish_reason="stop")),
                ]
            )
        )
    )
    model = offline_openai.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_openai.requests) == 1
    body = request_json(offline_openai.requests[0])
    assert body["messages"] == [{"role": "user", "content": "ping"}]
    assert body["stream_options"] == {"include_usage": True}

    assert len(chunks) == 4
    assert [c.content for c in chunks] == ["po", "ng", "", ""]
    # No chunk carries usage, and the aggregate is None -- not a zero dict.
    assert [c.usage_metadata for c in chunks] == [None, None, None, None]
    assert chunks[2].response_metadata["finish_reason"] == "stop"
    assert chunks[2].response_metadata["model_name"] == REPRESENTATIVE_MODEL
    assert chunks[2].chunk_position is None
    d5a_assert_core_sentinel(chunks[3])

    aggregated = d5a_add_chunks(chunks)
    assert aggregated.content == "pong"
    assert aggregated.usage_metadata is None
    assert aggregated.response_metadata["finish_reason"] == "stop"


async def test_d5a_truncated_ainvoke(offline_openai):
    """The protocol's own truncation reason, "length"."""
    offline_openai.respond(json_response(completion_body("par", finish_reason="length")))
    model = offline_openai.model()

    result = await model.ainvoke("ping")

    assert len(offline_openai.requests) == 1
    assert request_json(offline_openai.requests[0])["messages"] == [
        {"role": "user", "content": "ping"}
    ]

    assert result.content == "par"
    assert result.usage_metadata == {
        "input_tokens": 3,
        "output_tokens": 1,
        "total_tokens": 4,
        "input_token_details": {},
        "output_token_details": {},
    }
    # "length" is a real Choice.finish_reason literal; the adapter copies the
    # value through unchanged and adds no truncation flag of its own.
    assert result.response_metadata["finish_reason"] == "length"
    assert result.response_metadata["token_usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 1,
        "total_tokens": 4,
        "prompt_tokens_details": None,
        "completion_tokens_details": None,
    }
    assert result.response_metadata["id"] == "chatcmpl-offline"
    assert result.response_metadata["model_provider"] == "openai"
    assert result.response_metadata["model_name"] == REPRESENTATIVE_MODEL
    assert result.response_metadata["system_fingerprint"] is None
    assert result.response_metadata["logprobs"] is None


async def test_d5a_truncated_astream(offline_openai):
    """The same "length" reason through the streaming entry."""
    offline_openai.respond(
        lambda request: offline_openai.sse_response(
            offline_openai.sse_body(
                [
                    sse_event(stream_chunk("par")),
                    sse_event(stream_chunk(finish_reason="length")),
                ]
            )
        )
    )
    model = offline_openai.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_openai.requests) == 1
    assert request_json(offline_openai.requests[0])["messages"] == [
        {"role": "user", "content": "ping"}
    ]

    assert len(chunks) == 3
    assert [c.content for c in chunks] == ["par", "", ""]
    assert [c.usage_metadata for c in chunks] == [None, None, None]
    assert chunks[1].response_metadata["finish_reason"] == "length"
    assert chunks[1].response_metadata["model_name"] == REPRESENTATIVE_MODEL
    assert chunks[1].response_metadata["model_provider"] == "openai"
    assert chunks[1].chunk_position is None
    d5a_assert_core_sentinel(chunks[2])

    aggregated = d5a_add_chunks(chunks)
    assert aggregated.content == "par"
    assert aggregated.usage_metadata is None
    assert aggregated.response_metadata["finish_reason"] == "length"


async def test_d5a_unknown_ainvoke(offline_openai):
    """A synthetic unknown reason is accepted, not rejected.

    The SDK defaults to lenient construction, which does not validate Literal
    fields, so "d5a_unknown" survives to response_metadata verbatim. This is
    an observation about the locked SDK, not a product decision to tolerate
    unknown reasons.
    """
    offline_openai.respond(
        json_response(completion_body("par", finish_reason="d5a_unknown"))
    )
    model = offline_openai.model()

    result = await model.ainvoke("ping")

    assert len(offline_openai.requests) == 1
    assert request_json(offline_openai.requests[0])["messages"] == [
        {"role": "user", "content": "ping"}
    ]

    assert result.content == "par"
    assert result.usage_metadata == {
        "input_tokens": 3,
        "output_tokens": 1,
        "total_tokens": 4,
        "input_token_details": {},
        "output_token_details": {},
    }
    assert result.response_metadata["finish_reason"] == "d5a_unknown"
    # Everything else is identical to the "length" case.
    assert result.response_metadata["token_usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 1,
        "total_tokens": 4,
        "prompt_tokens_details": None,
        "completion_tokens_details": None,
    }
    assert result.response_metadata["id"] == "chatcmpl-offline"
    assert result.response_metadata["model_provider"] == "openai"
    assert result.response_metadata["model_name"] == REPRESENTATIVE_MODEL


async def test_d5a_terminal_frame_missing_astream(offline_openai):
    """A complete prefix of frames with no [DONE] terminator.

    The OpenAI stream only breaks on a literal [DONE] sentinel, so a raw EOF
    ends the stream without error. The resulting projection is the same shape
    as a terminated stream; that is exactly why this test cannot claim the SDK
    detected the missing terminator.
    """
    offline_openai.respond(
        lambda request: offline_openai.sse_response(
            offline_openai.sse_body(
                [
                    sse_event(stream_chunk("po")),
                    sse_event(stream_chunk("ng")),
                    sse_event(stream_chunk(finish_reason="stop")),
                    sse_event(
                        stream_chunk(
                            choices=[],
                            usage={
                                "prompt_tokens": 3,
                                "completion_tokens": 2,
                                "total_tokens": 5,
                            },
                        )
                    ),
                ],
                terminator=None,
            )
        )
    )
    model = offline_openai.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_openai.requests) == 1
    assert request_json(offline_openai.requests[0])["messages"] == [
        {"role": "user", "content": "ping"}
    ]

    assert len(chunks) == 5
    assert [c.content for c in chunks] == ["po", "ng", "", "", ""]
    assert [c.usage_metadata for c in chunks] == [
        None,
        None,
        None,
        {
            "input_tokens": 3,
            "output_tokens": 2,
            "total_tokens": 5,
            "input_token_details": {},
            "output_token_details": {},
        },
        None,
    ]
    assert chunks[2].response_metadata["finish_reason"] == "stop"
    assert chunks[3].response_metadata == {}
    # The core sentinel is still appended: the adapter never marks itself last.
    d5a_assert_core_sentinel(chunks[4])

    aggregated = d5a_add_chunks(chunks)
    assert aggregated.content == "pong"
    assert aggregated.usage_metadata == {
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
        "input_token_details": {},
        "output_token_details": {},
    }
    assert aggregated.response_metadata["finish_reason"] == "stop"
