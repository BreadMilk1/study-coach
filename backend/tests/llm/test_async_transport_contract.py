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
    offline_env,
    offline_openai,
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
    # Wait for the body to actually start being read; no sleep-based guessing.
    await asyncio.wait_for(body.started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(offline_openai.requests) == 1
    assert body.close_count >= 1, "response created before cancellation must close"


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
    await asyncio.wait_for(body.started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(offline_openai.requests) == 1
    assert received == []
    assert body.close_count >= 1


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
    await asyncio.wait_for(first_chunk_seen.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(offline_openai.requests) == 1
    assert "".join(c.content for c in received) == "first"
    assert body.close_count >= 1


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
