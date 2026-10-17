"""D4a-4: offline evidence for the real Gemini (Google GenAI) async transport.

Scope is one provider only. The chain that runs for real is ``get_chat_model``
-> ``ChatGoogleGenerativeAI`` -> a real ``google.genai.Client`` -> the real
``aio.models`` resource -> the SDK's own SSE frame parser and pydantic models
-> the adapter's message conversion. Only the socket is replaced, through the
adapter's existing ``client_args`` transport knob.

Nothing from the other providers' shapes is reused here: Gemini speaks
``candidates`` / ``content.parts`` / ``role: "model"`` with CamelCase wire
fields, and streams ``data: {json}`` frames terminated by a blank line.

The retry patch is deliberately narrow. Every async request builds a *new*
``tenacity.AsyncRetrying``, so patching a cached instance's sleep would never
take effect. This wraps only the constructor and injects an async no-op sleep,
passing every original ``stop`` / ``retry`` / ``wait`` / ``reraise`` argument
through untouched -- the retry predicate itself is never altered, only the
waiting is removed. The original objects are recorded so the evidence is that
the retry *policy* is intact while the *sleep* is gone.

``bind_tools`` is exercised schema-only. A ``function_declarations`` tool is
AFC-incompatible in the Google SDK, so it takes the single-request fallback; no
Python callable is passed, no tool runs, and nothing here claims the SDK
executed a tool or that AFC is enabled.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from typing import Any, Callable

import httpx
import pytest
import tenacity
from google.genai import _api_client as genai_api_client
from google.genai import errors as genai_errors
from langchain_core.messages import AIMessage, AIMessageChunk

# Not re-exported from the package __init__; it lives in the chat module.
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

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

SYNTHETIC_GEMINI_KEY = "gemini-offline-d4a4"
REPRESENTATIVE_MODEL = "gemini-2.5-flash"
DEVELOPER_API_BASE = "https://generativelanguage.googleapis.com/"

# A minimal declaration with no local implementation behind it. Nothing is ever
# executed; this only checks what the real adapter serialises onto the wire.
SCHEMA_ONLY_TOOL = {
    "name": "get_current_temperature",
    "description": "Return the current temperature for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}

GOOGLE_STEERING_ENV_VARS = (
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_GENAI_API_KEY",
    "GOOGLE_GENAI_USE_VERTEXAI",
    "GOOGLE_GENAI_USE_ENTERPRISE",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_GEMINI_BASE_URL",
    "GOOGLE_VERTEX_BASE_URL",
    "GOOGLE_CLOUD_QUOTA_PROJECT",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OLLAMA_HOST",
    "OLLAMA_API_KEY",
)


# ---------------------------------------------------------------------------
# Gemini wire builders
# ---------------------------------------------------------------------------


def candidate(text: str, *, finish_reason: str | None = "STOP") -> dict:
    block: dict = {"content": {"parts": [{"text": text}], "role": "model"}, "index": 0}
    if finish_reason is not None:
        block["finishReason"] = finish_reason
    return block


def generate_response(text: str = "pong", *, finish_reason: str | None = "STOP") -> dict:
    return {
        "candidates": [candidate(text, finish_reason=finish_reason)],
        "usageMetadata": {
            "promptTokenCount": 3,
            "candidatesTokenCount": 2,
            "totalTokenCount": 5,
        },
        "modelVersion": REPRESENTATIVE_MODEL,
    }


def sse_frame(payload: dict) -> bytes:
    """One Gemini streaming frame: ``data: <json>`` then a blank line."""
    return (f"data: {json.dumps(payload)}\n\n").encode()


def stream_frames(*texts: str, final: bool = True) -> list[bytes]:
    frames = [sse_frame({"candidates": [candidate(t, finish_reason=None)]}) for t in texts]
    if final:
        frames.append(sse_frame({"candidates": [candidate("", finish_reason="STOP")]}))
    return frames


def google_error_body(code: int, status: str, message: str = "nope") -> dict:
    """A valid Google error body for the given HTTP status."""
    return {"error": {"code": code, "message": message, "status": status}}


_HTTP_STATUS_BODIES = {
    401: google_error_body(401, "UNAUTHENTICATED"),
    429: google_error_body(429, "RESOURCE_EXHAUSTED"),
    503: google_error_body(503, "UNAVAILABLE"),
}


def json_response(payload: dict, *, status: int = 200, headers: dict | None = None):
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload, headers=headers or {})

    return responder


def request_json(request: httpx.Request) -> dict:
    return json.loads(request.content)


def request_timeout_fields(request: httpx.Request) -> dict:
    """The timeout the SDK actually put on this one request.

    This is the per-request extension, not the HTTPX client's default config;
    the two can disagree, and only this reflects what the request carried.
    """
    ext = request.extensions.get("timeout")
    if not isinstance(ext, dict):
        return {}
    return {key: ext.get(key) for key in ("connect", "read", "write", "pool")}


def assert_bound_declaration(body: dict) -> dict:
    """Verify the full serialised schema of the one declaration bound here.

    Both bound success cases reach this through ``assert_real_request``, so a
    name-only check cannot stand in for a parameter check on just one of them.
    """
    declarations = body["tools"][0]["functionDeclarations"]
    assert len(declarations) == 1
    declaration = declarations[0]
    assert declaration["name"] == SCHEMA_ONLY_TOOL["name"]
    assert declaration["description"] == SCHEMA_ONLY_TOOL["description"]
    parameters = declaration["parameters"]
    # The SDK normalises JSON-schema types into its own CamelCase enums; the
    # rest of the declaration is preserved as written.
    assert parameters["type"] == "OBJECT"
    assert parameters["properties"]["city"]["type"] == "STRING"
    assert parameters["required"] == ["city"]
    return declaration


def assert_real_request(
    request: httpx.Request,
    *,
    streaming: bool,
    expect_tools: bool = False,
) -> dict:
    """Pin the full wire shape of a real Gemini request.

    MockTransport would happily answer any path, so method, scheme, host, the
    complete endpoint path and the exact query are all asserted rather than
    substring-matched. When the request is bound, the declaration schema is
    verified here too, so every bound case carries the same depth of evidence.
    """
    assert request.method == "POST"
    assert request.url.scheme == "https"
    assert request.url.host == "generativelanguage.googleapis.com"

    endpoint = ":streamGenerateContent" if streaming else ":generateContent"
    other_endpoint = ":generateContent" if streaming else ":streamGenerateContent"
    assert request.url.path == f"/v1beta/models/{REPRESENTATIVE_MODEL}{endpoint}"
    assert other_endpoint not in request.url.path, "must not hit the other endpoint"

    params = dict(request.url.params)
    if streaming:
        assert params.get("alt") == "sse"
    else:
        assert "alt" not in params, "non-stream must not request SSE framing"

    assert request.headers["x-goog-api-key"] == SYNTHETIC_GEMINI_KEY

    body = request_json(request)
    assert body["contents"] == [{"role": "user", "parts": [{"text": "ping"}]}]
    if expect_tools:
        assert "tools" in body, "the bound request must carry its declarations"
        assert_bound_declaration(body)
    else:
        assert "tools" not in body, "the plain request must not carry declarations"
    return body


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class OfflineGemini:
    """MockTransport-backed stand-in for the Google socket."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.streams: list[ScriptedByteStream] = []
        self.models: list[Any] = []
        self.retry_observations: list[dict[str, Any]] = []
        self._responder: Callable[[httpx.Request], httpx.Response] | None = None

    def respond(self, responder) -> "OfflineGemini":
        self._responder = responder
        return self

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self._responder is None:
            raise AssertionError("offline transport has no responder installed")
        return self._responder(request)

    def body(self, steps, *, name: str = "sse") -> ScriptedByteStream:
        stream = ScriptedByteStream(steps, name=name)
        self.streams.append(stream)
        return stream

    def frame_body(self, frames, *, name: str = "sse") -> ScriptedByteStream:
        return self.body([send(f) for f in frames], name=name)

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

    def model(self, **kwargs):
        """Build through the product factory using its existing transport knob.

        ``client_args`` is what the adapter forwards into the Google
        ``HttpOptions`` for both the sync and async HTTPX clients, and a custom
        transport makes the SDK stay on HTTPX instead of aiohttp.
        """
        from app.llm.provider import LLMConfig, get_chat_model

        config = LLMConfig(
            provider="gemini",
            model=REPRESENTATIVE_MODEL,
            api_key=SYNTHETIC_GEMINI_KEY,
        )
        built = get_chat_model(
            config,
            client_args={
                "transport": httpx.MockTransport(self.handle),
                "trust_env": False,
            },
            **kwargs,
        )
        self.models.append(built)
        return built

    async def aclose(self) -> None:
        """Close both ends of every real SDK client, then drop the list.

        ``client.close()`` and ``client.aio.aclose()`` are separate entry
        points, so doing one and skipping the other leaves a client open.
        Models are kept until every close has been *attempted*, and one failing
        close never prevents the others; errors are collected and re-raised at
        the end rather than swallowed.

        Safe to call again with no models left: the fixture's final teardown
        call after the selfcheck is then a no-op.
        """
        errors: list[BaseException] = []
        for model in self.models:
            client = getattr(model, "client", None)
            if client is None:
                continue
            try:
                client.close()
            except BaseException as exc:  # noqa: BLE001 - collected, re-raised
                errors.append(exc)
            aio = getattr(client, "aio", None)
            if aio is None:
                continue
            try:
                await aio.aclose()
            except BaseException as exc:  # noqa: BLE001 - collected, re-raised
                errors.append(exc)
        self.models.clear()
        if errors:
            raise errors[0]


@pytest.fixture(autouse=True)
def gemini_env(monkeypatch):
    """Clear Google steering and credential variables, restored per test.

    No real ADC, project, location or API version is ever consulted.
    """
    for name in GOOGLE_STEERING_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("STUDY_COACH_TEST_MODE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    yield


@pytest.fixture
async def offline_gemini(monkeypatch) -> AsyncIterator[OfflineGemini]:
    harness = OfflineGemini()

    # Every async request constructs a fresh AsyncRetrying, so only the
    # constructor can be intercepted. All original policy kwargs are passed
    # through untouched; only the wait is neutralised.
    original_construct_retrying = genai_api_client.tenacity.AsyncRetrying

    async def _no_sleep(_seconds):  # noqa: ANN001 - tenacity's sleep signature
        return None

    def construct_retrying(*args, **kwargs):
        harness.retry_observations.append(
            {
                "stop": kwargs.get("stop"),
                "retry": kwargs.get("retry"),
                "wait": kwargs.get("wait"),
                "reraise": kwargs.get("reraise"),
            }
        )
        kwargs["sleep"] = _no_sleep
        return original_construct_retrying(*args, **kwargs)

    monkeypatch.setattr(
        genai_api_client.tenacity, "AsyncRetrying", construct_retrying
    )

    try:
        yield harness
    finally:
        # monkeypatch restores the constructor; close both ends of every real
        # SDK client. Nothing that body-level close_count assertions rely on
        # happens after this point.
        await harness.aclose()


# ---------------------------------------------------------------------------
# Harness self-checks: isolation, default config, source identity
# ---------------------------------------------------------------------------


def test_offline_harness_rejects_dns_and_connect_attempts():
    """The socket deny is real, not a printed claim."""
    import socket

    with pytest.raises(AssertionError, match="real network access"):
        socket.getaddrinfo("generativelanguage.googleapis.com", 443)
    with pytest.raises(AssertionError, match="real network access"):
        socket.create_connection(("generativelanguage.googleapis.com", 443))


def test_offline_harness_pins_developer_api_and_mock_transport(offline_gemini):
    """The factory's Developer API pinning survives, on a mock socket."""
    model = offline_gemini.model()

    assert type(model).__name__ == "ChatGoogleGenerativeAI"
    assert model.model == REPRESENTATIVE_MODEL
    assert model.vertexai is False, "factory pins the Developer API explicitly"
    assert model.client.vertexai is False, "the SDK client agrees"

    assert model.max_retries == 6, "adapter default attempts, first call included"

    api_client = model.client._api_client
    async_httpx = api_client._async_httpx_client
    sync_httpx = api_client._httpx_client
    assert isinstance(async_httpx, httpx.AsyncClient)
    assert isinstance(sync_httpx, httpx.Client)
    for httpx_client in (async_httpx, sync_httpx):
        assert isinstance(httpx_client._transport, httpx.MockTransport)
        assert httpx_client.trust_env is False
        # The SDK builds absolute request URLs itself, so the HTTPX clients
        # carry no base_url of their own; the host is pinned via HttpOptions.
        assert str(httpx_client.base_url) == ""
    assert api_client._http_options.base_url == DEVELOPER_API_BASE


def test_offline_harness_reports_actual_httpx_client_timeout_config(offline_gemini):
    """Client *configuration* only: the HTTPX default this adapter inherits.

    This says nothing about the timeout any single request actually carries --
    see ``test_offline_harness_request_timeout_differs_from_client_default``.
    Unlike OpenAI and Anthropic (where the adapter passes ``timeout=None``),
    this adapter leaves ``HttpOptions.timeout`` unset, so the SDK's own 5s
    HTTPX default shows up on the client object.
    """
    model = offline_gemini.model()

    api_client = model.client._api_client
    async_httpx = api_client._async_httpx_client
    sync_httpx = api_client._httpx_client
    for httpx_client in (async_httpx, sync_httpx):
        assert httpx_client.timeout == httpx.Timeout(timeout=5.0)
        assert httpx_client.timeout.connect == 5.0
        assert httpx_client.timeout.read == 5.0
        assert httpx_client.timeout.write == 5.0
        assert httpx_client.timeout.pool == 5.0


async def test_offline_harness_request_timeout_differs_from_client_default(offline_gemini):
    """The request actually carries no timeout at all.

    The SDK builds its ``HttpRequest`` with ``timeout=None`` and passes that
    explicitly into ``build_request``, which overrides the client's 5s default.
    So the two layers disagree, and this file reports both rather than treating
    the client value as a live budget. It is still a configuration observation,
    not a measurement of real network timing.
    """
    offline_gemini.respond(json_response(generate_response("ok")))
    model = offline_gemini.model()

    await model.ainvoke("ping")

    assert len(offline_gemini.requests) == 1
    assert request_timeout_fields(offline_gemini.requests[0]) == {
        "connect": None,
        "read": None,
        "write": None,
        "pool": None,
    }
    assert model.client._api_client._async_httpx_client.timeout == httpx.Timeout(timeout=5.0)


async def test_offline_harness_retry_policy_is_recorded_not_rewritten(offline_gemini):
    """The constructor wrapper only removes the wait; stop/retry survive."""
    offline_gemini.respond(json_response(generate_response("ok")))
    model = offline_gemini.model()

    result = await model.ainvoke("ping")

    assert result.content == "ok"
    assert offline_gemini.retry_observations, "a request-level retry object was built"
    observed = offline_gemini.retry_observations[-1]
    assert isinstance(observed["stop"], tenacity.stop_after_attempt)
    assert observed["stop"].max_attempt_number == 6
    assert observed["reraise"] is True
    assert observed["retry"] is not None
    assert observed["wait"] is not None, "the wait strategy is passed through, not replaced"
    assert len(offline_gemini.requests) == 1


async def test_cleanup_closes_both_ends_of_each_real_sdk_client(offline_gemini):
    """Explicitly close both ends, verified while strong refs are held.

    The model and both underlying HTTPX clients stay referenced for the whole
    test, so ``is_closed`` can only become True because this cleanup path ran,
    never because GC closed them behind us.
    """
    model = offline_gemini.model()
    api_client = model.client._api_client
    sync_httpx = api_client._httpx_client
    async_httpx = api_client._async_httpx_client

    assert sync_httpx.is_closed is False
    assert async_httpx.is_closed is False
    assert len(offline_gemini.models) == 1

    await offline_gemini.aclose()

    assert sync_httpx.is_closed is True, "the sync client must be closed explicitly"
    assert async_httpx.is_closed is True
    assert offline_gemini.models == []

    # The fixture calls this again during teardown; it must be safe.
    await offline_gemini.aclose()
    assert sync_httpx.is_closed is True
    assert async_httpx.is_closed is True


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
        sys.modules[__name__], "tests/llm/test_async_gemini_transport_contract.py"
    )


# ---------------------------------------------------------------------------
# 1. Plain happy path for both entries
# ---------------------------------------------------------------------------


async def test_ainvoke_sends_real_generate_content_request(offline_gemini):
    offline_gemini.respond(json_response(generate_response("pong")))
    model = offline_gemini.model()

    result = await model.ainvoke("ping")

    assert len(offline_gemini.requests) == 1
    request = offline_gemini.requests[0]
    assert_real_request(request, streaming=False)
    assert request_timeout_fields(request) == {
        "connect": None, "read": None, "write": None, "pool": None,
    }

    assert isinstance(result, AIMessage)
    assert result.content == "pong"


async def test_astream_reads_real_gemini_sse_frames(offline_gemini):
    offline_gemini.respond(
        lambda request: offline_gemini.sse_response(
            offline_gemini.frame_body(stream_frames("po", "ng"))
        )
    )
    model = offline_gemini.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_gemini.requests) == 1
    request = offline_gemini.requests[0]
    assert_real_request(request, streaming=True)
    assert request_timeout_fields(request) == {
        "connect": None, "read": None, "write": None, "pool": None,
    }

    assert all(isinstance(c, AIMessageChunk) for c in chunks)
    assert "".join(c.content for c in chunks if isinstance(c.content, str)) == "pong"


# ---------------------------------------------------------------------------
# 2. Schema tool-bound (AFC-disabled single-request fallback), both entries
# ---------------------------------------------------------------------------


async def test_ainvoke_with_bound_schema_serialises_declarations_only(offline_gemini):
    offline_gemini.respond(json_response(generate_response("sunny")))
    model = offline_gemini.model()
    bound = model.bind_tools([SCHEMA_ONLY_TOOL])

    result = await bound.ainvoke("ping")

    assert len(offline_gemini.requests) == 1, "no extra automatic HTTP request"
    request = offline_gemini.requests[0]
    # assert_real_request also verifies the full declaration schema here.
    assert_real_request(request, streaming=False, expect_tools=True)
    assert request_timeout_fields(request) == {
        "connect": None, "read": None, "write": None, "pool": None,
    }

    assert isinstance(result, AIMessage)
    assert result.content == "sunny"


async def test_astream_with_bound_schema_serialises_declarations_only(offline_gemini):
    offline_gemini.respond(
        lambda request: offline_gemini.sse_response(
            offline_gemini.frame_body(stream_frames("sunny"))
        )
    )
    model = offline_gemini.model()
    bound = model.bind_tools([SCHEMA_ONLY_TOOL])

    chunks = [chunk async for chunk in bound.astream("ping")]

    assert len(offline_gemini.requests) == 1
    request = offline_gemini.requests[0]
    # Same depth of evidence as the bound ainvoke case: assert_real_request
    # verifies the full declaration schema, not just the name.
    assert_real_request(request, streaming=True, expect_tools=True)
    assert request_timeout_fields(request) == {
        "connect": None, "read": None, "write": None, "pool": None,
    }

    assert "".join(c.content for c in chunks if isinstance(c.content, str)) == "sunny"


# ---------------------------------------------------------------------------
# 3. Empty / zero data
# ---------------------------------------------------------------------------


async def test_ainvoke_empty_candidates_returns_empty_message(offline_gemini):
    """candidates=[] is legal JSON; the adapter yields an empty message rather
    than raising, which is deliberately different from a hard failure."""
    offline_gemini.respond(json_response({"candidates": []}))
    model = offline_gemini.model()

    result = await model.ainvoke("ping")

    assert len(offline_gemini.requests) == 1
    assert isinstance(result, AIMessage)
    assert result.content == ""


async def test_astream_zero_frame_eof_raises_no_generation_chunks(offline_gemini):
    """A 200 stream that emits no frame at all is kept separate from an empty
    candidate: here the empty candidate did not raise, this does."""
    offline_gemini.respond(
        lambda request: offline_gemini.sse_response(offline_gemini.frame_body([]))
    )
    model = offline_gemini.model()

    with pytest.raises(ValueError, match="No generation chunks were returned"):
        async for _ in model.astream("ping"):
            pass

    assert len(offline_gemini.requests) == 1


async def test_astream_empty_candidates_frame_raises_index_error(offline_gemini):
    """A *legal* ``data:{"candidates":[]}`` frame is not the zero-frame case.

    Kept distinct from both neighbours: the empty-text cancel frame and the
    zero-frame EOF are different shapes, and this one names an empty candidate
    list on the wire.

    The narrow failure is observed, not invented: the SDK's AFC dispatch
    generator calls ``append_chunk_contents``, which indexes
    ``chunk.candidates[0]`` and raises ``IndexError``. That stack is itself
    evidence that the plain streaming entry does traverse the AFC dispatch
    branch -- what remains unverified is Python tool execution and multi-round
    remote calls, not the control flow.
    """
    offline_gemini.respond(
        lambda request: offline_gemini.sse_response(
            offline_gemini.frame_body([sse_frame({"candidates": []})])
        )
    )
    model = offline_gemini.model()

    with pytest.raises(IndexError) as excinfo:
        async for _ in model.astream("ping"):
            pass

    assert type(excinfo.value) is IndexError
    assert len(offline_gemini.requests) == 1, "the bad frame is not replayed"


# ---------------------------------------------------------------------------
# 4. Bad wire
# ---------------------------------------------------------------------------


async def test_ainvoke_bad_candidate_shape_raises_narrow_error(offline_gemini):
    """Legal fixture first, then exactly one field changed to a bad shape."""
    good = generate_response("pong")
    offline_gemini.respond(json_response(good))
    model = offline_gemini.model()
    ok = await model.ainvoke("ping")
    assert ok.content == "pong"

    bad = generate_response("pong")
    bad["candidates"][0]["content"] = {"parts": {"not": "a list"}, "role": "model"}
    offline_gemini.respond(json_response(bad))

    # Observed narrow failure: the SDK builds its response model leniently and
    # the malformed `parts` then breaks model construction.
    with pytest.raises(AttributeError) as excinfo:
        await model.ainvoke("ping")

    assert type(excinfo.value) is AttributeError
    assert len(offline_gemini.requests) == 2


async def test_astream_invalid_json_frame_raises_unknown_api_response(offline_gemini):
    body = offline_gemini.body([send(b"data: {not-json\n\n")])
    offline_gemini.respond(lambda request: offline_gemini.sse_response(body))
    model = offline_gemini.model()

    with pytest.raises(genai_errors.UnknownApiResponseError) as excinfo:
        async for _ in model.astream("ping"):
            pass

    assert type(excinfo.value) is genai_errors.UnknownApiResponseError
    assert len(offline_gemini.requests) == 1
    assert body.close_count >= 1


# ---------------------------------------------------------------------------
# 5. HTTP status for both plain entries
# ---------------------------------------------------------------------------


_HTTP_STATUS_ATTEMPTS = [
    pytest.param(401, 1, id="401"),
    pytest.param(429, 6, id="429"),
    pytest.param(503, 6, id="503"),
]

# ainvoke awaits a real coroutine inside the adapter's try, so a ClientError is
# converted by `_handle_client_error`.
_AINVOKE_STATUS_EXPECTED = {
    401: ChatGoogleGenerativeAIError,
    429: ChatGoogleGenerativeAIError,
    503: genai_errors.ServerError,
}

# Plain astream does NOT see the wrapped class. `generate_content_stream` is a
# coroutine, but on the AFC-enabled branch it returns an `async_generator`
# without sending anything; the HTTP request only happens when iteration
# begins, at `async for chunk in stream:` -- outside the adapter's
# try/except. The raw SDK error therefore escapes. (The bound/AFC-disabled
# branch awaits eagerly instead, and *is* wrapped -- see
# test_bound_schema_astream_http_status_is_wrapped.)
# Attempts are unchanged either way: the request-level retry still applies.
_ASTREAM_STATUS_EXPECTED = {
    401: genai_errors.ClientError,
    429: genai_errors.ClientError,
    503: genai_errors.ServerError,
}


@pytest.mark.parametrize("status,attempts", _HTTP_STATUS_ATTEMPTS)
async def test_ainvoke_http_status_class_and_attempts(offline_gemini, status, attempts):
    """A ClientError is wrapped by the adapter; a ServerError is not."""
    body = _HTTP_STATUS_BODIES[status]
    expected = _AINVOKE_STATUS_EXPECTED[status]

    def responder(request: httpx.Request) -> httpx.Response:
        # A fresh Response and body per attempt, never a reused one.
        return httpx.Response(status, json=dict(body))

    offline_gemini.respond(responder)
    model = offline_gemini.model()

    with pytest.raises(expected) as excinfo:
        await model.ainvoke("ping")

    assert type(excinfo.value) is expected, "exact class, not merely a subclass"
    assert len(offline_gemini.requests) == attempts


@pytest.mark.parametrize("status,attempts", _HTTP_STATUS_ATTEMPTS)
async def test_astream_http_status_class_and_attempts(offline_gemini, status, attempts):
    """Streaming status errors are *not* wrapped by the adapter.

    This is the observed behaviour of the locked adapter, pinned explicitly so
    the asymmetry against ainvoke cannot be mistaken for a flake.
    """
    body = _HTTP_STATUS_BODIES[status]
    expected = _ASTREAM_STATUS_EXPECTED[status]

    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=dict(body))

    offline_gemini.respond(responder)
    model = offline_gemini.model()

    with pytest.raises(expected) as excinfo:
        async for _ in model.astream("ping"):
            pass

    assert type(excinfo.value) is expected
    assert not isinstance(excinfo.value, ChatGoogleGenerativeAIError), (
        "astream status errors escape the adapter's handler"
    )
    assert len(offline_gemini.requests) == attempts


# ---------------------------------------------------------------------------
# 6. Send-phase failures for both plain entries
# ---------------------------------------------------------------------------


_SEND_FAILURE_MATRIX = [
    pytest.param(httpx.ReadTimeout, "simulated read timeout", id="readtimeout"),
    pytest.param(httpx.ConnectError, "simulated connect failure", id="connecterror"),
]


@pytest.mark.parametrize("exc_cls,message", _SEND_FAILURE_MATRIX)
async def test_ainvoke_send_failure_keeps_original_httpx_error(
    offline_gemini, exc_cls, message
):
    """The SDK predicate retries transport errors; the original class surfaces
    once the budget is exhausted. There is no OpenAI-style APIConnectionError."""
    errors_seen: list[BaseException] = []

    def responder(request: httpx.Request) -> httpx.Response:
        err = exc_cls(message)
        errors_seen.append(err)
        raise err

    offline_gemini.respond(responder)
    model = offline_gemini.model()

    with pytest.raises(exc_cls, match=message) as excinfo:
        await model.ainvoke("ping")

    assert type(excinfo.value) is exc_cls
    assert excinfo.value is errors_seen[-1], "the last injected instance surfaces"
    assert len(errors_seen) == 6, "each attempt builds its own exception"
    assert len(offline_gemini.requests) == 6


@pytest.mark.parametrize("exc_cls,message", _SEND_FAILURE_MATRIX)
async def test_astream_send_failure_keeps_original_httpx_error(
    offline_gemini, exc_cls, message
):
    errors_seen: list[BaseException] = []

    def responder(request: httpx.Request) -> httpx.Response:
        err = exc_cls(message)
        errors_seen.append(err)
        raise err

    offline_gemini.respond(responder)
    model = offline_gemini.model()

    with pytest.raises(exc_cls, match=message) as excinfo:
        async for _ in model.astream("ping"):
            pass

    assert type(excinfo.value) is exc_cls
    assert excinfo.value is errors_seen[-1]
    assert len(offline_gemini.requests) == 6


# ---------------------------------------------------------------------------
# 7. Retryable then success
# ---------------------------------------------------------------------------


async def test_retryable_429_then_success(offline_gemini):
    """One 429 then a 200 succeeds on attempt 2, with no policy invented."""
    calls = {"n": 0}

    def responder(request: httpx.Request) -> httpx.Response:
        index = calls["n"]
        calls["n"] += 1
        if index == 0:
            return httpx.Response(429, json=dict(_HTTP_STATUS_BODIES[429]))
        return offline_gemini.sse_response(
            offline_gemini.frame_body(stream_frames("recovered"))
        )

    offline_gemini.respond(responder)
    model = offline_gemini.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_gemini.requests) == 2
    assert "".join(c.content for c in chunks if isinstance(c.content, str)) == "recovered"


# ---------------------------------------------------------------------------
# 8. astream body-read failures: 2 exception types x before/after first content
# ---------------------------------------------------------------------------

_BODY_FAILURE_MATRIX = [
    pytest.param(httpx.ReadTimeout, "stalled on timeout", id="readtimeout"),
    pytest.param(httpx.ReadError, "stalled on read", id="readerror"),
]


@pytest.mark.parametrize("exc_cls,message", _BODY_FAILURE_MATRIX)
async def test_astream_body_failure_before_first_content(offline_gemini, exc_cls, message):
    injected = exc_cls(message)
    body = offline_gemini.body([fail_with(injected)])
    offline_gemini.respond(lambda request: offline_gemini.sse_response(body))
    model = offline_gemini.model()

    received = []
    with pytest.raises(exc_cls, match=message) as excinfo:
        async for chunk in model.astream("ping"):
            received.append(chunk)

    assert type(excinfo.value) is exc_cls
    assert excinfo.value is injected
    assert len(offline_gemini.requests) == 1, "a body failure cannot be replayed"
    assert received == []
    # Observed here, before the fixture closes the SDK clients.
    assert body.close_count >= 1


@pytest.mark.parametrize("exc_cls,message", _BODY_FAILURE_MATRIX)
async def test_astream_body_failure_after_first_content(offline_gemini, exc_cls, message):
    injected = exc_cls(message)
    body = offline_gemini.body(
        [
            send(sse_frame({"candidates": [candidate("par", finish_reason=None)]})),
            fail_with(injected),
        ]
    )
    offline_gemini.respond(lambda request: offline_gemini.sse_response(body))
    model = offline_gemini.model()

    received = []
    with pytest.raises(exc_cls, match=message) as excinfo:
        async for chunk in model.astream("ping"):
            received.append(chunk)

    assert type(excinfo.value) is exc_cls
    assert excinfo.value is injected
    assert len(offline_gemini.requests) == 1
    assert "".join(c.content for c in received if isinstance(c.content, str)) == "par"
    assert body.close_count >= 1


# ---------------------------------------------------------------------------
# 9. ainvoke non-stream body read (its budget boundary differs)
# ---------------------------------------------------------------------------


async def test_ainvoke_nonstream_body_read_timeout_uses_retry_budget(offline_gemini):
    """A body read failure on the non-stream entry happens inside the HTTPX
    send/retry scope, so it is *not* a single non-replayed request.

    Named separately from the astream body matrix precisely because the
    observable budget differs.
    """
    injected = httpx.ReadTimeout("stalled reading non-stream body")

    def responder(request: httpx.Request) -> httpx.Response:
        body = offline_gemini.body([fail_with(injected)])
        return offline_gemini.sse_response(body, headers={"content-type": "application/json"})

    offline_gemini.respond(responder)
    model = offline_gemini.model()

    with pytest.raises(httpx.ReadTimeout) as excinfo:
        await model.ainvoke("ping")

    assert type(excinfo.value) is httpx.ReadTimeout
    assert len(offline_gemini.requests) == 6, (
        "the non-stream body read sits inside the request-level retry budget"
    )


# ---------------------------------------------------------------------------
# 10. HTTP 200 SSE error frame (distinct from an HTTP 429 status)
# ---------------------------------------------------------------------------


async def test_http_200_sse_error_frame_is_client_error_not_wrapper(offline_gemini):
    """An error frame inside a 200 body is a different boundary from an HTTP
    429 status: the wire's error.code is 429, but the HTTP status is 200."""
    error_frame = sse_frame(
        {
            "error": {
                "code": 429,
                "message": "synthetic",
                "status": "RESOURCE_EXHAUSTED",
            }
        }
    )
    offline_gemini.respond(
        lambda request: offline_gemini.sse_response(
            offline_gemini.frame_body([error_frame])
        )
    )
    model = offline_gemini.model()

    with pytest.raises(genai_errors.ClientError) as excinfo:
        async for _ in model.astream("ping"):
            pass

    assert type(excinfo.value) is genai_errors.ClientError
    assert not isinstance(excinfo.value, ChatGoogleGenerativeAIError), (
        "a mid-stream error frame is not wrapped by the adapter"
    )
    assert excinfo.value.code == 429
    assert len(offline_gemini.requests) == 1, "an SSE error frame must not be retried"


# ---------------------------------------------------------------------------
# 11. Schema tool-bound failure on the same status boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status,attempts", [
    pytest.param(401, 1, id="401"),
    pytest.param(429, 6, id="429"),
])
async def test_bound_schema_astream_http_status_is_wrapped(
    offline_gemini, status, attempts
):
    """A bound (AFC-disabled) stream *is* wrapped -- unlike the plain stream.

    ``generate_content_stream`` is a coroutine. On the AFC-incompatible branch
    it awaits ``_generate_content_stream`` eagerly and only then wraps the
    result in ``base_async_generator``, so the HTTP request happens inside the
    adapter's try/except and the wrapper applies.
    """
    body = _HTTP_STATUS_BODIES[status]
    expected = _AINVOKE_STATUS_EXPECTED[status]

    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=dict(body))

    offline_gemini.respond(responder)
    model = offline_gemini.model()
    bound = model.bind_tools([SCHEMA_ONLY_TOOL])

    with pytest.raises(expected) as excinfo:
        async for _ in bound.astream("ping"):
            pass

    assert type(excinfo.value) is expected
    assert len(offline_gemini.requests) == attempts


@pytest.mark.parametrize("status,attempts", [
    pytest.param(401, 1, id="401"),
    pytest.param(429, 6, id="429"),
])
async def test_bound_schema_ainvoke_http_status_matches_plain_ainvoke(
    offline_gemini, status, attempts
):
    """The non-stream entry keeps the adapter wrapper, bound or not."""
    body = _HTTP_STATUS_BODIES[status]
    expected = _AINVOKE_STATUS_EXPECTED[status]

    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=dict(body))

    offline_gemini.respond(responder)
    model = offline_gemini.model()
    bound = model.bind_tools([SCHEMA_ONLY_TOOL])

    with pytest.raises(expected) as excinfo:
        await bound.ainvoke("ping")

    assert type(excinfo.value) is expected
    assert len(offline_gemini.requests) == attempts


async def test_bound_schema_http_200_error_frame_matches_plain_boundary(offline_gemini):
    error_frame = sse_frame(
        {"error": {"code": 401, "message": "synthetic", "status": "UNAUTHENTICATED"}}
    )
    offline_gemini.respond(
        lambda request: offline_gemini.sse_response(
            offline_gemini.frame_body([error_frame])
        )
    )
    model = offline_gemini.model()
    bound = model.bind_tools([SCHEMA_ONLY_TOOL])

    with pytest.raises(genai_errors.ClientError) as excinfo:
        async for _ in bound.astream("ping"):
            pass

    assert type(excinfo.value) is genai_errors.ClientError
    assert len(offline_gemini.requests) == 1


# ---------------------------------------------------------------------------
# 12. Explicit task cancellation, gated on events
# ---------------------------------------------------------------------------


async def test_cancel_active_ainvoke_reading_response(offline_gemini):
    gate = asyncio.Event()
    body = offline_gemini.body([wait_for(gate), send(b"never-reached")])
    offline_gemini.respond(lambda request: offline_gemini.sse_response(body))
    model = offline_gemini.model()

    task = asyncio.create_task(model.ainvoke("ping"))
    try:
        await asyncio.wait_for(body.started.wait(), timeout=5)
        task.cancel()
        await observe_explicit_cancellation(task, timeout=5)

        # Business assertions run while the gate is still unset and before
        # the fixture can close the client, so none of them is cleanup evidence.
        assert len(offline_gemini.requests) == 1
        assert body.close_count >= 1, "response created before cancellation must close"
    finally:
        await reclaim_own_task(task, gate=gate)



async def test_cancel_astream_before_first_content(offline_gemini):
    gate = asyncio.Event()
    body = offline_gemini.body(
        [
            send(sse_frame({"candidates": [candidate("", finish_reason=None)]})),
            wait_for(gate),
            send(sse_frame({"candidates": [candidate("late", finish_reason=None)]})),
        ]
    )
    offline_gemini.respond(lambda request: offline_gemini.sse_response(body))
    model = offline_gemini.model()

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
        assert len(offline_gemini.requests) == 1
        assert "".join(c.content for c in received if isinstance(c.content, str)) == "", (
            "an empty-text frame is not body text"
        )
        assert body.close_count >= 1
    finally:
        await reclaim_own_task(task, gate=gate)



async def test_cancel_astream_after_first_content_keeps_prefix(offline_gemini):
    gate = asyncio.Event()
    body = offline_gemini.body(
        [
            send(sse_frame({"candidates": [candidate("first", finish_reason=None)]})),
            wait_for(gate),
        ]
    )
    offline_gemini.respond(lambda request: offline_gemini.sse_response(body))
    model = offline_gemini.model()

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
        assert len(offline_gemini.requests) == 1
        assert "".join(c.content for c in received if isinstance(c.content, str)) == "first"
        assert body.close_count >= 1
    finally:
        await reclaim_own_task(task, gate=gate)



# ---------------------------------------------------------------------------
# 13. Transport EOF after a content frame (characterisation only)
# ---------------------------------------------------------------------------


async def test_stream_eof_after_content_frame_delivers_text(offline_gemini):
    """The candidate genuinely carries no finishReason, checked on the input.

    Characterisation only: this says nothing about usage/finish completeness,
    which stays out of scope for D4a-4.
    """
    frames = stream_frames("par", final=False)
    for frame in frames:
        payload = json.loads(frame.decode().split("data: ", 1)[1].strip())
        assert "finishReason" not in payload["candidates"][0], (
            "the EOF fixture must not smuggle a completion signal"
        )

    offline_gemini.respond(
        lambda request: offline_gemini.sse_response(offline_gemini.frame_body(frames))
    )
    model = offline_gemini.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_gemini.requests) == 1
    assert "".join(c.content for c in chunks if isinstance(c.content, str)) == "par"


# ===========================================================================
# D5A-1 R2: normal completion/usage projection + streamed usage position
#
# Appended at EOF only. Every byte above this marker is the D4 baseline and is
# unchanged. The usage numbers and chunk contents are carried over from the
# first run unchanged; only the sentinel and request assertions are added.
#
#   langchain_core/language_models/chat_models.py:935-950   terminal sentinel
#   langchain_google_genai/chat_models.py:846-880    _parse_response_candidate
#   langchain_google_genai/chat_models.py:1104-1161  _response_to_result usage
#   langchain_google_genai/chat_models.py:1163-1223  generation_info per frame
#   langchain_google_genai/chat_models.py:3199-3234  _astream prev_usage threading
#   langchain_core/messages/ai.py:676-715,721-740   chunk add sums usage
#   langchain_core/messages/ai.py:790-840          subtract_usage
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
    The Gemini adapter never sets one, so the sentinel is always present here
    and must be asserted rather than filtered out.
    """
    assert chunk.content == ""
    assert chunk.usage_metadata is None
    assert chunk.response_metadata == {}
    assert chunk.chunk_position == "last"


def d5a_frame(
    text: str,
    *,
    usage: dict | None = None,
    finish_reason: str | None = None,
) -> bytes:
    """One Gemini streaming frame that may carry ``usageMetadata``.

    ``stream_frames`` above never attaches usage, so it cannot express where
    usage arrives. ``modelVersion`` is always included because the adapter
    reads the model name from it.
    """
    payload: dict = {
        "modelVersion": REPRESENTATIVE_MODEL,
        "candidates": [candidate(text, finish_reason=finish_reason)],
    }
    if usage is not None:
        payload["usageMetadata"] = usage
    return sse_frame(payload)


async def test_d5a_baseline_ainvoke(offline_gemini):
    """One normal non-stream response pins the whole usage projection."""
    offline_gemini.respond(json_response(generate_response("pong")))
    model = offline_gemini.model()

    result = await model.ainvoke("ping")

    assert len(offline_gemini.requests) == 1
    assert_real_request(offline_gemini.requests[0], streaming=False)

    assert result.content == "pong"
    # totalTokenCount is read straight off the wire rather than recomputed.
    # Note this fixture cannot by itself separate "read the total" from
    # "recomputed 3 + 2", because 5 happens to equal 3 + 2 here.
    assert result.usage_metadata == {
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
        "input_token_details": {"cache_read": 0},
    }
    # cache_read is always projected -- as 0 when the wire omits
    # cachedContentTokenCount -- unlike OpenAI/Anthropic, which drop absent
    # detail keys entirely.
    assert result.response_metadata["finish_reason"] == "STOP"
    assert result.response_metadata["model_name"] == REPRESENTATIVE_MODEL
    assert result.response_metadata["model_provider"] == "google_genai"
    assert result.response_metadata["safety_ratings"] == []


async def test_d5a_baseline_astream(offline_gemini):
    """Streamed usage lands on the terminal frame, and its position is pinned."""
    offline_gemini.respond(
        lambda request: offline_gemini.sse_response(
            offline_gemini.frame_body(
                [
                    d5a_frame("po"),
                    d5a_frame("ng"),
                    d5a_frame(
                        "",
                        finish_reason="STOP",
                        usage={
                            "promptTokenCount": 3,
                            "candidatesTokenCount": 2,
                            "totalTokenCount": 5,
                        },
                    ),
                ]
            )
        )
    )
    model = offline_gemini.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_gemini.requests) == 1
    assert_real_request(offline_gemini.requests[0], streaming=True)

    # Two text frames, the terminal frame, and the langchain-core sentinel the
    # adapter never marks itself.
    assert len(chunks) == 4
    assert [c.content for c in chunks] == ["po", "ng", "", ""]
    # Only the frame that actually carries usageMetadata projects a usage
    # delta; the two text frames leave prev_usage at None and report nothing.
    assert [c.usage_metadata for c in chunks] == [
        None,
        None,
        {
            "input_tokens": 3,
            "output_tokens": 2,
            "total_tokens": 5,
            "input_token_details": {"cache_read": 0},
        },
        None,
    ]
    # model_name reaches response_metadata only on the finishing frame, so
    # concatenating chunks does not duplicate it. That frame is index 2; the
    # sentinel follows it.
    assert "finish_reason" not in chunks[0].response_metadata
    assert "model_name" not in chunks[0].response_metadata
    assert chunks[2].response_metadata["finish_reason"] == "STOP"
    assert chunks[2].response_metadata["model_name"] == REPRESENTATIVE_MODEL
    assert chunks[2].chunk_position is None
    d5a_assert_core_sentinel(chunks[3])

    aggregated = d5a_add_chunks(chunks)
    assert aggregated.content == "pong"
    assert aggregated.usage_metadata == {
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
        "input_token_details": {"cache_read": 0},
    }
    assert aggregated.response_metadata["finish_reason"] == "STOP"


async def test_d5a_usage_lifecycle_astream(offline_gemini):
    """The wire is cumulative; the adapter projects deltas; summing recovers it.

    Gemini repeats accumulated ``usageMetadata`` on later frames, so this is
    the one provider where a single request can distinguish cumulative from
    incremental reporting. Three frames report growing totals; the adapter
    subtracts the running previous value, so each chunk carries the increment
    (4,2,6) then (0,3,3) then (0,2,2). LangChain's own chunk addition then
    reproduces the cumulative (4,7,11) with no double counting.
    """
    offline_gemini.respond(
        lambda request: offline_gemini.sse_response(
            offline_gemini.frame_body(
                [
                    d5a_frame(
                        "po",
                        usage={
                            "promptTokenCount": 4,
                            "candidatesTokenCount": 2,
                            "totalTokenCount": 6,
                        },
                    ),
                    d5a_frame(
                        "ng",
                        usage={
                            "promptTokenCount": 4,
                            "candidatesTokenCount": 5,
                            "totalTokenCount": 9,
                        },
                    ),
                    d5a_frame(
                        "",
                        finish_reason="STOP",
                        usage={
                            "promptTokenCount": 4,
                            "candidatesTokenCount": 7,
                            "totalTokenCount": 11,
                        },
                    ),
                ]
            )
        )
    )
    model = offline_gemini.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_gemini.requests) == 1
    assert_real_request(offline_gemini.requests[0], streaming=True)

    # Per-frame deltas, not the wire's cumulative numbers, plus the core
    # sentinel which reports no usage at all.
    assert len(chunks) == 4
    assert [c.usage_metadata for c in chunks] == [
        {
            "input_tokens": 4,
            "output_tokens": 2,
            "total_tokens": 6,
            "input_token_details": {"cache_read": 0},
        },
        {
            "input_tokens": 0,
            "output_tokens": 3,
            "total_tokens": 3,
            "input_token_details": {"cache_read": 0},
        },
        {
            "input_tokens": 0,
            "output_tokens": 2,
            "total_tokens": 2,
            "input_token_details": {"cache_read": 0},
        },
        None,
    ]
    # Usage is present from the very first frame here, unlike the single
    # terminal report the other three providers send.
    assert chunks[0].usage_metadata is not None
    assert chunks[2].chunk_position is None
    d5a_assert_core_sentinel(chunks[3])

    aggregated = d5a_add_chunks(chunks)
    assert aggregated.content == "pong"
    assert aggregated.usage_metadata == {
        "input_tokens": 4,
        "output_tokens": 7,
        "total_tokens": 11,
        "input_token_details": {"cache_read": 0},
    }


# ===========================================================================
# D5A-2: completion boundaries and missing usage (7 nodes)
#
# Appended at EOF only. Every byte above this marker is the D4 baseline plus
# the accepted D5A-1 section, and is unchanged. Each fixed expectation was
# derived from the locked sources below, not from an observed run:
#
#   google/genai/types.py:425-443            FinishReason members
#   google/genai/_common.py:651              unknown value warns, is accepted
#   langchain_google_genai/chat_models.py:1163-1183  finish_reason -> .name
#   langchain_google_genai/chat_models.py:1116-1159  usage projection
#   langchain_core/language_models/chat_models.py:935-950  core sentinel
#   langchain_core/messages/ai.py:676-681    no usage -> None
#   backend/pyproject.toml                   no filterwarnings config
# ===========================================================================


def d5a2_null_finish_frame() -> bytes:
    """One frame whose candidate carries an explicit JSON null finishReason.

    ``candidate`` above omits the key whenever finish_reason is None, so an
    explicit null cannot be expressed with it. The usageMetadata is carried
    exactly as in the missing-key case, so the two closing frames differ only
    in how finishReason is expressed.
    """
    null_candidate = candidate("", finish_reason=None)
    null_candidate["finishReason"] = None
    return sse_frame(
        {
            "modelVersion": REPRESENTATIVE_MODEL,
            "candidates": [null_candidate],
            "usageMetadata": {
                "promptTokenCount": 3,
                "candidatesTokenCount": 2,
                "totalTokenCount": 5,
            },
        }
    )


async def test_d5a_completion_missing_astream(offline_gemini):
    """No finishReason key on the closing frame, usageMetadata still sent."""
    offline_gemini.respond(
        lambda request: offline_gemini.sse_response(
            offline_gemini.frame_body(
                [
                    d5a_frame("po"),
                    d5a_frame("ng"),
                    d5a_frame(
                        "",
                        usage={
                            "promptTokenCount": 3,
                            "candidatesTokenCount": 2,
                            "totalTokenCount": 5,
                        },
                    ),
                ]
            )
        )
    )
    model = offline_gemini.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_gemini.requests) == 1
    assert_real_request(offline_gemini.requests[0], streaming=True)

    assert len(chunks) == 4
    assert [c.content for c in chunks] == ["po", "ng", "", ""]
    assert [c.usage_metadata for c in chunks] == [
        None,
        None,
        {
            "input_tokens": 3,
            "output_tokens": 2,
            "total_tokens": 5,
            "input_token_details": {"cache_read": 0},
        },
        None,
    ]
    # model_name only rides along with a truthy finish_reason, so even though
    # the frame carries modelVersion this chunk has neither key.
    assert chunks[2].response_metadata == {
        "model_provider": "google_genai",
        "safety_ratings": [],
    }
    assert chunks[2].chunk_position is None
    d5a_assert_core_sentinel(chunks[3])

    aggregated = d5a_add_chunks(chunks)
    assert aggregated.content == "pong"
    assert aggregated.usage_metadata == {
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
        "input_token_details": {"cache_read": 0},
    }
    assert aggregated.response_metadata == {
        "model_provider": "google_genai",
        "safety_ratings": [],
    }


async def test_d5a_completion_null_astream(offline_gemini):
    """An explicit JSON null finishReason, kept apart from the missing key.

    Candidate.finish_reason is Optional, so an absent key and an explicit null
    decode identically and the projection matches
    test_d5a_completion_missing_astream. The wire really does differ.
    """
    offline_gemini.respond(
        lambda request: offline_gemini.sse_response(
            offline_gemini.frame_body(
                [
                    d5a_frame("po"),
                    d5a_frame("ng"),
                    d5a2_null_finish_frame(),
                ]
            )
        )
    )
    model = offline_gemini.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_gemini.requests) == 1
    assert_real_request(offline_gemini.requests[0], streaming=True)

    assert json.loads(
        d5a_frame(
            "",
            usage={
                "promptTokenCount": 3,
                "candidatesTokenCount": 2,
                "totalTokenCount": 5,
            },
        ).split(b"data: ", 1)[1]
    )["candidates"][0] != json.loads(
        d5a2_null_finish_frame().split(b"data: ", 1)[1]
    )["candidates"][0]
    # The two closing frames differ only in how finishReason is expressed:
    # absent key on one side, explicit null on the other, and nothing else.
    missing_frame = json.loads(
        d5a_frame(
            "",
            usage={
                "promptTokenCount": 3,
                "candidatesTokenCount": 2,
                "totalTokenCount": 5,
            },
        ).split(b"data: ", 1)[1]
    )
    null_frame = json.loads(d5a2_null_finish_frame().split(b"data: ", 1)[1])
    missing_candidate = dict(missing_frame["candidates"][0])
    null_candidate = dict(null_frame["candidates"][0])
    assert "finishReason" not in missing_candidate
    assert null_candidate["finishReason"] is None
    del null_candidate["finishReason"]
    assert missing_candidate == null_candidate
    # usage must not differ either: this case is about the reason key alone.
    assert missing_frame["usageMetadata"] == null_frame["usageMetadata"]

    assert len(chunks) == 4
    assert [c.content for c in chunks] == ["po", "ng", "", ""]
    assert [c.usage_metadata for c in chunks] == [
        None,
        None,
        {
            "input_tokens": 3,
            "output_tokens": 2,
            "total_tokens": 5,
            "input_token_details": {"cache_read": 0},
        },
        None,
    ]
    assert chunks[2].response_metadata == {
        "model_provider": "google_genai",
        "safety_ratings": [],
    }
    assert chunks[2].chunk_position is None
    d5a_assert_core_sentinel(chunks[3])

    aggregated = d5a_add_chunks(chunks)
    assert aggregated.content == "pong"
    assert aggregated.usage_metadata == {
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
        "input_token_details": {"cache_read": 0},
    }
    assert aggregated.response_metadata == {
        "model_provider": "google_genai",
        "safety_ratings": [],
    }


async def test_d5a_usage_missing_astream(offline_gemini):
    """Completion signal normal, usageMetadata simply not sent."""
    offline_gemini.respond(
        lambda request: offline_gemini.sse_response(
            offline_gemini.frame_body(
                [d5a_frame("po"), d5a_frame("ng"), d5a_frame("", finish_reason="STOP")]
            )
        )
    )
    model = offline_gemini.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_gemini.requests) == 1
    assert_real_request(offline_gemini.requests[0], streaming=True)

    assert len(chunks) == 4
    assert [c.content for c in chunks] == ["po", "ng", "", ""]
    # No frame carries usageMetadata, so every chunk reports none and the
    # aggregate is None rather than a zero dict.
    assert [c.usage_metadata for c in chunks] == [None, None, None, None]
    assert chunks[2].response_metadata["finish_reason"] == "STOP"
    assert chunks[2].response_metadata["model_name"] == REPRESENTATIVE_MODEL
    d5a_assert_core_sentinel(chunks[3])

    aggregated = d5a_add_chunks(chunks)
    assert aggregated.content == "pong"
    assert aggregated.usage_metadata is None
    assert aggregated.response_metadata["finish_reason"] == "STOP"


async def test_d5a_truncated_ainvoke(offline_gemini):
    """The protocol's own truncation reason, MAX_TOKENS."""
    offline_gemini.respond(
        json_response(generate_response("pong", finish_reason="MAX_TOKENS"))
    )
    model = offline_gemini.model()

    result = await model.ainvoke("ping")

    assert len(offline_gemini.requests) == 1
    assert_real_request(offline_gemini.requests[0], streaming=False)

    assert result.content == "pong"
    # cache_read is always projected, as 0 when the wire omits the field.
    assert result.usage_metadata == {
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
        "input_token_details": {"cache_read": 0},
    }
    # The adapter publishes the enum member's .name, never the value object,
    # and adds no truncation flag of its own.
    assert result.response_metadata["finish_reason"] == "MAX_TOKENS"
    assert result.response_metadata["model_name"] == REPRESENTATIVE_MODEL
    assert result.response_metadata["model_provider"] == "google_genai"
    assert result.response_metadata["safety_ratings"] == []


async def test_d5a_truncated_astream(offline_gemini):
    """The same MAX_TOKENS reason through the streaming entry."""
    offline_gemini.respond(
        lambda request: offline_gemini.sse_response(
            offline_gemini.frame_body(
                [
                    d5a_frame("po"),
                    d5a_frame("ng"),
                    d5a_frame(
                        "",
                        finish_reason="MAX_TOKENS",
                        usage={
                            "promptTokenCount": 3,
                            "candidatesTokenCount": 2,
                            "totalTokenCount": 5,
                        },
                    ),
                ]
            )
        )
    )
    model = offline_gemini.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_gemini.requests) == 1
    assert_real_request(offline_gemini.requests[0], streaming=True)

    assert len(chunks) == 4
    assert [c.content for c in chunks] == ["po", "ng", "", ""]
    assert [c.usage_metadata for c in chunks] == [
        None,
        None,
        {
            "input_tokens": 3,
            "output_tokens": 2,
            "total_tokens": 5,
            "input_token_details": {"cache_read": 0},
        },
        None,
    ]
    assert chunks[2].response_metadata["finish_reason"] == "MAX_TOKENS"
    assert chunks[2].response_metadata["model_name"] == REPRESENTATIVE_MODEL
    d5a_assert_core_sentinel(chunks[3])

    aggregated = d5a_add_chunks(chunks)
    assert aggregated.content == "pong"
    assert aggregated.usage_metadata == {
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
        "input_token_details": {"cache_read": 0},
    }
    assert aggregated.response_metadata["finish_reason"] == "MAX_TOKENS"


async def test_d5a_unknown_ainvoke(offline_gemini):
    """A synthetic unknown finishReason is accepted, with a warning.

    The CaseInSensitiveEnum accepts any string, warns that it is not a valid
    FinishReason, and keeps .name equal to the raw wire string. The warning is
    part of the accepted behaviour, not a failure.
    """
    offline_gemini.respond(
        json_response(generate_response("pong", finish_reason="d5a_unknown"))
    )
    model = offline_gemini.model()

    with pytest.warns(UserWarning, match="not a valid FinishReason"):
        result = await model.ainvoke("ping")

    assert len(offline_gemini.requests) == 1
    assert_real_request(offline_gemini.requests[0], streaming=False)

    assert result.content == "pong"
    assert result.usage_metadata == {
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
        "input_token_details": {"cache_read": 0},
    }
    # The raw string is passed through; the UNKNOWN_... fallback would need an
    # integer finish reason, which this wire does not send.
    assert result.response_metadata["finish_reason"] == "d5a_unknown"
    assert result.response_metadata["model_name"] == REPRESENTATIVE_MODEL
    assert result.response_metadata["model_provider"] == "google_genai"


async def test_d5a_terminal_frame_missing_astream(offline_gemini):
    """A complete content prefix with no closing finish frame.

    Every frame still carries a candidate, so the adapter never takes its
    empty-candidates branch; what is missing is the finish reason, the model
    name and the usage.
    """
    offline_gemini.respond(
        lambda request: offline_gemini.sse_response(
            offline_gemini.frame_body([d5a_frame("par"), d5a_frame("tial")])
        )
    )
    model = offline_gemini.model()

    chunks = [chunk async for chunk in model.astream("ping")]

    assert len(offline_gemini.requests) == 1
    assert_real_request(offline_gemini.requests[0], streaming=True)

    assert len(chunks) == 3
    assert [c.content for c in chunks] == ["par", "tial", ""]
    assert [c.usage_metadata for c in chunks] == [None, None, None]
    assert all(
        c.response_metadata
        == {"model_provider": "google_genai", "safety_ratings": []}
        for c in chunks[:2]
    )
    d5a_assert_core_sentinel(chunks[2])

    aggregated = d5a_add_chunks(chunks)
    assert aggregated.content == "partial"
    assert aggregated.usage_metadata is None
    assert aggregated.response_metadata == {
        "model_provider": "google_genai",
        "safety_ratings": [],
    }
