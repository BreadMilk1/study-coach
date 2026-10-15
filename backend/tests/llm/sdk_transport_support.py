"""Offline transport scaffolding for the D4a-1 OpenAI async contract tests.

The intent of this module is to replace *only the socket*. Everything else on
the path stays real: ``app.llm.provider.get_chat_model`` builds a real
``ChatOpenAI``, which builds a real ``openai.AsyncOpenAI``, which runs the real
SDK resource and parser, and the real adapter chunk conversion runs on the
result.

That rules out the obvious shortcut of passing ``http_async_client`` into
``get_chat_model``. The locked langchain-openai adapter reads
``http_async_client is not None`` when deciding whether to turn ``stream_usage``
on by default, so handing it the client directly would silently walk a
different production branch than the one the product uses. Instead the tests
patch ``langchain_openai.chat_models.base._get_default_async_httpx_client`` --
the constructor entry the adapter itself calls -- and return a genuine client
whose transport is ``httpx.MockTransport``.

The client is built on the locked SDK's *public* ``DefaultAsyncHttpxClient`` /
``DefaultHttpxClient`` rather than on bare ``httpx.AsyncClient``, because those
carry the SDK's own defaults (``follow_redirects=True`` and
``DEFAULT_CONNECTION_LIMITS``). A bare httpx client would silently ship
``follow_redirects=False`` and different pool limits, i.e. a different default
than production. Only ``base_url``/``timeout`` are forwarded, exactly as the
SDK factory received them, plus the mock transport and ``trust_env=False``.

The SDK default ``https://api.openai.com/v1`` is therefore the URL the request
carries; only ``MockTransport`` ever sees it, and no DNS lookup or connection
is ever attempted.
"""

from __future__ import annotations

import asyncio
import socket
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import openai
import pytest

from langchain_openai import chat_models as _openai_chat_models

from app.llm.provider import LLMConfig, get_chat_model

# ``base`` is where the adapter holds its module-level imports of the two client
# factories. Patching the names here is the same lookup the adapter performs.
_ADAPTER_MODULE = _openai_chat_models.base

SYNTHETIC_OPENAI_KEY = "sk-offline-d4a1-synthetic"
REPRESENTATIVE_MODEL = "gpt-4o-mini"
SDK_DEFAULT_BASE_URL = "https://api.openai.com/v1"

# Any of these would steer the adapter off its default branch (notably
# ``OPENAI_BASE_URL``, which suppresses the ``stream_usage`` default). The stdin
# runner clears credentials before pytest is imported; this fixture keeps the
# tests hermetic under any other entry point as well.
STEERING_ENV_VARS = (
    "OPENAI_API_KEY",
    "OPENAI_API_BASE",
    "OPENAI_BASE_URL",
    "OPENAI_PROXY",
    "OPENAI_ORG_ID",
    "OPENAI_ORGANIZATION",
    "OPENAI_PROJECT_ID",
    "LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S",
)

# Scripted body-step kinds.
BYTES = "bytes"
WAIT = "wait"
RAISE = "raise"


# ---------------------------------------------------------------------------
# Portable source identity
#
# The backend root is derived from this file's own location, so the same test
# passes from any checkout (remote, Mac, CI) without hard-coding an absolute
# path. Membership is checked by exact resolved equality against the expected
# relative location -- deliberately *not* ``startswith``/``is_relative_to``,
# which would happily accept a different checkout sharing a path prefix.
# ---------------------------------------------------------------------------


def backend_root() -> Path:
    """The backend root that owns this test package."""
    # <backend>/tests/llm/sdk_transport_support.py -> <backend>
    return Path(__file__).resolve().parents[2]


def assert_module_from_backend(module: ModuleType, relative_path: str) -> Path:
    """Assert ``module`` was imported from this checkout, at an exact location."""
    origin = Path(module.__file__).resolve()
    expected = (backend_root() / relative_path).resolve()
    assert origin == expected, (
        f"{module.__name__} was imported from {origin}, "
        f"but this checkout expects {expected}"
    )
    return origin


def assert_module_is_test_module(module: ModuleType) -> Path:
    """Assert ``module`` is the test module this run is executing."""
    return assert_module_from_backend(module, "tests/llm/test_async_transport_contract.py")


# ---------------------------------------------------------------------------
# Body scripting
# ---------------------------------------------------------------------------


def send(data: bytes) -> tuple[str, Any]:
    return (BYTES, data)


def wait_for(event: asyncio.Event) -> tuple[str, Any]:
    """Hold the body open until ``event`` is set, so tests gate on an event
    rather than on sleep timing."""
    return (WAIT, event)


def fail_with(exc: BaseException) -> tuple[str, Any]:
    return (RAISE, exc)


def sse_event(payload: str, *, event: str | None = None) -> bytes:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {payload}\n\n".encode()


DONE_EVENT = b"data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Bounded, test-owned cancellation observation and reclaim.
#
# These two helpers are deliberately narrow: they observe a cancellation the
# test itself requested, and they reclaim a task the test itself created. They
# are not a retry policy, a timeout policy, or a way to force any coroutine to
# stop -- an SDK that ignores cancellation entirely is still bounded only by the
# outer 180s process wall clock.
# ---------------------------------------------------------------------------


async def observe_explicit_cancellation(
    task: "asyncio.Task[None]", *, timeout: float = 5.0
) -> asyncio.CancelledError:
    """Require the exact ``CancelledError`` from a task the test cancelled.

    ``asyncio.shield`` is what makes this a *bounded observation* rather than a
    second cancellation: the timeout applies to this awaiting coroutine and can
    never be delivered to the task under test. A ``TimeoutError`` here means the
    task did not settle within ``timeout`` and must fail the case -- it is never
    treated as the expected cancellation. A task that swallows the cancellation
    and returns normally fails too, by falling through to the final ``raise``.
    """
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
    except asyncio.CancelledError as exc:
        assert type(exc) is asyncio.CancelledError, (
            "the cancellation must be exactly asyncio.CancelledError"
        )
        assert task.cancelled(), "the cancelled task must end in the cancelled state"
        return exc
    raise AssertionError(
        "the cancelled task completed normally instead of raising CancelledError"
    )


async def reclaim_own_task(
    task: "asyncio.Task[None]",
    *,
    gate: asyncio.Event | None = None,
    timeout: float = 5.0,
) -> None:
    """Bounded reclaim of a task this test created, after its assertions ran.

    This is cleanup only. It releases the caller's own ``gate`` and, if the task
    is still running, cancels it once more -- neither action may ever be counted
    as the cancellation under test, which is why this is called from ``finally``
    after every business assertion has already been made.

    ``asyncio.wait`` observes ``done``/``pending`` without cancelling anything
    itself, so it never manufactures a second cancellation to hide a hung task.
    A task still pending after ``timeout`` fails the case. Only the
    ``CancelledError`` this cleanup itself may cause is consumed; any other
    exception the task raised is re-raised untouched, so a real failure under
    test is still reported instead of being turned into a pass.
    """
    if gate is not None:
        gate.set()
    if not task.done():
        task.cancel()
    _done, pending = await asyncio.wait({task}, timeout=timeout)
    if pending:
        raise AssertionError(
            "test-owned task did not settle within the bounded cleanup window"
        )
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        raise exc


class ScriptedByteStream(httpx.AsyncByteStream):
    """A deterministic response body whose every step is explicit.

    Steps are consumed in order and the stream never invents bytes on its own,
    so a test can say exactly where a body starts, where it stalls and where it
    fails. ``aclose`` is only ever called by httpx/the SDK -- the stream does
    not close itself from ``__aiter__``, so the recorded ``close_count``
    reflects the real closing path rather than the fixture's cleanup.
    """

    def __init__(self, steps: Sequence[tuple[str, Any]], *, name: str = "body") -> None:
        self._steps = list(steps)
        self.name = name
        self.steps_reached = 0
        self.bytes_sent = 0
        self.close_count = 0
        # Set as soon as the SDK starts reading this body. Tests await it so
        # they cancel at a known point instead of guessing with sleep timing.
        self.started = asyncio.Event()

    async def __aiter__(self):
        self.started.set()
        for kind, payload in self._steps:
            self.steps_reached += 1
            if kind == BYTES:
                self.bytes_sent += len(payload)
                yield payload
            elif kind == WAIT:
                await payload.wait()
            elif kind == RAISE:
                raise payload
            else:
                raise AssertionError(f"unknown scripted body step: {kind!r}")

    async def aclose(self) -> None:
        self.close_count += 1


# ---------------------------------------------------------------------------
# Transport harness
# ---------------------------------------------------------------------------


class OfflineOpenAI:
    """MockTransport-backed stand-in for the provider socket.

    Holds every client it hands to the SDK so the fixture can close them all
    afterwards, and every request so tests can assert the real request count
    (which is how ``send`` retries are distinguished from non-replayed bodies).
    """

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.streams: list[ScriptedByteStream] = []
        self.async_clients: list[httpx.AsyncClient] = []
        self.sync_clients: list[httpx.Client] = []
        self._responder: Callable[[httpx.Request], httpx.Response] | None = None

    # -- wiring -----------------------------------------------------------
    def respond(self, responder: Callable[[httpx.Request], httpx.Response]) -> "OfflineOpenAI":
        self._responder = responder
        return self

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self._responder is None:
            raise AssertionError("offline transport has no responder installed")
        return self._responder(request)

    def async_client_factory(
        self,
        base_url: str | None,
        timeout: Any,
        socket_options: Iterable[Any] = (),
    ) -> httpx.AsyncClient:
        client = openai.DefaultAsyncHttpxClient(
            base_url=base_url or SDK_DEFAULT_BASE_URL,
            timeout=timeout,
            transport=httpx.MockTransport(self.handle),
            trust_env=False,
        )
        self.async_clients.append(client)
        return client

    def sync_client_factory(
        self,
        base_url: str | None,
        timeout: Any,
        socket_options: Iterable[Any] = (),
    ) -> httpx.Client:
        client = openai.DefaultHttpxClient(
            base_url=base_url or SDK_DEFAULT_BASE_URL,
            timeout=timeout,
            transport=httpx.MockTransport(self.handle),
            trust_env=False,
        )
        self.sync_clients.append(client)
        return client

    # -- bodies -----------------------------------------------------------
    def body(self, steps: Sequence[tuple[str, Any]], *, name: str = "body") -> ScriptedByteStream:
        stream = ScriptedByteStream(steps, name=name)
        self.streams.append(stream)
        return stream

    def sse_body(
        self,
        events: Sequence[bytes],
        *,
        terminator: bytes | None = DONE_EVENT,
        name: str = "sse",
        stall: bool = False,
    ) -> ScriptedByteStream:
        """Build an SSE body.

        ``terminator=None`` produces a body that simply runs out of bytes, which
        is a *transport EOF* rather than a provider-completed stream -- the two
        are deliberately kept distinct here.

        ``stall=True`` appends a gate that never opens, so the body stays open
        forever instead of ending; that is what a mid-stream stall looks like to
        the adapter's chunk guard.
        """
        steps: list[tuple[str, Any]] = [send(event) for event in events]
        if terminator is not None:
            steps.append(send(terminator))
        if stall:
            steps.append(wait_for(asyncio.Event()))
        return self.body(steps, name=name)

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

    # -- model ------------------------------------------------------------
    def model(self, **kwargs):
        """Build the real chat model through the product's own factory.

        Only ``api_key`` and ``model`` are supplied -- no ``base_url``, no
        ``http_async_client``, no ``stream_usage``. That is deliberate: those
        are exactly the kwargs that would move the adapter off its default
        branch.
        """
        config = LLMConfig(
            provider="openai",
            model=REPRESENTATIVE_MODEL,
            api_key=SYNTHETIC_OPENAI_KEY,
        )
        return get_chat_model(config, **kwargs)

    # -- teardown ---------------------------------------------------------
    async def aclose(self) -> None:
        for client in self.async_clients:
            await client.aclose()
        self.async_clients.clear()
        for client in self.sync_clients:
            client.close()
        self.sync_clients.clear()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def offline_env(monkeypatch):
    """Keep the adapter on its default branch regardless of the outer shell."""
    for name in STEERING_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("STUDY_COACH_TEST_MODE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    yield


@pytest.fixture(autouse=True)
def deny_sockets(monkeypatch):
    """Any real socket attempt is an immediate, fixed failure."""

    def deny(*args, **kwargs):
        raise AssertionError("D4a-1 offline test attempted real network access")

    monkeypatch.setattr(socket, "getaddrinfo", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    yield


@pytest.fixture
async def offline_openai(monkeypatch) -> AsyncIterator[OfflineOpenAI]:
    harness = OfflineOpenAI()

    # Only the two HTTPX client construction entries are patched. The factory,
    # ``init_chat_model``, the SDK resource, the SDK parser and the adapter
    # conversion functions all keep running for real.
    monkeypatch.setattr(
        _ADAPTER_MODULE, "_get_default_async_httpx_client", harness.async_client_factory
    )
    monkeypatch.setattr(
        _ADAPTER_MODULE, "_get_default_httpx_client", harness.sync_client_factory
    )

    # Remove the SDK's real backoff so retry-exhaustion cases do not sleep.
    # max_retries and the SDK retry predicate are untouched.
    monkeypatch.setattr(
        openai._base_client.BaseClient,
        "_calculate_retry_timeout",
        lambda *args, **kwargs: 0.0,
    )

    try:
        yield harness
    finally:
        await harness.aclose()
