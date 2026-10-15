"""D4b-4: real locked SDK -> real API/SSE boundary contract evidence.

Scope: the HTTP-facing boundary of the product -- ``routes.chat``'s SSE
generator, the production DI wiring in ``app.api.deps``, and the two model
probe routes. Everything from the socket up to the client stays real:
``deps.get_llm``/``get_judge_dependencies`` build real adapters through
``get_chat_model``, the real locked ``openai`` SDK resource and parser run, the
real ``production_v1`` Graph drives the turn, the real ``routes`` handlers make
every decision, and the real SQLAlchemy repositories write a real temporary
SQLite file.

Only the socket is replaced, through the frozen D4a seams. The non-LLM seams
that are unavoidable here are synthetic and local to this file: one
``retriever.search``, one seeded ``User``/``Document`` pair, and an in-memory
checkpointer. No factory, model, ``ainvoke``/``astream``, parser, route,
Graph, repository, tool argument, retry count, timeout or ``stream_usage``
default is faked or forced.

Two entry points are used, and every case declares which one it used:

* **direct iterator** -- ``routes.chat(...)`` is awaited directly and its real
  ``response.body_iterator`` is consumed frame by frame. This is the entry
  that can observe cancellation and mid-stream failure, because nothing
  buffers between the handler and this consumer.
* **HTTP ASGI** -- a bare ``FastAPI()`` served through
  ``httpx.ASGITransport``. ``ASGITransport`` accumulates the whole body before
  returning it, so an HTTP case proves routing, dependency injection and
  response assembly only. It is never used as incrementality or cancellation
  evidence.

Every document, chunk, payload, marker, score and credential in this file is
synthetic. A "valid" judge score here means "the payload was legal", never "a
provider really scored this". This file is read-only evidence: it fixes
current behaviour and asserts no product change.
"""

from __future__ import annotations

import asyncio
import json
import sys
import sysconfig
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from tests.llm.sdk_transport_support import (  # noqa: F401  (frozen, read-only)
    DONE_EVENT,
    SYNTHETIC_OPENAI_KEY,
    ScriptedByteStream,
    assert_module_from_backend,
    deny_sockets,
    fail_with,
    offline_openai,
    reclaim_own_task,
    send,
    sse_event,
    wait_for,
)
from tests.llm.sdk_transport_support import offline_env as offline_env  # noqa: F401

from app.agent.graph import _UNAVAILABLE_SUFFIX
from app.agent.production_judge import TUTOR_DIMENSIONS
from app.api import deps, routes
from app.db.models import Base, Citation, Message
from app.db.repositories import (
    ChatSessionRepository,
    DocumentRepository,
    MessageRepository,
    UserRepository,
)
from app.db.session import make_engine
from app.llm.content import EMPTY_CONTENT_MESSAGE
from app.llm.provider import LLMConfig


# ---------------------------------------------------------------------------
# Synthetic inputs.
#
# The question carries no quiz/plan keyword on purpose: `route_intent` must
# resolve it to "tutor", which is the only branch this slice exercises.
# ---------------------------------------------------------------------------

SYNTHETIC_QUESTION = "What does the synthetic retrieval note say?"
SYNTHETIC_FINGERPRINT = "d4b4-synthetic-fingerprint"

SYNTHETIC_CHUNK_A = {
    "chunk_id": "d4b4-synthetic:rrf:1:0",
    "content": "Synthetic note about reciprocal rank fusion.",
    "source": "d4b4-synthetic-retrieval.pdf",
    "page": 1,
    "score": 0.95,
}

# A marker the product must never echo. The tool-check route is the only place
# where a provider-controlled body could reach a response, so the probe cases
# assert its absence in the returned JSON.
SYNTHETIC_BODY_MARKER = "SYNTHETIC-PROVIDER-BODY-MARKER-d4b4"
SYNTHETIC_ERROR_MARKER = "SYNTHETIC-PROVIDER-PRIVATE-ERROR-MARKER-d4b4"

TUTOR_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o"

# The fixed safe projection for an exhausted 5xx. An SDK status error has no
# entry in the fixed category table, so both probe routes fall back to the one
# generic sentence. Written as a literal, not recomputed from the exception.
GENERIC_LLM_ERROR_TEXT = "LLMError: Model request failed."

# The persisted artifact schema string the assistant row must carry.
ARTIFACTS_SCHEMA = "assistant_artifacts.v1"

# The single synthetic tool the probe route declares, and the call id the
# scripted provider answers with. Both are fixed inputs, never derived from
# an observed response.
PING_TOOL_CALL_ID = "call_d4b4_synthetic_1"
PING_TOOL_NAME = "ping"
PING_TOOL_PROMPT = "Call the ping tool"


# ---------------------------------------------------------------------------
# Local observation seams (non-LLM only).
# ---------------------------------------------------------------------------


class SyntheticRetriever:
    """Allowed non-LLM seam: records the exact retrieval input."""

    def __init__(self, chunks: list[dict]):
        self.chunks = list(chunks)
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        self.calls.append((query, top_k))
        return list(self.chunks[:top_k])


class ObservedGate(asyncio.Event):
    """A never-opened gate that reports when the body actually entered its wait.

    ``wait`` is the only method overridden: it records the entry and then
    delegates to the real ``asyncio.Event.wait``. It never sets the event, so
    the body stays open exactly as long as the SDK leaves it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.waiting_event = asyncio.Event()

    async def wait(self) -> bool:
        self.waiting_event.set()
        return await super().wait()


class CloseObservingBody(ScriptedByteStream):
    """Local body subclass: records that a real ``aclose`` happened.

    ``aclose`` delegates to the frozen implementation first and only then
    records the fact, so the observation can never cause the close it reports.
    """

    def __init__(self, steps, *, name: str = "chat-body"):
        super().__init__(steps, name=name)
        self.closed_event = asyncio.Event()

    async def aclose(self) -> None:
        await super().aclose()
        self.closed_event.set()


# ---------------------------------------------------------------------------
# Wire builders. Small, provider-specific, local to this file.
# ---------------------------------------------------------------------------


def stream_delta_frame(
    *, model: str, text: str = "", finish_reason: str | None = None
) -> bytes:
    """One streaming completion chunk.

    ``text=""`` renders an empty delta, which is the legal way a provider
    reports "no content in this chunk" while the stream continues.
    """
    return sse_event(
        json.dumps(
            {
                "id": "chatcmpl-d4b4-synthetic",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": text} if text else {},
                        "logprobs": None,
                        "finish_reason": finish_reason,
                    }
                ],
            }
        )
    )


def completion_body(
    *,
    model: str,
    text: str | None = None,
    tool_calls: list[dict] | None = None,
    finish_reason: str = "stop",
) -> dict:
    """One non-streaming completion body."""
    message: dict = {"role": "assistant", "refusal": None}
    message["content"] = text
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-d4b4-synthetic",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def provider_error_body(marker: str) -> dict:
    """A 5xx body carrying a provider-private marker that must never surface."""
    return {
        "error": {
            "code": 503,
            "message": marker,
            "type": "synthetic",
        }
    }


def json_responder(payload: dict, *, status: int = 200):
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=json.loads(json.dumps(payload)))

    return responder


def provider_error_responder(status: int, marker: str):
    """A fresh error body per attempt, so no response object is ever reused."""

    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=provider_error_body(marker))

    return responder


def request_json(request: httpx.Request) -> dict:
    return json.loads(request.content)


def judge_pass_payload() -> str:
    """A legal strict rubric payload across all six TUTOR_DIMENSIONS."""
    return json.dumps(
        {**{dim: 4 for dim in TUTOR_DIMENSIONS}, "reasoning": "Synthetic pass."}
    )


# ---------------------------------------------------------------------------
# Expectations.
#
# Every expected prompt is rendered from the test-owned literals below plus this
# file's own synthetic inputs, before any request is sent. Nothing here reads a
# production template, a production rubric or a production renderer, and nothing
# compares an observed prompt with itself: when the product's wording changes,
# the expectations below stay put and the case fails.
# ---------------------------------------------------------------------------

TUTOR_SYSTEM_INSTRUCTION_GOLDEN = """You are a study coach answering questions based ONLY on the provided sources. Cite each fact you use with [N] referring to the source list. If the sources do not contain the answer, say you don't know — do not fabricate."""
TUTOR_SOURCE_LINE_GOLDEN = """[{index}] {source} p.{page}: {content}"""
TUTOR_EMPTY_CONTEXT_GOLDEN = """(no relevant sources retrieved)"""
JUDGE_RUBRIC_GOLDEN = """You are an impartial Study Coach judge. Your job: rate a Tutor's answer to a student question.

**Question**: {question}

**Available Context** (the only material the Tutor should rely on):
{context}

**Tutor's Answer to Evaluate**:
{answer}

---

## Bias-aware instruction (read first)

Evaluate the answer **critically**. Do NOT give high scores merely because the answer matches your style, vocabulary, or training distribution. Be especially harsh when the answer is fluent but ungrounded, or when it sounds confident without citing the available context. Self-preference bias is real — actively counteract it.

---

## Rubric

Rate each dimension from 1 to 5 (integer). All six dimensions are weighted equally — do not collapse pedagogical aspects into a single judgment.

### Grounding (3)

1. **relevance** (1-5): Does the answer directly address the question?
2. **accuracy** (1-5): Is the information accurate **and grounded in the Available Context**? Generally-true but unsupported claims cap at 3.
3. **citation_quality** (1-5): Are sources properly cited (e.g. `[N]`, source name, page)? Source names sufficient; page numbers a bonus.

### Pedagogy (3 — rate INDEPENDENTLY, do NOT average)

4. **accessibility** (1-5): How readable is the explanation for the target student?
   - 1 = heavy unexplained jargon, opaque
   - 3 = some unexplained terms but mostly readable
   - 5 = plain language; any necessary jargon is defined or scaffolded
5. **example_quality** (1-5): Are concrete examples or analogies given that aid understanding?
   - 1 = no example at all
   - 3 = vague or off-topic example
   - 5 = a vivid, on-topic example that makes the concept click
6. **learner_level_fit** (1-5): Is the depth pitched at an exam-prep student (undergrad / postgrad revising)?
   - 1 = too shallow (oversimplified) OR too deep (research-level)
   - 3 = approximately right but uneven
   - 5 = pitched precisely for someone preparing for an exam

**Important**: an answer with vivid examples (example_quality=5) BUT heavy jargon (accessibility=2) must still get accessibility=2 — do NOT inflate it because the example is good. Mixing aspects defeats the diagnostic value of separate dimensions.

---

## Few-shot calibration

Example A (well-grounded, well-paced):
- Question: "What is prompt engineering?"
- Context: "[1] Prompt engineering is the practice of designing inputs to elicit desired outputs from LLMs."
- Answer: "Prompt engineering is the practice of designing inputs to get useful outputs from large language models, as the source describes [1]. For exam revision, think of it like writing very precise instructions to a smart-but-literal assistant."
- Scores: relevance=5, accuracy=5, citation_quality=4, accessibility=5, example_quality=5, learner_level_fit=5

Example B (fluent but ungrounded — should score low on grounding even though it reads well):
- Question: "What is prompt engineering?"
- Context: "[1] Prompt engineering is the practice of designing inputs to elicit desired outputs from LLMs."
- Answer: "Prompt engineering was pioneered by OpenAI in 2020 and forms the basis of modern AI alignment."
- Scores: relevance=3, accuracy=1, citation_quality=1, accessibility=4, example_quality=1, learner_level_fit=3

Example C (correct but over-jargon — accessibility must call this out independently):
- Question: "What is BM25?"
- Context: "[1] BM25 is a lexical retrieval method that ranks documents by term frequency and inverse document frequency."
- Answer: "BM25 is an Okapi-family probabilistic ranking function leveraging IDF-weighted TF saturation with document-length normalisation."
- Scores: relevance=5, accuracy=4, citation_quality=2, accessibility=1, example_quality=1, learner_level_fit=2

Example D (good examples but lazy on grounding — pedagogy & grounding must diverge):
- Question: "What is BM25?"
- Context: "[1] BM25 is a lexical retrieval method that ranks documents by term frequency and inverse document frequency."
- Answer: "BM25 is like a librarian who finds books by counting how often your keywords appear and how rare those keywords are across the whole shelf — think 'cat' vs 'the'. It's the classic keyword search algorithm."
- Scores: relevance=5, accuracy=4, citation_quality=1, accessibility=5, example_quality=5, learner_level_fit=5

---

## Output format

Output **only** valid JSON, nothing else (no markdown fences, no prose preamble):

{{
    "relevance": <1-5>,
    "accuracy": <1-5>,
    "citation_quality": <1-5>,
    "accessibility": <1-5>,
    "example_quality": <1-5>,
    "learner_level_fit": <1-5>,
    "reasoning": "<one or two sentences citing the weakest dimension>"
}}
"""


def expected_context(chunks: list[dict]) -> str:
    """The source block, rebuilt independently of ``app.agent.prompt``.

    Empty input still uses the fixed no-sources line; a non-empty one rebuilds
    each source line from this file's own literal, so the layout is decided here
    rather than imported.
    """
    if not chunks:
        return TUTOR_EMPTY_CONTEXT_GOLDEN
    return "\n\n".join(
        TUTOR_SOURCE_LINE_GOLDEN.format(
            index=index, source=chunk["source"], page=chunk["page"], content=chunk["content"]
        )
        for index, chunk in enumerate(chunks, start=1)
    )


def expected_tutor_prompt(chunks: list[dict]) -> str:
    """The exact user content the Tutor must put on the wire for ``chunks``."""
    return (
        f"{TUTOR_SYSTEM_INSTRUCTION_GOLDEN}\n\n"
        f"Sources:\n{expected_context(chunks)}\n\n"
        f"Question: {SYNTHETIC_QUESTION}"
    )


def expected_judge_prompt(*, answer: str, context: str) -> str:
    """The exact user content the production Judge must put on the wire.

    ``context`` stays a caller-supplied string; the callers below pass the
    independently rebuilt :func:`expected_context`, never the production
    formatter's own output.
    """
    return JUDGE_RUBRIC_GOLDEN.format(
        question=SYNTHETIC_QUESTION, answer=answer, context=context
    )


def expected_ping_messages() -> list[dict]:
    return [{"role": "user", "content": "ping"}]


def expected_tool_check_first_messages() -> list[dict]:
    return [{"role": "user", "content": PING_TOOL_PROMPT}]


def expected_ping_tool_schema() -> list[dict]:
    """The one zero-argument `ping` function schema the probe declares.

    The locked adapter emits this JSON Schema without a ``required`` key at
    all -- the empty-argument tool is expressed by an empty ``properties``
    object, and no ``strict`` flag is added.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": PING_TOOL_NAME,
                "description": "Respond with 'pong' when called.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def expected_tool_call_wire(*, name: str = PING_TOOL_NAME, arguments: str = "{}") -> dict:
    return {
        "id": PING_TOOL_CALL_ID,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def expected_second_round_messages() -> list[dict]:
    """Round two history: the user turn, the native assistant tool call, pong.

    This is the semantic shape the probe is required to produce. The
    assistant turn is carried forward by the handler as the *original*
    ``AIMessage`` object, so it keeps its own aligned tool call and adds
    nothing of its own.
    """
    return [
        {"role": "user", "content": PING_TOOL_PROMPT},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [expected_tool_call_wire()],
        },
        {
            "role": "tool",
            "content": "pong",
            "tool_call_id": PING_TOOL_CALL_ID,
        },
    ]


# ---------------------------------------------------------------------------
# Request assertions.
# ---------------------------------------------------------------------------


# Sentinel meaning "this request must carry no ``tools`` key at all". It is
# deliberately not ``None``: ``None`` would read as "do not check tools", and
# every request in this file is checked on this branch.
EXPECT_NO_TOOLS: Any = object()


def assert_openai_request(
    request: httpx.Request,
    *,
    model: str,
    streaming: bool | None = None,
    expected_messages: list[dict] | None = None,
    expect_tools: Any = EXPECT_NO_TOOLS,
) -> dict:
    """Real request shape, the declared model, and the complete prompt.

    ``streaming`` is explicit per case so no case silently inherits the
    adapter's default branch.

    ``expect_tools`` defaults to the plain branch and then *requires* that no
    ``tools`` key was sent. A bound-model case must pass the schema it expects
    explicitly, so "plain" and "bound but unchecked" cannot be confused.
    """
    assert request.method == "POST"
    assert request.url.scheme == "https"
    assert request.url.host == "api.openai.com"
    assert request.url.path == "/v1/chat/completions"
    assert "api-key" not in request.headers
    assert "x-goog-api-key" not in request.headers
    body = request_json(request)
    assert body["model"] == model
    if streaming is True:
        assert body.get("stream") is True
    elif streaming is False:
        assert body.get("stream") is not True
    if expected_messages is not None:
        assert body["messages"] == expected_messages
    if expect_tools is EXPECT_NO_TOOLS:
        assert "tools" not in body, "a plain request must carry no tools key"
    else:
        assert body.get("tools") == expect_tools
    return body


def assert_second_round_messages(body: dict) -> None:
    """Check the complete second-round history, field by field.

    Every field is required exactly, except the nested
    ``function.arguments`` *text*, which is compared as decoded JSON so the
    SDK's own whitespace normalisation cannot decide the outcome. Nothing is
    skipped or defaulted: the message count and order, the roles, the native
    ``None`` assistant content, the call id/type/name, the ``pong`` tool body
    and its aligned ``tool_call_id`` are all still required.

    Plain (non tool-call) message lists are still compared with exact
    equality by ``assert_openai_request``; only this nested JSON field is
    given the semantic treatment.
    """
    expected = expected_second_round_messages()
    messages = body["messages"]

    assert len(messages) == len(expected) == 3, "strict second-round message count"
    assert [m["role"] for m in messages] == [e["role"] for e in expected]

    # The user turn and the pong turn are exact.
    assert messages[0] == expected[0]
    assert messages[2] == expected[2]

    actual_assistant = messages[1]
    expected_assistant = expected[1]
    assert actual_assistant.keys() == expected_assistant.keys()
    assert actual_assistant["role"] == expected_assistant["role"]
    assert actual_assistant["content"] is None, "the native None content is required"

    actual_calls = actual_assistant["tool_calls"]
    expected_calls = expected_assistant["tool_calls"]
    assert len(actual_calls) == len(expected_calls) == 1, "strict tool-call count"
    for actual_call, expected_call in zip(actual_calls, expected_calls):
        assert actual_call.keys() == expected_call.keys()
        assert actual_call["type"] == expected_call["type"]
        assert actual_call["id"] == expected_call["id"]
        assert actual_call["function"].keys() == expected_call["function"].keys()
        assert actual_call["function"]["name"] == expected_call["function"]["name"]
        # Only the JSON text's whitespace may differ; its meaning may not.
        assert json.loads(actual_call["function"]["arguments"]) == json.loads(
            expected_call["function"]["arguments"]
        ) == {}


def split_tutor_and_judge(offline, tutor_responder, judge_responder):
    """One harness, two real models; the stream flag selects the responder.

    Built from the same harness so the two real models can never overwrite
    each other's transport seam.
    """

    def responder(request: httpx.Request) -> httpx.Response:
        if request_json(request).get("stream"):
            return tutor_responder(request)
        return judge_responder(request)

    offline.respond(responder)


def assert_no_marker(payload: Any, marker: str, *, where: str) -> None:
    assert marker not in json.dumps(payload, default=str, ensure_ascii=False), (
        f"{where} leaked a provider-private marker"
    )


# ---------------------------------------------------------------------------
# SSE consumption helpers (direct iterator entry).
# ---------------------------------------------------------------------------


def parse_sse_frame(raw: Any) -> dict:
    """Parse one ``data:`` frame this file's own handler produced.

    Framing is stripped explicitly, so a parse failure can only mean the
    payload itself was not legal JSON.
    """
    text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
    assert isinstance(text, str)
    assert text.startswith("data: "), f"not a data frame: {text[:40]!r}"
    assert text.endswith("\n\n"), f"frame is not SSE-terminated: {text[-8:]!r}"
    return json.loads(text[len("data: ") : -2])


def parse_sse_block(block: str) -> dict:
    """Parse one already-separated SSE block of an assembled HTTP body.

    Used only where ``ASGITransport`` handed back the whole buffered body and
    the frame terminator was consumed by the split. The payload itself is
    still framed by the handler; only the trailing blank line is gone.
    """
    assert block.startswith("data: "), f"not a data block: {block[:40]!r}"
    return json.loads(block[len("data: ") :])


async def drain_events(response) -> list[dict]:
    events: list[dict] = []
    async for raw in response.body_iterator:
        events.append(parse_sse_frame(raw))
    return events


def event_types(events: list[dict]) -> list[str]:
    return [event["type"] for event in events]


def joined_tokens(events: list[dict]) -> str:
    return "".join(e["text"] for e in events if e.get("type") == "token")


def first_index(events: list[dict], event_type: str) -> int:
    for index, event in enumerate(events):
        if event.get("type") == event_type:
            return index
    raise AssertionError(f"no {event_type!r} event in {event_types(events)}")


# ---------------------------------------------------------------------------
# Fixtures: a real temporary SQLite file per case.
#
# `make_engine` is called with the tmp_path URL only; the module-global
# `_engine`/`_SessionLocal` are never touched, and no migration is run. The
# schema comes from `Base.metadata.create_all`, which is NOT Alembic
# migration evidence.
# ---------------------------------------------------------------------------


@pytest.fixture
def api_db(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'd4b4.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    session = factory()
    try:
        yield engine, factory, session
    finally:
        session.close()
        engine.dispose()


def seed_user_and_document(session: Session) -> str:
    """Create the one synthetic User/Document pair the Tutor path requires."""
    user = UserRepository(session).get_or_create(SYNTHETIC_FINGERPRINT)
    DocumentRepository(session).create(
        user_id=user.id,
        filename="d4b4-synthetic.pdf",
        hash_="d4b4-synthetic-hash",
        chunks_count=1,
    )
    return user.id


def real_chat_models() -> tuple[LLMConfig, Any, dict]:
    """The real DI chain: config -> tutor model -> judge dependencies.

    A different ``judge_model`` is declared in the configuration, so the real
    ``get_judge_dependencies`` builds a genuinely different Judge model and
    ``same_model`` is False on its own. No factory is patched or forced.
    """
    config = LLMConfig(
        provider="openai",
        model=TUTOR_MODEL,
        api_key=SYNTHETIC_OPENAI_KEY,
        judge_model=JUDGE_MODEL,
    )
    tutor_llm = deps.get_llm(config)
    judge = deps.get_judge_dependencies(config)
    assert judge["same_model"] is False, "a distinct judge model is required here"
    return config, tutor_llm, judge


async def call_real_chat(session: Session, user_id: str, *, graph, judge):
    """Invoke the real ``routes.chat`` handler with every dependency explicit.

    The quiz/planner collaborators are ``None`` and the modes are
    ``deterministic``: this is a pure Tutor turn, so those agents are never
    constructed and never run. The memory hydrator and writer are the real
    ``deps`` builders bound to this case's session.
    """
    return await routes.chat(
        body=routes.ChatRequest(message=SYNTHETIC_QUESTION),
        user_id=user_id,
        session=session,
        graph=graph,
        judge=judge,
        quiz_master=None,
        planner=None,
        planner_agent=None,
        planner_mode="deterministic",
        quiz_master_agent=None,
        quiz_mode="deterministic",
        memory_hydrator=deps.get_memory_hydrator(session),
        memory_writer=deps.get_memory_writer(session),
    )


def fresh_session(factory) -> Session:
    """A brand-new Session, used only to prove a commit is visible outside."""
    return factory()


def read_session_rows(factory, session_id: str) -> tuple[list[Message], list[Citation]]:
    with fresh_session(factory) as probe:
        messages = list(
            probe.execute(
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(Message.created_at.asc())
            )
            .scalars()
        )
        message_ids = [m.id for m in messages]
        citations = (
            list(
                probe.execute(
                    select(Citation).where(Citation.message_id.in_(message_ids))
                )
                .scalars()
            )
            if message_ids
            else []
        )
        return messages, citations


def build_probe_app() -> FastAPI:
    """A bare app registering only the real probe handlers.

    The production ``create_app`` factory is never called. The real handler
    objects are registered directly, with the same paths, methods and
    response models the product declares, so the handler body, its
    ``get_llm_config`` dependency and the SDK call underneath are all real.
    """
    app = FastAPI()
    app.add_api_route(
        "/api/models/ping",
        routes.ping_model,
        methods=["GET"],
        response_model=routes.PingOut,
    )
    app.add_api_route(
        "/api/models/tool-check",
        routes.tool_check,
        methods=["GET"],
        response_model=routes.ToolCapableOut,
    )
    return app


def probe_headers() -> dict[str, str]:
    """Synthetic BYOK headers parsed by the real ``get_llm_config``."""
    return {
        "x-provider": "openai",
        "x-model": TUTOR_MODEL,
        "x-api-key": SYNTHETIC_OPENAI_KEY,
    }


def probe_client(app: FastAPI) -> httpx.AsyncClient:
    """An in-process ASGI client. No socket, no TCP, no lifespan."""
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=True),
        base_url="http://testserver",
        trust_env=False,
    )


# ===========================================================================
# A1: Tutor streams two text segments and ends normally; Judge passes.
# Entry: direct iterator over the real response.body_iterator.
# Layout: 1 Tutor HTTP (stream) + 1 Judge HTTP (non-stream), both on gpt-4o*.
# ===========================================================================


async def test_A1_chat_tutor_stream_and_judge_pass(api_db, offline_openai):
    """A1: full SSE, committed rows and the real graph state all agree."""
    engine, factory, session = api_db
    user_id = seed_user_and_document(session)
    checkpointer = InMemorySaver()

    tutor_frames = [
        stream_delta_frame(model=TUTOR_MODEL, text="Reciprocal "),
        stream_delta_frame(model=TUTOR_MODEL, text="rank fusion."),
    ]
    answer = "Reciprocal rank fusion."
    split_tutor_and_judge(
        offline_openai,
        lambda request: offline_openai.sse_response(
            offline_openai.sse_body(tutor_frames)
        ),
        json_responder(completion_body(model=JUDGE_MODEL, text=judge_pass_payload())),
    )
    config, tutor_llm, judge = real_chat_models()
    graph = deps.get_graph(
        retriever=SyntheticRetriever([SYNTHETIC_CHUNK_A]),
        llm=tutor_llm,
        checkpointer=checkpointer,
    )

    response = await call_real_chat(session, user_id, graph=graph, judge=judge)
    assert response.media_type == "text/event-stream"

    events: list[dict] = []
    while True:
        try:
            raw = await anext(response.body_iterator)
        except StopAsyncIteration:
            break
        events.append(parse_sse_frame(raw))
        if events[-1].get("type") == "done":
            # The DB proof is taken at the moment ``done`` was delivered, on a
            # brand-new Session that never saw the handler's own session.
            session_id = events[0]["session_id"]
            messages, citations = read_session_rows(factory, session_id)
            break

    # Exactly one done, and the body really is exhausted after it.
    assert event_types(events).count("done") == 1
    with pytest.raises(StopAsyncIteration):
        await anext(response.body_iterator)

    # Frame order: session first, citations before the first token.
    assert events[0]["type"] == "session"
    assert first_index(events, "citations") < first_index(events, "token")
    assert joined_tokens(events) == answer

    # Two logical invokes, one each; no HTTP replay, no extra retry.
    assert len(offline_openai.requests) == 2
    assert_openai_request(
        offline_openai.requests[0],
        model=TUTOR_MODEL,
        streaming=True,
        expected_messages=[
            {"role": "user", "content": expected_tutor_prompt([SYNTHETIC_CHUNK_A])}
        ],
    )
    assert_openai_request(
        offline_openai.requests[1],
        model=JUDGE_MODEL,
        streaming=False,
        expected_messages=[
            {
                "role": "user",
                "content": expected_judge_prompt(
                    answer=answer,
                    context=expected_context([SYNTHETIC_CHUNK_A]),
                ),
            }
        ],
    )
    # The provider-completed stream carries exactly one upstream [DONE],
    # asserted separately from the downstream SSE ``done`` above.
    assert offline_openai.streams[0].bytes_sent == sum(
        len(frame) for frame in tutor_frames
    ) + len(DONE_EVENT)

    # Committed rows, seen only through a fresh Session.
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[0].content == SYNTHETIC_QUESTION
    assert messages[1].content == answer
    artifacts = messages[1].tool_calls_json
    citations_event = events[first_index(events, "citations")]
    assert artifacts["schema"] == ARTIFACTS_SCHEMA
    assert artifacts["citations"] == citations_event["citations"]
    assert artifacts["citations"] == [
        {
            "chunk_id": SYNTHETIC_CHUNK_A["chunk_id"],
            "source": SYNTHETIC_CHUNK_A["source"],
            "page": SYNTHETIC_CHUNK_A["page"],
            "span_start": 0,
            "span_end": len(SYNTHETIC_CHUNK_A["content"]),
        }
    ]
    # The real Citation row must be attributed to the assistant message and
    # cover the full retrieved span. The query above spans *every* message id
    # in the session, user included, so a row wrongly hung on the user message
    # is still counted here and fails these checks.
    assert len(citations) == 1, "exactly one persisted citation"
    assert citations[0].message_id == messages[1].id
    assert citations[0].chunk_id == SYNTHETIC_CHUNK_A["chunk_id"]
    assert citations[0].page == SYNTHETIC_CHUNK_A["page"]
    assert citations[0].span_start == 0
    assert citations[0].span_end == len(SYNTHETIC_CHUNK_A["content"])

    # The real in-memory checkpointer holds this thread's final state.
    final = await graph.aget_state({"configurable": {"thread_id": session_id}})
    ai_messages = [m for m in final.values["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) == 1
    assert ai_messages[0].content == answer
    assert final.values["judge_status"] == "valid"
    assert final.values["judge_score"] == 0.8
    assert final.values["retry_count"] == 0

    # The same row is readable through the real repository helper too.
    with fresh_session(factory) as probe:
        owned = ChatSessionRepository(probe).get_for_user(
            chat_id=session_id, user_id=user_id
        )
        assert owned is not None and owned.id == session_id
        assert len(MessageRepository(probe).list_by_session(session_id)) == 2


# ===========================================================================
# A2: a legal empty-text stream (real frames, real [DONE], no text body).
# Entry: direct iterator.
# Layout: 1 Tutor HTTP only; the Judge is never reached.
# ===========================================================================


async def test_A2_chat_legal_empty_text_stream_raises_empty_content(api_db, offline_openai):
    """A2: the narrow ValueError surfaces; nothing is fabricated or persisted."""
    engine, factory, session = api_db
    user_id = seed_user_and_document(session)

    frames = [stream_delta_frame(model=TUTOR_MODEL, text="")]
    split_tutor_and_judge(
        offline_openai,
        lambda request: offline_openai.sse_response(
            offline_openai.sse_body(frames)
        ),
        json_responder(completion_body(model=JUDGE_MODEL, text=judge_pass_payload())),
    )
    config, tutor_llm, judge = real_chat_models()
    graph = deps.get_graph(
        retriever=SyntheticRetriever([SYNTHETIC_CHUNK_A]),
        llm=tutor_llm,
        checkpointer=InMemorySaver(),
    )

    response = await call_real_chat(session, user_id, graph=graph, judge=judge)

    collected: list[dict] = []
    with pytest.raises(ValueError) as excinfo:
        while True:
            try:
                raw = await anext(response.body_iterator)
            except StopAsyncIteration:
                break
            collected.append(parse_sse_frame(raw))

    assert type(excinfo.value) is ValueError
    assert str(excinfo.value) == EMPTY_CONTENT_MESSAGE

    # A legal empty stream really did arrive and really did terminate.
    assert len(offline_openai.requests) == 1
    assert offline_openai.streams[0].bytes_sent == sum(len(f) for f in frames) + len(
        DONE_EVENT
    )

    # No Judge call, no fabricated body, no done.
    assert joined_tokens(collected) == ""
    assert "done" not in event_types(collected)
    assert request_json(offline_openai.requests[0]).get("stream") is True

    # The session and the user message committed before the graph ran; the
    # assistant message and its citations never did.
    assert collected[0]["type"] == "session"
    session_id = collected[0]["session_id"]
    messages, citations = read_session_rows(factory, session_id)
    assert [m.role for m in messages] == ["user"]
    assert messages[0].content == SYNTHETIC_QUESTION
    assert citations == []


# ===========================================================================
# A3: a delivered prefix, then the body fails with a raw HTTPX class.
# Entry: direct iterator. Layout: 1 Tutor HTTP, never replayed; 0 Judge HTTP.
# ===========================================================================


@pytest.mark.parametrize(
    "exc_cls,label",
    [
        pytest.param(httpx.ReadError, "read-error", id="readerror"),
        pytest.param(httpx.ReadTimeout, "read-timeout", id="readtimeout"),
    ],
)
async def test_A3_chat_prefix_then_body_failure_propagates_raw_httpx(
    api_db, offline_openai, exc_cls, label
):
    """A3: the original HTTPX class reaches the consumer; prefix is kept."""
    engine, factory, session = api_db
    user_id = seed_user_and_document(session)

    injected = exc_cls(f"synthetic {label} mid body")
    body = CloseObservingBody(
        [
            send(stream_delta_frame(model=TUTOR_MODEL, text="Partial ")),
            fail_with(injected),
        ],
        name="a3-body",
    )
    offline_openai.respond(lambda request: offline_openai.sse_response(body))
    config, tutor_llm, judge = real_chat_models()
    graph = deps.get_graph(
        retriever=SyntheticRetriever([SYNTHETIC_CHUNK_A]),
        llm=tutor_llm,
        checkpointer=InMemorySaver(),
    )

    response = await call_real_chat(session, user_id, graph=graph, judge=judge)

    collected: list[dict] = []
    with pytest.raises(exc_cls) as excinfo:
        while True:
            try:
                raw = await anext(response.body_iterator)
            except StopAsyncIteration:
                break
            collected.append(parse_sse_frame(raw))

    assert type(excinfo.value) is exc_cls, "the exact HTTPX class must survive"
    assert excinfo.value is injected, "the captured object is the injected one"

    # The prefix that was already delivered stays delivered.
    assert joined_tokens(collected) == "Partial "
    assert "done" not in event_types(collected)

    # One Tutor request, not replayed; the Judge was never reached.
    assert len(offline_openai.requests) == 1
    assert_openai_request(
        offline_openai.requests[0],
        model=TUTOR_MODEL,
        streaming=True,
        expected_messages=[
            {"role": "user", "content": expected_tutor_prompt([SYNTHETIC_CHUNK_A])}
        ],
    )

    session_id = collected[0]["session_id"]
    messages, citations = read_session_rows(factory, session_id)
    assert [m.role for m in messages] == ["user"]
    assert citations == []

    # Observed here: before this case's cleanup and before the fixture closes
    # the SDK clients. Recorded as a value, never asserted as a product rule --
    # it proves nothing about immediacy, connection release or billing.
    print(f"D4b4 A3[{label}] pre_cleanup_close_count={body.close_count}")


# ===========================================================================
# A4: prefix consumed, the SDK body parked on a real gate, explicit cancel.
# Entry: direct iterator. Layout: 1 Tutor HTTP; 0 Judge HTTP.
# ===========================================================================


async def test_A4_chat_cancel_after_prefix_keeps_prefix_and_closes(api_db, offline_openai):
    """A4: cancellation returns CancelledError; the committed rows are unchanged."""
    engine, factory, session = api_db
    user_id = seed_user_and_document(session)

    gate = ObservedGate()  # never set: the body stays open until cancellation
    body = CloseObservingBody(
        [
            send(stream_delta_frame(model=TUTOR_MODEL, text="Prefix ")),
            wait_for(gate),
            send(stream_delta_frame(model=TUTOR_MODEL, text="late")),
        ],
        name="a4-body",
    )
    offline_openai.respond(lambda request: offline_openai.sse_response(body))
    config, tutor_llm, judge = real_chat_models()
    graph = deps.get_graph(
        retriever=SyntheticRetriever([SYNTHETIC_CHUNK_A]),
        llm=tutor_llm,
        checkpointer=InMemorySaver(),
    )

    response = await call_real_chat(session, user_id, graph=graph, judge=judge)
    collected: list[dict] = []
    declared_prefix = "Prefix "
    prefix_ready = asyncio.Event()

    async def consume() -> None:
        async for raw in response.body_iterator:
            event = parse_sse_frame(raw)
            collected.append(event)
            # The consumer-side gate. It is set here and only here: after this
            # case's own parse-and-append of a real token frame, and only once
            # the declared prefix has been fully collected. The body gate, the
            # responder, this test's body and the ``finally`` never set it.
            if event.get("type") == "token" and joined_tokens(collected) == declared_prefix:
                prefix_ready.set()

    task = asyncio.create_task(consume())
    try:
        # Two independent, separately bounded gates. ``prefix_ready`` proves
        # the consumer really parsed and collected the prefix; the body's
        # ``waiting_event`` only proves the SDK body parked on its wait, which
        # by itself does not mean the Graph custom queue was consumed. Either
        # wait timing out fails the case.
        await asyncio.wait_for(prefix_ready.wait(), timeout=5)
        await asyncio.wait_for(gate.waiting_event.wait(), timeout=5)
        assert joined_tokens(collected) == declared_prefix
        assert "done" not in event_types(collected)

        # Fresh Session before cancelling: the session and the user message
        # were committed before the graph was ever entered.
        session_id = collected[0]["session_id"]
        messages, citations = read_session_rows(factory, session_id)
        assert [m.role for m in messages] == ["user"]
        assert citations == []

        task.cancel()
        # This wait carries its own deadline, so a cancellation that never
        # returns fails the case instead of hanging to the 180s process limit.
        # ``shield`` keeps that local timeout from cancelling the consumer
        # task a second time; the explicit cancel above is the only one.
        with pytest.raises(asyncio.CancelledError) as excinfo:
            await asyncio.wait_for(asyncio.shield(task), timeout=5)
        assert type(excinfo.value) is asyncio.CancelledError
        assert task.cancelled()

        # The inner stream task's cleanup may finish shortly after the handler
        # unwinds; the event is set by the body's own aclose, never by this
        # test, its ``finally``, or the fixture.
        await asyncio.wait_for(body.closed_event.wait(), timeout=5)
        observed_close_count = body.close_count
        assert isinstance(observed_close_count, int)

        # Nothing was invented after the cancellation.
        assert joined_tokens(collected) == declared_prefix
        assert "done" not in event_types(collected)

        messages_after, citations_after = read_session_rows(factory, session_id)
        assert [m.role for m in messages_after] == ["user"]
        assert citations_after == []

        assert len(offline_openai.requests) == 1
        assert_openai_request(
            offline_openai.requests[0],
            model=TUTOR_MODEL,
            streaming=True,
            expected_messages=[
                {
                    "role": "user",
                    "content": expected_tutor_prompt([SYNTHETIC_CHUNK_A]),
                }
            ],
        )

        # Observed before the ``finally`` and before the fixture closes the
        # SDK clients. Recorded only; it is not immediacy, connection-release
        # or billing evidence, and it is not promoted to a product rule.
        print(
            f"D4b4 A4 pre_cleanup_close_count={observed_close_count} "
            f"gate_wait_entered={gate.waiting_event.is_set()} "
            f"consumer_prefix_ready={prefix_ready.is_set()}"
        )
    finally:
        await reclaim_own_task(task)


# ===========================================================================
# A5: the user has no Document at all.
# Entry: direct iterator. Layout: 0 SDK requests, 0 retrieval.
# ===========================================================================


async def test_A5_chat_without_documents_never_calls_the_model(api_db, offline_openai):
    """A5: the fixed no-documents answer, with no provider request at all."""
    engine, factory, session = api_db
    user_id = seed_user_and_document(session)
    # Remove the only Document so the no-sources branch is the real one.
    documents = DocumentRepository(session).list_for_user(user_id)
    for document in documents:
        session.delete(document)
    session.commit()

    retriever = SyntheticRetriever([SYNTHETIC_CHUNK_A])
    offline_openai.respond(
        lambda request: pytest.fail("no SDK request may happen in A5")
    )
    config, tutor_llm, judge = real_chat_models()
    graph = deps.get_graph(
        retriever=retriever, llm=tutor_llm, checkpointer=InMemorySaver()
    )

    response = await call_real_chat(session, user_id, graph=graph, judge=judge)
    events = await drain_events(response)

    assert event_types(events) == ["session", "citations", "token", "done"]
    assert events[1]["citations"] == []
    assert joined_tokens(events) == routes._NO_DOCUMENTS_MESSAGE
    assert retriever.calls == [], "no retrieval happened"

    # Zero SDK requests and zero retrieval: the branch never reached a model.
    assert offline_openai.requests == []
    assert offline_openai.streams == []

    session_id = events[0]["session_id"]
    messages, citations = read_session_rows(factory, session_id)
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[0].content == SYNTHETIC_QUESTION
    assert messages[1].content == routes._NO_DOCUMENTS_MESSAGE
    assert messages[1].tool_calls_json["citations"] == []
    assert citations == []


# ===========================================================================
# A6: the Tutor succeeds; the Judge exhausts its retries on 503.
# Entry: direct iterator. Layout: 1 Tutor + 3 Judge HTTP attempts.
# ===========================================================================


async def test_A6_chat_judge_503_keeps_content_with_fixed_suffix(api_db, offline_openai):
    """A6: kept content + fixed suffix, natural graph end, no private marker."""
    engine, factory, session = api_db
    user_id = seed_user_and_document(session)

    tutor_frames = [stream_delta_frame(model=TUTOR_MODEL, text="Reciprocal rank fusion.")]
    answer = "Reciprocal rank fusion."
    split_tutor_and_judge(
        offline_openai,
        lambda request: offline_openai.sse_response(
            offline_openai.sse_body(tutor_frames)
        ),
        provider_error_responder(503, SYNTHETIC_ERROR_MARKER),
    )
    config, tutor_llm, judge = real_chat_models()
    graph = deps.get_graph(
        retriever=SyntheticRetriever([SYNTHETIC_CHUNK_A]),
        llm=tutor_llm,
        checkpointer=InMemorySaver(),
    )

    response = await call_real_chat(session, user_id, graph=graph, judge=judge)
    events = await drain_events(response)

    # 1 Tutor + 3 Judge attempts: the real SDK retried the 5xx to exhaustion.
    assert len(offline_openai.requests) == 4
    assert_openai_request(
        offline_openai.requests[0],
        model=TUTOR_MODEL,
        streaming=True,
        expected_messages=[
            {"role": "user", "content": expected_tutor_prompt([SYNTHETIC_CHUNK_A])}
        ],
    )
    expected_judge = expected_judge_prompt(
        answer=answer, context=expected_context([SYNTHETIC_CHUNK_A])
    )
    judge_bodies = [
        assert_openai_request(
            request,
            model=JUDGE_MODEL,
            streaming=False,
            expected_messages=[{"role": "user", "content": expected_judge}],
        )
        for request in offline_openai.requests[1:]
    ]
    # Every Judge attempt carried byte-identical parsed input, retries included.
    assert judge_bodies[1] == judge_bodies[0]
    assert judge_bodies[2] == judge_bodies[0]

    # The kept content is followed by the fixed production suffix, once.
    expected_body = answer + _UNAVAILABLE_SUFFIX
    assert joined_tokens(events) == expected_body
    assert answer in joined_tokens(events), "the original answer is kept verbatim"
    assert event_types(events).count("done") == 1

    # The graph ended naturally, so the assistant row and citations exist.
    session_id = events[0]["session_id"]
    messages, citations = read_session_rows(factory, session_id)
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[1].content == expected_body
    assert [c.chunk_id for c in citations] == [SYNTHETIC_CHUNK_A["chunk_id"]]

    final = await graph.aget_state({"configurable": {"thread_id": session_id}})
    assert final.values["judge_status"] == "unavailable"
    assert final.values["degraded"] is True
    assert final.values["judge_score"] is None
    ai_messages = [m for m in final.values["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) == 1
    assert ai_messages[0].content == expected_body

    # The provider-private error text never reaches any observable output.
    assert_no_marker(events, SYNTHETIC_ERROR_MARKER, where="the public SSE stream")
    assert_no_marker(
        messages[1].tool_calls_json, SYNTHETIC_ERROR_MARKER, where="the artifact"
    )
    assert_no_marker(
        dict(final.values), SYNTHETIC_ERROR_MARKER, where="the final graph state"
    )


# ===========================================================================
# A7: the same chat turn over a real HTTP POST on a bare app.
# Entry: HTTP ASGI. Proves routing/DI/assembly only.
# ===========================================================================


async def test_A7_chat_http_smoke_assembles_real_route_and_stream(api_db, offline_openai):
    """A7: HTTP 200 + real SSE body + committed rows, through real DI."""
    engine, factory, session = api_db
    user_id = seed_user_and_document(session)
    checkpointer = InMemorySaver()
    retriever = SyntheticRetriever([SYNTHETIC_CHUNK_A])

    tutor_frames = [
        stream_delta_frame(model=TUTOR_MODEL, text="Reciprocal "),
        stream_delta_frame(model=TUTOR_MODEL, text="rank fusion."),
    ]
    answer = "Reciprocal rank fusion."
    split_tutor_and_judge(
        offline_openai,
        lambda request: offline_openai.sse_response(
            offline_openai.sse_body(tutor_frames)
        ),
        json_responder(completion_body(model=JUDGE_MODEL, text=judge_pass_payload())),
    )

    # The real router, mounted on a bare app. No create_app, no middleware,
    # no auth: only the pieces this smoke is allowed to prove.
    app = FastAPI()
    app.include_router(routes.router)

    def override_session():
        yield session

    app.dependency_overrides[deps.get_session] = override_session
    app.dependency_overrides[deps.require_existing_user] = lambda: user_id
    app.dependency_overrides[deps.get_retriever] = lambda: retriever
    app.dependency_overrides[deps.get_checkpointer] = lambda: checkpointer
    # Unwalked collaborators for this pure-Tutor turn.
    for dependency in (
        deps.get_quiz_master,
        deps.get_planner,
        deps.get_planner_agent,
        deps.get_quiz_master_agent,
    ):
        app.dependency_overrides[dependency] = lambda: None

    headers = probe_headers() | {"x-judge-model": JUDGE_MODEL}

    async with probe_client(app) as client:
        response = await client.post(
            "/api/chat", json={"message": SYNTHETIC_QUESTION}, headers=headers
        )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    # ASGITransport buffers the whole body, so this list is the assembled
    # response and never incrementality or cancellation evidence.
    events = [
        parse_sse_block(block)
        for block in response.text.split("\n\n")
        if block.strip()
    ]
    assert events[0]["type"] == "session"
    assert event_types(events).count("done") == 1
    assert joined_tokens(events) == answer

    assert len(offline_openai.requests) == 2
    assert_openai_request(
        offline_openai.requests[0],
        model=TUTOR_MODEL,
        streaming=True,
        expected_messages=[
            {"role": "user", "content": expected_tutor_prompt([SYNTHETIC_CHUNK_A])}
        ],
    )
    assert_openai_request(
        offline_openai.requests[1],
        model=JUDGE_MODEL,
        streaming=False,
        expected_messages=[
            {
                "role": "user",
                "content": expected_judge_prompt(
                    answer=answer, context=expected_context([SYNTHETIC_CHUNK_A])
                ),
            }
        ],
    )
    assert retriever.calls == [(SYNTHETIC_QUESTION, 5)]

    # The HTTP response body and the committed rows say the same thing.
    session_id = events[0]["session_id"]
    messages, citations = read_session_rows(factory, session_id)
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[1].content == answer
    assert messages[1].tool_calls_json["citations"] == [
        events[first_index(events, "citations")]["citations"][0]
    ]
    assert [c.chunk_id for c in citations] == [SYNTHETIC_CHUNK_A["chunk_id"]]


# ===========================================================================
# P: the real ping handler over a bare app and real DI.
# Entry: HTTP ASGI.
# ===========================================================================


async def test_P1_ping_valid_text_succeeds_without_echoing_the_body(offline_openai):
    """P1: one plain request, a fixed success note, and no echoed body."""
    marker_body = f"Pong {SYNTHETIC_BODY_MARKER}"
    offline_openai.respond(
        json_responder(completion_body(model=TUTOR_MODEL, text=marker_body))
    )
    app = build_probe_app()

    async with probe_client(app) as client:
        response = await client.get("/api/models/ping", headers=probe_headers())

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["model"] == TUTOR_MODEL
    assert isinstance(payload["latency_ms"], (int, float))
    assert payload["latency_ms"] >= 0
    # The handler formats its own integer-rounded measurement into the note;
    # the response model widens the field itself to a float.
    assert payload["note"] == f"Connected — responded in {int(payload['latency_ms'])}ms"

    # The provider's own text never comes back to the client.
    assert SYNTHETIC_BODY_MARKER not in response.text

    assert len(offline_openai.requests) == 1
    assert_openai_request(
        offline_openai.requests[0],
        model=TUTOR_MODEL,
        streaming=False,
        expected_messages=expected_ping_messages(),
    )


async def test_P2_ping_empty_content_reports_the_fixed_no_text_note(offline_openai):
    """P2: a legal empty body is the fixed note, not a normal provider error."""
    offline_openai.respond(
        json_responder(completion_body(model=TUTOR_MODEL, text=""))
    )
    app = build_probe_app()

    async with probe_client(app) as client:
        response = await client.get("/api/models/ping", headers=probe_headers())

    payload = response.json()
    assert payload["ok"] is False
    assert payload["note"] == routes._PING_NO_TEXT_NOTE
    assert SYNTHETIC_BODY_MARKER not in response.text

    assert len(offline_openai.requests) == 1
    assert_openai_request(
        offline_openai.requests[0],
        model=TUTOR_MODEL,
        streaming=False,
        expected_messages=expected_ping_messages(),
    )


async def test_P3_ping_503_retries_and_normalizes_without_leaking(offline_openai):
    """P3: three real attempts, identical input, one safe fixed category."""
    offline_openai.respond(provider_error_responder(503, SYNTHETIC_ERROR_MARKER))
    app = build_probe_app()

    async with probe_client(app) as client:
        response = await client.get("/api/models/ping", headers=probe_headers())

    payload = response.json()
    assert payload["ok"] is False
    assert payload["model"] == TUTOR_MODEL
    assert payload["note"] == f"Failed: {GENERIC_LLM_ERROR_TEXT}"
    # The provider's private error text never leaves the boundary.
    assert SYNTHETIC_ERROR_MARKER not in response.text

    # Three attempts, each carrying the complete ping input. Real backoff
    # timing is explicitly not measured here.
    assert len(offline_openai.requests) == 3
    bodies = [
        assert_openai_request(
            request,
            model=TUTOR_MODEL,
            streaming=False,
            expected_messages=expected_ping_messages(),
        )
        for request in offline_openai.requests
    ]
    assert bodies[1] == bodies[0]
    assert bodies[2] == bodies[0]


# ===========================================================================
# K: the real tool-check handler over a bare app and real DI.
# Entry: HTTP ASGI.
# ===========================================================================


async def test_K1_tool_check_completes_one_legal_round_trip(offline_openai):
    """K1: one ping call, one pong, one aligned second round."""
    tool_calls = [expected_tool_call_wire()]
    calls: list[int] = []

    def responder(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(
                200,
                json=completion_body(
                    model=TUTOR_MODEL, tool_calls=tool_calls, finish_reason="tool_calls"
                ),
            )
        return httpx.Response(
            200, json=completion_body(model=TUTOR_MODEL, text="pong acknowledged")
        )

    offline_openai.respond(responder)
    app = build_probe_app()

    async with probe_client(app) as client:
        response = await client.get(
            "/api/models/tool-check", headers=probe_headers()
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["tool_capable"] is True
    assert payload["model"] == TUTOR_MODEL
    assert payload["note"] == routes._TOOL_CHECK_OK_NOTE

    # Two logical invokes: the round trip, not an SDK retry. Both requests are
    # checked in full: method/route/model/non-stream/messages/tool schema.
    assert len(offline_openai.requests) == 2
    first_body = assert_openai_request(
        offline_openai.requests[0],
        model=TUTOR_MODEL,
        streaming=False,
        expected_messages=expected_tool_check_first_messages(),
        expect_tools=expected_ping_tool_schema(),
    )
    second_body = assert_openai_request(
        offline_openai.requests[1],
        model=TUTOR_MODEL,
        streaming=False,
        expect_tools=expected_ping_tool_schema(),
    )

    # Nothing forces a tool choice or marks the schema strict. The declared
    # single zero-argument ping schema itself was checked on both requests
    # above, so it is not re-derived here.
    assert "tool_choice" not in first_body
    assert "tool_choice" not in second_body
    assert "strict" not in first_body["tools"][0]["function"]
    assert "strict" not in second_body["tools"][0]["function"]

    # Second round history: user, the native assistant tool call, the pong.
    assert_second_round_messages(second_body)
    assert second_body["messages"][2]["tool_call_id"] == PING_TOOL_CALL_ID


async def test_K2_tool_check_reports_false_when_no_calls_were_observed(offline_openai):
    """K2: a valid body with no tool calls is 'not observed this time'."""
    offline_openai.respond(
        json_responder(
            completion_body(model=TUTOR_MODEL, text=f"No tools needed {SYNTHETIC_BODY_MARKER}")
        )
    )
    app = build_probe_app()

    async with probe_client(app) as client:
        response = await client.get("/api/models/tool-check", headers=probe_headers())

    payload = response.json()
    assert payload["tool_capable"] is False
    assert payload["note"] == routes._TOOL_CHECK_NO_CALLS_NOTE
    assert SYNTHETIC_BODY_MARKER not in response.text

    # No calls observed, so the second round was never attempted. The one
    # request is checked in full, including its bound tool schema.
    assert len(offline_openai.requests) == 1
    assert_openai_request(
        offline_openai.requests[0],
        model=TUTOR_MODEL,
        streaming=False,
        expected_messages=expected_tool_check_first_messages(),
        expect_tools=expected_ping_tool_schema(),
    )


@pytest.mark.parametrize(
    "label,sensitive,first_body_factory",
    [
        pytest.param(
            "unknown-tool-name",
            "not_a_known_tool",
            lambda: completion_body(
                model=TUTOR_MODEL,
                tool_calls=[expected_tool_call_wire(name="not_a_known_tool")],
                finish_reason="tool_calls",
            ),
            id="unknown-tool-name",
        ),
        pytest.param(
            "malformed-arguments",
            "{not-json",
            lambda: completion_body(
                model=TUTOR_MODEL,
                tool_calls=[expected_tool_call_wire(arguments="{not-json")],
                finish_reason="tool_calls",
            ),
            id="malformed-arguments",
        ),
    ],
)
async def test_K3_tool_check_unknown_first_round_stays_incomplete(
    offline_openai, label, sensitive, first_body_factory
):
    """K3: an unusable first round is incomplete; no second call, no execution."""

    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=first_body_factory())

    offline_openai.respond(responder)
    app = build_probe_app()

    async with probe_client(app) as client:
        response = await client.get("/api/models/tool-check", headers=probe_headers())

    payload = response.json()
    assert payload["tool_capable"] is None
    assert payload["note"] == routes._TOOL_CHECK_INCOMPLETE_NOTE
    # Leak check on what this case *actually* injected: the unknown tool name
    # or the malformed argument text. SYNTHETIC_BODY_MARKER is not part of
    # either scripted body, so its absence proves nothing here.
    assert sensitive not in response.text

    # Exactly one request: the unknown shape never reaches a second round, and
    # no unknown tool was ever executed. It is still checked in full.
    assert len(offline_openai.requests) == 1
    assert_openai_request(
        offline_openai.requests[0],
        model=TUTOR_MODEL,
        streaming=False,
        expected_messages=expected_tool_check_first_messages(),
        expect_tools=expected_ping_tool_schema(),
    )


async def test_K4_tool_check_second_round_empty_body_stays_incomplete(offline_openai):
    """K4: a legal ping, then an empty second body: incomplete, tool ran once."""
    calls: list[int] = []

    def responder(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(
                200,
                json=completion_body(
                    model=TUTOR_MODEL,
                    tool_calls=[expected_tool_call_wire()],
                    finish_reason="tool_calls",
                ),
            )
        return httpx.Response(200, json=completion_body(model=TUTOR_MODEL, text="  "))

    offline_openai.respond(responder)
    app = build_probe_app()

    async with probe_client(app) as client:
        response = await client.get("/api/models/tool-check", headers=probe_headers())

    payload = response.json()
    assert payload["tool_capable"] is None
    assert payload["note"] == routes._TOOL_CHECK_INCOMPLETE_NOTE

    # Both requests are checked in full, the first one included: before the
    # repair it carried no method/route/model/non-stream/prompt/schema check.
    assert len(offline_openai.requests) == 2
    assert_openai_request(
        offline_openai.requests[0],
        model=TUTOR_MODEL,
        streaming=False,
        expected_messages=expected_tool_check_first_messages(),
        expect_tools=expected_ping_tool_schema(),
    )
    second_body = assert_openai_request(
        offline_openai.requests[1],
        model=TUTOR_MODEL,
        streaming=False,
        expect_tools=expected_ping_tool_schema(),
    )
    # The full second-round history is still checked even on failure.
    assert_second_round_messages(second_body)


async def test_K5_tool_check_second_round_503_exhausts_retries(offline_openai):
    """K5: one ping plus three second-round attempts, all with the full history."""
    attempts: list[int] = []

    def responder(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(
                200,
                json=completion_body(
                    model=TUTOR_MODEL,
                    tool_calls=[expected_tool_call_wire()],
                    finish_reason="tool_calls",
                ),
            )
        return httpx.Response(503, json=provider_error_body(SYNTHETIC_ERROR_MARKER))

    offline_openai.respond(responder)
    app = build_probe_app()

    async with probe_client(app) as client:
        response = await client.get("/api/models/tool-check", headers=probe_headers())

    payload = response.json()
    assert payload["tool_capable"] is None
    assert payload["note"] == f"Tool check failed: {GENERIC_LLM_ERROR_TEXT}"
    assert SYNTHETIC_ERROR_MARKER not in response.text

    # Four HTTP attempts total: the one ping plus the three retried failures.
    # "≤2 model calls" counts logical invokes and is never mixed with the SDK's
    # HTTP retries. Every attempt, the first included, is checked in full.
    assert len(offline_openai.requests) == 4
    assert_openai_request(
        offline_openai.requests[0],
        model=TUTOR_MODEL,
        streaming=False,
        expected_messages=expected_tool_check_first_messages(),
        expect_tools=expected_ping_tool_schema(),
    )
    retried = [
        assert_openai_request(
            request,
            model=TUTOR_MODEL,
            streaming=False,
            expect_tools=expected_ping_tool_schema(),
        )
        for request in offline_openai.requests[1:]
    ]
    for body in retried:
        assert_second_round_messages(body)
    # Each retried attempt carried byte-identical parsed input, retries
    # included; headers are deliberately not compared.
    assert retried[1] == retried[0]
    assert retried[2] == retried[0]


# ===========================================================================
# Source identity: exact resolved equality for this checkout and this venv.
# ===========================================================================


def venv_site_packages() -> Path:
    """The install root of *this* interpreter's site-packages.

    Derived from the running interpreter's own installation info, never from
    the location of a module under test. The expected ``purelib`` is also
    cross-checked against ``sys.prefix`` so a wrong interpreter is caught
    instead of silently redefining the target.
    """
    import sys

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


def test_Z1_source_identity_is_exact_for_this_checkout():
    """Z1: this checkout's modules and this venv's locked libraries."""
    import fastapi
    import langchain_openai.chat_models.base as openai_adapter
    import starlette
    import openai as openai_sdk

    import app.agent.graph as graph_module
    import app.agent.tutor_attempt as tutor_module
    import app.llm.provider as provider_module
    import tests.llm.sdk_transport_support as helper_module

    # The backend root comes from this file's own location, so no absolute
    # home path is hard-coded anywhere in this check.
    backend_root = Path(__file__).resolve().parents[2]
    assert backend_root.name == "backend"

    assert_module_from_backend(routes, "app/api/routes.py")
    assert_module_from_backend(deps, "app/api/deps.py")
    assert_module_from_backend(graph_module, "app/agent/graph.py")
    assert_module_from_backend(tutor_module, "app/agent/tutor_attempt.py")
    assert_module_from_backend(provider_module, "app/llm/provider.py")
    assert_module_from_backend(helper_module, "tests/llm/sdk_transport_support.py")
    assert_module_from_backend(
        sys.modules[__name__], "tests/api/test_routes_sdk_stream.py"
    )

    # The locked SDKs and the HTTP/adapter stack, pinned to this venv's
    # site-packages -- exact equality, never a prefix match.
    sdk_root = venv_site_packages()
    assert Path(openai_sdk.__file__).resolve() == sdk_root / "openai" / "__init__.py"
    assert Path(openai_adapter.__file__).resolve() == (
        sdk_root / "langchain_openai" / "chat_models" / "base.py"
    )
    assert Path(httpx.__file__).resolve() == sdk_root / "httpx" / "__init__.py"
    assert Path(fastapi.__file__).resolve() == sdk_root / "fastapi" / "__init__.py"
    assert Path(starlette.__file__).resolve() == (
        sdk_root / "starlette" / "__init__.py"
    )


# ---------------------------------------------------------------------------
# Prompt-oracle mechanism checks (P2).
#
# IMPORTANT: these three cases are NOT SDK, provider or route evidence. They are
# pure test logic over synthetic httpx requests: no route is called, no model is
# invoked, no production file is edited and no factory is patched. The real SDK
# route cases earlier in this file remain the healthy execution path.
#
# Each case proves exactly two things, in this order:
#   1. this file's own literal expectations ACCEPT the healthy text the real
#      production renderer / rubric loader / context formatter produce today.
#      Those three production entry points are imported here and only here, on
#      the healthy side: the candidate never comes from an oracle literal.
#   2. the same complete comparator -- assert_openai_request, including its
#      request-shape, model, streaming and tools guards -- REJECTS an in-memory
#      mutant that differs only at that one instruction/context marker, with the
#      role, the model and every other request field left untouched.
#
# Three green meta cases are not proof that the literals are independent and are
# not offered as such. The ping and tool-check oracles are already test-owned
# literals and are deliberately left alone by this section.
# ---------------------------------------------------------------------------

_ORACLE_MUTANT = " SYNTHETIC-ORACLE-MUTANT"

_API_ORACLE_ANSWER = "Reciprocal rank fusion."

_API_ORACLE_CASES = [
    ("A-TUTOR", TUTOR_SYSTEM_INSTRUCTION_GOLDEN),
    (
        "A-JUDGE",
        "You are an impartial Study Coach judge. "
        "Your job: rate a Tutor's answer to a student question.",
    ),
    ("A-CONTEXT", "[1] d4b4-synthetic-retrieval.pdf p.1:"),
]


@pytest.mark.parametrize("case_id,marker", _API_ORACLE_CASES)
def test_oracle_api_prompt_accepts_healthy_and_rejects_mutant(case_id, marker):
    """Healthy accepted, in-memory mutant rejected, by this file's own literals."""
    # Imported here and nowhere else: the healthy candidate side only.
    from app.agent.judge import load_tutor_rubric
    from app.agent.prompt import TutorPromptTemplate, format_context

    answer = _API_ORACLE_ANSWER
    # The real production render of this chunk list, and this file's own
    # independent rebuild of the very same thing.
    production_context = format_context([SYNTHETIC_CHUNK_A])
    rebuilt_context = expected_context([SYNTHETIC_CHUNK_A])
    rubric = load_tutor_rubric()

    if case_id == "A-TUTOR":
        healthy = TutorPromptTemplate.production_v2().render(
            SYNTHETIC_QUESTION, [SYNTHETIC_CHUNK_A]
        )
        expected = expected_tutor_prompt([SYNTHETIC_CHUNK_A])
        model, streaming = TUTOR_MODEL, True
    else:
        if case_id == "A-JUDGE":
            healthy = rubric.format(
                question=SYNTHETIC_QUESTION, answer=answer, context=production_context
            )
        else:  # A-CONTEXT: same judge prompt, only the source layout is in question
            healthy = rubric.format(
                question=SYNTHETIC_QUESTION, answer=answer, context=production_context
            )
        expected = expected_judge_prompt(answer=answer, context=rebuilt_context)
        model, streaming = JUDGE_MODEL, False

    # Where the marker must be unique, and therefore what the mutant changes.
    carrier = production_context if case_id == "A-CONTEXT" else healthy
    assert carrier.count(marker) == 1, f"{case_id}: marker must be unique in the candidate"
    mutant_carrier = carrier.replace(marker, marker + _ORACLE_MUTANT, 1)
    assert mutant_carrier != carrier, f"{case_id}: the mutant must differ from the candidate"

    if case_id == "A-CONTEXT":
        mutant = rubric.format(
            question=SYNTHETIC_QUESTION, answer=answer, context=mutant_carrier
        )
    else:
        mutant = mutant_carrier

    def request_for(prompt):
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        if streaming:
            payload["stream"] = True
        return httpx.Request(
            "POST", "https://api.openai.com/v1/chat/completions", json=payload
        )

    def compare(prompt):
        assert_openai_request(
            request_for(prompt),
            model=model,
            streaming=streaming,
            expected_messages=[{"role": "user", "content": expected}],
        )

    # 1. The healthy production text passes this file's own complete comparator.
    compare(healthy)

    # 3. Only the comparator call is inside the expected rejection.
    with pytest.raises(AssertionError):
        compare(mutant)
