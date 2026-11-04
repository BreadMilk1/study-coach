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
    reclaim_own_task,
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
# Expected prompt bodies -- INDEPENDENT oracle.
#
# These expectations deliberately do NOT share a source with the code under
# test. They are built from the two explicit golden constants registered below,
# plus this file's own synthetic fixture values. Nothing here calls
# ``TutorPromptTemplate.render``, ``TutorPromptTemplate.production_v2``,
# ``load_tutor_rubric`` or ``format_context``, and nothing is read from the
# production template or rubric file at import time. If the production
# template, the rubric text or the formatter changes, these expectations stay
# put and the comparison below fails -- which is the whole point of an anchor.
#
# The production builders remain in use for the *actual* side (the real consumer
# under test) at their original call sites; only the expectation side changed.
# ---------------------------------------------------------------------------

# Verbatim baseline of ``app.agent.prompt.SYSTEM_INSTRUCTION`` as of the D4
# accepted state. Registered here as a literal on purpose.
TUTOR_SYSTEM_INSTRUCTION_GOLDEN = (
    "You are a study coach answering questions based ONLY on the provided sources. "
    "Cite each fact you use with [N] referring to the source list. "
    "If the sources do not contain the answer, say you don't know — do not fabricate."
)

# Verbatim baseline of the source-list line format, and of the text used when a
# retrieval returns nothing. Both are literals, not calls into production.
TUTOR_SOURCE_LINE_GOLDEN = "[{index}] {source} p.{page}: {content}"
TUTOR_EMPTY_CONTEXT_GOLDEN = "(no relevant sources retrieved)"

# Verbatim baseline of ``app/agent/prompts/judge_tutor.txt`` as of the D4
# accepted state, INCLUDING every few-shot example (A, B, C and D) and the
# escaped braces of the JSON output block. It is formatted with the standard
# library ``str.format`` against this file's own test data below.
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
}}\n"""


def expected_tutor_prompt(chunks: list[dict]) -> str:
    """The exact user content the Tutor must put on the wire for ``chunks``.

    Assembled from the literal goldens above and this file's own synthetic
    chunk inputs. This intentionally re-derives the layout instead of calling
    the production renderer, so a production template drift cannot silently
    move the expectation with it.
    """
    if not chunks:
        source_block = TUTOR_EMPTY_CONTEXT_GOLDEN
    else:
        source_block = "\n\n".join(
            TUTOR_SOURCE_LINE_GOLDEN.format(
                index=index,
                source=chunk["source"],
                page=chunk["page"],
                content=chunk["content"],
            )
            for index, chunk in enumerate(chunks, start=1)
        )
    return (
        f"{TUTOR_SYSTEM_INSTRUCTION_GOLDEN}\n\n"
        f"Sources:\n{source_block}\n\n"
        f"Question: {SYNTHETIC_QUESTION}"
    )


def expected_judge_prompt(*, answer: str, context: str) -> str:
    """The exact user content the production Judge must put on the wire."""
    return JUDGE_RUBRIC_GOLDEN.format(
        question=SYNTHETIC_QUESTION, answer=answer, context=context
    )


# The Judge's inputs are fixed by ``_run_judge`` below, so the expected rubric
# prompt is fixed too and every J case (including every retry) checks against it.
# ``JUDGE_CONTEXT`` is the literal rendering of the single chunk the Judge sees,
# written out here rather than produced by the production ``format_context``.
JUDGE_ANSWER = "Synthetic answer body."
JUDGE_CONTEXT = (
    "[1] synthetic-retrieval.pdf p.1: Synthetic note about reciprocal rank fusion."
)
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
            answer=candidate["content"], context=JUDGE_CONTEXT
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
            answer=body, context=JUDGE_CONTEXT
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


# ---------------------------------------------------------------------------
# Pure-asyncio mechanism check for the shared `reclaim_own_task` this file now
# imports instead of its own deleted duplicate.
#
# IMPORTANT: this case is **not** consumer or SDK evidence. It uses plain asyncio
# primitives only -- no provider, no SDK transport, no socket, no database. Its
# only subject is the bounded cleanup contract every scenario above relies on
# from its `finally` block, and it neither replaces nor weakens any of them.
# ---------------------------------------------------------------------------


async def test_meta_reclaim_own_task_fails_fast_on_a_permanent_cancellation_swallow():
    """A task that swallows cancellation forever must fail fast, not hang.

    The old file-local duplicate bounded itself with ``asyncio.wait_for(task,
    timeout=...)``. On a target that refuses to die, ``wait_for`` cancels and
    then awaits that target *unbounded*, so the cleanup -- and with it the whole
    pytest process -- would hang until the 180s process limit. The shared helper
    instead uses ``asyncio.wait``, which returns on its own deadline whatever
    the target does, and fails with a precise ``AssertionError``.

    The proof is deliberately external to the helper: the cleanup runs as its own
    task and is watched by an *independent* ``asyncio.wait`` with a far larger
    budget. That observer never cancels anything, so it cannot manufacture the
    cancellation under test, and it cannot rescue the helper either -- if the
    helper regressed to a waiting-for-doom cleanup, the observer returns with the
    cleanup still pending and the assertion below fails instead of hanging.
    """
    START_BUDGET = 5.0
    RECLAIM_WINDOW = 0.2  # the helper's own short window, this case only
    OBSERVE_BUDGET = 5.0  # must be far larger than RECLAIM_WINDOW
    FINAL_BUDGET = 5.0

    started = asyncio.Event()
    swallowed = asyncio.Event()
    # Never set inside the body: it is the test's own gate, and releasing it is
    # the only way this stubborn task can ever finish.
    own_gate = asyncio.Event()

    async def stubborn():
        started.set()
        while True:
            try:
                await own_gate.wait()
            except asyncio.CancelledError:
                swallowed.set()  # deliberately keep ignoring cancellations
                continue
            return

    victim = asyncio.create_task(stubborn())
    cleanup = None
    try:
        await asyncio.wait_for(started.wait(), timeout=START_BUDGET)

        cleanup = asyncio.create_task(reclaim_own_task(victim, timeout=RECLAIM_WINDOW))

        done, pending = await asyncio.wait({cleanup}, timeout=OBSERVE_BUDGET)
        assert not pending, (
            "reclaim_own_task must fail its bounded window instead of waiting "
            "for a task that never settles"
        )
        assert cleanup in done

        exc = cleanup.exception()
        assert type(exc) is AssertionError, (
            f"a bounded-cleanup failure must be exactly AssertionError, got {exc!r}"
        )
        assert str(exc) == "test-owned task did not settle within the bounded cleanup window"

        # The failure is genuinely about the stubborn task, not an early exit:
        # it really was cancelled, really swallowed that, and is still running.
        await asyncio.wait_for(swallowed.wait(), timeout=START_BUDGET)
        assert victim.cancelling() == 1, "exactly one cancellation was requested"
        assert not victim.done(), "the stubborn task must still be pending"
        assert victim.cancelled() is False
    finally:
        # Releasing our own gate is what lets the stubborn task finish. Both
        # test-owned tasks are then reclaimed under one more bounded wait, so
        # this case cannot leave a pending task behind for pytest teardown even
        # when an assertion above failed.
        own_gate.set()
        targets = {t for t in (victim, cleanup) if t is not None}
        _done, still_pending = await asyncio.wait(targets, timeout=FINAL_BUDGET)

    # Asserted after the finally so a body failure is never masked by cleanup.
    assert not still_pending, (
        f"test-owned tasks must all settle in finally; still pending: {still_pending}"
    )
    assert victim.done() and victim.cancelled() is False, (
        "releasing the gate must let the stubborn task finish normally"
    )
    assert cleanup is not None and cleanup.done()


# ---------------------------------------------------------------------------
# Oracle mechanism checks: the independent expectation must actually bite.
#
# These two cases are pure test-logic checks. They use no provider, no SDK
# transport and no network; they only prove that the golden expectations
# registered above are independent anchors rather than a restatement of
# whatever the production side happens to produce.
#
# Each one does both halves:
#   * the HEALTHY baseline must be accepted, and
#   * a MUTANT of the production side must be rejected by the very same
#     existing full-prompt comparer the real cases use.
# The mutants are built in memory only. No production file is written, no
# loader is permanently changed, and a mutant is never used to generate an
# expected value.
# ---------------------------------------------------------------------------


def _openai_request_with_user_content(content: str, *, stream: bool) -> httpx.Request:
    return httpx.Request(
        "POST",
        "https://api.openai.com/v1/chat/completions",
        json={"model": OPENAI_MODEL, "messages": [{"role": "user", "content": content}],
              **({"stream": True} if stream else {})},
    )


def test_oracle_judge_rubric_mutant_losing_example_d_is_rejected():
    """Dropping Example D from the Judge rubric must fail the full comparison.

    The mutant keeps every ``str.format`` field intact, so it is still a legal
    rubric; only the calibration text shrinks. A fragment-level check would not
    notice, which is exactly the regression this anchors.
    """
    # Health: the production rubric text and the independent golden agree.
    assert load_tutor_rubric() == JUDGE_RUBRIC_GOLDEN, (
        "the registered golden must still describe the accepted rubric baseline"
    )
    healthy = _openai_request_with_user_content(EXPECTED_JUDGE_PROMPT, stream=False)
    assert_openai_chat_request(healthy, streaming=False, expected_prompt=EXPECTED_JUDGE_PROMPT)

    # Mutant: Example D removed in memory, every format field preserved.
    example_d_start = JUDGE_RUBRIC_GOLDEN.index("Example D (good examples")
    example_d_end = JUDGE_RUBRIC_GOLDEN.index("\n\n---\n\n## Output format")
    mutant_rubric = (
        JUDGE_RUBRIC_GOLDEN[:example_d_start] + JUDGE_RUBRIC_GOLDEN[example_d_end:]
    )
    mutant_prompt = mutant_rubric.format(
        question=SYNTHETIC_QUESTION, answer=JUDGE_ANSWER, context=JUDGE_CONTEXT
    )
    assert "{question}" not in mutant_prompt and "{context}" not in mutant_prompt
    assert mutant_prompt != EXPECTED_JUDGE_PROMPT, "the mutant must really differ"

    mutant = _openai_request_with_user_content(mutant_prompt, stream=False)
    with pytest.raises(AssertionError):
        assert_openai_chat_request(mutant, streaming=False, expected_prompt=EXPECTED_JUDGE_PROMPT)


def test_oracle_empty_evidence_tutor_mutant_without_system_instruction_is_rejected():
    """A Tutor prompt stripped of its system instruction must be rejected.

    The empty-evidence branch is the one that had no independent anchor, so the
    golden is compared both against the healthy production render and against a
    mutant produced by removing the system instruction in memory.
    """
    healthy_prompt = expected_tutor_prompt([])
    assert healthy_prompt.startswith(TUTOR_SYSTEM_INSTRUCTION_GOLDEN)
    assert TUTOR_EMPTY_CONTEXT_GOLDEN in healthy_prompt
    # Health: the independent expectation reproduces the production render.
    assert TutorPromptTemplate.production_v2().render(SYNTHETIC_QUESTION, []) == healthy_prompt
    healthy = _openai_request_with_user_content(healthy_prompt, stream=True)
    assert_openai_chat_request(healthy, streaming=True, expected_prompt=healthy_prompt)

    # Mutant: the real render with its system instruction removed in memory.
    production_render = TutorPromptTemplate.production_v2().render(SYNTHETIC_QUESTION, [])
    mutant_prompt = production_render.replace(TUTOR_SYSTEM_INSTRUCTION_GOLDEN, "", 1)
    assert mutant_prompt != healthy_prompt, "the mutant must really differ"
    assert TUTOR_EMPTY_CONTEXT_GOLDEN in mutant_prompt

    mutant = _openai_request_with_user_content(mutant_prompt, stream=True)
    with pytest.raises(AssertionError):
        assert_openai_chat_request(mutant, streaming=True, expected_prompt=healthy_prompt)


# ---------------------------------------------------------------------------
# D5B-1: the OpenAI Judge's completion-signal contract on the real SDK.
#
# This is a characterization of EXISTING behaviour, not a production repair
# and not a RED-to-GREEN claim. The legal rubric body stays valid when the
# provider's ``choices[0].finish_reason`` is absent, null or an unknown
# value; only a known negative completion signal ("length") turns the very
# same legal body into an unavailable result.
#
# The wire is the ONLY thing that changes between the four cases: every other
# field of the fresh ``openai_message_body(judge_payload())`` -- including
# the usage block -- is byte-identical, so any difference in the public
# result can only have come from ``choices[0].finish_reason``.
#
# A "valid" result here means "the payload was legal". It is NOT evidence
# that the provider really completed anything, and the SDK callback below
# only OBSERVES the real AIMessage -- it never manufactures a value and
# never mutates the response.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case_id,expected_finish_reason,expected_result",
    [
        pytest.param(
            "missing",
            None,
            {
                "status": "valid",
                "score": 0.8,
                "verdict": "pass",
                "weak_dims": [],
                "reasoning": SYNTHETIC_REASONING_MARKER,
            },
            id="missing",
        ),
        pytest.param(
            "null",
            None,
            {
                "status": "valid",
                "score": 0.8,
                "verdict": "pass",
                "weak_dims": [],
                "reasoning": SYNTHETIC_REASONING_MARKER,
            },
            id="null",
        ),
        pytest.param(
            "unknown",
            "d5b_unknown",
            {
                "status": "valid",
                "score": 0.8,
                "verdict": "pass",
                "weak_dims": [],
                "reasoning": SYNTHETIC_REASONING_MARKER,
            },
            id="unknown",
        ),
        pytest.param(
            "length",
            "length",
            {
                "status": "unavailable",
                "score": None,
                "verdict": None,
                "reason": "Judge response signalled an incomplete or blocked completion.",
            },
            id="length",
        ),
    ],
)
async def test_d5b_openai_judge_completion_contract(
    offline_openai, case_id, expected_finish_reason, expected_result
):
    """The Judge's public result follows ``choices[0].finish_reason`` only.

    Four bodies, one per completion-signal shape, each built fresh from the
    same legal rubric payload. The expected outcomes are literals fixed by
    the accepted contract; they are never derived from what the SDK returned.
    """
    from langchain_core.callbacks import AsyncCallbackHandler

    captured: list = []

    class Capture(AsyncCallbackHandler):
        """Local observer of the real SDK message; it changes nothing."""

        async def on_llm_end(self, response, **kwargs):
            captured.append(response.generations[0][0].message)

    payload_text = judge_payload()
    body = openai_message_body(payload_text)
    choice = body["choices"][0]

    # The single difference between the four bodies: choices[0].finish_reason.
    if case_id == "missing":
        choice.pop("finish_reason")
    elif case_id == "null":
        choice["finish_reason"] = None
    elif case_id == "unknown":
        choice["finish_reason"] = "d5b_unknown"
    else:
        choice["finish_reason"] = "length"

    # The original wire really distinguishes an absent key from a null value.
    if case_id == "missing":
        assert "finish_reason" not in choice, "the key is absent, not null"
    else:
        assert "finish_reason" in choice, "the key is present"
    if case_id == "null":
        assert choice["finish_reason"] is None, "an explicit null, not an absence"

    # Everything else about the body is the untouched original.
    assert body["id"] == "chatcmpl-synthetic"
    assert body["object"] == "chat.completion"
    assert body["created"] == 0
    assert body["model"] == OPENAI_MODEL
    assert body["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }
    assert choice["index"] == 0
    assert choice["logprobs"] is None
    assert choice["message"] == {
        "role": "assistant",
        "refusal": None,
        "content": payload_text,
    }

    offline_openai.respond(json_responder(body))
    model = offline_openai.model(callbacks=[Capture()])

    result = await _run_judge(model)

    # Exactly one real request on the wire, carrying the exact rubric prompt.
    assert len(offline_openai.requests) == 1
    assert_judge_requests(offline_openai.requests, streaming=True)

    # Exactly one real AIMessage, observed but never fabricated.
    assert len(captured) == 1
    message = captured[0]
    assert isinstance(message, AIMessage)
    assert message.content == payload_text
    assert "finish_reason" in message.response_metadata
    assert message.response_metadata["finish_reason"] == expected_finish_reason

    # The public result, compared against the predetermined literal outcome.
    assert result == expected_result


# ---------------------------------------------------------------------------
# D5B-2: the OpenAI Tutor's EOF / completion-signal contract on the real SDK.
#
# This is a characterization of EXISTING behaviour, not a production repair and
# not a RED-to-GREEN claim. The consumer keeps producing the same complete
# answer whether the stream ends with a [DONE], with an explicit
# ``finish_reason``, or with a plain transport EOF that carries neither.
#
# The wire is the ONLY thing that differs between the five cases: two fixed
# body frames, one fresh empty-delta terminal frame and one unchanged usage
# frame. Only the terminal frame's ``finish_reason`` differs, and in the two
# "absent" cases the key is DELETED (an absent key, never an explicit null) --
# which is what the raw streaming wire really distinguishes.
#
# A "completed" trace here is the local iteration/body completing, NOT evidence
# that the provider completed anything. The ``length`` case documents the
# consumer's current behaviour; it is not a new policy endorsement.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case_id,expected_finish_reason",
    [
        pytest.param("stop", "stop", id="stop"),
        pytest.param("missing", None, id="missing"),
        pytest.param("unknown", "d5b_unknown", id="unknown"),
        pytest.param("length", "length", id="length"),
        pytest.param("bare_eof", None, id="bare_eof"),
    ],
)
async def test_d5b_openai_tutor_eof_completion_contract(
    offline_openai, case_id, expected_finish_reason
):
    """The Tutor's answer follows the body, not the completion signal.

    Five scripted streams over one real model call: the same two body frames,
    the same usage frame and one terminal frame whose ``finish_reason`` is
    the only difference. ``bare_eof`` is the odd one out: it ends the HTTP
    body with a plain EOF, so the stream carries neither a finish key nor a
    [DONE].
    """
    from langchain_core.callbacks import AsyncCallbackHandler
    from langchain_core.messages import AIMessageChunk

    captured: list = []

    class Capture(AsyncCallbackHandler):
        """Local observer of the real SDK chunks; it changes nothing."""

        async def on_llm_end(self, response, **kwargs):
            captured.append(response.generations[0][0].message)

    # The terminal frame is built fresh per case and is the only frame that
    # differs. In the two "absent" cases the key is deleted, not set to null.
    if case_id in ("missing", "bare_eof"):
        terminal_payload = json.loads(openai_text_chunk(""))
        terminal_payload["choices"][0].pop("finish_reason")
        terminal = json.dumps(terminal_payload)
    else:
        terminal = openai_text_chunk("", finish_reason=expected_finish_reason)

    frames = openai_sse(
        openai_text_chunk("Reciprocal "),
        openai_text_chunk("rank fusion."),
        terminal,
        openai_usage_chunk(),
    )

    # The terminal frame on the wire really carries this case's shape.
    terminal_choice = sse_frame_json(frames[2])["choices"][0]
    if case_id in ("missing", "bare_eof"):
        assert "finish_reason" not in terminal_choice, "absent key, not null"
    else:
        assert terminal_choice["finish_reason"] == expected_finish_reason
    assert terminal_choice["delta"] == {}, "the terminal frame is an empty delta"

    # Every other frame and field is identical across the five cases.
    assert sse_frame_json(frames[0])["choices"][0]["delta"] == {"content": "Reciprocal "}
    assert sse_frame_json(frames[1])["choices"][0]["delta"] == {"content": "rank fusion."}
    assert sse_frame_json(frames[3])["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }
    assert DONE_EVENT not in b"".join(frames), "frames must not pre-terminate"

    if case_id == "bare_eof":
        # A normal HTTP body EOF: no finish key and no [DONE] at all.
        body = offline_openai.sse_body(frames, terminator=None)
    else:
        body = offline_openai.sse_body(frames, terminator=DONE_EVENT)
    offline_openai.respond(lambda request: offline_openai.sse_response(body))

    model = offline_openai.model(callbacks=[Capture()])
    retriever = SyntheticRetriever([SYNTHETIC_CHUNK_A])
    sink = RecordingEventSink()

    candidate = await run_tutor(offline_openai, model, retriever=retriever, sink=sink)

    # Exactly one real request on the wire, carrying the independent golden
    # prompt (the file's own literal expectation, not a production render).
    assert len(offline_openai.requests) == 1
    assert_openai_chat_request(
        offline_openai.requests[0],
        streaming=True,
        expected_prompt=expected_tutor_prompt([SYNTHETIC_CHUNK_A]),
    )

    # Exactly one recorded stream, with exactly the expected bytes delivered:
    # the frames plus one [DONE] for the four terminated cases, and the frames
    # alone for the bare EOF.
    assert len(offline_openai.streams) == 1
    if case_id == "bare_eof":
        assert offline_openai.streams[0].bytes_sent == sum(
            len(frame) for frame in frames
        )
    else:
        assert_single_done(offline_openai.streams[0], frames)

    # The same fixed consumer outcome in all five cases: the body decided the
    # answer, the completion signal did not.
    assert candidate.answer == "Reciprocal rank fusion."
    assert sink.tokens() == ["Reciprocal ", "rank fusion."]
    assert retriever.calls == [(SYNTHETIC_QUESTION, 5)]
    assert candidate.evidence == [SYNTHETIC_CHUNK_A]
    assert candidate.usage == {
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
    }

    traces = sink.traces()
    assert candidate.trace == sink.traces()
    assert [t["step"] for t in traces] == [
        "retrieval",
        "retrieval",
        "generation",
        "generation",
    ]
    assert traces[1]["status"] == "completed"
    assert traces[3]["status"] == "completed"
    assert traces[3]["answer_length"] == len(candidate.answer)
    assert traces[3]["usage"] == candidate.usage
    assert not any(t.get("status") == "failed" for t in traces)

    # Exactly one captured real AIMessageChunk, observed but never fabricated.
    assert len(captured) == 1
    message = captured[0]
    assert isinstance(message, AIMessageChunk)
    assert message.content == "Reciprocal rank fusion."
    if case_id in ("missing", "bare_eof"):
        # An absent key on the streaming wire is an absent key here too; it is
        # NOT the explicit null the non-streaming projection produces.
        assert "finish_reason" not in message.response_metadata, (
            "the key is absent, not null"
        )
    else:
        assert message.response_metadata["finish_reason"] == expected_finish_reason

    assert traces[0]["status"] == "started"
    assert traces[2]["status"] == "started"


# ---------------------------------------------------------------------------
# D5C-6: a real production_v1 Graph quality retry keeps each Tutor attempt's
# own usage separate.
#
# One real Graph turn: the Tutor streams, the production Judge rates the first
# candidate weak (2/5 on all six dimensions -> 0.40), the Graph retries the
# Tutor, and the second Judge rates the new candidate pass (4/5 -> 0.80). The
# subject is only the per-attempt usage the Graph really consumes: the real
# ``TutorAttemptEngine`` builds two real candidates, each carrying its own
# usage and its own completed-generation trace usage.
#
# This is NOT a whole-turn usage ledger. The Graph state and its stream carry
# no usage field, and nothing here sums the two attempts into a turn total --
# no such field exists and none is invented. Each attempt's usage is asserted
# in isolation, so the second attempt (7/3/10) is never allowed to arrive as
# the first plus the second (10/5/15).
#
# The delegating recorder is transparent: it holds a real engine, awaits it
# with the original keyword arguments untouched and returns the same candidate
# object. It manufactures no usage or trace and swallows no exception. The
# per-call responder serves the four scripted wires (Tutor stream, Judge
# non-stream, Tutor stream, Judge non-stream) in the order the Graph really
# drives them.
# ---------------------------------------------------------------------------


async def test_d5c6_graph_retry_keeps_tutor_attempt_usage_separate(offline_openai):
    from langchain_core.callbacks import AsyncCallbackHandler

    first_tutor_text = "First synthetic draft answer."
    second_tutor_text = "Second synthetic revised answer."

    # Independent, test-owned expectation of the production retry hint: the
    # verified previous score and the six fixed dimension names, never the
    # Judge's free reasoning, and never generated by a production helper.
    weak_dims_literal = (
        "relevance, accuracy, citation_quality, accessibility, "
        "example_quality, learner_level_fit"
    )
    retry_hint_literal = (
        "\n\n---\nPrevious attempt rating: previous_score=0.40 "
        f"(weak dimensions: {weak_dims_literal}). "
        "Re-read the sources carefully and produce a more grounded, well-paced answer "
        "for an exam-prep student. Address the weak dimensions specifically."
    )

    # --- Scripted wires, asserted before the run ---------------------------
    # Accepted OpenAI layout: full text frame(s), a terminal finish frame and a
    # usage-only frame; ``openai_sse`` leaves the single [DONE] to ``sse_body``.
    first_tutor_frames = openai_sse(
        openai_text_chunk(first_tutor_text),
        openai_text_chunk("", finish_reason="stop"),
        openai_usage_chunk(prompt=3, completion=2, total=5),
    )
    second_tutor_frames = openai_sse(
        openai_text_chunk(second_tutor_text),
        openai_text_chunk("", finish_reason="stop"),
        openai_usage_chunk(prompt=7, completion=3, total=10),
    )
    # The scripted usage/wire shapes, read straight off the frames.
    assert sse_frame_json(first_tutor_frames[0])["choices"][0]["delta"] == {
        "content": first_tutor_text
    }
    assert (
        sse_frame_json(first_tutor_frames[1])["choices"][0]["finish_reason"] == "stop"
    )
    assert sse_frame_json(first_tutor_frames[2])["choices"] == []
    assert sse_frame_json(first_tutor_frames[2])["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }
    assert sse_frame_json(second_tutor_frames[0])["choices"][0]["delta"] == {
        "content": second_tutor_text
    }
    assert (
        sse_frame_json(second_tutor_frames[1])["choices"][0]["finish_reason"] == "stop"
    )
    assert sse_frame_json(second_tutor_frames[2])["choices"] == []
    assert sse_frame_json(second_tutor_frames[2])["usage"] == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
    }

    # First Judge: all six dimensions at 2 -> 0.40 weak. Second Judge: all six
    # at 4 -> 0.80 pass. Both payloads are fixed literals, not run-derived.
    first_judge_body = openai_message_body(
        judge_payload(score=2, reasoning=SYNTHETIC_REASONING_MARKER)
    )
    second_judge_body = openai_message_body(judge_payload())
    assert json.loads(first_judge_body["choices"][0]["message"]["content"]) == {
        **{dim: 2 for dim in TUTOR_DIMENSIONS},
        "reasoning": SYNTHETIC_REASONING_MARKER,
    }
    assert json.loads(second_judge_body["choices"][0]["message"]["content"]) == {
        **{dim: 4 for dim in TUTOR_DIMENSIONS},
        "reasoning": SYNTHETIC_REASONING_MARKER,
    }

    # --- Per-call responder: two Tutor streams, two Judge bodies -----------
    counters = {"tutor": 0, "judge": 0}

    def per_call_responder(request: httpx.Request) -> httpx.Response:
        if request_json(request).get("stream") is True:
            index = counters["tutor"]
            counters["tutor"] += 1
            frames = (first_tutor_frames, second_tutor_frames)[index]
            return offline_openai.sse_response(offline_openai.sse_body(frames))
        index = counters["judge"]
        counters["judge"] += 1
        body = (first_judge_body, second_judge_body)[index]
        return json_responder(body)(request)

    # The independent observer of the real SDK aggregate: it changes nothing
    # and is attached only to the Tutor model, so the two Judge calls never
    # enter this capture.
    captured: list = []

    class TutorEndCapture(AsyncCallbackHandler):
        async def on_llm_end(self, response, **kwargs):
            captured.append(response.generations[0][0].message)

    class DelegatingTutorRecorder:
        """Transparent recorder around a real TutorAttemptEngine."""

        def __init__(self) -> None:
            self._delegate = TutorAttemptEngine()
            self.candidates: list = []

        async def answer(self, **kwargs):
            candidate = await self._delegate.answer(**kwargs)
            self.candidates.append(candidate)
            return candidate

    offline_openai.respond(per_call_responder)
    tutor_llm = offline_openai.model(callbacks=[TutorEndCapture()])
    judge_llm = offline_openai.model()
    recorder = DelegatingTutorRecorder()
    retriever = SyntheticRetriever([SYNTHETIC_CHUNK_A])
    writer = MemoryWriterSpy()
    graph = build_graph(
        retriever=retriever,
        llm=tutor_llm,
        tutor_attempt_engine=recorder,
        checkpointer=InMemorySaver(),
        judge_policy="production_v1",
    )

    observed, final_state = await _drive_graph(
        graph,
        thread_id="d5c6-retry-usage",
        messages=[HumanMessage(content=SYNTHETIC_QUESTION)],
        configurable={"judge_llm": judge_llm, "memory_writer": writer},
    )

    # --- Exactly four requests, in the fixed order, script exhausted -------
    assert len(offline_openai.requests) == 4, "two Tutor + two Judge calls only"
    assert counters == {"tutor": 2, "judge": 2}, "the script was consumed once"
    assert [
        request_json(r).get("stream") is True for r in offline_openai.requests
    ] == [True, False, True, False]
    assert_single_done(offline_openai.streams[0], first_tutor_frames)
    assert_single_done(offline_openai.streams[1], second_tutor_frames)

    # --- Request layer: full prompts, no reasoning leak --------------------
    base_tutor_prompt = expected_tutor_prompt([SYNTHETIC_CHUNK_A])
    assert_openai_chat_request(
        offline_openai.requests[0],
        streaming=True,
        expected_prompt=base_tutor_prompt,
    )
    assert_openai_chat_request(
        offline_openai.requests[2],
        streaming=True,
        expected_prompt=base_tutor_prompt + retry_hint_literal,
    )
    assert_openai_chat_request(
        offline_openai.requests[1],
        streaming=False,
        expected_prompt=expected_judge_prompt(
            answer=first_tutor_text, context=JUDGE_CONTEXT
        ),
    )
    assert_openai_chat_request(
        offline_openai.requests[3],
        streaming=False,
        expected_prompt=expected_judge_prompt(
            answer=second_tutor_text, context=JUDGE_CONTEXT
        ),
    )

    first_prompt = request_json(offline_openai.requests[0])["messages"][-1]["content"]
    second_prompt = request_json(offline_openai.requests[2])["messages"][-1]["content"]
    assert "Previous attempt rating" not in first_prompt, "no hint on attempt one"
    assert SYNTHETIC_REASONING_MARKER not in first_prompt
    assert "previous_score=0.40" in second_prompt
    assert weak_dims_literal in second_prompt
    assert SYNTHETIC_REASONING_MARKER not in second_prompt, (
        "the production retry hint never carries the Judge's free reasoning"
    )

    # --- Two real candidates, each with its own usage ----------------------
    candidates = recorder.candidates
    assert len(candidates) == 2, "exactly two real Tutor candidates, no third"
    assert candidates[0].answer == first_tutor_text
    assert candidates[1].answer == second_tutor_text
    assert candidates[0].answer != candidates[1].answer
    assert candidates[0].usage == {
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
    }
    assert candidates[1].usage == {
        "input_tokens": 7,
        "output_tokens": 3,
        "total_tokens": 10,
    }

    def completed_generation_usage(candidate) -> dict:
        matches = [
            event
            for event in candidate.trace
            if event.get("step") == "generation"
            and event.get("status") == "completed"
        ]
        assert len(matches) == 1, "exactly one completed generation trace"
        return matches[0]["usage"]

    assert completed_generation_usage(candidates[0]) == {
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
    }
    assert completed_generation_usage(candidates[1]) == {
        "input_tokens": 7,
        "output_tokens": 3,
        "total_tokens": 10,
    }

    # --- Independent callback: two real SDK aggregates, two fixed usages ---
    assert len(captured) == 2, "the Tutor model's callback saw both real attempts"
    assert captured[0].usage_metadata["input_tokens"] == 3
    assert captured[0].usage_metadata["output_tokens"] == 2
    assert captured[0].usage_metadata["total_tokens"] == 5
    assert captured[1].usage_metadata["input_tokens"] == 7
    assert captured[1].usage_metadata["output_tokens"] == 3
    assert captured[1].usage_metadata["total_tokens"] == 10

    # --- Real Graph state and observed responses ---------------------------
    observed_candidates = observed["candidates"]
    assert len(observed_candidates) == 2
    assert [c["content"] for c in observed_candidates] == [
        first_tutor_text,
        second_tutor_text,
    ]
    final_ai = [m for m in final_state.values["messages"] if isinstance(m, AIMessage)]
    assert len(final_ai) == 2
    assert final_ai[-1].content == candidates[1].answer
    assert final_state.values["retry_count"] == 1
    assert final_state.values["degraded"] is False
    assert final_state.values["judge_status"] == "valid"
    assert final_state.values["judge_score"] == 0.8
    assert writer.calls == 1, "the pass routes through the memory writer once"
