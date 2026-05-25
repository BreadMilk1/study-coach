"""End-to-end SSE tests proving routes.py goes through graph.astream_events.

These complement tests/api/test_routes.py (which lock the public SSE contract).
Together they prove:
- chat handler emits citations -> token -> done in order (contract preserved)
- chat handler routes through the multi-node graph (router intent observed)
- quiz / plan stubs surface their placeholder text via SSE token events
- P2.1-②: Judge Guard wired via config; same-model warning emitted exactly when
  x-judge-model is omitted AND the path is Tutor (real LLM output)
"""

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk

from app.main import create_app


class StubRetriever:
    def __init__(self):
        self.added: list[dict] = []
        self.search_returns: list[dict] = []
        self.last_query: str | None = None

    def add_chunks(self, chunks):
        self.added.extend(chunks)

    def search(self, query: str, top_k: int = 5):
        self.last_query = query
        return self.search_returns[:top_k]


class StubLLM:
    def __init__(self, tokens: list[str], quiz_json: str | None = None):
        self.tokens = tokens
        # ainvoke is used by generate_quiz (quiz GENERATE path). Default to an
        # empty array so non-quiz paths don't surprise the quiz parser.
        self.quiz_json = quiz_json or "[]"
        self.invoked: bool = False
        self.ainvoke_count = 0

    async def astream(self, messages, **_kwargs):
        self.invoked = True
        for t in self.tokens:
            yield AIMessageChunk(content=t)

    def invoke(self, messages, **_kwargs):
        self.invoked = True
        return AIMessage(content="".join(self.tokens))

    async def ainvoke(self, messages, **_kwargs):
        self.invoked = True
        self.ainvoke_count += 1
        return AIMessage(content=self.quiz_json)


class StubJudgeLLM:
    """Judge LLM stub that always returns a pass verdict."""

    _PASS = (
        '{"relevance":5,"accuracy":5,"citation_quality":4,'
        '"accessibility":4,"example_quality":5,"learner_level_fit":5,'
        '"reasoning":"Solid."}'
    )

    async def ainvoke(self, messages, **_kwargs):
        return AIMessage(content=self._PASS)


@pytest.fixture
def stub_retriever():
    r = StubRetriever()
    r.search_returns = [
        {"chunk_id": "a:1:0", "content": "HyDE rewrites queries.",
         "source": "a.pdf", "page": 1, "score": 0.9},
    ]
    return r


_QUIZ_GEN_JSON = """[
  {
    "prompt": "What does HyDE rewrite?",
    "options": ["A) Queries", "B) Documents", "C) Embeddings", "D) Answers"],
    "answer": "A",
    "explanation": "HyDE rewrites the user query into a hypothetical answer."
  }
]"""


@pytest.fixture
def stub_llm():
    return StubLLM(
        tokens=["HyDE", " is", " a", " technique", "."],
        quiz_json=_QUIZ_GEN_JSON,
    )


@pytest.fixture
def stub_judge_llm():
    return StubJudgeLLM()


def _make_app(tmp_path, monkeypatch, stub_retriever, stub_llm, stub_judge_llm, *, same_model: bool):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    from app.db import session as session_mod
    session_mod._engine = None
    session_mod._SessionLocal = None

    app = create_app()
    app.state.retriever = stub_retriever
    from app.api.deps import get_judge_dependencies, get_llm
    app.dependency_overrides[get_llm] = lambda: stub_llm
    app.dependency_overrides[get_judge_dependencies] = lambda: {
        "llm": stub_judge_llm,
        "same_model": same_model,
    }
    return app


@pytest.fixture
def app(tmp_path, stub_retriever, stub_llm, stub_judge_llm, monkeypatch):
    # Default: same_model=False (no warning) so generic SSE shape tests are
    # not perturbed by the P2.1-② bias warning. Tests that need the warning
    # build their own app via _make_app(same_model=True).
    return _make_app(tmp_path, monkeypatch, stub_retriever, stub_llm, stub_judge_llm,
                     same_model=False)


@pytest.fixture
def client(app):
    return TestClient(app)


def _read_sse_events(resp) -> list[dict]:
    events = []
    for line in resp.iter_lines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


_HEADERS = {
    "x-fingerprint": "fp-1",
    "x-provider": "ollama",
    "x-model": "gemma3:4b",
}


def test_chat_via_graph_tutor_path_emits_citations_then_tokens_then_done(
    client, stub_retriever, stub_llm
):
    with client.stream("POST", "/api/chat",
                       json={"message": "What is HyDE?"},
                       headers=_HEADERS) as resp:
        assert resp.status_code == 200
        events = _read_sse_events(resp)

    types = [e["type"] for e in events]
    assert types[0] == "citations", f"first event must be citations, got {types}"
    assert types[-1] == "done", f"last event must be done, got {types}"

    citation_event = events[0]
    assert citation_event["citations"][0]["chunk_id"] == "a:1:0"
    assert citation_event["citations"][0]["source"] == "a.pdf"

    token_events = [e for e in events if e["type"] == "token"]
    assert "".join(e["text"] for e in token_events) == "HyDE is a technique."

    # Tutor path actually queried the retriever
    assert stub_retriever.last_query == "What is HyDE?"
    assert stub_llm.invoked is True


def test_chat_via_graph_quiz_path_generates_real_question(
    client, stub_retriever, stub_llm
):
    """After Cut ④f production wiring, quiz path runs the real QuizMaster
    (generate_quiz LLM call → persist question). After Cut ④h grounding,
    QuizMaster also queries the retriever and embeds chunks in the LLM prompt."""
    with client.stream("POST", "/api/chat",
                       json={"message": "测我一下 HyDE"},
                       headers=_HEADERS) as resp:
        assert resp.status_code == 200
        events = _read_sse_events(resp)

    types = [e["type"] for e in events]
    assert types[0] == "citations"
    assert types[-1] == "done"

    # Quiz branch still emits empty citations (uniform frontend contract)
    assert events[0]["citations"] == []

    # Cut ④h: quiz path now grounds in retrieved chunks (topic name as query).
    assert stub_retriever.last_query == "HyDE"
    # Quiz GENERATE path uses ainvoke (not astream)
    assert stub_llm.ainvoke_count >= 1

    token_events = [e for e in events if e["type"] == "token"]
    joined = "".join(e["text"] for e in token_events)
    # Real question rendered + reply hint
    assert "What does HyDE rewrite?" in joined
    assert "Reply with A, B, C, or D" in joined


def test_chat_via_graph_plan_path_emits_empty_citations_then_stub_tokens(
    client, app, stub_retriever
):
    """After Cut ⑤g production wiring, plan path runs the real planner
    (generate_milestones LLM call → persist plan). Retriever is queried for
    topic-grounded planning."""
    from app.api.deps import get_llm

    _PLAN_JSON = (
        '[{"title": "Read §1", "due_at": "2026-05-25", "done": false, "topic": "HyDE"},'
        ' {"title": "Practice", "due_at": "2026-05-28", "done": false, "topic": "HyDE"},'
        ' {"title": "Review", "due_at": "2026-06-01", "done": false, "topic": "HyDE"}]'
    )
    plan_llm = StubLLM(tokens=[], quiz_json=_PLAN_JSON)
    app.dependency_overrides[get_llm] = lambda: plan_llm

    with client.stream("POST", "/api/chat",
                       json={"message": "帮我做学习计划 on HyDE"},
                       headers=_HEADERS) as resp:
        assert resp.status_code == 200
        events = _read_sse_events(resp)

    types = [e["type"] for e in events]
    assert types[0] == "citations"
    assert types[-1] == "done"

    # Plan branch still emits empty citations (uniform frontend contract)
    assert events[0]["citations"] == []

    # Cut ⑤g: plan path now grounds in retrieved chunks and runs the real planner.
    assert stub_retriever.last_query is not None
    assert plan_llm.ainvoke_count >= 1

    token_events = [e for e in events if e["type"] == "token"]
    joined = "".join(e["text"] for e in token_events)
    # Real milestones rendered (no more P2.1-⑤ stub message).
    assert "Read §1" in joined


def test_chat_quiz_two_turns_grade_after_generate(client, stub_llm):
    """Cut ④f: state persists across SSE requests via InMemorySaver + thread_id.

    Turn 1: "quiz me on HyDE" → question generated, active_quiz_question_id saved.
    Turn 2: same session_id, "A" → router state-aware override routes to quiz,
            grade path scores correct, mastery bumped.
    """
    session_id = "quiz-session-1"

    with client.stream(
        "POST", "/api/chat",
        json={"message": "quiz me on HyDE", "session_id": session_id},
        headers=_HEADERS,
    ) as resp:
        assert resp.status_code == 200
        turn1 = _read_sse_events(resp)
    joined1 = "".join(e["text"] for e in turn1 if e["type"] == "token")
    assert "What does HyDE rewrite?" in joined1

    with client.stream(
        "POST", "/api/chat",
        json={"message": "A", "session_id": session_id},
        headers=_HEADERS,
    ) as resp:
        assert resp.status_code == 200
        turn2 = _read_sse_events(resp)
    joined2 = "".join(e["text"] for e in turn2 if e["type"] == "token")
    assert "Correct" in joined2, f"turn 2 must grade as correct; got: {joined2!r}"


# --- P2.1-② Judge Guard same-model warning ---------------------------------


def test_chat_emits_same_model_warning_right_after_citations_on_tutor_path(
    tmp_path, monkeypatch, stub_retriever, stub_llm, stub_judge_llm
):
    """When the judge runs with the same model as the generator, the SSE stream
    must surface a bias warning right after `citations` (and before real LLM
    tokens) so the user can see the self-preference risk inline in the answer."""
    app = _make_app(tmp_path, monkeypatch, stub_retriever, stub_llm, stub_judge_llm,
                    same_model=True)
    client = TestClient(app)

    with client.stream("POST", "/api/chat",
                       json={"message": "What is HyDE?"},
                       headers=_HEADERS) as resp:
        events = _read_sse_events(resp)

    types = [e["type"] for e in events]
    assert types[0] == "citations"
    assert types[-1] == "done"
    token_events = [e for e in events if e["type"] == "token"]
    assert token_events, "expected at least one token event"
    # First token must be the bias warning (inline-prepended); subsequent
    # tokens are the real LLM stream.
    first_token_text = token_events[0]["text"].lower()
    assert "self-check" in first_token_text or "⚠️" in token_events[0]["text"], (
        f"first token must carry inline bias warning, got: {token_events[0]['text']!r}"
    )
    # Real LLM output must still be present after the warning
    joined = "".join(e["text"] for e in token_events)
    assert "HyDE is a technique." in joined


def test_chat_omits_warning_when_x_judge_model_explicitly_set(
    tmp_path, monkeypatch, stub_retriever, stub_llm, stub_judge_llm
):
    """When the user has set x-judge-model to a distinct model, no warning fires."""
    app = _make_app(tmp_path, monkeypatch, stub_retriever, stub_llm, stub_judge_llm,
                    same_model=False)
    client = TestClient(app)

    headers = {**_HEADERS, "x-judge-model": "qwen2.5:7b"}
    with client.stream("POST", "/api/chat",
                       json={"message": "What is HyDE?"},
                       headers=headers) as resp:
        events = _read_sse_events(resp)

    token_events = [e for e in events if e["type"] == "token"]
    joined = "".join(e["text"] for e in token_events)
    assert "self-check" not in joined.lower() and "⚠️" not in joined, (
        f"no warning expected when same_model=False, got tokens: {joined!r}"
    )
    # Strict equality: only the real LLM output flows when no warning
    assert joined == "HyDE is a technique."


def test_chat_stub_paths_do_not_emit_warning_even_when_same_model(
    tmp_path, monkeypatch, stub_retriever, stub_llm, stub_judge_llm
):
    """Quiz / Plan stubs don't invoke the LLM at all, so the bias warning is moot.
    The warning gate keys on non-empty citations (proxy for real LLM output)."""
    app = _make_app(tmp_path, monkeypatch, stub_retriever, stub_llm, stub_judge_llm,
                    same_model=True)
    client = TestClient(app)

    with client.stream("POST", "/api/chat",
                       json={"message": "测我一下"},
                       headers=_HEADERS) as resp:
        events = _read_sse_events(resp)

    token_events = [e for e in events if e["type"] == "token"]
    joined = "".join(e["text"] for e in token_events)
    assert "self-check" not in joined.lower() and "⚠️" not in joined, (
        f"stub path must not surface bias warning (no LLM ran), got: {joined!r}"
    )
