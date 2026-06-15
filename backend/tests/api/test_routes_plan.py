"""Cut ⑤g — Plan path through /api/chat.

Multi-turn: first request creates a plan (GENERATE), second request with the
same session_id reads the checkpointer state and enters CHECK-IN.
"""
import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.main import create_app
from app.api import deps


_GEN_JSON = """[
  {"title": "Read §1", "due_at": "2026-05-25", "done": false, "topic": "HyDE"},
  {"title": "Practice", "due_at": "2026-05-28", "done": false, "topic": "HyDE"},
  {"title": "Review", "due_at": "2026-06-01", "done": false, "topic": "HyDE"}
]"""

_CHECK_IN_JSON = """[
  {"title": "Read §1", "due_at": "2026-05-25", "done": true, "topic": "HyDE"},
  {"title": "Practice", "due_at": "2026-05-30", "done": false, "topic": "HyDE"},
  {"title": "Review", "due_at": "2026-06-03", "done": false, "topic": "HyDE"}
]"""


class StubRetriever:
    """Minimal retriever stub — planner only needs `search` to return chunks."""

    def __init__(self):
        self.last_query: str | None = None

    def add_chunks(self, chunks):  # pragma: no cover - unused in plan tests
        pass

    def search(self, query: str, top_k: int = 5):
        self.last_query = query
        return [
            {"chunk_id": "p:1:0", "content": "HyDE rewrites queries.",
             "source": "p.pdf", "page": 1, "score": 0.9},
        ]


class StubLLM:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.idx = 0

    async def ainvoke(self, messages, **_kwargs):
        text = self.responses[min(self.idx, len(self.responses) - 1)]
        self.idx += 1
        return AIMessage(content=text)

    async def astream(self, messages, **_kwargs):
        yield AIMessage(content=self.responses[0])


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/plan_test.db")
    monkeypatch.setenv("STUDY_COACH_TEST_MODE", "1")
    from app.db import session as session_mod
    session_mod._engine = None
    session_mod._SessionLocal = None

    app = create_app()
    app.state.retriever = StubRetriever()
    stub = StubLLM([_GEN_JSON, _CHECK_IN_JSON])
    app.dependency_overrides[deps.get_llm] = lambda: stub
    app.dependency_overrides[deps.get_judge_dependencies] = lambda: {"llm": None, "same_model": False}
    from app.db.repositories import DocumentRepository
    from app.db.session import session_scope
    with session_scope() as session:
        DocumentRepository(session).create(
            user_id="default-user",
            filename="fixture.pdf",
            hash_="fixture-hash",
            chunks_count=1,
        )
    with TestClient(app) as c:
        yield c


def _parse_events(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
    return events


def test_chat_plan_generate_emits_citations_token_done(client):
    response = client.post(
        "/api/chat",
        json={"message": "帮我做学习计划 on HyDE", "session_id": "sess-plan-1"},
        headers={"x-fingerprint": "fp-plan"},
    )
    assert response.status_code == 200
    events = _parse_events(response.text)
    types = [e["type"] for e in events]
    first_citations_idx = next(i for i, t in enumerate(types) if t == "citations")
    assert "token" in types
    assert types[-1] == "done"
    token_event = next(e for e in events if e["type"] == "token")
    assert "Read §1" in token_event["text"]


def test_chat_plan_two_turns_check_in_after_generate(client):
    # Turn 1: GENERATE
    client.post(
        "/api/chat",
        json={"message": "帮我做学习计划 on HyDE", "session_id": "sess-plan-2"},
        headers={"x-fingerprint": "fp-plan2"},
    )
    # Turn 2: CHECK-IN (same session_id → checkpointer retains active_plan_id)
    response = client.post(
        "/api/chat",
        json={"message": "进度怎么样了", "session_id": "sess-plan-2"},
        headers={"x-fingerprint": "fp-plan2"},
    )
    assert response.status_code == 200
    events = _parse_events(response.text)
    token_text = "".join(e.get("text", "") for e in events if e["type"] == "token")
    assert "Progress" in token_text or "进度" in token_text
