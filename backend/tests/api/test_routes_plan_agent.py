"""Cut P2.2-①e — /api/chat with x-planner-mode header.

Two contracts under test:
  1. x-planner-mode: agent_loop routes through the new planner_agent factory
     and produces the same SSE shape (citations → token → done) as the
     deterministic baseline.
  2. Default (no header) and unknown values fall back to deterministic; the
     existing P2.1-⑤g test_chat_plan_generate_emits_citations_token_done
     covers default — here we explicitly verify the unknown-value fallback.
"""
import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.api import deps
from app.auth import issue_token
from app.main import create_app
from tests.helpers import ensure_user


def _msg(content="", tool_calls=None, input_tokens=10, output_tokens=5):
    m = AIMessage(content=content, tool_calls=tool_calls or [])
    m.usage_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    return m


class ScriptedLLM:
    """Routes calls to the right script by sniffing for SystemMessage.

    The planner_agent always prepends a SystemMessage (agent system prompt);
    the deterministic planner only sends HumanMessage prompts. We use that to
    deliver each path's own script independently, even though both factories
    share the same LLM instance via `deps.get_llm` dependency caching.
    """

    def __init__(self, agent_responses, det_responses):
        self.agent_responses = list(agent_responses)
        self.det_responses = list(det_responses)
        self.agent_idx = 0
        self.det_idx = 0

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, messages, **_kwargs):
        from langchain_core.messages import SystemMessage
        is_agent = any(isinstance(m, SystemMessage) for m in messages)
        if is_agent:
            if self.agent_idx >= len(self.agent_responses):
                raise AssertionError("ScriptedLLM agent script exhausted")
            m = self.agent_responses[self.agent_idx]
            self.agent_idx += 1
            return m
        if self.det_idx >= len(self.det_responses):
            raise AssertionError("ScriptedLLM deterministic script exhausted")
        m = self.det_responses[self.det_idx]
        self.det_idx += 1
        return m

    async def astream(self, messages, **_kwargs):
        # Not used in these tests; provide a no-op generator for safety.
        if False:
            yield None


class StubRetriever:
    def add_chunks(self, _chunks):
        pass

    def search(self, query, top_k=5):
        return [{"chunk_id": "c1", "content": "HyDE def",
                 "source": "p.pdf", "page": 1, "score": 0.9}]


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/agent_test.db")
    monkeypatch.setenv("STUDY_COACH_TEST_MODE", "1")
    from app.db import session as session_mod
    session_mod._engine = None
    session_mod._SessionLocal = None

    app = create_app()
    app.state.retriever = StubRetriever()

    # Agent-mode LLM script: tool call → final summary
    agent_script = [
        _msg(tool_calls=[{
            "name": "update_study_plan",
            "args": {"milestones": [
                {"title": "M1", "due_at": "2026-05-30", "done": False, "topic": "HyDE"},
            ]},
            "id": "c1",
        }]),
        _msg(content="📋 Plan: M1 by 2026-05-30."),
    ]
    # Deterministic-mode LLM script: raw milestones JSON (one call only)
    det_script = ["""[
      {"title": "DET-M1", "due_at": "2026-05-30", "done": false, "topic": "HyDE"}
    ]"""]

    def get_llm_override():
        # ChatRequest goes through deps.get_llm once per request; both factories
        # (planner, planner_agent) receive the same LLM. ScriptedLLM sniffs the
        # SystemMessage presence to route to the right script (agent vs det).
        return ScriptedLLM(
            agent_responses=agent_script,
            det_responses=[_msg(content=det_script[0])],
        )

    app.dependency_overrides[deps.get_llm] = get_llm_override
    app.dependency_overrides[deps.get_judge_dependencies] = lambda: {"llm": None, "same_model": False}

    from app.db.repositories import DocumentRepository
    from app.db.session import session_scope
    with session_scope() as session:
        ensure_user(session, "default-user")
        DocumentRepository(session).create(
            user_id="default-user",
            filename="fixture.pdf",
            hash_="fixture-hash",
            chunks_count=1,
        )

    with TestClient(
        app,
        headers={"Authorization": f"Bearer {issue_token('default-user', 'guest')}"},
    ) as c:
        yield c


def _parse_events(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
    return events


def test_chat_agent_loop_mode_emits_citations_token_done(client):
    response = client.post(
        "/api/chat",
        json={"message": "make a plan on HyDE", "session_id": "sess-agent-1"},
        headers={
            "x-fingerprint": "fp-agent",
            "x-planner-mode": "agent_loop",
        },
    )
    assert response.status_code == 200
    events = _parse_events(response.text)
    types = [e["type"] for e in events]
    first_citations_idx = next(i for i, t in enumerate(types) if t == "citations")
    assert "token" in types
    assert types[-1] == "done"
    token_event = next(e for e in events if e["type"] == "token")
    # Agent-mode final summary contains a unique string the deterministic
    # path never produces (deterministic emits "DET-M1" instead).
    assert "📋 Plan: M1 by 2026-05-30." in token_event["text"]
    agent_run = next(e for e in events if e["type"] == "agent_run")
    assert agent_run["run"]["node"] == "planner"
    assert agent_run["run"]["mode"] == "agent_loop"
    assert agent_run["run"]["exit_reason"] == "natural_stop"
    assert agent_run["run"]["total_tool_calls"] == 1
    assert agent_run["run"]["tool_call_breakdown"] == {"update_study_plan": 1}


def test_chat_unknown_planner_mode_falls_back_to_deterministic(client):
    response = client.post(
        "/api/chat",
        json={"message": "帮我做学习计划 on HyDE", "session_id": "sess-fallback"},
        headers={
            "x-fingerprint": "fp-fallback",
            "x-planner-mode": "bogus-value",
        },
    )
    assert response.status_code == 200
    events = _parse_events(response.text)
    token_text = "".join(e.get("text", "") for e in events if e["type"] == "token")
    # Deterministic-mode output uses the deterministic LLM script (DET-M1)
    assert "DET-M1" in token_text
