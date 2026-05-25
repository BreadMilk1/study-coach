"""Cut P2.3-①e — route-level integration tests for x-quiz-mode + multi-turn dispatcher.

Three assertions:
  1. x-quiz-mode=agent_loop → SSE stream emits content from the agent path
  2. x-quiz-mode absent → SSE from deterministic quiz_master path
  3. Multi-turn: turn 1 = GENERATE in agent_loop mode (agent path), turn 2 = "A"
     (active_quiz_question_id set by turn 1) → state-aware override → deterministic GRADE
"""
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.api.deps import (
    get_quiz_master,
    get_quiz_master_agent,
)
from app.main import create_app


class StubRetriever:
    def add_chunks(self, _chunks):
        pass

    def search(self, query, top_k=5):
        return [{"chunk_id": "c1", "content": "stub",
                 "source": "p.pdf", "page": 1, "score": 0.9}]


@pytest.fixture
def client_with_stubs(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/p2_3_routes.db")
    monkeypatch.setenv("STUDY_COACH_TEST_MODE", "1")
    from app.db import session as session_mod
    session_mod._engine = None
    session_mod._SessionLocal = None

    app = create_app()
    app.state.retriever = StubRetriever()

    # Identifiable stubs
    async def agent_stub(state):
        from langgraph.config import get_stream_writer
        writer = get_stream_writer()
        writer({"type": "citations", "citations": []})
        writer({"type": "token", "text": "AGENT-QUIZ-PATH"})
        return {
            "messages": [AIMessage(content="AGENT-QUIZ-PATH")],
            "citations": [],
            "active_quiz_question_id": "agent-q-1",
            "quiz_action": "generate",
            "agent_trace": {
                "exit_reason": "natural_stop", "total_iterations": 2,
                "total_tool_calls": 2, "tool_call_breakdown": {},
                "tool_errors": 0, "input_tokens": 0,
                "output_tokens": 0, "wall_time_s": 0.1,
                "llm_error": None,
            },
        }

    async def quiz_master_stub(state):
        from langgraph.config import get_stream_writer
        writer = get_stream_writer()
        if state.get("active_quiz_question_id"):
            writer({"type": "citations", "citations": []})
            writer({"type": "token", "text": "DETERMINISTIC-GRADE-OK"})
            return {
                "messages": [AIMessage(content="DETERMINISTIC-GRADE-OK")],
                "citations": [],
                "active_quiz_question_id": None,
                "quiz_action": "grade",
            }
        writer({"type": "citations", "citations": []})
        writer({"type": "token", "text": "DETERMINISTIC-QUIZ-PATH"})
        return {
            "messages": [AIMessage(content="DETERMINISTIC-QUIZ-PATH")],
            "citations": [],
            "active_quiz_question_id": "deterministic-q-1",
            "quiz_action": "generate",
        }

    app.dependency_overrides[get_quiz_master] = lambda: quiz_master_stub
    app.dependency_overrides[get_quiz_master_agent] = lambda: agent_stub

    with TestClient(app) as c:
        yield c


def test_quiz_mode_header_agent_loop_routes_to_agent_path(client_with_stubs):
    response = client_with_stubs.post(
        "/api/chat",
        json={"message": "quiz me on HyDE", "session_id": "test-sess-1"},
        headers={"x-fingerprint": "fp-r-1", "x-quiz-mode": "agent_loop"},
    )
    assert response.status_code == 200
    body = response.text
    assert "AGENT-QUIZ-PATH" in body
    assert "citations" in body
    assert "token" in body
    assert "done" in body


def test_quiz_mode_header_absent_defaults_to_deterministic(client_with_stubs):
    response = client_with_stubs.post(
        "/api/chat",
        json={"message": "quiz me on chunking", "session_id": "test-sess-2"},
        headers={"x-fingerprint": "fp-r-2"},
    )
    assert response.status_code == 200
    body = response.text
    assert "DETERMINISTIC-QUIZ-PATH" in body
    assert "AGENT-QUIZ-PATH" not in body


def test_multi_turn_agent_loop_generate_then_deterministic_grade(client_with_stubs):
    """Turn 1: GENERATE in agent_loop mode → AGENT path persists active_quiz_question_id
    via stub. Turn 2 (same session) → graph dispatcher state-aware override → deterministic GRADE."""
    # Turn 1
    r1 = client_with_stubs.post(
        "/api/chat",
        json={"message": "quiz me on BM25", "session_id": "test-sess-3"},
        headers={"x-fingerprint": "fp-r-3", "x-quiz-mode": "agent_loop"},
    )
    assert r1.status_code == 200
    assert "AGENT-QUIZ-PATH" in r1.text

    # Turn 2: same session, expect state to have active_quiz_question_id from turn 1's checkpointer
    r2 = client_with_stubs.post(
        "/api/chat",
        json={"message": "A", "session_id": "test-sess-3"},
        headers={"x-fingerprint": "fp-r-3", "x-quiz-mode": "agent_loop"},
    )
    assert r2.status_code == 200
    assert "DETERMINISTIC-GRADE-OK" in r2.text
    assert "AGENT-QUIZ-PATH" not in r2.text
