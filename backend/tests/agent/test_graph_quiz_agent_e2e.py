"""Cut P2.3-①d — graph-level e2e tests for the Quiz mode-aware dispatcher.

Three assertions:
  1. quiz_mode="agent_loop" + GENERATE turn → routes to agent (agent_trace set)
  2. quiz_mode="agent_loop" + GRADE turn (active_quiz_question_id set) →
     routes to deterministic quiz_master (state-aware override)
  3. quiz_mode="deterministic" (default) + GENERATE → routes to quiz_master
     (baseline path, no agent_trace in state)
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.agent.graph import build_graph
from app.db.models import Base
from app.db.repositories import UserRepository


class _StubTutorLLM:
    """Unused in quiz path; satisfies build_graph(llm=...) signature."""

    async def astream(self, messages, **_kwargs):
        from langchain_core.messages import AIMessageChunk
        yield AIMessageChunk(content="tutor never runs in these tests")


class _StubRetriever:
    def search(self, query, top_k=5):
        return []


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


# Identifiable stubs: each path emits a distinct string so the test asserts which path served
async def _stub_quiz_master_generate(state):
    return {
        "messages": [AIMessage(content="DETERMINISTIC-PATH: generated question X")],
        "citations": [],
        "active_quiz_question_id": "deterministic-q-1",
        "quiz_action": "generate",
    }


async def _stub_quiz_master_grade(state):
    return {
        "messages": [AIMessage(content="DETERMINISTIC-GRADE: ✓ correct")],
        "citations": [],
        "active_quiz_question_id": None,
        "quiz_action": "grade",
    }


async def _stub_quiz_master(state):
    """Branch on state shape: GRADE if active_quiz_question_id set, else GENERATE."""
    if state.get("active_quiz_question_id"):
        return await _stub_quiz_master_grade(state)
    return await _stub_quiz_master_generate(state)


async def _stub_quiz_master_agent(state):
    return {
        "messages": [AIMessage(content="AGENT-PATH: generated question Y")],
        "citations": [],
        "active_quiz_question_id": "agent-q-1",
        "quiz_action": "generate",
        "agent_trace": {
            "total_iterations": 2,
            "total_tool_calls": 2,
            "tool_call_breakdown": {"retriever_search": 1, "persist_quiz_question": 1},
            "tool_errors": 0,
            "input_tokens": 100,
            "output_tokens": 50,
            "wall_time_s": 2.5,
            "exit_reason": "natural_stop",
            "llm_error": None,
        },
    }


async def test_quiz_mode_agent_loop_routes_to_agent_on_generate_turn(session):
    user = UserRepository(session).get_or_create("fp-graph-1")
    graph = build_graph(retriever=_StubRetriever(), llm=_StubTutorLLM())
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="quiz me on HyDE")], "user_id": user.id},
        config={"configurable": {
            "quiz_master": _stub_quiz_master,
            "quiz_master_agent": _stub_quiz_master_agent,
            "quiz_mode": "agent_loop",
        }},
    )
    # Route hit the agent stub
    assert "AGENT-PATH" in result["messages"][-1].content
    assert result.get("agent_trace") is not None
    assert result["agent_trace"]["exit_reason"] == "natural_stop"


async def test_quiz_mode_agent_loop_routes_to_deterministic_on_grade_turn(session):
    """State-aware override: active_quiz_question_id truthy → deterministic GRADE
    regardless of quiz_mode configured."""
    user = UserRepository(session).get_or_create("fp-graph-2")
    graph = build_graph(retriever=_StubRetriever(), llm=_StubTutorLLM())
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="A")],
            "user_id": user.id,
            "active_quiz_question_id": "pre-existing-q-99",
        },
        config={"configurable": {
            "quiz_master": _stub_quiz_master,
            "quiz_master_agent": _stub_quiz_master_agent,
            "quiz_mode": "agent_loop",  # still agent_loop, but GRADE state wins
        }},
    )
    # Route hit the deterministic stub even though mode is agent_loop
    assert "DETERMINISTIC-GRADE" in result["messages"][-1].content
    assert result.get("quiz_action") == "grade"
    assert result.get("agent_trace") is None


async def test_quiz_mode_deterministic_default_routes_to_quiz_master(session):
    user = UserRepository(session).get_or_create("fp-graph-3")
    graph = build_graph(retriever=_StubRetriever(), llm=_StubTutorLLM())
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="quiz me on chunking")], "user_id": user.id},
        config={"configurable": {
            "quiz_master": _stub_quiz_master,
            "quiz_master_agent": _stub_quiz_master_agent,
            "quiz_mode": "deterministic",
        }},
    )
    assert "DETERMINISTIC-PATH" in result["messages"][-1].content
    assert result.get("agent_trace") is None
