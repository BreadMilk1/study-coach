"""Cut P2.3-①c — unit tests for the quiz_master_agent loop body.

Test surface:
  1. natural_stop — model emits final summary with no tool calls
  2. budget_exhausted — model keeps calling tools past max_iter
  3. llm_call_failed — LLM ainvoke raises (e.g. Ollama 400 for no-tools model)
  4. tool_error_self_correction — invalid schema → ToolMessage → model retries
  5. valid_persist_round_trip — full happy path: persist → summary → active_quiz_question_id set
  6. quiz_action_always_generate — agent never sees GRADE turns by contract
"""
import json
from datetime import datetime

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.agent.agent_trace import AgentTrace
from app.agent.quiz_master_agent import (
    build_quiz_master_agent,
    _infer_quiz_action,
)
from app.db.models import Base
from app.db.repositories import (
    GoalRepository,
    QuestionRepository,
    TopicRepository,
    UserRepository,
)


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


class ScriptedLLM:
    """LLM that emits a scripted sequence of responses."""
    def __init__(self, responses: list[AIMessage]):
        self.responses = list(responses)
        self.bound_tools = None
        self.calls = 0

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    async def ainvoke(self, messages, **_kwargs):
        if self.calls >= len(self.responses):
            raise IndexError("ScriptedLLM ran out of responses")
        resp = self.responses[self.calls]
        self.calls += 1
        return resp


class FailingLLM:
    bound_tools = None
    def bind_tools(self, tools):
        self.bound_tools = tools
        return self
    async def ainvoke(self, messages, **_kwargs):
        raise ConnectionRefusedError("ollama is down")


def _ai(content="", tool_calls=None, input_tokens=10, output_tokens=5):
    msg = AIMessage(content=content, tool_calls=tool_calls or [])
    msg.usage_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    return msg


async def test_natural_stop_returns_final_summary(session):
    user = UserRepository(session).get_or_create("fp-loop-1")
    goal_repo = GoalRepository(session)
    topic_repo = TopicRepository(session)
    question_repo = QuestionRepository(session)

    valid_persist_args = {
        "topic": "HyDE",
        "prompt": "What does HyDE stand for?",
        "options": ["A) Hypothesis-Driven Experimentation",
                    "B) Hypothetical Document Embedding",
                    "C) High-Yield Data Encoding",
                    "D) Hybrid Document Engine"],
        "answer": "B",
        "explanation": "HyDE = Hypothetical Document Embedding.",
    }
    llm = ScriptedLLM([
        _ai(tool_calls=[{"name": "retriever_search",
                         "args": {"query": "HyDE"}, "id": "tc-1"}]),
        _ai(tool_calls=[{"name": "persist_quiz_question",
                         "args": valid_persist_args, "id": "tc-2"}]),
        _ai(content="Quiz on HyDE:\n\nWhat does HyDE stand for?\nA) ...\nB) ...\nReply with A-D."),
    ])

    agent = build_quiz_master_agent(
        llm=llm,
        topic_repo=topic_repo, question_repo=question_repo, goal_repo=goal_repo,
        retriever=None,
        now_fn=lambda: datetime(2026, 5, 24),
    )
    result = await agent({
        "messages": [HumanMessage(content="quiz me on HyDE")],
        "user_id": user.id,
    })

    assert result["quiz_action"] == "generate"
    assert result["active_quiz_question_id"] is not None
    assert "agent_trace" in result
    assert result["agent_trace"]["exit_reason"] == "natural_stop"
    assert result["agent_trace"]["total_iterations"] == 3
    assert result["agent_trace"]["total_tool_calls"] == 2
    assert "Quiz on HyDE" in result["messages"][0].content
    assert isinstance(result["messages"][0], AIMessage)


async def test_budget_exhausted_degrades_gracefully(session):
    user = UserRepository(session).get_or_create("fp-loop-2")
    goal_repo = GoalRepository(session)
    topic_repo = TopicRepository(session)
    question_repo = QuestionRepository(session)

    looping_responses = [
        _ai(tool_calls=[{"name": "retriever_search",
                         "args": {"query": "x"}, "id": f"tc-{i}"}])
        for i in range(10)
    ]
    llm = ScriptedLLM(looping_responses)

    agent = build_quiz_master_agent(
        llm=llm,
        topic_repo=topic_repo, question_repo=question_repo, goal_repo=goal_repo,
        retriever=None,
        max_iter=3,
    )
    result = await agent({
        "messages": [HumanMessage(content="quiz me on chunking")],
        "user_id": user.id,
    })

    assert result["degraded"] is True
    assert result["agent_trace"]["exit_reason"] == "budget_exhausted"
    assert "budget" in result["messages"][0].content.lower() or "⚠️" in result["messages"][0].content
    assert result.get("active_quiz_question_id") is None


async def test_llm_call_failed_degrades_gracefully(session):
    user = UserRepository(session).get_or_create("fp-loop-3")
    goal_repo = GoalRepository(session)
    topic_repo = TopicRepository(session)
    question_repo = QuestionRepository(session)

    agent = build_quiz_master_agent(
        llm=FailingLLM(),
        topic_repo=topic_repo, question_repo=question_repo, goal_repo=goal_repo,
    )
    result = await agent({
        "messages": [HumanMessage(content="quiz me on HyDE")],
        "user_id": user.id,
    })

    assert result["degraded"] is True
    assert result["agent_trace"]["exit_reason"] == "llm_call_failed"
    assert "ConnectionRefused" in (result["agent_trace"]["llm_error"] or "")


async def test_tool_error_self_correction_via_toolmessage(session):
    user = UserRepository(session).get_or_create("fp-loop-4")
    goal_repo = GoalRepository(session)
    topic_repo = TopicRepository(session)
    question_repo = QuestionRepository(session)

    bad_args = {
        "topic": "BM25",
        "prompt": "What is BM25?",
        "options": ["A) bad", "B) bad", "C) bad"],
        "answer": "A",
        "explanation": "BM25 is...",
    }
    good_args = {
        "topic": "BM25",
        "prompt": "What is BM25?",
        "options": ["A) embedding", "B) ranking function",
                    "C) tokenizer", "D) reranker"],
        "answer": "B",
        "explanation": "BM25 is a probabilistic ranking function.",
    }
    llm = ScriptedLLM([
        _ai(tool_calls=[{"name": "persist_quiz_question",
                         "args": bad_args, "id": "tc-bad"}]),
        _ai(tool_calls=[{"name": "persist_quiz_question",
                         "args": good_args, "id": "tc-good"}]),
        _ai(content="Quiz on BM25 ready"),
    ])

    agent = build_quiz_master_agent(
        llm=llm,
        topic_repo=topic_repo, question_repo=question_repo, goal_repo=goal_repo,
    )
    result = await agent({
        "messages": [HumanMessage(content="quiz me on BM25")],
        "user_id": user.id,
    })

    assert result["quiz_action"] == "generate"
    assert result["active_quiz_question_id"] is not None
    assert result["agent_trace"]["exit_reason"] == "natural_stop"
    assert result["agent_trace"]["tool_errors"] == 1
    breakdown = result["agent_trace"]["tool_call_breakdown"]
    assert breakdown.get("persist_quiz_question") == 2


async def test_valid_persist_round_trip_writes_active_quiz_question_id(session):
    user = UserRepository(session).get_or_create("fp-loop-5")
    goal_repo = GoalRepository(session)
    topic_repo = TopicRepository(session)
    question_repo = QuestionRepository(session)

    persist_args = {
        "topic": "embeddings",
        "prompt": "What is an embedding?",
        "options": ["A) A vector representation",
                    "B) A type of database",
                    "C) A search algorithm",
                    "D) A loss function"],
        "answer": "A",
        "explanation": "An embedding is a dense vector representation of text.",
    }
    llm = ScriptedLLM([
        _ai(tool_calls=[{"name": "persist_quiz_question",
                         "args": persist_args, "id": "tc-1"}]),
        _ai(content="Quiz ready"),
    ])

    agent = build_quiz_master_agent(
        llm=llm,
        topic_repo=topic_repo, question_repo=question_repo, goal_repo=goal_repo,
    )
    result = await agent({
        "messages": [HumanMessage(content="quiz me on embeddings")],
        "user_id": user.id,
    })

    persisted_id = result["active_quiz_question_id"]
    assert persisted_id is not None
    fetched = question_repo.get_by_id(persisted_id)
    assert fetched is not None
    assert fetched.answer == "A"


def test_infer_quiz_action_always_returns_generate():
    """Agent never sees GRADE turns by dispatcher contract."""
    import time
    trace = AgentTrace(t_start=time.monotonic())
    assert _infer_quiz_action(trace) == "generate"
    trace.record_tool_call("persist_quiz_question", {}, '{"question_id":"q"}', error=False)
    assert _infer_quiz_action(trace) == "generate"
