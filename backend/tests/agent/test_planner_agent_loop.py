"""Cut P2.2-①c — loop body + plan_action inference + error handling.

Stub LLM is scripted: each .ainvoke() returns the next preset AIMessage. Some
have tool_calls (forcing the loop to dispatch + iterate), the final one has
content but no tool_calls (natural stop).

Tests:
  1. natural_stop after retriever_search → update_study_plan → final summary
  2. budget_exhausted when LLM never stops calling tools
  3. llm_call_failed degrades cleanly without persisting
  4. tool error becomes a ToolMessage and loop continues (self-correction)
  5. plan_action inference: generate when get_existing_plan absent/null
  6. plan_action inference: check_in when get_existing_plan returned non-null
  7. plan_action inference fallback: generate when zero tools called
  8. _extract_topic regression — closes a P2.1-⑤i loose end
"""
from datetime import datetime

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.agent.planner_agent import build_planner_agent
from app.db.models import Base
from app.db.repositories import (
    GoalRepository,
    MasteryRepository,
    MistakeRepository,
    PlanRepository,
    UserRepository,
)


class ScriptedLLM:
    """Returns the next AIMessage from a preset list per .ainvoke() call.

    The harness wraps this with .bind_tools(tools); since our loop only reads
    the returned AIMessage and its .tool_calls, we don't need a real
    BaseChatModel — just a duck-typed ainvoke + bind_tools no-op.
    """
    def __init__(self, responses: list[AIMessage]):
        self.responses = list(responses)
        self.idx = 0
        self.calls = 0

    def bind_tools(self, _tools):
        return self  # pass-through; tools are dispatched outside the LLM

    async def ainvoke(self, messages, **_kwargs):
        self.calls += 1
        if self.idx >= len(self.responses):
            raise AssertionError("ScriptedLLM exhausted — loop called more times than expected")
        msg = self.responses[self.idx]
        self.idx += 1
        return msg


class CrashingLLM:
    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, messages, **_kwargs):
        raise ConnectionError("ollama unreachable")


class StubRetriever:
    def __init__(self, chunks=None):
        self.chunks = chunks or []

    def search(self, query, top_k=5):
        return self.chunks[:top_k]


def _msg(content="", tool_calls=None, input_tokens=10, output_tokens=5):
    msg = AIMessage(content=content, tool_calls=tool_calls or [])
    msg.usage_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    return msg


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


def _build(session, llm, retriever=None):
    return build_planner_agent(
        llm=llm,
        plan_repo=PlanRepository(session),
        goal_repo=GoalRepository(session),
        mastery_repo=MasteryRepository(session),
        mistake_repo=MistakeRepository(session),
        retriever=retriever or StubRetriever(),
        now_fn=lambda: datetime(2026, 5, 23, 12, 0),
        max_iter=10,
    )


async def test_loop_natural_stop_emits_plan_and_records_trace(session):
    user = UserRepository(session).get_or_create("fp-loop-1")
    goal_repo = GoalRepository(session)
    goal_repo.create(user_id=user.id, title="G")

    llm = ScriptedLLM([
        _msg(tool_calls=[{
            "name": "retriever_search",
            "args": {"query": "HyDE", "top_k": 3},
            "id": "c1",
        }]),
        _msg(tool_calls=[{
            "name": "update_study_plan",
            "args": {"milestones": [
                {"title": "Read HyDE", "due_at": "2026-05-25", "done": False, "topic": "HyDE"},
            ]},
            "id": "c2",
        }]),
        _msg(content="📋 Plan: Read HyDE by 2026-05-25.", tool_calls=[]),
    ])
    agent = _build(session, llm, retriever=StubRetriever(chunks=[
        {"chunk_id": "c1", "content": "HyDE definition", "page": 1},
    ]))

    update = await agent({
        "messages": [HumanMessage(content="make a plan on HyDE")],
        "user_id": user.id,
    })

    assert update["plan_action"] == "generate"
    assert update["active_plan_id"]
    assert "Read HyDE" in update["messages"][0].content
    trace = update["agent_trace"]
    assert trace["exit_reason"] == "natural_stop"
    assert trace["total_iterations"] == 3
    assert trace["total_tool_calls"] == 2
    assert trace["tool_call_breakdown"] == {
        "retriever_search": 1, "update_study_plan": 1,
    }


async def test_loop_budget_exhaustion_degrades_without_persisting(session):
    user = UserRepository(session).get_or_create("fp-loop-2")
    GoalRepository(session).create(user_id=user.id, title="G")

    # LLM keeps calling retriever_search forever; loop must bail at max_iter.
    forever_calls = [
        _msg(tool_calls=[{"name": "retriever_search", "args": {"query": "x"}, "id": f"c{i}"}])
        for i in range(12)
    ]
    llm = ScriptedLLM(forever_calls)
    agent = _build(session, llm)

    update = await agent({
        "messages": [HumanMessage(content="plan on X")],
        "user_id": user.id,
    })

    assert update["agent_trace"]["exit_reason"] == "budget_exhausted"
    # No plan should have been persisted on this path
    assert update.get("active_plan_id") is None
    # User-visible disclaimer text
    assert "reasoning budget" in update["messages"][0].content.lower() \
        or "budget" in update["messages"][0].content.lower()


async def test_loop_llm_error_degrades_with_disclaimer(session):
    user = UserRepository(session).get_or_create("fp-loop-3")
    GoalRepository(session).create(user_id=user.id, title="G")
    agent = _build(session, CrashingLLM())

    update = await agent({
        "messages": [HumanMessage(content="plan on something")],
        "user_id": user.id,
    })

    assert update["agent_trace"]["exit_reason"] == "llm_call_failed"
    assert "ConnectionError" in update["agent_trace"]["llm_error"]
    assert "could not reach" in update["messages"][0].content.lower() \
        or "model" in update["messages"][0].content.lower()


async def test_loop_tool_error_feeds_back_and_self_corrects(session):
    """First update_study_plan call has a bad milestone shape → ToolMessage
    error → LLM retries with correct shape → natural stop."""
    user = UserRepository(session).get_or_create("fp-loop-4")
    GoalRepository(session).create(user_id=user.id, title="G")

    llm = ScriptedLLM([
        _msg(tool_calls=[{
            "name": "update_study_plan",
            "args": {"milestones": [{"WRONG_KEY": "no title"}]},  # invalid → tool error
            "id": "c1",
        }]),
        _msg(tool_calls=[{
            "name": "update_study_plan",
            "args": {"milestones": [
                {"title": "M1", "due_at": None, "done": False, "topic": "X"},
            ]},
            "id": "c2",
        }]),
        _msg(content="Plan ready.", tool_calls=[]),
    ])
    agent = _build(session, llm)

    update = await agent({
        "messages": [HumanMessage(content="plan on X")],
        "user_id": user.id,
    })

    assert update["agent_trace"]["exit_reason"] == "natural_stop"
    assert update["agent_trace"]["tool_errors"] == 1
    # Second call succeeded → plan persisted
    assert update.get("active_plan_id")


async def test_plan_action_generate_when_no_get_existing_plan_called(session):
    user = UserRepository(session).get_or_create("fp-loop-5")
    GoalRepository(session).create(user_id=user.id, title="G")
    llm = ScriptedLLM([_msg(content="just text", tool_calls=[])])
    agent = _build(session, llm)

    update = await agent({
        "messages": [HumanMessage(content="x")],
        "user_id": user.id,
    })
    # No tools called → fallback inference is "generate"
    assert update["plan_action"] == "generate"


async def test_plan_action_check_in_when_get_existing_plan_returned_nonnull(session):
    user = UserRepository(session).get_or_create("fp-loop-6")
    goal_repo = GoalRepository(session)
    goal = goal_repo.create(user_id=user.id, title="G")
    plan_repo = PlanRepository(session)
    plan_repo.create(goal_id=goal.id, milestones_json=[
        {"title": "old M", "done": False, "topic": "HyDE"},
    ])

    llm = ScriptedLLM([
        _msg(tool_calls=[{"name": "get_existing_plan", "args": {}, "id": "c1"}]),
        _msg(content="progress: 0/1 done", tool_calls=[]),
    ])
    agent = _build(session, llm)

    update = await agent({
        "messages": [HumanMessage(content="how is the plan going")],
        "user_id": user.id,
    })

    assert update["plan_action"] == "check_in"


async def test_plan_action_generate_when_get_existing_plan_returned_null(session):
    """get_existing_plan called BUT returned the literal "null" sentinel → generate."""
    user = UserRepository(session).get_or_create("fp-loop-7")
    GoalRepository(session).create(user_id=user.id, title="G")

    llm = ScriptedLLM([
        _msg(tool_calls=[{"name": "get_existing_plan", "args": {}, "id": "c1"}]),
        # Loop now sees "null" — model decides no existing plan, drafts a new one
        _msg(tool_calls=[{
            "name": "update_study_plan",
            "args": {"milestones": [
                {"title": "M1", "due_at": None, "done": False, "topic": "x"},
            ]},
            "id": "c2",
        }]),
        _msg(content="Plan made.", tool_calls=[]),
    ])
    agent = _build(session, llm)

    update = await agent({
        "messages": [HumanMessage(content="plan")],
        "user_id": user.id,
    })

    assert update["plan_action"] == "generate"


async def test_extract_topic_strips_mindmap_suffix_without_corrupting_english():
    """Regression for the P2.1-⑤i character-set vs word-suffix strip bug.
    Topic 'Spam' must not become 'Sp' after suffix stripping."""
    from app.agent.planner_agent import _extract_topic_for_agent_prompt

    assert _extract_topic_for_agent_prompt("make a plan on Spam") == "Spam"
    assert _extract_topic_for_agent_prompt("帮我做学习计划 on HyDE 画脑图") == "HyDE"
    assert _extract_topic_for_agent_prompt("plan on BM25?") == "BM25"
    assert _extract_topic_for_agent_prompt("帮我做学习计划 on HyDE！") == "HyDE"
