"""Cut P2.2-①d — graph-level e2e for the agent_loop planner mode.

Three contracts under test:
  1. config.configurable.planner_mode='agent_loop' routes to planner_agent
     callable, NOT to the deterministic planner.
  2. agent_trace ends up in the final state so the eval harness can read it.
  3. Judge sees plan_action and applies the plan rubric (not the tutor one),
     same as deterministic — proving the agent path is observationally
     equivalent to deterministic from the judge's perspective.

Stub LLM is shared between planner_agent (tool-calling) and judge — scripted
to emit a generate→update→summary sequence, then a high-scoring rubric JSON.
"""
from datetime import datetime

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.agent.graph import build_graph
from app.agent.planner import build_planner
from app.agent.planner_agent import build_planner_agent
from app.db.models import Base
from app.db.repositories import (
    GoalRepository,
    MasteryRepository,
    MistakeRepository,
    PlanRepository,
    UserRepository,
)


def _msg(content="", tool_calls=None, input_tokens=10, output_tokens=5):
    m = AIMessage(content=content, tool_calls=tool_calls or [])
    m.usage_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    return m


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.idx = 0

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, messages, **_kwargs):
        if self.idx >= len(self.responses):
            raise AssertionError("ScriptedLLM exhausted")
        m = self.responses[self.idx]
        self.idx += 1
        return m

    async def astream(self, messages, **_kwargs):
        yield self.responses[0]


class StubRetriever:
    def search(self, query, top_k=5):
        return [{"chunk_id": "c1", "content": "HyDE def", "page": 1}]


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


def _wire(session, agent_llm, det_llm=None, judge_text=None):
    retriever = StubRetriever()
    graph = build_graph(retriever=retriever, llm=det_llm or agent_llm)
    planner = build_planner(
        llm=det_llm or agent_llm,
        plan_repo=PlanRepository(session),
        goal_repo=GoalRepository(session),
        mastery_repo=MasteryRepository(session),
        mistake_repo=MistakeRepository(session),
        retriever=retriever,
        now_fn=lambda: datetime(2026, 5, 23),
    )
    planner_agent = build_planner_agent(
        llm=agent_llm,
        plan_repo=PlanRepository(session),
        goal_repo=GoalRepository(session),
        mastery_repo=MasteryRepository(session),
        mistake_repo=MistakeRepository(session),
        retriever=retriever,
        now_fn=lambda: datetime(2026, 5, 23),
        max_iter=10,
    )
    judge_llm = ScriptedLLM([_msg(content=judge_text)]) if judge_text else None
    return graph, planner, planner_agent, judge_llm


async def test_graph_mode_agent_loop_dispatches_to_planner_agent_not_deterministic(session):
    user = UserRepository(session).get_or_create("fp-g-mode-agent")

    # Deterministic LLM scripted to crash if invoked → proves it wasn't picked
    class CrashIfCalled:
        async def ainvoke(self, *_a, **_kw):
            raise AssertionError("deterministic planner must not run in agent_loop mode")
        async def astream(self, *_a, **_kw):
            raise AssertionError("deterministic planner must not run in agent_loop mode")
        def bind_tools(self, _t): return self

    agent_llm = ScriptedLLM([
        _msg(tool_calls=[{
            "name": "update_study_plan",
            "args": {"milestones": [
                {"title": "M1", "due_at": "2026-05-30", "done": False, "topic": "HyDE"},
            ]},
            "id": "c1",
        }]),
        _msg(content="Plan done.", tool_calls=[]),
    ])
    graph, _, planner_agent, _ = _wire(session, agent_llm, det_llm=CrashIfCalled())

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="make a plan on HyDE")], "user_id": user.id},
        config={"configurable": {
            "planner_mode": "agent_loop",
            "planner_agent": planner_agent,
            "planner": CrashIfCalled(),  # would crash if dispatcher picked deterministic
        }},
    )

    assert result.get("plan_action") == "generate"
    assert result.get("active_plan_id")
    assert "agent_trace" in result and result["agent_trace"]["exit_reason"] == "natural_stop"


async def test_graph_default_mode_routes_to_deterministic_planner(session):
    """Absent planner_mode key → fallback to deterministic. Existing 157 tests
    pass this implicitly; making it explicit here defends against accidental
    dispatcher rewrites."""
    user = UserRepository(session).get_or_create("fp-g-mode-default")
    det_llm = ScriptedLLM([_msg(content="""[
      {"title": "M1", "due_at": "2026-05-30", "done": false, "topic": "HyDE"}
    ]""")])

    class CrashAgentIfCalled:
        async def ainvoke(self, *_a, **_kw):
            raise AssertionError("agent_loop must not run when mode is unset")
        def bind_tools(self, _t): return self

    graph, planner, planner_agent, _ = _wire(session, CrashAgentIfCalled(), det_llm=det_llm)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="帮我做学习计划 on HyDE")], "user_id": user.id},
        config={"configurable": {"planner": planner}},  # NO planner_mode key
    )

    assert result.get("plan_action") == "generate"
    assert result.get("agent_trace") is None or "agent_trace" not in result


async def test_graph_judge_runs_plan_rubric_on_agent_loop_output(session):
    """Judge sees plan_action='generate' from agent_loop → applies PLAN_DIMENSIONS
    rubric just like deterministic. Pass-verdict moves through memory_writer."""
    user = UserRepository(session).get_or_create("fp-g-judge-agent")
    agent_llm = ScriptedLLM([
        _msg(tool_calls=[{
            "name": "update_study_plan",
            "args": {"milestones": [
                {"title": "M1", "due_at": "2026-05-30", "done": False, "topic": "HyDE"},
            ]},
            "id": "c1",
        }]),
        _msg(content="Plan generated.", tool_calls=[]),
    ])
    judge_pass = (
        '{"milestone_specificity": 5, "milestone_granularity": 5, '
        '"time_feasibility": 5, "topic_coverage": 4, "actionability": 5, '
        '"reasoning": "ok"}'
    )
    graph, planner, planner_agent, judge_llm = _wire(session, agent_llm, judge_text=judge_pass)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="make a plan on HyDE")], "user_id": user.id},
        config={"configurable": {
            "planner_mode": "agent_loop",
            "planner_agent": planner_agent,
            "planner": planner,
            "judge_llm": judge_llm,
        }},
    )

    assert result.get("judge_score", 0) >= 0.6
    assert result.get("plan_action") == "generate"
