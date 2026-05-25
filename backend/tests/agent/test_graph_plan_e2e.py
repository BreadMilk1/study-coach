"""Cut ⑤f — Graph e2e for the Plan branch.

Exercises router state-aware override + planner injection + judge plan rubric
+ memory_writer no-op. LLM and judge are stubbed; nothing hits Ollama.
"""
from datetime import datetime

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.agent.graph import build_graph
from app.agent.planner import build_planner
from app.db.models import Base
from app.db.repositories import (
    GoalRepository,
    MasteryRepository,
    MistakeRepository,
    PlanRepository,
    UserRepository,
)


_GEN_JSON = """[
  {"title": "Read §1", "due_at": "2026-05-25", "done": false, "topic": "HyDE"},
  {"title": "Practice", "due_at": "2026-05-28", "done": false, "topic": "HyDE"},
  {"title": "Review", "due_at": "2026-06-01", "done": false, "topic": "HyDE"}
]"""


class StubLLM:
    def __init__(self, response_text: str):
        self.response_text = response_text

    async def ainvoke(self, messages, **_kwargs):
        return AIMessage(content=self.response_text)

    async def astream(self, messages, **_kwargs):
        # Plan path doesn't use astream; placeholder for compat.
        yield AIMessage(content=self.response_text)


class StubJudgeLLM(StubLLM):
    pass


class StubRetriever:
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


def _build(session, gen_llm_text=_GEN_JSON, judge_text=None):
    llm = StubLLM(gen_llm_text)
    retriever = StubRetriever()
    graph = build_graph(retriever=retriever, llm=llm)
    planner = build_planner(
        llm=llm,
        plan_repo=PlanRepository(session),
        goal_repo=GoalRepository(session),
        mastery_repo=MasteryRepository(session),
        mistake_repo=MistakeRepository(session),
        retriever=retriever,
        now_fn=lambda: datetime(2026, 5, 22),
    )
    judge_llm = StubJudgeLLM(judge_text) if judge_text else None
    return graph, planner, judge_llm


async def test_graph_routes_to_planner_on_plan_keyword(session):
    user = UserRepository(session).get_or_create("fp-graph-plan")
    graph, planner, _ = _build(session)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="帮我做学习计划 on HyDE")], "user_id": user.id},
        config={"configurable": {"planner": planner}},
    )

    assert result.get("plan_action") == "generate"
    assert result.get("active_plan_id")
    assert result["intent"] == "plan"


async def test_graph_state_aware_router_forces_plan_when_active_plan_id_set(session):
    """active_plan_id present + plain msg (no plan keyword) still routes to plan."""
    user = UserRepository(session).get_or_create("fp-graph-active-plan")
    goal = GoalRepository(session).create(user_id=user.id, title="G")
    plan_repo = PlanRepository(session)
    initial = plan_repo.create(
        goal_id=goal.id,
        milestones_json=[{"title": "M1", "due_at": "2026-05-25", "done": False, "topic": "HyDE"}],
    )
    graph, planner, _ = _build(session)

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="调整第二个里程碑标记为完成")],
            "user_id": user.id,
            "active_plan_id": initial.id,
        },
        config={"configurable": {"planner": planner}},
    )

    assert result["intent"] == "plan"
    assert result.get("plan_action") == "check_in"


async def test_graph_judge_runs_plan_rubric_on_generate_pass(session):
    user = UserRepository(session).get_or_create("fp-graph-plan-judge")
    judge_pass = (
        '{"milestone_specificity": 5, "milestone_granularity": 5, '
        '"time_feasibility": 5, "topic_coverage": 4, "actionability": 5, '
        '"reasoning": "ok"}'
    )
    graph, planner, judge_llm = _build(session, judge_text=judge_pass)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="帮我做学习计划 on HyDE")], "user_id": user.id},
        config={"configurable": {"planner": planner, "judge_llm": judge_llm}},
    )

    assert result.get("judge_score", 0) >= 0.6
    assert result.get("plan_action") == "generate"


async def test_graph_judge_skips_on_check_in(session):
    """plan_action='check_in' → judge short-circuits to pass without calling LLM."""
    user = UserRepository(session).get_or_create("fp-graph-plan-ci-skip")
    goal = GoalRepository(session).create(user_id=user.id, title="G")
    plan_repo = PlanRepository(session)
    initial = plan_repo.create(
        goal_id=goal.id,
        milestones_json=[{"title": "M1", "done": False, "topic": "HyDE"}],
    )

    class CrashJudgeLLM:
        async def ainvoke(self, messages, **_kwargs):
            raise AssertionError("judge must not be invoked on plan check_in")

    graph, planner, _ = _build(session)
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="进度怎么样了")],
            "user_id": user.id,
            "active_plan_id": initial.id,
        },
        config={"configurable": {"planner": planner, "judge_llm": CrashJudgeLLM()}},
    )

    assert result.get("plan_action") == "check_in"
    assert result.get("judge_score") == 1.0


async def test_graph_router_lets_tutor_escape_plan_chain(session):
    """Fix B: `What is HyDE?` with active_plan_id → tutor, not plan."""
    user = UserRepository(session).get_or_create("fp-graph-tutor-escape")
    goal = GoalRepository(session).create(user_id=user.id, title="G")
    plan_repo = PlanRepository(session)
    initial = plan_repo.create(
        goal_id=goal.id,
        milestones_json=[{"title": "M1", "done": False, "topic": "HyDE"}],
    )
    graph, planner, _ = _build(session)

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="What is HyDE?")],
            "user_id": user.id,
            "active_plan_id": initial.id,
        },
        config={"configurable": {"planner": planner}},
    )

    # Router should NOT force plan when message is a tutor question.
    assert result["intent"] == "tutor"


async def test_graph_router_keeps_plan_chain_on_edit_message(session):
    """Fix B: edit-shaped message ('调整第三个milestone') with active_plan_id → plan (CHECK-IN)."""
    user = UserRepository(session).get_or_create("fp-graph-plan-edit")
    goal = GoalRepository(session).create(user_id=user.id, title="G")
    plan_repo = PlanRepository(session)
    initial = plan_repo.create(
        goal_id=goal.id,
        milestones_json=[{"title": "M1", "done": False, "topic": "HyDE"}],
    )
    graph, planner, _ = _build(session)

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="调整第三个milestone的截止日期")],
            "user_id": user.id,
            "active_plan_id": initial.id,
        },
        config={"configurable": {"planner": planner}},
    )

    assert result["intent"] == "plan"
    assert result.get("plan_action") == "check_in"
