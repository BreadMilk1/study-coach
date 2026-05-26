"""Cut ⑤e — planner_node tests (GENERATE + CHECK-IN).

Mirrors test_quiz_master.py: factory built with real in-memory SQLite repos,
LLM stubbed. Exercises both decide() paths + edge cases.
"""
from datetime import datetime
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.agent.planner import build_planner
from app.db.models import Base
from app.db.repositories import (
    GoalRepository,
    MasteryRepository,
    MistakeRepository,
    PlanRepository,
    UserRepository,
)

_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "app" / "agent" / "prompts"


_GEN_JSON = """[
  {"title": "Read HyDE §1-§3", "due_at": "2026-05-25", "done": false, "topic": "HyDE"},
  {"title": "Implement HyDEGenerator", "due_at": "2026-05-28", "done": false, "topic": "HyDE"},
  {"title": "Compare HyDE vs BM25", "due_at": "2026-06-01", "done": false, "topic": "HyDE"}
]"""

_CHECK_IN_JSON = """[
  {"title": "Read HyDE §1-§3", "due_at": "2026-05-25", "done": true, "topic": "HyDE"},
  {"title": "Implement HyDEGenerator", "due_at": "2026-05-24", "done": false, "topic": "HyDE"},
  {"title": "Compare HyDE vs BM25", "due_at": "2026-06-01", "done": false, "topic": "HyDE"}
]"""


class StubPlannerLLM:
    def __init__(self, response_text: str = _GEN_JSON):
        self.response_text = response_text
        self.last_prompt: str | None = None

    async def ainvoke(self, messages, **_kwargs):
        self.last_prompt = messages[-1].content if messages else ""
        return AIMessage(content=self.response_text)


class StubRetriever:
    def __init__(self, chunks=None):
        self.chunks = chunks or []
        self.last_query: str | None = None

    def search(self, query, top_k=5):
        self.last_query = query
        return self.chunks[:top_k]


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


def _build_node(session, llm=None, retriever=None):
    return build_planner(
        llm=llm or StubPlannerLLM(),
        plan_repo=PlanRepository(session),
        goal_repo=GoalRepository(session),
        mastery_repo=MasteryRepository(session),
        mistake_repo=MistakeRepository(session),
        retriever=retriever,
        now_fn=lambda: datetime(2026, 5, 22, 12, 0),
    )


def test_planner_check_in_prompt_requires_id_and_topic_id_preservation():
    prompt = (_PROMPTS_DIR / "planner_check_in.txt").read_text(encoding="utf-8")

    assert "id/topic_id" in prompt


async def test_planner_generate_creates_plan_and_sets_active_plan_id(session):
    user = UserRepository(session).get_or_create("fp-plan-gen")
    node = _build_node(session)

    update = await node({
        "messages": [HumanMessage(content="帮我做学习计划 on HyDE")],
        "user_id": user.id,
    })

    assert update["plan_action"] == "generate"
    assert update["active_plan_id"]
    # Plan row created
    plan_repo = PlanRepository(session)
    goal = GoalRepository(session).list_active_for_user(user.id)[0]
    saved = plan_repo.get_by_goal(goal.id)
    assert saved is not None
    assert len(saved.milestones_json) == 3
    # Output text mentions milestones
    text = update["messages"][0].content
    assert "Read HyDE" in text


async def test_planner_generate_skips_mindmap_without_keyword(session):
    user = UserRepository(session).get_or_create("fp-plan-no-mm")
    node = _build_node(session)

    update = await node({
        "messages": [HumanMessage(content="帮我做学习计划 on HyDE")],
        "user_id": user.id,
    })

    text = update["messages"][0].content
    assert "mindmap" not in text.lower()
    assert "```" not in text  # no mermaid fence


async def test_planner_generate_calls_mindmap_on_keyword(session):
    user = UserRepository(session).get_or_create("fp-plan-mm")

    # Sequence two stub responses: first call = milestones JSON, second = mindmap text.
    class TwoStepLLM:
        def __init__(self):
            self.calls = 0

        async def ainvoke(self, messages, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content=_GEN_JSON)
            return AIMessage(content="```mermaid\nmindmap\n  root((HyDE))\n```\n- HyDE")

    llm = TwoStepLLM()
    node = _build_node(session, llm=llm)

    update = await node({
        "messages": [HumanMessage(content="帮我做学习计划 on HyDE 画脑图")],
        "user_id": user.id,
    })

    assert llm.calls == 2
    text = update["messages"][0].content
    assert "mindmap" in text.lower()


async def test_planner_generate_uses_retriever_chunks_in_prompt(session):
    user = UserRepository(session).get_or_create("fp-plan-rag")
    retriever = StubRetriever(chunks=[
        {"chunk_id": "t7:p1", "content": "HyDE = Hypothetical Document Embedding."},
    ])
    llm = StubPlannerLLM()
    node = _build_node(session, llm=llm, retriever=retriever)

    await node({
        "messages": [HumanMessage(content="帮我做学习计划 on HyDE")],
        "user_id": user.id,
    })

    assert retriever.last_query == "HyDE"
    assert "Hypothetical Document Embedding" in llm.last_prompt


async def test_planner_check_in_adjusts_existing_plan(session):
    user = UserRepository(session).get_or_create("fp-plan-ci")
    goal = GoalRepository(session).create(user_id=user.id, title="G")
    plan_repo = PlanRepository(session)
    initial = plan_repo.create(
        goal_id=goal.id,
        milestones_json=[
            {"title": "Read HyDE §1-§3", "due_at": "2026-05-25", "done": False, "topic": "HyDE"},
            {"title": "Implement HyDEGenerator", "due_at": "2026-05-20", "done": False, "topic": "HyDE"},
            {"title": "Compare HyDE vs BM25", "due_at": "2026-06-01", "done": False, "topic": "HyDE"},
        ],
    )
    llm = StubPlannerLLM(_CHECK_IN_JSON)
    node = _build_node(session, llm=llm)

    update = await node({
        "messages": [HumanMessage(content="Read HyDE §1-§3 完成了，Implement HyDEGenerator 需要延期")],
        "user_id": user.id,
        "active_plan_id": initial.id,
        "mastery_scores": {"BM25": 0.2, "HyDE": 0.6},
    })

    assert update["plan_action"] == "check_in"
    refreshed = plan_repo.get_by_goal(goal.id)
    assert len(refreshed.milestones_json) == 3
    assert refreshed.milestones_json[0]["done"] is True
    assert refreshed.milestones_json[1]["due_at"] == "2026-05-24"
    assert [m["title"] for m in refreshed.milestones_json] == [
        "Read HyDE §1-§3",
        "Implement HyDEGenerator",
        "Compare HyDE vs BM25",
    ]
    text = update["messages"][0].content
    assert "Done:" in text or "进度" in text  # progress card surfaced


async def test_planner_check_in_with_unparseable_llm_output_keeps_plan_and_notes_skip(session):
    user = UserRepository(session).get_or_create("fp-plan-ci-bad")
    goal = GoalRepository(session).create(user_id=user.id, title="G")
    plan_repo = PlanRepository(session)
    initial = plan_repo.create(
        goal_id=goal.id,
        milestones_json=[{"title": "Read HyDE §1-§3", "due_at": "2026-05-25", "done": False, "topic": "HyDE"}],
    )
    llm = StubPlannerLLM(response_text="Sorry, I cannot do that today.")
    node = _build_node(session, llm=llm)

    update = await node({
        "messages": [HumanMessage(content="进度怎么样了")],
        "user_id": user.id,
        "active_plan_id": initial.id,
        "mastery_scores": {},
    })

    refreshed = plan_repo.get_by_goal(goal.id)
    assert len(refreshed.milestones_json) == 1  # unchanged
    text = update["messages"][0].content
    assert "Auto-adjust skipped" in text


async def test_planner_check_in_falls_back_to_generate_when_plan_missing(session):
    """active_plan_id set but plan was deleted externally → recover via GENERATE path."""
    user = UserRepository(session).get_or_create("fp-plan-recover")
    node = _build_node(session)

    update = await node({
        "messages": [HumanMessage(content="帮我做学习计划 on HyDE")],
        "user_id": user.id,
        "active_plan_id": "deadbeef-not-in-db",
    })

    # GENERATE took over; new plan created
    assert update["plan_action"] == "generate"
    assert update["active_plan_id"] != "deadbeef-not-in-db"


async def test_planner_force_generate_when_create_keyword_present_with_active_plan(session):
    """Fix A: 帮我做学习计划 on X with active_plan_id → GENERATE (overwrites), not CHECK-IN."""
    user = UserRepository(session).get_or_create("fp-plan-force-gen")
    goal = GoalRepository(session).create(user_id=user.id, title="G")
    plan_repo = PlanRepository(session)
    plan_repo.create(
        goal_id=goal.id,
        milestones_json=[{"title": "old M1", "done": False, "topic": "OldTopic"}],
    )
    node = _build_node(session)  # StubPlannerLLM returns _GEN_JSON (3 fresh HyDE milestones)

    update = await node({
        "messages": [HumanMessage(content="帮我做学习计划 on HyDE")],
        "user_id": user.id,
        "active_plan_id": "stale-id-but-old-plan-exists",
    })

    assert update["plan_action"] == "generate"
    refreshed = plan_repo.get_by_goal(goal.id)
    # old plan overwritten (upsert), now has the 3 fresh milestones
    assert len(refreshed.milestones_json) == 3
    assert refreshed.milestones_json[0]["title"] == "Read HyDE §1-§3"


async def test_planner_check_in_progress_count_matches_final_milestone_count(session):
    """Fix C: Done: X / Y where Y == len(Updated Plan list)."""
    user = UserRepository(session).get_or_create("fp-plan-ci-count")
    goal = GoalRepository(session).create(user_id=user.id, title="G")
    plan_repo = PlanRepository(session)
    initial = plan_repo.create(
        goal_id=goal.id,
        milestones_json=[
            {"title": "Old A", "done": False, "topic": "HyDE"},
            {"title": "Old B", "done": False, "topic": "HyDE"},
        ],
    )
    check_in_json = """[
      {"title": "Old A", "done": true, "topic": "HyDE"},
      {"title": "Old B", "done": false, "topic": "HyDE"}
    ]"""
    llm = StubPlannerLLM(check_in_json)
    node = _build_node(session, llm=llm)

    update = await node({
        "messages": [HumanMessage(content="进度怎么样了")],
        "user_id": user.id,
        "active_plan_id": initial.id,
    })

    text = update["messages"][0].content
    # Find the "Done: X / Y" line
    done_line = next((line for line in text.splitlines() if line.startswith("- Done:")), None)
    assert done_line is not None
    # Y must equal the number of milestones in the Updated Plan section
    refreshed = plan_repo.get_by_goal(goal.id)
    expected_total = len(refreshed.milestones_json)
    assert f"/ {expected_total}" in done_line, f"expected '/ {expected_total}' in {done_line!r}"
