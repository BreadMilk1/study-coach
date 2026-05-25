"""Cut P2.2-①a — unit tests for the 5 LLM-facing tool wrappers in planner_agent.

Each test exercises:
  1. Closure injection — the LLM-visible args do not include user_id / repos / llm,
     yet the tool can use them via the factory closure.
  2. JSON return shape — every tool returns a JSON-serializable string the LLM
     can consume as a ToolMessage payload.
  3. Delegation correctness — the wrapper calls into the existing pure functions
     (update_study_plan_fn / generate_mindmap_fn / compute_progress_fn) rather
     than reimplementing the logic.

Business-logic depth is NOT re-tested here; that lives in test_plan_tools.py
and test_progress.py. The contract under test is the wrapper, not the work.
"""
import json
from datetime import datetime

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.agent.planner_agent import _make_planner_tools
from app.db.models import Base
from app.db.repositories import (
    GoalRepository,
    PlanRepository,
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


class StubRetriever:
    def __init__(self, chunks=None):
        self.chunks = chunks or []
        self.calls: list[tuple[str, int]] = []

    def search(self, query, top_k=5):
        self.calls.append((query, top_k))
        return self.chunks[:top_k]


class StubLLM:
    def __init__(self, response_text: str = ""):
        self.response_text = response_text
        self.calls = 0

    async def ainvoke(self, messages, **_kwargs):
        self.calls += 1
        return AIMessage(content=self.response_text)


async def test_retriever_search_uses_closure_retriever_and_returns_json():
    retriever = StubRetriever(chunks=[
        {"chunk_id": "c1", "content": "HyDE = Hypothetical Document Embedding.", "page": 1},
        {"chunk_id": "c2", "content": "Re-rank with BM25.", "page": 3},
    ])
    tools = _make_planner_tools(
        user_id="u1", llm=None, retriever=retriever,
        plan_repo=None, goal_repo=None,
        mastery_scores={}, recent_mistakes=[],
    )
    tool = next(t for t in tools if t.name == "retriever_search")

    # Schema: query is in the public args, retriever is NOT
    assert "query" in tool.args
    assert "top_k" in tool.args
    assert "retriever" not in tool.args
    assert "user_id" not in tool.args

    out = await tool.ainvoke({"query": "HyDE", "top_k": 1})
    parsed = json.loads(out)
    assert retriever.calls == [("HyDE", 1)]
    assert isinstance(parsed, list)
    assert parsed[0]["chunk_id"] == "c1"


async def test_get_existing_plan_returns_null_string_when_no_active_goal(session):
    user = UserRepository(session).get_or_create("fp-get-plan-1")
    goal_repo = GoalRepository(session)
    plan_repo = PlanRepository(session)
    tools = _make_planner_tools(
        user_id=user.id, llm=None, retriever=None,
        plan_repo=plan_repo, goal_repo=goal_repo,
        mastery_scores={}, recent_mistakes=[],
    )
    tool = next(t for t in tools if t.name == "get_existing_plan")

    out = await tool.ainvoke({})
    # No goal, no plan → tool returns literal "null" (LLM-friendly sentinel)
    assert out == "null"


async def test_update_study_plan_upserts_via_closure_and_returns_count(session):
    user = UserRepository(session).get_or_create("fp-upd-1")
    goal_repo = GoalRepository(session)
    plan_repo = PlanRepository(session)
    goal_repo.create(user_id=user.id, title="G")

    tools = _make_planner_tools(
        user_id=user.id, llm=None, retriever=None,
        plan_repo=plan_repo, goal_repo=goal_repo,
        mastery_scores={}, recent_mistakes=[],
    )
    tool = next(t for t in tools if t.name == "update_study_plan")

    milestones = [
        {"title": "Read HyDE §1", "due_at": "2026-05-25", "done": False, "topic": "HyDE"},
        {"title": "Practice HyDE", "due_at": "2026-05-28", "done": False, "topic": "HyDE"},
    ]
    out = await tool.ainvoke({"milestones": milestones})
    parsed = json.loads(out)
    assert parsed["milestones_count"] == 2
    assert parsed["plan_id"]

    # Verify persistence via the same repo (closure path actually wrote)
    goal = goal_repo.list_active_for_user(user.id)[0]
    saved = plan_repo.get_by_goal(goal.id)
    assert saved is not None
    assert len(saved.milestones_json) == 2


async def test_generate_mindmap_delegates_to_closure_llm():
    llm = StubLLM(response_text="```mermaid\nmindmap\n  root((HyDE))\n    Q1\n```\n- HyDE\n  - Q1")
    tools = _make_planner_tools(
        user_id="u", llm=llm, retriever=None,
        plan_repo=None, goal_repo=None,
        mastery_scores={}, recent_mistakes=[],
    )
    tool = next(t for t in tools if t.name == "generate_mindmap")

    # Closure means llm is NOT in args
    assert "llm" not in tool.args
    assert "topic" in tool.args
    assert "milestones" in tool.args

    out = await tool.ainvoke({
        "topic": "HyDE",
        "milestones": [{"title": "M1", "due_at": None, "done": False, "topic": "HyDE"}],
    })
    parsed = json.loads(out)
    assert llm.calls == 1
    assert "mindmap" in parsed["mermaid_src"].lower()
    assert isinstance(parsed["markdown_outline"], str)


async def test_compute_progress_uses_closure_state_and_returns_json(session):
    user = UserRepository(session).get_or_create("fp-prog-1")
    goal_repo = GoalRepository(session)
    plan_repo = PlanRepository(session)
    goal = goal_repo.create(user_id=user.id, title="G")
    plan_repo.create(
        goal_id=goal.id,
        milestones_json=[
            {"title": "M1", "done": True, "topic": "HyDE"},
            {"title": "M2", "done": False, "topic": "BM25", "due_at": "2020-01-01"},
        ],
    )

    tools = _make_planner_tools(
        user_id=user.id, llm=None, retriever=None,
        plan_repo=plan_repo, goal_repo=goal_repo,
        mastery_scores={"BM25": 0.2, "HyDE": 0.8},
        recent_mistakes=["m1", "m2"],
        now_fn=lambda: datetime(2026, 5, 23, 12, 0),
    )
    tool = next(t for t in tools if t.name == "compute_progress")

    # All inputs are closure-injected; tool has no public args
    assert tool.args == {}

    out = await tool.ainvoke({})
    parsed = json.loads(out)
    assert parsed["done_count"] == 1
    assert parsed["total_count"] == 2
    assert "M2" in parsed["overdue"]
    assert parsed["weak_topics"] == ["BM25"]
    assert parsed["recent_mistake_count"] == 2
