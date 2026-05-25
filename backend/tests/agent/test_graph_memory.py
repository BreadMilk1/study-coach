"""Cut ④ — Graph topology with memory_hydrator (entry) + memory_writer (exit).

These tests prove the new graph wiring:
    START → memory_hydrator → router → {tutor|quiz_stub|plan_stub} → judge → memory_writer → END

memory_hydrator/writer are injected via `RunnableConfig.configurable` (same
pattern as judge_llm in P2.1-②) — when absent, the nodes no-op so existing
P2.0/P2.1-①/② tests stay green without any rewrite.
"""
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.agent.graph import build_graph
from app.agent.memory_updater import build_memory_hydrator, build_memory_writer
from app.db.models import Base
from app.db.repositories import (
    GoalRepository,
    MasteryRepository,
    MistakeRepository,
    TopicRepository,
    UserRepository,
)


class StubLLM:
    async def astream(self, messages, **_kwargs):
        yield AIMessageChunk(content="answer")


class StubRetriever:
    def __init__(self, chunks=None):
        self.chunks = chunks or []

    def search(self, query, top_k=5):
        return self.chunks[:top_k]


@pytest.fixture
def session():
    # LangGraph runs nodes in worker threads. in-memory SQLite needs StaticPool +
    # check_same_thread=False so all "connections" share the same backing DB.
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _seed_user_with_topic(session):
    users = UserRepository(session)
    goals = GoalRepository(session)
    topics = TopicRepository(session)
    user = users.get_or_create("fp-graph")
    goal = goals.create(user_id=user.id, title="X")
    topic = topics.create(goal_id=goal.id, name="HyDE")
    return user, topic


async def test_memory_hydrator_populates_state_when_configured(session):
    user, topic = _seed_user_with_topic(session)
    mastery = MasteryRepository(session)
    mistakes = MistakeRepository(session)
    mastery.upsert(user_id=user.id, topic_id=topic.id, score=0.7)
    hydrator = build_memory_hydrator(mastery_repo=mastery, mistake_repo=mistakes)

    graph = build_graph(retriever=StubRetriever(), llm=StubLLM())

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="What is HyDE?")], "user_id": user.id},
        config={"configurable": {"memory_hydrator": hydrator}},
    )

    assert result.get("mastery_scores") == {"HyDE": pytest.approx(0.7)}
    # Tutor still produced an AIMessage; graph still exits cleanly.
    assert any(isinstance(m, AIMessage) for m in result["messages"])


async def test_memory_writer_drains_pending_mastery_delta_on_tutor_path(session):
    user, topic = _seed_user_with_topic(session)
    mastery = MasteryRepository(session)
    mistakes = MistakeRepository(session)
    writer = build_memory_writer(mastery_repo=mastery, mistake_repo=mistakes)

    graph = build_graph(retriever=StubRetriever(), llm=StubLLM())

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="What is HyDE?")],
            "user_id": user.id,
            "pending_mastery_delta": {topic.id: 0.4},
        },
        config={"configurable": {"memory_writer": writer}},
    )

    assert mastery.get_for_user(user.id) == {"HyDE": pytest.approx(0.4)}
    assert result.get("pending_mastery_delta") == {}


async def test_quiz_stub_path_also_runs_memory_writer(session):
    """Quiz/Plan stub branches must pass through memory_writer too (uniform exit)."""
    user, topic = _seed_user_with_topic(session)
    mastery = MasteryRepository(session)
    mistakes = MistakeRepository(session)
    writer = build_memory_writer(mastery_repo=mastery, mistake_repo=mistakes)

    graph = build_graph(retriever=StubRetriever(), llm=StubLLM())

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="quiz me on RAG")],
            "user_id": user.id,
            "pending_mastery_delta": {topic.id: 0.3},
        },
        config={"configurable": {"memory_writer": writer}},
    )

    assert mastery.get_for_user(user.id) == {"HyDE": pytest.approx(0.3)}
    assert result.get("pending_mastery_delta") == {}
    # Quiz branch still emitted the stub message (P2.1-① contract).
    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert ai_msgs and "P2.1-④" in ai_msgs[-1].content
