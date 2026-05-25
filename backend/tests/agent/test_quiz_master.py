"""Cut ④c — QuizMaster node tests.

QuizMaster is a deterministic factory-built async node:
  - state.active_quiz_question_id present → GRADE the user's reply
  - absent → GENERATE a new quiz from the user's message ("quiz me on X")

Tests exercise the factory directly with real in-memory SQLite repos
(no graph wiring; that's Cut ④e). LLM is stubbed.
"""
from datetime import datetime

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.agent.quiz_master import build_quiz_master
from app.db.models import Base
from app.db.repositories import (
    GoalRepository,
    MasteryRepository,
    MistakeRepository,
    QuestionRepository,
    TopicRepository,
    UserRepository,
)


_GEN_JSON = """[
  {
    "prompt": "What does HyDE rewrite?",
    "options": ["A) Queries", "B) Documents", "C) Embeddings", "D) Answers"],
    "answer": "A",
    "explanation": "HyDE rewrites the user query into a hypothetical answer."
  }
]"""


class StubGeneratorLLM:
    def __init__(self, response_text: str = _GEN_JSON):
        self.response_text = response_text
        self.last_prompt: str | None = None

    async def ainvoke(self, messages, **_kwargs):
        self.last_prompt = messages[-1].content if messages else ""
        return AIMessage(content=self.response_text)

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
    return build_quiz_master(
        llm=llm or StubGeneratorLLM(),
        topic_repo=TopicRepository(session),
        question_repo=QuestionRepository(session),
        mistake_repo=MistakeRepository(session),
        mastery_repo=MasteryRepository(session),
        goal_repo=GoalRepository(session),
        retriever=retriever,
        now_fn=lambda: datetime(2026, 5, 21, 12, 0),
    )


async def test_quiz_master_generate_path_auto_creates_goal_and_topic(session):
    users = UserRepository(session)
    user = users.get_or_create("fp-quizgen")
    goals = GoalRepository(session)
    topics = TopicRepository(session)
    node = _build_node(session)

    update = await node({
        "messages": [HumanMessage(content="quiz me on HyDE")],
        "user_id": user.id,
    })

    # Goal auto-created
    active = goals.list_active_for_user(user.id)
    assert len(active) == 1
    # Topic auto-created within that goal
    topic = topics.get_by_name(goal_id=active[0].id, name="HyDE")
    assert topic is not None
    # active_quiz_question_id set in returned state
    assert update["active_quiz_question_id"] is not None
    # AIMessage carries the question + options + reply hint
    ai_text = update["messages"][0].content
    assert "What does HyDE rewrite?" in ai_text
    assert "A) Queries" in ai_text
    assert "Reply with A, B, C, or D" in ai_text
    # Cut ④g: phase signal for downstream judge gating
    assert update.get("quiz_action") == "generate"


async def test_quiz_master_grade_path_correct_bumps_mastery_and_clears_active(session):
    users = UserRepository(session)
    goals = GoalRepository(session)
    topics = TopicRepository(session)
    questions = QuestionRepository(session)
    user = users.get_or_create("fp-correct")
    goal = goals.create(user_id=user.id, title="X")
    topic = topics.create(goal_id=goal.id, name="HyDE")
    question = questions.create(
        topic_id=topic.id,
        prompt="?",
        options_json=["A", "B", "C", "D"],
        answer="A",
        explanation="Because A.",
    )

    node = _build_node(session)

    update = await node({
        "messages": [HumanMessage(content="A")],
        "user_id": user.id,
        "active_quiz_question_id": question.id,
    })

    mastery = MasteryRepository(session)
    assert mastery.get_for_user(user.id) == {"HyDE": pytest.approx(0.1)}
    # Mistakes table untouched
    mistakes = MistakeRepository(session)
    assert mistakes.get_due_for_user(user.id, now=datetime(2026, 6, 1)) == []
    # State cleared
    assert update["active_quiz_question_id"] is None
    assert "Correct" in update["messages"][0].content
    # Cut ④g: grade phase signal so judge can skip
    assert update.get("quiz_action") == "grade"


async def test_quiz_master_grade_path_wrong_records_mistake_and_lowers_mastery(session):
    users = UserRepository(session)
    goals = GoalRepository(session)
    topics = TopicRepository(session)
    questions = QuestionRepository(session)
    mastery = MasteryRepository(session)
    user = users.get_or_create("fp-wrong")
    goal = goals.create(user_id=user.id, title="X")
    topic = topics.create(goal_id=goal.id, name="HyDE")
    question = questions.create(
        topic_id=topic.id,
        prompt="?",
        options_json=["A", "B", "C", "D"],
        answer="A",
        explanation="Because A.",
    )
    mastery.upsert(user_id=user.id, topic_id=topic.id, score=0.5)

    node = _build_node(session)

    update = await node({
        "messages": [HumanMessage(content="B")],
        "user_id": user.id,
        "active_quiz_question_id": question.id,
    })

    # Mastery lowered
    assert mastery.get_for_user(user.id) == {"HyDE": pytest.approx(0.4)}
    # Mistake persisted with SM-2 schedule (interval=1 → due tomorrow)
    mistakes = MistakeRepository(session)
    due = mistakes.get_due_for_user(user.id, now=datetime(2026, 5, 23))
    assert len(due) == 1
    # State cleared
    assert update["active_quiz_question_id"] is None
    ai_text = update["messages"][0].content
    assert "Incorrect" in ai_text
    assert "A" in ai_text  # correct answer surfaced
    assert "Because A." in ai_text  # explanation surfaced
    assert update.get("quiz_action") == "grade"


class StubRetriever:
    def __init__(self, chunks=None):
        self.chunks = chunks or []
        self.last_query: str | None = None

    def search(self, query, top_k=5):
        self.last_query = query
        return self.chunks[:top_k]


async def test_quiz_master_generate_path_uses_retriever_chunks_and_persists_topic_source(session):
    """Cut ④h: when retriever returns chunks for the topic, the generator
    sees them in its prompt AND the Topic.source_chunks field is updated so
    the chosen grounding is queryable later."""
    users = UserRepository(session)
    user = users.get_or_create("fp-rag-grounded")
    topics = TopicRepository(session)
    llm = StubGeneratorLLM()
    retriever = StubRetriever(chunks=[
        {"chunk_id": "topic7:p3:0", "content": "HyDE = Hypothetical Document Embedding."},
        {"chunk_id": "topic7:p4:0", "content": "It rewrites queries via hypothetical answers."},
    ])
    node = _build_node(session, llm=llm, retriever=retriever)

    await node({
        "messages": [HumanMessage(content="quiz me on HyDE")],
        "user_id": user.id,
    })

    # Retriever was queried with the topic name
    assert retriever.last_query == "HyDE"
    # LLM prompt embeds chunk content
    assert llm.last_prompt is not None
    assert "Hypothetical Document Embedding" in llm.last_prompt
    # Topic.source_chunks persisted (chunk_ids only, content stays in Chroma)
    goals = GoalRepository(session)
    goal = goals.list_active_for_user(user.id)[0]
    topic = topics.get_by_name(goal_id=goal.id, name="HyDE")
    assert topic.source_chunks == ["topic7:p3:0", "topic7:p4:0"]


async def test_quiz_master_generate_path_falls_back_when_retriever_returns_empty(session):
    """Empty retrieval still produces a question (ungrounded fallback)."""
    users = UserRepository(session)
    user = users.get_or_create("fp-rag-empty")
    llm = StubGeneratorLLM()
    retriever = StubRetriever(chunks=[])  # nothing indexed for this topic
    node = _build_node(session, llm=llm, retriever=retriever)

    update = await node({
        "messages": [HumanMessage(content="quiz me on UnknownTopic")],
        "user_id": user.id,
    })

    # Question still generated, no crash
    assert update.get("active_quiz_question_id") is not None
    # Prompt lacks source-grounded content (no chunks)
    assert "Hypothetical Document Embedding" not in llm.last_prompt


async def test_quiz_master_generate_path_works_without_retriever_injected(session):
    """Backward compat: no retriever = old behavior (ungrounded)."""
    users = UserRepository(session)
    user = users.get_or_create("fp-no-retriever")
    llm = StubGeneratorLLM()
    node = _build_node(session, llm=llm, retriever=None)

    update = await node({
        "messages": [HumanMessage(content="quiz me on HyDE")],
        "user_id": user.id,
    })

    assert update.get("active_quiz_question_id") is not None
