"""Cut ② — Repository tests for P2.1-③ schema (9 new tables).

Each table gets a minimal repository test that drives:
  - the SQLAlchemy 2.x model declaration in app/db/models.py
  - the matching Repository class in app/db/repositories.py

Per-test red→green cycles intentionally grow this file table-by-table.
Tests sharing this file all use the in-memory SQLite + create_all fixture
(same pattern as tests/db/test_repositories.py).
"""
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.models import Base
from app.db.repositories import (
    ChatSessionRepository,
    CitationRepository,
    GoalRepository,
    MasteryRepository,
    MessageRepository,
    MistakeRepository,
    PlanRepository,
    QuestionRepository,
    TopicRepository,
    UserRepository,
)


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def test_goal_repo_creates_and_lists_active_goals(session):
    users = UserRepository(session)
    goals = GoalRepository(session)
    user = users.get_or_create("fp-goal")

    g1 = goals.create(user_id=user.id, title="Pass CSCI3360 final")
    g2 = goals.create(
        user_id=user.id,
        title="Pass MATH calculus",
        exam_date=datetime(2026, 6, 1),
    )

    active = goals.list_active_for_user(user.id)
    assert {g.id for g in active} == {g1.id, g2.id}
    assert all(g.status == "active" for g in active)
    assert g2.exam_date == datetime(2026, 6, 1)


def test_topic_repo_create_and_find_by_name_within_goal(session):
    users = UserRepository(session)
    goals = GoalRepository(session)
    topics = TopicRepository(session)
    user = users.get_or_create("fp-topic")
    goal = goals.create(user_id=user.id, title="ML midterm")

    t1 = topics.create(goal_id=goal.id, name="HyDE", source_chunks=["c1", "c2"])
    found = topics.get_by_name(goal_id=goal.id, name="HyDE")
    missing = topics.get_by_name(goal_id=goal.id, name="not-a-topic")

    assert found is not None and found.id == t1.id
    assert found.source_chunks == ["c1", "c2"]
    assert missing is None


def test_topic_repo_set_source_chunks_overwrites(session):
    """Cut ④h: refresh source_chunks each time a new quiz is generated so the
    column reflects the latest grounding (not the first one ever)."""
    users = UserRepository(session)
    goals = GoalRepository(session)
    topics = TopicRepository(session)
    user = users.get_or_create("fp-topic-update")
    goal = goals.create(user_id=user.id, title="X")
    topic = topics.create(goal_id=goal.id, name="HyDE", source_chunks=["old1"])

    topics.set_source_chunks(topic_id=topic.id, chunk_ids=["c1", "c2", "c3"])

    refreshed = topics.get_by_name(goal_id=goal.id, name="HyDE")
    assert refreshed.source_chunks == ["c1", "c2", "c3"]


def test_plan_repo_create_and_get_by_goal(session):
    users = UserRepository(session)
    goals = GoalRepository(session)
    plans = PlanRepository(session)
    user = users.get_or_create("fp-plan")
    goal = goals.create(user_id=user.id, title="Final prep")

    milestones = [
        {"title": "Topic 1", "due_at": "2026-06-01", "done": False, "topic": "HyDE"},
        {"title": "Topic 2", "due_at": "2026-06-08", "done": False, "topic": "RAG"},
    ]
    plan = plans.create(goal_id=goal.id, milestones_json=milestones)

    found = plans.get_by_goal(goal.id)
    assert found is not None and found.id == plan.id
    assert found.milestones_json == milestones


def test_question_repo_create_and_get_by_id(session):
    users = UserRepository(session)
    goals = GoalRepository(session)
    topics = TopicRepository(session)
    questions = QuestionRepository(session)
    user = users.get_or_create("fp-q")
    goal = goals.create(user_id=user.id, title="Quiz session")
    topic = topics.create(goal_id=goal.id, name="HyDE")

    q = questions.create(
        topic_id=topic.id,
        prompt="What is HyDE?",
        options_json=["A) ...", "B) ...", "C) ...", "D) ..."],
        answer="A",
        explanation="HyDE = Hypothetical Document Embedding",
    )
    found = questions.get_by_id(q.id)
    assert found is not None
    assert found.prompt == "What is HyDE?"
    assert found.answer == "A"
    assert len(found.options_json) == 4


def test_mastery_repo_upsert_then_get_for_user_keyed_by_topic_name(session):
    users = UserRepository(session)
    goals = GoalRepository(session)
    topics = TopicRepository(session)
    mastery = MasteryRepository(session)
    user = users.get_or_create("fp-m")
    goal = goals.create(user_id=user.id, title="Final")
    t_hyde = topics.create(goal_id=goal.id, name="HyDE")
    t_rag = topics.create(goal_id=goal.id, name="RAG")

    mastery.upsert(user_id=user.id, topic_id=t_hyde.id, score=0.7)
    mastery.upsert(user_id=user.id, topic_id=t_rag.id, score=0.4)
    # Upsert again: overwrite, not insert.
    mastery.upsert(user_id=user.id, topic_id=t_hyde.id, score=0.85)

    assert mastery.get_for_user(user.id) == {"HyDE": 0.85, "RAG": 0.4}


def test_mastery_repo_apply_delta_increments_existing(session):
    users = UserRepository(session)
    goals = GoalRepository(session)
    topics = TopicRepository(session)
    mastery = MasteryRepository(session)
    user = users.get_or_create("fp-md")
    goal = goals.create(user_id=user.id, title="Final")
    t = topics.create(goal_id=goal.id, name="HyDE")

    mastery.upsert(user_id=user.id, topic_id=t.id, score=0.5)
    new_score = mastery.apply_delta(user_id=user.id, topic_id=t.id, delta=0.2)

    assert new_score == pytest.approx(0.7)
    assert mastery.get_for_user(user.id) == {"HyDE": pytest.approx(0.7)}


def test_mastery_repo_apply_delta_creates_row_when_missing(session):
    users = UserRepository(session)
    goals = GoalRepository(session)
    topics = TopicRepository(session)
    mastery = MasteryRepository(session)
    user = users.get_or_create("fp-mc")
    goal = goals.create(user_id=user.id, title="Final")
    t = topics.create(goal_id=goal.id, name="HyDE")

    new_score = mastery.apply_delta(user_id=user.id, topic_id=t.id, delta=0.3)

    assert new_score == pytest.approx(0.3)
    assert mastery.get_for_user(user.id) == {"HyDE": pytest.approx(0.3)}


def test_mistake_repo_filters_to_due_in_order(session):
    users = UserRepository(session)
    goals = GoalRepository(session)
    topics = TopicRepository(session)
    questions = QuestionRepository(session)
    mistakes = MistakeRepository(session)
    user = users.get_or_create("fp-mst")
    goal = goals.create(user_id=user.id, title="X")
    topic = topics.create(goal_id=goal.id, name="HyDE")
    q = questions.create(
        topic_id=topic.id,
        prompt="?",
        options_json=["A", "B", "C", "D"],
        answer="A",
        explanation="",
    )

    now = datetime(2026, 5, 20, 12, 0)
    older_due = mistakes.create(
        user_id=user.id,
        question_id=q.id,
        user_answer="B",
        srs_due_at=now - timedelta(days=3),
    )
    newer_due = mistakes.create(
        user_id=user.id,
        question_id=q.id,
        user_answer="B",
        srs_due_at=now - timedelta(hours=1),
    )
    not_due = mistakes.create(
        user_id=user.id,
        question_id=q.id,
        user_answer="B",
        srs_due_at=now + timedelta(days=3),
    )

    due_ids = mistakes.get_due_for_user(user.id, now=now)

    # Oldest-due first, future ones excluded.
    assert due_ids == [older_due.id, newer_due.id]
    assert not_due.id not in due_ids


def test_chat_session_repo_create_and_get(session):
    users = UserRepository(session)
    chats = ChatSessionRepository(session)
    user = users.get_or_create("fp-s")

    chat = chats.create(user_id=user.id)
    found = chats.get_by_id(chat.id)

    assert found is not None
    assert found.user_id == user.id
    assert found.summary is None
    assert found.started_at is not None


def test_message_repo_appends_and_lists_in_insertion_order(session):
    users = UserRepository(session)
    chats = ChatSessionRepository(session)
    msgs = MessageRepository(session)
    user = users.get_or_create("fp-msg")
    chat = chats.create(user_id=user.id)

    m1 = msgs.create(session_id=chat.id, role="user", content="hi")
    m2 = msgs.create(session_id=chat.id, role="assistant", content="hello")

    listed = msgs.list_by_session(chat.id)
    assert [m.id for m in listed] == [m1.id, m2.id]
    assert [m.role for m in listed] == ["user", "assistant"]


def test_citation_repo_bulk_create_for_message(session):
    users = UserRepository(session)
    chats = ChatSessionRepository(session)
    msgs = MessageRepository(session)
    cits = CitationRepository(session)
    user = users.get_or_create("fp-c")
    chat = chats.create(user_id=user.id)
    msg = msgs.create(session_id=chat.id, role="assistant", content="answer")

    rows = cits.bulk_create_for_message(
        message_id=msg.id,
        citations=[
            {"chunk_id": "c1", "page": 3, "span_start": 0, "span_end": 100},
            {"chunk_id": "c2", "page": 4, "span_start": 100, "span_end": 200},
        ],
    )

    assert len(rows) == 2
    assert {r.chunk_id for r in rows} == {"c1", "c2"}
    assert all(r.message_id == msg.id for r in rows)
