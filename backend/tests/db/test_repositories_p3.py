"""Cut A5 — MistakeRepository.list_due_with_details."""
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base
from app.db.repositories import (
    GoalRepository,
    MistakeRepository,
    QuestionRepository,
    TopicRepository,
    UserRepository,
)


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


def test_list_due_with_details_filters_by_due_at(session):
    user = UserRepository(session).get_or_create("fp-1")
    goal = GoalRepository(session).create(user_id=user.id, title="g", exam_date=None)
    topic = TopicRepository(session).create(goal_id=goal.id, name="HyDE", source_chunks=[])
    q = QuestionRepository(session).create(
        topic_id=topic.id,
        prompt="What is HyDE?",
        options_json=["A) X", "B) Y", "C) Z", "D) W"],
        answer="A",
        explanation="HyDE is...",
    )
    now = datetime.utcnow()
    # Past-due mistake — should appear
    MistakeRepository(session).create(
        user_id=user.id,
        question_id=q.id,
        user_answer="B",
        srs_due_at=now - timedelta(hours=1),
        srs_interval_days=1,
        srs_ease=2.5,
    )
    # Future-due mistake — must NOT appear
    MistakeRepository(session).create(
        user_id=user.id,
        question_id=q.id,
        user_answer="C",
        srs_due_at=now + timedelta(days=7),
        srs_interval_days=7,
        srs_ease=2.5,
    )

    rows = MistakeRepository(session).list_due_with_details(user.id, now=now)
    assert len(rows) == 1
    mistake, question, topic_row = rows[0]
    assert mistake.user_answer == "B"
    assert question.prompt == "What is HyDE?"
    assert topic_row.name == "HyDE"


def test_list_for_user_detailed_returns_topic_and_mastery(session):
    from app.db.repositories import (
        UserRepository, GoalRepository, TopicRepository, MasteryRepository,
    )
    user = UserRepository(session).get_or_create("fp-mastery")
    goal = GoalRepository(session).create(user_id=user.id, title="g", exam_date=None)
    t1 = TopicRepository(session).create(goal_id=goal.id, name="HyDE", source_chunks=[])
    t2 = TopicRepository(session).create(goal_id=goal.id, name="BM25", source_chunks=[])
    MasteryRepository(session).upsert(user_id=user.id, topic_id=t1.id, score=0.8)
    MasteryRepository(session).upsert(user_id=user.id, topic_id=t2.id, score=0.3)

    rows = MasteryRepository(session).list_for_user_detailed(user.id)
    assert len(rows) == 2
    by_name = {topic.name: mastery for topic, mastery in rows}
    assert by_name["HyDE"].score == 0.8
    assert by_name["BM25"].score == 0.3
    assert by_name["HyDE"].last_reviewed is not None
