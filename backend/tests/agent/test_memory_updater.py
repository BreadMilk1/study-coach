"""Cut ③ — Memory Updater node tests.

Two factory functions:
  - build_memory_hydrator(mastery_repo, mistake_repo) → state node that loads
    mastery_scores + recent_mistakes from DB into state (graph ENTRY).
  - build_memory_writer(mastery_repo, mistake_repo) → state node that drains
    state.pending_mastery_delta + state.pending_mistake into DB (graph EXIT).

Both no-op when state lacks user_id (anonymous / unit-test paths).
"""
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.agent.memory_updater import build_memory_hydrator, build_memory_writer
from app.db.models import Base
from app.db.repositories import (
    GoalRepository,
    MasteryRepository,
    MistakeRepository,
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


def _seed_user_with_topic_and_question(session):
    users = UserRepository(session)
    goals = GoalRepository(session)
    topics = TopicRepository(session)
    questions = QuestionRepository(session)
    user = users.get_or_create("fp-mem")
    goal = goals.create(user_id=user.id, title="Final prep")
    topic = topics.create(goal_id=goal.id, name="HyDE")
    question = questions.create(
        topic_id=topic.id,
        prompt="What is HyDE?",
        options_json=["A", "B", "C", "D"],
        answer="A",
        explanation="",
    )
    return user, topic, question


def test_memory_hydrator_loads_mastery_and_due_mistakes_into_state(session):
    user, topic, question = _seed_user_with_topic_and_question(session)
    mastery = MasteryRepository(session)
    mistakes = MistakeRepository(session)

    mastery.upsert(user_id=user.id, topic_id=topic.id, score=0.7)
    past_due = mistakes.create(
        user_id=user.id,
        question_id=question.id,
        user_answer="B",
        srs_due_at=datetime(2026, 5, 1),
    )

    hydrator = build_memory_hydrator(
        mastery_repo=mastery,
        mistake_repo=mistakes,
        now_fn=lambda: datetime(2026, 5, 20),
    )
    update = hydrator({"user_id": user.id})

    assert update["mastery_scores"] == {"HyDE": pytest.approx(0.7)}
    assert update["recent_mistakes"] == [past_due.id]


def test_memory_hydrator_noop_when_no_user_id(session):
    mastery = MasteryRepository(session)
    mistakes = MistakeRepository(session)
    hydrator = build_memory_hydrator(mastery_repo=mastery, mistake_repo=mistakes)

    update = hydrator({})

    assert update == {}


def test_memory_writer_applies_pending_mastery_delta_and_clears(session):
    user, topic, _ = _seed_user_with_topic_and_question(session)
    mastery = MasteryRepository(session)
    mistakes = MistakeRepository(session)
    mastery.upsert(user_id=user.id, topic_id=topic.id, score=0.5)

    writer = build_memory_writer(mastery_repo=mastery, mistake_repo=mistakes)
    update = writer({
        "user_id": user.id,
        "pending_mastery_delta": {topic.id: 0.2},
    })

    assert mastery.get_for_user(user.id) == {"HyDE": pytest.approx(0.7)}
    assert update["pending_mastery_delta"] == {}


def test_memory_writer_persists_pending_mistake_with_default_srs(session):
    user, _, question = _seed_user_with_topic_and_question(session)
    mastery = MasteryRepository(session)
    mistakes = MistakeRepository(session)

    writer = build_memory_writer(mastery_repo=mastery, mistake_repo=mistakes)
    update = writer({
        "user_id": user.id,
        "pending_mistake": {"question_id": question.id, "user_answer": "B"},
    })

    # Default SRS schedule = now + 1 day; ensure mistake is queryable after 2 days.
    after_two_days = datetime.utcnow() + timedelta(days=2)
    due = mistakes.get_due_for_user(user.id, now=after_two_days)
    assert len(due) == 1
    assert update["pending_mistake"] is None


def test_memory_writer_noop_without_user_id(session):
    mastery = MasteryRepository(session)
    mistakes = MistakeRepository(session)
    writer = build_memory_writer(mastery_repo=mastery, mistake_repo=mistakes)

    update = writer({"pending_mastery_delta": {"any-topic-id": 0.5}})

    assert update == {}
