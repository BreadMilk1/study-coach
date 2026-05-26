import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.db.models import Base
from app.db.repositories import (
    GoalRepository,
    MasteryRepository,
    PlanRepository,
    TopicRepository,
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

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _seed_plan(session):
    user = UserRepository(session).get_or_create("fp-plan-milestones")
    goal = GoalRepository(session).create(user_id=user.id, title="Master HyDE")
    topic = TopicRepository(session).create(goal_id=goal.id, name="HyDE")
    repo = PlanRepository(session)
    plan = repo.update_milestones(
        goal_id=goal.id,
        milestones=[
            {
                "title": "Read HyDE",
                "due_at": "2026-05-30",
                "done": False,
                "topic": "HyDE",
                "topic_id": topic.id,
            },
            {
                "title": "Quiz HyDE",
                "due_at": None,
                "done": False,
                "topic": "HyDE",
                "topic_id": topic.id,
            },
        ],
    )
    return user, goal, topic, plan, repo


def test_update_milestones_creates_normalized_rows_and_json_cache(session):
    _, _, topic, plan, repo = _seed_plan(session)

    rows = repo.list_milestones(plan.id)

    assert len(rows) == 2
    assert rows[0].id
    assert rows[0].title == "Read HyDE"
    assert rows[0].topic_id == topic.id
    assert rows[0].topic_name == "HyDE"
    assert rows[0].sort_order == 0
    assert plan.milestones_json[0]["id"] == rows[0].id
    assert plan.milestones_json[0]["due_at"] == "2026-05-30"
    assert plan.milestones_json[0]["topic"] == "HyDE"
    assert [e.action for e in repo.list_events(plan.id)] == ["created", "created"]


def test_update_milestones_refreshes_topic_and_logs_applied_event(session):
    _, _, topic, plan, repo = _seed_plan(session)
    new_topic = TopicRepository(session).create(goal_id=plan.goal_id, name="RAG")
    milestone = repo.list_milestones(plan.id)[0]

    updated = repo.update_milestones(
        goal_id=plan.goal_id,
        milestones=[
            {
                "id": milestone.id,
                "title": "Read RAG",
                "due_at": "2026-06-01",
                "done": False,
                "topic": "RAG",
                "topic_id": new_topic.id,
            },
        ],
    )
    row = repo.list_milestones(updated.id)[0]

    assert row.id == milestone.id
    assert row.topic_id == new_topic.id
    assert row.topic_id != topic.id
    assert row.topic_name == "RAG"
    assert repo.list_events(plan.id)[0].action == "applied"


def test_update_milestones_can_omit_old_rows_without_breaking_event_history(session):
    _, _, _, plan, repo = _seed_plan(session)
    kept, removed = repo.list_milestones(plan.id)
    repo.set_milestone_done(
        plan_id=plan.id,
        milestone_id=removed.id,
        done=True,
        actor="user",
        reason="User marked milestone complete",
    )

    updated = repo.update_milestones(
        goal_id=plan.goal_id,
        milestones=[
            {
                "id": kept.id,
                "title": kept.title,
                "due_at": "2026-05-30",
                "done": False,
                "topic": kept.topic_name,
                "topic_id": kept.topic_id,
            },
        ],
    )

    assert [row.id for row in repo.list_milestones(updated.id)] == [kept.id]
    assert any(
        event.action == "completed" and event.milestone_id is None
        for event in repo.list_events(plan.id, limit=20)
    )


def test_set_milestone_done_completes_and_reopens_with_events(session):
    _, _, _, plan, repo = _seed_plan(session)
    milestone = repo.list_milestones(plan.id)[0]

    updated = repo.set_milestone_done(
        plan_id=plan.id,
        milestone_id=milestone.id,
        done=True,
        actor="user",
        reason="User marked milestone complete",
    )

    assert updated.done is True
    assert updated.completed_at is not None
    assert repo.get_by_goal(plan.goal_id).milestones_json[0]["done"] is True
    assert repo.list_events(plan.id)[0].action == "completed"

    reopened = repo.set_milestone_done(
        plan_id=plan.id,
        milestone_id=milestone.id,
        done=False,
        actor="user",
        reason="User reopened milestone",
    )

    assert reopened.done is False
    assert reopened.completed_at is None
    events = repo.list_events(plan.id)
    assert [e.action for e in events[:2]] == ["reopened", "completed"]


def test_plan_repository_returns_mastery_score_by_topic_name(session):
    user, _, topic, plan, repo = _seed_plan(session)
    MasteryRepository(session).upsert(user_id=user.id, topic_id=topic.id, score=0.25)

    dto_rows = repo.list_milestone_dicts(plan.id, user_id=user.id)

    assert dto_rows[0]["mastery_score"] == pytest.approx(0.25)
    assert dto_rows[0]["validation_recommended"] is False

    repo.set_milestone_done(
        plan_id=plan.id,
        milestone_id=repo.list_milestones(plan.id)[0].id,
        done=True,
        actor="user",
        reason="User marked milestone complete",
    )

    completed_rows = repo.list_milestone_dicts(plan.id, user_id=user.id)
    assert completed_rows[0]["validation_recommended"] is True
