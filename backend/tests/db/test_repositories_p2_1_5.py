"""Cut ⑤b — PlanRepository.update_milestones upsert tests."""
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.db.models import Base
from app.db.repositories import GoalRepository, PlanRepository, UserRepository


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


def test_update_milestones_creates_plan_when_absent(session):
    user = UserRepository(session).get_or_create("fp-plan-create")
    goal = GoalRepository(session).create(user_id=user.id, title="G")
    repo = PlanRepository(session)

    plan = repo.update_milestones(
        goal_id=goal.id,
        milestones=[{"title": "Read §1", "due_at": "2026-05-30", "done": False, "topic": "HyDE"}],
    )

    assert plan.goal_id == goal.id
    assert len(plan.milestones_json) == 1
    assert plan.milestones_json[0]["title"] == "Read §1"
    assert isinstance(plan.updated_at, datetime)
    # Singleton: get_by_goal sees the new row
    fetched = repo.get_by_goal(goal.id)
    assert fetched.id == plan.id


def test_update_milestones_overwrites_existing_and_bumps_updated_at(session):
    user = UserRepository(session).get_or_create("fp-plan-upsert")
    goal = GoalRepository(session).create(user_id=user.id, title="G")
    repo = PlanRepository(session)
    original = repo.create(
        goal_id=goal.id,
        milestones_json=[{"title": "old", "done": False}],
    )
    old_updated = original.updated_at

    # Force a tick so updated_at advances on SQLite second-precision clocks.
    import time
    time.sleep(0.01)

    updated = repo.update_milestones(
        goal_id=goal.id,
        milestones=[{"title": "new", "done": True}],
    )

    assert updated.id == original.id  # same row, not a new one
    assert len(updated.milestones_json) == 1
    assert updated.milestones_json[0]["title"] == "new"
    assert updated.updated_at >= old_updated
