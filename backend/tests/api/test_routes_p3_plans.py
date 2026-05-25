"""Cut A1 — GET /api/plans/current — happy + 404."""
import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/p3_plans.db")
    monkeypatch.setenv("STUDY_COACH_TEST_MODE", "1")
    from app.db import session as session_mod
    session_mod._engine = None
    session_mod._SessionLocal = None
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_get_plans_current_404_when_no_plan(client):
    resp = client.get("/api/plans/current", headers={"x-fingerprint": "fp-1"})
    assert resp.status_code == 404
    assert resp.json() == {"detail": "no active plan for user"}


def test_get_plans_current_returns_active_plan(client):
    # Seed: create user, goal, plan via repos.
    from app.db.session import session_scope
    from app.db.repositories import (
        UserRepository, GoalRepository, PlanRepository,
    )
    with session_scope() as s:
        user = UserRepository(s).get_or_create("fp-2")
        goal = GoalRepository(s).create(user_id=user.id, title="Master HyDE", exam_date=None)
        PlanRepository(s).update_milestones(
            goal_id=goal.id,
            milestones=[
                {"title": "Read HyDE chapter", "due_at": "2026-05-26T00:00:00", "done": False, "topic": "HyDE"},
                {"title": "Quiz on HyDE", "due_at": None, "done": False, "topic": "HyDE"},
            ],
        )

    resp = client.get("/api/plans/current", headers={"x-fingerprint": "fp-2"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["goal_title"] == "Master HyDE"
    assert len(body["milestones"]) == 2
    assert body["milestones"][0]["title"] == "Read HyDE chapter"
    assert "plan_id" in body and "goal_id" in body and "updated_at" in body
