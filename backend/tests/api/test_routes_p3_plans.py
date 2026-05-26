"""Cut A1 — GET /api/plans/current — happy + 404."""
import pytest
from fastapi.testclient import TestClient

from app.auth import issue_token
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
    token = issue_token("default-user", "guest")
    resp = client.get("/api/plans/current", headers={"Authorization": f"Bearer {token}"})
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
        user_id = user.id

    token = issue_token(user_id, "guest")
    resp = client.get("/api/plans/current", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["goal_title"] == "Master HyDE"
    assert len(body["milestones"]) == 2
    assert body["milestones"][0]["title"] == "Read HyDE chapter"
    assert "plan_id" in body and "goal_id" in body and "updated_at" in body


def test_get_plans_current_returns_milestone_ids_and_mastery_hint(client):
    from app.db.session import session_scope
    from app.db.repositories import (
        UserRepository, GoalRepository, PlanRepository, TopicRepository, MasteryRepository,
    )

    with session_scope() as s:
        user = UserRepository(s).get_or_create("fp-plan-id")
        goal = GoalRepository(s).create(user_id=user.id, title="Master HyDE", exam_date=None)
        topic = TopicRepository(s).create(goal_id=goal.id, name="HyDE")
        MasteryRepository(s).upsert(user_id=user.id, topic_id=topic.id, score=0.25)
        PlanRepository(s).update_milestones(
            goal_id=goal.id,
            milestones=[{"title": "Read HyDE", "done": True, "topic": "HyDE", "topic_id": topic.id}],
        )
        user_id = user.id

    token = issue_token(user_id, "guest")
    resp = client.get("/api/plans/current", headers={"Authorization": f"Bearer {token}"})

    assert resp.status_code == 200
    milestone = resp.json()["milestones"][0]
    assert milestone["id"]
    assert milestone["topic_id"] == topic.id
    assert milestone["topic"] == "HyDE"
    assert milestone["mastery_score"] == 0.25
    assert milestone["validation_recommended"] is True


def test_patch_milestone_done_toggles_state_without_changing_mastery(client):
    from app.db.session import session_scope
    from app.db.repositories import (
        UserRepository, GoalRepository, PlanRepository, TopicRepository, MasteryRepository,
    )

    with session_scope() as s:
        user = UserRepository(s).get_or_create("fp-toggle")
        goal = GoalRepository(s).create(user_id=user.id, title="Master HyDE", exam_date=None)
        topic = TopicRepository(s).create(goal_id=goal.id, name="HyDE")
        MasteryRepository(s).upsert(user_id=user.id, topic_id=topic.id, score=0.25)
        plan = PlanRepository(s).update_milestones(
            goal_id=goal.id,
            milestones=[{"title": "Read HyDE", "done": False, "topic": "HyDE", "topic_id": topic.id}],
        )
        milestone_id = PlanRepository(s).list_milestones(plan.id)[0].id
        user_id = user.id

    token = issue_token(user_id, "guest")
    auth_headers = {"Authorization": f"Bearer {token}"}

    resp = client.patch(
        f"/api/plans/{plan.id}/milestones/{milestone_id}",
        headers=auth_headers,
        json={"done": True},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["plan"]["milestones"][0]["done"] is True
    assert body["plan"]["milestones"][0]["completed_at"] is not None
    assert body["event"]["action"] == "completed"
    assert body["validation_hint"]["show_quick_quiz"] is True

    reopened = client.patch(
        f"/api/plans/{plan.id}/milestones/{milestone_id}",
        headers=auth_headers,
        json={"done": False},
    )

    assert reopened.status_code == 200
    reopened_body = reopened.json()
    assert reopened_body["plan"]["milestones"][0]["done"] is False
    assert reopened_body["plan"]["milestones"][0]["completed_at"] is None
    assert reopened_body["event"]["action"] == "reopened"
    assert reopened_body["validation_hint"]["show_quick_quiz"] is False

    mastery = client.get("/api/mastery", headers=auth_headers).json()
    assert mastery["scores"][0]["score"] == 0.25


def test_plan_milestone_routes_reject_other_users_plan(client):
    from app.db.session import session_scope
    from app.db.repositories import UserRepository, GoalRepository, PlanRepository

    with session_scope() as s:
        owner = UserRepository(s).get_or_create("fp-owner")
        intruder = UserRepository(s).get_or_create("fp-intruder")
        owner_goal = GoalRepository(s).create(user_id=owner.id, title="Owner plan")
        GoalRepository(s).create(user_id=intruder.id, title="Intruder plan")
        plan = PlanRepository(s).update_milestones(
            goal_id=owner_goal.id,
            milestones=[{"title": "Read HyDE", "done": False, "topic": "HyDE"}],
        )
        milestone_id = PlanRepository(s).list_milestones(plan.id)[0].id
        intruder_id = intruder.id

    intruder_token = issue_token(intruder_id, "guest")
    patch_resp = client.patch(
        f"/api/plans/{plan.id}/milestones/{milestone_id}",
        headers={"Authorization": f"Bearer {intruder_token}"},
        json={"done": True},
    )
    events_resp = client.get(
        f"/api/plans/{plan.id}/events",
        headers={"Authorization": f"Bearer {intruder_token}"},
    )

    assert patch_resp.status_code == 404
    assert events_resp.status_code == 404


def test_plan_milestone_routes_find_owned_plan_beyond_first_active_goal(client):
    from app.db.session import session_scope
    from app.db.repositories import UserRepository, GoalRepository, PlanRepository

    with session_scope() as s:
        user = UserRepository(s).get_or_create("fp-many-goals")
        GoalRepository(s).create(user_id=user.id, title="Earlier goal")
        target_goal = GoalRepository(s).create(user_id=user.id, title="Target goal")
        plan = PlanRepository(s).update_milestones(
            goal_id=target_goal.id,
            milestones=[{"title": "Read HyDE", "done": False, "topic": "HyDE"}],
        )
        milestone_id = PlanRepository(s).list_milestones(plan.id)[0].id
        user_id = user.id

    token = issue_token(user_id, "guest")
    auth_headers = {"Authorization": f"Bearer {token}"}
    patch_resp = client.patch(
        f"/api/plans/{plan.id}/milestones/{milestone_id}",
        headers=auth_headers,
        json={"done": True},
    )
    events_resp = client.get(
        f"/api/plans/{plan.id}/events",
        headers=auth_headers,
    )

    assert patch_resp.status_code == 200
    assert events_resp.status_code == 200


def test_patch_milestone_done_returns_event_from_current_toggle(client, monkeypatch):
    from app.db.session import session_scope
    from app.db.repositories import UserRepository, GoalRepository, PlanRepository

    with session_scope() as s:
        user = UserRepository(s).get_or_create("fp-current-event")
        goal = GoalRepository(s).create(user_id=user.id, title="Master HyDE")
        plan = PlanRepository(s).update_milestones(
            goal_id=goal.id,
            milestones=[{"title": "Read HyDE", "done": False, "topic": "HyDE"}],
        )
        milestone_id = PlanRepository(s).list_milestones(plan.id)[0].id
        user_id = user.id

    original_list_events = PlanRepository.list_events

    def _wrong_recent_event(self, plan_id, *, limit=20):
        events = original_list_events(self, plan_id, limit=limit)
        if limit == 1 and events:
            events[0].action = "applied"
        return events

    monkeypatch.setattr(PlanRepository, "list_events", _wrong_recent_event)

    token = issue_token(user_id, "guest")
    resp = client.patch(
        f"/api/plans/{plan.id}/milestones/{milestone_id}",
        headers={"Authorization": f"Bearer {token}"},
        json={"done": True},
    )

    assert resp.status_code == 200
    assert resp.json()["event"]["action"] == "completed"


def test_get_plan_events_returns_recent_changes(client):
    from app.db.session import session_scope
    from app.db.repositories import UserRepository, GoalRepository, PlanRepository

    with session_scope() as s:
        user = UserRepository(s).get_or_create("fp-events")
        goal = GoalRepository(s).create(user_id=user.id, title="Master HyDE", exam_date=None)
        plan = PlanRepository(s).update_milestones(
            goal_id=goal.id,
            milestones=[{"title": "Read HyDE", "done": False, "topic": "HyDE"}],
        )
        milestone_id = PlanRepository(s).list_milestones(plan.id)[0].id
        PlanRepository(s).set_milestone_done(
            plan_id=plan.id,
            milestone_id=milestone_id,
            done=True,
            actor="user",
            reason="User marked milestone complete",
        )
        PlanRepository(s).set_milestone_done(
            plan_id=plan.id,
            milestone_id=milestone_id,
            done=False,
            actor="user",
            reason="User reopened milestone",
        )
        user_id = user.id

    token = issue_token(user_id, "guest")
    resp = client.get(f"/api/plans/{plan.id}/events?limit=1", headers={"Authorization": f"Bearer {token}"})

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["id"]
    assert body[0]["plan_id"] == plan.id
    assert body[0]["milestone_id"] == milestone_id
    assert body[0]["actor"] == "user"
    assert body[0]["action"] == "reopened"
    assert body[0]["before_json"]["done"] is True
    assert body[0]["after_json"]["done"] is False
    assert body[0]["reason"] == "User reopened milestone"
    assert body[0]["created_at"]
