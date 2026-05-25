"""Cut A9 — GET /api/mastery."""
from datetime import datetime, timedelta
import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/p3_mastery.db")
    monkeypatch.setenv("STUDY_COACH_TEST_MODE", "1")
    from app.db import session as session_mod
    session_mod._engine = None
    session_mod._SessionLocal = None
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_get_mastery_empty(client):
    resp = client.get("/api/mastery", headers={"x-fingerprint": "fp-1"})
    assert resp.status_code == 200
    assert resp.json() == {"scores": [], "weak_topics": [], "overdue_milestones_count": 0}


def test_get_mastery_returns_scores_and_weak_topics(client):
    from app.db.session import session_scope
    from app.db.repositories import (
        UserRepository, GoalRepository, TopicRepository, MasteryRepository,
    )
    with session_scope() as s:
        user = UserRepository(s).get_or_create("fp-2")
        goal = GoalRepository(s).create(user_id=user.id, title="g", exam_date=None)
        topics = {
            name: TopicRepository(s).create(goal_id=goal.id, name=name, source_chunks=[])
            for name in ["HyDE", "BM25", "RRF", "Reranker"]
        }
        MasteryRepository(s).upsert(user_id=user.id, topic_id=topics["HyDE"].id, score=0.85)
        MasteryRepository(s).upsert(user_id=user.id, topic_id=topics["BM25"].id, score=0.25)
        MasteryRepository(s).upsert(user_id=user.id, topic_id=topics["RRF"].id, score=0.4)
        MasteryRepository(s).upsert(user_id=user.id, topic_id=topics["Reranker"].id, score=0.7)

    resp = client.get("/api/mastery", headers={"x-fingerprint": "fp-2"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["scores"]) == 4
    # weak_topics: score < 0.5, sorted asc, max 5. BM25 (0.25) then RRF (0.4)
    assert body["weak_topics"] == ["BM25", "RRF"]
    assert body["overdue_milestones_count"] == 0


def test_get_mastery_overdue_count_from_active_plan(client):
    from app.db.session import session_scope
    from app.db.repositories import (
        UserRepository, GoalRepository, PlanRepository,
    )
    with session_scope() as s:
        user = UserRepository(s).get_or_create("fp-3")
        goal = GoalRepository(s).create(user_id=user.id, title="g", exam_date=None)
        past = (datetime.utcnow() - timedelta(days=1)).isoformat()
        future = (datetime.utcnow() + timedelta(days=7)).isoformat()
        PlanRepository(s).update_milestones(
            goal_id=goal.id,
            milestones=[
                {"title": "overdue done", "due_at": past, "done": True, "topic": None},
                {"title": "overdue!", "due_at": past, "done": False, "topic": None},
                {"title": "future", "due_at": future, "done": False, "topic": None},
            ],
        )

    resp = client.get("/api/mastery", headers={"x-fingerprint": "fp-3"})
    assert resp.json()["overdue_milestones_count"] == 1
