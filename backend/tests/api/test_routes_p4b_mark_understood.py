"""P4b — POST /api/mistakes/{id}/mark-understood."""
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.auth import issue_token
from app.main import create_app
from tests.helpers import ensure_user


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/p4b_mark_understood.db")
    monkeypatch.setenv("STUDY_COACH_TEST_MODE", "1")
    from app.db import session as session_mod
    session_mod._engine = None
    session_mod._SessionLocal = None
    app = create_app()
    from app.db.session import session_scope
    with session_scope() as session:
        ensure_user(session, "default-user")
    with TestClient(app) as c:
        yield c


def test_mark_understood_pushes_due_far_future(client):
    """After mark-understood, the mistake should not appear in GET /api/mistakes/due."""
    from app.db.repositories import (
        GoalRepository,
        MistakeRepository,
        QuestionRepository,
        TopicRepository,
        UserRepository,
    )
    from app.db.session import session_scope

    with session_scope() as s:
        user = UserRepository(s).get_or_create("fp-mark")
        goal = GoalRepository(s).create(user_id=user.id, title="g", exam_date=None)
        topic = TopicRepository(s).create(goal_id=goal.id, name="HyDE", source_chunks=[])
        q = QuestionRepository(s).create(
            topic_id=topic.id,
            prompt="What is HyDE?",
            options_json=["A) X", "B) Y", "C) Z", "D) W"],
            answer="A",
            explanation="HyDE is...",
        )
        now = datetime.utcnow()
        mistake = MistakeRepository(s).create(
            user_id=user.id,
            question_id=q.id,
            user_answer="B",
            srs_due_at=now - timedelta(hours=1),
            srs_interval_days=1,
            srs_ease=2.5,
        )
        user_id = user.id
        mistake_id = mistake.id

    token = issue_token(user_id, "guest")
    headers = {"Authorization": f"Bearer {token}"}

    # Verify it shows up as due before marking understood
    resp = client.get("/api/mistakes/due", headers=headers)
    assert resp.status_code == 200
    before_ids = [item["mistake_id"] for item in resp.json()]
    assert mistake_id in before_ids

    # Mark understood
    resp = client.post(
        f"/api/mistakes/{mistake_id}/mark-understood",
        headers=headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "mastery_score" in body
    assert "next_due_at" in body

    # Verify it no longer appears as due
    resp = client.get("/api/mistakes/due", headers=headers)
    assert resp.status_code == 200
    after_ids = [item["mistake_id"] for item in resp.json()]
    assert mistake_id not in after_ids


def test_mark_understood_returns_404_for_unknown_id(client):
    token = issue_token("default-user", "guest")
    resp = client.post(
        "/api/mistakes/unknown-id/mark-understood",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 404
