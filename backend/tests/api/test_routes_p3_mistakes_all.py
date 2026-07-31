"""Regression — Mistake Bank needs to show recorded mistakes, not only due-today rows."""
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.auth import issue_token
from app.main import create_app
from tests.helpers import ensure_user


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/p3_mistakes_all.db")
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


def test_get_mistakes_due_can_include_future_rows(client):
    from app.db.repositories import (
        GoalRepository,
        MistakeRepository,
        QuestionRepository,
        TopicRepository,
        UserRepository,
    )
    from app.db.session import session_scope

    with session_scope() as s:
        user = UserRepository(s).get_or_create("fp-future")
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
        MistakeRepository(s).create(
            user_id=user.id,
            question_id=q.id,
            user_answer="B",
            srs_due_at=now + timedelta(days=1),
            srs_interval_days=1,
            srs_ease=2.5,
        )
        user_id = user.id

    token = issue_token(user_id, "guest")
    resp = client.get(
        "/api/mistakes/due?include_future=true",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["question"]["prompt"] == "What is HyDE?"
    assert body[0]["topic_name"] == "HyDE"
