"""Learning routes require a signed bearer whose user row still exists."""

import pytest
from fastapi.testclient import TestClient

from app.auth import issue_token
from app.db.models import User
from app.db.session import session_scope
from app.main import create_app


class FakeRuntime:
    def __init__(self) -> None:
        self.retriever = object()

    def vector_count(self) -> int:
        return 0

    def reset_empty(self):
        self.retriever = object()
        return self.retriever


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDY_COACH_TEST_MODE", "1")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/learning-auth.db")
    monkeypatch.delenv("STUDY_COACH_LOCAL_MODE", raising=False)

    from app.db import session as session_module

    session_module._engine = None
    session_module._SessionLocal = None
    application = create_app()
    application.state.retriever_runtime = FakeRuntime()
    application.state.retriever = application.state.retriever_runtime.retriever
    yield application
    if session_module._engine is not None:
        session_module._engine.dispose()
    session_module._engine = None
    session_module._SessionLocal = None


@pytest.fixture
def client(app):
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_learning_write_rejects_missing_bearer(client):
    from sqlalchemy import func, select

    response = client.post("/api/goals", json={"title": "no auth"})
    assert response.status_code == 401
    with session_scope() as session:
        assert session.execute(select(func.count()).select_from(User)).scalar_one() == 0


def test_learning_write_rejects_invalid_bearer(client):
    response = client.post(
        "/api/goals",
        headers={"Authorization": "Bearer not-a-jwt"},
        json={"title": "bad token"},
    )
    assert response.status_code == 401


def test_learning_write_rejects_token_without_user_row(client):
    token = issue_token("ghost-user", "guest")
    response = client.post(
        "/api/goals",
        headers={"Authorization": f"Bearer {token}"},
        json={"title": "ghost write"},
    )
    assert response.status_code == 401
    summary = client.get(
        "/api/data/summary",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert summary.status_code == 200
    assert summary.json()["goals"] == 0
    assert summary.json()["has_learning_data"] is False


def test_learning_write_accepts_existing_anonymous_user(client):
    with session_scope() as session:
        session.add(User(id="anon-1", fingerprint="fp-anon-1"))
        session.commit()
    token = issue_token("anon-1", "guest")

    response = client.post(
        "/api/goals",
        headers={"Authorization": f"Bearer {token}"},
        json={"title": "real user goal"},
    )
    assert response.status_code == 200
    assert response.json()["title"] == "real user goal"
