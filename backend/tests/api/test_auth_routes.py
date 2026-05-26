import pytest
from fastapi.testclient import TestClient

from app.auth import issue_token
from app.main import create_app


class StubRetriever:
    def __init__(self):
        self.added: list[dict] = []
        self.search_returns: list[dict] = []

    def add_chunks(self, chunks):
        self.added.extend(chunks)

    def search(self, *args, **kwargs):
        return self.search_returns


class StubDocProcessor:
    def process_pdf(self, path):
        return [{"chunk_id": "c1", "content": "test", "source": "test.pdf", "page": 1}]


@pytest.fixture
def app(tmp_path):
    import os
    os.environ["STUDY_COACH_TEST_MODE"] = "1"
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    from app.db import session as session_mod
    session_mod._engine = None
    session_mod._SessionLocal = None

    app = create_app()
    app.state.retriever = StubRetriever()
    app.state.document_processor = StubDocProcessor()
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def test_anonymous_login_creates_user(client):
    resp = client.post("/api/auth/anonymous", json={"fingerprint": "test-fp-001"})
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["user_id"]
    assert data["tier"] == "guest"


def test_anonymous_login_same_fingerprint_same_user(client):
    r1 = client.post("/api/auth/anonymous", json={"fingerprint": "test-fp-002"})
    r2 = client.post("/api/auth/anonymous", json={"fingerprint": "test-fp-002"})
    assert r1.json()["user_id"] == r2.json()["user_id"]


def test_protected_route_accepts_valid_token(client):
    token = issue_token("test-user-id", "member")
    resp = client.get(
        "/api/documents",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200


def test_protected_route_rejects_invalid_token(client):
    resp = client.get(
        "/api/documents",
        headers={"Authorization": "Bearer garbage"},
    )
    assert resp.status_code == 401


def test_google_login_without_client_id_returns_401(client):
    resp = client.post("/api/auth/google", json={"credential": "fake"})
    assert resp.status_code == 401


def test_health_is_public(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
