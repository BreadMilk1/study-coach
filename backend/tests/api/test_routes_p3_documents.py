"""Cut A4 — GET /api/documents."""
import pytest
from fastapi.testclient import TestClient

from app.auth import issue_token
from app.main import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/p3_docs.db")
    monkeypatch.setenv("STUDY_COACH_TEST_MODE", "1")
    from app.db import session as session_mod
    session_mod._engine = None
    session_mod._SessionLocal = None
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_get_documents_empty(client):
    token = issue_token("default-user", "guest")
    resp = client.get("/api/documents", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_documents_lists_uploaded(client):
    from app.db.session import session_scope
    from app.db.repositories import UserRepository, DocumentRepository
    with session_scope() as s:
        user = UserRepository(s).get_or_create("fp-2")
        DocumentRepository(s).create(user_id=user.id, filename="a.pdf", hash_="h1", chunks_count=10)
        DocumentRepository(s).create(user_id=user.id, filename="b.pdf", hash_="h2", chunks_count=5)
        user_id = user.id

    token = issue_token(user_id, "guest")
    resp = client.get("/api/documents", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    assert {d["filename"] for d in body} == {"a.pdf", "b.pdf"}
    assert all(set(d.keys()) >= {"id", "filename", "chunks_count"} for d in body)
