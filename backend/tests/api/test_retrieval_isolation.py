"""Request-owner retrieval isolation through the production API boundary.

Builds the real app against a tmp SQLite database and a real per-test Chroma
collection, but swaps in the deterministic test embedder, a stub reranker, a
stub PDF processor, and a stub LLM/judge so nothing touches the network.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import chromadb
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk

from app.api import deps
from app.auth import issue_token
from app.db.repositories import DocumentRepository
from app.db.session import session_scope
from app.main import create_app
from app.rag.hybrid_retriever import BM25Index, HybridRetriever
from app.rag.reranking_retriever import RerankingRetriever
from app.rag.retriever import Retriever
from tests.helpers import ensure_user


class StubDocumentProcessor:
    def process_pdf(self, path):
        return [
            {"chunk_id": "tmp", "content": "HyDE hypothetical document expansion.",
             "source": Path(path).name, "page": 1},
            {"chunk_id": "tmp2", "content": "BM25 lexical retrieval scoring.",
             "source": Path(path).name, "page": 2},
        ]


class StubReranker:
    def score(self, query, documents):
        return [1.0 - 0.01 * i for i in range(len(documents))]


class StubLLM:
    async def astream(self, messages, **_kwargs):
        for token in ("HyDE", " is", " a", " technique", "."):
            yield AIMessageChunk(content=token)


class StubJudgeLLM:
    async def ainvoke(self, messages, **_kwargs):
        return AIMessage(content=(
            '{"relevance":5,"accuracy":5,"citation_quality":4,"accessibility":4,'
            '"example_quality":5,"learner_level_fit":5,"reasoning":"Solid."}'
        ))


def _read_sse(resp) -> list[dict]:
    events = []
    for line in resp.iter_lines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


@pytest.fixture
def app(tmp_path, monkeypatch, fake_embedder):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/retrieval_isolation.db")
    from app.db import session as session_mod
    session_mod._engine = None
    session_mod._SessionLocal = None

    application = create_app()
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    collection = client.get_or_create_collection("retrieval_isolation_chunks")
    dense = Retriever(collection=collection, embedder=fake_embedder)
    hybrid = HybridRetriever(dense=dense, bm25=BM25Index())
    application.state.retriever = RerankingRetriever(
        base=hybrid, reranker=StubReranker(), retrieval_depth=20
    )
    application.state.document_processor = StubDocumentProcessor()

    from app.api.deps import get_judge_dependencies, get_llm
    application.dependency_overrides[get_llm] = lambda: StubLLM()
    application.dependency_overrides[get_judge_dependencies] = lambda: {
        "llm": StubJudgeLLM(),
        "same_model": False,
    }
    with session_scope() as session:
        ensure_user(session, "user-a")
        ensure_user(session, "user-b")
    return application


def _client(app, user_id: str) -> TestClient:
    return TestClient(
        app, headers={"Authorization": f"Bearer {issue_token(user_id, 'guest')}"}
    )


def _upload(client, filename: str = "notes.pdf"):
    return client.post(
        "/api/documents",
        files={"file": (filename, b"%PDF-1.4 isolation-bytes", "application/pdf")},
        headers={"x-fingerprint": "fp"},
    )


def _chat_citations(client) -> list[dict]:
    with client.stream(
        "POST",
        "/api/chat",
        json={"message": "What is HyDE?"},
        headers={"x-provider": "ollama", "x-model": "gemma3:4b"},
    ) as resp:
        assert resp.status_code == 200
        events = _read_sse(resp)
    citation_events = [e for e in events if e["type"] == "citations"]
    assert citation_events, events
    return citation_events[0]["citations"]


def test_same_pdf_two_users_get_distinct_owner_scoped_chunks(app):
    a = _client(app, "user-a")
    b = _client(app, "user-b")

    response_a = _upload(a)
    response_b = _upload(b)
    assert response_a.status_code == response_b.status_code == 200
    assert response_a.json()["document_id"] != response_b.json()["document_id"]

    citations_a = _chat_citations(a)
    citations_b = _chat_citations(b)
    assert citations_a and all(c["chunk_id"].startswith("v2:user-a:") for c in citations_a)
    assert citations_b and all(c["chunk_id"].startswith("v2:user-b:") for c in citations_b)
    assert {c["chunk_id"] for c in citations_a}.isdisjoint(
        {c["chunk_id"] for c in citations_b}
    )


def test_v2_chunk_owner_scoped_and_non_owner_gets_404(app):
    a = _client(app, "user-a")
    b = _client(app, "user-b")
    _upload(a)
    _upload(b)

    ids = app.state.retriever.base.dense.collection.get(include=["metadatas"])["ids"]
    a_ids = [i for i in ids if i.startswith("v2:user-a:")]
    b_ids = [i for i in ids if i.startswith("v2:user-b:")]
    assert a_ids and b_ids

    assert a.get(f"/api/chunks/{a_ids[0]}").status_code == 200
    # Same PDF/hash, different owner: neither can read the other's v2 chunk.
    assert a.get(f"/api/chunks/{b_ids[0]}").status_code == 404
    assert b.get(f"/api/chunks/{a_ids[0]}").status_code == 404


def test_legacy_ownerless_chunk_readable_only_by_sql_owner(app):
    a = _client(app, "user-a")
    b = _client(app, "user-b")

    legacy_hash = "2" * 64
    legacy_id = f"{legacy_hash}:1:0"
    app.state.retriever.add_chunks([
        {"chunk_id": legacy_id, "content": "legacy ownerless content",
         "source": "old.pdf", "page": 1},
    ])
    with session_scope() as session:
        DocumentRepository(session).create(
            user_id="user-a", filename="old.pdf", hash_=legacy_hash, chunks_count=1,
        )

    owned = a.get(f"/api/chunks/{legacy_id}")
    assert owned.status_code == 200
    assert owned.json()["content"] == "legacy ownerless content"
    assert b.get(f"/api/chunks/{legacy_id}").status_code == 404

    unowned = "3" * 64 + ":1:0"
    app.state.retriever.add_chunks([
        {"chunk_id": unowned, "content": "unowned", "source": "u.pdf", "page": 1},
    ])
    assert a.get(f"/api/chunks/{unowned}").status_code == 404


def test_get_retriever_binds_request_owner_without_caching(app):
    with session_scope() as session:
        DocumentRepository(session).create(
            user_id="user-a", filename="a.pdf", hash_="a" * 64, chunks_count=1,
        )
        DocumentRepository(session).create(
            user_id="user-b", filename="b.pdf", hash_="b" * 64, chunks_count=1,
        )

    request = SimpleNamespace(app=app)
    with session_scope() as session:
        scoped_a = deps.get_retriever(request, "user-a", session)
        scoped_b = deps.get_retriever(request, "user-b", session)

    assert scoped_a.scope.owner == "user-a"
    assert scoped_b.scope.owner == "user-b"
    assert scoped_a.scope is not scoped_b.scope
    assert set(scoped_a.scope.documents) == {"a" * 64}
    assert set(scoped_b.scope.documents) == {"b" * 64}


def test_same_bytes_different_filenames_do_not_overwrite_source(app):
    a = _client(app, "user-a")
    b = _client(app, "user-b")
    response_a = _upload(a, filename="a.pdf")
    response_b = _upload(b, filename="b.pdf")
    assert response_a.status_code == response_b.status_code == 200
    assert response_a.json()["filename"] == "a.pdf"
    assert response_b.json()["filename"] == "b.pdf"

    stored = app.state.retriever.base.dense.collection.get(include=["metadatas"])
    sources_by_owner: dict[str, set] = {}
    for meta in stored["metadatas"]:
        sources_by_owner.setdefault(meta["owner"], set()).add(meta["source"])
    # Identical bytes must not let one owner overwrite the other's source name.
    assert sources_by_owner["user-a"] == {"a.pdf"}
    assert sources_by_owner["user-b"] == {"b.pdf"}


def test_sql_create_failure_leaves_unsearchable_orphan(app, monkeypatch):
    a = _client(app, "user-a")

    def boom(self, **kwargs):
        raise RuntimeError("forced sql create failure")

    monkeypatch.setattr(DocumentRepository, "create", boom)

    with pytest.raises(RuntimeError, match="forced sql create failure"):
        _upload(a)

    # Index-before-SQL wrote owner-scoped chunks, but they carry no document id
    # and no SQL row exists, so production search must not return them.
    stored = app.state.retriever.base.dense.collection.get(include=["metadatas"])
    assert len(stored["ids"]) == 2
    assert all("document_id" not in (meta or {}) for meta in stored["metadatas"])

    with session_scope() as session:
        scoped = deps.get_retriever(SimpleNamespace(app=app), "user-a", session)
    assert scoped.search("HyDE", top_k=5) == []
