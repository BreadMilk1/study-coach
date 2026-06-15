import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk

from app.api.deps import get_session
from app.main import create_app


class StubRetriever:
    def __init__(self):
        self.added: list[dict] = []
        self.search_returns: list[dict] = []

    def add_chunks(self, chunks):
        self.added.extend(chunks)

    def search(self, query: str, top_k: int = 5):
        return self.search_returns[:top_k]


class StubLLM:
    def __init__(self, tokens: list[str]):
        self.tokens = tokens

    async def astream(self, messages, **_kwargs):
        for t in self.tokens:
            yield AIMessageChunk(content=t)

    def invoke(self, messages, **_kwargs):
        return AIMessage(content="".join(self.tokens))


class StubDocumentProcessor:
    def __init__(self):
        self.calls = 0

    def process_pdf(self, path):
        self.calls += 1
        return [
            {"chunk_id": "stub:1:0", "content": "Stub chunk one content.",
             "source": "stub.pdf", "page": 1},
            {"chunk_id": "stub:2:0", "content": "Stub chunk two content.",
             "source": "stub.pdf", "page": 2},
        ]


class StubJudgeLLM:
    """Always-pass judge stub (P2.1-② Judge Guard wiring)."""

    _PASS = (
        '{"relevance":5,"accuracy":5,"citation_quality":4,'
        '"accessibility":4,"example_quality":5,"learner_level_fit":5,'
        '"reasoning":"Solid."}'
    )

    async def ainvoke(self, messages, **_kwargs):
        return AIMessage(content=self._PASS)


@pytest.fixture
def stub_retriever():
    r = StubRetriever()
    r.search_returns = [
        {"chunk_id": "a:1:0", "content": "HyDE rewrites queries.",
         "source": "a.pdf", "page": 1, "score": 0.9},
    ]
    return r


@pytest.fixture
def stub_llm():
    return StubLLM(tokens=["HyDE", " is", " a", " technique", "."])


@pytest.fixture
def stub_document_processor():
    return StubDocumentProcessor()


@pytest.fixture
def app(tmp_path, stub_retriever, stub_llm, stub_document_processor, monkeypatch):
    # Isolated SQLite per test
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    # Reset module-level state
    from app.db import session as session_mod
    session_mod._engine = None
    session_mod._SessionLocal = None

    app = create_app()
    app.state.retriever = stub_retriever
    app.state.document_processor = stub_document_processor

    from app.api.deps import get_judge_dependencies, get_llm
    app.dependency_overrides[get_llm] = lambda: stub_llm
    # same_model=False so the SSE strict-equality token assertion below
    # is not perturbed by the P2.1-② bias warning prefix.
    app.dependency_overrides[get_judge_dependencies] = lambda: {
        "llm": StubJudgeLLM(),
        "same_model": False,
    }
    from app.db.repositories import DocumentRepository
    from app.db.session import session_scope
    with session_scope() as session:
        DocumentRepository(session).create(
            user_id="default-user",
            filename="fixture.pdf",
            hash_="fixture-hash",
            chunks_count=1,
        )
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def test_health_endpoint(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "ollama_enabled" in r.json()


def test_chat_streams_citations_then_tokens_then_done(client):
    headers = {
        "x-fingerprint": "fp-1",
        "x-provider": "ollama",
        "x-model": "gemma3:4b",
    }
    with client.stream("POST", "/api/chat",
                       json={"message": "What is HyDE?"},
                       headers=headers) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        events = []
        for line in resp.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

    types = [e["type"] for e in events]
    assert types[-1] == "done"
    token_events = [e for e in events if e["type"] == "token"]
    assert "".join(e["text"] for e in token_events) == "HyDE is a technique."
    citation_events = [e for e in events if e["type"] == "citations"]
    assert len(citation_events) == 1
    assert citation_events[0]["citations"][0]["chunk_id"] == "a:1:0"


def test_upload_document_calls_processor_and_indexes_chunks(client, stub_retriever, stub_document_processor):
    files = {"file": ("lec.pdf", b"%PDF-1.4 fake bytes", "application/pdf")}
    headers = {"x-fingerprint": "fp-1"}

    r = client.post("/api/documents", files=files, headers=headers)

    assert r.status_code == 200
    body = r.json()
    assert body["filename"] == "lec.pdf"
    assert body["chunks_count"] == 2
    assert stub_document_processor.calls == 1
    assert len(stub_retriever.added) == 2
    # filename propagated as source
    assert all(c["source"] == "lec.pdf" for c in stub_retriever.added)
