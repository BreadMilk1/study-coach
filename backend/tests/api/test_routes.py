import asyncio
import json
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk

from app.api.deps import get_graph, get_session
from app.auth import issue_token
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
        self.paths: list[Path] = []

    def process_pdf(self, path):
        self.calls += 1
        self.paths.append(Path(path))
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


class FakeRuntime:
    def __init__(self) -> None:
        self.retriever = object()

    def vector_count(self) -> int:
        return 0

    def reset_empty(self) -> None:
        self.retriever = object()


class BlockingGraph:
    def __init__(self, started: threading.Event, release: threading.Event) -> None:
        self.started = started
        self.release = release

    async def astream(self, _input_state, **_kwargs):
        self.started.set()
        await asyncio.to_thread(self.release.wait)
        yield {"type": "citations", "citations": []}
        yield {"type": "token", "text": "done"}


class PublicModelStub:
    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, _messages):
        return AIMessage(content="pong")


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
    app.state.retriever_runtime = FakeRuntime()
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
    assert not stub_document_processor.paths[0].exists()


def test_same_pdf_uploads_use_distinct_temporary_paths(
    client,
    stub_document_processor,
):
    files = {"file": ("same.pdf", b"%PDF-1.4 same bytes", "application/pdf")}

    first = client.post("/api/documents", files=files)
    second = client.post("/api/documents", files=files)

    assert first.status_code == 200
    assert second.status_code == 200
    first_path, second_path = stub_document_processor.paths
    assert first_path != second_path
    assert first_path.name.startswith("sc_")
    assert second_path.name.startswith("sc_")
    assert first_path.suffix == ".pdf"
    assert second_path.suffix == ".pdf"
    assert not first_path.exists()
    assert not second_path.exists()


def test_upload_removes_exact_temporary_path_when_processing_fails(
    client,
    stub_document_processor,
    monkeypatch,
):
    captured: list[Path] = []

    def fail(path):
        captured.append(Path(path))
        raise RuntimeError("parse failed")

    monkeypatch.setattr(stub_document_processor, "process_pdf", fail)

    with pytest.raises(RuntimeError, match="parse failed"):
        client.post(
            "/api/documents",
            files={"file": ("broken.pdf", b"not a PDF", "application/pdf")},
        )

    assert len(captured) == 1
    assert not captured[0].exists()


def test_upload_removes_partial_temporary_file_when_write_fails(
    client,
    monkeypatch,
    tmp_path,
):
    partial_path = tmp_path / "partial-upload.pdf"

    class FailingTemporaryFile:
        name = str(partial_path)

        def __enter__(self):
            partial_path.touch()
            return self

        def write(self, content):
            partial_path.write_bytes(content[:4])
            raise OSError("disk full")

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        "app.api.routes.tempfile.NamedTemporaryFile",
        lambda **_kwargs: FailingTemporaryFile(),
    )

    with pytest.raises(OSError, match="disk full"):
        client.post(
            "/api/documents",
            files={"file": ("partial.pdf", b"%PDF-1.4", "application/pdf")},
        )

    assert not partial_path.exists()


def test_reset_is_rejected_until_streaming_chat_response_finishes(
    app,
    client,
    monkeypatch,
):
    started = threading.Event()
    release = threading.Event()
    app.dependency_overrides[get_graph] = lambda: BlockingGraph(started, release)
    monkeypatch.setenv("STUDY_COACH_LOCAL_MODE", "1")
    chat_result: dict[str, object] = {}

    def consume_chat() -> None:
        try:
            with TestClient(app) as stream_client:
                chat_result["response"] = stream_client.post(
                    "/api/chat",
                    json={"message": "hold this stream open"},
                )
        except BaseException as exc:  # pragma: no cover - reported by main thread
            chat_result["error"] = exc

    thread = threading.Thread(target=consume_chat)
    thread.start()
    try:
        assert started.wait(timeout=5), "chat stream did not start"
        reset = client.post(
            "/api/data/reset",
            headers={
                "Authorization": f"Bearer {issue_token('reset-user', 'member')}"
            },
            json={
                "scope": "learning",
                "confirmation": "CLEAR_LEARNING_DATA",
            },
        )
    finally:
        release.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert "error" not in chat_result
    assert reset.status_code == 409
    assert reset.json()["detail"]["code"] == "data_operation_in_progress"
    assert chat_result["response"].status_code == 200


@pytest.mark.parametrize(
    ("method", "path", "request_kwargs"),
    [
        (
            "post",
            "/api/documents",
            {"files": {"file": ("owned.pdf", b"pdf", "application/pdf")}},
        ),
        ("post", "/api/chat", {"json": {"message": "hello"}}),
        ("get", "/api/chat/sessions/current", {}),
        ("get", "/api/chat/sessions/missing/messages", {}),
        ("post", "/api/goals", {"json": {"title": "Exam"}}),
        ("get", "/api/plans/current", {}),
        (
            "patch",
            "/api/plans/plan/milestones/milestone",
            {"json": {"done": True}},
        ),
        ("get", "/api/plans/plan/events", {}),
        (
            "patch",
            "/api/plans/plan/milestones/reorder",
            {"json": {"milestone_ids": []}},
        ),
        ("get", "/api/documents", {}),
        ("get", "/api/mistakes/due", {}),
        (
            "post",
            "/api/mistakes/mistake/review",
            {"json": {"answer": "A"}},
        ),
        ("post", "/api/mistakes/mistake/mark-understood", {}),
        ("get", "/api/mastery", {}),
        ("get", "/api/users/me/stats", {}),
    ],
    ids=[
        "upload",
        "chat",
        "current-chat",
        "chat-messages",
        "goals",
        "plans",
        "milestone",
        "plan-events",
        "plan-reorder",
        "documents",
        "mistakes",
        "mistake-review",
        "mark-understood",
        "mastery",
        "stats",
    ],
)
def test_learning_route_family_is_rejected_during_reset(
    app,
    client,
    method,
    path,
    request_kwargs,
):
    with app.state.data_lifecycle_gate.exclusive_reset():
        response = client.request(method, path, **request_kwargs)

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "code": "reset_in_progress",
        "message": "Data reset is in progress.",
    }


@pytest.mark.parametrize(
    "path",
    ["/api/health", "/api/models/ping", "/api/models/tool-check"],
)
def test_public_route_is_available_during_reset(app, client, monkeypatch, path):
    monkeypatch.setattr("app.llm.provider.get_chat_model", lambda _config: PublicModelStub())

    with app.state.data_lifecycle_gate.exclusive_reset():
        response = client.get(
            path,
            headers={"x-provider": "ollama", "x-model": "stub-model"},
        )

    assert response.status_code == 200
