import asyncio
import json
import socket
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk

from app.api.deps import get_graph, get_session
from app.auth import issue_token
from app.main import create_app
from tests.helpers import ensure_user


class StubRetriever:
    def __init__(self):
        self.added: list[dict] = []
        self.search_returns: list[dict] = []

    def add_chunks(self, chunks):
        self.added.extend(chunks)

    def get_chunk(self, chunk_id: str):
        for chunk in reversed(self.added):
            if chunk.get("chunk_id") == chunk_id:
                return {
                    "chunk_id": chunk_id,
                    "content": chunk.get("content", ""),
                    "source": chunk.get("source", ""),
                    "page": chunk.get("page", -1),
                }
        return None

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
        path = Path(path)
        self.paths.append(path)
        source = path.name
        return [
            {
                "chunk_id": f"{source}:1:0",
                "content": "Stub chunk one content.",
                "source": source,
                "page": 1,
            },
            {
                "chunk_id": f"{source}:2:0",
                "content": "Stub chunk two content.",
                "source": source,
                "page": 2,
            },
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
        ensure_user(session, "default-user")
        DocumentRepository(session).create(
            user_id="default-user",
            filename="fixture.pdf",
            hash_="fixture-hash",
            chunks_count=1,
        )
    return app


@pytest.fixture
def client(app):
    return TestClient(
        app,
        headers={"Authorization": f"Bearer {issue_token('default-user', 'guest')}"},
    )


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


def test_get_chunk_returns_indexed_content(client, stub_retriever):
    owned_hash = "b" * 64
    chunk_id = f"{owned_hash}:1:0"
    missing_chunk_id = f"{owned_hash}:99:0"
    stub_retriever.add_chunks([
        {
            "chunk_id": chunk_id,
            "content": "Prompt engineering is the craft of designing prompts.",
            "source": "lec.pdf",
            "page": 3,
        },
    ])
    from app.db.repositories import DocumentRepository
    from app.db.session import session_scope

    with session_scope() as session:
        ensure_user(session, "chunk-user")
        DocumentRepository(session).create(
            user_id="chunk-user",
            filename="lec.pdf",
            hash_=owned_hash,
            chunks_count=1,
        )
    token = issue_token("chunk-user", "guest")
    headers = {"Authorization": f"Bearer {token}"}

    ok = client.get(f"/api/chunks/{chunk_id}", headers=headers)
    missing = client.get(f"/api/chunks/{missing_chunk_id}", headers=headers)

    assert ok.status_code == 200
    assert ok.json() == {
        "chunk_id": chunk_id,
        "content": "Prompt engineering is the craft of designing prompts.",
        "source": "lec.pdf",
        "page": 3,
    }
    assert missing.status_code == 404


def test_get_chunk_hides_content_not_owned_by_current_user(client, stub_retriever):
    """Cross-user: chunk exists for owner, but attacker must get neutral 404."""
    owner_hash = "a" * 64
    chunk_id = f"{owner_hash}:1:0"
    private_content = "PRIVATE_OWNER_ONLY_CHUNK_CONTENT"

    stub_retriever.add_chunks([
        {
            "chunk_id": chunk_id,
            "content": private_content,
            "source": "private.pdf",
            "page": 1,
        },
    ])
    assert stub_retriever.get_chunk(chunk_id) is not None
    assert stub_retriever.get_chunk(chunk_id)["content"] == private_content

    from app.db.repositories import DocumentRepository
    from app.db.session import session_scope

    with session_scope() as session:
        ensure_user(session, "chunk-owner")
        ensure_user(session, "chunk-attacker")
        owner_doc = DocumentRepository(session).create(
            user_id="chunk-owner",
            filename="private.pdf",
            hash_=owner_hash,
            chunks_count=1,
        )
        assert owner_doc.hash == owner_hash
        assert owner_doc.user_id == "chunk-owner"
        attacker_docs = DocumentRepository(session).list_for_user("chunk-attacker")
        assert attacker_docs == []

    token = issue_token("chunk-attacker", "guest")
    headers = {"Authorization": f"Bearer {token}"}

    response = client.get(f"/api/chunks/{chunk_id}", headers=headers)

    assert response.status_code == 404
    assert response.json()["detail"] == "chunk not found"
    assert private_content not in response.text


def test_upload_rejects_non_pdf_extension(client, stub_document_processor):
    response = client.post(
        "/api/documents",
        files={"file": ("notes.txt", b"%PDF-1.4 pretend", "text/plain")},
    )
    assert response.status_code == 415
    assert response.json()["detail"]["code"] == "unsupported_media_type"
    assert stub_document_processor.calls == 0


def test_upload_rejects_disallowed_mime_type(client, stub_document_processor):
    response = client.post(
        "/api/documents",
        files={"file": ("notes.pdf", b"%PDF-1.4 pretend", "text/html")},
    )
    assert response.status_code == 415
    assert response.json()["detail"]["code"] == "unsupported_media_type"
    assert stub_document_processor.calls == 0


def test_upload_rejects_non_pdf_magic_bytes(client, stub_document_processor):
    response = client.post(
        "/api/documents",
        files={"file": ("broken.pdf", b"not-a-pdf-at-all", "application/pdf")},
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_pdf"
    assert stub_document_processor.calls == 0


def test_upload_rejects_parser_failure_without_500(client, monkeypatch):
    def boom(self, _path):
        raise RuntimeError("secret parser path /tmp/inner.pdf")

    monkeypatch.setattr(type(client.app.state.document_processor), "process_pdf", boom)

    response = client.post(
        "/api/documents",
        files={"file": ("broken.pdf", b"%PDF-1.4 corrupt-body", "application/pdf")},
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_pdf"
    detail = response.json()["detail"]
    assert "secret parser" not in str(detail)
    assert "/tmp/inner" not in str(detail)


def test_upload_rejects_oversize_payload(client, monkeypatch, stub_document_processor):
    import app.api.routes as routes_mod

    monkeypatch.setattr(routes_mod, "MAX_UPLOAD_BYTES", 64)
    response = client.post(
        "/api/documents",
        files={"file": ("big.pdf", b"%PDF-1.4 " + b"x" * 64, "application/pdf")},
    )
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "payload_too_large"
    assert stub_document_processor.calls == 0


def test_upload_accepts_exact_size_limit_boundary(client, monkeypatch, stub_document_processor):
    import app.api.routes as routes_mod

    monkeypatch.setattr(routes_mod, "MAX_UPLOAD_BYTES", 32)
    payload = b"%PDF-1.4 " + b"y" * (32 - len(b"%PDF-1.4 "))
    assert len(payload) == 32
    response = client.post(
        "/api/documents",
        files={"file": ("edge.pdf", payload, "application/pdf")},
    )
    assert response.status_code == 200
    assert stub_document_processor.calls == 1
    assert not stub_document_processor.paths[0].exists()


def test_upload_cleans_temp_file_on_rejection(client, monkeypatch, stub_document_processor):
    import app.api.routes as routes_mod

    monkeypatch.setattr(routes_mod, "MAX_UPLOAD_BYTES", 16)
    before = set(Path(tempfile.gettempdir()).glob("sc_*.pdf"))
    response = client.post(
        "/api/documents",
        files={"file": ("big.pdf", b"%PDF-1.4 " + b"z" * 32, "application/pdf")},
    )
    assert response.status_code == 413
    after = set(Path(tempfile.gettempdir()).glob("sc_*.pdf"))
    assert after == before


@pytest.mark.asyncio
async def test_upload_cancelled_read_cleans_temp_and_propagates(monkeypatch):
    import app.api.routes as routes_mod

    created: list[Path] = []
    real_named = tempfile.NamedTemporaryFile

    def tracking_named_temporary_file(**kwargs):
        handle = real_named(**kwargs)
        created.append(Path(handle.name))
        return handle

    monkeypatch.setattr(routes_mod.tempfile, "NamedTemporaryFile", tracking_named_temporary_file)

    class CancellingUpload:
        filename = "cancel.pdf"
        content_type = "application/pdf"

        def __init__(self) -> None:
            self.calls = 0

        async def read(self, _size: int = -1) -> bytes:
            self.calls += 1
            if self.calls == 1:
                return b"%PDF-1.4 partial-chunk"
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await routes_mod._read_pdf_upload_to_temp(CancellingUpload())

    assert len(created) == 1
    assert not created[0].exists()


def _asgi_upload_scope(app, *, body: bytes, content_length: bytes | None, token: str):
    boundary = "----LimitUploadBoundary"
    headers = [
        (b"host", b"testserver"),
        (b"content-type", f"multipart/form-data; boundary={boundary}".encode()),
        (b"authorization", f"Bearer {token}".encode()),
        (b"origin", b"http://localhost:5173"),
    ]
    if content_length is not None:
        headers.append((b"content-length", content_length))
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/documents",
        "raw_path": b"/api/documents",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
        "app": app,
        "state": {},
    }, boundary


def test_upload_rejects_oversized_content_length_before_body(app, client, monkeypatch):
    import app.api.upload_limits as limits

    monkeypatch.setattr(limits, "MAX_UPLOAD_REQUEST_BYTES", 32)
    from app.db.session import session_scope

    with session_scope() as session:
        ensure_user(session, "upload-user")
    token = issue_token("upload-user", "guest")
    scope, _boundary = _asgi_upload_scope(
        app,
        body=b"",
        content_length=b"999999",
        token=token,
    )
    response_messages: list[dict] = []
    receive_calls = {"n": 0}

    async def receive():
        receive_calls["n"] += 1
        return {"type": "http.request", "body": b"should-not-be-needed", "more_body": False}

    async def send(message):
        response_messages.append(message)

    asyncio.run(app(scope, receive, send))
    starts = [m for m in response_messages if m.get("type") == "http.response.start"]
    assert starts and starts[0]["status"] == 413
    header_map = {
        k.decode().lower(): v.decode() for k, v in starts[0].get("headers", [])
    }
    assert header_map.get("access-control-allow-origin") == "http://localhost:5173"
    assert app.state.data_lifecycle_gate._active_operations == 0
    # Body receive must not be required for Content-Length rejection.
    assert receive_calls["n"] == 0


def test_slow_multipart_returns_413_before_remaining_body(
    app,
    client,
    monkeypatch,
):
    """Request-body cap must fire before multipart finishes spooling."""
    import app.api.upload_limits as limits

    monkeypatch.setattr(limits, "MAX_UPLOAD_REQUEST_BYTES", 40)
    monkeypatch.setenv("STUDY_COACH_LOCAL_MODE", "1")
    from app.db.session import session_scope

    with session_scope() as session:
        ensure_user(session, "upload-user")
    token = issue_token("upload-user", "guest")

    boundary = "----SlowOversizeBoundary"
    pdf_bytes = b"%PDF-1.4 " + (b"x" * 80)
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="big.pdf"\r\n'
        f"Content-Type: application/pdf\r\n\r\n"
    ).encode() + pdf_bytes + f"\r\n--{boundary}--\r\n".encode()
    first = body[:60]
    rest = body[60:]
    assert len(first) > 40

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/documents",
        "raw_path": b"/api/documents",
        "query_string": b"",
        "headers": [
            (b"host", b"testserver"),
            (
                b"content-type",
                f"multipart/form-data; boundary={boundary}".encode(),
            ),
            # Spoofed short Content-Length; stream still exceeds the request cap.
            (b"content-length", b"10"),
            (b"authorization", f"Bearer {token}".encode()),
            (b"origin", b"http://localhost:5173"),
        ],
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
        "app": app,
        "state": {},
    }

    body_blocked = threading.Event()
    release_body = threading.Event()
    response_started = threading.Event()
    upload_result: dict[str, object] = {}
    response_messages: list[dict] = []
    phase = {"n": 0}

    async def receive():
        if phase["n"] == 0:
            phase["n"] = 1
            return {"type": "http.request", "body": first, "more_body": True}
        if phase["n"] == 1:
            phase["n"] = 2
            body_blocked.set()
            await asyncio.to_thread(release_body.wait)
            return {"type": "http.request", "body": rest, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        response_messages.append(message)
        if message.get("type") == "http.response.start":
            response_started.set()

    def run_upload() -> None:
        try:
            asyncio.run(app(scope, receive, send))
            upload_result["ok"] = True
        except BaseException as exc:  # pragma: no cover
            upload_result["error"] = exc

    thread = threading.Thread(target=run_upload)
    thread.start()
    try:
        assert response_started.wait(timeout=5), "413 did not arrive before remaining body"
        assert not body_blocked.is_set(), "server waited on remaining body before 413"
        assert app.state.data_lifecycle_gate._active_operations == 0
        # Lease must be free so a concurrent reset is not blocked by the refusal.
        reset = client.post(
            "/api/data/reset",
            headers={"Authorization": f"Bearer {issue_token('reset-user', 'member')}"},
            json={
                "scope": "learning",
                "confirmation": "CLEAR_LEARNING_DATA",
            },
        )
        assert reset.status_code == 200
    finally:
        release_body.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert "error" not in upload_result
    starts = [m for m in response_messages if m.get("type") == "http.response.start"]
    assert starts and starts[0]["status"] == 413
    header_map = {
        k.decode().lower(): v.decode() for k, v in starts[0].get("headers", [])
    }
    assert header_map.get("access-control-allow-origin") == "http://localhost:5173"


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


def test_duplicate_pdf_upload_keeps_stable_chunk_ids_and_does_not_grow_index(
    client,
    stub_retriever,
    stub_document_processor,
):
    payload = b"%PDF-1.4 identical-bytes-for-dedup"
    first = client.post(
        "/api/documents",
        files={"file": ("notes.pdf", payload, "application/pdf")},
    )
    same_name = client.post(
        "/api/documents",
        files={"file": ("notes.pdf", payload, "application/pdf")},
    )
    renamed = client.post(
        "/api/documents",
        files={"file": ("renamed-notes.pdf", payload, "application/pdf")},
    )

    assert first.status_code == 200
    assert same_name.status_code == 200
    assert renamed.status_code == 200
    assert first.json()["document_id"] == same_name.json()["document_id"] == renamed.json()["document_id"]
    assert first.json()["filename"] == "notes.pdf"
    assert same_name.json()["filename"] == "notes.pdf"
    assert renamed.json()["filename"] == "notes.pdf"
    assert first.json()["chunks_count"] == 2

    first_ids = [c["chunk_id"] for c in stub_retriever.added[:2]]
    same_ids = [c["chunk_id"] for c in stub_retriever.added[2:4]]
    renamed_ids = [c["chunk_id"] for c in stub_retriever.added[4:]]
    assert first_ids == same_ids == renamed_ids
    assert all(not id_.startswith("sc_") for id_ in first_ids)
    assert len(set(first_ids)) == 2
    # Canonical filename is first-seen for this user+hash across SQL and indexes.
    assert all(c["source"] == "notes.pdf" for c in stub_retriever.added)


def test_duplicate_pdf_keeps_sql_chroma_bm25_source_metadata_aligned(
    client,
    app,
    fake_embedder,
    chroma_collection,
    stub_document_processor,
):
    from app.rag.hybrid_retriever import BM25Index, HybridRetriever
    from app.rag.retriever import Retriever

    hybrid = HybridRetriever(
        dense=Retriever(collection=chroma_collection, embedder=fake_embedder),
        bm25=BM25Index(),
    )
    app.state.retriever = hybrid
    payload = b"%PDF-1.4 canonical-source-alignment"
    headers = {"x-fingerprint": "fp-canonical"}

    first = client.post(
        "/api/documents",
        files={"file": ("notes.pdf", payload, "application/pdf")},
        headers=headers,
    )
    renamed = client.post(
        "/api/documents",
        files={"file": ("renamed-notes.pdf", payload, "application/pdf")},
        headers=headers,
    )
    # Idempotent retry after SQL already succeeded (not a true index-before-SQL failure).
    retry = client.post(
        "/api/documents",
        files={"file": ("retry-notes.pdf", payload, "application/pdf")},
        headers=headers,
    )

    assert first.status_code == renamed.status_code == retry.status_code == 200
    assert first.json()["filename"] == "notes.pdf"
    assert renamed.json()["filename"] == "notes.pdf"
    assert retry.json()["filename"] == "notes.pdf"
    assert first.json()["document_id"] == renamed.json()["document_id"] == retry.json()["document_id"]

    assert chroma_collection.count() == 2
    stored = chroma_collection.get(include=["metadatas"])
    assert len(stored["ids"]) == 2
    assert all(meta["source"] == "notes.pdf" for meta in stored["metadatas"])
    assert len(hybrid.bm25._chunks) == 2
    assert all(c["source"] == "notes.pdf" for c in hybrid.bm25._chunks)
    assert {c["chunk_id"] for c in hybrid.bm25._chunks} == set(stored["ids"])
    file_hash = __import__("hashlib").sha256(payload).hexdigest()
    assert all(cid.startswith(f"{file_hash}:") for cid in stored["ids"])


def test_partial_retry_after_sql_failure_converges_source_on_retry_filename(
    client,
    app,
    fake_embedder,
    chroma_collection,
    stub_document_processor,
    monkeypatch,
):
    """Indexes succeed, SQL create fails, renamed retry must realign all three stores."""
    from app.db.repositories import DocumentRepository
    from app.rag.hybrid_retriever import BM25Index, HybridRetriever
    from app.rag.retriever import Retriever

    hybrid = HybridRetriever(
        dense=Retriever(collection=chroma_collection, embedder=fake_embedder),
        bm25=BM25Index(),
    )
    app.state.retriever = hybrid
    payload = b"%PDF-1.4 partial-retry-sql-failure"
    headers = {"x-fingerprint": "fp-partial-retry"}

    real_create = DocumentRepository.create
    create_calls = {"n": 0}

    def flaky_create(self, **kwargs):
        create_calls["n"] += 1
        if create_calls["n"] == 1:
            raise RuntimeError("forced sql create failure after index write")
        return real_create(self, **kwargs)

    monkeypatch.setattr(DocumentRepository, "create", flaky_create)

    with pytest.raises(RuntimeError, match="forced sql create failure after index write"):
        client.post(
            "/api/documents",
            files={"file": ("notes.pdf", payload, "application/pdf")},
            headers=headers,
        )
    assert create_calls["n"] == 1
    assert chroma_collection.count() == 2
    assert len(hybrid.bm25._chunks) == 2
    assert all(meta["source"] == "notes.pdf" for meta in chroma_collection.get(include=["metadatas"])["metadatas"])
    assert all(c["source"] == "notes.pdf" for c in hybrid.bm25._chunks)

    retry = client.post(
        "/api/documents",
        files={"file": ("retry-notes.pdf", payload, "application/pdf")},
        headers=headers,
    )
    assert retry.status_code == 200
    assert retry.json()["filename"] == "retry-notes.pdf"
    assert create_calls["n"] == 2

    assert chroma_collection.count() == 2
    stored = chroma_collection.get(include=["metadatas"])
    assert len(stored["ids"]) == 2
    assert all(meta["source"] == "retry-notes.pdf" for meta in stored["metadatas"])
    assert len(hybrid.bm25._chunks) == 2
    assert all(c["source"] == "retry-notes.pdf" for c in hybrid.bm25._chunks)
    assert {c["chunk_id"] for c in hybrid.bm25._chunks} == set(stored["ids"])

    listed = client.get("/api/documents", headers=headers)
    assert listed.status_code == 200
    docs = listed.json()
    assert any(
        d["id"] == retry.json()["document_id"]
        and d["filename"] == "retry-notes.pdf"
        and d["chunks_count"] == 2
        for d in docs
    )


def test_concurrent_first_uploads_converge_on_winning_sql_filename(
    app,
    fake_embedder,
    chroma_collection,
    stub_document_processor,
    monkeypatch,
):
    """Two first uploads of the same bytes/different names must share one source."""
    from concurrent.futures import ThreadPoolExecutor

    from app.db.repositories import DocumentRepository
    from app.rag.hybrid_retriever import BM25Index, HybridRetriever
    from app.rag.retriever import Retriever

    hybrid = HybridRetriever(
        dense=Retriever(collection=chroma_collection, embedder=fake_embedder),
        bm25=BM25Index(),
    )
    app.state.retriever = hybrid
    payload = b"%PDF-1.4 concurrent-canonical-source"
    from app.db.session import session_scope

    with session_scope() as session:
        ensure_user(session, "same-user")
    lookup_lock = threading.Lock()
    lookup_count = {"n": 0}
    create_barrier = threading.Barrier(2)
    real_get = DocumentRepository.get_by_user_and_hash
    real_create = DocumentRepository.create

    def gated_get(self, *, user_id: str, hash_: str):
        with lookup_lock:
            lookup_count["n"] += 1
            n = lookup_count["n"]
        # Force both route-level first lookups to miss any SQL row.
        if n <= 2:
            return None
        return real_get(self, user_id=user_id, hash_=hash_)

    def gated_create(self, **kwargs):
        # Both requests finish indexing under their own filename, then race create.
        create_barrier.wait(timeout=15)
        return real_create(self, **kwargs)

    monkeypatch.setattr(DocumentRepository, "get_by_user_and_hash", gated_get)
    monkeypatch.setattr(DocumentRepository, "create", gated_create)

    token = issue_token("same-user", "guest")

    def upload(filename: str):
        with TestClient(app) as local_client:
            return local_client.post(
                "/api/documents",
                files={"file": (filename, payload, "application/pdf")},
                headers={"Authorization": f"Bearer {token}"},
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(upload, "a.pdf")
        future_b = pool.submit(upload, "b.pdf")
        response_a = future_a.result(timeout=30)
        response_b = future_b.result(timeout=30)

    assert response_a.status_code == 200
    assert response_b.status_code == 200
    body_a = response_a.json()
    body_b = response_b.json()
    assert body_a["document_id"] == body_b["document_id"]
    assert body_a["filename"] == body_b["filename"]
    winner = body_a["filename"]
    assert winner in {"a.pdf", "b.pdf"}

    assert chroma_collection.count() == 2
    stored = chroma_collection.get(include=["metadatas"])
    assert len(stored["ids"]) == 2
    assert all(meta["source"] == winner for meta in stored["metadatas"])
    assert len(hybrid.bm25._chunks) == 2
    assert len(hybrid.bm25._tokenized) == 2
    assert all(c["source"] == winner for c in hybrid.bm25._chunks)
    assert {c["chunk_id"] for c in hybrid.bm25._chunks} == set(stored["ids"])

    with TestClient(app) as local_client:
        listed = local_client.get(
            "/api/documents",
            headers={"Authorization": f"Bearer {token}"},
        )
    assert listed.status_code == 200
    user_docs = [d for d in listed.json() if d["id"] == body_a["document_id"]]
    assert len(user_docs) == 1
    assert user_docs[0]["filename"] == winner
    assert user_docs[0]["chunks_count"] == 2


def test_upload_rewrites_chunk_ids_from_content_hash_not_temp_basename(
    client,
    stub_retriever,
    stub_document_processor,
):
    response = client.post(
        "/api/documents",
        files={"file": ("lecture.pdf", b"%PDF-1.4 hash-me", "application/pdf")},
    )

    assert response.status_code == 200
    temp_name = stub_document_processor.paths[0].name
    for chunk in stub_retriever.added:
        assert temp_name not in chunk["chunk_id"]
        assert chunk["source"] == "lecture.pdf"
        assert chunk["chunk_id"].count(":") >= 2



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

    response = client.post(
        "/api/documents",
        files={"file": ("broken.pdf", b"%PDF-1.4 still broken", "application/pdf")},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_pdf"
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
                    headers={
                        "Authorization": (
                            f"Bearer {issue_token('default-user', 'guest')}"
                        )
                    },
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


def test_reset_is_rejected_while_multipart_upload_body_is_pending(
    app,
    client,
    monkeypatch,
):
    """Shared lease must be held before multipart body completes."""
    monkeypatch.setenv("STUDY_COACH_LOCAL_MODE", "1")

    body_blocked = threading.Event()
    release_body = threading.Event()
    upload_result: dict[str, object] = {}

    boundary = "----SlowUploadBoundary"
    pdf_bytes = b"%PDF-1.4 slow-upload-body-content"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="slow.pdf"\r\n'
        f"Content-Type: application/pdf\r\n\r\n"
    ).encode() + pdf_bytes + f"\r\n--{boundary}--\r\n".encode()
    first, rest = body[:48], body[48:]
    from app.db.session import session_scope

    with session_scope() as session:
        ensure_user(session, "upload-user")
    token = issue_token("upload-user", "guest")

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/documents",
        "raw_path": b"/api/documents",
        "query_string": b"",
        "headers": [
            (b"host", b"testserver"),
            (
                b"content-type",
                f"multipart/form-data; boundary={boundary}".encode(),
            ),
            (b"content-length", str(len(body)).encode()),
            (b"authorization", f"Bearer {token}".encode()),
        ],
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
        "app": app,
        "state": {},
    }
    phase = {"n": 0}
    response_messages: list[dict] = []

    async def receive():
        if phase["n"] == 0:
            phase["n"] = 1
            return {"type": "http.request", "body": first, "more_body": True}
        if phase["n"] == 1:
            phase["n"] = 2
            body_blocked.set()
            await asyncio.to_thread(release_body.wait)
            return {"type": "http.request", "body": rest, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        response_messages.append(message)

    def run_upload() -> None:
        try:
            asyncio.run(app(scope, receive, send))
            upload_result["ok"] = True
        except BaseException as exc:  # pragma: no cover - reported by main thread
            upload_result["error"] = exc

    thread = threading.Thread(target=run_upload)
    thread.start()
    try:
        assert body_blocked.wait(timeout=5), "multipart body never blocked"
        assert app.state.data_lifecycle_gate._active_operations > 0
        reset = client.post(
            "/api/data/reset",
            headers={"Authorization": f"Bearer {issue_token('reset-user', 'member')}"},
            json={
                "scope": "learning",
                "confirmation": "CLEAR_LEARNING_DATA",
            },
        )
    finally:
        release_body.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert "error" not in upload_result
    assert reset.status_code == 409
    assert reset.json()["detail"]["code"] == "data_operation_in_progress"
    starts = [m for m in response_messages if m.get("type") == "http.response.start"]
    assert starts and starts[0]["status"] == 200


def test_shared_lease_releases_after_learning_route_exception(app, client):
    def boom():
        raise RuntimeError("forced learning failure")

    app.dependency_overrides[get_session] = boom

    with pytest.raises(RuntimeError, match="forced learning failure"):
        client.get("/api/documents")

    with app.state.data_lifecycle_gate.exclusive_reset():
        pass


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


def test_middleware_reset_conflict_includes_cors_headers(app, client):
    with app.state.data_lifecycle_gate.exclusive_reset():
        response = client.get(
            "/api/documents",
            headers={"Origin": "http://localhost:5173"},
        )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "reset_in_progress"
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert response.headers.get("access-control-allow-credentials") == "true"


def test_middleware_recovery_conflict_includes_cors_headers(app, client):
    app.state.data_lifecycle_gate.mark_recovery_required("learning")

    response = client.get(
        "/api/documents",
        headers={"Origin": "http://localhost:5173"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "code": "reset_recovery_required",
        "required_scope": "learning",
        "message": "A previous data reset is incomplete. Retry that reset.",
    }
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert response.headers.get("access-control-allow-credentials") == "true"


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


# --- Batch A: LLM error detail boundary ------------------------------------

_MARKER = "SECRET_OPAQUE_MARKER_7f3a"
_SAFE_CONNECTION = "ConnectionError: Could not connect to the model service."
_SAFE_AUTH = "AuthenticationError: Model authentication failed."
_SAFE_RATE_LIMIT = "RateLimitError: The model service rate limit was reached."

_CHAT_HEADERS = {
    "x-fingerprint": "fp-1",
    "x-provider": "ollama",
    "x-model": "gemma3:4b",
}


def _raw_agent_run(llm_error: str | None) -> dict:
    """A `serialize_public()`-shaped run as a pre-fix emitter would send it."""
    return {
        "node": "planner",
        "mode": "agent_loop",
        "total_iterations": 0,
        "total_tool_calls": 0,
        "tool_call_breakdown": {},
        "tool_errors": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "wall_time_s": 0.01,
        "exit_reason": "llm_call_failed",
        "llm_error": llm_error,
        "tool_calls": [],
    }


def _read_sse_events(resp) -> list[dict]:
    return [json.loads(line[6:]) for line in resp.iter_lines() if line.startswith("data: ")]


def _rich_agent_run(llm_error: str | None) -> dict:
    """A non-trivial but valid `serialize_public()` run.

    Non-zero counters, tokens and a tool breakdown make the cross-boundary
    assertion meaningful: only `llm_error` may change.
    """
    return {
        "node": "quiz",
        "mode": "agent_loop",
        "total_iterations": 3,
        "total_tool_calls": 2,
        "tool_call_breakdown": {"retriever_search": 1, "persist_quiz_question": 1},
        "tool_errors": 1,
        "input_tokens": 412,
        "output_tokens": 137,
        "wall_time_s": 1.25,
        "exit_reason": "llm_call_failed",
        "llm_error": llm_error,
        "tool_calls": [
            {
                "name": "retriever_search",
                "error": False,
                "args_preview": '{"query":"HyDE","top_k":3}',
                "output_preview": "3 chunks",
            },
            {
                "name": "persist_quiz_question",
                "error": True,
                "args_preview": '{"topic":"HyDE"}',
                "output_preview": "Error calling persist_quiz_question: invalid options",
            },
        ],
    }


def test_ping_note_is_safe_when_model_factory_raises(client, monkeypatch):
    def boom(_config):
        raise ConnectionError(
            f"cannot reach http://127.0.0.1:11434/api/chat?token={_MARKER}"
        )

    monkeypatch.setattr("app.llm.provider.get_chat_model", boom)

    response = client.get(
        "/api/models/ping",
        headers={"x-provider": "ollama", "x-model": "gemma3:4b"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert body["model"] == "gemma3:4b"
    assert isinstance(body["latency_ms"], (int, float))
    assert body["note"] == f"Failed: {_SAFE_CONNECTION}"
    assert _MARKER not in response.text


def test_ping_note_is_safe_when_ainvoke_raises_sdk_style_error(client, monkeypatch):
    # Class *name* only: Batch A never imports a provider SDK.
    synthetic_auth_error = type("AuthenticationError", (Exception,), {})

    class FailingModel:
        async def ainvoke(self, _messages, **_kwargs):
            raise synthetic_auth_error(
                f"Incorrect API key provided: sk-live-{_MARKER}. "
                f'{{"api_key": "sk-live-{_MARKER}"}}'
            )

    monkeypatch.setattr("app.llm.provider.get_chat_model", lambda _config: FailingModel())

    response = client.get(
        "/api/models/ping",
        headers={"x-provider": "ollama", "x-model": "gemma3:4b"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert body["note"] == f"Failed: {_SAFE_AUTH}"
    assert _MARKER not in response.text


def test_ping_success_contract_is_unchanged(client, monkeypatch):
    class PongModel:
        async def ainvoke(self, _messages, **_kwargs):
            return AIMessage(content="pong")

    monkeypatch.setattr("app.llm.provider.get_chat_model", lambda _config: PongModel())

    response = client.get(
        "/api/models/ping",
        headers={"x-provider": "ollama", "x-model": "gemma3:4b"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["model"] == "gemma3:4b"
    assert body["note"].startswith("Connected — responded in")
    assert body["note"].endswith("ms")


def test_ping_empty_response_reports_failure_with_fixed_note(client, monkeypatch):
    """D2 deliberately changes the Batch A legacy behaviour: a ping response
    without a usable text body is a failed check (ok=False) with a fixed safe
    note, never a success. The route boundary (200 result, fixed fields) is
    preserved on purpose."""

    class EmptyModel:
        async def ainvoke(self, _messages, **_kwargs):
            return AIMessage(content="")

    monkeypatch.setattr("app.llm.provider.get_chat_model", lambda _config: EmptyModel())

    response = client.get(
        "/api/models/ping",
        headers={"x-provider": "ollama", "x-model": "gemma3:4b"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert body["model"] == "gemma3:4b"
    assert body["note"] == _PING_NO_TEXT_NOTE


# --- D2: ping requires a valid response body --------------------------------

_PING_NO_TEXT_NOTE = (
    "Failed: the model response did not contain a usable text body."
)


def test_ping_whitespace_response_reports_failure_with_fixed_note(
    client, monkeypatch, deny_model_route_network
):
    class WhitespaceModel:
        async def ainvoke(self, _messages, **_kwargs):
            return AIMessage(content="   \n\t  ")

    monkeypatch.setattr(
        "app.llm.provider.get_chat_model", lambda _config: WhitespaceModel()
    )

    response = client.get(
        "/api/models/ping",
        headers={"x-provider": "ollama", "x-model": "gemma3:4b"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert body["note"] == _PING_NO_TEXT_NOTE


def test_ping_skipped_only_blocks_report_failure_without_echoing_payload(
    client, monkeypatch, deny_model_route_network
):
    class ReasoningOnlyModel:
        async def ainvoke(self, _messages, **_kwargs):
            return AIMessage(
                content=[{"type": "reasoning", "thinking": f"secret-{_MARKER}"}]
            )

    monkeypatch.setattr(
        "app.llm.provider.get_chat_model", lambda _config: ReasoningOnlyModel()
    )

    response = client.get(
        "/api/models/ping",
        headers={"x-provider": "ollama", "x-model": "gemma3:4b"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert body["note"] == _PING_NO_TEXT_NOTE
    assert _MARKER not in response.text


def test_ping_malformed_shape_reports_failure_without_echoing_payload(
    client, monkeypatch, deny_model_route_network
):
    class MalformedModel:
        async def ainvoke(self, _messages, **_kwargs):
            return AIMessage(
                content=[{"type": "text", "text": {"leak": _MARKER}}]
            )

    monkeypatch.setattr(
        "app.llm.provider.get_chat_model", lambda _config: MalformedModel()
    )

    response = client.get(
        "/api/models/ping",
        headers={"x-provider": "ollama", "x-model": "gemma3:4b"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert body["note"] == _PING_NO_TEXT_NOTE
    assert _MARKER not in response.text


def test_ping_note_for_body_failure_is_not_a_normalized_llm_error(
    client, monkeypatch, deny_model_route_network
):
    """The body-validation failure must keep its own fixed note — it is not
    routed through normalize_llm_error like factory/ainvoke failures."""

    class EmptyModel:
        async def ainvoke(self, _messages, **_kwargs):
            return AIMessage(content="")

    monkeypatch.setattr("app.llm.provider.get_chat_model", lambda _config: EmptyModel())

    response = client.get(
        "/api/models/ping",
        headers={"x-provider": "ollama", "x-model": "gemma3:4b"},
    )

    assert response.status_code == 200
    note = response.json()["note"]
    assert note == _PING_NO_TEXT_NOTE
    assert note not in {
        f"Failed: {_SAFE_CONNECTION}",
        f"Failed: {_SAFE_AUTH}",
        f"Failed: {_SAFE_RATE_LIMIT}",
    }


def test_ping_text_block_body_succeeds(client, monkeypatch, deny_model_route_network):
    class TextBlockModel:
        async def ainvoke(self, _messages, **_kwargs):
            return AIMessage(
                content=[
                    {"type": "text", "text": "pon"},
                    {"type": "text", "text": "g"},
                ]
            )

    monkeypatch.setattr(
        "app.llm.provider.get_chat_model", lambda _config: TextBlockModel()
    )

    response = client.get(
        "/api/models/ping",
        headers={"x-provider": "ollama", "x-model": "gemma3:4b"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["note"].startswith("Connected — responded in")


# --- D2: tool-check completes a real tool round trip ------------------------

_TOOL_CHECK_OK_NOTE = "Tool call round trip completed."
_TOOL_CHECK_NO_CALLS_NOTE = (
    "No tool calls observed in this probe — deterministic mode will be used."
)
_TOOL_CHECK_INCOMPLETE_NOTE = "Tool check did not complete a valid round trip."
_SAFE_TIMEOUT = "APITimeoutError: The model request timed out."


class _ScriptedBound:
    """Stands in for the bound runnable; replays a fixed response script."""

    def __init__(self, script: list, calls: list):
        self._script = script
        self._calls = calls

    async def ainvoke(self, messages, **_kwargs):
        self._calls.append(list(messages))
        step = self._script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


class _ScriptedToolModel:
    """Stands in for the factory model; bind_tools returns one bound runnable."""

    def __init__(self, script: list):
        self._script = script
        self.ainvoke_calls: list[list] = []
        self.bind_calls = 0

    def bind_tools(self, _tools):
        self.bind_calls += 1
        return _ScriptedBound(self._script, self.ainvoke_calls)


def _install_scripted_tool_model(monkeypatch, script) -> _ScriptedToolModel:
    model = _ScriptedToolModel(script)
    monkeypatch.setattr("app.llm.provider.get_chat_model", lambda _config: model)
    return model


def _ping_tool_call(call_id: str = "call-1") -> dict:
    return {"name": "ping", "args": {}, "id": call_id}


_HEADERS = {"x-provider": "ollama", "x-model": "gemma3:4b"}


def test_tool_check_round_trip_executes_tool_once_and_reports_capable(
    client, monkeypatch, deny_model_route_network
):
    first = AIMessage(content="", tool_calls=[_ping_tool_call("call-1")])
    second = AIMessage(content="pong after tool")
    model = _install_scripted_tool_model(monkeypatch, [first, second])

    response = client.get("/api/models/tool-check", headers=_HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["tool_capable"] is True
    assert body["model"] == "gemma3:4b"
    assert body["note"] == _TOOL_CHECK_OK_NOTE
    # Exactly two model invocations, one bind, and the synthetic ping tool
    # produced the real ToolMessage in the second-round history.
    assert model.bind_calls == 1
    assert len(model.ainvoke_calls) == 2
    first_round, second_round = model.ainvoke_calls
    assert len(first_round) == 1
    assert first_round[0].content == "Call the ping tool"
    assert second_round[1] is first  # original AIMessage passed through as-is
    tool_message = second_round[2]
    assert type(tool_message).__name__ == "ToolMessage"
    assert tool_message.tool_call_id == "call-1"
    assert tool_message.content == "pong"


def test_tool_check_reports_false_when_probe_observes_no_tool_calls(
    client, monkeypatch, deny_model_route_network
):
    model = _install_scripted_tool_model(
        monkeypatch, [AIMessage(content="I am a plain text model")]
    )

    response = client.get("/api/models/tool-check", headers=_HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["tool_capable"] is False
    assert body["note"] == _TOOL_CHECK_NO_CALLS_NOTE
    assert len(model.ainvoke_calls) == 1


def test_tool_check_unknown_when_first_round_carries_invalid_tool_calls(
    client, monkeypatch, deny_model_route_network
):
    model = _install_scripted_tool_model(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[_ping_tool_call()],
                invalid_tool_calls=[
                    {"name": "ping", "args": "not-json", "id": "bad", "error": None}
                ],
            )
        ],
    )

    response = client.get("/api/models/tool-check", headers=_HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["tool_capable"] is None
    assert body["note"] == _TOOL_CHECK_INCOMPLETE_NOTE
    assert len(model.ainvoke_calls) == 1


def test_tool_check_unknown_when_tool_call_args_are_not_empty(
    client, monkeypatch, deny_model_route_network
):
    model = _install_scripted_tool_model(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "ping", "args": {"q": "x"}, "id": "call-1"}],
            )
        ],
    )

    response = client.get("/api/models/tool-check", headers=_HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["tool_capable"] is None
    assert body["note"] == _TOOL_CHECK_INCOMPLETE_NOTE
    assert len(model.ainvoke_calls) == 1  # unknown tool shape is never executed


def test_tool_check_unknown_when_tool_call_name_is_not_ping(
    client, monkeypatch, deny_model_route_network
):
    model = _install_scripted_tool_model(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "mystery_tool", "args": {}, "id": "call-1"}],
            )
        ],
    )

    response = client.get("/api/models/tool-check", headers=_HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["tool_capable"] is None
    assert body["note"] == _TOOL_CHECK_INCOMPLETE_NOTE
    assert len(model.ainvoke_calls) == 1  # unknown tool is never executed


@pytest.mark.parametrize("bad_id", ["", None])
def test_tool_check_unknown_when_tool_call_id_is_not_a_nonempty_string(
    client, monkeypatch, deny_model_route_network, bad_id
):
    # Note: LangChain's AIMessage validation rejects a non-string id outright
    # (e.g. 42), so only the empty/None shapes are constructible here; the
    # route additionally guards with isinstance(id, str).
    _install_scripted_tool_model(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "ping", "args": {}, "id": bad_id}],
            )
        ],
    )

    response = client.get("/api/models/tool-check", headers=_HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["tool_capable"] is None
    assert body["note"] == _TOOL_CHECK_INCOMPLETE_NOTE


def test_tool_check_unknown_when_first_round_returns_multiple_tool_calls(
    client, monkeypatch, deny_model_route_network
):
    _install_scripted_tool_model(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[_ping_tool_call("call-1"), _ping_tool_call("call-2")],
            )
        ],
    )

    response = client.get("/api/models/tool-check", headers=_HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["tool_capable"] is None
    assert body["note"] == _TOOL_CHECK_INCOMPLETE_NOTE


def test_tool_check_unknown_when_second_round_starts_new_tool_calls(
    client, monkeypatch, deny_model_route_network
):
    _install_scripted_tool_model(
        monkeypatch,
        [
            AIMessage(content="", tool_calls=[_ping_tool_call("call-1")]),
            AIMessage(content="", tool_calls=[_ping_tool_call("call-2")]),
        ],
    )

    response = client.get("/api/models/tool-check", headers=_HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["tool_capable"] is None
    assert body["note"] == _TOOL_CHECK_INCOMPLETE_NOTE


def test_tool_check_unknown_when_second_round_body_is_empty(
    client, monkeypatch, deny_model_route_network
):
    _install_scripted_tool_model(
        monkeypatch,
        [
            AIMessage(content="", tool_calls=[_ping_tool_call()]),
            AIMessage(content="   "),
        ],
    )

    response = client.get("/api/models/tool-check", headers=_HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["tool_capable"] is None
    assert body["note"] == _TOOL_CHECK_INCOMPLETE_NOTE


def test_tool_check_unknown_when_probe_without_calls_has_no_usable_body(
    client, monkeypatch, deny_model_route_network
):
    _install_scripted_tool_model(monkeypatch, [AIMessage(content="")])

    response = client.get("/api/models/tool-check", headers=_HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["tool_capable"] is None
    assert body["note"] == _TOOL_CHECK_INCOMPLETE_NOTE


def test_tool_check_note_is_safe_when_factory_raises(
    client, monkeypatch, deny_model_route_network
):
    def boom(_config):
        raise ConnectionError(f"cannot reach http://127.0.0.1:11434?token={_MARKER}")

    monkeypatch.setattr("app.llm.provider.get_chat_model", boom)

    response = client.get("/api/models/tool-check", headers=_HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["tool_capable"] is None
    assert body["note"] == f"Tool check failed: {_SAFE_CONNECTION}"
    assert _MARKER not in response.text


def test_tool_check_note_is_safe_when_first_ainvoke_raises(
    client, monkeypatch, deny_model_route_network
):
    synthetic_auth_error = type("AuthenticationError", (Exception,), {})
    _install_scripted_tool_model(
        monkeypatch, [synthetic_auth_error(f"bad key {_MARKER}")]
    )

    response = client.get("/api/models/tool-check", headers=_HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["tool_capable"] is None
    assert body["note"] == f"Tool check failed: {_SAFE_AUTH}"
    assert _MARKER not in response.text
    assert "AuthenticationError(" not in body["note"]


def test_tool_check_note_is_safe_when_second_ainvoke_raises(
    client, monkeypatch, deny_model_route_network
):
    synthetic_timeout_error = type("APITimeoutError", (Exception,), {})
    _install_scripted_tool_model(
        monkeypatch,
        [
            AIMessage(content="", tool_calls=[_ping_tool_call()]),
            synthetic_timeout_error(f"timed out {_MARKER}"),
        ],
    )

    response = client.get("/api/models/tool-check", headers=_HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["tool_capable"] is None
    assert body["note"] == f"Tool check failed: {_SAFE_TIMEOUT}"
    assert _MARKER not in response.text


# --- D2 repair: synthetic malformed response containers → fixed 200 results -
#
# These stubs are clearly synthetic model-boundary shapes used to pin the
# route's protocol defense. They are NOT evidence that the locked real SDKs
# produce such values, and the assertions make no real-provider claim: any
# non-standard response container must yield the fixed "did not complete"
# null result (200) instead of a 500, without extra tool executions or model
# calls. The real-message controls are the existing round-trip / no-call
# tests above.

_SYNTHETIC_TOOL_CHECK_CASES = [
    pytest.param(None, id="synthetic-missing-response"),
    pytest.param(
        SimpleNamespace(tool_calls=[], invalid_tool_calls=[]),
        id="synthetic-missing-content",
    ),
    pytest.param(
        SimpleNamespace(tool_calls=["synthetic-malformed"], invalid_tool_calls=[], content="body"),
        id="synthetic-non-mapping-call",
    ),
    pytest.param(
        SimpleNamespace(tool_calls=42, invalid_tool_calls=[], content="body"),
        id="synthetic-non-list-calls",
    ),
]


@pytest.mark.parametrize("synthetic_response", _SYNTHETIC_TOOL_CHECK_CASES)
def test_tool_check_synthetic_malformed_first_round_reports_fixed_null(
    client, monkeypatch, deny_model_route_network, synthetic_response
):
    model = _install_scripted_tool_model(monkeypatch, [synthetic_response])

    response = client.get("/api/models/tool-check", headers=_HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["tool_capable"] is None
    assert body["note"] == _TOOL_CHECK_INCOMPLETE_NOTE
    assert _MARKER not in response.text
    # No second model call and no tool execution for a malformed container.
    assert len(model.ainvoke_calls) == 1


def test_tool_check_synthetic_malformed_second_round_reports_fixed_null(
    client, monkeypatch, deny_model_route_network
):
    model = _install_scripted_tool_model(
        monkeypatch,
        [
            AIMessage(content="", tool_calls=[_ping_tool_call()]),
            SimpleNamespace(tool_calls=42, invalid_tool_calls=[], content="body"),
        ],
    )

    response = client.get("/api/models/tool-check", headers=_HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["tool_capable"] is None
    assert body["note"] == _TOOL_CHECK_INCOMPLETE_NOTE
    # Exactly the two invocations of the round trip: the malformed second
    # response neither restarts the probe nor triggers extra tool work.
    assert len(model.ainvoke_calls) == 2


def test_ping_synthetic_malformed_response_reports_fixed_failure(
    client, monkeypatch, deny_model_route_network
):
    for synthetic_response, case_id in [
        (None, "missing-response"),
        (SimpleNamespace(), "missing-content"),
    ]:
        class SyntheticPingModel:
            async def ainvoke(self, _messages, **_kwargs):
                return synthetic_response

        monkeypatch.setattr(
            "app.llm.provider.get_chat_model", lambda _config: SyntheticPingModel()
        )

        response = client.get(
            "/api/models/ping",
            headers={"x-provider": "ollama", "x-model": "gemma3:4b"},
        )

        assert response.status_code == 200, case_id
        body = response.json()
        assert body["ok"] is False, case_id
        assert body["note"] == _PING_NO_TEXT_NOTE, case_id
        assert _MARKER not in response.text, case_id


# --- D2: strict connection config → stable 400 before the model factory -----

_INVALID_PROVIDER_MESSAGE = "Provider must be ollama, openai, anthropic, or gemini."
_INVALID_MODEL_MESSAGE = "A valid model is required for the selected provider."
_INVALID_API_KEY_MESSAGE = "An API key is required for the selected provider."


@pytest.fixture()
def deny_model_route_network(monkeypatch):
    """Offline guard for the D2 model-route consumer tests.

    Every real DNS resolution or socket connection attempt fails the test
    immediately; the probe test below confirms the guard covers the real
    socket surface while the tests themselves only use inert stubs.
    """

    def deny(*args, **kwargs):
        raise AssertionError("model route test attempted real network access")

    monkeypatch.setattr(socket, "getaddrinfo", deny)
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)


def test_model_route_network_guard_covers_real_socket_surface(deny_model_route_network):
    probe = socket.socket()
    try:
        with pytest.raises(AssertionError):
            socket.getaddrinfo("localhost", 80)
        with pytest.raises(AssertionError):
            probe.connect(("127.0.0.1", 80))
        with pytest.raises(AssertionError):
            probe.connect_ex(("127.0.0.1", 80))
        with pytest.raises(AssertionError):
            socket.create_connection(("127.0.0.1", 80))
    finally:
        probe.close()


def _install_counting_factory(monkeypatch):
    """Stub the model factory and fail loudly if it is ever reached."""
    calls: list[object] = []

    def counting_factory(_config):
        calls.append(_config)
        raise AssertionError("model factory must not run for an invalid config")

    monkeypatch.setattr("app.llm.provider.get_chat_model", counting_factory)
    return calls


@pytest.mark.parametrize(
    ("headers", "field", "message"),
    [
        ({"x-provider": "vertexai", "x-model": "m", "x-api-key": "sk-x"},
         "provider", _INVALID_PROVIDER_MESSAGE),
        ({"x-provider": "", "x-model": "m"},
         "provider", _INVALID_PROVIDER_MESSAGE),
        ({"x-provider": "   ", "x-model": "m"},
         "provider", _INVALID_PROVIDER_MESSAGE),
        ({"x-provider": "openai", "x-api-key": "sk-x"},
         "model", _INVALID_MODEL_MESSAGE),
        ({"x-provider": "openai", "x-model": "   ", "x-api-key": "sk-x"},
         "model", _INVALID_MODEL_MESSAGE),
        ({"x-provider": "ollama", "x-model": ""},
         "model", _INVALID_MODEL_MESSAGE),
        ({"x-provider": "openai", "x-model": "m"},
         "api_key", _INVALID_API_KEY_MESSAGE),
        ({"x-provider": "openai", "x-model": "m", "x-api-key": "   "},
         "api_key", _INVALID_API_KEY_MESSAGE),
        ({"x-provider": "gemini", "x-model": "m"},
         "api_key", _INVALID_API_KEY_MESSAGE),
    ],
)
@pytest.mark.parametrize("path", ["/api/models/ping", "/api/models/tool-check"])
def test_model_routes_reject_invalid_config_with_stable_400_before_factory(
    client, monkeypatch, deny_model_route_network, headers, field, message, path
):
    calls = _install_counting_factory(monkeypatch)

    response = client.get(path, headers=headers)

    assert response.status_code == 400
    body = response.json()["detail"]
    assert body == {"code": "invalid_llm_config", "field": field, "message": message}
    assert calls == []


@pytest.mark.parametrize("path", ["/api/models/ping", "/api/models/tool-check"])
def test_model_routes_never_echo_the_rejected_header_value(
    client, monkeypatch, deny_model_route_network, path
):
    _install_counting_factory(monkeypatch)
    marker_provider = f"bad-provider-{_MARKER}"

    response = client.get(
        path,
        headers={"x-provider": marker_provider, "x-model": "m", "x-api-key": "sk-x"},
    )

    assert response.status_code == 400
    assert marker_provider not in response.text


def test_models_ping_passes_normalized_config_to_the_factory(
    client, monkeypatch, deny_model_route_network
):
    captured: dict[str, object] = {}

    class PongModel:
        async def ainvoke(self, _messages, **_kwargs):
            return AIMessage(content="pong")

    def fake_factory(config):
        captured["config"] = config
        return PongModel()

    monkeypatch.setattr("app.llm.provider.get_chat_model", fake_factory)

    response = client.get(
        "/api/models/ping",
        headers={
            "x-provider": "  google_genai ",
            "x-model": "  gemini-2.5-flash  ",
            "x-api-key": "gm-offline-key",
        },
    )

    assert response.status_code == 200
    config = captured["config"]
    assert config.provider == "google_genai"
    assert config.model == "gemini-2.5-flash"
    assert config.api_key == "gm-offline-key"


def test_chat_projects_raw_agent_run_before_sse_and_persistence(app, client):
    from sqlalchemy import text

    from app.api.deps import get_graph
    from app.db.session import session_scope

    raw = f"AuthenticationError: Incorrect API key provided: sk-live-{_MARKER}"
    chunk = {"type": "agent_run", "run": _rich_agent_run(raw)}
    # Full-fidelity snapshot: the graph chunk must come back byte-identical.
    chunk_snapshot = json.dumps(chunk, sort_keys=True)
    expected = _rich_agent_run(_SAFE_AUTH)

    class RawAgentRunGraph:
        async def astream(self, _input_state, **_kwargs):
            yield {"type": "citations", "citations": []}
            yield {"type": "token", "text": "done"}
            yield chunk

    app.dependency_overrides[get_graph] = lambda: RawAgentRunGraph()

    with client.stream(
        "POST",
        "/api/chat",
        json={"message": "plan on HyDE", "session_id": "batch-a-sse"},
        headers=_CHAT_HEADERS,
    ) as resp:
        assert resp.status_code == 200
        events = _read_sse_events(resp)

    session_id = next(e["session_id"] for e in events if e["type"] == "session")
    sse_run = next(e["run"] for e in events if e["type"] == "agent_run")
    # Only llm_error is replaced; counters, tokens, breakdown, tool previews and
    # the rest of the envelope are identical to the input.
    assert sse_run == expected
    assert sse_run["total_iterations"] == 3
    assert sse_run["total_tool_calls"] == 2
    assert sse_run["tool_call_breakdown"] == {
        "retriever_search": 1,
        "persist_quiz_question": 1,
    }
    assert sse_run["tool_errors"] == 1
    assert sse_run["input_tokens"] == 412
    assert sse_run["output_tokens"] == 137
    assert sse_run["exit_reason"] == "llm_call_failed"
    assert sse_run["llm_error"] == _SAFE_AUTH
    assert _MARKER not in json.dumps(events)

    # The graph-emitted chunk is not mutated in place.
    assert json.dumps(chunk, sort_keys=True) == chunk_snapshot
    assert chunk["run"]["llm_error"] == raw

    with session_scope() as db:
        stored = db.execute(
            text(
                "SELECT tool_calls_json FROM messages "
                "WHERE session_id = :sid AND role = 'assistant'"
            ),
            {"sid": session_id},
        ).scalar()

    assert stored is not None
    assert _MARKER not in stored
    assert json.loads(stored)["agent_run"] == expected

    history = client.get(
        f"/api/chat/sessions/{session_id}/messages",
        headers=_CHAT_HEADERS,
    )
    assert history.status_code == 200
    assistant = history.json()["messages"][-1]
    assert assistant["content"] == "done"
    assert assistant["agent_run"] == expected
    assert _MARKER not in history.text


def test_history_projects_legacy_raw_llm_error_without_rewriting_the_sql_row(app, client):
    from sqlalchemy import text

    from app.db.repositories import (
        ChatSessionRepository,
        CitationRepository,
        MessageRepository,
    )
    from app.db.session import session_scope

    raw = f"RateLimitError: Rate limit reached for org-abc; Bearer sk-live-{_MARKER}"
    envelope = {
        "schema": "assistant_artifacts.v1",
        "citations": [{"chunk_id": "legacy:1:0", "source": "legacy.pdf", "page": 5}],
        "agent_run": _raw_agent_run(raw),
        "quiz_question_id": "legacy-question-id",
    }

    with session_scope() as db:
        ChatSessionRepository(db).create(user_id="default-user", chat_id="batch-a-legacy")
        message = MessageRepository(db).create(
            session_id="batch-a-legacy",
            role="assistant",
            content="legacy answer",
            tool_calls_json=envelope,
        )
        message_id = message.id
        CitationRepository(db).bulk_create_for_message(
            message_id=message_id,
            citations=[{
                "chunk_id": "legacy:1:0",
                "page": 5,
                "span_start": 0,
                "span_end": 4,
            }],
        )

    with session_scope() as db:
        before = db.execute(
            text("SELECT tool_calls_json FROM messages WHERE id = :mid"),
            {"mid": message_id},
        ).scalar()

    assert before is not None
    assert _MARKER in before  # the raw detail really is in the stored row

    response = client.get(
        "/api/chat/sessions/batch-a-legacy/messages",
        headers=_CHAT_HEADERS,
    )

    assert response.status_code == 200
    assistant = response.json()["messages"][0]
    assert assistant["content"] == "legacy answer"
    assert assistant["agent_run"]["exit_reason"] == "llm_call_failed"
    assert assistant["agent_run"]["llm_error"] == _SAFE_RATE_LIMIT
    assert assistant["citations"][0]["chunk_id"] == "legacy:1:0"
    assert assistant["citations"][0]["source"] == "legacy.pdf"
    assert assistant["citations"][0]["page"] == 5
    assert assistant["quiz_question_id"] == "legacy-question-id"
    assert _MARKER not in response.text

    with session_scope() as db:
        after = db.execute(
            text("SELECT tool_calls_json FROM messages WHERE id = :mid"),
            {"mid": message_id},
        ).scalar()

    # Read-only projection: history never rewrites the stored original.
    assert after == before


def test_history_legacy_and_missing_agent_run_semantics_are_preserved(app, client):
    from app.db.repositories import (
        ChatSessionRepository,
        CitationRepository,
        MessageRepository,
    )
    from app.db.session import session_scope

    run_without_error_key = _raw_agent_run(None)
    del run_without_error_key["llm_error"]
    run_without_error_key["exit_reason"] = "natural_stop"

    with session_scope() as db:
        ChatSessionRepository(db).create(user_id="default-user", chat_id="batch-a-compat")
        legacy_message = MessageRepository(db).create(
            session_id="batch-a-compat",
            role="assistant",
            content="legacy citation list",
            tool_calls_json=[{"chunk_id": "c:1:0", "source": "s.pdf"}],
        )
        # Citations are returned from the SQL `citations` table; the envelope
        # only carried the legacy source metadata.
        CitationRepository(db).bulk_create_for_message(
            message_id=legacy_message.id,
            citations=[{
                "chunk_id": "c:1:0",
                "page": 2,
                "span_start": 0,
                "span_end": 4,
            }],
        )
        MessageRepository(db).create(
            session_id="batch-a-compat",
            role="assistant",
            content="null agent run",
            tool_calls_json={
                "schema": "assistant_artifacts.v1",
                "citations": [],
                "agent_run": None,
            },
        )
        MessageRepository(db).create(
            session_id="batch-a-compat",
            role="assistant",
            content="run without llm_error key",
            tool_calls_json={
                "schema": "assistant_artifacts.v1",
                "citations": [],
                "agent_run": run_without_error_key,
            },
        )
        MessageRepository(db).create(
            session_id="batch-a-compat",
            role="assistant",
            content="invalid agent run",
            tool_calls_json={
                "schema": "assistant_artifacts.v1",
                "citations": [],
                "agent_run": "not-a-dict",
            },
        )

    response = client.get(
        "/api/chat/sessions/batch-a-compat/messages",
        headers=_CHAT_HEADERS,
    )

    assert response.status_code == 200
    by_content = {m["content"]: m for m in response.json()["messages"]}
    assert set(by_content) == {
        "legacy citation list",
        "null agent run",
        "run without llm_error key",
        "invalid agent run",
    }
    # Plain legacy citation list keeps its citations and has no agent_run.
    legacy_citation = by_content["legacy citation list"]["citations"][0]
    assert legacy_citation["chunk_id"] == "c:1:0"
    assert legacy_citation["page"] == 2
    assert legacy_citation["source"] == "s.pdf"
    assert by_content["legacy citation list"]["agent_run"] is None
    assert by_content["legacy citation list"]["quiz_question_id"] is None
    # null / invalid run keep the existing None handling.
    assert by_content["null agent run"]["agent_run"] is None
    assert by_content["invalid agent run"]["agent_run"] is None
    # A run that predates llm_error stays valid and reports None.
    assert by_content["run without llm_error key"]["agent_run"]["exit_reason"] == "natural_stop"
    assert by_content["run without llm_error key"]["agent_run"]["llm_error"] is None


def test_history_projection_keeps_session_ownership_isolation(app, client):
    from app.db.repositories import ChatSessionRepository, MessageRepository
    from app.db.session import session_scope

    with session_scope() as db:
        ensure_user(db, "batch-a-other-user")
        ChatSessionRepository(db).create(
            user_id="batch-a-other-user",
            chat_id="batch-a-other-session",
        )
        MessageRepository(db).create(
            session_id="batch-a-other-session",
            role="assistant",
            content="other user answer",
            tool_calls_json={
                "schema": "assistant_artifacts.v1",
                "citations": [],
                "agent_run": _raw_agent_run(f"AuthenticationError: sk-live-{_MARKER}"),
            },
        )

    response = client.get(
        "/api/chat/sessions/batch-a-other-session/messages",
        headers=_CHAT_HEADERS,
    )

    assert response.status_code == 404
    assert "other user answer" not in response.text
    assert _MARKER not in response.text


_INVALID_RUNS = [
    pytest.param(
        f"AuthenticationError: Incorrect API key provided: sk-live-{_MARKER}",
        id="string-run",
    ),
    pytest.param(
        [{"llm_error": f"ConnectionError: {_MARKER}"}],
        id="list-run",
    ),
]


@pytest.mark.parametrize("invalid_run", _INVALID_RUNS)
def test_chat_collapses_invalid_agent_run_to_none_across_all_boundaries(
    app, client, invalid_run
):
    """A malformed `agent_run` event must not skip the SSE projection.

    The invalid value collapses to None on the wire, in the persisted envelope
    and in history — the same handling the persistence path already used.
    """
    from sqlalchemy import text

    from app.api.deps import get_graph
    from app.db.session import session_scope

    chunk = {"type": "agent_run", "run": invalid_run}
    snapshot = json.dumps(chunk, sort_keys=True)

    class InvalidRunGraph:
        async def astream(self, _input_state, **_kwargs):
            yield {"type": "citations", "citations": []}
            yield {"type": "token", "text": "done"}
            yield chunk

    app.dependency_overrides[get_graph] = lambda: InvalidRunGraph()

    with client.stream(
        "POST",
        "/api/chat",
        json={"message": "plan on HyDE", "session_id": "batch-a-invalid-run"},
        headers=_CHAT_HEADERS,
    ) as resp:
        assert resp.status_code == 200
        events = _read_sse_events(resp)

    session_id = next(e["session_id"] for e in events if e["type"] == "session")
    run_event = next(e for e in events if e["type"] == "agent_run")
    # The run field stays explicit on the wire and is None, never the raw value.
    assert "run" in run_event
    assert run_event["run"] is None
    assert _MARKER not in json.dumps(events)

    # The graph-emitted chunk keeps its original value.
    assert json.dumps(chunk, sort_keys=True) == snapshot

    with session_scope() as db:
        stored = db.execute(
            text(
                "SELECT tool_calls_json FROM messages "
                "WHERE session_id = :sid AND role = 'assistant'"
            ),
            {"sid": session_id},
        ).scalar()

    assert stored is not None
    assert _MARKER not in stored
    assert json.loads(stored)["agent_run"] is None

    history = client.get(
        f"/api/chat/sessions/{session_id}/messages",
        headers=_CHAT_HEADERS,
    )
    assert history.status_code == 200
    assert history.json()["messages"][-1]["agent_run"] is None
    assert _MARKER not in history.text


def test_chat_agent_run_event_shapes_and_order_are_preserved(app, client):
    """Compatibility control for the projection rewrite.

    - an `agent_run` event without a `run` field must not gain one
    - an explicit `run: None` stays None
    - non-agent_run events pass through unchanged
    - relative event order is unchanged
    """
    from app.api.deps import get_graph

    chunks = [
        {"type": "citations", "citations": []},
        {"type": "trace", "stage": "router"},
        {"type": "agent_run"},
        {"type": "agent_run", "run": None},
        {"type": "token", "text": "done"},
        {"type": "quiz_question", "question_id": "q-1"},
    ]
    snapshot = json.dumps(chunks, sort_keys=True)

    class MixedEventGraph:
        async def astream(self, _input_state, **_kwargs):
            for item in chunks:
                yield item

    app.dependency_overrides[get_graph] = lambda: MixedEventGraph()

    with client.stream(
        "POST",
        "/api/chat",
        json={"message": "hi", "session_id": "batch-a-shapes"},
        headers=_CHAT_HEADERS,
    ) as resp:
        assert resp.status_code == 200
        events = _read_sse_events(resp)

    assert [e["type"] for e in events] == [
        "session",
        "citations",
        "trace",
        "agent_run",
        "agent_run",
        "token",
        "quiz_question",
        "done",
    ]
    run_events = [e for e in events if e["type"] == "agent_run"]
    assert "run" not in run_events[0]
    assert run_events[1]["run"] is None
    assert events[1] == {"type": "citations", "citations": []}
    assert events[2] == {"type": "trace", "stage": "router"}
    assert events[5] == {"type": "token", "text": "done"}
    assert events[6] == {"type": "quiz_question", "question_id": "q-1"}

    # No graph chunk was mutated in place.
    assert json.dumps(chunks, sort_keys=True) == snapshot
