import asyncio
import json
import tempfile
import threading
from pathlib import Path

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
