"""Isolation and integrity contracts for the evaluation CorpusSnapshot loader."""

import asyncio
from dataclasses import replace
import gc
import threading
import time

import pytest

from app.eval.learning_run import corpus as corpus_module
from app.eval.learning_run.contracts import canonical_hash
from app.eval.learning_run.registry import RegistryError, TaskRegistry
from app.eval.learning_run.corpus import CorpusSnapshotLoader


class ExplodingDependency:
    def __init__(self, label: str) -> None:
        self.label = label
        self.calls = 0

    def __getattr__(self, name: str):
        self.calls += 1
        raise AssertionError(f"{self.label} must not be touched: {name}")


class FakeCollection:
    def __init__(self, name: str) -> None:
        self.name = name
        self.added_chunks: list[dict] = []


class FakeClient:
    def __init__(self) -> None:
        self.collections: dict[str, FakeCollection] = {}
        self.created_names: list[str] = []
        self.deleted_names: list[str] = []
        self.close_calls = 0

    def get_or_create_collection(self, name: str) -> FakeCollection:
        self.created_names.append(name)
        collection = self.collections.setdefault(name, FakeCollection(name))
        return collection

    def delete_collection(self, name: str) -> None:
        self.deleted_names.append(name)
        self.collections.pop(name, None)

    def close(self) -> None:
        self.close_calls += 1


class FakeRetriever:
    def __init__(self, collection: FakeCollection) -> None:
        self.collection = collection
        self.add_calls: list[list[dict]] = []
        self.close_calls = 0

    def add_chunks(self, chunks: list[dict]) -> None:
        copied = [dict(chunk) for chunk in chunks]
        self.add_calls.append(copied)
        self.collection.added_chunks.extend(copied)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        return self.collection.added_chunks[:top_k]

    def close(self) -> None:
        self.close_calls += 1


class FakeIsolatedBuilder:
    def __init__(self) -> None:
        self.calls: list[FakeCollection] = []

    def __call__(self, collection: FakeCollection, **kwargs) -> FakeRetriever:
        self.calls.append(collection)
        return FakeRetriever(collection)


@pytest.fixture
def snapshot():
    return TaskRegistry.load_default().corpus


def test_eval_corpus_never_uses_global_retriever_and_loads_exact_chunks(snapshot):
    global_retriever = ExplodingDependency("global retriever")
    client = FakeClient()
    builder = FakeIsolatedBuilder()
    built_retrievers: list[FakeRetriever] = []

    def recording_builder(collection):
        retriever = builder(collection)
        built_retrievers.append(retriever)
        return retriever

    loader = CorpusSnapshotLoader(
        client_factory=lambda: client,
        builder=recording_builder,
    )

    retriever = loader.load(snapshot=snapshot, global_retriever=global_retriever)

    assert global_retriever.calls == 0
    assert len(builder.calls) == 1
    assert builder.calls[0].name != "study_coach_chunks"
    assert len(client.created_names) == 1
    assert client.created_names[0] == builder.calls[0].name
    assert client.created_names[0] != "study_coach_chunks"
    assert not hasattr(retriever, "client")
    assert not hasattr(retriever, "collection")
    assert not hasattr(retriever, "retriever")
    assert not hasattr(retriever, "add_chunks")
    assert retriever.search("RRF", top_k=10) == [
        {
            "chunk_id": chunk.chunk_id,
            "content": chunk.content,
            "source": chunk.source,
            "page": chunk.page,
        }
        for chunk in snapshot.chunks
    ]
    assert built_retrievers[0].add_calls == [
        [
            {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "source": chunk.source,
                "page": chunk.page,
            }
            for chunk in snapshot.chunks
        ]
    ]


@pytest.mark.parametrize(
    "tampered_snapshot",
    [
        pytest.param(
            lambda snapshot: replace(snapshot, chunks=snapshot.chunks[:-1]),
            id="missing-chunk",
        ),
        pytest.param(
            lambda snapshot: replace(
                snapshot,
                chunks=snapshot.chunks
                + (replace(snapshot.chunks[0], chunk_id="tgqa-c-extra"),),
            ),
            id="extra-chunk",
        ),
        pytest.param(
            lambda snapshot: replace(
                snapshot,
                chunks=(replace(snapshot.chunks[0], content_hash="0" * 64),)
                + snapshot.chunks[1:],
            ),
            id="content-hash-mismatch",
        ),
        pytest.param(
            lambda snapshot: replace(snapshot, aggregate_hash="0" * 64),
            id="aggregate-hash-mismatch",
        ),
    ],
)
def test_loader_rejects_incomplete_or_tampered_snapshot_before_collection_creation(
    snapshot, tampered_snapshot
):
    client = FakeClient()
    builder = FakeIsolatedBuilder()
    loader = CorpusSnapshotLoader(
        client_factory=lambda: client,
        builder=builder,
    )

    with pytest.raises((RegistryError, ValueError), match="chunk|hash|mismatch"):
        loader.load(snapshot=tampered_snapshot(snapshot))

    assert client.created_names == []
    assert builder.calls == []


def test_loader_keeps_isolated_client_alive_for_retriever_lifetime(snapshot):
    clients: list[FakeClient] = []

    def client_factory():
        client = FakeClient()
        clients.append(client)
        return client

    loader = CorpusSnapshotLoader(
        client_factory=client_factory,
        builder=FakeIsolatedBuilder(),
    )
    retriever = loader.load(snapshot=snapshot)

    assert retriever.search("evidence", top_k=1)[0]["chunk_id"] == "tgqa-c01-rrf"


def test_builder_type_error_is_not_retried_or_swallowed(snapshot):
    client = FakeClient()
    calls: list[FakeCollection] = []

    def builder(collection):
        calls.append(collection)
        raise TypeError("builder contract failure")

    loader = CorpusSnapshotLoader(
        client_factory=lambda: client,
        builder=builder,
    )

    with pytest.raises(TypeError, match="builder contract failure"):
        loader.load(snapshot=snapshot)

    assert len(calls) == 1


def test_default_builder_uses_snapshot_embedding_mapping_not_environment(
    snapshot, monkeypatch
):
    monkeypatch.delenv("EMBED_MODEL", raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama.test")
    client = FakeClient()
    captured: dict[str, object] = {}

    def fake_builder(collection, *, embed_model, embed_host):
        captured["embed_model"] = embed_model
        captured["embed_host"] = embed_host
        return FakeRetriever(collection)

    monkeypatch.setattr(corpus_module, "build_retriever_for_collection", fake_builder)
    loader = CorpusSnapshotLoader(client_factory=lambda: client)

    loader.load(snapshot=snapshot)

    assert captured == {
        "embed_model": "nomic-embed-text",
        "embed_host": "http://ollama.test",
    }


def test_conflicting_embedding_environment_fails_before_client_creation(
    snapshot, monkeypatch
):
    monkeypatch.setenv("EMBED_MODEL", "other-model")
    client = FakeClient()
    loader = CorpusSnapshotLoader(client_factory=lambda: client)

    with pytest.raises(RegistryError, match="embedding|model"):
        loader.load(snapshot=snapshot)

    assert client.created_names == []


def test_unknown_snapshot_embedding_config_fails_before_client_creation(snapshot):
    tampered = replace(snapshot, embedding_config_version="unknown-embedding-v9")
    aggregate_hash = canonical_hash(tampered.aggregate_payload())
    tampered = replace(tampered, aggregate_hash=aggregate_hash)
    definition_payload = tampered.to_dict()
    definition_payload.pop("definition_hash", None)
    tampered = replace(tampered, definition_hash=canonical_hash(definition_payload))
    client = FakeClient()
    loader = CorpusSnapshotLoader(client_factory=lambda: client)

    with pytest.raises(RegistryError, match="embedding"):
        loader.load(snapshot=tampered)

    assert client.created_names == []


@pytest.mark.asyncio
async def test_materializer_controller_rejects_busy_admission_and_drains_late_result(snapshot):
    from app.eval.learning_run.corpus import (
        CorpusMaterializerBusyError,
        CorpusMaterializerController,
    )

    entered = threading.Event()
    allow = threading.Event()
    retrievers: list[FakeRetriever] = []

    class BlockingLoader:
        calls = 0

        def load(self, *, snapshot, stop=None, deadline=None):
            del snapshot, stop, deadline
            self.calls += 1
            entered.set()
            assert allow.wait(timeout=5)
            retriever = FakeRetriever(FakeCollection(f"late-{self.calls}"))
            retrievers.append(retriever)
            return retriever

    loader = BlockingLoader()
    controller = CorpusMaterializerController(loader)
    loop = asyncio.get_running_loop()
    first = asyncio.create_task(
        controller.acquire(snapshot=snapshot, deadline=loop.time() + 0.01)
    )
    assert await asyncio.to_thread(entered.wait, 5)

    for _ in range(2):
        with pytest.raises(CorpusMaterializerBusyError):
            await controller.acquire(snapshot=snapshot, deadline=loop.time() + 1)
    with pytest.raises(TimeoutError):
        await first
    assert controller.state == "draining"
    assert controller.outstanding_count == 1
    assert loader.calls == 1

    allow.set()
    assert await asyncio.to_thread(lambda: _wait_for_state(controller, "idle"))
    assert retrievers[0].close_calls == 1
    assert controller.outstanding_count == 0

    lease = await controller.acquire(snapshot=snapshot, deadline=loop.time() + 1)
    assert controller.state == "active"
    lease.release()
    assert await asyncio.to_thread(lambda: _wait_for_state(controller, "idle"))
    assert controller.outstanding_count == 0
    assert retrievers[1].close_calls == 1
    controller.shutdown(wait=False)


def _wait_for_state(controller, expected: str) -> bool:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if controller.state == expected:
            return True
        time.sleep(0.001)
    return controller.state == expected


@pytest.mark.asyncio
async def test_loader_stop_after_partial_add_deletes_collection_and_closes_client(snapshot):
    from app.eval.learning_run.corpus import CorpusMaterializerController

    client = FakeClient()
    entered_add = threading.Event()
    allow_add = threading.Event()
    loop = asyncio.get_running_loop()
    loop_errors: list[dict] = []
    previous_exception_handler = loop.get_exception_handler()
    loop.set_exception_handler(
        lambda _loop, context: loop_errors.append(dict(context))
    )

    class BlockingRetriever(FakeRetriever):
        def add_chunks(self, chunks):
            entered_add.set()
            assert allow_add.wait(timeout=5)
            super().add_chunks(chunks)

    def builder(collection):
        return BlockingRetriever(collection)

    loader = CorpusSnapshotLoader(client_factory=lambda: client, builder=builder)
    controller = CorpusMaterializerController(loader)
    task = asyncio.create_task(
        controller.acquire(snapshot=snapshot, deadline=loop.time() + 0.01)
    )
    try:
        assert await asyncio.to_thread(entered_add.wait, 5)
        with pytest.raises(TimeoutError):
            await task
        del task
        allow_add.set()
        assert await asyncio.to_thread(lambda: _wait_for_state(controller, "idle"))

        # Let the late worker result reach the wrapped asyncio Future, then
        # force its destructor while this test owns the loop exception hook.
        for _ in range(8):
            gc.collect()
            await asyncio.sleep(0)

        assert client.deleted_names == [client.created_names[0]]
        assert client.close_calls == 1
        assert not [
            context
            for context in loop_errors
            if context.get("message") == "Future exception was never retrieved"
        ]
    finally:
        loop.set_exception_handler(previous_exception_handler)
        controller.shutdown(wait=False)


def test_isolated_retriever_close_waits_for_search_operation_and_is_idempotent(snapshot):
    from app.eval.learning_run.corpus import IsolatedCorpusRetriever

    client = FakeClient()
    collection = FakeCollection("isolated-lock")
    search_started = threading.Event()
    release_search = threading.Event()

    class BlockingRetriever(FakeRetriever):
        def search(self, query, top_k=5):
            del query, top_k
            search_started.set()
            assert release_search.wait(timeout=5)
            return []

    retriever = BlockingRetriever(collection)
    isolated = IsolatedCorpusRetriever(
        client=client,
        collection=collection,
        retriever=retriever,
    )
    search_thread = threading.Thread(target=lambda: isolated.search("q"))
    close_done = threading.Event()
    close_thread = threading.Thread(
        target=lambda: (isolated.close(), close_done.set())
    )
    search_thread.start()
    assert search_started.wait(timeout=5)
    close_thread.start()
    assert not close_done.wait(timeout=0.01)
    assert client.deleted_names == []
    release_search.set()
    search_thread.join(timeout=5)
    close_thread.join(timeout=5)
    assert close_done.is_set()
    assert retriever.close_calls == 1
    assert client.deleted_names == [collection.name]
    assert client.close_calls == 1
    isolated.close()
    assert retriever.close_calls == 1
    assert client.close_calls == 1


@pytest.mark.asyncio
async def test_shutdown_active_schedules_controller_cleanup_and_closes_on_worker(snapshot):
    from app.eval.learning_run.corpus import CorpusMaterializerBusyError, CorpusMaterializerController, IsolatedCorpusRetriever

    client = FakeClient()
    collection = FakeCollection("shutdown-active")
    raw = FakeRetriever(collection)
    search_started = threading.Event()
    release_search = threading.Event()

    def blocking_search(_query: str, top_k: int = 5):
        del top_k
        search_started.set()
        release_search.wait(timeout=5)
        return []

    raw.search = blocking_search  # type: ignore[method-assign]

    class Loader:
        def load(self, *, snapshot, stop=None, deadline=None):
            del snapshot, stop, deadline
            return IsolatedCorpusRetriever(client=client, collection=collection, retriever=raw)

    controller = CorpusMaterializerController(Loader())
    loop = asyncio.get_running_loop()
    lease = await controller.acquire(snapshot=snapshot, deadline=loop.time() + 1)
    search_thread = threading.Thread(target=lambda: lease.retriever.search("blocked"))
    search_thread.start()
    assert await asyncio.to_thread(search_started.wait, 1)

    shutdown_done = threading.Event()
    shutdown_thread = threading.Thread(
        target=lambda: (controller.shutdown(wait=False), shutdown_done.set())
    )
    shutdown_thread.start()
    shutdown_returned = await asyncio.to_thread(shutdown_done.wait, 0.2)
    if not shutdown_returned:
        release_search.set()
        search_thread.join(timeout=5)
        shutdown_thread.join(timeout=5)
    assert shutdown_returned
    assert controller.state == "draining"
    assert controller.outstanding_count == 1
    with pytest.raises(CorpusMaterializerBusyError):
        await controller.acquire(snapshot=snapshot, deadline=loop.time() + 1)
    assert raw.close_calls == 0

    release_search.set()
    search_thread.join(timeout=5)
    assert await asyncio.to_thread(lambda: _wait_for_state(controller, "closed"))
    assert controller.outstanding_count == 0
    assert raw.close_calls == 1
    lease.release()
    controller.shutdown(wait=False)


@pytest.mark.asyncio
async def test_shutdown_materializing_closes_late_result_once_and_stays_closed(snapshot):
    from app.eval.learning_run.corpus import CorpusMaterializerBusyError, CorpusMaterializerController

    entered = threading.Event()
    allow = threading.Event()
    late: list[FakeRetriever] = []

    class BlockingLoader:
        def load(self, *, snapshot, stop=None, deadline=None):
            del snapshot, stop, deadline
            entered.set()
            allow.wait(timeout=5)
            retriever = FakeRetriever(FakeCollection("shutdown-late"))
            late.append(retriever)
            return retriever

    controller = CorpusMaterializerController(BlockingLoader())
    loop = asyncio.get_running_loop()
    acquire_task = asyncio.create_task(
        controller.acquire(snapshot=snapshot, deadline=loop.time() + 5)
    )
    assert await asyncio.to_thread(entered.wait, 1)
    controller.shutdown(wait=False)
    assert controller.state == "draining"
    with pytest.raises(CorpusMaterializerBusyError):
        await controller.acquire(snapshot=snapshot, deadline=loop.time() + 1)

    allow.set()
    with pytest.raises((TimeoutError, CorpusMaterializerBusyError)):
        await acquire_task
    assert await asyncio.to_thread(lambda: _wait_for_state(controller, "closed"))
    assert late and late[0].close_calls == 1
    assert controller.outstanding_count == 0
    controller.shutdown(wait=False)
