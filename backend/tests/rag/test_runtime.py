import chromadb
import pytest
from chromadb.errors import NotFoundError

from app.rag import runtime as runtime_module
from app.rag.runtime import RetrieverRuntime


class FakeCollection:
    def __init__(
        self,
        name: str,
        count: int = 0,
        documents: list[str] | None = None,
        metadatas: list[dict] | None = None,
    ) -> None:
        self.name = name
        self._count = count
        self._documents = documents or []
        self._metadatas = metadatas or []
        self._deleted = False

    def count(self) -> int:
        if self._deleted:
            raise NotFoundError(f"Collection {self.name} does not exist")
        return self._count

    def get(self, *, include):
        return {
            "ids": [f"chunk-{index}" for index in range(len(self._documents))],
            "documents": self._documents,
            "metadatas": self._metadatas,
        }


class FakeClient:
    def __init__(self) -> None:
        self.collections: dict[str, FakeCollection] = {}
        self.created_names: list[str] = []
        self.deleted_names: list[str] = []
        self._created_once = False

    def get_or_create_collection(self, name: str) -> FakeCollection:
        self.created_names.append(name)
        if name not in self.collections:
            count = 3 if not self._created_once else 0
            self.collections[name] = FakeCollection(name, count=count)
            self._created_once = True
        return self.collections[name]

    def delete_collection(self, name: str) -> None:
        self.deleted_names.append(name)
        if name not in self.collections:
            raise NotFoundError(f"Collection {name} does not exist")
        collection = self.collections.pop(name)
        collection._deleted = True


def test_runtime_builds_retriever_and_reports_vector_count():
    client = FakeClient()
    built_for: list[FakeCollection] = []

    def builder(collection):
        built_for.append(collection)
        return object()

    runtime = RetrieverRuntime(
        client=client,
        collection_name="study_coach_chunks",
        builder=builder,
    )

    assert runtime.vector_count() == 3
    assert runtime.retriever is not None
    assert built_for == [runtime.collection]


def test_vector_count_propagates_non_not_found_errors():
    class FailingCountCollection(FakeCollection):
        def count(self) -> int:
            raise RuntimeError("storage unavailable")

    class FailingCountClient(FakeClient):
        def get_or_create_collection(self, name: str) -> FakeCollection:
            return FailingCountCollection(name)

    runtime = RetrieverRuntime(
        client=FailingCountClient(),
        collection_name="study_coach_chunks",
        builder=lambda collection: object(),
    )

    with pytest.raises(RuntimeError, match="storage unavailable"):
        runtime.vector_count()


def test_reset_empty_rebuilds_empty_collection_and_replaces_retriever_each_time():
    client = FakeClient()
    built_for: list[FakeCollection] = []

    def builder(collection):
        built_for.append(collection)
        return object()

    runtime = RetrieverRuntime(
        client=client,
        collection_name="study_coach_chunks",
        builder=builder,
    )
    previous_collection = runtime.collection
    previous_retriever = runtime.retriever

    rebuilt_retriever = runtime.reset_empty()
    first_reset_collection = runtime.collection
    twice_rebuilt_retriever = runtime.reset_empty()

    assert client.deleted_names == ["study_coach_chunks", "study_coach_chunks"]
    assert first_reset_collection is not previous_collection
    assert runtime.collection is not first_reset_collection
    assert runtime.vector_count() == 0
    assert rebuilt_retriever is not previous_retriever
    assert twice_rebuilt_retriever is runtime.retriever
    assert twice_rebuilt_retriever is not rebuilt_retriever
    assert built_for == [previous_collection, first_reset_collection, runtime.collection]


def test_reset_empty_recovers_when_collection_was_already_deleted():
    client = FakeClient()
    built_for: list[FakeCollection] = []

    def builder(collection):
        built_for.append(collection)
        return object()

    runtime = RetrieverRuntime(
        client=client,
        collection_name="study_coach_chunks",
        builder=builder,
    )
    client.delete_collection("study_coach_chunks")

    runtime.reset_empty()

    assert runtime.collection is client.collections["study_coach_chunks"]
    assert len(built_for) == 2


def test_reset_empty_propagates_non_not_found_delete_errors():
    class FailingDeleteClient(FakeClient):
        def delete_collection(self, name: str) -> None:
            raise RuntimeError("storage unavailable")

    runtime = RetrieverRuntime(
        client=FailingDeleteClient(),
        collection_name="study_coach_chunks",
        builder=lambda collection: object(),
    )

    with pytest.raises(RuntimeError, match="storage unavailable"):
        runtime.reset_empty()


def test_reset_empty_keeps_retry_reachable_when_builder_fails():
    client = FakeClient()
    build_count = 0

    def builder(collection):
        nonlocal build_count
        build_count += 1
        if build_count == 2:
            raise RuntimeError("reranker unavailable")
        return object()

    runtime = RetrieverRuntime(
        client=client,
        collection_name="study_coach_chunks",
        builder=builder,
    )
    previous_collection = runtime.collection
    initial_retriever = runtime.retriever

    with pytest.raises(RuntimeError, match="reranker unavailable"):
        runtime.reset_empty()

    with pytest.raises(NotFoundError):
        previous_collection.count()
    assert runtime.collection is not previous_collection
    assert runtime.vector_count() == 0
    assert runtime.retriever is initial_retriever

    rebuilt_retriever = runtime.reset_empty()

    assert runtime.vector_count() == 0
    assert runtime.retriever is rebuilt_retriever
    assert rebuilt_retriever is not initial_retriever


def test_reset_empty_retries_after_recreate_temporarily_fails():
    class RecreateFailsOnceClient(FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self._remaining_recreate_failures = 1

        def get_or_create_collection(self, name: str) -> FakeCollection:
            if self._created_once and self._remaining_recreate_failures:
                self._remaining_recreate_failures -= 1
                raise RuntimeError("storage temporarily unavailable")
            return super().get_or_create_collection(name)

    client = RecreateFailsOnceClient()
    runtime = RetrieverRuntime(
        client=client,
        collection_name="study_coach_chunks",
        builder=lambda collection: object(),
    )
    previous_collection = runtime.collection
    initial_retriever = runtime.retriever

    with pytest.raises(RuntimeError, match="storage temporarily unavailable"):
        runtime.reset_empty()

    with pytest.raises(NotFoundError):
        previous_collection.count()
    assert runtime.vector_count() == 0
    assert runtime.retriever is initial_retriever

    rebuilt_retriever = runtime.reset_empty()

    assert runtime.vector_count() == 0
    assert runtime.retriever is rebuilt_retriever
    assert rebuilt_retriever is not initial_retriever


def test_reset_empty_retries_after_real_chroma_builder_failure():
    client = chromadb.EphemeralClient()
    build_count = 0

    def builder(collection):
        nonlocal build_count
        build_count += 1
        if build_count == 2:
            raise RuntimeError("reranker unavailable")
        return object()

    runtime = RetrieverRuntime(
        client=client,
        collection_name="runtime_retry_recovery",
        builder=builder,
    )
    previous_collection = runtime.collection
    initial_retriever = runtime.retriever

    with pytest.raises(RuntimeError, match="reranker unavailable"):
        runtime.reset_empty()

    with pytest.raises(NotFoundError):
        previous_collection.count()
    assert runtime.vector_count() == 0
    assert runtime.retriever is initial_retriever

    rebuilt_retriever = runtime.reset_empty()

    assert runtime.vector_count() == 0
    assert runtime.retriever is rebuilt_retriever
    assert rebuilt_retriever is not initial_retriever


def test_default_retriever_builder_hydrates_bm25_from_existing_collection(
    monkeypatch,
):
    collection = FakeCollection(
        "study_coach_chunks",
        count=2,
        documents=["BM25 is lexical", "Dense retrieval uses embeddings"],
        metadatas=[
            {"source": "a.pdf", "page": 1},
            {"source": "b.pdf", "page": 2},
        ],
    )
    captured: dict[str, object] = {}

    class FakeDense:
        def __init__(self, *, collection, embedder) -> None:
            captured["dense_collection"] = collection
            captured["embedder"] = embedder

    class FakeBM25:
        def add_chunks(self, chunks) -> None:
            captured["bm25_chunks"] = chunks

    class FakeHybrid:
        def __init__(self, *, dense, bm25) -> None:
            captured["hybrid"] = (dense, bm25)

    class FakeReranking:
        def __init__(self, *, base, reranker, retrieval_depth) -> None:
            captured["reranking"] = (base, reranker, retrieval_depth)

    monkeypatch.setattr(
        runtime_module,
        "OllamaEmbedder",
        lambda *, model, base_url: (model, base_url),
    )
    monkeypatch.setattr(runtime_module, "Retriever", FakeDense)
    monkeypatch.setattr(runtime_module, "BM25Index", FakeBM25)
    monkeypatch.setattr(runtime_module, "HybridRetriever", FakeHybrid)
    monkeypatch.setattr(runtime_module, "FastembedReranker", lambda: "reranker")
    monkeypatch.setattr(runtime_module, "RerankingRetriever", FakeReranking)

    runtime_module.build_retriever_for_collection(
        collection,
        embed_model="test-model",
        embed_host="http://ollama.test",
    )

    assert captured["dense_collection"] is collection
    assert captured["embedder"] == ("test-model", "http://ollama.test")
    assert captured["bm25_chunks"] == [
        {"chunk_id": "chunk-0", "content": "BM25 is lexical", "source": "a.pdf", "page": 1},
        {"chunk_id": "chunk-1", "content": "Dense retrieval uses embeddings", "source": "b.pdf", "page": 2},
    ]
    assert captured["reranking"][2] == 20


def test_build_default_runtime_uses_configured_chroma_path_and_fixed_collection(
    monkeypatch,
):
    client = FakeClient()
    captured: dict[str, object] = {}

    def persistent_client(*, path):
        captured["path"] = path
        return client

    def build_retriever(collection, *, embed_model, embed_host):
        captured["build_args"] = (collection, embed_model, embed_host)
        return object()

    monkeypatch.setenv("CHROMA_PATH", "/tmp/study-coach-chroma")
    monkeypatch.setenv("EMBED_MODEL", "test-model")
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama.test")
    monkeypatch.setattr(runtime_module.chromadb, "PersistentClient", persistent_client)
    monkeypatch.setattr(runtime_module, "build_retriever_for_collection", build_retriever)

    runtime = runtime_module.build_default_runtime()

    assert captured["path"] == "/tmp/study-coach-chroma"
    assert client.created_names == ["study_coach_chunks"]
    assert captured["build_args"] == (
        runtime.collection,
        "test-model",
        "http://ollama.test",
    )


def test_main_retriever_factory_returns_runtime_retriever(monkeypatch):
    from app import main

    runtime = type("Runtime", (), {"retriever": object()})()
    monkeypatch.setattr(main, "build_default_runtime", lambda: runtime)

    assert main._build_default_retriever() is runtime.retriever


def test_create_app_keeps_runtime_and_retriever_in_app_state(monkeypatch):
    from app import main
    from app.db import session

    runtime = type("Runtime", (), {"retriever": object()})()
    monkeypatch.delenv("STUDY_COACH_TEST_MODE", raising=False)
    monkeypatch.setattr(main, "build_default_runtime", lambda: runtime)
    monkeypatch.setattr(session, "migrate_to_head", lambda: None)

    app = main.create_app()

    assert app.state.retriever_runtime is runtime
    assert app.state.retriever is runtime.retriever
