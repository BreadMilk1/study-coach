"""Owner-scoped retrieval isolation at the retriever layer.

Uses a real in-memory Chroma collection plus the deterministic test embedder
and a stub cross-encoder reranker so no network or provider is touched.
"""

import chromadb
import pytest

from app.rag.hybrid_retriever import BM25Index, HybridRetriever
from app.rag.reranking_retriever import RerankingRetriever
from app.rag.retriever import Retriever
from app.rag.scope import (
    RetrievalScope,
    ScopedRetriever,
    UnscopedRetrieverError,
    authorized_get_chunk,
    build_v2_chunk_id,
    parse_legacy_chunk_id,
    parse_v2_chunk_id,
)


class StubReranker:
    def __init__(self):
        self.seen_documents: list[str] = []

    def score(self, query, documents):
        self.seen_documents = list(documents)
        return [1.0 - 0.01 * i for i in range(len(documents))]


class DictBase:
    """Minimal chunk store for the authorized readback unit tests."""

    def __init__(self, chunks):
        self._by_id = {chunk["chunk_id"]: chunk for chunk in chunks}

    def add_chunks(self, chunks):
        for chunk in chunks:
            self._by_id[chunk["chunk_id"]] = chunk

    def search(self, query, top_k=5):
        return []

    def get_chunk(self, chunk_id):
        return self._by_id.get(chunk_id)


def _collection(name: str):
    client = chromadb.Client()
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return client.create_collection(name)


def _build_chain(collection, embedder, reranker):
    dense = Retriever(collection=collection, embedder=embedder)
    hybrid = HybridRetriever(dense=dense, bm25=BM25Index())
    return RerankingRetriever(base=hybrid, reranker=reranker, retrieval_depth=20)


def _scoped_chunk(owner, hash_, doc_id, content, page=1, index=0, source="a.pdf"):
    return {
        "chunk_id": build_v2_chunk_id(owner=owner, hash_=hash_, page=page, index=index),
        "content": content,
        "source": source,
        "page": page,
        "owner": owner,
        "document_id": doc_id,
        "hash": hash_,
    }


def test_scope_excludes_ownerless_and_other_owner_before_rrf_and_rerank(fake_embedder):
    collection = _collection("scope_exclude")
    reranker = StubReranker()
    chain = _build_chain(collection, fake_embedder, reranker)
    a_chunk = _scoped_chunk("A", "a" * 64, "doc-a", "alpha unique keyword A")
    b_chunk = _scoped_chunk("B", "b" * 64, "doc-b", "alpha unique keyword B")
    orphan = _scoped_chunk("A", "c" * 64, "doc-orphan", "alpha unique keyword orphan")
    legacy_id = "d" * 64 + ":1:0"
    legacy = {"chunk_id": legacy_id, "content": "alpha unique keyword legacy",
              "source": "old.pdf", "page": 1}
    chain.add_chunks([a_chunk, b_chunk, orphan, legacy])

    scope_a = RetrievalScope(owner="A", documents={"a" * 64: "doc-a"})

    dense_only = chain.base.dense.search("alpha unique keyword", top_k=10, scope=scope_a)
    assert [c["chunk_id"] for c in dense_only] == [a_chunk["chunk_id"]]

    results = chain.search_scoped("alpha unique keyword", top_k=10, scope=scope_a)
    assert [c["chunk_id"] for c in results] == [a_chunk["chunk_id"]]
    # Only the filtered candidate reached the cross-encoder.
    assert reranker.seen_documents == ["alpha unique keyword A"]

    # Unscoped search still sees every chunk, proving the scope did the filter.
    unscoped = chain.search("alpha unique keyword", top_k=10)
    assert {c["chunk_id"] for c in unscoped} == {
        a_chunk["chunk_id"],
        b_chunk["chunk_id"],
        orphan["chunk_id"],
        legacy_id,
    }


def test_other_owner_high_bm25_score_does_not_crowd_owner():
    # Five docs so the shared query terms keep a positive BM25 IDF (a term in
    # every document is filtered out by BM25's negative-IDF fallback).
    a_chunk = _scoped_chunk("A", "a" * 64, "doc-a", "react agent")
    b_chunk = _scoped_chunk("B", "b" * 64, "doc-b", "react react react agent agent agent")
    fillers = [
        _scoped_chunk("A", f"{i:064x}", f"doc-f{i}", f"filler{i} unrelated content",
                      index=i)
        for i in range(1, 4)
    ]
    bm25 = BM25Index()
    bm25.add_chunks([a_chunk, b_chunk, *fillers])

    scope_a = RetrievalScope(owner="A", documents={"a" * 64: "doc-a"})
    assert [c["chunk_id"] for c in bm25.search("react agent", top_k=1)] == [
        b_chunk["chunk_id"]
    ]
    assert [c["chunk_id"] for c in bm25.search("react agent", top_k=1, scope=scope_a)] == [
        a_chunk["chunk_id"]
    ]


def test_missing_owner_metadata_fails_closed(fake_embedder):
    collection = _collection("scope_missing")
    chain = _build_chain(collection, fake_embedder, StubReranker())
    legacy_id = "e" * 64 + ":1:0"
    ownerless = {"chunk_id": legacy_id, "content": "react agent framework",
                 "source": "x.pdf", "page": 1}
    fillers = [
        {"chunk_id": f"filler-{i}", "content": f"filler{i} unrelated content",
         "source": "x.pdf", "page": i + 1}
        for i in range(4)
    ]
    chain.add_chunks([ownerless, *fillers])

    # Even when the caller owns the hash in SQL, an ownerless stored chunk is
    # never returned by production search — dense, BM25, or hybrid.
    scope = RetrievalScope(owner="A", documents={"e" * 64: "doc-e"})
    assert chain.base.dense.search("react agent", top_k=5, scope=scope) == []
    assert chain.base.bm25.search("react agent", top_k=5, scope=scope) == []
    assert chain.search_scoped("react agent", top_k=5, scope=scope) == []
    # Unscoped BM25 does find the ownerless chunk, so the scope is the filter.
    assert [c["chunk_id"] for c in chain.base.bm25.search("react agent", top_k=5)] == [
        legacy_id
    ]


def test_restart_hydration_preserves_owner_metadata(monkeypatch, fake_embedder):
    from app.rag import runtime as runtime_module

    collection = _collection("scope_hydrate")
    dense = Retriever(collection=collection, embedder=fake_embedder)
    seeded = HybridRetriever(dense=dense, bm25=BM25Index())
    a_id = build_v2_chunk_id(owner="A", hash_="a" * 64, page=1, index=0)
    legacy_id = "d" * 64 + ":1:0"
    seeded.add_chunks([
        {"chunk_id": a_id, "content": "owner A HyDE chunk", "source": "a.pdf",
         "page": 1, "owner": "A", "document_id": "doc-a", "hash": "a" * 64},
        {"chunk_id": legacy_id, "content": "ownerless legacy chunk",
         "source": "old.pdf", "page": 1},
    ])

    monkeypatch.setattr(
        runtime_module, "OllamaEmbedder", lambda *, model, base_url: fake_embedder
    )
    monkeypatch.setattr(runtime_module, "FastembedReranker", lambda: StubReranker())

    rebuilt = runtime_module.build_retriever_for_collection(collection, embed_model="fake")
    by_id = {c["chunk_id"]: c for c in rebuilt.base.bm25._chunks}
    assert by_id[a_id]["owner"] == "A"
    assert by_id[a_id]["document_id"] == "doc-a"
    assert by_id[a_id]["hash"] == "a" * 64
    assert "owner" not in by_id[legacy_id]

    scope_a = RetrievalScope(owner="A", documents={"a" * 64: "doc-a"})
    assert [c["chunk_id"] for c in rebuilt.search_scoped("HyDE", top_k=5, scope=scope_a)] == [a_id]


def test_v2_same_hash_other_owner_not_readable():
    hash_ = "f" * 64
    a_id = build_v2_chunk_id(owner="A", hash_=hash_, page=1, index=0)
    b_id = build_v2_chunk_id(owner="B", hash_=hash_, page=1, index=0)
    base = DictBase([
        {"chunk_id": a_id, "content": "A secret", "source": "a.pdf", "page": 1,
         "owner": "A", "document_id": "doc-a", "hash": hash_},
        {"chunk_id": b_id, "content": "B secret", "source": "b.pdf", "page": 1,
         "owner": "B", "document_id": "doc-b", "hash": hash_},
    ])
    scope_a = RetrievalScope(owner="A", documents={hash_: "doc-a"})
    assert authorized_get_chunk(base, a_id, scope=scope_a)["content"] == "A secret"
    # A cannot read B's same-hash v2 chunk.
    assert authorized_get_chunk(base, b_id, scope=scope_a) is None
    # A's own id with a document id that is not A's SQL document is refused.
    mismatched = RetrievalScope(owner="A", documents={hash_: "some-other-doc"})
    assert authorized_get_chunk(base, a_id, scope=mismatched) is None


def test_legacy_ownerless_readable_only_with_sql_ownership():
    hash_ = "1" * 64
    legacy_id = f"{hash_}:1:0"
    base = DictBase([
        {"chunk_id": legacy_id, "content": "old", "source": "o.pdf", "page": 1},
    ])
    owner_scope = RetrievalScope(owner="A", documents={hash_: "doc-a"})
    assert authorized_get_chunk(base, legacy_id, scope=owner_scope)["content"] == "old"
    assert authorized_get_chunk(base, legacy_id, scope=RetrievalScope(owner="A", documents={})) is None
    # A non-strict id (not 64hex:page:index) is refused.
    assert authorized_get_chunk(base, f"{hash_}:1", scope=owner_scope) is None
    # A legacy id whose stored metadata carries an owner is not a legacy read.
    owned = DictBase([
        {"chunk_id": legacy_id, "content": "x", "source": "o.pdf", "page": 1, "owner": "A"},
    ])
    assert authorized_get_chunk(owned, legacy_id, scope=owner_scope) is None
    # A 63-hex id is not a strict legacy id even when the caller owns the hash.
    assert authorized_get_chunk(base, f"{hash_[:-1]}:1:0", scope=owner_scope) is None


def test_scoped_search_requires_search_scoped_on_base():
    # A production base without a scoped entry point fails closed, never falls
    # back to an unscoped search.
    scoped = ScopedRetriever(
        base=DictBase([]), scope=RetrievalScope(owner="A", documents={})
    )
    with pytest.raises(UnscopedRetrieverError):
        scoped.search("anything", top_k=5)


def test_scope_rejects_empty_or_colon_owner():
    for bad in ("", ":", "owner:with:colon"):
        with pytest.raises(ValueError):
            RetrievalScope(owner=bad, documents={})


def test_scope_documents_is_read_only():
    scope = RetrievalScope(owner="A", documents={"a" * 64: "doc-a"})
    with pytest.raises(TypeError):
        scope.documents["b" * 64] = "doc-b"  # type: ignore[index]


def test_v2_wrong_document_id_rejected():
    hash_ = "a" * 64
    chunk_id = build_v2_chunk_id(owner="A", hash_=hash_, page=1, index=0)
    chunk = {"chunk_id": chunk_id, "content": "x", "source": "a.pdf", "page": 1,
             "owner": "A", "document_id": "wrong-doc", "hash": hash_}
    scope = RetrievalScope(owner="A", documents={hash_: "doc-a"})
    assert scope.allows(chunk) is False
    assert authorized_get_chunk(DictBase([chunk]), chunk_id, scope=scope) is None


def test_meta_and_id_mismatch_rejected():
    hash_ = "a" * 64
    chunk_id = build_v2_chunk_id(owner="A", hash_=hash_, page=1, index=0)
    scope = RetrievalScope(owner="A", documents={hash_: "doc-a"})
    # Metadata hash disagrees with the id hash.
    assert scope.allows({
        "chunk_id": chunk_id, "content": "x", "source": "a.pdf", "page": 1,
        "owner": "A", "document_id": "doc-a", "hash": "b" * 64,
    }) is False
    # Metadata owner disagrees with the id owner.
    assert scope.allows({
        "chunk_id": chunk_id, "content": "x", "source": "a.pdf", "page": 1,
        "owner": "B", "document_id": "doc-a", "hash": hash_,
    }) is False
    # Missing document id fails closed.
    assert scope.allows({
        "chunk_id": chunk_id, "content": "x", "source": "a.pdf", "page": 1,
        "owner": "A", "hash": hash_,
    }) is False


def test_malicious_trailing_newline_id_rejected():
    hash_ = "a" * 64
    chunk_id = build_v2_chunk_id(owner="A", hash_=hash_, page=1, index=0)
    scope = RetrievalScope(owner="A", documents={hash_: "doc-a"})
    newline_id = chunk_id + "\n"
    assert parse_v2_chunk_id(newline_id) is None
    assert parse_legacy_chunk_id(f"{hash_}:1:0\n") is None
    assert scope.allows({"chunk_id": newline_id, "content": "x", "source": "a.pdf",
                         "page": 1, "owner": "A", "document_id": "doc-a",
                         "hash": hash_}) is False
    # A lenient store that resolves the newline id still refuses to authorize it.
    base = DictBase([{"chunk_id": newline_id, "content": "x", "source": "a.pdf",
                      "page": 1, "owner": "A", "document_id": "doc-a", "hash": hash_}])
    assert authorized_get_chunk(base, newline_id, scope=scope) is None


def test_build_v2_chunk_id_validates_arguments():
    with pytest.raises(ValueError):
        build_v2_chunk_id(owner="", hash_="a" * 64, page=1, index=0)
    with pytest.raises(ValueError):
        build_v2_chunk_id(owner="a:b", hash_="a" * 64, page=1, index=0)
    with pytest.raises(ValueError):
        build_v2_chunk_id(owner="A", hash_="not-a-hash", page=1, index=0)
    with pytest.raises(ValueError):
        build_v2_chunk_id(owner="A", hash_="a" * 64, page=-2, index=0)
    with pytest.raises(ValueError):
        build_v2_chunk_id(owner="A", hash_="a" * 64, page=1, index=-1)


def test_same_owner_orphan_does_not_crowd_dense_topk(fake_embedder):
    collection = _collection("scope_orphan_dense")
    chain = _build_chain(collection, fake_embedder, StubReranker())
    hash_a = "a" * 64
    legit = _scoped_chunk("A", hash_a, "doc-a", "hyde hypothetical")
    # Same owner but no document id (index-before-SQL orphan), stronger match.
    orphan = {"chunk_id": build_v2_chunk_id(owner="A", hash_="c" * 64, page=1, index=0),
              "content": "hyde hypothetical hyde hypothetical", "source": "a.pdf",
              "page": 1, "owner": "A", "hash": "c" * 64}
    # Same owner but a different document id.
    wrong = {"chunk_id": build_v2_chunk_id(owner="A", hash_="d" * 64, page=1, index=0),
             "content": "hyde hypothetical hyde hypothetical hyde", "source": "a.pdf",
             "page": 1, "owner": "A", "document_id": "not-mine", "hash": "d" * 64}
    chain.add_chunks([legit, orphan, wrong])

    scope_a = RetrievalScope(owner="A", documents={hash_a: "doc-a"})
    assert [c["chunk_id"] for c in chain.base.dense.search("hyde hypothetical", top_k=1, scope=scope_a)] == [
        legit["chunk_id"]
    ]
    assert [c["chunk_id"] for c in chain.search_scoped("hyde hypothetical", top_k=1, scope=scope_a)] == [
        legit["chunk_id"]
    ]


def test_same_owner_orphan_does_not_crowd_bm25_topk():
    hash_a = "a" * 64
    legit = _scoped_chunk("A", hash_a, "doc-a", "hyde hypothetical")
    orphan = {"chunk_id": build_v2_chunk_id(owner="A", hash_="c" * 64, page=1, index=0),
              "content": "hyde hyde hyde hypothetical hypothetical hypothetical",
              "source": "a.pdf", "page": 1, "owner": "A", "hash": "c" * 64}
    wrong = {"chunk_id": build_v2_chunk_id(owner="A", hash_="d" * 64, page=1, index=0),
             "content": "hyde hyde hyde hyde hypothetical hypothetical hypothetical hypothetical",
             "source": "a.pdf", "page": 1, "owner": "A", "document_id": "not-mine",
             "hash": "d" * 64}
    fillers = [
        {"chunk_id": f"filler-{i}", "content": f"filler{i} unrelated content",
         "source": "f.pdf", "page": i + 1}
        for i in range(4)
    ]
    bm25 = BM25Index()
    bm25.add_chunks([legit, orphan, wrong, *fillers])

    scope_a = RetrievalScope(owner="A", documents={hash_a: "doc-a"})
    # Unscoped ranking is crowded by the stronger same-owner orphans...
    assert bm25.search("hyde hypothetical", top_k=1)[0]["chunk_id"] in {
        orphan["chunk_id"], wrong["chunk_id"],
    }
    # ...but the scoped ranking returns only the SQL-authorized chunk.
    assert [c["chunk_id"] for c in bm25.search("hyde hypothetical", top_k=1, scope=scope_a)] == [
        legit["chunk_id"]
    ]
