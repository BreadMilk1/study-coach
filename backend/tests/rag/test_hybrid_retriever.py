from app.rag.hybrid_retriever import (
    BM25Index,
    HybridRetriever,
    reciprocal_rank_fusion,
)


class StubDenseRetriever:
    """Returns chunks in a fixed order, ignoring query.

    Lets us assert hybrid behavior without depending on the WordBagEmbedder's
    accidental lexical semantics (which would have BM25 and Dense agreeing).
    """

    def __init__(self, ordered_ids: list[str]):
        self._ordered_ids = ordered_ids
        self._chunks: dict[str, dict] = {}

    def add_chunks(self, chunks):
        for c in chunks:
            self._chunks[c["chunk_id"]] = c

    def search(self, query, top_k=5):
        return [
            {**self._chunks[cid], "score": 1.0 - i * 0.1}
            for i, cid in enumerate(self._ordered_ids[:top_k])
            if cid in self._chunks
        ]


def test_bm25_index_retrieves_chunk_with_exact_keyword():
    bm25 = BM25Index()
    bm25.add_chunks([
        {"chunk_id": "a:1:0", "content": "Prompt engineering techniques",
         "source": "a.pdf", "page": 1},
        {"chunk_id": "b:1:0", "content": "ReAct framework for tool use agents",
         "source": "b.pdf", "page": 1},
        {"chunk_id": "c:1:0", "content": "Neural network backpropagation",
         "source": "c.pdf", "page": 1},
    ])

    results = bm25.search("ReAct", top_k=2)

    assert results
    assert results[0]["chunk_id"] == "b:1:0"
    assert results[0]["source"] == "b.pdf"
    assert results[0]["score"] > 0


def test_bm25_index_returns_empty_when_no_match():
    bm25 = BM25Index()
    bm25.add_chunks([
        {"chunk_id": "a:1:0", "content": "Prompt engineering",
         "source": "a.pdf", "page": 1},
    ])

    results = bm25.search("xyzqwerty", top_k=5)

    assert results == []


def test_bm25_index_add_chunks_updates_source_without_growing():
    bm25 = BM25Index()
    chunk = {
        "chunk_id": "hash:1:0",
        "content": "HyDE rewrites queries before embedding",
        "source": "a.pdf",
        "page": 1,
    }
    other = {
        "chunk_id": "other:1:0",
        "content": "Unrelated spaced repetition schedule",
        "source": "c.pdf",
        "page": 1,
    }
    bm25.add_chunks([chunk, other])
    bm25.add_chunks([{**chunk, "source": "b.pdf"}])

    assert [c["chunk_id"] for c in bm25._chunks] == ["hash:1:0", "other:1:0"]
    assert bm25._chunks[0]["source"] == "b.pdf"
    assert bm25._chunks[1]["source"] == "c.pdf"


def test_bm25_concurrent_add_chunks_does_not_duplicate_under_lock():
    """ControllableLock proves writer/writer and writer/search serialize on real RLock."""
    import threading
    import time

    class ControllableLock:
        """Test-only wrapper: real RLock with deterministic hold/blocked signals."""

        def __init__(self) -> None:
            self._lock = threading.RLock()
            self.hold_after_acquire = False
            self.acquired = threading.Event()
            self.allow_continue = threading.Event()
            self.blocked = threading.Event()

        def __enter__(self):
            if not self._lock.acquire(blocking=False):
                self.blocked.set()
                self._lock.acquire(blocking=True)
            if self.hold_after_acquire:
                self.hold_after_acquire = False
                self.acquired.set()
                assert self.allow_continue.wait(timeout=5)
            return self

        def __exit__(self, exc_type, exc, tb):
            self._lock.release()
            return False

    chunks = [
        {
            "chunk_id": f"id-{i}",
            "content": f"unique keyword{i} retrieval content",
            "source": "a.pdf",
            "page": i,
        }
        for i in range(20)
    ]

    bm25 = BM25Index()
    lock = ControllableLock()
    bm25._lock = lock

    lock.hold_after_acquire = True
    lock.allow_continue.clear()
    lock.acquired.clear()
    lock.blocked.clear()

    writer_a_done = threading.Event()
    writer_b_done = threading.Event()

    def writer_a() -> None:
        bm25.add_chunks(chunks)
        writer_a_done.set()

    def writer_b() -> None:
        assert lock.acquired.wait(timeout=5)
        bm25.add_chunks(chunks)
        writer_b_done.set()

    thread_a = threading.Thread(target=writer_a)
    thread_b = threading.Thread(target=writer_b)
    thread_a.start()
    assert lock.acquired.wait(timeout=5)
    thread_b.start()
    assert lock.blocked.wait(timeout=5)
    time.sleep(0.05)
    assert not writer_b_done.is_set()
    lock.allow_continue.set()
    thread_a.join(timeout=5)
    thread_b.join(timeout=5)
    assert writer_a_done.is_set() and writer_b_done.is_set()
    assert len(bm25._chunks) == 20
    assert len(bm25._tokenized) == 20
    assert len({c["chunk_id"] for c in bm25._chunks}) == 20

    # Writer holds the lock while search contends and must wait.
    gate = BM25Index()
    gate_lock = ControllableLock()
    gate._lock = gate_lock
    gate.add_chunks(chunks[:10])

    gate_lock.hold_after_acquire = True
    gate_lock.allow_continue.clear()
    gate_lock.acquired.clear()
    gate_lock.blocked.clear()
    search_done = threading.Event()

    def writer() -> None:
        gate.add_chunks([chunks[10]])

    def reader() -> None:
        assert gate_lock.acquired.wait(timeout=5)
        gate.search("keyword0")
        search_done.set()

    writer_thread = threading.Thread(target=writer)
    reader_thread = threading.Thread(target=reader)
    writer_thread.start()
    assert gate_lock.acquired.wait(timeout=5)
    reader_thread.start()
    assert gate_lock.blocked.wait(timeout=5)
    time.sleep(0.05)
    assert not search_done.is_set()
    gate_lock.allow_continue.set()
    writer_thread.join(timeout=5)
    reader_thread.join(timeout=5)
    assert search_done.is_set()
    assert len(gate._chunks) == 11
    assert len(gate._tokenized) == 11
    assert len({c["chunk_id"] for c in gate._chunks}) == 11
    hit = gate.search("keyword0", top_k=5)
    assert hit and hit[0]["chunk_id"] == "id-0"
    assert gate._bm25 is not None
    assert len(gate._bm25.get_scores(["keyword0"])) == len(gate._chunks)


def test_reciprocal_rank_fusion_weights_chunks_in_multiple_lists_higher():
    fused = reciprocal_rank_fusion([
        ["a", "b"],
        ["c", "a", "b"],
    ])

    # a: rank 1 in list1, rank 2 in list2 -> 1/61 + 1/62
    # b: rank 2 in list1, rank 3 in list2 -> 1/62 + 1/63
    # c: rank 1 in list2 only             -> 1/61
    assert fused["a"] > fused["b"]
    assert fused["b"] > fused["c"]


def test_hybrid_retriever_promotes_chunk_via_bm25_when_dense_misses_it():
    chunks = [
        {"chunk_id": "A", "content": "common content one",
         "source": "A.pdf", "page": 1},
        {"chunk_id": "B", "content": "ReAct is a tool use agent framework",
         "source": "B.pdf", "page": 1},
        {"chunk_id": "C", "content": "common content two",
         "source": "C.pdf", "page": 1},
        {"chunk_id": "D", "content": "common content three",
         "source": "D.pdf", "page": 1},
    ]
    dense = StubDenseRetriever(ordered_ids=["A", "C", "D"])
    hybrid = HybridRetriever(dense=dense)
    hybrid.add_chunks(chunks)

    results = hybrid.search("ReAct", top_k=5, retrieval_depth=10)
    result_ids = [r["chunk_id"] for r in results]

    assert "B" in result_ids, (
        "Hybrid must surface chunk B (BM25-only hit) when Dense misses it"
    )


def test_hybrid_retriever_returns_chunks_with_required_keys():
    chunks = [
        {"chunk_id": "a:1:0", "content": "HyDE rewrites the query",
         "source": "a.pdf", "page": 1},
        {"chunk_id": "b:1:0", "content": "BM25 is lexical retrieval",
         "source": "b.pdf", "page": 2},
    ]
    dense = StubDenseRetriever(ordered_ids=["a:1:0", "b:1:0"])
    hybrid = HybridRetriever(dense=dense)
    hybrid.add_chunks(chunks)

    results = hybrid.search("HyDE", top_k=2)

    assert results
    for r in results:
        for key in ("chunk_id", "content", "source", "page", "score"):
            assert key in r, f"missing key {key} in {r}"
