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
