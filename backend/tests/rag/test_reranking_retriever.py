from app.rag.reranking_retriever import RerankingRetriever


class StubBaseRetriever:
    def __init__(self, return_chunks: list[dict]):
        self._returns = return_chunks
        self.last_top_k: int | None = None

    def add_chunks(self, chunks):
        pass

    def search(self, query, top_k):
        self.last_top_k = top_k
        return self._returns[:top_k]

    def close(self):
        self.closed = getattr(self, "closed", 0) + 1


class StubReranker:
    def __init__(self, scores: list[float]):
        self._scores = scores
        self.last_query: str | None = None
        self.last_documents: list[str] | None = None

    def score(self, query, documents):
        self.last_query = query
        self.last_documents = list(documents)
        return self._scores[: len(documents)]

    def close(self):
        self.closed = getattr(self, "closed", 0) + 1


def test_reranking_retriever_reorders_candidates_by_reranker_score():
    chunks = [
        {"chunk_id": "A", "content": "doc A", "source": "a.pdf", "page": 1},
        {"chunk_id": "B", "content": "doc B", "source": "b.pdf", "page": 1},
        {"chunk_id": "C", "content": "doc C", "source": "c.pdf", "page": 1},
    ]
    base = StubBaseRetriever(return_chunks=chunks)
    reranker = StubReranker(scores=[0.1, 0.9, 0.5])  # B > C > A
    rr = RerankingRetriever(base=base, reranker=reranker, retrieval_depth=10)

    results = rr.search("query", top_k=3)

    assert [r["chunk_id"] for r in results] == ["B", "C", "A"]
    assert results[0]["score"] == 0.9


def test_reranking_retriever_truncates_to_top_k():
    chunks = [
        {"chunk_id": f"c{i}", "content": f"doc{i}", "source": "x.pdf", "page": 1}
        for i in range(10)
    ]
    base = StubBaseRetriever(return_chunks=chunks)
    reranker = StubReranker(scores=[float(i) for i in range(10)])  # c9 highest
    rr = RerankingRetriever(base=base, reranker=reranker, retrieval_depth=10)

    results = rr.search("query", top_k=3)

    assert [r["chunk_id"] for r in results] == ["c9", "c8", "c7"]


def test_reranking_retriever_returns_empty_when_base_returns_empty():
    base = StubBaseRetriever(return_chunks=[])
    reranker = StubReranker(scores=[])
    rr = RerankingRetriever(base=base, reranker=reranker)

    results = rr.search("query", top_k=5)

    assert results == []


def test_reranking_retriever_calls_base_with_retrieval_depth_not_top_k():
    chunks = [{"chunk_id": "A", "content": "doc A", "source": "a.pdf", "page": 1}]
    base = StubBaseRetriever(return_chunks=chunks)
    reranker = StubReranker(scores=[0.5])
    rr = RerankingRetriever(base=base, reranker=reranker, retrieval_depth=20)

    rr.search("query", top_k=5)

    assert base.last_top_k == 20


def test_reranking_retriever_close_closes_base_and_reranker_once():
    base = StubBaseRetriever(return_chunks=[])
    reranker = StubReranker(scores=[])
    rr = RerankingRetriever(base=base, reranker=reranker)

    rr.close()
    rr.close()

    assert base.closed == 1
    assert reranker.closed == 1
