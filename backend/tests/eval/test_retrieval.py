from app.eval.retrieval import EvalQuery, evaluate, hit_rate_at_k, mrr


def test_hit_rate_at_k_returns_one_when_expected_chunk_in_top_k():
    returned = [
        {"source": "X.pdf", "page": 1},
        {"source": "Y.pdf", "page": 5},
        {"source": "Z.pdf", "page": 2},
    ]
    expected = [{"source": "Y.pdf", "pages": [5, 6]}]

    assert hit_rate_at_k(returned, expected, k=3) == 1.0
    assert hit_rate_at_k(returned, expected, k=1) == 0.0


def test_hit_rate_at_k_returns_zero_when_no_chunk_matches_expected():
    returned = [
        {"source": "X.pdf", "page": 1},
        {"source": "W.pdf", "page": 9},
    ]
    expected = [{"source": "Y.pdf", "pages": [5]}]

    assert hit_rate_at_k(returned, expected, k=5) == 0.0


def test_mrr_returns_reciprocal_of_first_hit_rank():
    returned = [
        {"source": "X.pdf", "page": 1},
        {"source": "Y.pdf", "page": 5},
        {"source": "Z.pdf", "page": 2},
    ]
    expected = [{"source": "Y.pdf", "pages": [5]}]

    assert mrr(returned, expected) == 0.5


def test_mrr_returns_zero_when_no_hit():
    returned = [{"source": "X.pdf", "page": 1}]
    expected = [{"source": "Y.pdf", "pages": [5]}]

    assert mrr(returned, expected) == 0.0


def test_evaluate_aggregates_hit_rate_and_mrr_across_queries():
    class StubRetriever:
        def __init__(self, results_by_query):
            self._results = results_by_query

        def search(self, query, top_k=10):
            return self._results[query]

    queries = [
        EvalQuery(
            query="HyDE",
            expected=[{"source": "Topic7.pdf", "pages": [3]}],
            classification="short_kw",
        ),
        EvalQuery(
            query="ReAct",
            expected=[{"source": "Topic9.pdf", "pages": [10]}],
            classification="short_kw",
        ),
    ]

    retriever = StubRetriever({
        "HyDE": [
            {"source": "Topic7.pdf", "page": 3, "chunk_id": "a"},
            {"source": "X.pdf", "page": 1, "chunk_id": "b"},
        ],
        "ReAct": [
            {"source": "Y.pdf", "page": 1, "chunk_id": "c"},
            {"source": "Z.pdf", "page": 2, "chunk_id": "d"},
        ],
    })

    report = evaluate(retriever, queries, top_k=10)

    assert report.aggregate["hit_rate@5"] == 0.5
    assert report.aggregate["mrr"] == 0.5
    assert len(report.per_query) == 2
    assert report.per_query[0]["hit_rank"] == 1
    assert report.per_query[1]["hit_rank"] is None
