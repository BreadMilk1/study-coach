from app.eval.p2_3_cloud_capability.corpus import (
    RecordingRetriever,
    retriever_config_hash,
    snapshot_hash,
)


def test_snapshot_hash_sorts_by_chunk_id_and_uses_four_fields():
    chunks = [
        {"chunk_id": "b", "content": "B", "source": "s", "page": 2, "score": 0.9},
        {"chunk_id": "a", "content": "A", "source": "s", "page": 1, "score": 0.1},
    ]
    h1 = snapshot_hash(chunks)
    h2 = snapshot_hash([
        {"chunk_id": "a", "content": "A", "source": "s", "page": 1},
        {"chunk_id": "b", "content": "B", "source": "s", "page": 2},
    ])
    assert h1 == h2
    assert len(h1) == 64


def test_score_field_does_not_affect_snapshot():
    a = [{"chunk_id": "a", "content": "A", "source": "s", "page": 1, "score": 1}]
    b = [{"chunk_id": "a", "content": "A", "source": "s", "page": 1, "score": 0}]
    assert snapshot_hash(a) == snapshot_hash(b)


def test_recording_retriever_empty_evidence_is_ok():
    class Empty:
        def search(self, query, top_k=5):
            return []
    rec = RecordingRetriever(Empty())
    assert rec.search("HyDE", top_k=5) == []
    cap = rec.captures[-1]
    assert cap["capture_status"] == "ok"
    assert cap["evidence"] == []
    assert cap["query"] == "HyDE"
    assert cap["top_k"] == 5


def test_recording_retriever_exception_is_missing_not_empty_list():
    class Boom:
        def search(self, query, top_k=5):
            raise RuntimeError("down")
    rec = RecordingRetriever(Boom())
    try:
        rec.search("HyDE")
    except RuntimeError:
        pass
    cap = rec.captures[-1]
    assert cap["capture_status"] == "missing"
    assert cap["evidence"] is None


def test_retriever_config_hash_stable():
    cfg = {
        "embedding_model": "nomic-embed-text",
        "chunking": {"chunk_size": 512, "chunk_overlap": 50, "min_chunk_length": 20},
        "reranker": "jinaai/jina-reranker-v2-base-multilingual",
        "top_k": 5,
        "retrieval_depth": 20,
    }
    assert retriever_config_hash(cfg) == retriever_config_hash(dict(cfg))
