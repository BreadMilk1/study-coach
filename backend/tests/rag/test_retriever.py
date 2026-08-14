from app.rag.retriever import Retriever


def test_retriever_returns_chunk_most_similar_to_query_first(fake_embedder, chroma_collection):
    retriever = Retriever(collection=chroma_collection, embedder=fake_embedder)
    retriever.add_chunks([
        {"chunk_id": "a:1:0", "content": "Prompt engineering basics and tactics",
         "source": "a.pdf", "page": 1},
        {"chunk_id": "b:1:0", "content": "Retrieval augmented generation HyDE hypothetical document",
         "source": "b.pdf", "page": 1},
        {"chunk_id": "c:1:0", "content": "Neural network training backpropagation",
         "source": "c.pdf", "page": 1},
    ])

    results = retriever.search("What is HyDE hypothetical document?", top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "b:1:0"
    assert results[0]["source"] == "b.pdf"
    assert results[0]["page"] == 1
    assert "score" in results[0]


def test_retriever_add_chunks_is_idempotent_for_stable_ids(fake_embedder, chroma_collection):
    retriever = Retriever(collection=chroma_collection, embedder=fake_embedder)
    chunk = {
        "chunk_id": "abc123:1:0",
        "content": "Stable content about HyDE retrieval.",
        "source": "notes.pdf",
        "page": 1,
    }

    retriever.add_chunks([chunk])
    retriever.add_chunks([{**chunk, "source": "renamed.pdf"}])

    assert chroma_collection.count() == 1
    stored = chroma_collection.get(ids=["abc123:1:0"])
    assert stored["metadatas"][0]["source"] == "renamed.pdf"


def test_retriever_get_chunk_returns_stored_document(fake_embedder, chroma_collection):
    retriever = Retriever(collection=chroma_collection, embedder=fake_embedder)
    retriever.add_chunks([
        {
            "chunk_id": "abc:1:0",
            "content": "HyDE rewrites the query into a hypothetical answer.",
            "source": "notes.pdf",
            "page": 2,
        },
    ])

    found = retriever.get_chunk("abc:1:0")
    missing = retriever.get_chunk("missing")

    assert found == {
        "chunk_id": "abc:1:0",
        "content": "HyDE rewrites the query into a hypothetical answer.",
        "source": "notes.pdf",
        "page": 2,
    }
    assert missing is None


def test_retriever_close_closes_embedder_once(fake_embedder, chroma_collection):
    calls = []
    fake_embedder.close = lambda: calls.append(1)
    retriever = Retriever(collection=chroma_collection, embedder=fake_embedder)

    retriever.close()
    retriever.close()

    assert calls == [1]
