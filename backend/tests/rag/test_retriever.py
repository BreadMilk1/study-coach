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
