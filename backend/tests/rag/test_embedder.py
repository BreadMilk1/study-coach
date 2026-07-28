from app.rag.embedder import OllamaEmbedder


def test_ollama_embedder_ignores_process_proxy_settings(monkeypatch):
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def embeddings(self, *, model, prompt):
            return {"embedding": [1.0]}

    monkeypatch.setattr("ollama.Client", FakeClient)

    embedder = OllamaEmbedder(
        model="nomic-embed-text",
        base_url="http://127.0.0.1:11434",
    )

    assert embedder.embed(["local document"]) == [[1.0]]
    assert captured == {
        "host": "http://127.0.0.1:11434",
        "trust_env": False,
    }
