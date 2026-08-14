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


def test_ollama_embedder_close_is_idempotent_and_does_not_initialize_client(monkeypatch):
    calls = []

    class ExplodingClient:
        def __init__(self, **_kwargs):
            calls.append("init")

    monkeypatch.setattr("ollama.Client", ExplodingClient)
    embedder = OllamaEmbedder()

    embedder.close()
    embedder.close()
    assert calls == []


def test_ollama_embedder_close_closes_existing_client_once(monkeypatch):
    close_calls = []

    class FakeClient:
        def close(self):
            close_calls.append(1)

    monkeypatch.setattr("ollama.Client", lambda **_kwargs: FakeClient())
    embedder = OllamaEmbedder()
    embedder._ensure_client()
    embedder.close()
    embedder.close()
    assert close_calls == [1]
