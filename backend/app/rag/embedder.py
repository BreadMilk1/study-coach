from typing import Protocol


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OllamaEmbedder:
    def __init__(self, model: str = "nomic-embed-text", base_url: str | None = None):
        self.model = model
        self.base_url = base_url
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import ollama
            self._client = ollama.Client(host=self.base_url) if self.base_url else ollama.Client()
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        client = self._ensure_client()
        out: list[list[float]] = []
        for text in texts:
            resp = client.embeddings(model=self.model, prompt=text)
            out.append(list(resp["embedding"]))
        return out
