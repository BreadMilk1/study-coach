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
            client_kwargs: dict[str, object] = {"trust_env": False}
            if self.base_url:
                client_kwargs["host"] = self.base_url
            self._client = ollama.Client(**client_kwargs)
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        client = self._ensure_client()
        out: list[list[float]] = []
        for text in texts:
            resp = client.embeddings(model=self.model, prompt=text)
            out.append(list(resp["embedding"]))
        return out

    def close(self) -> None:
        """Close an initialized local client without creating one."""

        client = self._client
        self._client = None
        if client is not None:
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
