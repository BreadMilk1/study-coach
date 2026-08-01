import hashlib
import os

# Ensure test collection never imports app.main in production mode, even when
# the outer environment explicitly sets STUDY_COACH_TEST_MODE=0.
os.environ["STUDY_COACH_TEST_MODE"] = "1"

import chromadb
import pytest


class WordBagEmbedder:
    """Deterministic test embedder. Texts sharing words map to overlapping dims."""

    DIM = 64

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.DIM
            for word in text.lower().split():
                idx = int(hashlib.md5(word.encode()).hexdigest()[:8], 16) % self.DIM
                vec[idx] += 1.0
            n = sum(v * v for v in vec) ** 0.5
            if n:
                vec = [v / n for v in vec]
            out.append(vec)
        return out


@pytest.fixture
def fake_embedder() -> WordBagEmbedder:
    return WordBagEmbedder()


@pytest.fixture
def chroma_collection():
    client = chromadb.Client()  # in-memory
    name = "test_collection"
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return client.create_collection(name)
