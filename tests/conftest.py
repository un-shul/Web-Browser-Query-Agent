"""Shared fixtures.

The embedder is stubbed everywhere so the suite never downloads MiniLM, and
Chroma runs in-memory so nothing touches ./chroma_db.
"""

import hashlib
import math

import chromadb
import pytest

import cache_chromadb
import embeddings

STUB_DIM = 32


def _stub_vector(text: str):
    """Deterministic pseudo-embedding.

    Identical text gives an identical vector, and unrelated text gives a
    roughly orthogonal one, which is all these tests need. Tests that care
    about specific similarities set the vectors explicitly.
    """
    digest = hashlib.sha256(text.strip().lower().encode()).digest()
    raw = [(digest[i % len(digest)] - 128) / 128.0 for i in range(STUB_DIM)]
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]


@pytest.fixture(autouse=True)
def stub_embedder(monkeypatch):
    monkeypatch.setattr(embeddings, "encode_one", _stub_vector)
    monkeypatch.setattr(embeddings, "encode_many", lambda ts: [_stub_vector(t) for t in ts])
    monkeypatch.setattr(embeddings, "embedding_dim", lambda: STUB_DIM)
    monkeypatch.setattr(embeddings, "is_available", lambda: True)
    monkeypatch.setattr(embeddings, "get_embedder", lambda: None)
    yield


@pytest.fixture
def cache(monkeypatch):
    """A fresh in-memory cache per test."""
    client = chromadb.EphemeralClient()

    class MemoryCache(cache_chromadb.ChromaDBCache):
        def __init__(self):
            self.client = client
            self.collection = client.get_or_create_collection(
                name="test_cache", metadata={"hnsw:space": "cosine"}
            )
            self._migration_epoch = None

    instance = MemoryCache()
    monkeypatch.setattr(cache_chromadb, "_cache_db", instance)
    monkeypatch.setattr(cache_chromadb, "get_cache", lambda: instance)
    yield instance
    try:
        client.delete_collection("test_cache")
    except Exception:
        pass
