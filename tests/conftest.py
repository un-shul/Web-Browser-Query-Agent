"""Shared fixtures.

The embedder is stubbed everywhere so the suite never downloads MiniLM, and
Chroma runs in-memory so nothing touches ./chroma_db.
"""

import hashlib
import math
import os

# Set before transformers or huggingface_hub is imported. Tests that touch the
# summariser's tokeniser otherwise let huggingface_hub phone home to check for
# model updates -- 84 outbound requests in a suite that is supposed to make
# none. Offline mode uses the local cache only.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pytest

from queryagent import cache as cachemod
from queryagent import config
from queryagent import embeddings
from queryagent.llm import providers, router

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
def no_llm(monkeypatch):
    """Disable the LLM for every test unless it opts in.

    Without this, a developer with keys in .env runs the whole suite against
    the real providers: slow, non-deterministic, and quietly spending a daily
    quota. Tests that need a model install a FakeProvider explicitly.
    """
    monkeypatch.setattr(config, "LLM_DISABLED", True)
    monkeypatch.setattr(config, "LLM_PROVIDER", "none")
    # Also neutralise the search key. A developer with .env populated would
    # otherwise see different behaviour from CI, which is how a test asserting
    # /healthz reported ok passed locally and failed on the runner.
    monkeypatch.setattr(config, "TAVILY_API_KEY", "")
    providers.set_chain([])
    router.clear_memo()
    providers.get_budget().reset()
    yield
    providers.reset_chain()
    router.clear_memo()


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
    """A fresh in-memory cache per test.

    chromadb is imported here rather than at module scope so that test files
    which do not use this fixture -- test_production_bundle.py in particular --
    can run in an environment with only the production dependencies installed.
    """
    import chromadb

    client = chromadb.EphemeralClient()

    class MemoryCache(cachemod.ChromaDBCache):
        def __init__(self):
            self.client = client
            self.collection = client.get_or_create_collection(
                name="test_cache", metadata={"hnsw:space": "cosine"}
            )
            self._migration_epoch = None

    instance = MemoryCache()
    monkeypatch.setattr(cachemod, "_cache_db", instance)
    monkeypatch.setattr(cachemod, "get_cache", lambda: instance)
    yield instance
    try:
        client.delete_collection("test_cache")
    except Exception:
        pass
