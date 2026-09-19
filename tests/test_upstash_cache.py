"""The Upstash backend, against an in-memory stand-in for its REST API.

No network. The fake mirrors Upstash's actual response shapes -- notably that
query() returns a cosine *similarity*, where Chroma returns a distance. Getting
that backwards would invert every threshold, so it is pinned here.
"""

import json
import math

import pytest

import cache_chromadb as cc
import config
import volatility_policy as vp


class FakeUpstash:
    """Minimal Upstash Vector, with its metadata and scoring semantics."""

    def __init__(self):
        self.rows = {}
        self.calls = {"upsert": 0, "query": 0, "fetch": 0, "range": 0, "delete": 0}

    def upsert(self, entry_id, vector, document, metadata):
        self.calls["upsert"] += 1
        payload = dict(metadata)
        payload["_document"] = document
        self.rows[entry_id] = {"vector": list(vector), "metadata": payload}

    def query(self, vector, top_k):
        self.calls["query"] += 1
        def cos(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a)) or 1.0
            nb = math.sqrt(sum(x * x for x in b)) or 1.0
            return dot / (na * nb)
        scored = []
        for entry_id, row in self.rows.items():
            meta = dict(row["metadata"])
            scored.append({
                "id": entry_id,
                "score": cos(vector, row["vector"]),  # similarity, not distance
                "document": meta.pop("_document", ""),
                "metadata": meta,
            })
        scored.sort(key=lambda r: -r["score"])
        return scored[:top_k]

    def fetch(self, entry_id):
        self.calls["fetch"] += 1
        row = self.rows.get(entry_id)
        if not row:
            return None
        meta = dict(row["metadata"])
        return {"id": entry_id, "document": meta.pop("_document", ""), "metadata": meta}

    def list_all(self, limit=1000):
        self.calls["range"] += 1
        out = []
        for entry_id, row in list(self.rows.items())[:limit]:
            meta = dict(row["metadata"])
            out.append({"id": entry_id, "document": meta.pop("_document", ""),
                        "metadata": meta})
        return out

    def delete(self, ids):
        self.calls["delete"] += 1
        n = 0
        for entry_id in ids:
            if self.rows.pop(entry_id, None) is not None:
                n += 1
        return n

    def reset(self):
        self.rows.clear()

    def count(self):
        return len(self.rows)


@pytest.fixture
def upstash(monkeypatch):
    fake = FakeUpstash()
    instance = cc.UpstashCache(store=fake)
    monkeypatch.setattr(cc, "_cache_db", instance)
    monkeypatch.setattr(cc, "get_cache", lambda: instance)
    instance.fake = fake
    return instance


def test_write_then_read(upstash):
    upstash.add_to_cache("what is photosynthesis", "A summary.", volatility=vp.STATIC)
    fresh, expired = upstash.find_similar_candidates("what is photosynthesis", floor=0.9)
    assert len(fresh) == 1
    assert fresh[0].summary == "A summary."
    assert not expired


def test_score_is_read_as_similarity_not_distance(upstash):
    """Upstash returns cosine similarity; Chroma returns distance. Treating one
    as the other inverts every threshold in the system."""
    upstash.add_to_cache("what is photosynthesis", "S", volatility=vp.STATIC)
    fresh, _ = upstash.find_similar_candidates("what is photosynthesis", floor=0.0)
    assert fresh[0].similarity > 0.99  # identical text


def test_unrelated_query_scores_low(upstash):
    upstash.add_to_cache("what is photosynthesis", "S", volatility=vp.STATIC)
    fresh, _ = upstash.find_similar_candidates("completely different thing", floor=0.9)
    assert fresh == []


def test_expiry_is_enforced(upstash):
    import time
    upstash.add_to_cache("live cricket score", "Old.", volatility=vp.REALTIME)
    later = int(time.time()) + 400
    fresh, expired = upstash.find_similar_candidates("live cricket score", floor=0.9, now=later)
    assert fresh == []
    assert len(expired) == 1


def test_document_survives_the_metadata_round_trip(upstash):
    """The query text rides inside Upstash metadata, since there is no separate
    documents array."""
    upstash.add_to_cache("what is photosynthesis", "S", volatility=vp.STATIC)
    fresh, _ = upstash.find_similar_candidates("what is photosynthesis", floor=0.9)
    assert fresh[0].query == "what is photosynthesis"


def test_source_urls_round_trip(upstash):
    srcs = [{"url": "https://a.example", "title": "A"}]
    upstash.add_to_cache("q here", "S", volatility=vp.STATIC, source_urls=srcs)
    fresh, _ = upstash.find_similar_candidates("q here", floor=0.9)
    assert fresh[0].source_urls == srcs


def test_rewrite_updates_in_place(upstash):
    a = upstash.add_to_cache("what is photosynthesis", "First", volatility=vp.STATIC)
    b = upstash.add_to_cache("what is photosynthesis", "Second", volatility=vp.STATIC)
    assert a == b
    assert upstash.fake.count() == 1


def test_zero_ttl_is_not_written(upstash, monkeypatch):
    monkeypatch.setitem(vp.TTL_POLICY, vp.REALTIME, {"default": 0, "min": 0, "max": 0})
    assert upstash.add_to_cache("live score", "S", volatility=vp.REALTIME) is None
    assert upstash.fake.count() == 0


def test_stats_report_the_backend(upstash):
    upstash.add_to_cache("q here now", "S", volatility=vp.STATIC)
    stats = upstash.get_cache_stats()
    assert stats["backend"] == "upstash"
    assert stats["total_queries"] == 1
    # Chroma-only fields must not leak into the hosted backend's stats.
    assert "database_path" not in stats


def test_delete_and_purge(upstash):
    import time
    upstash.add_to_cache("static thing here", "S", volatility=vp.STATIC)
    upstash.add_to_cache("live score now", "S", volatility=vp.REALTIME)
    assert upstash.purge_expired(now=int(time.time()) + 400) == 1
    assert upstash.fake.count() == 1


def test_clear_resets_the_index(upstash):
    upstash.add_to_cache("q here now", "S", volatility=vp.STATIC)
    upstash.clear_cache()
    assert upstash.fake.count() == 0


def test_touch_makes_no_api_calls(upstash):
    """Upstash cannot patch metadata without re-sending the vector, so keeping
    a hit counter would spend an embedding call and an upsert per cache hit."""
    upstash.add_to_cache("q here now", "S", volatility=vp.STATIC)
    before = dict(upstash.fake.calls)
    upstash.touch("anything")
    assert upstash.fake.calls == before


def test_view_all_cache_works_on_this_backend(upstash):
    upstash.add_to_cache("q here now", "S", volatility=vp.STATIC)
    items = cc.view_all_cache()
    assert len(items) == 1
    assert items[0]["query"] == "q here now"
    assert items[0]["is_expired"] is False


def test_find_similar_query_shim(upstash):
    upstash.add_to_cache("what is photosynthesis", "A summary.", volatility=vp.STATIC)
    summary, sim = upstash.find_similar_query("what is photosynthesis", threshold=0.9)
    assert summary == "A summary."
    assert sim > 0.9
