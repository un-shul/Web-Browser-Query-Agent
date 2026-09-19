"""Freshness behaviour of the semantic cache."""

import json
import time

import pytest

import cache_chromadb as cc
import volatility_policy as vp


def test_write_then_read_roundtrip(cache):
    cache.add_to_cache("what is photosynthesis", "A summary.", volatility=vp.STATIC)
    fresh, expired = cache.find_similar_candidates("what is photosynthesis", floor=0.9)
    assert len(fresh) == 1
    assert fresh[0].summary == "A summary."
    assert fresh[0].volatility == vp.STATIC
    assert not expired


@pytest.mark.parametrize("volatility", vp.VOLATILITIES)
def test_each_volatility_sets_its_own_ttl(cache, volatility):
    cache.add_to_cache(f"query about {volatility}", "S", volatility=volatility)
    fresh, _ = cache.find_similar_candidates(f"query about {volatility}", floor=0.9)
    assert fresh[0].ttl_seconds == vp.ttl_for(volatility)


def test_expired_entry_is_not_offered_as_fresh(cache):
    """The hole this whole change exists to close."""
    cache.add_to_cache("live cricket score", "Old score: 120/3", volatility=vp.REALTIME)
    later = int(time.time()) + vp.ttl_for(vp.REALTIME) + 5
    fresh, expired = cache.find_similar_candidates("live cricket score", floor=0.9, now=later)
    assert fresh == []
    assert len(expired) == 1
    assert expired[0].summary == "Old score: 120/3"


def test_static_entry_survives_far_into_the_future(cache):
    cache.add_to_cache("what is photosynthesis", "S", volatility=vp.STATIC)
    a_month = int(time.time()) + 30 * 86400
    fresh, _ = cache.find_similar_candidates("what is photosynthesis", floor=0.9, now=a_month)
    assert len(fresh) == 1


def test_realtime_entry_expires_within_minutes(cache):
    cache.add_to_cache("live score", "S", volatility=vp.REALTIME)
    fresh, _ = cache.find_similar_candidates("live score", floor=0.9, now=int(time.time()) + 301)
    assert fresh == []


def test_zero_ttl_is_not_written_at_all(cache, monkeypatch):
    monkeypatch.setitem(vp.TTL_POLICY, vp.REALTIME, {"default": 0, "min": 0, "max": 0})
    assert cache.add_to_cache("live score", "S", volatility=vp.REALTIME) is None
    fresh, expired = cache.find_similar_candidates("live score", floor=0.5)
    assert not fresh and not expired


def test_ttl_hint_is_clamped_on_write(cache):
    cache.add_to_cache("latest news", "S", volatility=vp.DYNAMIC, ttl_seconds=10**9)
    fresh, _ = cache.find_similar_candidates("latest news", floor=0.9)
    assert fresh[0].ttl_seconds == vp.TTL_POLICY[vp.DYNAMIC]["max"]


# --- retrieval ---------------------------------------------------------------


def test_returns_multiple_candidates(cache):
    for i in range(4):
        cache.add_to_cache(f"query variant {i}", f"summary {i}", volatility=vp.STATIC)
    fresh, _ = cache.find_similar_candidates("query variant 0", k=5, floor=-1.0)
    assert len(fresh) >= 4


def test_candidates_are_ordered_by_similarity(cache):
    for i in range(4):
        cache.add_to_cache(f"query variant {i}", f"summary {i}", volatility=vp.STATIC)
    fresh, _ = cache.find_similar_candidates("query variant 0", k=5, floor=-1.0)
    assert fresh == sorted(fresh, key=lambda c: -c.similarity)


def test_floor_excludes_weak_matches(cache):
    cache.add_to_cache("completely unrelated topic", "S", volatility=vp.STATIC)
    fresh, _ = cache.find_similar_candidates("what is photosynthesis", floor=0.99)
    assert fresh == []


def test_fresher_entry_is_reachable_even_when_less_similar(cache):
    """Regression: n_results=1 meant only the single nearest neighbour was
    ever considered, so a fresh entry ranked second was invisible behind an
    expired one."""
    cache.add_to_cache("live cricket score", "STALE", volatility=vp.REALTIME)
    cache.add_to_cache("live cricket score update", "FRESH", volatility=vp.STATIC)
    later = int(time.time()) + 400
    fresh, expired = cache.find_similar_candidates("live cricket score", k=5, floor=-1.0, now=later)
    assert "FRESH" in [c.summary for c in fresh]
    assert "STALE" in [c.summary for c in expired]


def test_source_urls_survive_the_json_round_trip(cache):
    urls = ["https://a.example/one", "https://b.example/two"]
    cache.add_to_cache("q", "S", volatility=vp.STATIC, source_urls=urls)
    fresh, _ = cache.find_similar_candidates("q", floor=0.9)
    assert fresh[0].source_urls == urls


def test_entries_without_a_summary_are_skipped(cache):
    cache.collection.add(
        ids=["blank"], embeddings=[[0.1] * 32], documents=["q"],
        metadatas=[{"summary": "", "schema_version": 2, "created_at": int(time.time()),
                    "expires_at": int(time.time()) + 999}],
    )
    fresh, _ = cache.find_similar_candidates("q", floor=-1.0)
    assert fresh == []


def test_migration_epoch_marker_is_never_returned(cache):
    cache.add_to_cache("q", "S", volatility=vp.STATIC)
    _ = cache.migration_epoch
    fresh, expired = cache.find_similar_candidates("q", k=10, floor=-1.0)
    ids = [c.id for c in fresh + expired]
    assert cc._MIGRATION_EPOCH_KEY not in ids


# --- legacy entries ----------------------------------------------------------


def _write_legacy(cache, query, summary):
    """Write exactly what the pre-schema-2 code wrote: summary only."""
    cache.collection.add(
        ids=[f"legacy-{query}"],
        embeddings=[cc.embeddings.encode_one(query)],
        documents=[query],
        metadatas=[{"summary": summary}],
    )


def test_legacy_entry_is_readable(cache):
    _write_legacy(cache, "what is photosynthesis", "Legacy summary.")
    fresh, expired = cache.find_similar_candidates("what is photosynthesis", floor=0.9)
    found = (fresh + expired)[0]
    assert found.summary == "Legacy summary."
    assert found.schema_version == 1
    assert found.volatility == vp.UNKNOWN


def test_legacy_entry_is_served_inside_the_grace_window(cache):
    _write_legacy(cache, "what is photosynthesis", "Legacy summary.")
    fresh, _ = cache.find_similar_candidates("what is photosynthesis", floor=0.9)
    assert len(fresh) == 1


def test_legacy_entry_expires_after_the_grace_window(cache):
    """Fail-closed: an entry whose real age is unknowable stops being served
    rather than being served forever."""
    _write_legacy(cache, "what is photosynthesis", "Legacy summary.")
    beyond = cache.migration_epoch + cc.LEGACY_GRACE_SECONDS + 10
    fresh, expired = cache.find_similar_candidates(
        "what is photosynthesis", floor=0.9, now=beyond
    )
    assert fresh == []
    assert len(expired) == 1


def test_backfill_upgrades_a_legacy_entry(cache):
    _write_legacy(cache, "live cricket score", "Old score.")
    fresh, expired = cache.find_similar_candidates("live cricket score", floor=0.9)
    candidate = (fresh + expired)[0]
    assert candidate.schema_version == 1

    cache.backfill(candidate)

    fresh2, expired2 = cache.find_similar_candidates("live cricket score", floor=0.9)
    upgraded = (fresh2 + expired2)[0]
    assert upgraded.schema_version == cc.SCHEMA_VERSION
    # Classified by heuristics alone -- backfill must never spend an LLM call.
    assert upgraded.volatility == vp.REALTIME


def test_backfill_is_a_noop_on_current_entries(cache):
    cache.add_to_cache("q", "S", volatility=vp.STATIC)
    fresh, _ = cache.find_similar_candidates("q", floor=0.9)
    before = fresh[0].created_at
    cache.backfill(fresh[0])
    after, _ = cache.find_similar_candidates("q", floor=0.9)
    assert after[0].created_at == before


def test_normalize_meta_tolerates_garbage():
    for bad in [None, {}, {"summary": "x"}, {"schema_version": "nonsense"},
                {"created_at": None, "expires_at": ""}]:
        meta = cc._normalize_meta(bad, 1_000_000)
        assert isinstance(meta["created_at"], int)
        assert isinstance(meta["expires_at"], int)
        assert isinstance(meta["volatility"], str)


@pytest.mark.parametrize("raw,expected", [
    ('["a","b"]', ["a", "b"]), ("[]", []), ("not json", []),
    (None, []), ("", []), ('{"not":"a list"}', []), (["a"], ["a"]),
])
def test_decode_urls_never_raises(raw, expected):
    assert cc._decode_urls(raw) == expected


# --- admin -------------------------------------------------------------------


def test_touch_increments_hit_count(cache):
    cache.add_to_cache("q", "S", volatility=vp.STATIC)
    fresh, _ = cache.find_similar_candidates("q", floor=0.9)
    cache.touch(fresh[0].id)
    again, _ = cache.find_similar_candidates("q", floor=0.9)
    assert again[0].hit_count == 1


def test_touch_on_a_missing_id_is_silent(cache):
    cache.touch("does-not-exist")


def test_purge_removes_only_expired(cache):
    cache.add_to_cache("static thing", "S", volatility=vp.STATIC)
    cache.add_to_cache("live score", "S", volatility=vp.REALTIME)
    removed = cache.purge_expired(now=int(time.time()) + 400)
    assert removed == 1
    stats = cache.get_cache_stats()
    assert stats["total_queries"] == 1


def test_stats_break_down_by_volatility(cache):
    cache.add_to_cache("what is photosynthesis", "S", volatility=vp.STATIC)
    cache.add_to_cache("latest news", "S", volatility=vp.DYNAMIC)
    stats = cache.get_cache_stats()
    assert stats["total_queries"] == 2
    assert stats["by_volatility"][vp.STATIC] == 1
    assert stats["by_volatility"][vp.DYNAMIC] == 1
    assert stats["schema_version"] == cc.SCHEMA_VERSION


def test_stats_count_legacy_entries(cache):
    _write_legacy(cache, "old query", "Old.")
    assert cache.get_cache_stats()["legacy_entries"] == 1


def test_clear_empties_the_cache(cache):
    cache.add_to_cache("q", "S", volatility=vp.STATIC)
    cache.clear_cache()
    assert cache.get_cache_stats()["total_queries"] == 0


# --- legacy single-candidate API --------------------------------------------


def test_find_similar_query_still_works(cache):
    cache.add_to_cache("what is photosynthesis", "A summary.", volatility=vp.STATIC)
    summary, similarity = cache.find_similar_query("what is photosynthesis", threshold=0.9)
    assert summary == "A summary."
    assert similarity > 0.9


def test_find_similar_query_returns_none_on_miss(cache):
    assert cache.find_similar_query("nothing cached", threshold=0.9) == (None, None)


def test_find_similar_query_will_not_return_an_expired_entry(cache):
    """The no-LLM path must respect TTLs too."""
    cache.add_to_cache("live score", "Stale.", volatility=vp.REALTIME)
    time.sleep(0)
    import cache_chromadb
    original = cache_chromadb._now
    try:
        cache_chromadb._now = lambda: original() + 400
        assert cache.find_similar_query("live score", threshold=0.9) == (None, None)
    finally:
        cache_chromadb._now = original


def test_view_all_cache_exposes_freshness(cache):
    cache.add_to_cache("q", "S", volatility=vp.STATIC, source_urls=["https://x.example"])
    items = cc.view_all_cache()
    assert len(items) == 1
    assert items[0]["volatility"] == vp.STATIC
    assert items[0]["is_expired"] is False
    assert items[0]["source_urls"] == ["https://x.example"]


def test_view_all_cache_survives_legacy_metadata(cache):
    """Regression: view_all_cache read metadata['summary'] directly, which
    raises KeyError on any entry written by a different schema version."""
    _write_legacy(cache, "old query", "Old summary.")
    items = cc.view_all_cache()
    assert len(items) == 1
    assert items[0]["schema_version"] == 1


def test_search_cache_filters_by_query_text(cache):
    cache.add_to_cache("photosynthesis basics", "S", volatility=vp.STATIC)
    cache.add_to_cache("cricket score", "S", volatility=vp.STATIC)
    assert len(cc.search_cache("photosynthesis")) == 1


def test_delete_by_query(cache):
    cache.add_to_cache("photosynthesis basics", "S", volatility=vp.STATIC)
    assert cc.delete_cache_by_query("photosynthesis") == 1
    assert cache.get_cache_stats()["total_queries"] == 0


def test_rewriting_the_same_query_updates_rather_than_duplicates(cache):
    """Regression: a force-refresh appended a second row for the same
    question, so re-running a query grew the cache without bound."""
    first = cache.add_to_cache("what is photosynthesis", "First answer.",
                               volatility=vp.STATIC)
    second = cache.add_to_cache("what is photosynthesis", "Second answer.",
                                volatility=vp.STATIC)
    assert first == second
    assert cache.get_cache_stats()["total_queries"] == 1
    fresh, _ = cache.find_similar_candidates("what is photosynthesis", floor=0.9)
    assert fresh[0].summary == "Second answer."


def test_exact_match_ignores_case_and_spacing(cache):
    a = cache.add_to_cache("What Is  Photosynthesis", "A", volatility=vp.STATIC)
    b = cache.add_to_cache("what is photosynthesis", "B", volatility=vp.STATIC)
    assert a == b


def test_different_queries_still_get_their_own_rows(cache):
    cache.add_to_cache("what is photosynthesis", "A", volatility=vp.STATIC)
    cache.add_to_cache("what is respiration", "B", volatility=vp.STATIC)
    assert cache.get_cache_stats()["total_queries"] == 2
