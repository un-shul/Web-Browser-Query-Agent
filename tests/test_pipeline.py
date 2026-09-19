"""Pipeline flow, with search and summarisation faked.

Asserts the decision sequence rather than answer quality -- in particular
that a realtime query never consults the semantic cache.
"""

import pytest

import pipeline
import volatility_policy as vp
from web_search import PageContent, SearchBundle, SearchResult


@pytest.fixture
def fake_web(monkeypatch):
    """Stand in for Tavily and the summariser. Records what was asked."""
    calls = {"search": [], "fetch": 0, "summarize": 0}

    def fake_search(query, topic="general", time_range=None, max_results=None):
        calls["search"].append({"query": query, "topic": topic, "time_range": time_range})
        return SearchBundle(
            query=query,
            results=[
                SearchResult(url=f"https://example.test/{i}", title=f"Result {i}",
                             snippet="snippet", score=0.9 - i / 10, raw_content="raw")
                for i in range(3)
            ],
        )

    def fake_fetch(results, limit=None, timeout=None):
        calls["fetch"] += 1
        return [
            PageContent(r.url, r.title, f"Body text for {r.url}. " * 40, "scrape")
            for r in results
        ]

    def fake_summarize(text, query=None, **kwargs):
        calls["summarize"] += 1
        return f"Summary for {query}."

    monkeypatch.setattr(pipeline, "search", fake_search)
    monkeypatch.setattr(pipeline, "fetch_contents", fake_fetch)
    monkeypatch.setattr(pipeline, "summarize_text", fake_summarize)
    return calls


def stages(query, **kwargs):
    return [e.stage for e in pipeline.process_query(query, **kwargs)]


def final(query, **kwargs):
    return list(pipeline.process_query(query, **kwargs))[-1]


# --- classification ----------------------------------------------------------


@pytest.mark.parametrize("query,expected", [
    ("live cricket score india vs australia", vp.REALTIME),
    ("latest ai news", vp.DYNAMIC),
    ("best laptops for programming", vp.SLOW),
    ("what is photosynthesis", vp.STATIC),
])
def test_classify_assigns_volatility(cache, query, expected):
    assert pipeline.classify(query).volatility == expected


@pytest.mark.parametrize("query", ["", "  ", "ab", "x" * 500, "!!!!!", "12345"])
def test_cheap_gates_reject_without_a_model(cache, query):
    verdict = pipeline.classify(query)
    assert not verdict.is_valid
    assert verdict.source == "gate"


def test_valid_queries_survive_the_gates(cache):
    for query in ["weather in bangalore", "tesla stock price today",
                  "live cricket score", "what is photosynthesis"]:
        assert pipeline.classify(query).is_valid, query


def test_ttl_matches_the_volatility_class(cache):
    """With no LLM there is no TTL hint, so each class gets its default."""
    for query in ["live cricket score", "latest ai news", "what is photosynthesis"]:
        verdict = pipeline.classify(query)
        assert verdict.ttl_seconds == vp.ttl_for(verdict.volatility)


def test_news_topic_is_selected_for_volatile_queries(cache):
    assert pipeline.classify("latest ai news").search_topic == "news"
    assert pipeline.classify("what is photosynthesis").search_topic == "general"


def test_realtime_queries_request_a_day_time_range(cache):
    assert pipeline.classify("live cricket score").time_range == "day"
    assert pipeline.classify("what is photosynthesis").time_range is None


# --- the realtime guarantee --------------------------------------------------


def test_realtime_query_does_not_reuse_a_similar_cached_answer(cache, fake_web):
    """The core of the feature.

    A realtime query must not be answered from another query's cache entry,
    however high the cosine similarity.
    """
    cache.add_to_cache("live cricket score india vs australia",
                       "STALE: India 247/4", volatility=vp.REALTIME)

    result = final("live cricket score ind vs aus")
    assert result.stage == "complete"
    assert result.data["is_cached"] is False
    assert "STALE" not in result.data["summary"]


def test_realtime_query_reaches_the_search_backend(cache, fake_web):
    cache.add_to_cache("live cricket score", "STALE", volatility=vp.REALTIME)
    stage_list = stages("live cricket score india vs australia")
    assert "searching" in stage_list
    assert len(fake_web["search"]) == 1


def test_static_query_does_reuse_its_cached_answer(cache, fake_web):
    cache.add_to_cache("what is photosynthesis", "CACHED ANSWER", volatility=vp.STATIC)
    result = final("what is photosynthesis")
    assert result.data["is_cached"] is True
    assert result.data["summary"] == "CACHED ANSWER"
    assert fake_web["search"] == []  # never hit the network


def test_expired_static_entry_triggers_a_fresh_search(cache, fake_web, monkeypatch):
    cache.add_to_cache("what is photosynthesis", "OLD", volatility=vp.STATIC)
    import cache_chromadb
    real_now = cache_chromadb._now
    monkeypatch.setattr(cache_chromadb, "_now", lambda: real_now() + 400 * 86400)
    result = final("what is photosynthesis")
    assert result.data["is_cached"] is False
    assert len(fake_web["search"]) == 1


def test_force_refresh_bypasses_a_valid_cache_entry(cache, fake_web):
    cache.add_to_cache("what is photosynthesis", "CACHED", volatility=vp.STATIC)
    result = final("what is photosynthesis", force_refresh=True)
    assert result.data["is_cached"] is False
    assert len(fake_web["search"]) == 1


# --- flow --------------------------------------------------------------------


def test_cache_miss_runs_the_full_pipeline(cache, fake_web):
    stage_list = stages("what is photosynthesis")
    for stage in ["validating", "classified", "cache", "searching", "found",
                  "scraping", "read", "summarizing", "caching", "complete"]:
        assert stage in stage_list, stage


def test_invalid_query_stops_before_searching(cache, fake_web):
    stage_list = stages("!!!!!")
    assert stage_list[-1] == "error"
    assert "searching" not in stage_list
    assert fake_web["search"] == []


def test_progress_never_decreases(cache, fake_web):
    values = [e.progress for e in pipeline.process_query("what is photosynthesis")
              if e.stage != "error"]
    assert values == sorted(values)


def test_complete_event_carries_sources(cache, fake_web):
    result = final("what is photosynthesis")
    assert len(result.data["sources"]) == 3
    assert all("url" in s and "title" in s for s in result.data["sources"])


def test_complete_event_carries_the_verdict(cache, fake_web):
    verdict = final("what is photosynthesis").data["verdict"]
    assert verdict["volatility"] == vp.STATIC
    assert verdict["source"]


def test_cache_trail_survives_into_the_completion_event(cache, fake_web):
    """Regression: the completion event replaced the trail with a stub that had
    no decision and no candidates, so the UI lost the miss reasoning -- exactly
    the case worth showing."""
    cache.add_to_cache("what is photosynthesis in plants", "Earlier answer.",
                       volatility=vp.STATIC)
    trail = final("what is photosynthesis").data["cache"]
    assert trail["decision"]
    assert "candidates" in trail
    assert "verifier" in trail


def test_answer_is_written_back_to_the_cache(cache, fake_web):
    final("what is photosynthesis")
    assert cache.get_cache_stats()["total_queries"] == 1


def test_search_failure_surfaces_as_an_error(cache, fake_web, monkeypatch):
    from web_search import SearchError

    def boom(*a, **k):
        raise SearchError("no TAVILY_API_KEY set")

    monkeypatch.setattr(pipeline, "search", boom)
    result = final("what is photosynthesis")
    assert result.stage == "error"
    assert "TAVILY" in result.message


def test_empty_search_results_surface_as_an_error(cache, fake_web, monkeypatch):
    monkeypatch.setattr(pipeline, "search",
                        lambda *a, **k: SearchBundle(query="q", results=[]))
    assert final("what is photosynthesis").stage == "error"


def test_unreadable_pages_surface_as_an_error(cache, fake_web, monkeypatch):
    monkeypatch.setattr(pipeline, "fetch_contents", lambda *a, **k: [])
    assert final("what is photosynthesis").stage == "error"


def test_summariser_failure_surfaces_as_an_error(cache, fake_web, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(pipeline, "summarize_text", boom)
    result = final("what is photosynthesis")
    assert result.stage == "error"
    assert "model exploded" in result.message


def test_run_collapses_the_stream_into_one_result(cache, fake_web):
    result = pipeline.run("what is photosynthesis")
    assert result["stage"] == "complete"
    assert "summary" in result


def test_run_reports_errors(cache, fake_web):
    assert "error" in pipeline.run("!!!!!")


# --- audit trail -------------------------------------------------------------


def test_cache_miss_explains_itself(cache, fake_web):
    events = {e.stage: e for e in pipeline.process_query("what is photosynthesis")}
    assert events["cache_miss"].data["cache"]["decision"]


def test_realtime_bypass_is_recorded_in_the_trail(cache, fake_web):
    events = {e.stage: e for e in pipeline.process_query("live cricket score")}
    assert "bypassed" in events["cache_miss"].data["cache"]["decision"]


def test_candidate_scores_are_exposed(cache, fake_web):
    cache.add_to_cache("what is photosynthesis", "A", volatility=vp.STATIC)
    cache.add_to_cache("explain photosynthesis briefly", "B", volatility=vp.STATIC)
    trail = pipeline.lookup_cache("what is photosynthesis",
                                  pipeline.classify("what is photosynthesis"))
    assert trail["candidates"]
    assert "similarity" in trail["candidates"][0]


@pytest.mark.parametrize("seconds,expected", [
    (0, "not at all"), (90, "90s"), (600, "10m"), (7200, "2h"), (86400 * 180, "180d"),
])
def test_human_readable_durations(seconds, expected):
    assert pipeline._human(seconds) == expected
