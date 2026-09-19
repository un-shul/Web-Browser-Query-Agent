"""Routes, SSE framing, and template wiring.

Search and summarisation are faked, so nothing here touches the network.
"""

import json

import pytest

import pipeline
import volatility_policy as vp
from web_search import PageContent, SearchBundle, SearchResult


@pytest.fixture
def client(cache, monkeypatch):
    import app as app_module

    monkeypatch.setattr(
        pipeline, "search",
        lambda q, topic="general", time_range=None, max_results=None: SearchBundle(
            query=q,
            results=[SearchResult(url=f"https://example.test/{i}", title=f"Result {i}",
                                  snippet="s", score=0.9, raw_content="raw")
                     for i in range(3)],
        ),
    )
    monkeypatch.setattr(
        pipeline, "fetch_contents",
        lambda results, limit=None, timeout=None: [
            PageContent(r.url, r.title, "Body. " * 60, "scrape") for r in results
        ],
    )
    monkeypatch.setattr(pipeline, "_summarize",
                        lambda pages, combined, query: f"Summary for {query}.")
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


# --- pages -------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/", "/cache"])
def test_pages_render(client, path):
    resp = client.get(path)
    assert resp.status_code == 200
    assert b"<html" in resp.data


def test_search_page_links_its_assets(client):
    html = client.get("/").data.decode()
    assert "style.css" in html
    assert "app.js" in html


def test_cache_page_links_its_assets(client):
    html = client.get("/cache").data.decode()
    assert "cache.js" in html


def test_theme_is_set_before_first_paint(client):
    """A theme applied after load would flash the wrong colours."""
    html = client.get("/").data.decode()
    assert "localStorage.getItem('theme')" in html
    assert html.index("localStorage.getItem('theme')") < html.index("</head>")


def test_nav_marks_the_current_page(client):
    assert 'aria-current="page"' in client.get("/").data.decode()


# --- healthz -----------------------------------------------------------------


def test_healthz_reports_backends(client):
    body = client.get("/healthz").get_json()
    assert body["ok"] is True
    for key in ["search_configured", "llm_provider", "vector_store",
                "classifier_loaded", "embedder_available"]:
        assert key in body


# --- SSE ---------------------------------------------------------------------


def _events(resp):
    out = []
    for line in resp.data.decode().splitlines():
        if line.startswith("data: "):
            out.append(json.loads(line[6:]))
    return out


def test_sse_streams_progress_then_completes(client):
    resp = client.get("/search_progress?query=what+is+photosynthesis")
    assert resp.status_code == 200
    assert resp.mimetype == "text/event-stream"
    events = _events(resp)
    assert events[-1]["stage"] == "complete"
    assert "summary" in events[-1]


def test_sse_disables_proxy_buffering(client):
    """Without this an intermediary can hold the stream until it completes."""
    resp = client.get("/search_progress?query=test+query+here")
    assert resp.headers.get("X-Accel-Buffering") == "no"
    assert "no-cache" in resp.headers.get("Cache-Control", "")


def test_sse_rejects_an_empty_query(client):
    events = _events(client.get("/search_progress?query="))
    assert events[0]["stage"] == "error"


def test_sse_carries_the_verdict_and_trace(client):
    events = _events(client.get("/search_progress?query=what+is+photosynthesis"))
    final = events[-1]
    assert final["verdict"]["volatility"] == vp.STATIC
    assert "cache" in final


def test_sse_carries_sources(client):
    events = _events(client.get("/search_progress?query=what+is+photosynthesis"))
    assert len(events[-1]["sources"]) == 3


def test_sse_reports_a_cache_hit(client, cache):
    cache.add_to_cache("what is photosynthesis", "CACHED", volatility=vp.STATIC)
    events = _events(client.get("/search_progress?query=what+is+photosynthesis"))
    assert events[-1]["is_cached"] is True
    assert events[-1]["summary"] == "CACHED"


def test_refresh_flag_bypasses_the_cache(client, cache):
    cache.add_to_cache("what is photosynthesis", "CACHED", volatility=vp.STATIC)
    events = _events(
        client.get("/search_progress?query=what+is+photosynthesis&refresh=1"))
    assert events[-1]["is_cached"] is False


def test_realtime_query_never_reports_a_cache_hit(client, cache):
    cache.add_to_cache("live cricket score india vs australia", "STALE",
                       volatility=vp.REALTIME)
    events = _events(
        client.get("/search_progress?query=live+cricket+score+ind+vs+aus"))
    assert events[-1]["is_cached"] is False


# --- JSON endpoint -----------------------------------------------------------


def test_json_search(client):
    body = client.post("/search", data={"query": "what is photosynthesis"}).get_json()
    assert "summary" in body
    assert len(body["sources"]) == 3


def test_json_search_rejects_empty(client):
    assert client.post("/search", data={"query": "  "}).status_code == 400


def test_json_search_rejects_junk(client):
    assert client.post("/search", data={"query": "!!!!!"}).status_code == 400


# --- cache admin -------------------------------------------------------------


def test_cache_stats_endpoint(client, cache):
    cache.add_to_cache("q one here", "S", volatility=vp.STATIC)
    body = client.get("/cache-stats").get_json()
    assert body["total_queries"] == 1
    assert "by_volatility" in body


def test_cache_view_exposes_freshness(client, cache):
    cache.add_to_cache("q one here", "S", volatility=vp.STATIC)
    item = client.get("/cache-view").get_json()["items"][0]
    for key in ["volatility", "is_expired", "ttl_seconds", "hit_count", "age_seconds"]:
        assert key in item


def test_cache_search_requires_a_term(client):
    assert client.get("/cache-search").status_code == 400


def test_cache_search_filters(client, cache):
    cache.add_to_cache("photosynthesis basics", "S", volatility=vp.STATIC)
    cache.add_to_cache("cricket scores today", "S", volatility=vp.STATIC)
    assert client.get("/cache-search?q=photosynthesis").get_json()["found_items"] == 1


def test_cache_delete_item(client, cache):
    cache.add_to_cache("q one here", "S", volatility=vp.STATIC)
    entry_id = client.get("/cache-view").get_json()["items"][0]["id"]
    assert client.post("/cache-delete-item", json={"id": entry_id}).status_code == 200
    assert client.get("/cache-stats").get_json()["total_queries"] == 0


def test_cache_delete_item_needs_an_id(client):
    assert client.post("/cache-delete-item", json={}).status_code == 400


def test_cache_purge_removes_expired_only(client, cache, monkeypatch):
    cache.add_to_cache("static thing here", "S", volatility=vp.STATIC)
    cache.add_to_cache("live score now", "S", volatility=vp.REALTIME)
    import cache_chromadb
    real_now = cache_chromadb._now
    monkeypatch.setattr(cache_chromadb, "_now", lambda: real_now() + 400)
    assert client.post("/cache-purge").get_json()["purged"] == 1


def test_cache_clear(client, cache):
    cache.add_to_cache("q one here", "S", volatility=vp.STATIC)
    assert client.post("/cache-clear").status_code == 200
