"""The single query flow, shared by the CLI and the web app.

app.py previously carried this logic twice -- once in the SSE generator and
once in the plain JSON endpoint -- with a third copy in main.py. Any new stage
had to be written three times, so the volatility layer lives here instead and
both entry points consume the same event stream.

Emits ProgressEvent objects. The web app serialises them as SSE; the CLI
prints them.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterator, List, Optional

import agent
import cache_chromadb as cache
import config
import embeddings
import volatility_policy as vp
from llm_gateway import reranker, router
from web_search import SearchError, fetch_contents, search

log = logging.getLogger(__name__)


@dataclass
class QueryVerdict:
    """How the router classified a query."""

    is_valid: bool = True
    volatility: str = vp.DEFAULT_VOLATILITY
    ttl_seconds: int = 0
    search_topic: str = "general"
    time_range: Optional[str] = None
    reason: str = ""
    source: str = ""  # gate | lr | heuristic:<rule> | llm | memo(...) | default
    search_query: Optional[str] = None
    confidence: Optional[float] = None
    p_valid: Optional[float] = None
    llm_calls: int = 0
    degraded: bool = False
    # Set when the regex floor overrode a less volatile LLM verdict.
    escalated_from: Optional[str] = None
    heuristic_floor: Optional[str] = None


@dataclass
class ProgressEvent:
    stage: str
    message: str
    progress: int = 0
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = {"stage": self.stage, "message": self.message, "progress": self.progress}
        out.update(self.data)
        return out


def classify(query: str, embedding=None, allow_llm: bool = True) -> QueryVerdict:
    """Decide validity and volatility.

    Delegates to llm_gateway.router, which spends at most one LLM call and
    skips it entirely for junk, memoised repeats, and anything the regex
    heuristics can settle.
    """
    verdict = router.route(query, embedding=embedding, allow_llm=allow_llm)
    return QueryVerdict(**{k: v for k, v in verdict.items() if k in QueryVerdict.__dataclass_fields__})


def lookup_cache(query: str, verdict: QueryVerdict, embedding=None) -> Dict[str, Any]:
    """Find a reusable cached answer, or explain why there isn't one.

    Returns an audit trail rather than a bare hit/miss, so the UI can show
    which candidates were considered and why the decision went as it did.
    """
    trail: Dict[str, Any] = {
        "hit": False, "summary": None, "similarity": None, "candidates": [],
        "expired_skipped": 0, "decision": "", "entry_id": None, "sources": [],
        "llm_calls": 0, "verifier": "", "reason": "", "degraded": False,
    }

    # A realtime query never reuses another query's answer, however similar.
    # Only its own short-TTL entry can serve it, which makes this structural
    # rather than a threshold to tune.
    if verdict.volatility == vp.REALTIME:
        fresh, expired = cache.find_similar_candidates(
            query, k=1, floor=0.98, embedding=embedding
        )
        trail["expired_skipped"] = len(expired)
        if fresh:
            cache.touch(fresh[0].id)
            trail.update(
                hit=True, summary=fresh[0].summary, similarity=fresh[0].similarity,
                entry_id=fresh[0].id, sources=fresh[0].source_urls,
                verifier="realtime_window",
                decision="exact realtime re-ask inside its 90s window",
            )
        else:
            trail["decision"] = "realtime query: cache bypassed"
        return trail

    fresh, expired = cache.find_similar_candidates(query, embedding=embedding)
    trail["expired_skipped"] = len(expired)

    # Upgrade any legacy entry we touched, heuristics only -- migration must
    # never spend an LLM call.
    for candidate in list(fresh) + list(expired):
        if candidate.schema_version < cache.SCHEMA_VERSION:
            cache.get_cache().backfill(candidate)

    outcome = reranker.verify(query, fresh)
    trail["candidates"] = outcome.candidates
    trail["llm_calls"] = outcome.llm_calls
    trail["verifier"] = outcome.decision
    trail["reason"] = outcome.reason
    trail["degraded"] = outcome.degraded

    if outcome.accepted and outcome.index is not None:
        best = fresh[outcome.index]
        cache.touch(best.id)
        trail.update(
            hit=True, summary=best.summary, similarity=best.similarity,
            entry_id=best.id, sources=best.source_urls,
            decision=outcome.reason,
        )
    else:
        trail["decision"] = outcome.reason or "no usable cached answer"
        if expired and not fresh:
            trail["decision"] += f" ({len(expired)} expired)"
    return trail


def process_query(query: str, force_refresh: bool = False) -> Iterator[ProgressEvent]:
    """Run the full pipeline, yielding progress as it goes."""
    query = (query or "").strip()

    yield ProgressEvent("validating", "Checking the query...", 5)

    # One embedding, reused by the classifier and the cache lookup.
    embedding = None
    try:
        if embeddings.is_available():
            embedding = embeddings.encode_one(query)
    except Exception as exc:
        log.warning("embedding unavailable: %s", exc)

    verdict = classify(query, embedding)
    if not verdict.is_valid:
        yield ProgressEvent("error", verdict.reason or "That is not a searchable query.", 0,
                            {"verdict": asdict(verdict)})
        return

    yield ProgressEvent(
        "classified",
        f"{verdict.volatility} query (cache for {_human(verdict.ttl_seconds)})",
        12,
        {"verdict": asdict(verdict)},
    )

    # --- cache ---
    if force_refresh:
        trail = {"hit": False, "decision": "force refresh requested", "candidates": [],
                 "expired_skipped": 0, "sources": [], "llm_calls": 0, "verifier": "bypassed",
                 "reason": "force refresh requested", "degraded": False}
        yield ProgressEvent("cache", "Bypassing cache (force refresh)", 18, {"cache": trail})
        cache_trail = trail
    else:
        yield ProgressEvent("cache", "Checking the cache...", 15)
        trail = lookup_cache(query, verdict, embedding)
        if trail["hit"]:
            yield ProgressEvent(
                "complete", "Answered from cache", 100,
                {"summary": trail["summary"], "is_cached": True,
                 "similarity": trail["similarity"], "sources": trail["sources"],
                 "verdict": asdict(verdict), "cache": trail},
            )
            return
        yield ProgressEvent("cache_miss", trail["decision"], 18, {"cache": trail})
    cache_trail = trail

    # --- search ---
    yield ProgressEvent("searching", "Searching the web...", 25)
    try:
        bundle = search(verdict.search_query or query,
                        topic=verdict.search_topic, time_range=verdict.time_range)
    except SearchError as exc:
        yield ProgressEvent("error", str(exc), 0)
        return
    if not bundle.results:
        yield ProgressEvent("error", "No search results found. Try a different query.", 0)
        return

    yield ProgressEvent("found", f"Found {len(bundle.results)} results", 35)

    # --- fetch ---
    yield ProgressEvent("scraping", f"Reading {len(bundle.results)} pages...", 45)
    pages = fetch_contents(bundle.results)
    if not pages:
        yield ProgressEvent("error", "Could not read any of the result pages.", 0)
        return

    sources = [{"url": p.url, "title": p.title, "via": p.via} for p in pages]
    yield ProgressEvent("read", f"Read {len(pages)} pages", 70, {"sources": sources})

    # --- summarise ---
    yield ProgressEvent("summarizing", "Summarising...", 80)
    combined = "\n\n".join(p.text[:5000] for p in pages)
    try:
        summary = _summarize(pages, combined, query)
    except Exception as exc:
        log.exception("summarisation failed")
        yield ProgressEvent("error", f"Summarisation failed: {exc}", 0)
        return
    if not summary:
        yield ProgressEvent("error", "Could not produce an answer from those pages.", 0)
        return

    # --- cache write ---
    yield ProgressEvent("caching", "Saving...", 95)
    entry_id = cache.add_to_cache(
        query, summary,
        volatility=verdict.volatility,
        ttl_seconds=verdict.ttl_seconds,
        source_urls=[{'url': p.url, 'title': p.title} for p in pages],
        router_source=verdict.source,
        embedding=embedding,
    )

    yield ProgressEvent(
        "complete", "Done", 100,
        {"summary": summary, "is_cached": False, "sources": sources,
         "pages_scraped": len(pages), "total_content_length": len(combined),
         "verdict": asdict(verdict), "cached_as": entry_id,
         "cache": dict(cache_trail, stored=entry_id is not None)},
    )


def _summarize(pages, combined: str, query: str) -> Optional[str]:
    """Summarise with the configured backend, falling back to the other.

    SUMMARIZER=llm is required on serverless, where distilbart plus torch is
    ~1.7GB against a 500MB bundle limit. The fallback runs the other way too,
    so a rate-limited provider degrades to the local model rather than failing
    the request -- when the local model is actually installed.
    """
    if config.SUMMARIZER == "llm":
        import summarize_llm

        answer = summarize_llm.summarize(pages, query)
        if answer:
            return answer
        log.info("llm summariser unavailable; trying the local model")
        try:
            from summarizer import summarize_text

            return summarize_text(combined, query)
        except ImportError:
            log.warning("no local summariser installed either")
            return None

    try:
        from summarizer import summarize_text

        return summarize_text(combined, query)
    except ImportError:
        log.info("local summariser not installed; trying the llm")
        import summarize_llm

        return summarize_llm.summarize(pages, query)


def _human(seconds: int) -> str:
    if seconds <= 0:
        return "not at all"
    if seconds < 120:
        return f"{seconds}s"
    if seconds < 7200:
        return f"{seconds // 60}m"
    if seconds < 172800:
        return f"{seconds // 3600}h"
    return f"{seconds // 86400}d"


def run(query: str, force_refresh: bool = False) -> Dict[str, Any]:
    """Collect the whole pipeline into one result. Used by the JSON endpoint."""
    last: Optional[ProgressEvent] = None
    events: List[ProgressEvent] = []
    for event in process_query(query, force_refresh):
        events.append(event)
        last = event
    if last is None:
        return {"error": "pipeline produced no result"}
    if last.stage == "error":
        return {"error": last.message, **last.data}
    return last.to_dict()
