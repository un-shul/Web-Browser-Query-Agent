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
from summarizer import summarize_text
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
    source: str = ""  # heuristic:<rule> | lr | llm:<provider> | default
    p_valid: Optional[float] = None
    llm_calls: int = 0


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


def classify(query: str, embedding=None) -> QueryVerdict:
    """Decide validity and volatility.

    Deliberately ordered so the common cases cost nothing. Once the LLM router
    lands it slots in after step 3, and everything before it stays free.
    """
    # 1. Cheap character-level gates. No model, no embedding.
    stripped = (query or "").strip()
    if len(stripped) < 3 or len(stripped) > 400:
        return QueryVerdict(is_valid=False, reason="query length out of range", source="gate")
    if not any(ch.isalpha() for ch in stripped):
        return QueryVerdict(is_valid=False, reason="no alphabetic characters", source="gate")

    # 2. Logistic-regression gate. See agent.is_junk for why the threshold is
    #    0.05 and not the model's own 0.5 boundary.
    label, p_valid = agent.classify_query_with_confidence(stripped, embedding)
    if p_valid is not None and p_valid < config.LR_REJECT_P:
        return QueryVerdict(
            is_valid=False,
            reason=f"classifier confident this is not a query (p={p_valid:.3f})",
            source="lr",
            p_valid=p_valid,
        )

    # 3. Deterministic volatility heuristics. Free, and the escalation floor
    #    for whatever the router later says.
    guess, rule = vp.heuristic_volatility(stripped)
    volatility = guess or vp.DEFAULT_VOLATILITY
    source = f"heuristic:{rule}" if rule else "default"

    return QueryVerdict(
        is_valid=True,
        volatility=volatility,
        ttl_seconds=vp.ttl_for(volatility),
        search_topic="news" if volatility in (vp.DYNAMIC, vp.REALTIME) else "general",
        time_range="day" if volatility == vp.REALTIME else None,
        reason=f"matched {rule}" if rule else "no heuristic matched; using default",
        source=source,
        p_valid=p_valid,
    )


def lookup_cache(query: str, verdict: QueryVerdict, embedding=None) -> Dict[str, Any]:
    """Find a reusable cached answer, or explain why there isn't one.

    Returns an audit trail rather than just a hit/miss, so the UI can show why
    a decision was made.
    """
    trail: Dict[str, Any] = {
        "hit": False, "summary": None, "similarity": None, "candidates": [],
        "expired_skipped": 0, "decision": "", "entry_id": None, "sources": [],
    }

    # A realtime query never reuses another query's answer, however similar.
    # Only its own short-TTL entry can serve it, which is what makes this
    # structural rather than a threshold to tune.
    if verdict.volatility == vp.REALTIME:
        fresh, expired = cache.find_similar_candidates(
            query, k=1, floor=0.98, embedding=embedding
        )
        trail["expired_skipped"] = len(expired)
        if fresh:
            trail.update(
                hit=True, summary=fresh[0].summary, similarity=fresh[0].similarity,
                entry_id=fresh[0].id, sources=fresh[0].source_urls,
                decision="exact realtime re-ask inside its 90s window",
            )
        else:
            trail["decision"] = "realtime query: cache bypassed"
        return trail

    fresh, expired = cache.find_similar_candidates(query, embedding=embedding)
    trail["expired_skipped"] = len(expired)
    trail["candidates"] = [
        {"query": c.query, "similarity": round(c.similarity, 4),
         "volatility": c.volatility, "age_seconds": c.age_seconds}
        for c in fresh
    ]

    # Upgrade any legacy entry we touched, using heuristics only.
    for candidate in list(fresh) + list(expired):
        if candidate.schema_version < cache.SCHEMA_VERSION:
            cache.get_cache().backfill(candidate)

    if not fresh:
        trail["decision"] = (
            f"no fresh candidate above {config.CANDIDATE_FLOOR:.2f}"
            + (f" ({len(expired)} expired)" if expired else "")
        )
        return trail

    best = fresh[0]
    # Without an LLM verifier this is the legacy threshold, so degraded mode is
    # never worse than the pre-router app. The reranker replaces this branch.
    if best.similarity >= config.LEGACY_SIM_THRESHOLD:
        cache.touch(best.id)
        trail.update(
            hit=True, summary=best.summary, similarity=best.similarity,
            entry_id=best.id, sources=best.source_urls,
            decision=f"similarity {best.similarity:.3f} >= {config.LEGACY_SIM_THRESHOLD:.2f}",
        )
    else:
        trail["decision"] = (
            f"best similarity {best.similarity:.3f} below "
            f"{config.LEGACY_SIM_THRESHOLD:.2f}"
        )
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
                 "expired_skipped": 0, "sources": []}
        yield ProgressEvent("cache", "Bypassing cache (force refresh)", 18, {"cache": trail})
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

    # --- search ---
    yield ProgressEvent("searching", "Searching the web...", 25)
    try:
        bundle = search(query, topic=verdict.search_topic, time_range=verdict.time_range)
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
        summary = summarize_text(combined, query)
    except Exception as exc:
        log.exception("summarisation failed")
        yield ProgressEvent("error", f"Summarisation failed: {exc}", 0)
        return

    # --- cache write ---
    yield ProgressEvent("caching", "Saving...", 95)
    entry_id = cache.add_to_cache(
        query, summary,
        volatility=verdict.volatility,
        ttl_seconds=verdict.ttl_seconds,
        source_urls=[p.url for p in pages],
        router_source=verdict.source,
        embedding=embedding,
    )

    yield ProgressEvent(
        "complete", "Done", 100,
        {"summary": summary, "is_cached": False, "sources": sources,
         "pages_scraped": len(pages), "total_content_length": len(combined),
         "verdict": asdict(verdict), "cached_as": entry_id,
         "cache": {"hit": False, "stored": entry_id is not None}},
    )


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
