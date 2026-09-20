"""LLM query router: validity and volatility in one call.

Bundling both into one call is the single biggest quota decision here. Asking
separately would double the spend for no extra information.

Everything cheap runs first, and each early exit is a call not spent:

    cheap gates      ->  0 calls   length, charset
    memo             ->  0 calls   this query was routed recently
    classifier       ->  0 calls   confidently junk
    realtime regex   ->  0 calls   "live score", "stock price", "weather now"
    router           ->  1 call    everything else

The regex result is also kept as an escalation floor, so the LLM can raise
volatility but never lower it below what the heuristics detected.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Optional, Sequence, Tuple

from queryagent import classifier
from queryagent import config
from queryagent import safety
from queryagent import volatility as vp

from . import prompts
from .providers import call_json
from .schemas import ROUTER_SCHEMA

log = logging.getLogger(__name__)

MEMO_TTL_SECONDS = 3600
MEMO_MAX_ENTRIES = 512

_memo: Dict[str, Tuple[float, dict]] = {}
_memo_lock = threading.Lock()


def normalize(query: str) -> str:
    return " ".join((query or "").strip().lower().split())


def _memo_get(key: str) -> Optional[dict]:
    with _memo_lock:
        entry = _memo.get(key)
        if not entry:
            return None
        stored_at, value = entry
        if time.time() - stored_at > MEMO_TTL_SECONDS:
            _memo.pop(key, None)
            return None
        return dict(value)


def _memo_put(key: str, value: dict) -> None:
    with _memo_lock:
        if len(_memo) >= MEMO_MAX_ENTRIES:
            oldest = min(_memo, key=lambda k: _memo[k][0])
            _memo.pop(oldest, None)
        _memo[key] = (time.time(), dict(value))


def clear_memo() -> None:
    with _memo_lock:
        _memo.clear()


def route(
    query: str,
    embedding: Optional[Sequence[float]] = None,
    allow_llm: bool = True,
) -> dict:
    """Classify a query.

    Returns a plain dict so pipeline.QueryVerdict can absorb it without this
    module importing the pipeline.
    """
    stripped = (query or "").strip()
    key = normalize(stripped)

    # 1. Cheap gates. No model, no embedding, no network.
    if len(stripped) < 3 or len(stripped) > 400:
        return _invalid("query length out of range", "gate")
    if not any(ch.isalpha() for ch in stripped):
        return _invalid("no alphabetic characters", "gate")

    # 2. Refusal, deterministic. Before the memo deliberately: a refused
    #    query must not be memoised, embedded, searched or stored, and this
    #    check costs nothing.
    allowed, category, message = safety.check(stripped)
    if not allowed:
        log.info("refused a query (%s)", category)  # text deliberately not logged
        return _refused(category, message, "safety")

    # 3. Memo. A repeat inside the hour is free.
    cached = _memo_get(key)
    if cached:
        cached["source"] = f"memo({cached.get('source', '')})"
        cached["llm_calls"] = 0
        return cached

    # 4. Classifier gate. Rejects junk before any LLM call. The 0.05 threshold
    #    rather than the model's own 0.5 boundary -- see classifier.is_junk.
    label, p_valid = classifier.classify_query_with_confidence(stripped, embedding)
    if p_valid is not None and p_valid < config.LR_REJECT_P:
        return _invalid(
            f"classifier confident this is not a query (p={p_valid:.3f})",
            "lr", p_valid=p_valid,
        )

    # 5. Deterministic volatility. A realtime match settles it outright: this
    #    is the highest-value skip, since "live score" and "stock price" are
    #    common and now cost nothing, forever.
    guess, rule = vp.heuristic_volatility(stripped)
    if guess == vp.REALTIME:
        verdict = _build(vp.REALTIME, f"heuristic:{rule}", f"matched {rule}", p_valid=p_valid)
        _memo_put(key, verdict)
        return verdict

    # 6. Router. One call.
    llm_result = None
    if allow_llm and not config.LLM_DISABLED:
        llm_result, meta = call_json(
            prompts.ROUTER_SYSTEM,
            prompts.router_user_prompt(stripped),
            ROUTER_SCHEMA,
            purpose="router",
            max_output_tokens=300,
        )
        if llm_result is None:
            log.info("router degraded: %s", meta.error)

    if llm_result is None:
        # No LLM: keep the heuristic if it had an opinion, else default.
        volatility = guess or vp.DEFAULT_VOLATILITY
        verdict = _build(
            volatility,
            f"heuristic:{rule}" if rule else "default",
            f"matched {rule}" if rule else "no heuristic matched; using default",
            p_valid=p_valid, degraded=True,
        )
        # Deliberately not memoized: a degraded verdict would then be reused
        # for an hour even after the LLM came back.
        return verdict

    if llm_result.get("refuse") is True:
        category = str(llm_result.get("refuse_category") or "other")
        if category == "none":
            category = "other"
        log.info("router refused a query (%s)", category)
        return _refused(category, safety.message_for(category), "llm", llm_calls=1)

    intent = str(llm_result.get("intent", "information_seeking"))
    if not llm_result.get("valid", True) or intent in {"navigation", "command", "gibberish"}:
        return _invalid(
            str(llm_result.get("reason") or f"classified as {intent}"),
            "llm", p_valid=p_valid, llm_calls=1,
        )

    # Escalation: the regex floor can raise volatility, never lower it. This
    # is what makes "router calls a live-score query static" harmless.
    volatility = vp.escalate(llm_result.get("volatility"), guess)
    verdict = _build(
        volatility,
        "llm",
        str(llm_result.get("reason", ""))[:300],
        ttl_hint=llm_result.get("ttl_hint_seconds"),
        topic=llm_result.get("topic"),
        search_query=llm_result.get("search_query"),
        confidence=llm_result.get("confidence"),
        p_valid=p_valid,
        llm_calls=1,
        escalated_from=llm_result.get("volatility"),
        heuristic_floor=guess,
    )
    _memo_put(key, verdict)
    return verdict


def _build(
    volatility: str,
    source: str,
    reason: str,
    ttl_hint=None,
    topic=None,
    search_query=None,
    confidence=None,
    p_valid=None,
    llm_calls: int = 0,
    degraded: bool = False,
    escalated_from=None,
    heuristic_floor=None,
) -> dict:
    volatility = vp.normalize_volatility(volatility)
    resolved_topic = topic if topic in {"general", "news"} else (
        "news" if volatility in (vp.DYNAMIC, vp.REALTIME) else "general"
    )
    return {
        "is_valid": True,
        "volatility": volatility,
        "ttl_seconds": vp.ttl_for(volatility, ttl_hint),
        "search_topic": resolved_topic,
        "time_range": "day" if volatility == vp.REALTIME else None,
        "search_query": search_query or None,
        "reason": reason,
        "source": source,
        "confidence": float(confidence) if isinstance(confidence, (int, float)) else None,
        "p_valid": p_valid,
        "llm_calls": llm_calls,
        "degraded": degraded,
        "escalated_from": escalated_from if escalated_from != volatility else None,
        "heuristic_floor": heuristic_floor,
        "refused": False, "refuse_category": None,
    }


def _refused(category: Optional[str], message: str, source: str,
             llm_calls: int = 0) -> dict:
    """A query the agent will not answer.

    Distinct from invalid: an invalid query is malformed, a refused one is
    understood and declined. The pipeline must not search or cache either, but
    only this one carries a message written for the user to read.
    """
    verdict = _invalid(message, f"refused:{source}", llm_calls=llm_calls)
    verdict["refused"] = True
    verdict["refuse_category"] = category
    return verdict


def _invalid(reason: str, source: str, p_valid=None, llm_calls: int = 0) -> dict:
    return {
        "is_valid": False, "volatility": vp.DEFAULT_VOLATILITY, "ttl_seconds": 0,
        "search_topic": "general", "time_range": None, "search_query": None,
        "reason": reason, "source": source, "confidence": None,
        "p_valid": p_valid, "llm_calls": llm_calls, "degraded": False,
        "escalated_from": None, "heuristic_floor": None,
        "refused": False, "refuse_category": None,
    }
