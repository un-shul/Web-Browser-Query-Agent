"""How fast does an answer go stale, and how long may it be cached?

Pure, deterministic, and dependency-free on purpose. This module must work
with no LLM, no network, and no model, because it is the fallback the whole
caching layer degrades into when nothing else is reachable.

Four classes, ordered by how quickly the correct answer changes:

    static    an answer that does not change      "what is photosynthesis"
    slow      changes over weeks or months        "best laptops 2026"
    dynamic   changes over hours                  "latest news on X"
    realtime  changes continuously                "live cricket score"

The router may propose a class and a TTL hint, but this module owns the final
TTL, and `escalate` guarantees the regex heuristics can only ever raise
volatility, never lower it. That is what makes the worst failure -- a model
labelling "live cricket score" as static -- structurally impossible rather
than merely unlikely.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

STATIC = "static"
SLOW = "slow"
DYNAMIC = "dynamic"
REALTIME = "realtime"
UNKNOWN = "unknown"  # only for cache entries written before TTLs existed

VOLATILITIES = (STATIC, SLOW, DYNAMIC, REALTIME)

# Ordering used by escalate(). UNKNOWN is deliberately absent: it is a storage
# state, not a routing decision.
ORDER: Dict[str, int] = {STATIC: 0, SLOW: 1, DYNAMIC: 2, REALTIME: 3}

DAY = 86400

# min/max bound whatever the model suggests; default applies when it suggests
# nothing usable.
TTL_POLICY: Dict[str, Dict[str, int]] = {
    STATIC:   {"default": 180 * DAY, "min": 7 * DAY, "max": 365 * DAY},
    SLOW:     {"default": 14 * DAY,  "min": DAY,     "max": 60 * DAY},
    DYNAMIC:  {"default": 6 * 3600,  "min": 900,     "max": DAY},
    # Short but non-zero: long enough to absorb a double-click or an
    # EventSource reconnect, too short to serve anything meaningfully stale.
    REALTIME: {"default": 90,        "min": 0,       "max": 300},
    UNKNOWN:  {"default": 3 * DAY,   "min": 3600,    "max": 7 * DAY},
}

DEFAULT_VOLATILITY = SLOW

# Written to metadata instead of a null TTL, since Chroma metadata cannot hold
# None. 2100-01-01.
NEVER_SENTINEL = 4102444800


# --- Heuristics --------------------------------------------------------------
# Ordered: the first match wins, so the most specific patterns come first.
# Each entry is (rule_name, volatility, pattern).

_RULES: Tuple[Tuple[str, str, re.Pattern], ...] = (
    # --- realtime ---
    ("live_event", REALTIME, re.compile(
        r"\b(live|in[- ]play)\b.*\b(score|scores|match|game|stream|updates?|results?)\b"
        r"|\b(score|scorecard)\b.*\blive\b"
        r"|\blive\s+(score|scorecard|cricket|football|match)\b", re.I)),
    ("scorecard", REALTIME, re.compile(
        r"\b(scorecard|score\s*card|full\s*score|current\s*score|latest\s*score)\b"
        r"|\b(vs|v/s)\b.*\bscore\b", re.I)),
    ("market_price", REALTIME, re.compile(
        r"\b(stock|share|crypto|bitcoin|btc|eth|ethereum|forex|nifty|sensex|nasdaq)\b"
        r".*\b(price|quote|rate|value|today|now)\b"
        r"|\b(price|rate)\s+of\b.*\b(gold|silver|bitcoin|crude|oil|dollar|rupee)\b"
        r"|\bexchange\s+rate\b|\bstock\s+price\b", re.I)),
    ("weather_now", REALTIME, re.compile(
        r"\b(weather|temperature|forecast|aqi|air\s+quality|rainfall|humidity)\b"
        r"(?!.*\b(climate|average|historical|annual|typical)\b)", re.I)),
    ("right_now", REALTIME, re.compile(
        r"\b(right\s+now|at\s+the\s+moment|as\s+of\s+now|currently\s+happening"
        r"|happening\s+now|trending\s+now|live\s+updates?)\b", re.I)),
    ("traffic_transit", REALTIME, re.compile(
        r"\b(traffic|flight\s+status|train\s+status|delay(ed)?|departures?|arrivals?)\b", re.I)),

    # --- dynamic ---
    ("news", DYNAMIC, re.compile(
        r"\b(news|headlines?|breaking|announced?|announcement"
        r"|press\s+release|just\s+released)\b", re.I)),
    ("recency_word", DYNAMIC, re.compile(
        r"\b(latest|newest|recent|today|tonight|yesterday|this\s+(week|morning|month)"
        r"|so\s+far|up\s*to\s*date|current)\b", re.I)),
    ("ongoing_event", DYNAMIC, re.compile(
        r"\b(election\s+results?|poll\s+results?|winner\s+of|who\s+won"
        r"|release\s+date|launch\s+date)\b", re.I)),

    # --- slow ---
    ("recommendation", SLOW, re.compile(
        r"\b(best|top|cheapest|fastest|recommended|should\s+i|worth\s+(it|buying)"
        r"|compare|comparison|vs\.?|alternatives?|review)\b"
        r"|\bis\b.{0,30}\ba\s+good\b", re.I)),
    ("year_ref", SLOW, re.compile(r"\b20[2-9][0-9]\b")),

    # --- static ---
    ("definition", STATIC, re.compile(
        r"^\s*(what\s+(is|are|was|were|does|do)\b|who\s+(is|was|were)\b"
        r"|define\b|definition\s+of\b|meaning\s+of\b|how\s+(does|do|did)\b"
        r"|why\s+(is|are|does|do|did)\b|explain\b"
        r"|when\s+(was|were|did)\b)", re.I)),
    ("historical", STATIC, re.compile(
        r"\b(history\s+of|invented|discovered|founded|born\s+in|died\s+in"
        r"|in\s+18\d\d|in\s+19\d\d|in\s+20[01]\d)\b", re.I)),
)


_YEAR = re.compile(r"\b(1[89]\d\d|20\d\d)\b")
_RESULT_WORD = re.compile(
    r"\b(won|winner|champions?|result|results|final|finals|held|hosted"
    r"|score|scores|scorecard)\b",
    re.I,
)


def _settled_past(query: str) -> bool:
    """True for a resolved outcome in a year that has already ended.

    "who won the 2011 cricket world cup" is as static as a definition. The
    current year is excluded: "who won the 2026 election" may still be live.
    """
    if not _RESULT_WORD.search(query):
        return False
    this_year = datetime.now(timezone.utc).year
    return any(int(y) < this_year for y in _YEAR.findall(query))


def heuristic_volatility(query: str) -> Tuple[Optional[str], Optional[str]]:
    """Classify by regex alone.

    Returns (volatility, rule_name), or (None, None) when nothing matches.
    A None result is not a failure -- it means "ask the router", and the caller
    uses it as the escalation floor either way.
    """
    if not query:
        return None, None
    if _settled_past(query):
        return STATIC, "settled_past"
    for rule, volatility, pattern in _RULES:
        if pattern.search(query):
            return volatility, rule
    return None, None


# --- Policy ------------------------------------------------------------------


def normalize_volatility(value: Optional[str]) -> str:
    """Coerce anything a model might return into a known class."""
    if isinstance(value, str) and value.strip().lower() in ORDER:
        return value.strip().lower()
    return DEFAULT_VOLATILITY


def escalate(volatility: Optional[str], floor: Optional[str]) -> str:
    """Return the more volatile of the two.

    The invariant the cache depends on: escalate(v, floor) is never less
    volatile than either argument. So a regex that recognises a live-score
    query overrides a model that called it static.
    """
    v = normalize_volatility(volatility)
    if floor is None or floor not in ORDER:
        return v
    return v if ORDER[v] >= ORDER[floor] else floor


def ttl_for(volatility: str, hint_seconds: Optional[int] = None) -> int:
    """Resolve a TTL in seconds.

    The hint is advisory. Models are inconsistent at arithmetic and this value
    decides whether stale answers get served, so it is always clamped into the
    band for its class.
    """
    key = volatility if volatility in TTL_POLICY else DEFAULT_VOLATILITY
    policy = TTL_POLICY[key]
    if hint_seconds is None:
        return policy["default"]
    try:
        hint = int(hint_seconds)
    except (TypeError, ValueError):
        return policy["default"]
    if hint < 0:
        return policy["default"]
    return max(policy["min"], min(policy["max"], hint))


def expires_at(created_at: int, ttl_seconds: int) -> int:
    """Absolute expiry, using the sentinel for effectively-never."""
    if ttl_seconds <= 0:
        return created_at
    horizon = created_at + ttl_seconds
    return min(horizon, NEVER_SENTINEL)


def is_cacheable(volatility: str) -> bool:
    return ttl_for(volatility) > 0
