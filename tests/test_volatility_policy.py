"""Tests for the deterministic volatility policy.

No network, no model, no LLM -- this module is the fallback everything else
degrades into, so its tests must run anywhere.
"""

from datetime import datetime, timezone

import pytest

from queryagent import volatility as vp

REALTIME_QUERIES = [
    "live cricket score india vs australia",
    "ind vs aus scorecard",
    "tesla stock price today",
    "bitcoin price now",
    "price of gold",
    "weather in bangalore",
    "delhi aqi",
    "what is happening right now in gaza",
    "flight status AI302",
    "usd to inr exchange rate",
    "nifty today",
    "live updates on the hurricane",
]

DYNAMIC_QUERIES = [
    "latest ai news",
    "breaking news india",
    "who won the election results",
    "gta 6 release date",
    "recent developments in fusion energy",
    "headlines today",
]

SLOW_QUERIES = [
    "best laptops for programming",
    "react vs vue",
    "is tesla a good investment",
    "top python web frameworks",
    "iphone 17 review",
    "best laptops 2026",
    "should i learn rust",
]

STATIC_QUERIES = [
    "what is photosynthesis",
    "who was alan turing",
    "define entropy",
    "how does the python GIL work",
    "why is the sky blue",
    "history of the roman empire",
    "who won the 2011 cricket world cup",
    "who won the 2019 world cup final",
    "when was the eiffel tower built",
]


@pytest.mark.parametrize("query", REALTIME_QUERIES)
def test_realtime_queries(query):
    assert vp.heuristic_volatility(query)[0] == vp.REALTIME


@pytest.mark.parametrize("query", DYNAMIC_QUERIES)
def test_dynamic_queries(query):
    assert vp.heuristic_volatility(query)[0] == vp.DYNAMIC


@pytest.mark.parametrize("query", SLOW_QUERIES)
def test_slow_queries(query):
    assert vp.heuristic_volatility(query)[0] == vp.SLOW


@pytest.mark.parametrize("query", STATIC_QUERIES)
def test_static_queries(query):
    assert vp.heuristic_volatility(query)[0] == vp.STATIC


@pytest.mark.parametrize("query", REALTIME_QUERIES)
def test_realtime_never_downgraded(query):
    """The property the cache depends on.

    A realtime query must never be classified as anything less volatile, since
    that is exactly what would let a stale answer be served.
    """
    got, _ = vp.heuristic_volatility(query)
    assert vp.ORDER[got] >= vp.ORDER[vp.REALTIME]


# --- escalation --------------------------------------------------------------


@pytest.mark.parametrize("a", vp.VOLATILITIES)
@pytest.mark.parametrize("b", vp.VOLATILITIES)
def test_escalate_is_monotonic(a, b):
    """escalate() may only ever raise volatility, never lower it."""
    result = vp.escalate(a, b)
    assert vp.ORDER[result] == max(vp.ORDER[a], vp.ORDER[b])


def test_escalate_overrides_a_wrong_model_verdict():
    # The highest-severity failure: the router calls a live-score query static.
    assert vp.escalate(vp.STATIC, vp.REALTIME) == vp.REALTIME


def test_escalate_with_no_floor_passes_through():
    assert vp.escalate(vp.STATIC, None) == vp.STATIC


@pytest.mark.parametrize("bogus", ["weekly", "", "REALTIME ", None, 7, [], "static "])
def test_normalize_coerces_bad_input(bogus):
    assert vp.normalize_volatility(bogus) in vp.ORDER


def test_normalize_accepts_case_and_whitespace():
    assert vp.normalize_volatility("  ReAlTiMe ") == vp.REALTIME


def test_escalate_with_unknown_floor_ignores_it():
    # "unknown" is a storage state, not a routing class, so it is not a floor.
    assert vp.escalate(vp.STATIC, vp.UNKNOWN) == vp.STATIC


# --- TTL ---------------------------------------------------------------------


@pytest.mark.parametrize("volatility", vp.VOLATILITIES)
def test_ttl_default_within_band(volatility):
    policy = vp.TTL_POLICY[volatility]
    assert policy["min"] <= vp.ttl_for(volatility) <= policy["max"]


@pytest.mark.parametrize("volatility", vp.VOLATILITIES)
def test_ttl_clamps_absurdly_large_hint(volatility):
    assert vp.ttl_for(volatility, 10**12) == vp.TTL_POLICY[volatility]["max"]


@pytest.mark.parametrize("volatility", vp.VOLATILITIES)
def test_ttl_clamps_small_hint_up_to_min(volatility):
    assert vp.ttl_for(volatility, 1) >= vp.TTL_POLICY[volatility]["min"]


@pytest.mark.parametrize("hint", [None, -1, -99999, "abc", "", [], {}, 3.7])
def test_ttl_survives_unusable_hints(hint):
    # A model returning junk must not produce a junk TTL.
    assert vp.ttl_for(vp.DYNAMIC, hint) > 0


def test_ttl_honours_a_reasonable_hint():
    assert vp.ttl_for(vp.DYNAMIC, 3600) == 3600


def test_ttl_for_unknown_class_falls_back_to_default():
    assert vp.ttl_for("nonsense") == vp.TTL_POLICY[vp.DEFAULT_VOLATILITY]["default"]


def test_realtime_ttl_is_short_but_nonzero():
    """Long enough to absorb an EventSource reconnect, short enough to be safe."""
    ttl = vp.ttl_for(vp.REALTIME)
    assert 0 < ttl <= 300


def test_static_ttl_is_much_longer_than_dynamic():
    assert vp.ttl_for(vp.STATIC) > 100 * vp.ttl_for(vp.DYNAMIC)


# --- expiry ------------------------------------------------------------------


def test_expires_at_adds_ttl():
    assert vp.expires_at(1_000_000, 3600) == 1_003_600


def test_expires_at_with_zero_ttl_is_immediately_expired():
    assert vp.expires_at(1_000_000, 0) == 1_000_000


def test_expires_at_is_capped_by_sentinel():
    assert vp.expires_at(1_000_000, 10**15) == vp.NEVER_SENTINEL


def test_is_cacheable():
    assert vp.is_cacheable(vp.STATIC)
    assert vp.is_cacheable(vp.REALTIME)  # briefly, by design


# --- settled-past detection --------------------------------------------------


def test_current_year_result_is_not_treated_as_settled():
    """This year's outcomes may still be in flux, so they stay dynamic."""
    year = datetime.now(timezone.utc).year
    got, _ = vp.heuristic_volatility(f"who won the {year} election")
    assert got == vp.DYNAMIC


def test_past_year_result_is_settled():
    got, rule = vp.heuristic_volatility("who won the 2011 cricket world cup")
    assert got == vp.STATIC
    assert rule == "settled_past"


def test_settled_past_needs_a_result_word():
    # A bare past year is not enough; "best laptops 2019" is still a
    # recommendation, not a settled outcome.
    assert not vp._settled_past("best laptops 2019")


# --- edge cases --------------------------------------------------------------


@pytest.mark.parametrize("query", ["", "   ", None])
def test_empty_query_returns_no_match(query):
    assert vp.heuristic_volatility(query) == (None, None)


def test_unmatched_query_returns_none_rather_than_guessing():
    """No match means 'ask the router', which is different from a default."""
    got, rule = vp.heuristic_volatility("zorblax quimbly")
    assert got is None and rule is None


def test_climate_is_not_mistaken_for_weather():
    """The weather rule must not fire on historical climate questions."""
    got, _ = vp.heuristic_volatility("climate of delhi")
    assert got != vp.REALTIME
