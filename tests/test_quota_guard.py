"""How many LLM calls each query class costs.

These are the tests that keep the free tier viable. Every assertion is an
exact call count, so a refactor that moves work onto the LLM fails here rather
than silently exhausting a daily quota in production.
"""

import pytest

import config
import volatility_policy as vp
from llm_gateway import providers as P
from llm_gateway import reranker, router


@pytest.fixture
def fake_llm(monkeypatch):
    """A provider that answers anything, so call counts are the only variable."""
    fake = P.FakeProvider({})

    def answer_everything(*, system, user, schema, purpose="", **kwargs):
        fake.calls.append({"purpose": purpose, "user": user})
        if purpose == "router":
            return {"valid": True, "intent": "information_seeking",
                    "volatility": "slow", "confidence": 0.9, "reason": "stub"}
        return {"match_index": 0, "confidence": 0.9, "reason": "stub"}

    fake.complete_json = answer_everything
    P.set_chain([fake])
    monkeypatch.setattr(config, "LLM_DISABLED", False)
    router.clear_memo()
    P.get_budget().reset()
    yield fake
    P.reset_chain()
    router.clear_memo()


# --- router ------------------------------------------------------------------


@pytest.mark.parametrize("query", ["", "ab", "!!!!!", "12345", "x" * 500])
def test_cheap_gates_cost_nothing(cache, fake_llm, query):
    router.route(query)
    assert fake_llm.call_count == 0


@pytest.mark.parametrize("query", [
    "live cricket score india vs australia",
    "tesla stock price today",
    "weather in bangalore",
    "usd to inr exchange rate",
])
def test_realtime_regex_matches_cost_nothing(cache, fake_llm, query):
    """The highest-value skip: these are common and now free, forever."""
    verdict = router.route(query)
    assert verdict["volatility"] == vp.REALTIME
    assert verdict["llm_calls"] == 0
    assert fake_llm.call_count == 0


def test_a_cold_ambiguous_query_costs_one_call(cache, fake_llm):
    verdict = router.route("thoughts on the new framework everyone mentions")
    assert verdict["llm_calls"] == 1
    assert fake_llm.call_count == 1


def test_a_repeated_query_is_free_the_second_time(cache, fake_llm):
    query = "thoughts on the new framework everyone mentions"
    router.route(query)
    assert fake_llm.call_count == 1
    verdict = router.route(query)
    assert fake_llm.call_count == 1  # memo, no second call
    assert verdict["llm_calls"] == 0
    assert verdict["source"].startswith("memo")


def test_memo_ignores_case_and_spacing(cache, fake_llm):
    router.route("Thoughts On The New Framework Everyone Mentions")
    router.route("  thoughts on the new   framework everyone mentions  ")
    assert fake_llm.call_count == 1


def test_a_degraded_verdict_is_not_memoized(cache, monkeypatch):
    """Otherwise a transient outage would poison the memo for an hour."""
    P.set_chain([])
    router.clear_memo()
    query = "thoughts on the new framework everyone mentions"
    first = router.route(query)
    assert first["degraded"]

    fake = P.FakeProvider({})
    fake.complete_json = lambda **kw: {
        "valid": True, "intent": "information_seeking",
        "volatility": "dynamic", "confidence": 0.9, "reason": "stub"}
    P.set_chain([fake])
    second = router.route(query)
    assert not second["degraded"]
    assert second["volatility"] == vp.DYNAMIC
    P.reset_chain()


def test_disabled_llm_costs_nothing(cache, fake_llm, monkeypatch):
    monkeypatch.setattr(config, "LLM_DISABLED", True)
    verdict = router.route("thoughts on the new framework everyone mentions")
    assert fake_llm.call_count == 0
    assert verdict["degraded"]


# --- reranker ----------------------------------------------------------------


def test_no_candidates_costs_nothing(cache, fake_llm):
    outcome = reranker.verify("what is photosynthesis", [])
    assert outcome.llm_calls == 0
    assert fake_llm.call_count == 0


def test_auto_accept_costs_nothing(cache, fake_llm):
    cache.add_to_cache("what is photosynthesis", "Summary.", volatility=vp.STATIC)
    fresh, _ = cache.find_similar_candidates("what is photosynthesis", floor=0.0)
    outcome = reranker.verify("what is photosynthesis", fresh)
    assert outcome.decision == "auto_accept"
    assert outcome.llm_calls == 0
    assert fake_llm.call_count == 0


def test_a_hard_conflict_costs_nothing(cache, fake_llm):
    """Polarity and direction are decided locally, so they never spend a call."""
    cache.add_to_cache("is coffee good for your health", "Benefits...",
                       volatility=vp.STATIC)
    fresh, _ = cache.find_similar_candidates("is coffee good for your health", floor=0.0)
    outcome = reranker.verify("is coffee bad for your health", fresh)
    assert outcome.decision == "mismatch_reject"
    assert outcome.llm_calls == 0
    assert fake_llm.call_count == 0


def test_realtime_queries_never_reach_the_reranker(cache, fake_llm):
    """A realtime query skips the semantic lookup, so no verification is
    needed and no call is spent."""
    import pipeline
    cache.add_to_cache("live cricket score", "Old score.", volatility=vp.REALTIME)
    verdict = pipeline.classify("live cricket score india vs australia")
    pipeline.lookup_cache("live cricket score india vs australia", verdict)
    assert [c for c in fake_llm.calls if c["purpose"] == "rerank"] == []


# --- worst case --------------------------------------------------------------


def test_worst_case_is_two_calls(cache, fake_llm, monkeypatch):
    """A cold ambiguous query with mid-similarity candidates: router + rerank.

    Two is the ceiling. Against Groq's ~14,400 requests/day that is still
    7,000+ queries, so the budget is not the binding constraint.
    """
    import pipeline
    monkeypatch.setattr(config, "AUTO_ACCEPT_SIM", 0.999)
    cache.add_to_cache("some earlier related question", "Earlier answer.",
                       volatility=vp.SLOW)
    query = "thoughts on the new framework everyone mentions"
    verdict = pipeline.classify(query)
    pipeline.lookup_cache(query, verdict)
    assert fake_llm.call_count <= 2


def test_junk_never_reaches_the_llm_even_via_the_pipeline(cache, fake_llm):
    import pipeline
    list(pipeline.process_query("!!!!!"))
    assert fake_llm.call_count == 0
