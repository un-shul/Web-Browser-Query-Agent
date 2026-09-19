"""Provider chain, degradation, and quota discipline.

No network: every provider is stubbed or replayed from a cassette. The quota
tests are the important ones -- they pin how many LLM calls each query class
costs, so a later refactor cannot quietly start burning the free tier.
"""

import json

import pytest
import responses

from queryagent import config
from queryagent.llm import providers as P
from queryagent.llm import reranker, router
from queryagent.llm.budget import Budget
from queryagent.llm.schemas import RERANK_SCHEMA, ROUTER_SCHEMA

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    P.reset_chain()
    router.clear_memo()
    P.get_budget().reset()
    monkeypatch.setattr(config, "LLM_DISABLED", False)
    yield
    P.reset_chain()
    router.clear_memo()
    P.get_budget().reset()


def _groq_only(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
    P.reset_chain()


def _groq_body(payload: dict) -> dict:
    return {"choices": [{"message": {"content": json.dumps(payload)}}]}


# --- JSON extraction ---------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ('{"ok": true}', {"ok": True}),
    ('```json\n{"ok": true}\n```', {"ok": True}),
    ('```\n{"ok": true}\n```', {"ok": True}),
    ('Sure! {"ok": true} hope that helps', {"ok": True}),
    ('  \n {"ok": true}  \n ', {"ok": True}),
    ('{"nested": {"a": 1}, "ok": true}', {"nested": {"a": 1}, "ok": True}),
    ('{"text": "a } brace in a string", "ok": true}',
     {"text": "a } brace in a string", "ok": True}),
])
def test_extract_json_handles_common_wrappers(raw, expected):
    assert P.extract_json(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "no json here", "[1,2,3]", "{unterminated",
                                 '{"broken": }'])
def test_extract_json_raises_on_unusable_input(raw):
    with pytest.raises(P.LLMBadOutput):
        P.extract_json(raw)


# --- chain construction ------------------------------------------------------


def test_missing_keys_produce_an_empty_chain(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    monkeypatch.setattr(config, "LLM_PROVIDER", "chain")
    P.reset_chain()
    assert P.get_chain() == []


def test_disabled_flag_produces_an_empty_chain(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "x")
    monkeypatch.setattr(config, "LLM_DISABLED", True)
    P.reset_chain()
    assert P.get_chain() == []


def test_chain_order_is_gemini_then_groq(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "x")
    monkeypatch.setattr(config, "GROQ_API_KEY", "y")
    monkeypatch.setattr(config, "LLM_PROVIDER", "chain")
    P.reset_chain()
    assert [p.name for p in P.get_chain()] == ["gemini", "groq"]


def test_unknown_provider_name_is_ignored(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "nonsense")
    P.reset_chain()
    assert P.get_chain() == []


# --- call_json never raises --------------------------------------------------


def test_call_json_with_no_provider_returns_none(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    monkeypatch.setattr(config, "LLM_PROVIDER", "chain")
    P.reset_chain()
    result, meta = P.call_json("s", "u", SCHEMA, purpose="t")
    assert result is None
    assert meta.degraded


@responses.activate
def test_successful_call(monkeypatch):
    _groq_only(monkeypatch)
    responses.add(responses.POST, GROQ_URL, json=_groq_body({"ok": True}), status=200)
    result, meta = P.call_json("s", "u", SCHEMA, purpose="t")
    assert result == {"ok": True}
    assert not meta.degraded and meta.provider == "groq"


@responses.activate
@pytest.mark.parametrize("status", [401, 403, 404, 500, 503])
def test_http_errors_degrade_without_raising(monkeypatch, status):
    _groq_only(monkeypatch)
    responses.add(responses.POST, GROQ_URL, json={"error": "x"}, status=status)
    result, meta = P.call_json("s", "u", SCHEMA, purpose="t")
    assert result is None and meta.degraded


@responses.activate
def test_rate_limit_opens_the_breaker(monkeypatch):
    _groq_only(monkeypatch)
    responses.add(responses.POST, GROQ_URL, json={"error": "rate"}, status=429,
                  headers={"Retry-After": "30"})
    result, meta = P.call_json("s", "u", SCHEMA, purpose="t")
    assert result is None and meta.degraded
    assert "groq" in P.get_budget().stats()["breakers_open"]


@responses.activate
def test_an_open_breaker_prevents_further_requests(monkeypatch):
    _groq_only(monkeypatch)
    responses.add(responses.POST, GROQ_URL, json={"error": "rate"}, status=429,
                  headers={"Retry-After": "60"})
    P.call_json("s", "u", SCHEMA, purpose="t")
    before = len(responses.calls)
    P.call_json("s", "u2", SCHEMA, purpose="t")
    assert len(responses.calls) == before  # no new HTTP request


@responses.activate
def test_rate_limit_is_not_retried_on_the_same_provider(monkeypatch):
    """A 429 means the window is exhausted; retrying wastes the next slot."""
    _groq_only(monkeypatch)
    responses.add(responses.POST, GROQ_URL, json={"error": "rate"}, status=429)
    P.call_json("s", "u", SCHEMA, purpose="t")
    assert len(responses.calls) == 1


@responses.activate
def test_transient_failure_is_retried_once(monkeypatch):
    _groq_only(monkeypatch)
    responses.add(responses.POST, GROQ_URL, json={"error": "boom"}, status=503)
    responses.add(responses.POST, GROQ_URL, json=_groq_body({"ok": True}), status=200)
    result, _ = P.call_json("s", "u", SCHEMA, purpose="t")
    assert result == {"ok": True}


@responses.activate
def test_unparseable_response_degrades(monkeypatch):
    _groq_only(monkeypatch)
    responses.add(responses.POST, GROQ_URL,
                  json={"choices": [{"message": {"content": "sorry, no JSON"}}]}, status=200)
    result, meta = P.call_json("s", "u", SCHEMA, purpose="t")
    assert result is None and meta.degraded


@responses.activate
def test_bad_output_costs_only_one_call(monkeypatch):
    """A repair round-trip would be another request against the free tier."""
    _groq_only(monkeypatch)
    responses.add(responses.POST, GROQ_URL,
                  json={"choices": [{"message": {"content": "not json"}}]}, status=200)
    P.call_json("s", "u", SCHEMA, purpose="t")
    assert len(responses.calls) == 1


@responses.activate
def test_chain_falls_through_to_the_second_provider(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "g")
    monkeypatch.setattr(config, "GROQ_API_KEY", "k")
    monkeypatch.setattr(config, "LLM_PROVIDER", "chain")
    P.reset_chain()
    responses.add(responses.POST,
                  P.GeminiProvider.ENDPOINT.format(model=config.GEMINI_MODEL),
                  json={"error": "quota"}, status=429)
    responses.add(responses.POST, GROQ_URL, json=_groq_body({"ok": True}), status=200)
    result, meta = P.call_json("s", "u", SCHEMA, purpose="t")
    assert result == {"ok": True}
    assert meta.provider == "groq"


# --- budget ------------------------------------------------------------------


def test_budget_blocks_once_rpm_is_reached():
    budget = Budget()
    for _ in range(5):
        budget.record_attempt("p")
    assert not budget.allow("p", rpm=5)
    assert budget.allow("p", rpm=10)


def test_success_closes_the_breaker():
    budget = Budget()
    budget.record_rate_limited("p", 60)
    assert not budget.allow("p")
    budget.record_success("p")
    assert budget.allow("p")


def test_gemini_schema_strips_unsupported_keys():
    cleaned = P._to_gemini_schema({
        "type": "object", "additionalProperties": False, "$schema": "x",
        "properties": {"a": {"type": "string", "minLength": 2}},
        "required": ["a"],
    })
    assert "additionalProperties" not in cleaned
    assert "$schema" not in cleaned
    assert "minLength" not in cleaned["properties"]["a"]


# --- cassette ----------------------------------------------------------------


def test_fake_provider_replays_a_recorded_response():
    key = P.FakeProvider.key("router", "sys", "usr")
    fake = P.FakeProvider({key: {"ok": True}})
    P.set_chain([fake])
    result, _ = P.call_json("sys", "usr", SCHEMA, purpose="router")
    assert result == {"ok": True}
    assert fake.call_count == 1


def test_cassette_miss_does_not_reach_the_network():
    """A miss must fail loudly, or a test could silently start spending quota."""
    fake = P.FakeProvider({})
    P.set_chain([fake])
    result, meta = P.call_json("sys", "usr", SCHEMA, purpose="router")
    assert result is None and meta.degraded
    assert "cassette miss" in meta.error


# --- summariser confidence ---------------------------------------------------


def test_summariser_reports_low_confidence():
    from queryagent.summarize import hosted

    key = P.FakeProvider.key("summarize", hosted.SYSTEM, "x")
    fake = P.FakeProvider({})
    fake.complete_json = lambda **kw: {
        "answer": "The extracts do not answer the question.", "confident": False}
    P.set_chain([fake])

    class Page:
        text = "Some body text. " * 40
        title = "T"
        url = "https://example.test"

    result = hosted.summarize([Page()], "what is photosynthesis")
    assert result is not None
    assert result.confident is False
    assert "do not answer" in result.text


def test_summariser_defaults_to_confident_when_the_flag_is_absent():
    """A model that omits the field should not have its answer discarded."""
    from queryagent.summarize import hosted

    fake = P.FakeProvider({})
    fake.complete_json = lambda **kw: {"answer": "A real answer."}
    P.set_chain([fake])

    class Page:
        text = "Body. " * 40
        title = "T"
        url = "https://example.test"

    result = hosted.summarize([Page()], "q")
    assert result.confident is True
