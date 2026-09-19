"""Prompts for the router and the reranker."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

ROUTER_SYSTEM = """\
You classify web search queries for a caching search classifier.

Decide two things.

1. valid -- can this be answered from web pages? Mark invalid when the input
   is gibberish, a device command ("set an alarm"), a request to open an app
   or site ("open youtube"), or pure small talk.

2. volatility -- how soon could the correct answer change? Ask yourself how
   long from now the answer you would give today becomes wrong:
     realtime  within the hour   (live scores, market prices, weather now)
     dynamic   within a day      (news, ongoing events, fast-moving topics)
     slow      within a month    (recommendations, reviews, comparisons)
     static    not really        (definitions, history, settled facts)

Judge the answer's shelf life, not the topic's popularity. "who won the 2011
world cup" is static because the result is settled; "who is winning" is
realtime. "climate of delhi" is static; "weather in delhi" is realtime.

Also set topic to "news" for realtime or dynamic queries, and give
search_query as a cleaned-up version of the query suited to a search engine.

Reply with JSON only."""

ROUTER_EXAMPLES = """\
Examples:
live cricket score -> {"valid":true,"intent":"information_seeking","volatility":"realtime","topic":"news"}
who won the 2011 cricket world cup -> {"valid":true,"intent":"information_seeking","volatility":"static","topic":"general"}
tesla stock price -> {"valid":true,"intent":"information_seeking","volatility":"realtime","topic":"news"}
is tesla a good investment -> {"valid":true,"intent":"information_seeking","volatility":"slow","topic":"general"}
weather in delhi -> {"valid":true,"intent":"information_seeking","volatility":"realtime","topic":"news"}
climate of delhi -> {"valid":true,"intent":"information_seeking","volatility":"static","topic":"general"}
latest ai news -> {"valid":true,"intent":"information_seeking","volatility":"dynamic","topic":"news"}
what is photosynthesis -> {"valid":true,"intent":"information_seeking","volatility":"static","topic":"general"}
open youtube -> {"valid":false,"intent":"navigation","volatility":"static","topic":"general"}
asdkjhaskdjh -> {"valid":false,"intent":"gibberish","volatility":"static","topic":"general"}"""


def router_user_prompt(query: str) -> str:
    # The date is not optional. Without it the model cannot tell "results of
    # tomorrow's match" from "results of the 2019 final".
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"{ROUTER_EXAMPLES}\n\nToday (UTC): {today}\nQuery: {query}\nJSON:"


RERANK_SYSTEM = """\
You decide whether a cached answer can be reused for a new question.

The candidates were already retrieved by embedding similarity, so "these are
about the same subject" is not the question. The question is narrower: would
the cached answer actually answer the new query correctly?

Reject a candidate when:
  - it is about a different entity ("capital of France" vs "capital of Italy")
  - it covers a different time window ("2024 results" vs "2025 results")
  - it is broader or narrower in a way that changes the answer
    ("python" vs "python decorators")
  - it uses a different sense of an ambiguous word
    ("python the language" vs "python the snake")
  - it answers a different question about the same subject
    ("how tall is Everest" vs "who first climbed Everest")

Accept only when the cached answer genuinely satisfies the new query. When in
doubt, reject: re-running a search is cheap, and serving a wrong answer is not.

Set match_index to 0 for "none of these", or to the number of the candidate
that matches. Reply with JSON only."""


def rerank_user_prompt(
    query: str, candidates: List[dict], conflict: str = ""
) -> str:
    lines = [f"New query: {query}"]
    if conflict:
        # A deterministic check already found something specific. Say what it
        # is, so the model scrutinises that rather than reasoning from the
        # general case -- it only sees the first 200 characters of each
        # answer and cannot tell what the rest covers.
        lines += [
            "",
            f"WARNING: an automated check flagged a likely mismatch with the "
            f"top candidate ({conflict}).",
            "Do not assume the cached answer covers the other side unless the "
            "excerpt below shows that it does. If you cannot confirm it from "
            "the excerpt, answer 0.",
        ]
    lines += ["", "Candidates:"]
    for i, cand in enumerate(candidates, 1):
        head = (cand.get("summary") or "")[:200].replace("\n", " ")
        lines.append(
            f'[{i}] query="{cand.get("query", "")}" '
            f'| similarity={cand.get("similarity", 0):.3f} '
            f'| age={cand.get("age", "unknown")} '
            f'| answer_begins="{head}..."'
        )
    lines += ["", "JSON:"]
    return "\n".join(lines)
