"""Web search and page content extraction.

Search goes through Tavily. The previous implementation scraped DuckDuckGo's
HTML endpoint with a Google fallback; both are now blocked. Verified
2026-09-19: duckduckgo.com/html and lite.duckduckgo.com/lite both return
HTTP 202 with an anti-bot page, and Google returns HTTP 200 with no
extractable result links. So search_duckduckgo() returned [] on every call
and the pipeline always died at "No search results found".

Content extraction keeps BeautifulSoup as the primary path, because a real
scrape gets the whole article where a search snippet gets a paragraph. Each
URL falls through a chain:

  1. requests + BeautifulSoup
  2. Playwright, if enabled -- for pages that render via JS (local only,
     cannot run on serverless)
  3. Tavily's raw_content from the original search response

Step 3 matters for deployment: datacenter IPs get blocked by many sites, and
raw_content was already paid for by the search call, so it costs nothing
extra.
"""

from __future__ import annotations

import logging
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

from queryagent import config

log = logging.getLogger(__name__)

TAVILY_ENDPOINT = "https://api.tavily.com/search"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0",
]

# Minimum characters for extracted text to count as real content. Below this a
# page is almost always a cookie wall, a JS shell, or a bot check.
MIN_CONTENT_CHARS = 200


class SearchError(RuntimeError):
    """Search could not be performed at all (missing key, provider down)."""


@dataclass
class SearchResult:
    url: str
    title: str = ""
    snippet: str = ""
    score: float = 0.0
    published_date: Optional[str] = None
    # Provider-supplied full text, used as the last link in the content chain.
    raw_content: str = ""


@dataclass
class PageContent:
    url: str
    title: str
    text: str
    via: str  # "scrape" | "playwright" | "provider" | "none"


@dataclass
class SearchBundle:
    query: str
    results: List[SearchResult] = field(default_factory=list)
    provider_answer: str = ""


def _headers() -> Dict[str, str]:
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }


# --- Search ------------------------------------------------------------------


def search(
    query: str,
    topic: str = "general",
    time_range: Optional[str] = None,
    max_results: Optional[int] = None,
) -> SearchBundle:
    """Search the web.

    topic and time_range come straight from the query router, so a news query
    automatically gets topic="news", time_range="day".
    """
    max_results = max_results or config.MAX_SEARCH_RESULTS

    if not config.TAVILY_API_KEY:
        raise SearchError(
            "No TAVILY_API_KEY set. Web search requires it -- DuckDuckGo and "
            "Google both block scraped requests. Free key: https://tavily.com"
        )

    payload: Dict[str, object] = {
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",  # 1 credit; "advanced" costs 2
        "include_raw_content": "markdown",
        "topic": topic if topic in {"general", "news"} else "general",
    }
    if time_range in {"day", "week", "month", "year"}:
        payload["time_range"] = time_range

    try:
        resp = requests.post(
            TAVILY_ENDPOINT,
            headers={
                "Authorization": f"Bearer {config.TAVILY_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
    except requests.RequestException as exc:
        raise SearchError(f"search request failed: {exc}") from exc

    if resp.status_code == 401:
        raise SearchError("Tavily rejected the API key (401).")
    if resp.status_code == 429:
        raise SearchError("Tavily rate limit or monthly credits exhausted (429).")
    if resp.status_code != 200:
        raise SearchError(f"Tavily returned HTTP {resp.status_code}: {resp.text[:200]}")

    body = resp.json()
    results = [
        SearchResult(
            url=item.get("url", ""),
            title=item.get("title", "") or "",
            snippet=item.get("content", "") or "",
            score=float(item.get("score") or 0.0),
            published_date=item.get("published_date"),
            raw_content=item.get("raw_content") or "",
        )
        for item in body.get("results", [])
        if item.get("url")
    ]
    log.info("search %r -> %d results", query, len(results))
    return SearchBundle(
        query=query, results=results, provider_answer=body.get("answer") or ""
    )


# --- Content extraction ------------------------------------------------------

_JUNK_SELECTORS = (
    "nav, header, footer, aside, script, style, iframe, noscript, form, "
    "[role=navigation], [role=banner], [role=contentinfo], "
    ".sidebar, .nav, .navigation, .menu, .trending, .popular, .related, "
    ".recommended, .widget, .widgets, .ads, .advertisement, .social, .share, "
    ".follow, .subscribe, .comments, .comment-section, .cookie, .newsletter"
)

_JUNK_ATTR = re.compile(
    r"sidebar|trending|popular|widget|advertisement|social|share|comment|"
    r"related|recommended|cookie|newsletter|paywall",
    re.I,
)


def extract_main_text(html: str) -> str:
    """Pull the readable body text out of an HTML document."""
    soup = BeautifulSoup(html, "html.parser")

    for el in soup.select(_JUNK_SELECTORS):
        el.decompose()
    for attr in ("class", "id"):
        for el in soup.find_all(attrs={attr: _JUNK_ATTR}):
            el.decompose()

    # Prefer semantic containers; fall back to body.
    node = (
        soup.find("article")
        or soup.find("main")
        or soup.find(attrs={"role": "main"})
        or soup.find(class_=re.compile(r"\b(content|article|post|entry)\b", re.I))
        or soup.body
        or soup
    )
    return re.sub(r"\s+", " ", node.get_text(separator=" ", strip=True)).strip()


def _scrape_requests(url: str, timeout: int) -> str:
    resp = requests.get(url, headers=_headers(), timeout=timeout)
    if resp.status_code != 200:
        log.debug("scrape %s -> HTTP %s", url, resp.status_code)
        return ""
    return extract_main_text(resp.text)


def _scrape_playwright(url: str, timeout: int) -> str:
    """Render the page in a headless browser. Local development only."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log.debug("playwright not installed; skipping")
        return ""
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            try:
                page = browser.new_page(user_agent=random.choice(USER_AGENTS))
                page.goto(url, timeout=timeout * 1000, wait_until="domcontentloaded")
                return extract_main_text(page.content())
            finally:
                browser.close()
    except Exception as exc:
        log.debug("playwright failed on %s: %s", url, exc)
        return ""


def fetch_content(result: SearchResult, timeout: Optional[int] = None) -> PageContent:
    """Get the text of one page, walking the fallback chain."""
    timeout = timeout or config.SCRAPE_TIMEOUT_S

    try:
        text = _scrape_requests(result.url, timeout)
    except requests.RequestException as exc:
        log.debug("scrape %s failed: %s", result.url, exc)
        text = ""
    if len(text) >= MIN_CONTENT_CHARS:
        return PageContent(result.url, result.title, text, "scrape")

    if config.ENABLE_PLAYWRIGHT:
        text = _scrape_playwright(result.url, timeout)
        if len(text) >= MIN_CONTENT_CHARS:
            return PageContent(result.url, result.title, text, "playwright")

    # Already retrieved by the search call, so this is free.
    if len(result.raw_content) >= MIN_CONTENT_CHARS:
        return PageContent(result.url, result.title, result.raw_content, "provider")

    best = max([text, result.raw_content, result.snippet], key=len)
    return PageContent(result.url, result.title, best, "none" if not best else "snippet")


def fetch_contents(
    results: List[SearchResult],
    limit: Optional[int] = None,
    timeout: Optional[int] = None,
) -> List[PageContent]:
    """Fetch several pages concurrently, preserving result order.

    The previous code fetched serially with time.sleep(1) between pages, so
    five pages cost ~15s. Concurrently it is bounded by the slowest page.
    """
    subset = results[: (limit or config.MAX_SEARCH_RESULTS)]
    if not subset:
        return []

    out: List[Optional[PageContent]] = [None] * len(subset)
    workers = min(config.SCRAPE_WORKERS, len(subset))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_content, r, timeout): i for i, r in enumerate(subset)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                out[i] = fut.result()
            except Exception as exc:
                log.warning("content fetch failed for %s: %s", subset[i].url, exc)
                out[i] = PageContent(subset[i].url, subset[i].title, "", "none")

    return [p for p in out if p and p.text]


# --- Backwards-compatible shims ---------------------------------------------
# app.py and main.py still call these; the pipeline refactor removes them.


def search_duckduckgo(query: str, max_results: int = 5, max_retries: int = 3) -> List[str]:
    """Deprecated: returns bare URLs. Kept so existing callers keep working."""
    try:
        return [r.url for r in search(query, max_results=max_results).results]
    except SearchError as exc:
        log.error("%s", exc)
        return []


def scrape_page(url: str, timeout: int = 10) -> str:
    """Deprecated: no provider fallback available when called with a bare URL."""
    return fetch_content(SearchResult(url=url), timeout=timeout).text
