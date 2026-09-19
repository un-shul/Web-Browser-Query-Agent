"""Summarisation via a hosted LLM.

Used when SUMMARIZER=llm, which is the only workable option on serverless:
distilbart plus torch is ~1.7GB against Vercel's 500MB Python bundle limit.

It also produces a better answer than the local path can. distilbart cannot be
instructed, so local.py has to approximate query focus by selecting which
sentences reach the model. Here the query is simply part of the prompt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

from queryagent import config
from queryagent.llm.providers import call_json

log = logging.getLogger(__name__)

@dataclass
class Summary:
    text: str
    # False when the model says the extracts do not answer the question. The
    # caller must not cache these: the answer is a statement about a failed
    # fetch, not about the world, and a retry deserves fresh sources.
    confident: bool = True


SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confident": {"type": "boolean"},
    },
    "required": ["answer", "confident"],
    "propertyOrdering": ["answer", "confident"],
}

SYSTEM = """\
You answer a question using only the page extracts provided.

Rules:
- Answer the question that was asked. Do not summarise the pages in general.
- Use only what the extracts contain. Do not add outside knowledge.
- If the extracts do not answer the question, say so plainly and set
  confident to false.
- 3 to 6 sentences of plain prose, or a short markdown list where the answer is
  genuinely a list. No preamble and no "based on the extracts".
- Where sources disagree, say what they disagree about rather than picking one.

Reply with JSON only."""

# Roughly 30k characters of input. Enough for five pages of trimmed article
# text while staying well inside a Flash model's context and the free tier's
# per-minute token allowance.
MAX_CHARS_PER_PAGE = 6000
MAX_TOTAL_CHARS = 30000


def summarize(
    pages: Sequence,
    query: str,
    max_chars_per_page: int = MAX_CHARS_PER_PAGE,
) -> Optional[Summary]:
    """Answer `query` from `pages`.

    Returns None when no provider is reachable, so the caller can fall back to
    the local summariser rather than failing the request.
    """
    extracts: List[str] = []
    total = 0
    for i, page in enumerate(pages, 1):
        text = (getattr(page, "text", "") or "").strip()
        if not text:
            continue
        chunk = text[:max_chars_per_page]
        if total + len(chunk) > MAX_TOTAL_CHARS:
            chunk = chunk[: MAX_TOTAL_CHARS - total]
        if not chunk:
            break
        host = getattr(page, "url", "") or ""
        extracts.append(f"[{i}] {getattr(page, 'title', '') or host}\n{chunk}")
        total += len(chunk)
        if total >= MAX_TOTAL_CHARS:
            break

    if not extracts:
        return None

    user = f"Question: {query}\n\nPage extracts:\n\n" + "\n\n".join(extracts) + "\n\nJSON:"
    result, meta = call_json(
        SYSTEM, user, SUMMARY_SCHEMA, purpose="summarize",
        max_output_tokens=1200, timeout_s=45.0,
    )
    if result is None:
        log.info("llm summariser unavailable: %s", meta.error)
        return None

    answer = str(result.get("answer") or "").strip()
    if not answer:
        return None
    confident = result.get("confident") is not False
    if not confident:
        log.info("extracts did not answer %r; not cacheable", query[:60])
    return Summary(answer, confident)


def summarize_text(text: str, query: Optional[str] = None, **kwargs) -> Optional[str]:
    """Signature-compatible shim for the local summariser's entry point."""
    class _Page:
        def __init__(self, body): self.text = body; self.title = ""; self.url = ""
    result = summarize([_Page(text)], query or "")
    return result.text if result else None
