"""Extractive-then-abstractive summarisation with a local model.

Used when SUMMARIZER=local. distilbart cannot be instructed, so the query is
applied by *selecting* which sentences reach the model rather than by
prompting: sentences are scored on overlap with the query, the best are kept
in document order, and only those are summarised. That keeps a summary of
"what is photosynthesis" from wandering into deforestation policy because the
source page happened to mention it.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import sys
import threading
import warnings
from typing import Iterable, Iterator, List, Optional, Set

import config

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

log = logging.getLogger(__name__)

# distilbart-cnn-12-6 has a 1024-token encoder. Leave headroom for the special
# tokens the tokenizer adds.
MODEL_TOKEN_LIMIT = 1024
CHUNK_TOKEN_BUDGET = 900
SUMMARY_MAX_TOKENS = 160

_summarizer = None
_summarizer_lock = threading.Lock()

_STOPWORDS: Set[str] = {
    "a", "about", "an", "and", "are", "as", "at", "be", "by", "do", "does",
    "for", "from", "how", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "to", "was", "were", "what", "when", "where", "which", "who", "why",
    "with", "explain", "define", "definition", "meaning", "tell", "me",
}


def get_summarizer():
    """Build the summarisation pipeline on first use.

    This used to run at import time, pulling ~1.2GB of weights into memory
    before the app could serve a request.
    """
    global _summarizer
    if _summarizer is not None:
        return _summarizer
    with _summarizer_lock:
        if _summarizer is None:  # re-check under the lock
            from transformers import pipeline

            _summarizer = pipeline(
                "summarization", model=config.LOCAL_SUMMARIZER_MODEL
            )
    return _summarizer


@contextlib.contextmanager
def suppress_warnings():
    """Silence the pipeline's stdout/stderr chatter."""
    with open(os.devnull, "w") as devnull:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err


# --- Cleaning ----------------------------------------------------------------

# Generic page furniture. Deliberately pattern-based: an earlier version
# hardcoded the names of specific 2024 news stories ("Bigg Boss", "RBI
# Governor", ...), which both aged instantly and silently mangled any
# legitimate query about those subjects.
_BOILERPLATE = re.compile(
    r"\b("
    r"hot picks|trending now|also read|related articles|most popular"
    r"|you are already registered|please log ?in|don'?t have an account"
    r"|sign up|create (an )?account|subscribe( now| to)?|newsletter"
    r"|advertisement|sponsored( content)?|click here|buy now|shop now"
    r"|(this website|we) use[s]? cookies|accept (all )?cookies|cookie policy"
    r"|share on (facebook|twitter|whatsapp)|follow us"
    r"|all rights reserved|terms of (use|service)|privacy policy"
    r")\b[^.]*\.?",
    re.I,
)

# Keep hyphens, parentheses, percent, and slashes. Stripping hyphens turned
# "light-dependent" into "lightdependent" and "3-phosphoglyceric" into
# "3phosphoglyceric" in real output.
_ALLOWED_CHARS = re.compile(r"[^\w\s,.!?'\-()%/:&]")

_NAV_RUN = re.compile(r"(?:\b[A-Z][a-z]+\b[ \t]*){6,}(?=[A-Z][a-z]+\b)")


def clean_text(text: str) -> str:
    """Strip page furniture without damaging prose."""
    text = _BOILERPLATE.sub(" ", text)
    # Long runs of Capitalised Words With No Punctuation are navigation menus
    # and headline lists. This replaces a filter that dropped any sentence
    # whose words were >40% capitalised -- which deleted most content about
    # people and places.
    text = _NAV_RUN.sub(" ", text)
    text = _ALLOWED_CHARS.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fix_punctuation(text: str) -> str:
    text = re.sub(r"\s+([.!?,:;])", r"\1", text)
    text = re.sub(r"([.!?])\s*", r"\1 ", text)
    text = re.sub(r"([,:;])\s*", r"\1 ", text)
    text = re.sub(r"(\.)\s+([a-z])", lambda m: ". " + m.group(2).upper(), text)
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


# --- Query-aware selection ---------------------------------------------------


def _terms(query: str) -> Set[str]:
    return {
        w for w in re.findall(r"[a-z0-9]+", (query or "").lower())
        if len(w) > 2 and w not in _STOPWORDS
    }


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def select_relevant(text: str, query: Optional[str], keep_chars: int) -> str:
    """Keep the sentences most relevant to the query, in document order.

    This is where the query actually influences the output. Without it, the
    model summarises whatever happened to be on the page.
    """
    terms = _terms(query or "")
    sentences = split_sentences(text)
    if not terms or not sentences:
        return text[:keep_chars]

    base: List[float] = []
    for sentence in sentences:
        words = set(re.findall(r"[a-z0-9]+", sentence.lower()))
        if not words:
            base.append(0.0)
            continue
        overlap = len(terms & words)
        # Normalise so a long sentence containing one query term does not
        # outrank a short, dense one.
        base.append(overlap + (overlap / len(words)) * 2)

    # Credit each sentence with a fraction of its neighbours' scores.
    # Term overlap alone is too literal: for "what is photosynthesis" the only
    # query term is "photosynthesis", so the sentences that actually explain
    # it -- about chlorophyll, ATP, wavelengths -- score zero and would rank
    # below unrelated filler. Explanations cluster around the mention, so
    # neighbour credit recovers them while leaving isolated off-topic
    # sentences at the bottom.
    scored = []
    for i, sentence in enumerate(sentences):
        context = base[i]
        if i > 0:
            context += 0.4 * base[i - 1]
        if i + 1 < len(base):
            context += 0.4 * base[i + 1]
        scored.append((context, i, sentence))

    scored.sort(key=lambda t: (-t[0], t[1]))

    # Anything scoring zero is unrelated to the query and to its neighbours.
    # Keep it only to pad a budget that relevant material has not filled.
    relevant = [t for t in scored if t[0] > 0]
    filler = [t for t in scored if t[0] <= 0]

    chosen, total = [], 0
    for _, i, sentence in relevant + filler:
        if total + len(sentence) > keep_chars:
            continue
        chosen.append((i, sentence))
        total += len(sentence)
    if not chosen:
        return text[:keep_chars]

    chosen.sort(key=lambda t: t[0])  # restore reading order
    return " ".join(s for _, s in chosen)


# --- Chunking ----------------------------------------------------------------


def split_text(text: str, max_tokens: int = CHUNK_TOKEN_BUDGET) -> Iterator[str]:
    """Chunk on sentence boundaries, measured in *tokens*.

    The previous version chunked by word count (380 words), but the limit is a
    token limit. 380 words of technical prose tokenised to 1152 tokens, over
    the model's 1024, and the pipeline raised
    "index 1026 is out of bounds" -- which was caught and logged as a skipped
    chunk, silently discarding that content from the summary.
    """
    try:
        tokenizer = get_summarizer().tokenizer

        def n_tokens(s: str) -> int:
            return len(tokenizer.encode(s, add_special_tokens=False))
    except Exception:
        # Tokeniser unavailable; approximate conservatively. English prose runs
        # ~1.3 tokens/word, so 1/2 is a safe margin.
        def n_tokens(s: str) -> int:
            return int(len(s.split()) * 2)

    buf: List[str] = []
    buf_tokens = 0
    for sentence in split_sentences(text) or [text]:
        count = n_tokens(sentence)
        if count > max_tokens:
            # A single oversized sentence: hard-split it by words.
            if buf:
                yield " ".join(buf)
                buf, buf_tokens = [], 0
            words = sentence.split()
            step = max(1, len(words) * max_tokens // max(count, 1))
            for i in range(0, len(words), step):
                yield " ".join(words[i : i + step])
            continue
        if buf_tokens + count > max_tokens and buf:
            yield " ".join(buf)
            buf, buf_tokens = [], 0
        buf.append(sentence)
        buf_tokens += count
    if buf:
        yield " ".join(buf)


def _drop_truncated_tail(summary: str) -> str:
    """Remove a final sentence that the generation limit cut off mid-thought.

    The model stops at SUMMARY_MAX_TOKENS whether or not it has finished a
    sentence, and it still emits a period, so the result looks complete:
    "...synthesize food directly from carb." Rather than guess from the text,
    check whether generation actually hit the ceiling -- if it did, the last
    sentence is unreliable and there are others to fall back on.
    """
    try:
        tokenizer = get_summarizer().tokenizer
        used = len(tokenizer.encode(summary, add_special_tokens=False))
    except Exception:
        return summary
    if used < SUMMARY_MAX_TOKENS - 2:
        return summary  # finished on its own terms
    sentences = split_sentences(summary)
    if len(sentences) > 1:
        return " ".join(sentences[:-1])
    return summary


def _dedupe(sentences: Iterable[str]) -> List[str]:
    """Drop near-duplicate sentences across per-chunk summaries.

    Each chunk is summarised independently, so several chunks covering the
    same source material produce the same opening sentence. Real output
    repeated "Photosynthesis is the process by which..." three times.
    """
    seen: Set[str] = set()
    out: List[str] = []
    for sentence in sentences:
        key = " ".join(sorted(re.findall(r"[a-z0-9]+", sentence.lower())[:12]))
        if key and key in seen:
            continue
        seen.add(key)
        out.append(sentence)
    return out


def trim_to_sentence_boundary(text: str, max_chars: int = 3500) -> str:
    text = fix_punctuation(text)
    if len(text) <= max_chars:
        return text
    out = ""
    for sentence in split_sentences(text):
        if len(out) + len(sentence) > max_chars:
            break
        out += sentence + " "
    return fix_punctuation(out.strip())


# --- Entry point -------------------------------------------------------------


def summarize_text(
    text: str,
    query: Optional[str] = None,
    max_input_chars: int = 12000,
) -> str:
    """Summarise scraped page text, focused on `query`."""
    text = clean_text(text)
    if len(text.split()) < 30:
        return "Content too short to summarize."

    # Select before chunking, so model passes are spent on relevant material.
    text = select_relevant(text, query, max_input_chars)

    summaries: List[str] = []
    skipped = 0
    for chunk in split_text(text):
        if len(chunk.split()) < 20:
            continue
        try:
            with suppress_warnings():
                result = get_summarizer()(
                    chunk, max_length=SUMMARY_MAX_TOKENS, min_length=40,
                    do_sample=False, truncation=True,
                )
            summaries.append(_drop_truncated_tail(result[0]["summary_text"].strip()))
        except Exception as exc:
            skipped += 1
            log.warning("chunk skipped: %s", exc)

    if not summaries:
        return "Unable to generate summary."
    if skipped:
        log.warning("%d chunk(s) failed to summarise", skipped)

    sentences = _dedupe(s for summary in summaries for s in split_sentences(summary))
    return trim_to_sentence_boundary(" ".join(sentences))
