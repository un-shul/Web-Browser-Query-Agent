"""Cheap checks for the things sentence embeddings systematically miss.

Measured against all-MiniLM-L6-v2, cosine similarity handles different
entities well but is nearly blind to polarity, direction and quantity:

    what is the capital of France  / ...of Italy            0.464   fine
    who is the ceo of google       / ...of microsoft        0.624   fine
    how to install docker          / how to uninstall       0.765
    symptoms of type 1 diabetes    / type 2 diabetes        0.806
    best laptops under 50000       / under 100000           0.867
    2024 election results          / 2025 election results  0.914
    is coffee good for your health / bad for your health    0.954
    flights from delhi to mumbai   / from mumbai to delhi   0.997

The last three sit above the 0.93 auto-accept threshold, so the fast path
would serve them without ever consulting the verifier -- defeating the
verifier on precisely the cases it exists for.

These checks are free and deterministic. They never accept anything; they
only force a high-similarity pair to be verified rather than waved through.
"""

from __future__ import annotations

import re
from typing import List, Optional, Set, Tuple

# Opposites that embeddings tend to place close together, since they share
# their context. Order within a pair does not matter.
ANTONYMS: Tuple[Tuple[str, str], ...] = (
    ("good", "bad"), ("best", "worst"), ("better", "worse"),
    ("install", "uninstall"), ("enable", "disable"), ("start", "stop"),
    ("open", "close"), ("add", "remove"), ("create", "delete"),
    ("increase", "decrease"), ("rise", "fall"), ("gain", "loss"),
    ("safe", "dangerous"), ("healthy", "unhealthy"), ("legal", "illegal"),
    ("pros", "cons"), ("advantages", "disadvantages"),
    ("benefits", "risks"), ("benefits", "harms"),
    ("buy", "sell"), ("import", "export"), ("upload", "download"),
    ("more", "less"), ("higher", "lower"), ("cheaper", "expensive"),
    ("before", "after"), ("with", "without"), ("include", "exclude"),
    ("win", "lose"), ("won", "lost"), ("accept", "reject"),
)

NEGATORS: Set[str] = {
    "not", "no", "never", "without", "isn't", "aren't", "doesn't", "don't",
    "didn't", "won't", "can't", "cannot", "shouldn't", "un", "non",
}

# Prepositions that give a query a direction, so the same words in a
# different order mean a different question.
DIRECTIONAL: Set[str] = {"to", "from", "into", "vs", "versus", "over", "against", "than"}

_STOPWORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did", "of",
    "in", "on", "at", "for", "and", "or", "what", "which", "who", "how",
    "why", "when", "where", "me", "my", "i", "it", "its", "be", "been",
    "that", "this", "with", "about", "please", "tell", "explain", "give",
}

_TOKEN = re.compile(r"[a-z0-9]+")
_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?\b")


def _tokens(text: str) -> List[str]:
    return _TOKEN.findall((text or "").lower())


def _content_tokens(text: str) -> List[str]:
    return [t for t in _tokens(text) if t not in _STOPWORDS]


def _numbers(text: str) -> Set[str]:
    return {n.replace(",", "") for n in _NUMBER.findall((text or "").lower())}


def polarity_conflict(a: str, b: str) -> Optional[str]:
    """One query asserts what the other denies."""
    ta, tb = set(_tokens(a)), set(_tokens(b))

    for left, right in ANTONYMS:
        if (left in ta and right in tb) or (right in ta and left in tb):
            return f"opposite terms: {left}/{right}"

    # A negator on one side only. Counting rather than set membership so
    # "not safe" vs "not unsafe" is not mistaken for a conflict.
    na = sum(1 for t in _tokens(a) if t in NEGATORS)
    nb = sum(1 for t in _tokens(b) if t in NEGATORS)
    if (na > 0) != (nb > 0):
        return "negation on one side only"
    return None


def quantity_conflict(a: str, b: str) -> Optional[str]:
    """Different numbers or years, e.g. a price cap or an election year."""
    na, nb = _numbers(a), _numbers(b)
    if na != nb and (na or nb):
        only_a = sorted(na - nb)
        only_b = sorted(nb - na)
        return f"different numbers: {only_a or '-'} vs {only_b or '-'}"
    return None


def direction_conflict(a: str, b: str) -> Optional[str]:
    """Same words, different order, around a directional preposition.

    "flights from delhi to mumbai" and "flights from mumbai to delhi" are
    0.997 apart in embedding space and are different questions.
    """
    ta, tb = _tokens(a), _tokens(b)
    if not (set(ta) & DIRECTIONAL and set(tb) & DIRECTIONAL):
        return None
    ca, cb = _content_tokens(a), _content_tokens(b)
    if sorted(ca) == sorted(cb) and ca != cb:
        return "same terms in a different order around a directional word"
    return None


def find_conflict(a: str, b: str) -> Optional[str]:
    """Why these two queries may not share an answer, or None."""
    for check in (polarity_conflict, quantity_conflict, direction_conflict):
        found = check(a, b)
        if found:
            return found
    return None


def is_hard_conflict(a: str, b: str) -> Optional[str]:
    """A conflict decided here rather than by the verifier.

    Polarity and direction. Measured behaviour drove the split.

    Polarity: asked whether a benefits-only summary of "is coffee good for
    your health" answers "is coffee bad for your health", the verifier
    accepted three times running -- reasoning that a health overview "covers
    both risks and benefits" -- even with the conflict named in the prompt and
    an excerpt that plainly discussed only benefits.

    Direction: "flights from delhi to mumbai" and "flights from mumbai to
    delhi" sit at 0.997 cosine, so retrieval offers no signal at all, and the
    verifier accepted on the grounds that flights are "bidirectional with the
    same duration and airlines". True of duration, false of schedules and
    fares.

    Both are cases where the model rationalises a difference it was told
    about. Quantity is different -- it correctly rejected 2024 -> 2025 results
    and "under 50000" -> "under 100000" -- so that stays its decision.

    The asymmetry justifies deciding locally: wrongly rejecting costs one web
    search, wrongly accepting answers a different question than the one asked.
    Both checks are narrow -- an explicit antonym list, and an exact
    token-multiset match with a changed order -- so false positives are rare.
    """
    return polarity_conflict(a, b) or direction_conflict(a, b)


def safe_to_auto_accept(new_query: str, cached_query: str) -> Tuple[bool, str]:
    """May a high-similarity pair skip verification?

    Returns (safe, reason). A False here does not reject the candidate -- it
    routes it to the verifier instead of the fast path.
    """
    conflict = find_conflict(new_query, cached_query)
    if conflict:
        return False, conflict
    return True, ""
