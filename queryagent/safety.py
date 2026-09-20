"""Refuse queries the agent should not answer.

Two layers, mirroring how volatility is decided:

  1. A narrow deterministic check for unambiguous cases. Free, and it runs
     before anything is embedded, searched or cached.
  2. The router's own judgement, which rides along in the call it already
     makes -- so the LLM layer costs nothing extra.

The hard design constraint is **false positives**, not coverage. A blocklist
wide enough to catch every phrasing of explicit content will also refuse
"breast cancer symptoms", "how does HIV spread" and "civilian casualties in
Gaza" -- all legitimate questions a search agent should answer, and the kind of
refusal that makes a tool feel broken and patronising.

So the deterministic layer is deliberately narrow: it matches only phrasings
that are requests *for* explicit material, and it stands down entirely when
the query carries clinical, educational or news framing. Anything ambiguous is
left to the router, which sees the whole query and can tell "how does HIV
spread" from a request for pornography.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

# Shown to the user on refusal. Deliberately short, neutral, and not a lecture:
# it says what happened and moves on.
REFUSAL_MESSAGE = (
    "This agent doesn't answer that kind of query. Try asking something else."
)

# Signals that a query is clinical, educational, legal or news-related. When
# any of these is present the deterministic layer stands down and lets the
# router judge, because these are the contexts where a keyword match is most
# likely to be wrong.
_LEGITIMATE_CONTEXT = re.compile(
    r"\b("
    r"symptom|symptoms|diagnos\w*|treatment|prevent\w*|vaccine|infection|"
    r"disease|cancer|health|medical|medicine|clinical|doctor|hospital|therapy|"
    r"anatomy|biology|physiolog\w*|puberty|pregnan\w*|contracepti\w*|fertility|"
    r"consent|abuse|assault|traffick\w*|exploitation|survivor|helpline|"
    r"law|legal|illegal|statute|rights|policy|regulation|court|convicted|"
    r"news|report|reported|investigation|documentary|history|historical|"
    r"statistics|research|study|studies|education|curriculum|awareness|"
    r"casualt\w*|war|conflict|genocide|famine"
    r")\b",
    re.I,
)

# Requests *for* explicit sexual material. Phrase-level rather than word-level,
# so a single clinical term cannot trigger a refusal on its own.
_EXPLICIT_REQUEST = re.compile(
    r"\b(porn|pornhub|xvideos|xnxx|onlyfans|nsfw|hentai|rule\s?34)\b"
    r"|\b(nude|naked|nudes|topless)\s+(pic|pics|picture|photo|photos|image|"
    r"images|video|videos|clip|clips|leak|leaks)\b"
    r"|\b(sex|porn|xxx|erotic)\s+(video|videos|clip|clips|site|sites|movie|"
    r"movies|story|stories)\b"
    r"|\bwatch\s+\w{0,12}\s?(porn|xxx|hentai)\b"
    r"|\b(sexting|camgirl|escort\s+service)\b",
    re.I,
)

# Absolute: refused regardless of framing, since no framing makes these
# answerable.
_ABSOLUTE = re.compile(
    r"\b(child|minor|underage|teen|kid|kids)\s*\w{0,8}\s*"
    r"(porn|nude|nudes|sexual|sex\b|explicit|csam)\b"
    r"|\b(cp|csam)\s+(download|link|links|site|sites)\b"
    r"|\bhow\s+to\s+(groom|lure)\s+(a\s+)?(child|minor|kid)\b",
    re.I,
)

# Actionable harm: instructions rather than information. "how bombs work" is a
# physics question; "how to build a bomb at home" is not.
_HARM_INSTRUCTIONS = re.compile(
    r"\bhow\s+to\s+\w{0,10}\s?("
    r"make|build|construct|synthesi[sz]e|manufacture|cook|obtain|buy"
    r")\s+\w{0,12}\s?("
    r"bomb|explosive|ied|napalm|thermite|nerve\s+agent|sarin|ricin|"
    r"meth|methamphetamine|fentanyl|heroin|cocaine|lsd|"
    r"ghost\s+gun|untraceable\s+(gun|firearm)|silencer|suppressor"
    r")\b"
    r"|\bhow\s+to\s+(kill|murder|poison)\s+(someone|a\s+person|my\s+\w+)\b",
    re.I,
)

# Self-harm gets a different response entirely -- see check().
_SELF_HARM = re.compile(
    r"\b(how\s+to\s+(kill|hurt|harm)\s+myself|how\s+to\s+(commit\s+)?suicide|"
    r"painless\s+way\s+to\s+die|best\s+way\s+to\s+(die|end\s+my\s+life))\b",
    re.I,
)

SELF_HARM_MESSAGE = (
    "This agent can't help with that. If you're struggling, talking to someone "
    "helps — in India, Tele-MANAS is available free on 14416, any time."
)


def check(query: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """Decide whether to answer `query`.

    Returns (allowed, category, message). `category` is for logging and the
    decision trace; `message` is what the user sees.
    """
    text = (query or "").strip()
    if not text:
        return True, None, None

    # Ordered by severity. Self-harm first, because it needs a different
    # response rather than a flat refusal.
    if _SELF_HARM.search(text):
        return False, "self_harm", SELF_HARM_MESSAGE
    if _ABSOLUTE.search(text):
        return False, "csam", REFUSAL_MESSAGE
    if _HARM_INSTRUCTIONS.search(text):
        return False, "harm_instructions", REFUSAL_MESSAGE

    # Explicit-material requests only when no legitimate framing is present.
    # "how does HIV spread" and "sexual health education" must pass.
    if _EXPLICIT_REQUEST.search(text) and not _LEGITIMATE_CONTEXT.search(text):
        return False, "explicit", REFUSAL_MESSAGE

    return True, None, None


def message_for(category: Optional[str]) -> str:
    """The user-facing message for a category the router flagged."""
    return SELF_HARM_MESSAGE if category == "self_harm" else REFUSAL_MESSAGE
