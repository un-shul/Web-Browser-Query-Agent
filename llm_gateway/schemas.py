"""Response schemas for the two LLM calls."""

ROUTER_SCHEMA = {
    "type": "object",
    "properties": {
        "valid": {"type": "boolean"},
        "intent": {
            "type": "string",
            "enum": ["information_seeking", "navigation", "command", "chitchat", "gibberish"],
        },
        "volatility": {
            "type": "string",
            "enum": ["static", "slow", "dynamic", "realtime"],
        },
        # Advisory only. The server clamps it into the band for its class --
        # models are inconsistent at arithmetic and this value decides whether
        # a stale answer gets served.
        "ttl_hint_seconds": {"type": "integer"},
        "search_query": {"type": "string"},
        "topic": {"type": "string", "enum": ["general", "news"]},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["valid", "intent", "volatility", "confidence", "reason"],
    "propertyOrdering": [
        "valid", "intent", "volatility", "ttl_hint_seconds",
        "search_query", "topic", "confidence", "reason",
    ],
}

# match_index is 0 for "none of these", else 1..N into the candidate list as
# presented. Small integers rather than ids: fewer tokens per candidate, and
# any hallucinated value falls outside 0..N and is detectable.
RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "match_index": {"type": "integer"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["match_index", "confidence", "reason"],
    "propertyOrdering": ["match_index", "confidence", "reason"],
}
