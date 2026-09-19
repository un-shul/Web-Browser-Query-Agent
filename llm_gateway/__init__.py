"""Free-tier LLM access with graceful degradation.

Everything the router and reranker need is `call_json`, which never raises and
returns None when no provider is reachable. That single property is what lets
both features degrade to their deterministic fallbacks instead of failing.
"""

from .providers import (  # noqa: F401
    CallMeta,
    LLMBadOutput,
    LLMError,
    LLMRateLimited,
    LLMUnavailable,
    call_json,
    get_chain,
    reset_chain,
)

__all__ = [
    "call_json", "get_chain", "reset_chain", "CallMeta",
    "LLMError", "LLMRateLimited", "LLMUnavailable", "LLMBadOutput",
]
