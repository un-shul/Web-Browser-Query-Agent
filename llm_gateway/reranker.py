"""LLM verification of semantic cache hits.

Embedding similarity answers "are these about the same subject", which is not
the same as "does this cached answer answer the new question". On MiniLM,
"capital of France" and "capital of Italy" sit around 0.8 cosine, comfortably
above the old 0.75 accept threshold -- so the cache would serve Rome for
Paris.

Retrieval therefore keeps a low floor (0.60) for recall, and acceptance moves
here. One call judges every candidate at once:

    top similarity >= 0.93   ->  0 calls   accept; effectively the same query
    no fresh candidates      ->  0 calls   miss
    anything in between      ->  1 call    the model decides
    no LLM reachable         ->  0 calls   fall back to the old 0.75 rule
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import config

from . import prompts
from .mismatch import find_conflict, is_hard_conflict, safe_to_auto_accept
from .providers import call_json
from .schemas import RERANK_SCHEMA

log = logging.getLogger(__name__)


@dataclass
class RerankResult:
    accepted: bool = False
    index: Optional[int] = None  # 0-based index into the candidates given
    confidence: Optional[float] = None
    reason: str = ""
    decision: str = ""  # auto_accept | llm_accept | llm_reject | threshold | no_candidates
    llm_calls: int = 0
    degraded: bool = False
    candidates: List[Dict[str, Any]] = field(default_factory=list)


def _describe(candidates: Sequence[Any]) -> List[Dict[str, Any]]:
    out = []
    for cand in candidates:
        age = getattr(cand, "age_seconds", -1)
        out.append({
            "query": getattr(cand, "query", ""),
            "summary": getattr(cand, "summary", ""),
            "similarity": float(getattr(cand, "similarity", 0.0)),
            "age": _human_age(age),
            "volatility": getattr(cand, "volatility", ""),
        })
    return out


def _human_age(seconds: int) -> str:
    if seconds is None or seconds < 0:
        return "unknown"
    if seconds < 3600:
        return f"{max(1, seconds // 60)}m"
    if seconds < 86400:
        return f"{seconds // 3600}h"
    return f"{seconds // 86400}d"


def verify(
    query: str,
    candidates: Sequence[Any],
    allow_llm: bool = True,
) -> RerankResult:
    """Decide which candidate, if any, may answer `query`.

    `candidates` are cache_chromadb.Candidate objects, already TTL-filtered
    and sorted by descending similarity.
    """
    described = _describe(candidates)

    if not candidates:
        return RerankResult(
            decision="no_candidates", reason="nothing fresh above the floor",
            candidates=described,
        )

    top = candidates[0]
    top_similarity = float(getattr(top, "similarity", 0.0))
    top_query = getattr(top, "query", "")

    # Decided here, not by the verifier: see mismatch.is_hard_conflict for the
    # measurements behind that.
    hard = is_hard_conflict(query, top_query)
    if hard:
        return RerankResult(
            decision="mismatch_reject", confidence=top_similarity,
            reason=f"rejected on {hard} (similarity {top_similarity:.3f})",
            candidates=described,
        )

    # Near-identical query: re-asks, casing changes, trailing punctuation.
    #
    # Guarded, because high cosine is not sufficient on its own. Measured on
    # MiniLM: "good for your health" vs "bad for your health" is 0.954, and
    # "delhi to mumbai" vs "mumbai to delhi" is 0.997 -- both above this
    # threshold, and both different questions. The guard is free and only
    # diverts such pairs to the verifier; it never accepts anything itself.
    if top_similarity >= config.AUTO_ACCEPT_SIM:
        safe, conflict = safe_to_auto_accept(query, top_query)
        if safe:
            return RerankResult(
                accepted=True, index=0, confidence=top_similarity,
                reason=f"similarity {top_similarity:.3f} >= {config.AUTO_ACCEPT_SIM:.2f}",
                decision="auto_accept", candidates=described,
            )
        log.info("similarity %.3f but %s; verifying", top_similarity, conflict)
        trap_reason = conflict
    else:
        trap_reason = ""

    if not allow_llm or config.LLM_DISABLED:
        # Without a verifier the guard has to decide alone. It found a real
        # conflict, so reject rather than serve a likely-wrong answer.
        if trap_reason:
            return RerankResult(
                decision="mismatch_reject", confidence=top_similarity,
                reason=f"no verifier; rejected on {trap_reason}",
                degraded=True, candidates=described,
            )
        return _threshold_fallback(top_similarity, described, "LLM disabled")

    # If the guard found nothing but similarity is still mid-range, check
    # the top candidate anyway -- the guard is cheap and its findings sharpen
    # the prompt.
    if not trap_reason:
        trap_reason = find_conflict(query, top_query) or ""

    result, meta = call_json(
        prompts.RERANK_SYSTEM,
        prompts.rerank_user_prompt(query, described, conflict=trap_reason),
        RERANK_SCHEMA,
        purpose="rerank",
        max_output_tokens=250,
    )

    if result is None:
        if trap_reason:
            return RerankResult(
                decision="mismatch_reject", confidence=top_similarity,
                reason=f"no verifier; rejected on {trap_reason}",
                degraded=True, candidates=described,
            )
        return _threshold_fallback(top_similarity, described, meta.error)

    raw_index = result.get("match_index")
    try:
        index = int(raw_index)
    except (TypeError, ValueError):
        log.warning("reranker returned non-integer match_index %r", raw_index)
        index = 0

    confidence = result.get("confidence")
    confidence = float(confidence) if isinstance(confidence, (int, float)) else 0.0
    reason = str(result.get("reason", ""))[:300]

    # Fail closed on anything out of range: re-searching is cheap, serving a
    # wrong answer is not.
    if index < 0 or index > len(candidates):
        log.warning("reranker index %s outside 0..%d; treating as a miss",
                    index, len(candidates))
        return RerankResult(
            decision="llm_reject", reason=f"invalid index {index}",
            llm_calls=1, candidates=described,
        )

    if index == 0:
        return RerankResult(
            decision="llm_reject", confidence=confidence,
            reason=reason or "model rejected every candidate",
            llm_calls=1, candidates=described,
        )

    if confidence < config.RERANK_MIN_CONFIDENCE:
        return RerankResult(
            decision="llm_reject", confidence=confidence,
            reason=f"confidence {confidence:.2f} below "
                   f"{config.RERANK_MIN_CONFIDENCE:.2f}: {reason}",
            llm_calls=1, candidates=described,
        )

    return RerankResult(
        accepted=True, index=index - 1, confidence=confidence,
        reason=reason or "model accepted this candidate",
        decision="llm_accept", llm_calls=1, candidates=described,
    )


def _threshold_fallback(
    top_similarity: float, described: List[Dict[str, Any]], why: str
) -> RerankResult:
    """No verifier available, so use the pre-LLM rule.

    Deliberately the original 0.75 threshold: degraded mode should be exactly
    as good as the app was before this layer existed, not worse.
    """
    accept = top_similarity >= config.LEGACY_SIM_THRESHOLD
    return RerankResult(
        accepted=accept,
        index=0 if accept else None,
        confidence=top_similarity,
        reason=(
            f"no verifier ({why}); similarity {top_similarity:.3f} "
            f"{'>=' if accept else '<'} {config.LEGACY_SIM_THRESHOLD:.2f}"
        ),
        decision="threshold", degraded=True, candidates=described,
    )
