"""Query validity classifier.

A logistic regression over sentence embeddings, trained by train_classifier.py
on augmented_query_dataset.csv. It is the cheap first gate in the pipeline:
obvious junk is rejected here without spending an LLM call, and it remains the
offline fallback whenever no LLM is reachable.

Artifacts are loaded in this order:

  1. classifier.json -- plain coefficients, ~8KB, committed to the repo.
     No sklearn needed at runtime, no pickle version risk, no unpickle cost
     on a cold start.
  2. classifier.pkl  -- the original sklearn artifact, if present.

If neither loads, classify_query_with_confidence returns ("unknown", None)
rather than raising, so the app still starts and the router handles validity.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from typing import List, Optional, Sequence, Tuple

import config
import embeddings

log = logging.getLogger(__name__)

JSON_ARTIFACT = "classifier.json"
PICKLE_ARTIFACT = "classifier.pkl"

_state: Optional[dict] = None
_load_attempted = False
_lock = threading.Lock()


def _load_json_artifact() -> Optional[dict]:
    if not os.path.isfile(JSON_ARTIFACT):
        return None
    try:
        with open(JSON_ARTIFACT) as fh:
            blob = json.load(fh)
        coef = [float(x) for x in blob["coef"]]
        return {
            "kind": "json",
            "coef": coef,
            "intercept": float(blob["intercept"]),
            "dim": int(blob.get("dim", len(coef))),
        }
    except Exception as exc:
        log.warning("could not read %s: %s", JSON_ARTIFACT, exc)
        return None


def _load_pickle_artifact() -> Optional[dict]:
    if not os.path.isfile(PICKLE_ARTIFACT):
        return None
    try:
        import pickle

        with open(PICKLE_ARTIFACT, "rb") as fh:
            clf = pickle.load(fh)
        return {"kind": "sklearn", "clf": clf}
    except Exception as exc:
        log.warning("could not read %s: %s", PICKLE_ARTIFACT, exc)
        return None


def _get_state() -> Optional[dict]:
    global _state, _load_attempted
    if _load_attempted:
        return _state
    with _lock:
        if not _load_attempted:
            _state = _load_json_artifact() or _load_pickle_artifact()
            _load_attempted = True
            if _state is None:
                log.warning(
                    "no classifier artifact found (%s / %s); validity gate disabled",
                    JSON_ARTIFACT,
                    PICKLE_ARTIFACT,
                )
    return _state


def is_available() -> bool:
    return _get_state() is not None


def _sigmoid(z: float) -> float:
    # Branch to avoid overflow in exp for large-magnitude z.
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def classify_query_with_confidence(
    query: str, embedding: Optional[Sequence[float]] = None
) -> Tuple[str, Optional[float]]:
    """Return (label, p_valid).

    label is "valid", "invalid", or "unknown" when no artifact is loadable.

    Pass `embedding` when the caller has already embedded the query -- the
    pipeline does, so the query is encoded exactly once per request.
    """
    state = _get_state()
    if state is None:
        return "unknown", None

    try:
        vec: List[float] = (
            [float(x) for x in embedding]
            if embedding is not None
            else embeddings.encode_one(query)
        )
    except Exception as exc:
        log.warning("embedding failed during classification: %s", exc)
        return "unknown", None

    try:
        if state["kind"] == "json":
            coef = state["coef"]
            if len(vec) != len(coef):
                log.warning(
                    "embedding dim %d != classifier dim %d; skipping gate",
                    len(vec),
                    len(coef),
                )
                return "unknown", None
            z = sum(c * v for c, v in zip(coef, vec)) + state["intercept"]
            p_valid = _sigmoid(z)
        else:
            p_valid = float(state["clf"].predict_proba([vec])[0][1])
    except Exception as exc:
        log.warning("classifier inference failed: %s", exc)
        return "unknown", None

    return ("valid" if p_valid >= 0.5 else "invalid"), p_valid


def is_junk(query: str, embedding: Optional[Sequence[float]] = None) -> bool:
    """True only when the classifier is *confident* the query is unusable.

    Use this, not classify_query, as a pipeline gate.

    The 0.5 decision boundary is not safe here. augmented_query_dataset.csv is
    templated -- "valid" rows are question forms ("What is X", "How to X") and
    "invalid" rows are commands ("Navigate to X", "Turn on X") -- so the model
    really learned "is this phrased as a question?". Bare noun-phrase queries
    are neither, and land mid-range:

        weather in bangalore                    p_valid = 0.397
        tesla stock price today                 p_valid = 0.399
        live cricket score india vs australia   p_valid = 0.469

    All three are perfectly good queries, and all three fall below 0.5. Gating
    on classify_query therefore rejects exactly the realtime queries this
    project most cares about.

    So the gate is deliberately asymmetric: reject only below LR_REJECT_P
    (0.05), which catches real junk (punctuation-only at 0.030, "set an alarm for 7am"
    at 0.035) while letting anything ambiguous through to the router. Some junk
    survives this gate and costs one LLM call; that is the correct trade, since
    the alternative is silently refusing to answer valid questions.
    """
    label, p_valid = classify_query_with_confidence(query, embedding)
    if label == "unknown" or p_valid is None:
        return False  # no classifier -> never reject
    return p_valid < config.LR_REJECT_P


def classify_query(query: str) -> str:
    """Original API: "valid" or "invalid".

    Retained for backwards compatibility, but note it applies the 0.5 boundary,
    which is too aggressive to gate on -- see is_junk. Callers wanting a gate
    should use is_junk instead.
    """
    label, _ = classify_query_with_confidence(query)
    return "invalid" if label == "invalid" else "valid"
