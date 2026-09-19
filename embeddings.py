"""Lazily-loaded, process-shared sentence embedder.

Previously agent.py and cache_chromadb.py each constructed their own
SentenceTransformer at import time, against a gitignored "embedding_model"
directory. That meant two copies of the model in memory and an ImportError on
any fresh clone that had not run train_classifier.py first.

This module loads one instance, on first use, and resolves the model from the
first source that exists:

  1. EMBEDDING_MODEL_PATH, if set
  2. ./embedding_model, if train_classifier.py has been run
  3. the hub id (downloads ~90MB once, cached by huggingface)

Step 3 is what lets a fresh clone work with no training step at all.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import List, Optional

import config

log = logging.getLogger(__name__)

_model = None
_lock = threading.Lock()


def _resolve_source() -> str:
    if config.EMBEDDING_MODEL_PATH:
        return config.EMBEDDING_MODEL_PATH
    if os.path.isdir("embedding_model"):
        return "embedding_model"
    return config.LOCAL_EMBED_MODEL


def get_embedder():
    """Return the shared SentenceTransformer, loading it on first call."""
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is None:  # re-check under the lock
            from sentence_transformers import SentenceTransformer

            source = _resolve_source()
            log.info("loading embedding model from %s", source)
            _model = SentenceTransformer(source)
    return _model


def encode_one(text: str) -> List[float]:
    """Embed a single string. Reuse the result -- this is the hot path."""
    return [float(x) for x in get_embedder().encode([text])[0]]


def encode_many(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    return [[float(x) for x in row] for row in get_embedder().encode(texts)]


def embedding_dim() -> Optional[int]:
    try:
        return int(get_embedder().get_sentence_embedding_dimension())
    except Exception:  # model unavailable -- callers degrade rather than crash
        return None


def is_available() -> bool:
    """True if the model can actually be loaded. Never raises."""
    try:
        get_embedder()
        return True
    except Exception as exc:
        log.warning("embedding model unavailable: %s", exc)
        return False
