"""Text embeddings, from a local model or a hosted API.

Two backends behind one interface, chosen by EMBED_PROVIDER:

  local   sentence-transformers all-MiniLM-L6-v2, 384 dims. Free, offline,
          no key. The development default.
  gemini  gemini-embedding-001 truncated to 768 dims. Needed for serverless,
          where torch (547MB on its own) cannot fit inside Vercel's 500MB
          Python bundle limit.

Dimensions differ between backends, so a vector store populated by one cannot
be queried by the other. That is why local development stays on ChromaDB while
production uses a separate Upstash index rather than sharing one.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from typing import List, Optional, Sequence

from queryagent import config

log = logging.getLogger(__name__)

_model = None
_model_lock = threading.Lock()

GEMINI_EMBED_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"
)
# One request may carry many texts, and the rate limit counts requests.
GEMINI_BATCH_LIMIT = 100


class EmbeddingError(RuntimeError):
    """Embedding could not be produced."""


# --- local backend -----------------------------------------------------------


def _resolve_local_source() -> str:
    if config.EMBEDDING_MODEL_PATH:
        return config.EMBEDDING_MODEL_PATH
    if os.path.isdir("embedding_model"):
        return "embedding_model"
    return config.LOCAL_EMBED_MODEL


def get_embedder():
    """The shared SentenceTransformer, loaded on first call."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:  # re-check under the lock
            from sentence_transformers import SentenceTransformer

            source = _resolve_local_source()
            log.info("loading embedding model from %s", source)
            _model = SentenceTransformer(source)
    return _model


def _local_encode(texts: List[str]) -> List[List[float]]:
    return [[float(x) for x in row] for row in get_embedder().encode(texts)]


# --- gemini backend ----------------------------------------------------------


def _l2_normalize(vector: Sequence[float]) -> List[float]:
    """Unit-length the vector.

    Required when gemini-embedding-001 output is truncated below its native
    3072 dims: the Matryoshka prefix is not normalised, and cosine distance in
    the vector store assumes it is.
    """
    norm = math.sqrt(sum(float(x) * float(x) for x in vector))
    if norm == 0:
        return [float(x) for x in vector]
    return [float(x) / norm for x in vector]


def _gemini_encode(texts: List[str], task_type: str = "SEMANTIC_SIMILARITY") -> List[List[float]]:
    import requests

    if not config.GEMINI_API_KEY:
        raise EmbeddingError(
            "EMBED_PROVIDER=gemini but no GEMINI_API_KEY is set. "
            "Free key: https://aistudio.google.com/apikey"
        )

    out: List[List[float]] = []
    url = GEMINI_EMBED_ENDPOINT.format(model=config.GEMINI_EMBED_MODEL)
    headers = {"x-goog-api-key": config.GEMINI_API_KEY, "Content-Type": "application/json"}

    for start in range(0, len(texts), GEMINI_BATCH_LIMIT):
        batch = texts[start : start + GEMINI_BATCH_LIMIT]
        body = {
            "requests": [
                {
                    "model": f"models/{config.GEMINI_EMBED_MODEL}",
                    "content": {"parts": [{"text": text}]},
                    "taskType": task_type,
                    "outputDimensionality": config.GEMINI_EMBED_DIM,
                }
                for text in batch
            ]
        }
        try:
            resp = requests.post(
                url.replace(":embedContent", ":batchEmbedContents"),
                headers=headers, json=body, timeout=45,
            )
        except requests.RequestException as exc:
            raise EmbeddingError(f"gemini embedding request failed: {exc}") from exc

        if resp.status_code != 200:
            raise EmbeddingError(
                f"gemini embedding HTTP {resp.status_code}: {resp.text[:200]}"
            )
        payload = resp.json()
        rows = payload.get("embeddings") or []
        if len(rows) != len(batch):
            raise EmbeddingError(
                f"gemini returned {len(rows)} embeddings for {len(batch)} texts"
            )
        for row in rows:
            values = row.get("values") or row.get("embedding", {}).get("values")
            if not values:
                raise EmbeddingError("gemini embedding response had no values")
            out.append(_l2_normalize(values))
    return out


# --- dispatch ----------------------------------------------------------------


def provider_name() -> str:
    return "gemini" if config.EMBED_PROVIDER == "gemini" else "local"


def encode_many(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    if provider_name() == "gemini":
        return _gemini_encode(texts)
    return _local_encode(texts)


def encode_one(text: str) -> List[float]:
    """Embed one string. The hot path -- callers should reuse the result."""
    return encode_many([text])[0]


def embedding_dim() -> Optional[int]:
    if provider_name() == "gemini":
        return config.GEMINI_EMBED_DIM
    try:
        return int(get_embedder().get_sentence_embedding_dimension())
    except Exception:
        return None


def is_available() -> bool:
    """Whether embeddings can actually be produced. Never raises."""
    if provider_name() == "gemini":
        return bool(config.GEMINI_API_KEY)
    try:
        get_embedder()
        return True
    except Exception as exc:
        log.warning("local embedding model unavailable: %s", exc)
        return False
