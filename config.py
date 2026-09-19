"""Central configuration.

Every environment variable the app reads is declared here, so there is exactly
one place to look when wiring up a new deployment. Nothing else in the codebase
should touch os.environ directly.

Defaults are chosen so that a fresh clone with no API keys and no .env file
runs the fully local stack. Adding keys is what switches it to hosted APIs.
"""

from __future__ import annotations

import os

try:  # optional -- absent in the minimal serverless bundle
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _str(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --- Backend selection -------------------------------------------------------
# Defaults keep a keyless clone working on the local stack.
LLM_PROVIDER = _str("LLM_PROVIDER", "none")        # gemini|groq|chain|fake|none
EMBED_PROVIDER = _str("EMBED_PROVIDER", "local")   # gemini|local
VECTOR_STORE = _str("VECTOR_STORE", "chroma")      # upstash|chroma
SUMMARIZER = _str("SUMMARIZER", "local")           # llm|local
SEARCH_PROVIDER = _str("SEARCH_PROVIDER", "tavily")

LLM_DISABLED = _bool("LLM_DISABLED", False)

# --- Credentials -------------------------------------------------------------
GEMINI_API_KEY = _str("GEMINI_API_KEY", "")
GROQ_API_KEY = _str("GROQ_API_KEY", "")
TAVILY_API_KEY = _str("TAVILY_API_KEY", "")
UPSTASH_VECTOR_REST_URL = _str("UPSTASH_VECTOR_REST_URL", "")
UPSTASH_VECTOR_REST_TOKEN = _str("UPSTASH_VECTOR_REST_TOKEN", "")

# --- Models ------------------------------------------------------------------
GEMINI_MODEL = _str("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_EMBED_MODEL = _str("GEMINI_EMBED_MODEL", "gemini-embedding-001")
# Upstash free tier caps indexes at 1536 dimensions and gemini-embedding-001
# defaults to 3072, so truncate. Truncated Matryoshka output must be
# L2-renormalized, which the embedding provider does.
GEMINI_EMBED_DIM = _int("GEMINI_EMBED_DIM", 768)
GROQ_MODEL = _str("GROQ_MODEL", "llama-3.3-70b-versatile")

LOCAL_EMBED_MODEL = _str("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_MODEL_PATH = _str("EMBEDDING_MODEL_PATH", "")
LOCAL_SUMMARIZER_MODEL = _str("LOCAL_SUMMARIZER_MODEL", "sshleifer/distilbart-cnn-12-6")

# --- Storage -----------------------------------------------------------------
CHROMA_PATH = _str("CHROMA_PATH", "./chroma_db")
CHROMA_COLLECTION = _str("CHROMA_COLLECTION", "query_cache")

# --- Retrieval thresholds ----------------------------------------------------
# Floor is deliberately well below the legacy 0.75: recall now comes from the
# reranker, not from the similarity threshold.
CANDIDATE_FLOOR = _float("CANDIDATE_FLOOR", 0.60)
RERANK_K = _int("RERANK_K", 5)
AUTO_ACCEPT_SIM = _float("AUTO_ACCEPT_SIM", 0.93)
RERANK_MIN_CONFIDENCE = _float("RERANK_MIN_CONFIDENCE", 0.70)
# Used only when no LLM is reachable. This is the legacy behaviour, so degraded
# mode is never worse than the pre-LLM app.
LEGACY_SIM_THRESHOLD = _float("LEGACY_SIM_THRESHOLD", 0.75)

# --- Classifier gate ---------------------------------------------------------
LR_REJECT_P = _float("LR_REJECT_P", 0.05)
LR_ACCEPT_P = _float("LR_ACCEPT_P", 0.95)

# --- Search / scrape ---------------------------------------------------------
MAX_SEARCH_RESULTS = _int("MAX_SEARCH_RESULTS", 5)
SCRAPE_TIMEOUT_S = _int("SCRAPE_TIMEOUT_S", 8)
SCRAPE_WORKERS = _int("SCRAPE_WORKERS", 5)
ENABLE_PLAYWRIGHT = _bool("ENABLE_PLAYWRIGHT", False)

# --- LLM call behaviour ------------------------------------------------------
LLM_TIMEOUT_S = _float("LLM_TIMEOUT_S", 8.0)

FLASK_DEBUG = _bool("FLASK_DEBUG", False)
