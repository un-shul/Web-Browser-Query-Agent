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

# ChromaDB posts anonymised usage telemetry by default. Off here: it is an
# outbound request nobody asked for, and it slows cold starts. Set before
# chromadb is imported anywhere.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_IMPL", "chromadb.telemetry.NoopTelemetry")


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
# The "-latest" alias deliberately: pinned ids retire. gemini-2.0-flash was
# the pinned default here and now returns 404 "no longer available". An alias
# may shift behaviour under us, but for JSON classification that is a better
# trade than an app that stops working.
GEMINI_MODEL = _str("GEMINI_MODEL", "gemini-flash-lite-latest")
GEMINI_EMBED_MODEL = _str("GEMINI_EMBED_MODEL", "gemini-embedding-001")
# Upstash free tier caps indexes at 1536 dimensions and gemini-embedding-001
# defaults to 3072, so truncate. Truncated Matryoshka output must be
# L2-renormalized, which the embedding provider does.
GEMINI_EMBED_DIM = _int("GEMINI_EMBED_DIM", 768)
# llama-3.3-70b-versatile was the default and is no longer served.
GROQ_MODEL = _str("GROQ_MODEL", "openai/gpt-oss-20b")

LOCAL_EMBED_MODEL = _str("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_MODEL_PATH = _str("EMBEDDING_MODEL_PATH", "")
LOCAL_SUMMARIZER_MODEL = _str("LOCAL_SUMMARIZER_MODEL", "sshleifer/distilbart-cnn-12-6")

# --- Storage -----------------------------------------------------------------
CHROMA_PATH = _str("CHROMA_PATH", "./chroma_db")
CHROMA_COLLECTION = _str("CHROMA_COLLECTION", "query_cache")

# --- Retrieval thresholds ----------------------------------------------------
# Cosine ranges differ sharply between embedding models, so a single set of
# numbers cannot serve both. Measured on the same 13 query pairs:
#
#                        MiniLM (384d)     gemini-embedding-001 (768d)
#   equivalent           0.918 - 0.965         0.968 - 0.990
#   quantity conflict    0.806 - 0.914         0.962 - 0.971
#   polarity conflict    0.954                 0.966
#   direction conflict   0.997                 0.988
#   different entity     0.464 - 0.624         0.883 - 0.892
#   unrelated            0.016 - 0.052         0.652 - 0.704
#
# Two consequences. MiniLM's 0.60 floor would admit every unrelated pair under
# Gemini, where unrelated tops out at 0.704. And under Gemini the equivalent
# and quantity-conflict bands *overlap* (0.968 vs 0.971), so no auto-accept
# threshold can separate a genuine paraphrase from "under 50000" vs "under
# 100000" -- auto-accept is therefore set high enough to be near-exact-match
# only, and the verifier sees almost everything.
#
# This is also why the deterministic mismatch guard matters more, not less,
# with better embeddings: it works on the query text, so it is unaffected by
# whichever model is behind it.
_THRESHOLDS = {
    "local":  {"floor": 0.60, "auto_accept": 0.93,  "legacy": 0.75},
    "gemini": {"floor": 0.80, "auto_accept": 0.995, "legacy": 0.93},
}
_t = _THRESHOLDS.get(EMBED_PROVIDER, _THRESHOLDS["local"])

CANDIDATE_FLOOR = _float("CANDIDATE_FLOOR", _t["floor"])
RERANK_K = _int("RERANK_K", 5)
AUTO_ACCEPT_SIM = _float("AUTO_ACCEPT_SIM", _t["auto_accept"])
RERANK_MIN_CONFIDENCE = _float("RERANK_MIN_CONFIDENCE", 0.70)
# Used only when no LLM is reachable, so degraded mode is no worse than the
# pre-verifier app was on the same embeddings.
LEGACY_SIM_THRESHOLD = _float("LEGACY_SIM_THRESHOLD", _t["legacy"])

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
