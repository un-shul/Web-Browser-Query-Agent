"""Provider implementations and the fallback chain.

Plain `requests`, no vendor SDKs: each provider is a single POST, and skipping
google-genai and groq keeps the deploy bundle and cold start small.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

from queryagent import config
from .budget import get_budget

log = logging.getLogger(__name__)


class LLMError(Exception):
    """Base class. Callers should catch nothing -- use call_json."""


class LLMRateLimited(LLMError):
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class LLMUnavailable(LLMError):
    """No key, timeout, 5xx, DNS failure."""


class LLMBadOutput(LLMError):
    """Response could not be parsed or validated."""


@dataclass(frozen=True)
class CallMeta:
    provider: str = ""
    model: str = ""
    latency_ms: int = 0
    attempts: int = 0
    degraded: bool = False
    error: str = ""
    purpose: str = ""


# --- JSON repair -------------------------------------------------------------

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.I)


def extract_json(text: str) -> Dict[str, Any]:
    """Parse a model response into a dict, tolerating common wrappers.

    Repair is local only. A second call to ask for better-formed JSON would
    cost another request against the free tier, which is a bad trade when the
    deterministic fallback is free.
    """
    if not text or not text.strip():
        raise LLMBadOutput("empty response")

    cleaned = _FENCE.sub("", text.strip())
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
        raise LLMBadOutput(f"expected an object, got {type(parsed).__name__}")
    except json.JSONDecodeError:
        pass

    # Fall back to the first balanced {...} span.
    start = cleaned.find("{")
    if start == -1:
        raise LLMBadOutput(f"no JSON object found in: {cleaned[:120]}")
    depth, in_string, escaped = 0, False, False
    for i, ch in enumerate(cleaned[start:], start):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[start : i + 1])
                except json.JSONDecodeError as exc:
                    raise LLMBadOutput(f"malformed JSON span: {exc}") from exc
    raise LLMBadOutput("unterminated JSON object")


# --- Providers ---------------------------------------------------------------


class LLMProvider(ABC):
    name = "base"

    @abstractmethod
    def is_configured(self) -> bool: ...

    @abstractmethod
    def rpm(self) -> Optional[int]: ...

    @abstractmethod
    def complete_json(
        self, *, system: str, user: str, schema: Dict[str, Any],
        max_output_tokens: int, temperature: float, timeout_s: float,
    ) -> Dict[str, Any]: ...


class GeminiProvider(LLMProvider):
    name = "gemini"
    ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, model: Optional[str] = None):
        self.model = model or config.GEMINI_MODEL

    def is_configured(self) -> bool:
        return bool(config.GEMINI_API_KEY)

    def rpm(self) -> Optional[int]:
        return 15  # free tier

    def complete_json(self, *, system, user, schema, max_output_tokens,
                      temperature, timeout_s):
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": temperature,
                # Generous: a low ceiling makes the model return an empty
                # candidate with finishReason MAX_TOKENS rather than truncated
                # JSON, which looks like an outage.
                "maxOutputTokens": max(max_output_tokens, 2048),
                "responseMimeType": "application/json",
                "responseSchema": _to_gemini_schema(schema),
            },
        }
        try:
            resp = requests.post(
                self.ENDPOINT.format(model=self.model),
                headers={"x-goog-api-key": config.GEMINI_API_KEY,
                         "Content-Type": "application/json"},
                json=body, timeout=timeout_s,
            )
        except requests.RequestException as exc:
            raise LLMUnavailable(f"gemini request failed: {exc}") from exc

        if resp.status_code == 429:
            raise LLMRateLimited("gemini quota exceeded", _retry_after(resp))
        if resp.status_code in (401, 403):
            raise LLMUnavailable(f"gemini rejected the key ({resp.status_code})")
        if resp.status_code == 404:
            raise LLMUnavailable(f"gemini model {self.model} not available")
        if resp.status_code >= 500:
            raise LLMUnavailable(f"gemini HTTP {resp.status_code}")
        if resp.status_code != 200:
            raise LLMBadOutput(f"gemini HTTP {resp.status_code}: {resp.text[:200]}")

        payload = resp.json()
        candidates = payload.get("candidates") or []
        if not candidates:
            reason = (payload.get("promptFeedback") or {}).get("blockReason", "no candidates")
            raise LLMBadOutput(f"gemini returned nothing ({reason})")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts)
        if not text.strip():
            raise LLMBadOutput(
                f"gemini empty text (finishReason={candidates[0].get('finishReason')})"
            )
        return extract_json(text)


class GroqProvider(LLMProvider):
    name = "groq"
    ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, model: Optional[str] = None):
        self.model = model or config.GROQ_MODEL

    def is_configured(self) -> bool:
        return bool(config.GROQ_API_KEY)

    def rpm(self) -> Optional[int]:
        return 30  # free tier

    def complete_json(self, *, system, user, schema, max_output_tokens,
                      temperature, timeout_s):
        # Groq's json_object mode takes no schema, so it goes in the prompt.
        system_with_schema = (
            f"{system}\n\nReply with JSON only, matching this schema:\n"
            f"{json.dumps(schema)}"
        )
        try:
            resp = requests.post(
                self.ENDPOINT,
                headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_with_schema},
                        {"role": "user", "content": user},
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": temperature,
                    "max_tokens": max_output_tokens,
                },
                timeout=timeout_s,
            )
        except requests.RequestException as exc:
            raise LLMUnavailable(f"groq request failed: {exc}") from exc

        if resp.status_code == 429:
            raise LLMRateLimited("groq quota exceeded", _retry_after(resp))
        if resp.status_code in (401, 403):
            raise LLMUnavailable(f"groq rejected the key ({resp.status_code})")
        if resp.status_code == 404:
            raise LLMUnavailable(f"groq model {self.model} not available")
        if resp.status_code >= 500:
            raise LLMUnavailable(f"groq HTTP {resp.status_code}")
        if resp.status_code != 200:
            raise LLMBadOutput(f"groq HTTP {resp.status_code}: {resp.text[:200]}")

        choices = resp.json().get("choices") or []
        if not choices:
            raise LLMBadOutput("groq returned no choices")
        return extract_json((choices[0].get("message") or {}).get("content", ""))


class NullProvider(LLMProvider):
    """Explicit terminator. Makes "no LLM" a configuration, not an accident."""

    name = "null"
    model = "none"

    def is_configured(self) -> bool:
        return False

    def rpm(self) -> Optional[int]:
        return None

    def complete_json(self, **kwargs):
        raise LLMUnavailable("no LLM provider configured")


class FakeProvider(LLMProvider):
    """Replays recorded responses so tests never spend quota.

    A cassette miss raises rather than falling through to the network, so a
    test can never quietly start making real calls.
    """

    name = "fake"
    model = "cassette"

    def __init__(self, cassette: Optional[Dict[str, Any]] = None, record: bool = False):
        self.cassette: Dict[str, Any] = cassette if cassette is not None else {}
        self.record = record
        self.calls: List[Dict[str, Any]] = []

    @staticmethod
    def key(purpose: str, system: str, user: str) -> str:
        return hashlib.sha256(
            f"{purpose}|{system}|{user}".encode()
        ).hexdigest()[:32]

    def is_configured(self) -> bool:
        return True

    def rpm(self) -> Optional[int]:
        return None

    def complete_json(self, *, system, user, schema, max_output_tokens,
                      temperature, timeout_s, purpose=""):
        key = self.key(purpose, system, user)
        self.calls.append({"purpose": purpose, "key": key, "user": user})
        if key in self.cassette:
            return self.cassette[key]
        raise LLMBadOutput(
            f"cassette miss for {purpose} (key {key}). Re-record instead of "
            f"letting the test reach the network."
        )

    @property
    def call_count(self) -> int:
        return len(self.calls)


def _retry_after(resp) -> Optional[float]:
    raw = resp.headers.get("Retry-After")
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    # Gemini puts a RetryInfo duration in the error body.
    try:
        for detail in (resp.json().get("error") or {}).get("details") or []:
            delay = detail.get("retryDelay")
            if isinstance(delay, str) and delay.endswith("s"):
                return float(delay[:-1])
    except Exception:
        pass
    return None


def _to_gemini_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Strip JSON Schema keys Gemini's responseSchema does not accept."""
    allowed = {"type", "properties", "required", "items", "enum",
               "description", "nullable", "propertyOrdering"}
    out: Dict[str, Any] = {}
    for key, value in schema.items():
        if key not in allowed:
            continue
        if key == "properties" and isinstance(value, dict):
            out[key] = {k: _to_gemini_schema(v) for k, v in value.items()}
        elif key == "items" and isinstance(value, dict):
            out[key] = _to_gemini_schema(value)
        else:
            out[key] = value
    return out


# --- Chain -------------------------------------------------------------------

_chain: Optional[List[LLMProvider]] = None
_chain_lock = threading.Lock()

_REGISTRY = {
    "gemini": GeminiProvider,
    "groq": GroqProvider,
    "null": NullProvider,
    "none": NullProvider,
}


def get_chain() -> List[LLMProvider]:
    """Providers to try in order, unconfigured ones dropped."""
    global _chain
    if _chain is not None:
        return _chain
    with _chain_lock:
        if _chain is None:
            _chain = _build_chain()
    return _chain


def reset_chain() -> None:
    """Force a rebuild. Used by tests and after a config change."""
    global _chain
    with _chain_lock:
        _chain = None


def set_chain(providers: List[LLMProvider]) -> None:
    global _chain
    with _chain_lock:
        _chain = list(providers)


def _build_chain() -> List[LLMProvider]:
    if config.LLM_DISABLED:
        log.info("LLM_DISABLED set; running fully deterministic")
        return []

    spec = (config.LLM_PROVIDER or "none").strip().lower()
    if spec == "fake":
        return [FakeProvider(_load_cassette())]
    names = ["gemini", "groq"] if spec == "chain" else [n.strip() for n in spec.split(",")]

    chain: List[LLMProvider] = []
    for name in names:
        cls = _REGISTRY.get(name)
        if cls is None:
            log.warning("unknown LLM provider %r; ignoring", name)
            continue
        provider = cls()
        if provider.is_configured():
            chain.append(provider)
        elif name not in ("null", "none"):
            log.info("%s not configured (no API key); skipping", name)
    if not chain:
        log.info("no LLM provider available; deterministic fallbacks only")
    return chain


CASSETTE_PATH = os.path.join("tests", "fixtures", "llm_cassette.json")


def _load_cassette() -> Dict[str, Any]:
    try:
        with open(CASSETTE_PATH) as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        log.warning("could not load cassette %s: %s", CASSETTE_PATH, exc)
        return {}


def call_json(
    system: str,
    user: str,
    schema: Dict[str, Any],
    purpose: str = "",
    max_output_tokens: int = 512,
    temperature: float = 0.0,
    timeout_s: Optional[float] = None,
) -> Tuple[Optional[Dict[str, Any]], CallMeta]:
    """Ask the chain for a JSON object.

    Never raises. Returns (None, meta) when every provider fails, so callers
    always have a single degraded branch to write rather than a try/except
    around each feature.
    """
    timeout_s = timeout_s or config.LLM_TIMEOUT_S
    budget = get_budget()
    attempts = 0
    last_error = "no provider configured"
    started = time.time()

    for provider in get_chain():
        if not budget.allow(provider.name, provider.rpm()):
            last_error = f"{provider.name}: rate limited locally"
            continue

        # One retry for transient failures only. A 429 is not transient -- it
        # means the window is exhausted, so we move on rather than retry.
        for attempt in range(2):
            attempts += 1
            budget.record_attempt(provider.name)
            try:
                kwargs = dict(
                    system=system, user=user, schema=schema,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature, timeout_s=timeout_s,
                )
                if isinstance(provider, FakeProvider):
                    kwargs["purpose"] = purpose
                result = provider.complete_json(**kwargs)
                budget.record_success(provider.name)
                return result, CallMeta(
                    provider=provider.name,
                    model=getattr(provider, "model", ""),
                    latency_ms=int((time.time() - started) * 1000),
                    attempts=attempts, degraded=False, purpose=purpose,
                )
            except LLMRateLimited as exc:
                budget.record_rate_limited(provider.name, exc.retry_after)
                last_error = f"{provider.name}: {exc}"
                break
            except LLMUnavailable as exc:
                budget.record_failure(provider.name)
                last_error = f"{provider.name}: {exc}"
                if attempt == 0:
                    time.sleep(0.3 + random.random() * 0.3)
                    continue
                break
            except LLMBadOutput as exc:
                budget.record_failure(provider.name)
                last_error = f"{provider.name}: {exc}"
                break
            except Exception as exc:  # never let a provider bug escape
                budget.record_failure(provider.name)
                last_error = f"{provider.name}: unexpected {type(exc).__name__}: {exc}"
                break

    log.info("call_json(%s) degraded: %s", purpose, last_error)
    return None, CallMeta(
        latency_ms=int((time.time() - started) * 1000),
        attempts=attempts, degraded=True, error=last_error, purpose=purpose,
    )
