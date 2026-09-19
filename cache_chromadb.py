"""Semantic cache over ChromaDB, with freshness.

Answers are keyed by embedding, so a paraphrase of an earlier question reuses
its answer. The addition here is that reuse is now bounded by time: every
entry carries the volatility class it was created under and an absolute
expiry, and an expired entry is not a candidate.

That closes a real hole. Previously the only test was "nearest neighbour,
cosine >= 0.75", so "live cricket score" would keep serving whatever summary
it computed the first time, forever.

Metadata schema 2:

    summary          str   the cached answer
    canonical_query  str   normalised query text ("" if unknown)
    volatility       str   static | slow | dynamic | realtime | unknown
    created_at       int   epoch seconds, UTC
    expires_at       int   epoch seconds, UTC
    ttl_seconds      int   as resolved by volatility_policy
    source_urls      str   json.dumps([...]) -- Chroma rejects list values
    router_source    str   how volatility was decided
    schema_version   int   2
    hit_count        int
    last_hit_at      int

Chroma metadata values must be str/int/float/bool: no None, no lists. Upstash
would accept nested JSON, but designing to Chroma's stricter rules lets one
schema serve both backends.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import config
import embeddings
import volatility_policy as vp

# chromadb is NOT imported at module level. It is absent from the production
# bundle -- torch and friends are 1.3GB against a 500MB limit -- and importing
# it here made this module unimportable on serverless, which crashed the whole
# app at startup. ChromaDBCache imports it when it is actually constructed.

log = logging.getLogger(__name__)

SCHEMA_VERSION = 2


class CacheUnavailable(RuntimeError):
    """The configured cache backend cannot be used, with a reason worth reading."""

# Entries written before schema 2 have no timestamp, so their real age is
# unknowable. They share one stable pseudo-birthday recorded on first boot
# after the upgrade, and stop being served once the grace window passes --
# fail-closed, rather than serving something of unknown provenance forever.
LEGACY_GRACE_SECONDS = 7 * 86400
_MIGRATION_EPOCH_KEY = "__migration_epoch__"


@dataclass
class Candidate:
    """One cache entry retrieved as a possible match."""

    id: str
    query: str
    summary: str
    similarity: float
    volatility: str = vp.UNKNOWN
    created_at: int = 0
    expires_at: int = 0
    ttl_seconds: int = 0
    source_urls: List[str] = field(default_factory=list)
    router_source: str = ""
    schema_version: int = 1
    hit_count: int = 0

    @property
    def age_seconds(self) -> int:
        return max(0, int(time.time()) - self.created_at) if self.created_at else -1

    def is_expired(self, now: Optional[int] = None) -> bool:
        return (now or int(time.time())) >= self.expires_at


def _now() -> int:
    return int(time.time())


def _as_int(value: Any, default: int = 0) -> int:
    """Coerce a metadata value to int without raising.

    Metadata can be hand-edited, written by an older version, or corrupted.
    The normalizer's whole purpose is that no read path can raise, so this
    must swallow anything.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_meta(meta: Optional[Dict[str, Any]], migration_epoch: int) -> Dict[str, Any]:
    """Fill in every field a pre-schema-2 entry is missing.

    Every read path must go through this. The old writer stored only
    {"summary": ...}, so anything reading another key directly raises
    KeyError on older data.
    """
    meta = dict(meta or {})
    version = _as_int(meta.get("schema_version"), 1) or 1

    if version >= SCHEMA_VERSION:
        return {
            "summary": meta.get("summary", ""),
            "canonical_query": meta.get("canonical_query", ""),
            "volatility": meta.get("volatility", vp.UNKNOWN),
            "created_at": _as_int(meta.get("created_at")),
            "expires_at": _as_int(meta.get("expires_at")),
            "ttl_seconds": _as_int(meta.get("ttl_seconds")),
            "source_urls": meta.get("source_urls", "[]"),
            "router_source": meta.get("router_source", ""),
            "schema_version": version,
            "hit_count": _as_int(meta.get("hit_count")),
            "last_hit_at": _as_int(meta.get("last_hit_at")),
        }

    # Legacy entry: date it from the migration epoch and give it the
    # "unknown" TTL band, which is deliberately short.
    ttl = vp.ttl_for(vp.UNKNOWN)
    return {
        "summary": meta.get("summary", ""),
        "canonical_query": meta.get("canonical_query", ""),
        "volatility": vp.UNKNOWN,
        "created_at": migration_epoch,
        "expires_at": min(migration_epoch + ttl, migration_epoch + LEGACY_GRACE_SECONDS),
        "ttl_seconds": ttl,
        "source_urls": "[]",
        "router_source": "legacy",
        "schema_version": 1,
        "hit_count": 0,
        "last_hit_at": 0,
    }


def _decode_urls(raw: Any) -> List[Any]:
    """Decode stored sources.

    Entries written before titles were captured hold a list of bare URL
    strings; newer ones hold {"url", "title"} objects. Both are returned as
    given, and the UI handles either shape.
    """
    loaded = raw if isinstance(raw, list) else None
    if loaded is None:
        try:
            loaded = json.loads(raw or "[]")
        except (ValueError, TypeError):
            return []
    if not isinstance(loaded, list):
        return []
    out: List[Any] = []
    for item in loaded:
        if isinstance(item, dict) and item.get("url"):
            out.append({"url": str(item["url"]), "title": str(item.get("title") or "")})
        elif isinstance(item, str):
            out.append(item)
    return out


class ChromaDBCache:
    def __init__(self, collection_name=None, db_path=None):
        try:
            import chromadb  # local: see the note at the top of this module
        except ImportError as exc:
            # The default backend cannot work on serverless: no chromadb in the
            # bundle and no writable disk. Left to surface on its own this
            # arrives as a bare "No module named 'chromadb'" mid-query, which
            # says nothing about the actual mistake.
            raise CacheUnavailable(
                "VECTOR_STORE=chroma, but chromadb is not installed.\n"
                "  Deployed?  set VECTOR_STORE=upstash (plus EMBED_PROVIDER=gemini "
                "and SUMMARIZER=llm) in your host's environment variables, then "
                "redeploy -- the local model stack does not fit in a serverless "
                "bundle.\n"
                "  Local?     pip install -r requirements-local.txt"
            ) from exc

        collection_name = collection_name or config.CHROMA_COLLECTION
        db_path = db_path or config.CHROMA_PATH

        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._migration_epoch: Optional[int] = None
        log.info("ChromaDB cache ready (%d entries)", self.collection.count())

    # The embedder is shared process-wide and loaded on first use, so
    # constructing this class no longer pulls a ~90MB model into memory.
    @property
    def model(self):
        return embeddings.get_embedder()

    # --- migration epoch ----------------------------------------------------

    @property
    def migration_epoch(self) -> int:
        """One stable timestamp shared by all pre-schema-2 entries.

        Persisted so restarts do not keep extending legacy entries' lives.
        """
        if self._migration_epoch is not None:
            return self._migration_epoch
        try:
            got = self.collection.get(ids=[_MIGRATION_EPOCH_KEY])
            metas = got.get("metadatas") or []
            if metas and metas[0] and metas[0].get("epoch"):
                epoch = _as_int(metas[0]["epoch"])
                if epoch > 0:
                    self._migration_epoch = epoch
                    return self._migration_epoch
        except Exception as exc:
            log.debug("could not read migration epoch: %s", exc)

        epoch = _now()
        try:
            dim = embeddings.embedding_dim() or 384
            self.collection.add(
                ids=[_MIGRATION_EPOCH_KEY],
                embeddings=[[0.0] * dim],
                documents=[_MIGRATION_EPOCH_KEY],
                metadatas=[{"epoch": epoch, "note": "pseudo-birthday for legacy entries"}],
            )
        except Exception as exc:
            log.debug("could not persist migration epoch: %s", exc)
        self._migration_epoch = epoch
        return epoch

    # --- retrieval ----------------------------------------------------------

    def find_similar_candidates(
        self,
        query: str,
        k: Optional[int] = None,
        floor: Optional[float] = None,
        embedding: Optional[Sequence[float]] = None,
        include_expired: bool = False,
        now: Optional[int] = None,
    ) -> Tuple[List[Candidate], List[Candidate]]:
        """Return (fresh, expired) candidates, best similarity first.

        Freshness is filtered in Python rather than with a Chroma `where`
        clause. A clause like {"expires_at": {"$gt": now}} reads better but
        silently excludes documents that lack the key -- which is every legacy
        entry -- giving a behaviour change with nothing logged. Filtering here
        means expired entries can be counted, logged, and migrated.
        """
        k = k or config.RERANK_K
        floor = config.CANDIDATE_FLOOR if floor is None else floor
        now = now or _now()

        try:
            vector = list(embedding) if embedding is not None else embeddings.encode_one(query)
        except Exception as exc:
            log.warning("could not embed query: %s", exc)
            return [], []

        # Over-fetch: the freshness filter runs after retrieval, so asking for
        # exactly k would silently return fewer. The old code used
        # n_results=1, which also meant a fresher but slightly less similar
        # entry could never be found.
        n_fetch = max(3 * k, 15)

        try:
            res = self.collection.query(query_embeddings=[vector], n_results=n_fetch)
        except Exception as exc:
            log.warning("cache query failed: %s", exc)
            return [], []

        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        epoch = self.migration_epoch
        fresh: List[Candidate] = []
        expired: List[Candidate] = []

        for i, entry_id in enumerate(ids):
            if entry_id == _MIGRATION_EPOCH_KEY:
                continue
            similarity = 1.0 - float(dists[i])
            if similarity < floor:
                continue
            meta = _normalize_meta(metas[i] if i < len(metas) else {}, epoch)
            if not meta["summary"]:
                continue
            cand = Candidate(
                id=entry_id,
                query=docs[i] if i < len(docs) else "",
                summary=meta["summary"],
                similarity=similarity,
                volatility=meta["volatility"],
                created_at=meta["created_at"],
                expires_at=meta["expires_at"],
                ttl_seconds=meta["ttl_seconds"],
                source_urls=_decode_urls(meta["source_urls"]),
                router_source=meta["router_source"],
                schema_version=meta["schema_version"],
                hit_count=meta["hit_count"],
            )
            (expired if cand.is_expired(now) else fresh).append(cand)

        fresh.sort(key=lambda c: -c.similarity)
        expired.sort(key=lambda c: -c.similarity)

        if expired:
            log.info(
                "cache: %d fresh, %d expired above floor %.2f", len(fresh), len(expired), floor
            )
        return fresh[:k], expired[:k]

    # --- writing ------------------------------------------------------------

    def add_to_cache(
        self,
        query: str,
        summary: str,
        volatility: str = vp.DEFAULT_VOLATILITY,
        ttl_seconds: Optional[int] = None,
        canonical_query: Optional[str] = None,
        source_urls: Optional[Sequence[Any]] = None,
        router_source: str = "",
        embedding: Optional[Sequence[float]] = None,
    ) -> Optional[str]:
        volatility = vp.normalize_volatility(volatility)
        ttl = vp.ttl_for(volatility, ttl_seconds)
        if ttl <= 0:
            log.info("not caching %r (ttl 0 for %s)", query[:50], volatility)
            return None

        created = _now()
        try:
            vector = list(embedding) if embedding is not None else embeddings.encode_one(query)
        except Exception as exc:
            log.warning("could not embed for cache write: %s", exc)
            return None

        # Reuse the row for an identical question instead of adding a second
        # one. A force-refresh or a re-run otherwise appends a near-duplicate
        # every time, so the cache grows without bound and the explorer shows
        # the same query repeatedly.
        entry_id = self._find_exact(query) or str(uuid.uuid4())
        try:
            self.collection.upsert(
                ids=[entry_id],
                embeddings=[vector],
                documents=[query],
                metadatas=[{
                    "summary": summary,
                    "canonical_query": canonical_query or query,
                    "volatility": volatility,
                    "created_at": created,
                    "expires_at": vp.expires_at(created, ttl),
                    "ttl_seconds": ttl,
                    "source_urls": json.dumps(list(source_urls or [])),
                    "router_source": router_source,
                    "schema_version": SCHEMA_VERSION,
                    "hit_count": 0,
                    "last_hit_at": 0,
                }],
            )
        except Exception as exc:
            log.warning("cache write failed: %s", exc)
            return None

        log.info("cached %r as %s (ttl %ds)", query[:50], volatility, ttl)
        return entry_id

    def _find_exact(self, query: str) -> Optional[str]:
        """Existing entry id for this exact question, if any."""
        wanted = " ".join((query or "").strip().lower().split())
        try:
            for entry_id, doc, _meta in self.list_entries():
                if " ".join((doc or "").strip().lower().split()) == wanted:
                    return entry_id
        except Exception:
            return None
        return None

    def touch(self, entry_id: str) -> None:
        """Record a cache hit. Best-effort; never raises."""
        try:
            got = self.collection.get(ids=[entry_id])
            metas = got.get("metadatas") or []
            if not metas or not metas[0]:
                return
            meta = dict(metas[0])
            meta["hit_count"] = _as_int(meta.get("hit_count")) + 1
            meta["last_hit_at"] = _now()
            self.collection.update(ids=[entry_id], metadatas=[meta])
        except Exception as exc:
            log.debug("touch failed for %s: %s", entry_id, exc)

    def backfill(self, candidate: Candidate) -> None:
        """Upgrade a legacy entry to schema 2 in place.

        Volatility comes from the deterministic heuristics only -- migration
        must never spend an LLM call.
        """
        if candidate.schema_version >= SCHEMA_VERSION:
            return
        guess, rule = vp.heuristic_volatility(candidate.query)
        volatility = guess or vp.UNKNOWN
        ttl = vp.ttl_for(volatility)
        created = candidate.created_at or self.migration_epoch
        try:
            self.collection.update(
                ids=[candidate.id],
                metadatas=[{
                    "summary": candidate.summary,
                    "canonical_query": candidate.query,
                    "volatility": volatility,
                    "created_at": created,
                    "expires_at": vp.expires_at(created, ttl),
                    "ttl_seconds": ttl,
                    "source_urls": json.dumps(candidate.source_urls),
                    "router_source": f"backfill:{rule}" if rule else "backfill:default",
                    "schema_version": SCHEMA_VERSION,
                    "hit_count": candidate.hit_count,
                    "last_hit_at": _now(),
                }],
            )
            log.info("backfilled %s as %s", candidate.id, volatility)
        except Exception as exc:
            log.debug("backfill failed for %s: %s", candidate.id, exc)

    def purge_expired(self, limit: int = 500, now: Optional[int] = None) -> int:
        now = now or _now()
        epoch = self.migration_epoch
        try:
            doomed = [
                entry_id
                for entry_id, _doc, meta in self.list_entries()[:limit]
                if now >= _normalize_meta(meta, epoch)["expires_at"]
            ]
        except Exception as exc:
            log.warning("purge failed: %s", exc)
            return 0
        if doomed:
            try:
                self.delete(doomed)
            except Exception as exc:
                log.warning("purge delete failed: %s", exc)
                return 0
        return len(doomed)

    # --- stats / admin ------------------------------------------------------

    def get_cache_stats(self) -> Dict[str, Any]:
        try:
            epoch = self.migration_epoch
            now = _now()
            by_volatility: Dict[str, int] = {}
            expired = legacy = 0
            ages: List[int] = []
            for _entry_id, _doc, raw_meta in self.list_entries():
                meta = _normalize_meta(raw_meta, epoch)
                by_volatility[meta["volatility"]] = by_volatility.get(meta["volatility"], 0) + 1
                if now >= meta["expires_at"]:
                    expired += 1
                if meta["schema_version"] < SCHEMA_VERSION:
                    legacy += 1
                if meta["created_at"]:
                    ages.append(now - meta["created_at"])
            total = sum(by_volatility.values())
            return {
                "total_queries": total,
                "fresh": total - expired,
                "expired": expired,
                "legacy_entries": legacy,
                "by_volatility": by_volatility,
                "avg_age_seconds": int(sum(ages) / len(ages)) if ages else 0,
                "schema_version": SCHEMA_VERSION,
                **self._backend_stats(),
            }
        except Exception as exc:
            log.warning("stats failed: %s", exc)
            return {"error": str(exc)}

    def _backend_stats(self) -> Dict[str, Any]:
        """Fields only this backend can report. Overridden per backend so the
        shared stats builder never touches storage internals."""
        return {
            "backend": "chroma",
            "collection_name": self.collection.name,
            "database_path": config.CHROMA_PATH,
        }

    def clear_cache(self) -> None:
        try:
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection.name,
                metadata={"hnsw:space": "cosine"},
            )
            self._migration_epoch = None
            log.info("cache cleared")
        except Exception as exc:
            log.warning("clear failed: %s", exc)

    def list_entries(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        """(id, document, metadata) for every entry, marker excluded."""
        got = self.collection.get()
        out = []
        for i, entry_id in enumerate(got.get("ids") or []):
            if entry_id == _MIGRATION_EPOCH_KEY:
                continue
            out.append((
                entry_id,
                (got.get("documents") or [""])[i] or "",
                (got.get("metadatas") or [{}])[i] or {},
            ))
        return out

    def delete(self, ids: Sequence[str]) -> None:
        self.collection.delete(ids=list(ids))

    # --- legacy single-candidate API ---------------------------------------

    def find_similar_query(self, new_query, threshold=None):
        """Nearest fresh neighbour above `threshold`, or (None, None).

        This is the no-LLM path: used when no provider is reachable, and by
        callers that predate the candidate API.
        """
        threshold = config.LEGACY_SIM_THRESHOLD if threshold is None else threshold
        fresh, _ = self.find_similar_candidates(new_query, k=1, floor=threshold)
        if not fresh:
            return None, None
        return fresh[0].summary, fresh[0].similarity


_cache_db = None
_cache_lock = threading.Lock()


def get_cache() -> "ChromaDBCache":
    """Return the shared cache, constructing it on first use."""
    global _cache_db
    if _cache_db is not None:
        return _cache_db
    with _cache_lock:
        if _cache_db is None:  # re-check under the lock
            if config.VECTOR_STORE == "upstash":
                _cache_db = UpstashCache()
            else:
                _cache_db = ChromaDBCache()
    return _cache_db


# --- module-level API --------------------------------------------------------


def find_similar_query(new_query, threshold=None):
    return get_cache().find_similar_query(new_query, threshold)


def find_similar_candidates(query, **kwargs):
    return get_cache().find_similar_candidates(query, **kwargs)


def add_to_cache(query, summary, **kwargs):
    return get_cache().add_to_cache(query, summary, **kwargs)


def touch(entry_id):
    return get_cache().touch(entry_id)


def backfill(candidate):
    return get_cache().backfill(candidate)


def purge_expired(limit=500):
    return get_cache().purge_expired(limit)


def get_cache_stats():
    return get_cache().get_cache_stats()


def clear_cache():
    return get_cache().clear_cache()


def view_all_cache():
    """All cached entries, newest first."""
    cache = get_cache()
    try:
        rows = cache.list_entries()
    except Exception as exc:
        log.warning("view failed: %s", exc)
        return []

    epoch = cache.migration_epoch
    now = _now()
    items = []
    for entry_id, document, raw_meta in rows:
        meta = _normalize_meta(raw_meta, epoch)
        summary = meta["summary"]
        items.append({
            "id": entry_id,
            "query": document,
            "summary": summary[:100] + "..." if len(summary) > 100 else summary,
            "full_summary": summary,
            "volatility": meta["volatility"],
            "created_at": meta["created_at"],
            "expires_at": meta["expires_at"],
            "ttl_seconds": meta["ttl_seconds"],
            "is_expired": now >= meta["expires_at"],
            "age_seconds": now - meta["created_at"] if meta["created_at"] else -1,
            "hit_count": meta["hit_count"],
            "source_urls": _decode_urls(meta["source_urls"]),
            "schema_version": meta["schema_version"],
        })
    items.sort(key=lambda it: -it["created_at"])
    return items


def search_cache(search_term):
    term = (search_term or "").lower()
    return [it for it in view_all_cache() if term in it["query"].lower()]


def delete_cache_item(query_id):
    try:
        get_cache().delete([query_id])
        return True
    except Exception as exc:
        log.warning("delete failed for %s: %s", query_id, exc)
        return False


def delete_cache_by_query(query_text):
    return sum(1 for it in search_cache(query_text) if delete_cache_item(it["id"]))


class UpstashCache(ChromaDBCache):
    """The same cache over Upstash Vector instead of local ChromaDB.

    Subclasses ChromaDBCache to inherit the parts that are storage-agnostic --
    TTL policy, metadata normalisation, legacy backfill, stats -- and replaces
    only the four operations that touch storage. Needed because serverless has
    no writable filesystem for chromadb.PersistentClient.
    """

    def __init__(self, store=None):
        import vector_store

        self.store = store or vector_store.UpstashVectorStore()
        self.collection = None  # nothing Chroma-shaped exists here
        self._migration_epoch = None

    @property
    def migration_epoch(self) -> int:
        """Pseudo-birthday for pre-TTL entries.

        Upstash indexes are created by this project, so nothing predates the
        schema and there is no real migration to date. Boot time is fine.
        """
        if self._migration_epoch is None:
            self._migration_epoch = _now()
        return self._migration_epoch

    def list_entries(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        return [
            (row["id"], row.get("document", ""), row.get("metadata") or {})
            for row in self.store.list_all()
        ]

    def delete(self, ids: Sequence[str]) -> None:
        self.store.delete(list(ids))

    def find_similar_candidates(
        self, query, k=None, floor=None, embedding=None,
        include_expired=False, now=None,
    ):
        k = k or config.RERANK_K
        floor = config.CANDIDATE_FLOOR if floor is None else floor
        now = now or _now()

        try:
            vector = list(embedding) if embedding is not None else embeddings.encode_one(query)
        except Exception as exc:
            log.warning("could not embed query: %s", exc)
            return [], []

        try:
            rows = self.store.query(vector, top_k=max(3 * k, 15))
        except Exception as exc:
            log.warning("upstash query failed: %s", exc)
            return [], []

        epoch = self.migration_epoch
        fresh: List[Candidate] = []
        expired: List[Candidate] = []
        for row in rows:
            # Upstash returns cosine similarity directly, not a distance.
            similarity = float(row["score"])
            if similarity < floor:
                continue
            meta = _normalize_meta(row.get("metadata") or {}, epoch)
            if not meta["summary"]:
                continue
            cand = Candidate(
                id=row["id"], query=row.get("document", ""), summary=meta["summary"],
                similarity=similarity, volatility=meta["volatility"],
                created_at=meta["created_at"], expires_at=meta["expires_at"],
                ttl_seconds=meta["ttl_seconds"],
                source_urls=_decode_urls(meta["source_urls"]),
                router_source=meta["router_source"],
                schema_version=meta["schema_version"], hit_count=meta["hit_count"],
            )
            (expired if cand.is_expired(now) else fresh).append(cand)

        fresh.sort(key=lambda c: -c.similarity)
        expired.sort(key=lambda c: -c.similarity)
        return fresh[:k], expired[:k]

    def add_to_cache(
        self, query, summary, volatility=vp.DEFAULT_VOLATILITY, ttl_seconds=None,
        canonical_query=None, source_urls=None, router_source="", embedding=None,
    ):
        volatility = vp.normalize_volatility(volatility)
        ttl = vp.ttl_for(volatility, ttl_seconds)
        if ttl <= 0:
            log.info("not caching %r (ttl 0 for %s)", query[:50], volatility)
            return None

        created = _now()
        try:
            vector = list(embedding) if embedding is not None else embeddings.encode_one(query)
        except Exception as exc:
            log.warning("could not embed for cache write: %s", exc)
            return None

        entry_id = self._find_exact(query) or str(uuid.uuid4())
        try:
            self.store.upsert(entry_id, vector, query, {
                "summary": summary,
                "canonical_query": canonical_query or query,
                "volatility": volatility,
                "created_at": created,
                "expires_at": vp.expires_at(created, ttl),
                "ttl_seconds": ttl,
                "source_urls": json.dumps(list(source_urls or [])),
                "router_source": router_source,
                "schema_version": SCHEMA_VERSION,
                "hit_count": 0,
                "last_hit_at": 0,
            })
        except Exception as exc:
            log.warning("upstash write failed: %s", exc)
            return None
        log.info("cached %r as %s (ttl %ds)", query[:50], volatility, ttl)
        return entry_id

    def touch(self, entry_id: str) -> None:
        """Record a hit.

        Upstash has no metadata-only update, so bumping a counter means
        re-sending the vector. Doing that on every cache hit would spend an
        embedding call and an upsert to increment an integer nobody reads in
        the hot path, and the free tier's daily budget is shared with real
        queries. Hit counts are therefore only maintained on the ChromaDB
        backend, where the update is local and free.
        """
        return

    def backfill(self, candidate: Candidate) -> None:
        if candidate.schema_version >= SCHEMA_VERSION:
            return
        guess, rule = vp.heuristic_volatility(candidate.query)
        volatility = guess or vp.UNKNOWN
        ttl = vp.ttl_for(volatility)
        created = candidate.created_at or self.migration_epoch
        try:
            # Re-embed rather than fetch with includeVectors: one call either
            # way, and this path is rare.
            vector = embeddings.encode_one(candidate.query)
            self.store.upsert(candidate.id, vector, candidate.query, {
                "summary": candidate.summary,
                "canonical_query": candidate.query,
                "volatility": volatility,
                "created_at": created,
                "expires_at": vp.expires_at(created, ttl),
                "ttl_seconds": ttl,
                "source_urls": json.dumps(candidate.source_urls),
                "router_source": f"backfill:{rule}" if rule else "backfill:default",
                "schema_version": SCHEMA_VERSION,
                "hit_count": candidate.hit_count,
                "last_hit_at": _now(),
            })
        except Exception as exc:
            log.debug("backfill failed for %s: %s", candidate.id, exc)

    def _backend_stats(self) -> Dict[str, Any]:
        return {"backend": "upstash"}

    def clear_cache(self) -> None:
        try:
            self.store.reset()
            self._migration_epoch = None
            log.info("upstash index reset")
        except Exception as exc:
            log.warning("clear failed: %s", exc)
