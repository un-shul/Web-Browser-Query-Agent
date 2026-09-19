"""Vector-store backends behind one interface.

  chroma    on-disk ChromaDB. Development default; needs a writable filesystem.
  upstash   Upstash Vector over REST. Needed for serverless, where there is no
            writable disk, so chromadb.PersistentClient has nowhere to live.

Both speak the Candidate/metadata shape defined in cache_chromadb, so the
pipeline does not know or care which is active.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import config

log = logging.getLogger(__name__)


class VectorStoreError(RuntimeError):
    pass


class UpstashVectorStore:
    """Upstash Vector via its REST API.

    Plain requests rather than the upstash-vector SDK: four endpoints, and one
    fewer dependency in a bundle that has to stay under 500MB.
    """

    def __init__(self, url: Optional[str] = None, token: Optional[str] = None):
        self.url = (url or config.UPSTASH_VECTOR_REST_URL).rstrip("/")
        self.token = token or config.UPSTASH_VECTOR_REST_TOKEN
        if not self.url or not self.token:
            raise VectorStoreError(
                "VECTOR_STORE=upstash needs UPSTASH_VECTOR_REST_URL and "
                "UPSTASH_VECTOR_REST_TOKEN. Free index: https://console.upstash.com"
            )

    def _post(self, path: str, body: Any) -> Any:
        import requests

        try:
            resp = requests.post(
                f"{self.url}/{path.lstrip('/')}",
                headers={"Authorization": f"Bearer {self.token}",
                         "Content-Type": "application/json"},
                json=body, timeout=20,
            )
        except requests.RequestException as exc:
            raise VectorStoreError(f"upstash request failed: {exc}") from exc
        if resp.status_code == 401:
            raise VectorStoreError("upstash rejected the token (401)")
        if resp.status_code != 200:
            raise VectorStoreError(
                f"upstash {path} returned HTTP {resp.status_code}: {resp.text[:200]}"
            )
        return resp.json().get("result")

    # --- operations ---------------------------------------------------------

    def upsert(self, entry_id: str, vector: Sequence[float],
               document: str, metadata: Dict[str, Any]) -> None:
        # Upstash metadata is JSON, so the document rides along inside it and
        # there is no separate documents array to keep in step.
        payload = dict(metadata)
        payload["_document"] = document
        self._post("upsert", {"id": entry_id, "vector": list(vector), "metadata": payload})

    def query(self, vector: Sequence[float], top_k: int) -> List[Dict[str, Any]]:
        result = self._post("query", {
            "vector": list(vector), "topK": top_k,
            "includeMetadata": True, "includeVectors": False,
        }) or []
        out = []
        for row in result:
            meta = dict(row.get("metadata") or {})
            out.append({
                "id": row.get("id"),
                # Upstash returns a cosine *similarity* already normalised to
                # 0..1, not a distance, so it is not converted here.
                "score": float(row.get("score") or 0.0),
                "document": meta.pop("_document", ""),
                "metadata": meta,
            })
        return out

    def fetch(self, entry_id: str) -> Optional[Dict[str, Any]]:
        result = self._post("fetch", {"ids": [entry_id], "includeMetadata": True})
        rows = [r for r in (result or []) if r]
        if not rows:
            return None
        meta = dict(rows[0].get("metadata") or {})
        return {
            "id": rows[0].get("id"),
            "document": meta.pop("_document", ""),
            "metadata": meta,
        }

    def list_all(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Page through the whole index. Free tier caps at 1GB, so this is fine
        at portfolio scale; it is not a pattern for a large index."""
        out: List[Dict[str, Any]] = []
        cursor = "0"
        while cursor is not None and len(out) < limit:
            result = self._post("range", {
                "cursor": cursor, "limit": min(100, limit - len(out)),
                "includeMetadata": True,
            }) or {}
            for row in result.get("vectors") or []:
                meta = dict(row.get("metadata") or {})
                out.append({
                    "id": row.get("id"),
                    "document": meta.pop("_document", ""),
                    "metadata": meta,
                })
            cursor = result.get("nextCursor") or None
            if cursor == "":
                cursor = None
        return out

    def delete(self, ids: Sequence[str]) -> int:
        if not ids:
            return 0
        result = self._post("delete", {"ids": list(ids)}) or {}
        return int(result.get("deleted") or 0)

    def reset(self) -> None:
        self._post("reset", {})

    def count(self) -> int:
        result = self._post("info", {}) or {}
        return int(result.get("vectorCount") or 0)


def get_store() -> Optional[UpstashVectorStore]:
    """The configured hosted store, or None when running on ChromaDB."""
    if config.VECTOR_STORE != "upstash":
        return None
    return UpstashVectorStore()
