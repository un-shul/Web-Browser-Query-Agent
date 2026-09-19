#!/usr/bin/env python3
"""Check a deployment's configuration before pushing it.

Every check is a live call, because the failures that matter are the ones a
config file cannot show: a retired model id, an index created with the wrong
dimension, a key that was never activated.

    python preflight.py            # check the current environment
    python preflight.py --prod     # check it as production would be configured
"""

from __future__ import annotations

import argparse
import os
import sys

OK, WARN, FAIL = "ok  ", "warn", "FAIL"


def line(status: str, name: str, detail: str = "") -> None:
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prod", action="store_true",
                        help="check the hosted stack rather than the local one")
    args = parser.parse_args()

    if args.prod:
        os.environ.update({
            "EMBED_PROVIDER": "gemini", "VECTOR_STORE": "upstash",
            "SUMMARIZER": "llm", "LLM_PROVIDER": "chain",
        })

    import config

    failures = 0
    print(f"\nConfiguration ({'production' if args.prod else 'current'})")
    for key in ["LLM_PROVIDER", "EMBED_PROVIDER", "VECTOR_STORE", "SUMMARIZER"]:
        line(OK, f"{key}={getattr(config, key)}")
    line(OK, f"thresholds floor={config.CANDIDATE_FLOOR} "
             f"auto_accept={config.AUTO_ACCEPT_SIM} legacy={config.LEGACY_SIM_THRESHOLD}")

    print("\nSearch")
    if not config.TAVILY_API_KEY:
        line(FAIL, "TAVILY_API_KEY", "not set; web search cannot work")
        failures += 1
    else:
        import web_search
        try:
            bundle = web_search.search("preflight connectivity check", max_results=1)
            line(OK, "tavily", f"{len(bundle.results)} result(s)")
        except Exception as exc:
            line(FAIL, "tavily", str(exc)[:120])
            failures += 1

    print("\nInference")
    import llm_gateway.providers as providers
    providers.reset_chain()
    chain = providers.get_chain()
    if not chain:
        line(WARN, "no LLM provider", "router and verifier fall back to heuristics")
    for provider in chain:
        result, meta = providers.call_json(
            "Reply with JSON only.", "Return {\"ok\": true}. JSON:",
            {"type": "object", "properties": {"ok": {"type": "boolean"}},
             "required": ["ok"]},
            purpose="preflight",
        )
        if result is None:
            line(FAIL, provider.name, (meta.error or "no response")[:120])
            failures += 1
        else:
            line(OK, f"{provider.name} ({provider.model})", f"{meta.latency_ms}ms")
        providers.set_chain([p for p in chain if p is not provider] or [])
        providers.set_chain(chain)

    print("\nEmbeddings")
    import embeddings
    try:
        vector = embeddings.encode_one("preflight")
        expected = embeddings.embedding_dim()
        if expected and len(vector) != expected:
            line(FAIL, "dimension", f"got {len(vector)}, expected {expected}")
            failures += 1
        else:
            line(OK, f"{embeddings.provider_name()}", f"{len(vector)} dims")
    except Exception as exc:
        line(FAIL, "embeddings", str(exc)[:120])
        failures += 1
        vector = None

    print("\nVector store")
    if config.VECTOR_STORE == "upstash":
        import vector_store
        try:
            store = vector_store.UpstashVectorStore()
            count = store.count()
            line(OK, "upstash", f"reachable, {count} vector(s)")
            if vector:
                probe = store.query(vector, top_k=1)
                line(OK, "upstash query", f"{len(probe)} row(s)")
        except Exception as exc:
            msg = str(exc)
            line(FAIL, "upstash", msg[:150])
            if "dimension" in msg.lower():
                line(WARN, "hint",
                     f"create the index with {embeddings.embedding_dim()} dimensions "
                     "and COSINE distance")
            failures += 1
    else:
        try:
            import cache_chromadb
            line(OK, "chroma", f"{cache_chromadb.get_cache_stats().get('total_queries', 0)} entries")
        except Exception as exc:
            line(FAIL, "chroma", str(exc)[:120])
            failures += 1

    print("\nClassifier")
    import agent
    if agent.is_available():
        _, p = agent.classify_query_with_confidence("what is photosynthesis")
        if p is None:
            line(WARN, "classifier loaded but unusable",
                 "embedding dimension does not match the trained artifact")
        else:
            line(OK, "classifier", f"p_valid={p:.3f}")
    else:
        line(WARN, "no classifier artifact", "the router handles validity alone")

    print()
    if failures:
        print(f"{failures} check(s) failed.\n")
        return 1
    print("All checks passed.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
