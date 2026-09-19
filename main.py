"""Command-line entry point. Same pipeline as the web app, printed."""

import logging
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from queryagent import pipeline

ICONS = {
    "validating": "🔎", "classified": "🏷️ ", "cache": "💾", "cache_miss": "💾",
    "searching": "🌐", "found": "✅", "scraping": "📄", "read": "📄",
    "summarizing": "✂️ ", "caching": "💾", "complete": "✨", "error": "❌",
}


def process(query: str, force_refresh: bool = False) -> None:
    for event in pipeline.process_query(query, force_refresh=force_refresh):
        icon = ICONS.get(event.stage, "  ")

        if event.stage == "error":
            print(f"{icon} {event.message}")
            return

        if event.stage == "read":
            print(f"{icon} {event.message}")
            for i, src in enumerate(event.data.get("sources", []), 1):
                print(f"     [{i}] via {src['via']:<10} {src['url']}")
            continue

        if event.stage == "complete":
            verdict = event.data.get("verdict") or {}
            print(f"\n{icon} {'From cache' if event.data.get('is_cached') else 'Fresh answer'}", end="")
            if event.data.get("similarity") is not None:
                print(f" (similarity {event.data['similarity']:.3f})", end="")
            print(f"\n\n📄 Summary:\n{event.data.get('summary', '')}")

            sources = event.data.get("sources") or []
            if sources:
                print("\n🔗 Sources:")
                for src in sources:
                    if isinstance(src, dict):
                        print(f"   - {src.get('title') or src.get('url')}")
                        print(f"     {src.get('url')}")
                    else:
                        print(f"   - {src}")

            if verdict:
                print(f"\n🏷️  {verdict.get('volatility')} "
                      f"(via {verdict.get('source')}) "
                      f"ttl={pipeline._human(verdict.get('ttl_seconds', 0))}")
            continue

        print(f"{icon} {event.message}")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")
    print("Web Browser Query Agent. Prefix a query with '!' to force a refresh.")
    while True:
        try:
            raw = input("\nEnter your query (or 'exit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue
        if raw.lower() in {"exit", "quit"}:
            break
        force = raw.startswith("!")
        process(raw.lstrip("!").strip(), force_refresh=force)


if __name__ == "__main__":
    main()
