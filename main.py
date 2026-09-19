"""Command-line entry point.

Same pipeline as the web app, without the SSE plumbing. Useful for testing the
agent without a browser.
"""

import logging
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from agent import is_junk
from cache_chromadb import add_to_cache, find_similar_query
from summarizer import summarize_text
from web_search import SearchError, fetch_contents, search


def process(query: str) -> None:
    # Step 1: cheap validity gate
    if is_junk(query):
        print("❌ That does not look like a searchable query.")
        return

    # Step 2: semantic cache
    cached, similarity = find_similar_query(query)
    if cached:
        print(f"✅ Cached result (similarity {similarity:.2f})")
        print("\n📄 Summary:\n" + cached)
        return

    # Step 3: search
    print("🌐 Searching...")
    try:
        bundle = search(query)
    except SearchError as exc:
        print(f"❌ {exc}")
        return
    if not bundle.results:
        print("⚠️  No search results found.")
        return

    # Step 4: fetch pages concurrently
    print(f"📄 Reading {len(bundle.results)} pages...")
    pages = fetch_contents(bundle.results)
    if not pages:
        print("⚠️  Could not extract content from any page.")
        return
    for i, page in enumerate(pages, 1):
        print(f"   [{i}] via {page.via:<10} {len(page.text):>6} chars  {page.url}")

    # Step 5: summarise
    print("\n✂️  Summarising...")
    try:
        summary = summarize_text("\n\n".join(p.text[:5000] for p in pages), query)
    except Exception as exc:
        print(f"❌ Summarisation failed: {exc}")
        return

    # Step 6: cache and report
    add_to_cache(query, summary)
    print("\n📄 Summary:\n" + summary)
    print("\n🔗 Sources:")
    for page in pages:
        print(f"   - {page.title or page.url}")
        print(f"     {page.url}")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")
    while True:
        try:
            query = input("\nEnter your query (or 'exit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break
        process(query)


if __name__ == "__main__":
    main()
