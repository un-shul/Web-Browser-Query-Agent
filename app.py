"""Flask web app.

Thin layer over pipeline.process_query: this module handles HTTP and SSE, and
the pipeline owns the query logic. It used to carry two full copies of that
logic (the SSE generator and the JSON endpoint), with a third in main.py.
"""

import json
import logging
import os

# Must be set before any tokenizer is constructed.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from flask import Flask, Response, jsonify, render_template, request

from queryagent import cache as cache
from queryagent import config
from queryagent import pipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

app = Flask(__name__)
# Jinja ties template auto-reload to debug mode, so with debug off an edited
# template keeps serving from the compiled cache until restart. Harmless in
# production, confusing in development.
app.config["TEMPLATES_AUTO_RELOAD"] = config.FLASK_DEBUG or bool(
    os.environ.get("TEMPLATES_AUTO_RELOAD")
)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/cache")
def cache_page():
    """Cache explorer. The /cache-* endpoints predate this and had no UI."""
    return render_template("cache.html")


@app.route("/search_progress")
def search_progress():
    """Stream pipeline progress as Server-Sent Events."""
    query = request.args.get("query", "").strip()
    force = request.args.get("refresh", "").lower() in {"1", "true", "yes"}

    if not query:
        return Response(
            _sse({"stage": "error", "message": "Please enter a query", "progress": 0}),
            mimetype="text/event-stream",
        )

    def stream():
        try:
            for event in pipeline.process_query(query, force_refresh=force):
                yield _sse(event.to_dict())
        except cache.CacheUnavailable as exc:
            # A configuration problem, not a bug. Pass the guidance through
            # rather than burying it under "Unexpected error".
            app.logger.error("cache backend unavailable: %s", exc)
            yield _sse({"stage": "error", "message": str(exc), "progress": 0})
        except Exception as exc:  # never leave the client hanging
            app.logger.exception("pipeline failed")
            yield _sse({"stage": "error", "message": f"Unexpected error: {exc}", "progress": 0})

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # stop nginx-style proxies buffering the stream
        },
    )


@app.route("/search", methods=["POST"])
def search():
    """Non-streaming equivalent, for clients that cannot use SSE."""
    query = (request.form.get("query") or (request.json or {}).get("query", "")).strip()
    if not query:
        return jsonify({"error": "Please enter a query"}), 400

    force = str(request.form.get("refresh", "")).lower() in {"1", "true", "yes"}
    result = pipeline.run(query, force_refresh=force)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


# --- cache administration ----------------------------------------------------


@app.route("/cache-stats")
def cache_stats():
    return jsonify(cache.get_cache_stats())


@app.route("/cache-view")
def cache_view():
    items = cache.view_all_cache()
    return jsonify({"total_items": len(items), "items": items})


@app.route("/cache-search")
def cache_search():
    term = request.args.get("q", "")
    if not term:
        return jsonify({"error": "Provide a search term with ?q="}), 400
    items = cache.search_cache(term)
    return jsonify({"search_term": term, "found_items": len(items), "items": items})


@app.route("/cache-delete-item", methods=["POST"])
def cache_delete_item():
    data = request.get_json(silent=True) or {}
    if "id" not in data:
        return jsonify({"error": "Provide an item id"}), 400
    if cache.delete_cache_item(data["id"]):
        return jsonify({"message": "Deleted"})
    return jsonify({"error": "Delete failed"}), 500


@app.route("/cache-delete-query", methods=["POST"])
def cache_delete_query():
    data = request.get_json(silent=True) or {}
    if "query" not in data:
        return jsonify({"error": "Provide query text"}), 400
    count = cache.delete_cache_by_query(data["query"])
    return jsonify({"message": f"Deleted {count} items", "deleted_count": count})


@app.route("/cache-purge", methods=["POST"])
def cache_purge():
    """Drop entries whose TTL has elapsed."""
    return jsonify({"purged": cache.purge_expired()})


@app.route("/cache-clear", methods=["POST"])
def cache_clear():
    try:
        cache.clear_cache()
        return jsonify({"message": "Cache cleared"})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/healthz")
def healthz():
    """Which backends are wired up, and whether each can actually run.

    Reports reachability rather than just configuration, so a deployment whose
    environment variables were never set is diagnosable from one request
    instead of from a failed query.
    """
    from queryagent import classifier
    from queryagent import embeddings

    problems = []
    if not config.TAVILY_API_KEY:
        problems.append("TAVILY_API_KEY is not set; web search cannot work")

    cache_ok = True
    try:
        cache.get_cache()
    except Exception as exc:
        cache_ok = False
        problems.append(str(exc).splitlines()[0])

    if not embeddings.is_available():
        problems.append(
            f"EMBED_PROVIDER={config.EMBED_PROVIDER} is not usable here"
        )

    return jsonify({
        "ok": not problems,
        "problems": problems,
        "cache_available": cache_ok,
        "search_configured": bool(config.TAVILY_API_KEY),
        "llm_provider": config.LLM_PROVIDER,
        "embed_provider": config.EMBED_PROVIDER,
        "queryagent.upstash": config.VECTOR_STORE,
        "queryagent.local": config.SUMMARIZER,
        "classifier_loaded": classifier.is_available(),
        "embedder_available": embeddings.is_available(),
    })


if __name__ == "__main__":
    # Debug mode is opt-in via FLASK_DEBUG; it must never be on in production,
    # where it would expose an interactive debugger.
    app.run(debug=config.FLASK_DEBUG)
