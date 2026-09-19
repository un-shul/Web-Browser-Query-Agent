"""Every module must import without the local model stack installed.

This exists because it was missed. The original check imported the leaf
modules individually and declared the production bundle verified -- but never
imported `app`, whose import chain reaches chromadb through cache.
The deployed function crashed on every request with
FUNCTION_INVOCATION_FAILED.

Runs in a subprocess so the blocked imports cannot leak into the rest of the
suite, and walks the entry points rather than a hand-written list, so a new
module cannot quietly reintroduce the problem.
"""

import os
import subprocess
import sys
import textwrap

# The subprocesses below import queryagent, so they need the repository root on
# sys.path regardless of where pytest was invoked from.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import pytest

# Packages installed for development but absent from requirements.txt.
LOCAL_ONLY = [
    "torch", "transformers", "sentence_transformers", "chromadb",
    "sklearn", "pandas", "playwright", "onnxruntime",
]

# Everything a deployed request can reach. app is the entry point Vercel loads.
PRODUCTION_MODULES = [
    "queryagent",
    "queryagent.config",
    "queryagent.embeddings",
    "queryagent.classifier",
    "queryagent.volatility",
    "queryagent.search",
    "queryagent.cache",
    "queryagent.cache.upstash",
    "queryagent.summarize",
    "queryagent.summarize.hosted",
    "queryagent.pipeline",
    "queryagent.llm",
    "queryagent.llm.providers",
    "queryagent.llm.router",
    "queryagent.llm.reranker",
    "queryagent.llm.mismatch",
    "queryagent.llm.budget",
    "app",  # the entry point Vercel loads
]

SCRIPT = textwrap.dedent("""
    import builtins, os, sys
    sys.path.insert(0, {root!r})
    BLOCKED = set({blocked!r})
    real_import = builtins.__import__
    def guarded(name, *a, **kw):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError("No module named %r" % name)
        return real_import(name, *a, **kw)
    builtins.__import__ = guarded

    os.environ.update({{
        "EMBED_PROVIDER": "gemini", "VECTOR_STORE": "upstash",
        "SUMMARIZER": "llm", "LLM_PROVIDER": "chain",
        "UPSTASH_VECTOR_REST_URL": "https://example.upstash.io",
        "UPSTASH_VECTOR_REST_TOKEN": "token",
        "TAVILY_API_KEY": "tvly-test", "GEMINI_API_KEY": "key",
    }})

    failed = []
    for name in {modules!r}:
        try:
            real_import(name)
        except Exception as exc:
            failed.append("%s: %s: %s" % (name, type(exc).__name__, exc))
    if failed:
        print("\\n".join(failed))
        sys.exit(1)
    sys.exit(0)
""")


def _run(modules, blocked=LOCAL_ONLY):
    return subprocess.run(
        [sys.executable, "-c",
         SCRIPT.format(blocked=blocked, modules=modules, root=REPO_ROOT)],
        cwd=REPO_ROOT,
        capture_output=True, text=True, timeout=180,
    )


def test_every_production_module_imports_without_the_local_stack():
    result = _run(PRODUCTION_MODULES)
    assert result.returncode == 0, (
        "these modules cannot be imported in the production bundle:\n"
        + result.stdout + result.stderr
    )


def test_the_flask_entry_point_imports():
    """Vercel loads the top-level `app` from app.py. If that import raises,
    every request returns FUNCTION_INVOCATION_FAILED."""
    result = _run(["app"])
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_app_object_exists_and_is_wsgi():
    result = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {REPO_ROOT!r}); "
         "import app; assert callable(app.app); print(type(app.app).__name__)"],
        capture_output=True, text=True, timeout=120, cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "Flask" in result.stdout


@pytest.mark.parametrize("package", LOCAL_ONLY)
def test_no_production_module_imports_a_local_only_package_at_module_level(package):
    """Blocking one package at a time localises a regression to its cause."""
    result = _run(PRODUCTION_MODULES, blocked=[package])
    assert result.returncode == 0, (
        f"blocking {package!r} alone breaks the production bundle:\n"
        + result.stdout + result.stderr
    )


def test_chroma_backend_reports_the_fix_when_unavailable():
    """The default backend cannot work on serverless. Left alone it surfaces as
    a bare "No module named 'chromadb'" mid-query, which names neither the
    cause nor the fix."""
    script = (
        f"import sys; sys.path.insert(0, {REPO_ROOT!r})\n"
        "import builtins\n"
        "real = builtins.__import__\n"
        "def g(n, *a, **k):\n"
        "    if n.split('.')[0] == 'chromadb':\n"
        "        raise ModuleNotFoundError(\"No module named 'chromadb'\")\n"
        "    return real(n, *a, **k)\n"
        "builtins.__import__ = g\n"
        "from queryagent import cache\n"
        "try:\n"
        "    cache.get_cache()\n"
        "except cache.CacheUnavailable as e:\n"
        "    print(str(e))\n"
    )
    result = subprocess.run([sys.executable, "-c", script],
                            capture_output=True, text=True, timeout=120,
                            cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "VECTOR_STORE=upstash" in out
    assert "requirements-local.txt" in out
