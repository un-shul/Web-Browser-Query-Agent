# Web Browser Query Agent

[![tests](https://github.com/un-shul/Web-Browser-Query-Agent/actions/workflows/tests.yml/badge.svg)](https://github.com/un-shul/Web-Browser-Query-Agent/actions/workflows/tests.yml)

A query agent that answers natural-language questions by searching the web,
scraping the results, and summarising them — while reusing earlier answers from
a semantic cache when, and only when, they are still valid.

The interesting part is the decision layer in front of the cache, and it is
built on measurements rather than intuition.

Cosine similarity is better at some distinctions than you would guess and far
worse at others. Measured on `all-MiniLM-L6-v2`:

| cached query | new query | cosine | same answer? |
|---|---|---:|---|
| capital of France | capital of Italy | 0.464 | no |
| ceo of google | ceo of microsoft | 0.624 | no |
| install docker | uninstall docker | 0.765 | no |
| type 1 diabetes symptoms | type 2 diabetes symptoms | 0.806 | no |
| best laptops under 50000 | best laptops under 100000 | 0.867 | no |
| 2024 election results | 2025 election results | 0.914 | no |
| coffee **good** for health | coffee **bad** for health | 0.954 | no |
| flights delhi→mumbai | flights mumbai→delhi | 0.997 | no |

Different entities are separated cleanly. Polarity, direction and quantity are
not — and the bottom three sit *above* the 0.93 threshold at which a cache
would reuse an answer without a second thought. A single similarity threshold
cannot fix this: set it low and you serve wrong answers, set it high and you
never get a cache hit.

So the agent adds two checks:

- a **query router** that classifies how quickly an answer goes stale
  (`static` / `slow` / `dynamic` / `realtime`) and assigns a TTL, so
  time-sensitive queries are never served from cache
- an **LLM reranker** over the top-k candidates, which has to agree that a
  cached answer genuinely answers the new question before it is reused, with a
  free deterministic guard in front of it for the failure modes above

Both layers degrade to a logistic-regression classifier plus a fixed similarity
threshold when no LLM is reachable, so the app never hard-depends on an API key.

## Architecture

```
query
  │
  ├─ cheap gates (length, charset)                        no model
  ├─ logistic regression over MiniLM embeddings           local, instant
  ├─ memo: seen this query in the last hour?              local, instant
  ├─ regex volatility heuristics                          local, instant
  ├─ router: validity + volatility + TTL                  1 LLM call, skippable
  │
  ├─ semantic cache: top-5 candidates, TTL-filtered       vector store
  ├─ mismatch guard: polarity / direction / quantity      local, instant
  ├─ reranker: does a candidate really answer this?       1 LLM call, skippable
  │
  └─ on a miss: search → scrape → summarise → cache
```

Everything above the router exists to keep LLM calls off the common paths.
Measured over a 11-case benchmark of near-miss cache lookups, the agent gets
**11/11** correct using **4 LLM calls** — the other 7 decisions are settled for
free.

| query class | LLM calls |
|---|---|
| gibberish, commands, navigation | 0 |
| realtime by regex (*live score*, *stock price*) | 0 |
| repeat within the hour | 0 |
| near-identical cache hit | 0 |
| polarity or direction conflict | 0 |
| cold informational query | 1 |
| cold query with mid-similarity candidates | 2 |

Two is the ceiling. Against Groq's ~14,400 requests/day that is still 7,000+
queries, so the free tier is not the binding constraint. See `docs/` for the
full decision flow.

**Nothing hard-depends on an LLM.** Delete the `llm_gateway/` package and the
app still runs on the classifier, the regex heuristics and a fixed similarity
threshold — which is exactly how it behaved before these layers existed.

## Layout

```
app.py                  Flask app -- Vercel loads the top-level `app` from here
main.py                 CLI

queryagent/
  config.py             every environment variable, read in one place
  pipeline.py           the one query flow, shared by both entry points
  classifier.py         logistic-regression validity gate
  embeddings.py         local MiniLM or hosted Gemini, behind one interface
  volatility.py         staleness classes and the TTL policy (pure functions)
  search.py             Tavily search plus the scrape fallback chain
  cache/                semantic cache: __init__.py is ChromaDB, upstash.py is hosted
  summarize/            local.py is distilbart, hosted.py is an LLM
  llm/                  provider chain, router, verifier, mismatch guard

scripts/                train_classifier.py, preflight.py, cache_manager.py
data/                   datasets and the committed classifier artifact
templates/  static/     server-rendered UI
tests/                  383 tests, no network
```

## Setup

Requires Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-local.txt   # local model stack

cp .env.example .env     # then add a TAVILY_API_KEY, see below
```

`requirements.txt` holds only the four production dependencies; the local
model stack lives in `requirements-local.txt`. That split exists because
Vercel's Python bundle limit is 500 MB and torch alone is 547 MB.

Train the validity classifier (downloads ~90MB on first run):

```bash
python scripts/train_classifier.py
```

This writes `classifier.json` (the artifact the app actually loads),
`classifier.pkl`, and `embedding_model/`. Only `classifier.json` is required,
and it is committed — so you can skip this step unless you want to retrain.

Run it:

```bash
python main.py               # CLI
python app.py                # web UI at http://127.0.0.1:5000
python scripts/preflight.py  # check every backend with a live call
```

On macOS, Control Center's AirPlay Receiver also listens on port 5000. Flask
still binds to `127.0.0.1:5000` and wins, but if the page will not load, turn
AirPlay Receiver off in System Settings → General → AirDrop & Handoff, or run
`python app.py` behind `FLASK_RUN_PORT`.

`scripts/preflight.py` is worth running before any deploy. Every check makes a real
request, because the failures that matter are the ones a config file cannot
show: a retired model id, an index created with the wrong dimension, a key
that was never activated. Two of the three model ids this project started with
had been withdrawn by the time it was deployed.

## API keys

**One key is required.** DuckDuckGo and Google both block scraped requests now
(DuckDuckGo returns an anti-bot page, Google returns markup with no extractable
links), so web search needs a real search API.

| Service | Purpose | Free tier | Card? |
|---|---|---|---|
| [Tavily](https://tavily.com) | web search | 1,000/month | No |
| [Google AI Studio](https://aistudio.google.com/apikey) | LLM + embeddings | ~1,500/day | No |
| [Groq](https://console.groq.com/keys) | LLM fallback on 429 | ~14,400/day | No |
| [Upstash Vector](https://console.upstash.com) | hosted vector store (deploy only) | 1GB | No |

Everything runs at **$0**. Without the LLM keys the app still works — it falls
back to the classifier and a fixed similarity threshold, which is the behaviour
it had before the LLM layers existed.

Note that Google may use free-tier prompts to improve their models. Use Groq as
the primary if that matters to you.

## Configuration

Backends are swapped by environment variable, so the same code runs the local
model stack in development and hosted APIs in production:

| Variable | Values | Default |
|---|---|---|
| `LLM_PROVIDER` | `gemini` \| `groq` \| `chain` \| `fake` \| `none` | `none` |
| `EMBED_PROVIDER` | `gemini` \| `local` | `local` |
| `VECTOR_STORE` | `upstash` \| `chroma` | `chroma` |
| `SUMMARIZER` | `llm` \| `local` | `local` |

All defaults are the local stack, so a fresh clone works with no keys at all.
See `.env.example` for the full list.

## Deploying

The app runs two interchangeable stacks, selected by environment variable:

| | local (default) | production |
|---|---|---|
| Embeddings | all-MiniLM-L6-v2, 384d | `gemini-embedding-001`, 768d |
| Inference | — (heuristics only) | Gemini, Groq fallback |
| Summarising | distilbart | Gemini / Groq |
| Vector store | ChromaDB on disk | Upstash Vector |
| Install size | ~1.3 GB | ~25 MB |

Production is not a preference — it is a requirement. Serverless has no
writable filesystem, so `chromadb.PersistentClient` has nowhere to live, and
the local model stack is 2.6× the bundle limit on its own.

**1. Create a free Upstash Vector index** with **768 dimensions** and
**COSINE** distance. The dimension is fixed at creation and must equal
`GEMINI_EMBED_DIM`.

**2. Set the environment variables.** Two groups, and the distinction matters:

The five secrets are the same as your `.env`:

```
TAVILY_API_KEY  GEMINI_API_KEY  GROQ_API_KEY
UPSTASH_VECTOR_REST_URL  UPSTASH_VECTOR_REST_TOKEN
```

The four selectors are **different** from your `.env`, which holds local
values:

```
LLM_PROVIDER=chain  EMBED_PROVIDER=gemini
VECTOR_STORE=upstash  SUMMARIZER=llm
```

Do not bulk-copy `.env` into your host. Three of those four selectors point at
the local stack there, which cannot run on serverless -- and the resulting
failure looks like a missing dependency rather than a configuration mistake.

**3. Check the same configuration locally, then deploy:**

```bash
python scripts/preflight.py --prod
npx vercel --prod          # env changes need a redeploy to take effect
```

**4. Confirm it took.** `GET /healthz` reports whether each backend can
actually run, not just what it is set to:

```json
{"ok": true, "problems": [], "vector_store": "upstash",
 "embed_provider": "gemini", "cache_available": true}
```

A non-empty `problems` array names the variable to fix.

Vercel detects Flask from the top-level `app` in `app.py`; no handler wrapper
or `api/` directory is needed. `vercel.json` sets a 120 s ceiling and excludes
the local-stack files from the bundle.

### Thresholds are per-embedding-model

Cosine ranges differ enough between the two embedders that one set of numbers
cannot serve both. Measured on the same 13 query pairs:

| | MiniLM (384d) | Gemini (768d) |
|---|---|---|
| equivalent | 0.918 – 0.965 | 0.968 – 0.990 |
| quantity conflict | 0.806 – 0.914 | 0.962 – 0.971 |
| polarity conflict | 0.954 | 0.966 |
| direction conflict | 0.997 | 0.988 |
| different entity | 0.464 – 0.624 | 0.883 – 0.892 |
| unrelated | 0.016 – 0.052 | 0.652 – 0.704 |

Two things follow. MiniLM's 0.60 retrieval floor would admit every unrelated
pair under Gemini, where unrelated tops out at 0.704. And under Gemini the
equivalent and quantity-conflict bands *overlap* — 0.968 against 0.971 — so no
threshold can separate a real paraphrase from "under 50000" versus "under
100000". Auto-accept is therefore near-exact-match only in production, and the
verifier sees almost everything.

This is also why the deterministic mismatch guard matters more with better
embeddings rather than less: it compares query text, so it is unaffected by
whichever model sits behind it.

## Tests

```bash
pip install -r requirements-test.txt
pytest
```

CI runs three jobs on every push: the suite on Python 3.12 and 3.13, the
suite again with every non-loopback socket blocked, and an import of every
module the deployed function loads using only the production dependencies.

383 tests, and the suite makes **no network calls** — verified by running it
with every non-loopback socket blocked, not by assumption. An autouse fixture
disables the LLM, a stub embedder avoids downloading MiniLM, and
`HF_HUB_OFFLINE` stops huggingface_hub checking for model updates.

That last one was not theoretical. Before it was set the suite made 84
outbound requests and took 77 seconds; it now takes 7.

`scripts/run_tests_offline.py` enforces it: it blocks every non-loopback
socket and fails if anything tried to connect, even when the test itself
swallowed the error. CI runs it, because this property stopped being true
twice without anyone noticing.

Tests need none of the local model stack -- the embedder is stubbed and the
classifier loads from JSON -- so `requirements-test.txt` omits torch and
friends, keeping CI to 96 packages rather than 1.3GB.

`tests/test_quota_guard.py` asserts the exact number of LLM calls for each
query class, which is what stops a refactor from quietly burning through a
daily quota.
