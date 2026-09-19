# Web Browser Query Agent

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

## Setup

Requires Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env     # then add a TAVILY_API_KEY, see below
```

Train the validity classifier (downloads ~90MB on first run):

```bash
python train_classifier.py
```

This writes `classifier.json` (the artifact the app actually loads),
`classifier.pkl`, and `embedding_model/`. Only `classifier.json` is required,
and it is committed — so you can skip this step unless you want to retrain.

Run it:

```bash
python main.py     # CLI
python app.py      # web UI at http://127.0.0.1:5000
```

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

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite makes **no network calls**. LLM responses are replayed from a
recorded cassette, and a stub embedder avoids downloading MiniLM in CI. One
test asserts the exact number of LLM calls made per query class, which is what
stops a refactor from quietly burning through the free tier.
