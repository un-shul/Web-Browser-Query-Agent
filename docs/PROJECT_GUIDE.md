# Project Guide

A complete reference for the Web Browser Query Agent: what it does, why each
part is built the way it is, every measurement behind a design decision, and
the questions the design invites.

Written to be read cold, months later, with no other context.

---

## Contents

1. [The short version](#1-the-short-version)
2. [Plain-language explanation](#2-plain-language-explanation)
3. [Key terms](#3-key-terms)
4. [The problem this solves](#4-the-problem-this-solves)
5. [Architecture](#5-architecture)
6. [The measurements](#6-the-measurements)
7. [Design decisions and why](#7-design-decisions-and-why)
8. [Cost and rate limits](#8-cost-and-rate-limits)
9. [Deployment](#9-deployment)
10. [Testing](#10-testing)
11. [Bugs found, and what each taught](#11-bugs-found-and-what-each-taught)
12. [Known weaknesses](#12-known-weaknesses)
13. [Questions this design invites](#13-questions-this-design-invites)
14. [Numbers worth remembering](#14-numbers-worth-remembering)
15. [Operating it](#15-operating-it)

---

## 1. The short version

Ask a question. The agent searches the web, reads the top results, and returns
one summarised answer with its sources. Ask something similar later and it
answers from a cache instead of searching again — but only when reusing the old
answer is actually safe.

"Actually safe" is the whole project. Two things have to hold:

- **The answer must not have expired.** "What is photosynthesis" is safe to
  reuse for months. "Live cricket score" is not safe to reuse for 90 seconds.
- **The cached question must genuinely be the same question.** Embedding
  similarity says "is coffee good for your health" and "is coffee **bad** for
  your health" are 0.95 similar. They are opposite questions.

Everything else — the search, the scraping, the summarising — is plumbing.

**Live:** https://web-browser-query-agent.vercel.app
**Stack:** Python, Flask, Gemini/Groq, Tavily, Upstash Vector, ChromaDB
**Cost:** $0/month on free tiers, no credit card

---

## 2. Plain-language explanation

### What a semantic cache is

A normal cache is keyed on exact text. Ask "what is photosynthesis" twice and
the second one is free; ask "explain photosynthesis" and it is a miss, even
though it is the same question.

A **semantic** cache keys on *meaning*. Each question is converted into a list
of numbers (an **embedding**) that encodes what it is about. Two questions with
similar meanings get similar number-lists. So "explain photosynthesis" finds
the entry stored under "what is photosynthesis" and reuses it.

The comparison is **cosine similarity**: 1.0 means identical direction, 0.0
means unrelated. In this project that number ranges roughly 0.0–1.0 with
MiniLM and 0.65–1.0 with Gemini (see §6 — the ranges differ a lot, and that
matters).

### Why that is not enough

Two independent failure modes.

**Staleness.** A semantic cache has no clock. It will happily serve last
month's cricket score to today's question, because the *question* is identical
and the cache has no notion that the *answer* rotted. Fixed by classifying how
fast each answer goes stale and attaching a time-to-live (TTL).

**False matches.** Embeddings are good at topic and bad at some distinctions
that completely change the answer. Measured on this project's own embedder:

- "is coffee **good** for your health" vs "**bad** for your health" → **0.95**
- "flights from **Delhi to Mumbai**" vs "**Mumbai to Delhi**" → **0.997**

Both are near-identical to the embedding and different questions to a human.
No similarity threshold can separate them, because genuine paraphrases score in
the same range. Fixed by two layers in front of the cache: a free
pattern-matching guard for these known blind spots, and an LLM asked to
confirm a borderline match really answers the new question.

### The flow, in words

1. Is this even a question? (too short, no letters, gibberish → reject, free)
2. Have I answered this exact thing in the last hour? (→ reuse, free)
3. Does it look time-sensitive by pattern? ("live score", "stock price" →
   mark realtime, free)
4. Otherwise ask an LLM: is this valid, and how fast does the answer go stale?
   (one API call)
5. Look in the cache for similar questions, ignoring expired entries.
6. Is the best match obviously the same question, obviously a known trap, or
   borderline? Only borderline costs an LLM call to verify.
7. On a miss: search the web, fetch the pages, summarise, store the answer with
   its TTL.

---

## 3. Key terms

| Term | Meaning here |
|---|---|
| **Embedding** | A list of numbers representing a piece of text's meaning. 384 numbers with MiniLM, 768 with Gemini. |
| **Cosine similarity** | How aligned two embeddings are. 1.0 = same direction, 0.0 = unrelated. The cache's matching metric. |
| **Vector store** | A database that finds the embeddings closest to a query embedding. ChromaDB locally, Upstash Vector in production. |
| **Semantic cache** | Cache keyed on meaning rather than exact text. |
| **TTL** | Time to live. How long a cached answer stays valid. |
| **Volatility** | This project's term for how fast an answer goes stale: static / slow / dynamic / realtime. |
| **Router** | The step that decides validity and volatility for a query. |
| **Reranker / verifier** | The step that decides whether a retrieved cache candidate genuinely answers the new question. |
| **Top-k** | Retrieving the k nearest candidates rather than only the single nearest. k = 5 here. |
| **Logistic regression** | A simple linear classifier. Used here as a cheap first filter for unusable queries. |
| **SSE** | Server-Sent Events. One-way streaming from server to browser, used for live progress. |
| **Cold start** | The first request to a serverless function, which must load the code before responding. |
| **Free tier** | The no-cost usage allowance of an API. The binding constraint on this project's design. |
| **Deuteranopia** | Red-green colour blindness. Relevant because the UI's accept/reject colours are indistinguishable under it, so they carry icons and words too. |

---

## 4. The problem this solves

### What it looked like before

The project existed and did not run. Two independent failures:

**It could not start.** Two modules loaded a sentence-transformer from a
directory that was in `.gitignore` and absent from disk, and the classifier was
loaded from a `.pkl` that was also gitignored. Importing the entry point raised
before Flask started.

**It could not search.** Both backends scraped search engines rather than using
an API, and both were blocked. Verified by request: DuckDuckGo's HTML endpoint
returned **HTTP 202 with an anti-bot page** (3 of 3 attempts, including the
`lite` variant), and Google returned **HTTP 200 with zero extractable result
links**. The search function returned an empty list on every call, so the
pipeline always terminated at "No search results found."

Neither was fixable by tweaking headers. Search needed a real API.

### What the cache did wrong

The original matching rule was one line: take the single nearest neighbour, and
accept it if cosine similarity ≥ 0.75. Three problems:

1. **No clock.** Metadata held only the summary — no timestamp, no TTL. So a
   realtime answer was reusable forever.
2. **`n_results=1`.** Only the single nearest neighbour was ever considered, so
   a fresh entry ranked second was invisible behind a stale one ranked first.
3. **A single threshold cannot work.** Set it low and you serve wrong answers;
   set it high and you never get a hit. §6 shows why: the "same question" and
   "different question" bands overlap.

---

## 5. Architecture

```
app.py                  Flask app. Vercel loads the top-level `app` from here.
main.py                 CLI. Same pipeline, printed instead of streamed.

queryagent/
  config.py             Every environment variable, read in one place.
  pipeline.py           The one query flow, yielding progress events.
  classifier.py         Logistic-regression validity gate.
  embeddings.py         Local MiniLM or hosted Gemini, behind one interface.
  volatility.py         Staleness classes and the TTL policy. Pure functions.
  search.py             Tavily search plus the page-content fallback chain.
  cache/
    __init__.py         ChromaDB backend (local).
    upstash.py          Upstash Vector backend (production).
  summarize/
    local.py            distilbart. Offline, no key.
    hosted.py           Gemini/Groq. Required on serverless.
  llm/
    providers.py        Provider chain, retries, circuit breaker, JSON repair.
    router.py           Validity + volatility in one call.
    reranker.py         Cache-candidate verification.
    mismatch.py         Deterministic checks for embedding blind spots.
    prompts.py          Prompt text and few-shot examples.
    schemas.py          Response schemas for structured output.
    budget.py           Rate-limit bookkeeping.

scripts/                train_classifier, preflight, cache_manager,
                        run_tests_offline
data/                   Datasets and the committed classifier artifact.
templates/  static/     Server-rendered UI.
tests/                  386 tests, no network.
```

### The decision flow, precisely

Each step's cost in LLM calls is marked. Every early exit is a call not spent.

```
STEP 0  Cheap gates                                        [0 calls]
        length < 3 or > 400, or no alphabetic character -> invalid.
        No model, no embedding, no network.

STEP 1  Memo                                               [0 calls]
        In-process LRU keyed on the normalised query, 1 hour TTL.
        A repeat inside the hour reuses the previous verdict.

STEP 2  Logistic-regression gate                           [0 calls]
        Reject only when p(valid) < 0.05. Deliberately NOT the model's
        own 0.5 boundary -- see §7.

STEP 3  Regex volatility heuristics                        [0 calls]
        13 rules. A realtime match settles the query outright, which is
        the highest-value skip: "live score", "stock price", "weather"
        are common and now cost nothing.

STEP 4  Router                                             [1 call]
        One call decides validity AND volatility. Splitting them would
        double the spend for no extra information.
        The regex result is kept as an escalation floor: the model may
        raise volatility but never lower it below what a pattern matched.

STEP 5  Cache retrieval                                    [0 calls]
        realtime -> skip the semantic lookup entirely. Only the query's
        own 90s entry can serve it, so it can never be answered from a
        different question's entry.
        Otherwise: top-5 candidates above the floor, expired ones dropped.

STEP 6  Verification                                       [0 or 1 call]
        hard conflict (polarity/direction)  -> reject       [0]
        similarity >= auto-accept threshold -> accept       [0]
        no candidates                       -> miss         [0]
        anything else                       -> ask the LLM  [1]

STEP 7  Search, fetch, summarise                           [0 or 1 call]
        Tavily search -> parallel page fetch -> summarise.
        The hosted summariser is one more call; the local one is free.

STEP 8  Write back
        Store with the TTL from STEP 4 -- unless the summariser reported
        that the pages did not answer the question, in which case nothing
        is stored.
```

### Volatility classes and TTLs

| Class | Default TTL | Min | Max | Example |
|---|---|---|---|---|
| `static` | 180 days | 7 d | 365 d | what is photosynthesis |
| `slow` | 14 days | 1 d | 60 d | best laptops for programming |
| `dynamic` | 6 hours | 15 m | 1 d | latest AI news |
| `realtime` | 90 seconds | 0 | 5 m | live cricket score |
| `unknown` | 3 days | 1 h | 7 d | entries predating TTLs |

The model may suggest a TTL, but it is **advisory only** — the server clamps it
into the band for its class. Models are inconsistent at arithmetic, and this
number decides whether stale answers get served.

`realtime` gets 90 seconds rather than 0 so that a double-click or an
`EventSource` reconnect does not trigger a second full search, while nothing
meaningfully stale is ever served.

### The content fallback chain

Per URL, in order:

1. `requests` + BeautifulSoup — a real scrape gets the whole article
2. Playwright, if enabled — for pages that render via JavaScript (local only;
   cannot run on serverless)
3. Tavily's `raw_content` from the original search response

Step 3 earns its place twice over. In testing, 4 of 5 pages scraped directly
and one (Britannica) yielded nothing and was recovered from `raw_content`. It
matters more in production, where datacenter IPs are blocked more aggressively,
and it is free because the search call already paid for it.

---

## 6. The measurements

This is the most important section. Every threshold in the project comes from
here rather than from intuition.

### Where cosine similarity misleads

Measured on `all-MiniLM-L6-v2`, the local embedder:

| Cached question | New question | Cosine | Same answer? |
|---|---|---:|---|
| capital of France | capital of Italy | 0.464 | No |
| ceo of google | ceo of microsoft | 0.624 | No |
| how tall is Everest | who first climbed Everest | 0.637 | No |
| python tutorial | python decorators | 0.404 | No |
| how to **install** docker | how to **uninstall** docker | **0.765** | No |
| type **1** diabetes symptoms | type **2** diabetes symptoms | **0.806** | No |
| laptops under **50000** | laptops under **100000** | **0.867** | No |
| **2024** election results | **2025** election results | **0.914** | No |
| coffee **good** for health | coffee **bad** for health | **0.954** | No |
| flights **Delhi→Mumbai** | flights **Mumbai→Delhi** | **0.997** | No |
| what is photosynthesis | explain photosynthesis | 0.918 | Yes |
| capital of France | which city is France's capital | 0.962 | Yes |
| how does a car engine work | explain how car engines work | 0.947 | Yes |

**The conclusion.** Different *entities* separate cleanly — France/Italy at
0.464 poses no problem. But polarity, direction and quantity do not separate at
all, and the last three sit **above** the 0.93 threshold at which a cache would
reuse an answer without a second thought. Genuine paraphrases score 0.918–0.962,
which **overlaps** the false matches. No single threshold works.

This also corrected a wrong assumption. The plan for this work assumed
France/Italy would be around 0.8 and defeat the old 0.75 rule. It measures
0.464 — the old rule rejected it correctly. The real failures were elsewhere,
and only measuring found them.

### Thresholds differ by embedding model

The same 13 pairs, compared across both embedders:

| | MiniLM (384d) | Gemini (768d) |
|---|---|---|
| equivalent questions | 0.918 – 0.965 | 0.968 – 0.990 |
| quantity conflict | 0.806 – 0.914 | 0.962 – 0.971 |
| polarity conflict | 0.954 | 0.966 |
| direction conflict | 0.997 | 0.988 |
| different entity | 0.464 – 0.624 | 0.883 – 0.892 |
| unrelated | 0.016 – 0.052 | 0.652 – 0.704 |

Two consequences:

1. **MiniLM's 0.60 retrieval floor would admit every unrelated pair under
   Gemini**, where unrelated tops out at 0.704. Every query would retrieve
   every cached entry as a candidate.
2. **Under Gemini the equivalent and quantity bands overlap** — 0.968 against
   0.971 — so no auto-accept threshold can separate a real paraphrase from
   "under 50000" vs "under 100000".

So thresholds are per-provider, and auto-accept in production is
near-exact-match only:

| | floor | auto-accept | no-LLM fallback |
|---|---|---|---|
| MiniLM (local) | 0.60 | 0.93 | 0.75 |
| Gemini (production) | 0.80 | 0.995 | 0.93 |

**The counter-intuitive result:** the deterministic guard matters *more* with
the better embedding model, not less. It compares query text, so it is
unaffected by which model sits behind it — and it is the only thing left
separating those cases once the embeddings stop distinguishing them.

### Verification benchmark

11 near-miss cache lookups, realistic summary bodies, production stack:

| New question | Cosine | Decision | LLM calls |
|---|---:|---|---:|
| is coffee bad for your health | 0.954 | mismatch_reject | 0 |
| best way to uninstall docker | 0.774 | mismatch_reject | 0 |
| disadvantages of remote work | 0.929 | mismatch_reject | 0 |
| flights from mumbai to delhi | 0.997 | mismatch_reject | 0 |
| 2025 election results | 0.914 | llm_reject | 1 |
| best laptops under 100000 | 0.867 | llm_reject | 1 |
| symptoms of type 2 diabetes | 0.806 | llm_reject | 1 |
| explain photosynthesis | 0.918 | llm_accept | 1 |
| explain how car engines work | 0.947 | auto_accept | 0 |
| which city is France's capital | 0.962 | auto_accept | 0 |
| what is the height of mount everest | 0.965 | auto_accept | 0 |

**11/11 correct, 4 LLM calls.** Seven decisions settled for free.

### Why polarity and direction are decided without the LLM

The verifier was asked and got both wrong, repeatedly.

**Polarity.** Given a benefits-only summary of "is coffee good for your
health" and asked whether it answers "is coffee **bad** for your health", it
accepted **three times running** — including with the conflict named explicitly
in the prompt and an excerpt that plainly discussed only benefits. Its reason:
a health overview "covers both risks and benefits."

**Direction.** For "flights Delhi→Mumbai" against "Mumbai→Delhi" it accepted
because flights are "bidirectional with the same duration and airlines." True
of duration. False of schedules and fares.

**Quantity it judges correctly** — it rejected 2024→2025 and 50000→100000 — so
that stays its decision.

The asymmetry justifies deciding the first two locally: wrongly rejecting costs
one web search; wrongly accepting answers a different question than the one
asked. Both checks are narrow (an explicit list of 31 antonym pairs, and an
exact token-multiset match with changed order), so false positives are rare.

### Latency

Measured on the deployed stack:

| Path | Time |
|---|---|
| Cold query (search + 5 pages + summarise) | 14.4 s |
| Cache hit via the verifier | 5.6 s |
| Realtime query (cache deliberately bypassed) | 13.0 s |

**Note what the cache actually saves.** The router call happens either way, so
the saving is not LLM calls — it is the search, five page fetches, and the
summarisation. That is the real argument for the cache, and it is stronger than
any call count.

---

## 7. Design decisions and why

### The logistic-regression gate rejects at p < 0.05, not 0.5

The classifier reports 99.5% cross-validated accuracy, and that number is
misleading. `augmented_query_dataset.csv` is **templated**: valid rows are
question forms ("What is X", "How to X") and invalid rows are commands
("Navigate to X", "Turn on X"). So the model learned *"is this phrased as a
question?"*, not *"is this searchable?"*

Bare noun-phrase queries are neither, and land mid-range:

| Query | p(valid) |
|---|---:|
| weather in bangalore | 0.397 |
| tesla stock price today | 0.399 |
| live cricket score india vs australia | 0.469 |

All three are perfectly good queries and all three fall below 0.5. The original
code gated on the 0.5 boundary, so the app answered *"Invalid query"* to
exactly the realtime questions the caching feature exists to handle.

The gate is therefore asymmetric: reject only below 0.05, which still catches
punctuation-only input (0.030) and "set an alarm for 7am" (0.035) while letting
anything ambiguous through. Some junk survives and costs one LLM call; that
beats refusing to answer valid questions.

**If you quote the accuracy anywhere, say "99.5% on a held-out split of a 2,000-row
synthetic dataset."** The number is real; it just does not mean generalisation.

### Escalation: heuristics may raise volatility, never lower it

After the router answers, `volatility = max(llm_verdict, regex_floor)` on the
ordering static < slow < dynamic < realtime.

This makes the highest-severity failure — the model calling "live cricket
score" static, and the cache then serving a month-old score — **structurally
impossible** rather than merely unlikely. Tested as a property across all 16
class pairs.

The asymmetry is deliberate: over-classifying costs a redundant web fetch
(slower, never wrong); under-classifying serves a stale answer. Every tie-break
leans toward freshness.

### Realtime queries skip the cache lookup entirely

Not "a very high threshold" — the semantic lookup is not performed at all. Only
the query's own 90-second entry can serve it. So a realtime query can never be
answered from a *different* question's entry, whatever the cosine score. The
guarantee is structural, with no threshold to tune wrong.

### Retrieval floor dropped from 0.75 to 0.60

Recall moved to the verifier. A low floor with a strict verifier finds fresher
entries that a high floor would miss, and the verifier catches what the floor
used to. Over-fetching also fixed the `n_results=1` bug: a fresh entry ranked
second is now reachable.

### One LLM call decides validity and volatility together

Splitting them would double the spend for no extra information. The single
biggest quota decision in the project.

### `call_json` never raises

It returns `None` when every provider fails. So each feature has exactly one
degraded branch to write, instead of a try/except at every call site. That
single property is what makes both LLM layers optional.

**The hard rule:** no feature may hard-depend on an LLM being reachable. Delete
the whole `llm/` package and the app still runs on the classifier, the regex
heuristics and a fixed threshold — which is exactly how it behaved before those
layers existed.

### The circuit breaker is reactive, not predictive

On a 429, a per-provider breaker opens until `Retry-After` elapses and the
chain moves on. A predictive token bucket would assume a long-lived warm
process, which is wrong for serverless: every cold container would start with a
full bucket and cheerfully re-exceed a limit the provider is still enforcing.

### Malformed JSON is repaired locally, never with a second call

Strip code fences, extract the first balanced `{…}`, coerce types. A repair
round-trip would be another request against the free tier — a bad trade when
the deterministic fallback is free.

### Gemini's model id is the `-latest` alias, deliberately

Pinned ids retire. `gemini-2.0-flash` was the default here and now returns 404
"no longer available"; `llama-3.3-70b-versatile` is no longer served by Groq.
An alias may shift behaviour under you, but for JSON classification that beats
an app that stops working.

### Truncated Gemini embeddings are L2-renormalised

Upstash's free tier caps indexes at 1,536 dimensions and
`gemini-embedding-001` defaults to 3,072, so output is truncated to 768. The
Matryoshka prefix is not unit-length, and cosine distance in the store assumes
it is.

### Low-confidence answers are shown but not cached

The summariser reports whether the pages actually answered the question. A "the
extracts do not cover this" response describes a failed fetch, not the world.
Caching it would serve that failure to every paraphrase for the entry's full
TTL — 30 days in practice — when a retry might pick different sources and
succeed. It is shown to the user (they asked, and "I could not find this" is a
legitimate reply) and marked *incomplete — not cached*.

### Hit counts are not maintained on Upstash

Upstash has no metadata-only update, so incrementing a counter means
re-embedding the query and re-sending the vector: two calls against a shared
daily budget, per cache hit, for a number nothing reads in the hot path.
ChromaDB keeps them, because there the update is local and free.

### The UI's accept/reject never relies on colour

A palette validator measured green against red at **ΔE 4.1 under
deuteranopia** — a red-green colourblind reader cannot distinguish them. So
every such badge carries a glyph and a word; colour is decoration on top.

Similarly, the volatility breakdown uses **one hue with monotone lightness**
rather than four categorical colours, because static < slow < dynamic <
realtime is an ordered scale, not unordered identity.

---

## 8. Cost and rate limits

Everything runs at **$0/month**. No credit card at any step.

| Service | Purpose | Free allowance |
|---|---|---|
| Tavily | web search | 1,000 searches/month |
| Google Gemini | router, verifier, summaries, embeddings | ~15 req/min, ~1,500/day |
| Groq | automatic fallback on 429 | 30 req/min, ~14,400/day |
| Upstash Vector | production vector store | 1 GB, 10k ops/day, max 1,536 dims |
| Vercel | hosting | Hobby plan |

### What a query actually costs

| Query class | LLM calls |
|---|---:|
| Gibberish, commands, navigation | 0 |
| Realtime matched by regex | 0 |
| Repeat within the hour | 0 |
| Near-identical cache hit | 0 |
| Polarity or direction conflict | 0 |
| **Cold informational query** | **1** |
| Cold query with borderline candidates | 2 |

**Measured on a realistic 16-query mix: 11 calls, 0.69 per query.**

Be precise about this: **one call is the normal case.** Zero happens only for
regex-matched realtime queries, repeats within the hour, and rejected input.
The design goal was a *ceiling* of two, not a typical of zero. Against Groq's
~14,400/day the ceiling allows 7,000+ queries, so the budget is not the binding
constraint — Gemini's 15 requests/minute is.

Gemini's free tier means **Google may use prompts to improve their models**.
Fine for a portfolio project; worth knowing.

---

## 9. Deployment

Two interchangeable stacks, selected by environment variable:

| | local (default) | production |
|---|---|---|
| Embeddings | all-MiniLM-L6-v2, 384d | `gemini-embedding-001`, 768d |
| Inference | none (heuristics only) | Gemini, Groq fallback |
| Summarising | distilbart | Gemini / Groq |
| Vector store | ChromaDB on disk | Upstash Vector |
| Dependencies | 9 packages, ~1.3 GB | 4 packages, ~25 MB |

### Why production cannot use the local stack

Not a preference — a requirement, for two independent reasons:

1. **Bundle size.** Vercel's Python limit is 500 MB uncompressed. The installed
   local dependencies come to ~1.3 GB; **torch alone is 547 MB.**
2. **No writable filesystem.** `chromadb.PersistentClient(path="./chroma_db")`
   has nowhere to live on serverless.

### Environment variables

Two groups, and the distinction matters. The **five secrets** are identical to
your `.env`:

```
TAVILY_API_KEY  GEMINI_API_KEY  GROQ_API_KEY
UPSTASH_VECTOR_REST_URL  UPSTASH_VECTOR_REST_TOKEN
```

The **four selectors** are *different* from `.env`, which holds local values:

```
LLM_PROVIDER=chain  EMBED_PROVIDER=gemini
VECTOR_STORE=upstash  SUMMARIZER=llm
```

**Do not bulk-copy `.env` into the host.** Three of those four point at the
local stack there, and the resulting failure looks like a missing dependency
rather than a configuration mistake. This actually happened — see §11.

### Vercel specifics

- Flask is detected natively from the top-level `app` in `app.py` at the repo
  root. No handler wrapper, no `api/` directory.
- Hobby's function limit is **300 s** by default with Fluid compute on;
  `vercel.json` sets 120 s. (The commonly-quoted 60 s and 250 MB figures are
  outdated / Node-specific.)
- Streaming is enabled by default for Python, so the SSE endpoint works.
- Env changes require a redeploy.
- The Upstash index dimension is **fixed at creation** and must equal
  `GEMINI_EMBED_DIM` (768), with **COSINE** distance.

`python scripts/preflight.py --prod` checks every backend with a live request,
because a config file cannot show a retired model id, an index created with the
wrong dimension, or a key that was never activated. Two of the three model ids
this project started with had been withdrawn by the time it deployed.

---

## 10. Testing

**386 tests. No network calls.** Verified by blocking every non-loopback socket
and confirming the suite still passes — not by assumption.

| File | Tests | Covers |
|---|---:|---|
| `test_volatility_policy.py` | 109 | Classification, TTL clamping, escalation as a property |
| `test_cache_ttl.py` | 46 | Freshness, retrieval, legacy migration, admin |
| `test_pipeline.py` | 45 | Flow, the realtime guarantee, error paths |
| `test_summarizer.py` | 39 | Cleaning, query-aware selection, chunking |
| `test_llm_gateway.py` | 38 | Provider chain, 429s, JSON repair, degradation |
| `test_mismatch.py` | 34 | Polarity, direction, quantity detection |
| `test_web.py` | 29 | Routes, SSE framing, cache admin |
| `test_quota_guard.py` | 20 | **Exact LLM call counts per query class** |
| `test_upstash_cache.py` | 14 | The hosted backend against a faithful stub |
| `test_production_bundle.py` | 12 | Every deployed module imports without the local stack |

Three of these are load-bearing in a way worth explaining:

**`test_quota_guard.py`** asserts the *exact* number of LLM calls for each
query class. A refactor that quietly moves work onto the model fails here
rather than draining a daily quota in production.

**`test_production_bundle.py`** imports every module the deployed function
loads, in a subprocess with the local-only packages blocked. It exists because
the first deployment returned `FUNCTION_INVOCATION_FAILED` on every request.

**`scripts/run_tests_offline.py`** runs the suite with every non-loopback
socket blocked and fails on any outbound attempt — **even when the test itself
swallowed the error.** It exists because "the tests make no network calls"
silently stopped being true twice.

### CI

Three jobs on every push:

| Job | What | Why |
|---|---|---|
| `test` | suite on Python 3.12 and 3.13 | 3.12 is what Vercel runs |
| `offline` | suite with sockets blocked | the property decayed twice |
| `bundle` | production deps only | the first deploy died on this |

CI installs `requirements-test.txt`, not the local stack: the suite stubs the
embedder and the classifier loads from JSON, so torch and friends add ~1.3 GB
for no extra coverage. 96 packages, under a minute per job.

---

## 11. Bugs found, and what each taught

Worth keeping because the *class* of each mistake is the transferable part.

### In the original code

**The validity gate rejected valid queries.** Gating on the classifier's 0.5
boundary meant `weather in bangalore`, `tesla stock price today` and `live
cricket score` were all answered with *"Invalid query."* — precisely the
queries the caching work targets. *Lesson: a model's default decision boundary
is not automatically the right operating point for your use case.*

**Summary chunks were silently dropped.** Chunking counted **words** (380)
against a **token** limit (1024). 380 words of technical prose tokenises to
1,152, the pipeline raised "index 1026 is out of bounds", and the handler
logged it as a skipped chunk — so content vanished from summaries with only a
warning. *Lesson: measure in the unit the limit is expressed in.*

**The query parameter was accepted and never used.** `summarize_text(text,
query)` ignored `query` entirely, so every summary was query-agnostic.
*Lesson: an unused parameter is a silent behavioural bug, not dead code.*

**Sentences with >40% capitalised words were deleted** as a nav-menu proxy —
removing most prose about people and places. A summary of "who was Alan Turing"
lost its subject. *Lesson: a heuristic's false-positive class matters as much
as its true-positive rate.*

**Hyphens were stripped**, producing "lightdependent" and
"3phosphoglyceric". **Unicode apostrophes** became spaces, so "Earth's" read
"Earth s". *Lesson: normalise typography before filtering characters, not
after.*

**Non-answers were cached for 30 days.** The summariser already reported
"the extracts do not answer this" and that flag was logged and discarded.
*Lesson: a signal you compute and ignore is worse than one you never had — it
looks like the case is handled.*

### Mine, during this work

**The first deployment crashed on every request.** `FUNCTION_INVOCATION_FAILED`
— a module-level `chromadb` import in code the production bundle does not
install. I had "verified" the bundle by importing the leaf modules and never
`app`, the one module Vercel actually loads.

**CI failed on `No module named 'queryagent'`.** `python -m pytest` puts the
working directory on `sys.path`; the `pytest` console script does not. I
verified with the former; CI runs the latter.

**CI failed on `assert False is True`.** A `/healthz` test asserted `ok` was
true, which held only because my `.env` had a key. CI has no secrets, so the
endpoint correctly reported a problem.

**A bulk rename corrupted the API and its test identically.** A regex renaming
module references also matched string literals, rewriting two `/healthz`
response keys. Nothing failed, because the same regex rewrote the key the test
looked for — so the assertion and the bug moved together. Found by reading the
live response, not by any test.

**One lesson, four times over: passing on my machine was never the same claim
as passing.** Every one came from verifying against something *adjacent* to the
real thing. The fixes are structural — `pytest.ini` pins the path so any
invocation behaves the same; `conftest` neutralises the search key so a
populated `.env` cannot diverge from CI; the bundle test walks the real entry
points; the `/healthz` test compares the full key set so it cannot drift in
step with the code.

---

## 12. Known weaknesses

State these plainly if asked. They are not disqualifying, and pretending
otherwise is worse than owning them.

**It is a demo, not a service.** No authentication, no per-user rate limiting,
no metrics, no tracing, no alerting. One shared cache for all visitors.

**The cache's value needs repeat traffic.** With one person asking different
things, hit rates are low. The mechanism solves a problem that appears at
scale.

**The LLM does the language work.** The contribution is the decision layer
around it — when reuse is safe, what expires, what the model should not be
trusted to judge. That is a real contribution, but it is orchestration, not
novel modelling.

**Cold queries take ~14 s.** Most of it is fetching five pages. Three pages
would roughly halve it.

**The classifier does not generalise well** — see §7. It works as a
low-threshold junk filter and nothing more.

**The 768-dim classifier is not trained.** Gemini's embedding quota ran out
partway through. Production runs without the gate; the router covers validity.
`python scripts/train_classifier.py --provider gemini` resumes from a
checkpoint.

**Playwright cannot run on serverless**, so the JS-heavy-page fallback is local
only. Production relies on Tavily's `raw_content` instead.

**ChromaDB's `PersistentClient` is single-process.** Fine at one Flask worker;
multi-worker needs the HTTP client or the hosted store.

---

## 13. Questions this design invites

### "Isn't this just a wrapper around Gemini?"

The LLM does the language work. What is mine is the decision layer: classifying
how fast an answer goes stale, deriving a TTL, deciding which cache candidates
are even eligible, and deciding what the model should *not* be trusted with.

That last part is the most concrete answer: the polarity and direction checks
are deterministic **because the LLM got them wrong**. Asked whether a
benefits-only summary of "is coffee good for you" answers "is coffee bad for
you", it said yes three times running, including when the conflict was named in
the prompt. So that decision was taken away from it. I can show the
measurements.

### "Why not just use a higher similarity threshold?"

Because the bands overlap. Genuine paraphrases score 0.918–0.962 on MiniLM;
false matches score 0.765–0.997. Under Gemini it is worse — paraphrases
0.968–0.990, quantity conflicts 0.962–0.971. Any threshold either serves wrong
answers or never hits. That is the measurement that motivated the whole
verification layer.

### "How would you scale this?"

The cache gets *more* valuable with traffic, since hit rate rises with repeats.
What would need adding: authentication and per-user rate limiting; metrics on
hit rate, verifier accept rate and LLM call volume; a shared memo (currently
in-process, so it is per-container on serverless); and batching embedding calls.
The vector store is already hosted and horizontally fine.

The first thing I would actually measure is verifier precision in production —
I have an 11-case benchmark, which is enough to design against and not enough
to trust at volume.

### "What is the hardest bug you hit?"

The one where a bulk rename corrupted two API response keys *and* the test that
would have caught it, identically, so the suite stayed green while the deployed
endpoint returned the wrong field names. I found it by reading the live
response. The fix was to assert the full key set rather than spot-check names,
so the test cannot drift in step with the code.

The lesson generalised: I had three separate failures from verifying against
something adjacent to the real thing rather than the real thing.

### "Why Flask and server-rendered HTML rather than React?"

The interesting work is the decision layer, and one service is simpler to
deploy on a free tier than a frontend plus an API. All the logic has to be
Python regardless. A React frontend would have added a second deployment, CORS,
and two cold starts for no functional gain.

### "What would you do differently?"

Measure the embedding behaviour *before* choosing thresholds — I set 0.75 and
0.93 from intuition and both were wrong, one of them in a way that defeated the
layer it was protecting. And stop treating "it passes locally" as verification;
three of my failures came from a non-representative local environment.

### "Why is the classifier still there if it does not generalise?"

It is a cheap junk filter, not a decision-maker. It rejects only below p=0.05,
which catches punctuation-only input and device commands without an API call.
Deleting it would push that traffic onto the LLM for no benefit. But I would
not claim it does more than that — the 99.5% figure measures templates, not
generalisation.

---

## 14. Numbers worth remembering

| | |
|---|---|
| Source lines | ~4,900 |
| Test lines | ~2,400 |
| Tests | 386, no network |
| Commits in this work | 22 |
| Production dependencies | 4 packages (~25 MB) |
| Local dependencies | 9 packages (~1.3 GB) |
| Torch alone | 547 MB, vs Vercel's 500 MB limit |
| LLM calls per query | 0.69 measured, 1 typical, 2 ceiling |
| Verification benchmark | 11/11 correct, 4 LLM calls |
| Cold query / cache hit | 14.4 s / 5.6 s |
| Volatility heuristic rules | 13 (6 realtime, 3 dynamic, 2 slow, 2 static) |
| Antonym pairs in the guard | 31 |
| Monthly cost | $0 |

Three figures to be careful with:

- **99.5% classifier accuracy** — on a synthetic templated dataset. Always
  qualify it.
- **"0 LLM calls in most cases"** — false. One is typical. Zero is for
  regex-matched realtime, repeats, and rejected input.
- **Cache saving** — not LLM calls (the router runs either way) but the search,
  five page fetches, and the summarisation: 14.4 s → 5.6 s.

---

## 15. Operating it

```bash
# Setup (Python 3.12)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-local.txt     # local stack
cp .env.example .env                      # add TAVILY_API_KEY at minimum

# Run
python main.py                            # CLI; prefix a query with ! to force refresh
python app.py                             # web UI at http://127.0.0.1:5000
python scripts/preflight.py               # check every backend with a live request
python scripts/preflight.py --prod        # check the hosted stack instead

# Tests
pip install -r requirements-test.txt
pytest                                    # 386 tests
python scripts/run_tests_offline.py       # same, with the network blocked

# Cache admin
python scripts/cache_manager.py stats
python scripts/cache_manager.py view --limit 20
python scripts/cache_manager.py delete --query "photosynthesis"

# Retrain the classifier
python scripts/train_classifier.py                     # local, 384d
python scripts/train_classifier.py --provider gemini   # hosted, 768d, resumable

# Deploy (auto-deploys on push to main)
npx vercel --prod
```

Useful endpoints: `/healthz` reports whether each backend can actually run and
names any problem; `/cache` is the cache explorer; `/cache-stats` and
`/cache-view` return JSON.

On macOS, Control Center's AirPlay Receiver also listens on port 5000. Flask
still binds and wins, but if the page will not load, that is why.
