"""Train the query-validity classifier.

Fits a logistic regression over sentence embeddings of
augmented_query_dataset.csv and writes three artifacts:

  classifier.json  -- plain coefficients. Small, committed to the repo, and
                      loadable with no sklearn and no pickle. This is what
                      agent.py prefers at runtime.
  classifier.pkl   -- the sklearn estimator, for anyone who wants it.
  embedding_model/ -- a local copy of the sentence transformer.

Only classifier.json is required to run the app; the other two are
conveniences. Reports cross-validated accuracy so the number quoted for this
model is measured rather than assumed.

The artifact is named for the embedding model's dimensionality, because a
classifier trained on 384-dim MiniLM vectors is meaningless applied to 768-dim
Gemini ones. agent.py selects by dimension at load time, so both can coexist
and the gate keeps working whichever backend is active.

    python train_classifier.py                # local MiniLM, 384 dims
    python train_classifier.py --provider gemini   # hosted, 768 dims
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

import config

DATASET = "augmented_query_dataset.csv"


CACHE_PATH = ".embed_cache.json"


def _embed_with_resume(queries, embeddings, batch=100, pause=6.0):
    """Embed via the hosted API, checkpointing as it goes.

    The free tier will refuse partway through a 2000-row dataset, and losing
    twenty successful calls to the twenty-first failing is a poor trade. Each
    batch is written to disk as it lands, so a rate-limited run can simply be
    repeated once the quota resets and it resumes where it stopped.
    """
    import json as _json
    import os as _os
    import time as _time

    done = {}
    if _os.path.isfile(CACHE_PATH):
        try:
            with open(CACHE_PATH) as fh:
                done = _json.load(fh)
            print(f"resuming: {len(done)} of {len(queries)} already embedded")
        except (OSError, ValueError):
            done = {}

    todo = [q for q in queries if q not in done]
    if todo:
        print(f"embedding {len(todo)} queries via {config.GEMINI_EMBED_MODEL} "
              f"at {config.GEMINI_EMBED_DIM} dims "
              f"({-(-len(todo) // batch)} requests, ~{pause:.0f}s apart)...")
    for start in range(0, len(todo), batch):
        chunk = todo[start : start + batch]
        try:
            vectors = embeddings.encode_many(chunk)
        except Exception as exc:
            with open(CACHE_PATH, "w") as fh:
                _json.dump(done, fh)
            raise SystemExit(
                f"\nembedding stopped after {len(done)}/{len(queries)}: {exc}\n"
                f"Progress is saved in {CACHE_PATH}. Re-run this command once the "
                f"quota resets and it will continue from here."
            ) from exc
        done.update(dict(zip(chunk, vectors)))
        with open(CACHE_PATH, "w") as fh:
            _json.dump(done, fh)
        print(f"  {len(done)}/{len(queries)}")
        if start + batch < len(todo):
            _time.sleep(pause)  # stay inside the per-minute request limit

    return [done[q] for q in queries]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["local", "gemini"], default="local")
    args = parser.parse_args()

    df = pd.read_csv(DATASET)
    queries = df["query"].astype(str).tolist()
    labels = df["label"].map({"valid": 1, "invalid": 0}).values

    if set(df["label"].unique()) - {"valid", "invalid"}:
        raise SystemExit(f"unexpected labels in {DATASET}: {df['label'].unique()}")

    import os
    os.environ["EMBED_PROVIDER"] = args.provider
    import importlib
    importlib.reload(config)
    import embeddings
    importlib.reload(embeddings)

    model = None
    if args.provider == "local":
        from sentence_transformers import SentenceTransformer
        print(f"loading embedding model: {config.LOCAL_EMBED_MODEL}")
        model = SentenceTransformer(config.LOCAL_EMBED_MODEL)
        print(f"embedding {len(queries)} queries...")
        X = model.encode(queries, show_progress_bar=True)
    else:
        X = _embed_with_resume(queries, embeddings)

    # Measure before fitting on everything, so the reported number means
    # something. Fitting on the full set afterwards is deliberate: this is a
    # cheap gate and we want every labelled row in it.
    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, X, labels, cv=5, scoring="accuracy")
    cv_accuracy = float(scores.mean())
    print(f"5-fold CV accuracy: {cv_accuracy:.4f} (+/- {scores.std():.4f})")

    clf = LogisticRegression(max_iter=1000).fit(X, labels)

    coef = clf.coef_[0].tolist()
    embed_name = (config.LOCAL_EMBED_MODEL if args.provider == "local"
                  else f"{config.GEMINI_EMBED_MODEL}@{config.GEMINI_EMBED_DIM}")
    artifact = {
        "schema": 1,
        "model": "logistic_regression",
        "embedding_model": embed_name,
        "dim": len(coef),
        "coef": coef,
        "intercept": float(clf.intercept_[0]),
        "classes": ["invalid", "valid"],
        "n_train": int(len(queries)),
        "cv_accuracy": round(cv_accuracy, 4),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = f"classifier-{len(coef)}.json"
    with open(path, "w") as fh:
        json.dump(artifact, fh, indent=1)
    print(f"wrote {path} ({len(coef)} coefficients)")

    if args.provider == "local":
        with open("classifier.pkl", "wb") as fh:
            pickle.dump(clf, fh)
        print("wrote classifier.pkl")
        model.save("embedding_model")
        print("wrote embedding_model/")


if __name__ == "__main__":
    main()
