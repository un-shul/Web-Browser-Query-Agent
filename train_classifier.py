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

Usage:  python train_classifier.py
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

import config

DATASET = "augmented_query_dataset.csv"


def main() -> None:
    df = pd.read_csv(DATASET)
    queries = df["query"].astype(str).tolist()
    labels = df["label"].map({"valid": 1, "invalid": 0}).values

    if set(df["label"].unique()) - {"valid", "invalid"}:
        raise SystemExit(f"unexpected labels in {DATASET}: {df['label'].unique()}")

    print(f"loading embedding model: {config.LOCAL_EMBED_MODEL}")
    model = SentenceTransformer(config.LOCAL_EMBED_MODEL)

    print(f"embedding {len(queries)} queries...")
    X = model.encode(queries, show_progress_bar=True)

    # Measure before fitting on everything, so the reported number means
    # something. Fitting on the full set afterwards is deliberate: this is a
    # cheap gate and we want every labelled row in it.
    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, X, labels, cv=5, scoring="accuracy")
    cv_accuracy = float(scores.mean())
    print(f"5-fold CV accuracy: {cv_accuracy:.4f} (+/- {scores.std():.4f})")

    clf = LogisticRegression(max_iter=1000).fit(X, labels)

    coef = clf.coef_[0].tolist()
    artifact = {
        "schema": 1,
        "model": "logistic_regression",
        "embedding_model": config.LOCAL_EMBED_MODEL,
        "dim": len(coef),
        "coef": coef,
        "intercept": float(clf.intercept_[0]),
        "classes": ["invalid", "valid"],
        "n_train": int(len(queries)),
        "cv_accuracy": round(cv_accuracy, 4),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open("classifier.json", "w") as fh:
        json.dump(artifact, fh, indent=1)
    print(f"wrote classifier.json ({len(coef)} coefficients)")

    with open("classifier.pkl", "wb") as fh:
        pickle.dump(clf, fh)
    print("wrote classifier.pkl")

    model.save("embedding_model")
    print("wrote embedding_model/")


if __name__ == "__main__":
    main()
