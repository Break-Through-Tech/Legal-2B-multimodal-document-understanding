"""Pipeline step 1: give each document a legal type, or send it to review (issue #10).

Uses the team's choices:
- Classifier: logistic regression on the ColQwen2 embeddings (Grace, Nov 10 lab).
- Out-of-scope check: the `either` rule on the same embeddings. A document goes to REVIEW
  if the confidence detector or the distance detector flags it.

The models and thresholds come from analysis/detection.py (`fit_detectors`), so these routes
match results/detection_scores.csv. On test, all 39 legal documents get their true type and
2 of the 175 out-of-scope documents get through (ids 36 and 991). The `either` rule was
picked after seeing these test numbers, so they are a little optimistic (see
docs/out_of_scope_detection.md).

Output:
    results/extraction/routing.csv   one row per val and test document:
        id, split, format (true type), scope, predicted_type,
        confidence_score, distance_score, route (the predicted type, or REVIEW)

Usage (from the repository root, needs cache/embeddings/vision_colqwen2.npy):
    python -m extraction.route
"""

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.channels import RESULTS_DIR, load_channels
from analysis.detection import REVIEW, fit_detectors
from analysis.legal_classifier import CHOSEN_CHANNEL
from data.build_dataset import load_documents

DETECTOR = "either"
OUT_DIR = RESULTS_DIR / "extraction"
ROUTING_CSV = OUT_DIR / "routing.csv"


def route_documents(docs: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:
    """The route for every val and test document."""
    model, _, scores, _, flagged = fit_detectors(docs, X)
    predicted = model.predict(X)
    rows = np.flatnonzero(docs["split"].isin(["val", "test"]).to_numpy())
    return pd.DataFrame({
        "id": docs["id"].to_numpy()[rows],
        "split": docs["split"].to_numpy()[rows],
        "format": docs["format"].to_numpy()[rows],
        "scope": docs["scope"].to_numpy()[rows],
        "predicted_type": predicted[rows],
        "confidence_score": scores["confidence"][rows],
        "distance_score": scores["distance"][rows],
        "route": np.where(flagged[DETECTOR][rows], REVIEW, predicted[rows]),
    })


def load_routes(path: Path = ROUTING_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def main() -> None:
    docs = load_documents()
    channels = load_channels([CHOSEN_CHANNEL])
    if CHOSEN_CHANNEL not in channels:
        raise SystemExit(f"Needs cache/embeddings/{CHOSEN_CHANNEL}.npy (see features/vision_embeddings.py).")
    routes = route_documents(docs, channels[CHOSEN_CHANNEL])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    routes.round(4).to_csv(ROUTING_CSV, index=False)

    for split in ["val", "test"]:
        part = routes[routes["split"] == split]
        legal = part[part["scope"] == "in_scope"]
        other = part[part["scope"] != "in_scope"]
        leaked = other[other["route"] != REVIEW]
        print(f"{split}: {(legal['route'] == legal['format']).sum()} of {len(legal)} legal documents get their true type, "
              f"{(legal['route'] == REVIEW).sum()} go to review. "
              f"{len(leaked)} of {len(other)} out-of-scope documents get through: {leaked['id'].tolist()}")


if __name__ == "__main__":
    main()
