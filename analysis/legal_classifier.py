"""Classify in-scope documents into the 5 legal types, on each embedding channel (issue #9).

This is the classifier the confidence-based out-of-scope detector (analysis/detection.py)
is built on. It follows the October plan: a majority-class baseline, then logistic
regression on frozen embeddings with the regularization strength picked on val, then one
run on test.

- Trained only on in-scope `train` documents from data/splits.csv.
- C is chosen by macro F1 on in-scope `val` documents. Test is used once, at the end.
- Same model as Chinmaya's embedding notebook on the data/preprocessing branch
  (StandardScaler + LogisticRegression), now on the shared split and with a validation step.

Outputs:
    results/classifier_comparison.csv   macro F1 and accuracy per channel, baseline included
    results/classifier_per_class.csv    precision / recall / F1 / support per type, test split
    results/figures/classifier_confusion_vision_colqwen2.png   for ColQwen2, the team's pick

Usage (from the repository root):
    python -m analysis.legal_classifier
"""

import warnings
from collections.abc import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, log_loss, precision_recall_fscore_support
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis.channels import FIGURES_DIR, RESULTS_DIR, load_channels
from data.build_dataset import load_documents

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
CHOSEN_CHANNEL = "vision_colqwen2"  # the team's pick for the classifier (Nov 10 lab)


def fit_classifier(X: np.ndarray, y: np.ndarray, C: float) -> Pipeline:
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return model.fit(X, y)


def in_scope_indices(docs: pd.DataFrame, split: str) -> np.ndarray:
    return np.flatnonzero((docs["scope"] == "in_scope").to_numpy() & (docs["split"] == split).to_numpy())


def tune_and_fit(X: np.ndarray, y: np.ndarray, train: np.ndarray, val: np.ndarray,
                 tie_break: Callable[[Pipeline], float] | None = None) -> tuple[Pipeline, float, float]:
    """Pick C by val macro F1, then return (model fit on train, C, val macro F1).

    Ties are common here: most channels reach F1 1.00 at every C. `tie_break(model)`
    decides between tied models (higher is better). By default it picks the model with
    the lowest val log loss. The out-of-scope detector passes its own rule, because C
    changes how spread out the probabilities are, and that matters more to the detector
    than to the classifier (see docs/out_of_scope_detection.md).
    """
    if tie_break is None:
        def tie_break(model):
            return -log_loss(y[val], model.predict_proba(X[val]), labels=model.classes_)

    best, best_key = None, None
    for C in C_GRID:
        model = fit_classifier(X[train], y[train], C)
        macro_f1 = f1_score(y[val], model.predict(X[val]), average="macro")
        key = (round(macro_f1, 9), tie_break(model))
        if best_key is None or key > best_key:
            best, best_key = (model, C, macro_f1), key
    return best


def train_on_channel(docs: pd.DataFrame, X: np.ndarray,
                     tie_break: Callable[[Pipeline], float] | None = None) -> tuple[Pipeline, float, float]:
    """The tuned classifier for one channel: (model, chosen C, val macro F1)."""
    y = docs["format"].to_numpy()
    return tune_and_fit(X, y, in_scope_indices(docs, "train"), in_scope_indices(docs, "val"), tie_break)


def main() -> None:
    docs = load_documents()
    channels = load_channels()
    y = docs["format"].to_numpy()
    train, val, test = (in_scope_indices(docs, s) for s in ["train", "val", "test"])

    # Baseline: always predict the most common legal type in train. It needs no features.
    baseline = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(train), 1)), y[train])
    baseline_pred = baseline.predict(np.zeros((len(test), 1)))
    rows = [{
        "channel": "majority_baseline", "C": None, "val_macro_f1": None,
        "test_macro_f1": f1_score(y[test], baseline_pred, average="macro", zero_division=0),
        "test_accuracy": accuracy_score(y[test], baseline_pred),
    }]

    per_class, test_predictions = [], {}
    for channel, X in channels.items():
        model, C, val_f1 = tune_and_fit(X, y, train, val)
        pred = model.predict(X[test])
        test_predictions[channel] = pred
        rows.append({
            "channel": channel, "C": C, "val_macro_f1": val_f1,
            "test_macro_f1": f1_score(y[test], pred, average="macro"),
            "test_accuracy": accuracy_score(y[test], pred),
        })
        labels = sorted(set(y[train]))
        p, r, f, s = precision_recall_fscore_support(y[test], pred, labels=labels, zero_division=0)
        per_class += [
            {"channel": channel, "format": t, "precision": p[i], "recall": r[i], "f1": f[i], "support": s[i]}
            for i, t in enumerate(labels)
        ]

    comparison = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    comparison.round(4).to_csv(RESULTS_DIR / "classifier_comparison.csv", index=False)
    pd.DataFrame(per_class).round(4).to_csv(RESULTS_DIR / "classifier_per_class.csv", index=False)
    print(comparison.round(3).to_string(index=False))

    # Confusion matrix for the channel the team chose for the classifier (ColQwen2, Nov 10 lab).
    # Almost every channel ties at val macro F1 1.00, so the scores alone can't pick one.
    if CHOSEN_CHANNEL not in test_predictions:
        print(f"\n{CHOSEN_CHANNEL} is not cached, so no confusion matrix.")
        return
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y[test], test_predictions[CHOSEN_CHANNEL], ax=ax,
                                            xticks_rotation=45, colorbar=False)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")  # keep long type names apart
    ax.set_title(f"Legal-type classifier on test, {CHOSEN_CHANNEL}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"classifier_confusion_{CHOSEN_CHANNEL}.png", dpi=150)
    print(f"\nConfusion matrix saved for {CHOSEN_CHANNEL}.")


if __name__ == "__main__":
    main()
