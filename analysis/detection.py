"""Out-of-scope detection: classifier confidence vs. distance to legal documents (October tasks 3b to 3f).

Two detectors, each giving every document a score where higher means "more likely out of scope":

- `confidence` (3b): 1 minus the legal classifier's top probability (analysis/legal_classifier.py).
  The idea: if the classifier can't decide which legal type a document is, it may not be legal.
- `distance` (3c): 1 minus the mean cosine similarity to the 5 nearest in-scope training
  documents. This is the k-NN detector from analysis/out_of_scope.py, reused as is.
- `either`: flags a document if either detector flags it. Only to see whether the two
  catch different documents; it has no single score, so no ROC-AUC.

Threshold (3d): for each detector and channel, the score that maximizes F1 for flagging on
the val split (in-scope val documents should pass, out-of-scope val documents should be
flagged). Same rule as analysis/out_of_scope.py, so the numbers are comparable. The 75 tail
documents are never used to choose it, as the scope policy requires.

Evaluation (3e): on test, scored separately for each out-of-scope group (common types,
legal-adjacent, tail) and for all of them together. Positive class = out of scope. The
confusion matrix counts and the C the detector picked are in the comparison table.

Error analysis (3f): every test document a detector gets wrong is written out with its type,
quality and score. The 4 in-scope PHOTO documents are all in train, so they are scored
leave-one-out: the classifier is refit without the document, and the distance detector
skips the document itself.

Outputs:
    results/detection_comparison.csv    metrics and confusion counts per channel, detector, negative set
    results/detection_scores.csv        score, threshold, flag and predicted type, every val/test document
    results/detection_failures.csv      test documents each detector got wrong
    results/detection_photo_check.csv   the 4 in-scope photos: flagged or not
    results/figures/detection_confusion.png

Usage (from the repository root):
    python -m analysis.detection
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from analysis.channels import FIGURES_DIR, RESULTS_DIR, load_channels
from analysis.legal_classifier import fit_classifier, in_scope_indices, train_on_channel
from analysis.out_of_scope import best_threshold, novelty_scores
from data.build_dataset import load_documents

REVIEW = "REVIEW"
DETECTORS = ["confidence", "distance", "either"]


# --- Task 3b: the confidence detector as a reusable function ---------------------------------


def confidence_scores(model, X: np.ndarray) -> np.ndarray:
    """1 minus the top class probability. 0 = completely sure of one legal type."""
    return 1 - model.predict_proba(X).max(axis=1)


def route(model, X: np.ndarray, threshold: float) -> np.ndarray:
    """The predicted legal type for each document, or REVIEW if the model is too unsure.

    `threshold` is on the confidence score: a document goes to REVIEW when
    1 - top probability >= threshold. Lower thresholds send more documents to a person.
    """
    predicted = model.predict(X)
    return np.where(confidence_scores(model, X) >= threshold, REVIEW, predicted)


# --- Evaluation ----------------------------------------------------------------------------


def negative_sets(docs: pd.DataFrame) -> dict[str, np.ndarray]:
    """Test documents that should be flagged, grouped the way the scope policy asks."""
    scope, split = docs["scope"].to_numpy(), docs["split"].to_numpy()
    head_oos = (scope == "out_of_scope") & (split == "test")
    sets = {
        "head_oos": np.flatnonzero(head_oos),
        "legal_adjacent": np.flatnonzero(head_oos & docs["is_legal_adjacent"].to_numpy()),
        "tail": np.flatnonzero(scope == "tail"),
    }
    sets["all_oos"] = np.concatenate([sets["head_oos"], sets["tail"]])
    return sets


def metrics(truth: np.ndarray, flagged: np.ndarray, scores: np.ndarray | None) -> dict:
    tn, fp, fn, tp = confusion_matrix(truth, flagged, labels=[False, True]).ravel()
    return {
        "ROC_AUC": roc_auc_score(truth, scores) if scores is not None else np.nan,
        "precision": precision_score(truth, flagged, zero_division=0),
        "recall": recall_score(truth, flagged, zero_division=0),
        "F1": f1_score(truth, flagged, zero_division=0),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


def photo_check(docs: pd.DataFrame, X: np.ndarray, C: float, thresholds: dict, reference: np.ndarray) -> list[dict]:
    """Would a legal photo be flagged just for being a photo? Scored leave-one-out."""
    y = docs["format"].to_numpy()
    train = in_scope_indices(docs, "train")
    distance = novelty_scores(X, reference)  # already skips each document's match with itself
    rows = []
    for i in np.flatnonzero((docs["scope"] == "in_scope").to_numpy() & (docs["document_quality"] == "PHOTO").to_numpy()):
        others = train[train != i]
        conf = confidence_scores(fit_classifier(X[others], y[others], C), X[[i]])[0]
        flags = {"confidence": conf >= thresholds["confidence"], "distance": distance[i] >= thresholds["distance"]}
        flags["either"] = flags["confidence"] or flags["distance"]
        rows.append({"id": i, "format": y[i], "split": docs["split"][i],
                     "confidence_score": conf, "distance_score": distance[i],
                     **{f"flagged_{d}": bool(flags[d]) for d in DETECTORS}})
    return rows


def evaluate_channel(docs: pd.DataFrame, X: np.ndarray) -> tuple[list, list, list, dict]:
    in_scope = (docs["scope"] == "in_scope").to_numpy()
    split = docs["split"].to_numpy()
    val = np.flatnonzero((split == "val") & (docs["scope"] != "tail").to_numpy())
    test_pass = in_scope_indices(docs, "test")
    reference = in_scope_indices(docs, "train")
    groups = negative_sets(docs)

    # Among equally accurate classifiers, keep the one whose confidence best separates legal
    # from out-of-scope val documents. ROC-AUC needs no threshold; test is never looked at.
    def separates_val(model) -> float:
        return roc_auc_score(~in_scope[val], confidence_scores(model, X[val]))

    model, C, _ = train_on_channel(docs, X, tie_break=separates_val)
    scores = {"confidence": confidence_scores(model, X), "distance": novelty_scores(X, reference)}
    thresholds = {d: best_threshold(scores[d][val], ~in_scope[val]) for d in scores}
    flagged = {d: scores[d] >= thresholds[d] for d in scores}
    flagged["either"] = flagged["confidence"] | flagged["distance"]

    predicted_type = model.predict(X)
    top_probability = 1 - scores["confidence"]

    comparison = []
    for name, negatives in groups.items():
        idx = np.concatenate([test_pass, negatives])
        truth = ~in_scope[idx]
        for d in DETECTORS:
            comparison.append({
                "detector": d, "negatives": name, "n_in_scope": len(test_pass), "n_out_of_scope": len(negatives),
                **metrics(truth, flagged[d][idx], scores[d][idx] if d in scores else None),
                "threshold": thresholds.get(d, np.nan),
                "C": np.nan if d == "distance" else C,
            })

    scored = np.flatnonzero(np.isin(split, ["val", "test"]))
    per_document = []
    for d in DETECTORS:
        per_document.append(pd.DataFrame({
            "id": docs["id"].to_numpy()[scored], "detector": d, "split": split[scored],
            "format": docs["format"].to_numpy()[scored], "scope": docs["scope"].to_numpy()[scored],
            "document_quality": docs["document_quality"].to_numpy()[scored],
            "score": scores[d][scored] if d in scores else np.nan,
            "threshold": thresholds.get(d, np.nan),
            "flagged": flagged[d][scored],
            "predicted_type": predicted_type[scored], "top_probability": top_probability[scored],
        }))

    photos = photo_check(docs, X, C, thresholds, reference)
    all_test = np.concatenate([test_pass, groups["all_oos"]])
    confusion = {d: confusion_matrix(~in_scope[all_test], flagged[d][all_test], labels=[False, True])
                 for d in DETECTORS}
    return comparison, per_document, photos, confusion


def plot_confusions(confusions: dict) -> None:
    """One 2x2 matrix per channel and detector, on all test documents."""
    channels = list(confusions)
    fig, axes = plt.subplots(len(channels), len(DETECTORS), figsize=(4 * len(DETECTORS), 3 * len(channels)),
                             squeeze=False)
    for r, channel in enumerate(channels):
        for c, d in enumerate(DETECTORS):
            ConfusionMatrixDisplay(confusions[channel][d], display_labels=["legal", "out of scope"]).plot(
                ax=axes[r][c], colorbar=False)
            axes[r][c].set_title(f"{channel}\n{d}", fontsize=9)
            axes[r][c].set_xlabel("flagged as" if r == len(channels) - 1 else "")
            axes[r][c].set_ylabel("true" if c == 0 else "")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "detection_confusion.png", dpi=130)


def main() -> None:
    docs = load_documents()
    channels = load_channels()

    comparison, per_document, photos, confusions = [], [], [], {}
    for channel, X in channels.items():
        rows, frames, photo_rows, confusion = evaluate_channel(docs, X)
        comparison += [{"channel": channel, **row} for row in rows]
        per_document += [frame.assign(channel=channel) for frame in frames]
        photos += [{"channel": channel, **row} for row in photo_rows]
        confusions[channel] = confusion

    comparison = pd.DataFrame(comparison)
    per_document = pd.concat(per_document, ignore_index=True)
    test_rows = per_document[per_document["split"] == "test"]
    failures = test_rows[test_rows["flagged"] != (test_rows["scope"] != "in_scope")].assign(
        error=lambda df: np.where(df["scope"] == "in_scope", "legal flagged", "out of scope accepted"))

    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    comparison.round(4).to_csv(RESULTS_DIR / "detection_comparison.csv", index=False)
    per_document.round(4).to_csv(RESULTS_DIR / "detection_scores.csv", index=False)
    failures.round(4).to_csv(RESULTS_DIR / "detection_failures.csv", index=False)
    pd.DataFrame(photos).round(4).to_csv(RESULTS_DIR / "detection_photo_check.csv", index=False)
    plot_confusions(confusions)

    view = comparison[comparison["negatives"] == "all_oos"]
    print(view[["channel", "detector", "ROC_AUC", "precision", "recall", "F1", "TP", "FP", "FN", "TN", "threshold", "C"]]
          .round(3).to_string(index=False))
    print("\nTest errors per channel and detector:")
    print(failures.groupby(["channel", "detector", "error"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
