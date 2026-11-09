"""Evaluate a first out-of-scope detector on every channel with precision, recall and ROC-AUC.

Detector (deliberately simple, so differences come from the embedding):
    score(doc) = 1 - mean cosine similarity to its 5 nearest in-scope *training* documents
A high score means "unlike every legal document we know", so the document is flagged.
Only in-scope training documents are used to fit it; it never sees an out-of-scope
document during training (pure novelty detection).

Threshold: chosen on the val split as the score that maximizes F1 for flagging, using
in-scope val documents (should pass) and out-of-scope val documents (should be flagged).
Tail documents are never used to choose it.

Reported on the test split, with three negative sets kept apart, as the scope policy asks
(docs/scope_policy.md, "Confound: the tail is all photos"):
- `head_oos`:       held-out out-of-scope head documents (same rendering pipeline)
- `legal_adjacent`: the subset of head_oos that is ACCOUNT_STATEMENT / CREDIT_CARD / FORM_1040
- `tail`:           the 75 novel-type photo documents
Positive class = "out of scope" (should be flagged). Every row also has the in-scope test
documents as the documents that should pass.

Outputs:
    results/out_of_scope_detection.csv
    results/out_of_scope_photo_check.csv   the 4 in-scope PHOTO documents: flagged or not

Usage (from the repository root):
    python -m analysis.out_of_scope
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from analysis.channels import RESULTS_DIR, channel_family, load_channels
from data.build_dataset import load_documents

K_NEIGHBORS = 5


def novelty_scores(X: np.ndarray, reference: np.ndarray) -> np.ndarray:
    similarity = X @ X[reference].T
    # A training document's nearest neighbor is itself; that only matters for train rows,
    # which are never scored, but drop exact self-matches anyway.
    similarity[np.isin(np.arange(len(X)), reference)[:, None] & (similarity > 0.9999)] = -1
    top = np.sort(similarity, axis=1)[:, -K_NEIGHBORS:]
    return 1 - top.mean(axis=1)


def best_threshold(scores: np.ndarray, is_oos: np.ndarray) -> float:
    candidates = np.unique(scores)
    f1s = [f1_score(is_oos, scores >= t, zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(f1s))])


def main() -> None:
    docs = load_documents()
    channels = load_channels()

    split, scope = docs["split"].to_numpy(), docs["scope"].to_numpy()
    in_scope = scope == "in_scope"
    reference = np.flatnonzero(in_scope & (split == "train"))
    val = np.flatnonzero((split == "val") & (scope != "tail"))
    test_pass = np.flatnonzero(in_scope & (split == "test"))
    negative_sets = {
        "head_oos": np.flatnonzero((scope == "out_of_scope") & (split == "test")),
        "legal_adjacent": np.flatnonzero((scope == "out_of_scope") & docs["is_legal_adjacent"].to_numpy() & (split == "test")),
        "tail": np.flatnonzero(scope == "tail"),
    }
    negative_sets["all_oos"] = np.concatenate([negative_sets["head_oos"], negative_sets["tail"]])

    # All 4 in-scope PHOTO documents landed in train. They are scored leave-self-out
    # (novelty_scores drops each document's match with itself), which is enough to ask
    # "would the detector flag a legal document just because it is a photo?"
    photos = docs[in_scope & (docs["document_quality"] == "PHOTO")]

    rows, photo_rows = [], []
    for channel, X in channels.items():
        scores = novelty_scores(X, reference)
        threshold = best_threshold(scores[val], ~in_scope[val])
        flagged = scores >= threshold
        for negatives_name, negatives in negative_sets.items():
            idx = np.concatenate([test_pass, negatives])
            truth = ~in_scope[idx]
            rows.append(
                {
                    "channel": channel,
                    "family": channel_family(channel),
                    "negatives": negatives_name,
                    "n_in_scope": len(test_pass),
                    "n_out_of_scope": len(negatives),
                    "ROC_AUC": roc_auc_score(truth, scores[idx]),
                    "precision": precision_score(truth, flagged[idx], zero_division=0),
                    "recall": recall_score(truth, flagged[idx], zero_division=0),
                    "F1": f1_score(truth, flagged[idx], zero_division=0),
                    "in_scope_wrongly_flagged": int(flagged[test_pass].sum()),
                    "threshold": threshold,
                }
            )
        for doc_id, row in photos.iterrows():
            photo_rows.append(
                {"channel": channel, "id": doc_id, "format": row["format"], "split": row["split"],
                 "flagged": bool(flagged[doc_id]), "score": float(scores[doc_id]), "threshold": threshold}
            )

    RESULTS_DIR.mkdir(exist_ok=True)
    results = pd.DataFrame(rows)
    results.round(4).to_csv(RESULTS_DIR / "out_of_scope_detection.csv", index=False)
    pd.DataFrame(photo_rows).round(4).to_csv(RESULTS_DIR / "out_of_scope_photo_check.csv", index=False)

    for metric in ["ROC_AUC", "precision", "recall"]:
        print(f"\n{metric} on test (positive = out of scope)")
        print(results.pivot(index="channel", columns="negatives", values=metric).round(3).to_string())
    if len(photos):
        print(f"\nIn-scope PHOTO documents (scored leave-self-out), flagged? ids {photos.index.tolist()}")
        print(pd.DataFrame(photo_rows).pivot(index="channel", columns="id", values="flagged").to_string())


if __name__ == "__main__":
    main()
