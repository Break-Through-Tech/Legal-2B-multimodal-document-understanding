"""Build the team's train / val / test split manifest (task #2d).

Writes data/splits.csv with two columns, `id` and `split`. The file is committed to git,
so every teammate and every experiment uses exactly the same split.

How documents are split (see docs/splits.md for the reasoning):
- in_scope and out_of_scope (the 18 head classes): 70 / 15 / 15, stratified by `format`,
  so every class keeps the same proportions in every split.
- tail (the 75 rare-class documents): test only. They are the detector's novelty test and
  are never trained on or used to tune thresholds.

Usage:
    python data/load_data.py      # writes data/labels.csv
    python data/splits.py         # writes data/splits.csv and prints the counts
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

try:  # imported as data.splits, e.g. from a notebook
    from .load_data import LABELS_CSV, build_labels
    from .scope import IN_SCOPE, OUT_OF_SCOPE, TAIL, apply_scope_policy
except ImportError:  # run as a script
    from load_data import LABELS_CSV, build_labels
    from scope import IN_SCOPE, OUT_OF_SCOPE, TAIL, apply_scope_policy

SPLITS_CSV = Path(__file__).parent / "splits.csv"
SEED = 42
VAL_FRAC = 0.15
TEST_FRAC = 0.15
SPLITS = ["train", "val", "test"]


def load_scoped_labels() -> pd.DataFrame:
    """Labels with scope columns, rebuilt from the dataset if labels.csv is missing."""
    labels = pd.read_csv(LABELS_CSV) if LABELS_CSV.exists() else build_labels()
    if "scope" not in labels.columns:
        labels = apply_scope_policy(labels)
    return labels.sort_values("id").reset_index(drop=True)


def build_splits(labels: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    head = labels[labels["scope"].isin([IN_SCOPE, OUT_OF_SCOPE])]
    train, rest = train_test_split(
        head, test_size=VAL_FRAC + TEST_FRAC, stratify=head["format"], random_state=seed
    )
    val, test = train_test_split(
        rest, test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC), stratify=rest["format"], random_state=seed
    )

    split = pd.Series(index=labels["id"], dtype=object)
    split[train["id"]] = "train"
    split[val["id"]] = "val"
    split[test["id"]] = "test"
    split[labels.loc[labels["scope"] == TAIL, "id"]] = "test"

    if split.isna().any():
        raise RuntimeError(f"{split.isna().sum()} documents got no split")
    return split.rename("split").reset_index()


def check_splits(labels: pd.DataFrame, splits: pd.DataFrame) -> None:
    """Fail loudly if the manifest breaks any rule in docs/splits.md."""
    df = labels.merge(splits, on="id", validate="one_to_one")
    assert len(df) == len(labels) == splits["id"].nunique(), "every document needs exactly one split"
    assert set(df["split"]) == set(SPLITS)
    assert (df.loc[df["scope"] == TAIL, "split"] == "test").all(), "tail documents are test-only"

    head = df[df["scope"] != TAIL]
    counts = pd.crosstab(head["format"], head["split"])
    assert (counts[SPLITS] > 0).all().all(), "every head class must appear in every split"
    share = counts["train"] / counts.sum(axis=1)
    assert share.between(0.65, 0.75).all(), f"train share drifted from 70%: {share.round(2).to_dict()}"


def print_summary(labels: pd.DataFrame, splits: pd.DataFrame) -> None:
    df = labels.merge(splits, on="id")
    print("\nDocuments per scope and split")
    print(pd.crosstab(df["scope"], df["split"], margins=True)[SPLITS + ["All"]].to_string())
    print("\nIn-scope documents per format and split")
    in_scope = df[df["scope"] == IN_SCOPE]
    print(pd.crosstab(in_scope["format"], in_scope["split"], margins=True)[SPLITS + ["All"]].to_string())


def main() -> None:
    labels = load_scoped_labels()
    splits = build_splits(labels)
    check_splits(labels, splits)
    splits.to_csv(SPLITS_CSV, index=False)
    print(f"Wrote {SPLITS_CSV} ({len(splits)} rows, seed {SEED})")
    print_summary(labels, splits)


if __name__ == "__main__":
    main()
