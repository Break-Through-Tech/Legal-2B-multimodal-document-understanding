"""Checks on the committed split manifest and processed document table.

These read data/splits.csv and data/documents.csv only, so they run in seconds and need
no dataset download.
"""

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPLITS = pd.read_csv(ROOT / "data" / "splits.csv")
DOCS = pd.read_csv(ROOT / "data" / "documents.csv")


def test_every_document_has_exactly_one_split():
    assert SPLITS["id"].is_unique
    assert sorted(SPLITS["id"]) == list(range(1000))
    assert set(SPLITS["split"]) == {"train", "val", "test"}


def test_documents_table_matches_manifest():
    merged = DOCS[["id", "split"]].merge(SPLITS, on="id", suffixes=("_docs", "_manifest"))
    assert len(merged) == 1000
    assert (merged["split_docs"] == merged["split_manifest"]).all()


def test_tail_is_test_only():
    assert (DOCS.loc[DOCS["scope"] == "tail", "split"] == "test").all()
    assert (DOCS["scope"] == "tail").sum() == 75


def test_scope_counts_match_policy():
    assert DOCS["scope"].value_counts().to_dict() == {"out_of_scope": 661, "in_scope": 264, "tail": 75}


@pytest.mark.parametrize("split, low, high", [("train", 0.65, 0.75), ("val", 0.12, 0.18), ("test", 0.12, 0.18)])
def test_head_classes_are_stratified(split, low, high):
    head = DOCS[DOCS["scope"] != "tail"]
    share = pd.crosstab(head["format"], head["split"], normalize="index")[split]
    assert share.between(low, high).all(), share.round(2).to_dict()


def test_splits_are_reproducible():
    """Rebuilding from the committed labels gives the identical manifest."""
    from data.splits import build_splits

    labels = DOCS[["id", "format", "document_quality", "is_head_class", "scope", "is_legal_adjacent"]]
    rebuilt = build_splits(labels).set_index("id")["split"].sort_index()
    committed = SPLITS.set_index("id")["split"].sort_index()
    assert rebuilt.astype(str).equals(committed.astype(str))
