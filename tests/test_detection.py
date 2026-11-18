"""Tests for the out-of-scope detectors (analysis/detection.py).

A tiny 2-D example where the right answer is obvious, plus checks on how test documents are
grouped. They read only data/documents.csv: no embeddings or downloads needed.
"""

import numpy as np
import pytest

from analysis.detection import REVIEW, confidence_scores, negative_sets, route
from analysis.legal_classifier import fit_classifier
from data.build_dataset import load_documents

# Two well-separated legal types, and one point halfway between them.
X_TRAIN = np.array([[-3.0, 0], [-3.2, 0.1], [-2.8, -0.1], [3.0, 0], [3.2, 0.1], [2.8, -0.1]])
Y_TRAIN = np.array(["LEASE"] * 3 + ["PATENT"] * 3)
CLEAR_LEASE, CLEAR_PATENT, IN_BETWEEN = [-3.0, 0.05], [3.0, 0.05], [0.0, 0.0]


@pytest.fixture(scope="module")
def model():
    return fit_classifier(X_TRAIN, Y_TRAIN, C=1.0)


def test_confident_documents_keep_their_type(model):
    assert list(route(model, np.array([CLEAR_LEASE, CLEAR_PATENT]), threshold=0.3)) == ["LEASE", "PATENT"]


def test_unsure_document_goes_to_review(model):
    assert route(model, np.array([IN_BETWEEN]), threshold=0.3)[0] == REVIEW


def test_threshold_controls_how_much_goes_to_review(model):
    scores = confidence_scores(model, np.array([CLEAR_LEASE, IN_BETWEEN]))
    assert 0 <= scores[0] < scores[1] <= 0.5  # with 2 types the top probability is at least 0.5
    assert (route(model, np.array([CLEAR_LEASE, IN_BETWEEN]), threshold=1.0) != REVIEW).all()  # never review
    assert (route(model, np.array([CLEAR_LEASE, IN_BETWEEN]), threshold=0.0) == REVIEW).all()  # always review


def test_negative_sets_follow_the_scope_policy():
    docs = load_documents()
    sets = negative_sets(docs)
    in_scope = set(docs.loc[docs["scope"] == "in_scope", "id"])
    assert not in_scope & set(sets["all_oos"])
    assert set(sets["legal_adjacent"]) <= set(sets["head_oos"])
    assert len(sets["tail"]) == 75
    assert (docs.loc[sets["head_oos"], "split"] == "test").all()
    assert len(sets["all_oos"]) == len(set(sets["all_oos"]))


def test_documents_table_is_in_id_order():
    # Embedding rows are matched to documents by position, so this order matters.
    docs = load_documents()
    assert docs["id"].tolist() == list(range(len(docs)))
