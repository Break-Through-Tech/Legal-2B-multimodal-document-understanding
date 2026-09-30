"""Apply the document scope policy to the parsed labels.

The policy and the reasoning behind it live in docs/scope_policy.md. Change the
constants here and the doc together.

Usage:
    python data/load_data.py          # writes data/labels.csv first
    python data/scope.py --summary
"""

import argparse

import pandas as pd

try:  # imported as data.scope, e.g. from a notebook
    from .load_data import HEAD_CLASS_MIN_COUNT, LABELS_CSV
except ImportError:  # run as a script
    from load_data import HEAD_CLASS_MIN_COUNT, LABELS_CSV

# The classifier's label space. Core legal types from Challenge-Project-Overview.md.
IN_SCOPE_FORMATS = [
    "COMMERCIAL_LEASE_AGREEMENT",
    "PATENT",
    "PETITION_FORM",
    "PROXY_VOTING",
    "REAL_ESTATE",
]

# Legal-adjacent types we deliberately leave out of the label space for now. They are
# treated as out-of-scope, but tagged so results can be broken out for them and the
# label space can be widened later without re-deriving anything.
LEGAL_ADJACENT_FORMATS = [
    "ACCOUNT_STATEMENT",
    "CREDIT_CARD_STATEMENT",
    "FORM_1040",
]

# Scope values
IN_SCOPE = "in_scope"            # train / val / test for the classifier
OUT_OF_SCOPE = "out_of_scope"    # head-class non-legal types; the detector's negatives
TAIL = "tail"                    # < HEAD_CLASS_MIN_COUNT examples; held-out novelty test only


def apply_scope_policy(labels: pd.DataFrame) -> pd.DataFrame:
    df = labels.copy()
    counts = df["format"].value_counts()
    is_head = df["format"].map(counts) >= HEAD_CLASS_MIN_COUNT

    unknown = set(IN_SCOPE_FORMATS + LEGAL_ADJACENT_FORMATS) - set(df["format"][is_head])
    if unknown:
        raise ValueError(f"Scope policy names formats that are not head classes: {sorted(unknown)}")

    df["scope"] = OUT_OF_SCOPE
    df.loc[~is_head, "scope"] = TAIL
    df.loc[df["format"].isin(IN_SCOPE_FORMATS), "scope"] = IN_SCOPE
    df["is_legal_adjacent"] = df["format"].isin(LEGAL_ADJACENT_FORMATS)
    return df


def print_summary(df: pd.DataFrame) -> None:
    for scope in [IN_SCOPE, OUT_OF_SCOPE, TAIL]:
        part = df[df["scope"] == scope]
        print(f"\n{scope.upper()} — {part['format'].nunique()} formats, {len(part)} documents")
        print(part["document_quality"].value_counts().to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true", help="print counts per scope")
    args = parser.parse_args()

    df = apply_scope_policy(pd.read_csv(LABELS_CSV))
    df.to_csv(LABELS_CSV, index=False)
    print(f"Added scope columns to {LABELS_CSV}")

    if args.summary:
        print_summary(df)


if __name__ == "__main__":
    main()
