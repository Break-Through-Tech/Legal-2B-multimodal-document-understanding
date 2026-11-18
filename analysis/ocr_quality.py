"""Score the cached Tesseract text against the ground-truth transcription.

Word recall    = share of ground-truth words (as a multiset, lowercased, alphanumeric)
                 that also appear in the OCR text.
Word precision = share of OCR words that also appear in the ground truth.
Order is ignored, which matches how bag-of-words channels (TF-IDF) see the text.

Outputs:
    results/ocr_quality_by_document.csv
    results/ocr_quality_summary.csv     means by document_quality, scope, in-scope format, rotation

Usage (from the repository root, after `python -m features.ocr`):
    python -m analysis.ocr_quality
    python -m analysis.ocr_quality --ocr-file cache/ocr/other_run.jsonl   # compare a variant
"""

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from analysis.channels import RESULTS_DIR
from data.build_dataset import decode_markdown, load_documents, load_raw_dataset
from features.ocr import OCR_JSONL, load_ocr_text


def words(text: str) -> Counter:
    return Counter(re.findall(r"[a-z0-9]+", text.lower()))


def score(ocr: dict[int, str]) -> pd.DataFrame:
    docs = load_documents()
    truth = [decode_markdown(raw) for raw in load_raw_dataset()["true_markdown_output"]]
    rows = []
    for i in docs["id"]:
        g, o = words(truth[i]), words(ocr.get(i, ""))
        hit = sum((g & o).values())
        rows.append(
            {
                "id": i,
                "gt_words": sum(g.values()),
                "ocr_words": sum(o.values()),
                "word_recall": hit / max(1, sum(g.values())),
                "word_precision": hit / max(1, sum(o.values())),
            }
        )
    return docs[["id", "format", "scope", "document_quality", "rotation"]].merge(pd.DataFrame(rows), on="id")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["word_recall", "word_precision"]
    parts = [
        df[metrics].mean().to_frame().T.assign(group="all", value="all"),
        df.groupby("document_quality")[metrics].mean().reset_index(names="value").assign(group="document_quality"),
        df.groupby("scope")[metrics].mean().reset_index(names="value").assign(group="scope"),
        df[df["scope"] == "in_scope"].groupby("format")[metrics].mean().reset_index(names="value").assign(group="in_scope_format"),
        df.assign(value=df["rotation"].ne(0).map({True: "rotated", False: "upright"}))
        .groupby("value")[metrics].mean().reset_index().assign(group="rotation"),
    ]
    counts = pd.concat(
        [
            pd.Series({"all": len(df)}),
            df["document_quality"].value_counts(),
            df["scope"].value_counts(),
            df[df["scope"] == "in_scope"]["format"].value_counts(),
            df["rotation"].ne(0).map({True: "rotated", False: "upright"}).value_counts(),
        ]
    )
    out = pd.concat(parts, ignore_index=True)[["group", "value"] + metrics]
    out["n_docs"] = out["value"].map(counts)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-file", type=Path, default=OCR_JSONL)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    df = score(load_ocr_text(args.ocr_file))
    summary = summarize(df)
    print(summary.round(3).to_string(index=False))
    if not args.no_save:
        RESULTS_DIR.mkdir(exist_ok=True)
        df.round(4).to_csv(RESULTS_DIR / "ocr_quality_by_document.csv", index=False)
        summary.round(4).to_csv(RESULTS_DIR / "ocr_quality_summary.csv", index=False)


if __name__ == "__main__":
    main()
