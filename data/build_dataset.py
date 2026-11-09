"""Save the clean, processed document table (task #2e).

Writes data/documents.csv: one row per document with its cleaned label, scope, split and
metadata. It holds no images or text, so it is small enough to commit, and it is the one
table every later stage should join on (by `id`).

Columns:
    id, format, document_quality, scope, is_head_class, is_legal_adjacent, split,
    font_family, rotation, width, height, image_mode,
    schema_id         short hash of the document's json_schema (same hash = same schema)
    gt_json_valid     True if true_json_output parses as JSON
    gt_markdown_chars length of the decoded ground-truth transcription

The ground-truth markdown ships JSON-encoded (quoted, with escaped newlines) on every row.
`decode_markdown` undoes that; use it anywhere the transcription is read.

Usage:
    python data/build_dataset.py
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset

try:  # imported as data.build_dataset
    from .load_data import CACHE_DIR, DATASET
    from .splits import SPLITS_CSV, load_scoped_labels
except ImportError:  # run as a script
    from load_data import CACHE_DIR, DATASET
    from splits import SPLITS_CSV, load_scoped_labels

DOCUMENTS_CSV = Path(__file__).parent / "documents.csv"


def load_raw_dataset():
    """The benchmark's single split of 1,000 rows, cached in data/raw/."""
    return load_dataset(DATASET, split="test", cache_dir=str(CACHE_DIR))


def decode_markdown(raw: str) -> str:
    """Ground-truth markdown is a JSON string literal; return the plain text."""
    try:
        text = json.loads(raw)
        return text if isinstance(text, str) else raw
    except (json.JSONDecodeError, TypeError):
        return raw


def load_documents() -> pd.DataFrame:
    """The processed table, indexed in id order (row i is document id i)."""
    docs = pd.read_csv(DOCUMENTS_CSV).sort_values("id").reset_index(drop=True)
    assert (docs["id"] == range(len(docs))).all(), "documents.csv must cover ids 0..N-1"
    return docs


def build_documents() -> pd.DataFrame:
    labels = load_scoped_labels()
    splits = pd.read_csv(SPLITS_CSV)
    ds = load_raw_dataset()

    rows = []
    for record in ds:
        meta = json.loads(record["metadata"])
        image = record["image"]
        try:
            json.loads(record["true_json_output"])
            gt_valid = True
        except json.JSONDecodeError:
            gt_valid = False
        rows.append(
            {
                "id": record["id"],
                "font_family": meta.get("fontFamily"),
                "rotation": meta.get("rotation", 0),
                "width": image.width,
                "height": image.height,
                "image_mode": image.mode,
                "schema_id": hashlib.sha1(record["json_schema"].encode()).hexdigest()[:10],
                "gt_json_valid": gt_valid,
                "gt_markdown_chars": len(decode_markdown(record["true_markdown_output"])),
            }
        )

    docs = (
        labels.merge(splits, on="id", validate="one_to_one")
        .merge(pd.DataFrame(rows), on="id", validate="one_to_one")
        .sort_values("id")
    )
    columns = [
        "id", "format", "document_quality", "scope", "is_head_class", "is_legal_adjacent", "split",
        "font_family", "rotation", "width", "height", "image_mode",
        "schema_id", "gt_json_valid", "gt_markdown_chars",
    ]
    return docs[columns]


def main() -> None:
    docs = build_documents()
    docs.to_csv(DOCUMENTS_CSV, index=False)
    print(f"Wrote {DOCUMENTS_CSV} ({len(docs)} rows, {len(docs.columns)} columns)")
    print(docs.groupby(["scope", "split"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
