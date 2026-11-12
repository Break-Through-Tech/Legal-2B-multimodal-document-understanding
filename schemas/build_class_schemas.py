"""Build one extraction schema per document class from the benchmark's per-document schemas.

Usage (from the repository root):
    python schemas/build_class_schemas.py
    python schemas/build_class_schemas.py --classes PATENT PROXY_VOTING
    python schemas/build_class_schemas.py --split-manifest data/splits.csv

Outputs:
    schemas/classes/<CLASS>.json   one JSON schema per class
    schemas/REPORT.md              how well each schema fits the data

See schemas/README.md for the method and the reasoning behind each decision.
"""

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match

# Make the repository root importable, so this file works both as
# `python schemas/build_class_schemas.py` and as `python -m schemas.build_class_schemas`.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.load_data import CACHE_DIR, DATASET, build_labels
from data.scope import IN_SCOPE_FORMATS, LEGAL_ADJACENT_FORMATS
from schemas.merge import Conflict, Schema, allow_null, merge_schemas

CLASSES_DIR = REPO_ROOT / "schemas" / "classes"
REPORT_MD = REPO_ROOT / "schemas" / "REPORT.md"
JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"

# Which types get a schema comes from the team's scope policy (data/scope.py, explained in
# docs/scope_policy.md), so this script follows the policy automatically when it changes.
# The legal-adjacent types are built too: the team is still choosing between 5 and 8
# classes, and having their schemas ready keeps both options open.
DEFAULT_CLASSES = IN_SCOPE_FORMATS + [
    name for name in LEGAL_ADJACENT_FORMATS if name not in IN_SCOPE_FORMATS
]

# Stand-in split, used only until the team's official train/val/test manifest exists.
PROVISIONAL_SEED = 42
PROVISIONAL_TRAIN_FRAC = 0.7

# With a team manifest, rows with this label are used to evaluate the schemas. Rows with any
# other label (like "test") are never read, so they stay clean for the final evaluation.
EVAL_SPLIT = "val"

MAX_FAILURES_LISTED = 5


@dataclass
class ClassResult:
    """Everything the report needs for one class."""

    name: str
    scope: str  # where the team's scope policy puts this type, e.g. "in scope"
    schema: Schema
    conflicts: list[Conflict]
    n_train: int
    n_heldout: int
    n_train_schemas: int  # how many different schemas the training documents have
    field_support: dict[str, float]  # field path -> share of training documents whose schema has it
    # Validity on held-out documents: share of ground truths that pass the schema.
    own_valid: float  # each document against its own schema (the dataset's baseline)
    own_with_nulls_valid: float  # same, after allow_null(), to isolate the effect of nulls
    class_valid: float  # against the merged class schema
    coverage: float  # share of filled-in held-out fields the class schema describes
    other_accepted: float  # share of other classes' held-out ground truths the class schema accepts
    failures: list[tuple[int, str]]  # (document id, validation error) for held-out failures


# --- Loading data and splitting it --------------------------------------------------------


def load_documents() -> pd.DataFrame:
    """Every document with its class, parsed schema, and parsed ground truth."""
    # Reuse the team's loader for class labels, so names match the rest of the project.
    labels = build_labels()[["id", "format"]]

    dataset = load_dataset(DATASET, split="test", cache_dir=str(CACHE_DIR))
    content = dataset.select_columns(["id", "json_schema", "true_json_output"]).to_pandas()

    # Sorting by id matters: the seeded split and the order of fields in each merged schema
    # both depend on row order, so a fixed order makes every run give the same output.
    docs = labels.merge(content, on="id").sort_values("id", ignore_index=True)
    docs["schema"] = docs["json_schema"].map(json.loads)
    docs["truth"] = docs["true_json_output"].map(json.loads)
    return docs[["id", "format", "schema", "truth"]]


def provisional_split(docs: pd.DataFrame) -> pd.Series:
    """A stand-in split until the team's official one exists: "train" or "heldout" per document.

    Stratified: each class is sampled separately, so every class gets the same share of
    training documents, and the fixed seed makes the split identical on every run.
    """
    split = pd.Series("heldout", index=docs.index)
    for _, class_docs in docs.groupby("format"):
        train_docs = class_docs.sample(frac=PROVISIONAL_TRAIN_FRAC, random_state=PROVISIONAL_SEED)
        split[train_docs.index] = "train"
    return split


def split_from_manifest(docs: pd.DataFrame, path: Path) -> pd.Series:
    """Read the team's split manifest: a CSV with 'id' and 'split' columns.

    Returns "train", "heldout", or NaN (not used) per document. Rows labeled 'train' build
    the schemas, rows labeled EVAL_SPLIT ('val') evaluate them, and everything else (like
    'test') is left out completely.
    """
    # Check the file and its contents first, so mistakes get a clear message instead of a
    # confusing Python error.
    if not path.exists():
        raise SystemExit(f"Split manifest not found: {path}")
    manifest = pd.read_csv(path)
    if not {"id", "split"}.issubset(manifest.columns):
        raise SystemExit(f"{path} needs 'id' and 'split' columns, found {list(manifest.columns)}")

    ids = pd.to_numeric(manifest["id"], errors="coerce")  # anything that isn't a number becomes NaN
    if ids.isna().any() or (ids % 1 != 0).any():
        raise SystemExit(f"{path}: every 'id' must be a whole number")
    if manifest["split"].isna().any():
        raise SystemExit(f"{path}: some rows have an empty 'split'")
    manifest["id"] = ids.astype(int)

    # Be forgiving about spelling: " Train" and "train" mean the same thing.
    manifest["split"] = manifest["split"].astype(str).str.strip().str.lower()

    # An id listed twice (say, as both train and test) would leak between splits.
    repeated = manifest["id"][manifest["id"].duplicated()].unique().tolist()
    if repeated:
        raise SystemExit(f"{path}: these ids appear more than once: {sorted(repeated)[:10]}")

    labels_found = set(manifest["split"])
    for label in ["train", EVAL_SPLIT]:
        if label not in labels_found:
            raise SystemExit(f"{path}: no rows labeled '{label}' (labels found: {sorted(labels_found)})")

    unknown_ids = set(manifest["id"]) - set(docs["id"])
    if unknown_ids:
        print(f"Note: {len(unknown_ids)} id(s) in {path} are not in the dataset and were skipped.")

    # Look up each document's label, then translate it: "train" stays "train", the evaluation
    # label becomes "heldout", and everything else (including ids not in the manifest) is NaN.
    split_by_id = dict(zip(manifest["id"], manifest["split"]))
    labels = docs["id"].map(split_by_id)
    return labels.map({"train": "train", EVAL_SPLIT: "heldout"})


# --- Walking schemas and ground truths ----------------------------------------------------


def _join(prefix: str, name: str) -> str:
    """Build a field path like "company.address"."""
    return f"{prefix}.{name}" if prefix else name


def iter_leaves(schema: Schema, prefix: str = "") -> Iterator[tuple[str, Schema]]:
    """Yield (path, schema) for every leaf field, e.g. "transactions[].amount".

    Nested fields are joined with dots, and "[]" marks the entries of a list.
    """
    children = []
    for name, child in schema.get("properties", {}).items():
        children.append((_join(prefix, name), child))
    if "items" in schema:
        children.append((prefix + "[]", schema["items"]))

    if not children:
        yield prefix, schema
    for path, child in children:
        yield from iter_leaves(child, path)


def filled_paths(value: object, prefix: str = "") -> set[str]:
    """Paths of every filled-in (non-null) field in a JSON value, in the same format as iter_leaves."""
    paths: set[str] = set()
    if isinstance(value, dict):
        for name, child in value.items():
            paths |= filled_paths(child, _join(prefix, name))
    elif isinstance(value, list):
        for child in value:
            paths |= filled_paths(child, prefix + "[]")
    elif value is not None:
        paths.add(prefix)
    return paths


# --- Building and measuring one class -----------------------------------------------------


def make_validator(schema: Schema) -> Draft202012Validator:
    # format_checker makes "format" rules (like "date") actually get checked.
    return Draft202012Validator(schema, format_checker=Draft202012Validator.FORMAT_CHECKER)


def share(flags: list[bool]) -> float:
    """Share of True values, or NaN when there's nothing to measure."""
    return sum(flags) / len(flags) if flags else float("nan")


def field_coverage(schema: Schema, truths: list[object]) -> float:
    """Share of filled-in ground-truth fields that the schema describes at all."""
    known = {path for path, _ in iter_leaves(schema)}
    total = 0
    covered = 0
    for truth in truths:
        for path in filled_paths(truth):
            total += 1
            if path in known:
                covered += 1
    return covered / total if total else float("nan")


def scope_label(name: str) -> str:
    """Where the team's scope policy (data/scope.py) puts a document type."""
    if name in IN_SCOPE_FORMATS:
        return "in scope"
    if name in LEGAL_ADJACENT_FORMATS:
        return "legal-adjacent"
    return "out of scope"


def build_class(name: str, docs: pd.DataFrame, split: pd.Series) -> ClassResult:
    """Merge one class's training schemas, then measure how well the result fits the data."""
    in_class = docs["format"] == name
    train = docs[in_class & (split == "train")]
    heldout = docs[in_class & (split == "heldout")]
    # Other classes are also limited to held-out rows, so test rows are never read.
    other_heldout = docs[~in_class & (split == "heldout")]
    if train.empty:
        raise SystemExit(f"{name}: no training documents in the split")

    # 1. Build: merge the training schemas, allow null for blank fields, add a title.
    merged, conflicts = merge_schemas(list(train["schema"]))
    body = allow_null(merged)
    body.pop("title", None)
    # Put "$schema" and "title" first, then copy in every key of body (** unpacks a dict).
    schema = {"$schema": JSON_SCHEMA_DIALECT, "title": name, **body}

    # 2. Measure validity on held-out documents, plus two baselines for comparison.
    validator = make_validator(schema)
    heldout_pairs = list(zip(heldout["schema"], heldout["truth"]))
    own_valid = share([make_validator(s).is_valid(t) for s, t in heldout_pairs])
    own_with_nulls_valid = share([make_validator(allow_null(s)).is_valid(t) for s, t in heldout_pairs])
    class_valid = share([validator.is_valid(t) for t in heldout["truth"]])

    # 3. A schema that also accepts other document types is a weak check; measure that too.
    other_accepted = share([validator.is_valid(t) for t in other_heldout["truth"]])

    failures = []
    for doc_id, truth in zip(heldout["id"], heldout["truth"]):
        error = best_match(validator.iter_errors(truth))  # None when the truth is valid
        if error is not None:
            failures.append((doc_id, error.message))

    # 4. Field support: in how many training documents' schemas each field appears.
    docs_with_field: Counter[str] = Counter()
    for doc_schema in train["schema"]:
        docs_with_field.update({path for path, _ in iter_leaves(doc_schema)})
    field_support = {path: docs_with_field[path] / len(train) for path, _ in iter_leaves(schema)}

    distinct_train_schemas = {json.dumps(s, sort_keys=True) for s in train["schema"]}

    return ClassResult(
        name=name,
        scope=scope_label(name),
        schema=schema,
        conflicts=conflicts,
        n_train=len(train),
        n_heldout=len(heldout),
        n_train_schemas=len(distinct_train_schemas),
        field_support=field_support,
        own_valid=own_valid,
        own_with_nulls_valid=own_with_nulls_valid,
        class_valid=class_valid,
        coverage=field_coverage(schema, list(heldout["truth"])),
        other_accepted=other_accepted,
        failures=failures,
    )


# --- Writing the report -------------------------------------------------------------------


def pct(value: float) -> str:
    """Format 0.5 as "50%", and NaN (nothing to measure) as "n/a"."""
    return "n/a" if pd.isna(value) else f"{value:.0%}"


def render_report(results: list[ClassResult], split_note: str) -> str:
    """Build the text of REPORT.md: a summary table, then one section per class."""
    lines = [
        "# Class Schema Report",
        "",
        (
            "Generated by `python schemas/build_class_schemas.py`; do not edit by hand. "
            "It covers the classes built in the latest run. Method: [README.md](README.md)."
        ),
        "",
        f"**Split:** {split_note}",
        "",
        "## Summary",
        "",
        (
            "All validity numbers are on **held-out** documents: the share of ground-truth JSONs "
            "that pass `jsonschema` validation."
        ),
        "",
        "- **Scope**: where the team's scope policy (`docs/scope_policy.md`) puts the type.",
        "- **Own schema**: each document's own schema from the dataset (the baseline).",
        "- **Own + nulls**: the same, but allowing null. Shows how much of the gain is just the null rule.",
        "- **Class schema**: the merged schema in `classes/`.",
        "- **Coverage**: share of filled-in held-out fields the class schema describes.",
        (
            "- **Other docs accepted**: share of *other* classes' held-out ground truths the class schema "
            "accepts. Lower is better; a high number means the schema is a weak check. Expect 0%: different "
            "types use different field names, so this is a sanity check rather than proof of quality."
        ),
        "",
        (
            "| Class | Scope | Train / held-out docs | Distinct train schemas | Fields | Own schema | "
            "Own + nulls | Class schema | Coverage | Other docs accepted |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| `{r.name}` | {r.scope} | {r.n_train} / {r.n_heldout} | {r.n_train_schemas} | {len(r.field_support)} | "
            f"{pct(r.own_valid)} | {pct(r.own_with_nulls_valid)} | {pct(r.class_valid)} | "
            f"{pct(r.coverage)} | {pct(r.other_accepted)} |"
        )
    for r in results:
        lines.append("")
        lines.extend(class_section(r))
    return "\n".join(lines) + "\n"


def class_section(r: ClassResult) -> list[str]:
    """The REPORT.md section for one class: warnings, conflicts, failures, and field support."""
    lines = [f"## `{r.name}`", "", f"Schema: [`classes/{r.name}.json`](classes/{r.name}.json)", ""]

    if not r.schema.get("required"):
        lines.append("⚠️ No field is required at the top level, so an empty object `{}` passes this schema.")
        lines.append("")

    if r.conflicts:
        lines.append("Conflicts resolved while merging (choices the merge had to make):")
        for conflict in r.conflicts:
            lines.append(f"- `{conflict.path}` {conflict.keyword}: {conflict.detail}")
    elif r.n_train_schemas == 1:
        lines.append("All training schemas are identical, so there was nothing to merge.")
    else:
        lines.append(
            f"The {r.n_train_schemas} distinct training schemas never disagree on a shared field's type, "
            "allowed values, constraints, or required status. They differ in which fields they include "
            "(and sometimes in description wording, where the most common wording is kept)."
        )

    if r.failures:
        lines.append("")
        lines.append(f"Held-out documents that fail the class schema ({len(r.failures)}):")
        for doc_id, message in r.failures[:MAX_FAILURES_LISTED]:
            lines.append(f"- id {doc_id}: {message[:120]}")

    lines += [
        "",
        "<details><summary>Field support</summary>",
        "",
        "| Field | Type | In % of training documents |",
        "|---|---|---|",
    ]
    for path, node in iter_leaves(r.schema):
        declared = node.get("type", "any")
        # A plain "|" would start a new table column in Markdown, so it's written as "\|".
        shown = " \\| ".join(declared) if isinstance(declared, list) else declared
        lines.append(f"| `{path}` | {shown} | {pct(r.field_support[path])} |")
    lines += ["", "</details>"]
    return lines


# --- Command line -------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES,
                        help="document classes to build (default: in-scope + legal-adjacent from data/scope.py)")
    parser.add_argument("--split-manifest", type=Path,
                        help="CSV with 'id' and 'split' columns (default: a provisional split)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    docs = load_documents()

    unknown = sorted(set(args.classes) - set(docs["format"]))
    if unknown:
        raise SystemExit(f"Unknown class(es): {unknown}")

    if args.split_manifest:
        split = split_from_manifest(docs, args.split_manifest)
        split_note = (f"team split manifest `{args.split_manifest}`: 'train' builds the schemas, "
                      f"'{EVAL_SPLIT}' evaluates them.")
    else:
        split = provisional_split(docs)
        split_note = (f"**provisional** stratified split (seed {PROVISIONAL_SEED}, {PROVISIONAL_TRAIN_FRAC:.0%} train "
                      "per class). Rerun with `--split-manifest` once the team's official split exists.")

    CLASSES_DIR.mkdir(exist_ok=True)
    results = []
    for name in args.classes:
        result = build_class(name, docs, split)
        schema_text = json.dumps(result.schema, indent=2, ensure_ascii=False) + "\n"
        (CLASSES_DIR / f"{name}.json").write_text(schema_text)
        results.append(result)
        print(f"{name:<28} fields={len(result.field_support):>3}   held-out valid={pct(result.class_valid):>4}   "
              f"other docs accepted={pct(result.other_accepted):>4}")

    REPORT_MD.write_text(render_report(results, split_note))
    print(f"\nWrote {len(results)} schemas to {CLASSES_DIR.relative_to(REPO_ROOT)}/ "
          f"and the report to {REPORT_MD.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
