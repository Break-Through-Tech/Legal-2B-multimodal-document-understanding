# Train / Val / Test Splits

**Task:** #2d (stratified splits) and #2e (processed dataset)
**Code:** [`data/splits.py`](../data/splits.py) writes [`data/splits.csv`](../data/splits.csv). [`data/build_dataset.py`](../data/build_dataset.py) writes [`data/documents.csv`](../data/documents.csv). [`tests/test_splits.py`](../tests/test_splits.py) checks both.

Both CSVs are committed. **Use them; don't re-split in a notebook.** Every experiment then trains and scores on the same documents, so results are comparable across teammates.

```bash
python data/load_data.py     # labels.csv (not committed)
python data/scope.py         # adds scope columns
python data/splits.py        # data/splits.csv
python data/build_dataset.py # data/documents.csv
pytest tests/test_splits.py
```

## Strategy

| Scope | Docs | Split | Why |
|---|---|---|---|
| `in_scope` (5 legal types) | 264 | 70 / 15 / 15, stratified by `format` | Classifier train / tune / report |
| `out_of_scope` (13 head types) | 661 | 70 / 15 / 15, stratified by `format` | Detector negatives; held-out ones never overlap with training |
| `tail` (18 rare types) | 75 | **test only** | Novelty test. Never trained on, never used to tune a threshold |

- **Seed:** `42` (`SEED` in `data/splits.py`). Two `train_test_split` calls: 70% train, then the remaining 30% split in half.
- **Stratification:** by `format`, so each of the 18 head types keeps about 70/15/15 in every split. The test file checks this.
- **Not stratified by `document_quality`.** Some format × quality cells have 1–2 documents (for example the 4 in-scope `PHOTO` documents), which can't be split three ways. Quality is roughly balanced anyway. Break results down by `document_quality` in `documents.csv` when you report them.

### Resulting counts

| scope | train | val | test | total |
|---|---|---|---|---|
| in_scope | 184 | 41 | 39 | 264 |
| out_of_scope | 463 | 98 | 100 | 661 |
| tail | 0 | 0 | 75 | 75 |
| **total** | **647** | **139** | **214** | **1,000** |

Per in-scope type: train 35–41, val 8–9, test 7–9.

### Where tail documents go: a change from the planning sheet

The September planning sheet said tail documents should go to **validation and test**. The scope policy merged in PR #3 says **test only**. This manifest follows the scope policy, because any tail document in val could be used to tune the detector's threshold. Then the tail would no longer test "flag what you have never seen." If the team prefers the sheet's version, change the one line that assigns `TAIL` in `build_splits` and re-run. The test file will need its `test_tail_is_test_only` check updated too.

### 5 vs. 8 in-scope classes

The splits follow the current scope policy (5 classes). The 3 legal-adjacent types are `out_of_scope` but are stratified exactly like the in-scope types. If the team widens the label space to 8 classes, **the split assignments don't change.** Every legal-adjacent document already has a train/val/test label.

## `data/documents.csv` (task #2e)

One row per document: no images and no text, about 100 KB. Join on `id`.

| Column | Meaning |
|---|---|
| `id` | 0–999, same as the dataset |
| `format` | Cleaned label (`strip().upper()`, 36 values) |
| `document_quality` | CLEAN / HIGH_QUALITY / LOW_QUALITY / PHOTO |
| `scope`, `is_head_class`, `is_legal_adjacent` | From `data/scope.py` |
| `split` | From `data/splits.csv` |
| `font_family`, `rotation` | From metadata. Missing `font_family` = tail or `CHART`. **Analysis only, never model features:** they leak the source |
| `width`, `height`, `image_mode` | Image size and PIL mode (RGB / RGBA / L / P) |
| `schema_id` | First 10 hex characters of the SHA-1 of `json_schema`. Equal ids mean an identical schema (526 distinct) |
| `gt_json_valid` | `true_json_output` parses as JSON (all 1,000 do) |
| `gt_markdown_chars` | Length of the decoded ground-truth transcription |

`data/build_dataset.py` also provides `decode_markdown()`, which undoes the JSON encoding on `true_markdown_output`, and `load_documents()`, which loads the table in id order.
