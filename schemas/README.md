# Class-Level Extraction Schemas

Every document in the benchmark comes with its own `json_schema`, a list of the fields to extract from it. In the real pipeline, a new document arrives with **no schema**: the classifier predicts its type (say, `PATENT`), and extraction then needs a schema for that type.

This folder builds **one schema per document type** by merging the schemas of that type's training documents, and it measures how well each merged schema fits the data.

## Quick start

Run these from the repository root (the folder that contains `requirements.txt`). You need **Python 3.12 or newer**; check with `python3 --version`.

```bash
# 1. Create an isolated environment and install the pinned packages (one time only)
python3 -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Build the schemas and the report
python schemas/build_class_schemas.py

# 3. Run the tests
pytest
```

The first run downloads the dataset from Hugging Face (about 360 MB, no account needed). Later runs take a few seconds.

**Then look at:**
- `schemas/REPORT.md`, the results. In VS Code, open it and press `Cmd+Shift+V` (`Ctrl+Shift+V` on Windows) for a formatted preview.
- `schemas/classes/PATENT.json` (or any other type), an actual merged schema.

### On Google Colab

```python
!git clone https://github.com/Break-Through-Tech/Legal-2B-multimodal-document-understanding.git
%cd Legal-2B-multimodal-document-understanding
!git checkout schemas/class-schemas   # only needed until this branch is merged
!pip install -r requirements.txt
!python schemas/build_class_schemas.py
```

### Options

| Command | What it does |
|---|---|
| `python schemas/build_class_schemas.py` | Builds the team's 5 in-scope types plus the 3 legal-adjacent ones, with a provisional split |
| `... --classes PATENT PROXY_VOTING` | Builds only the types you list |
| `... --split-manifest data/splits.csv` | Uses the team's official split instead of the provisional one |
| `... --help` | Shows every option |

A split manifest is a CSV with `id` and `split` columns. Rows labeled `train` build the schemas and rows labeled `val` evaluate them. Rows with any other label, like `test`, are never read, so they stay clean for the final evaluation.

## Files

| File | What it is |
|---|---|
| `merge.py` | The merge rules. Pure functions: no file or network access, so they're easy to test. |
| `build_class_schemas.py` | The script you run: loads data, splits it, builds each schema, writes the outputs |
| `classes/<TYPE>.json` | Output: one merged schema per type (JSON Schema draft 2020-12) |
| `REPORT.md` | Output: the numbers, conflicts found while merging, and how many training documents have each field |
| `../tests/test_merge.py` | Unit tests for the merge rules |

## How it works

1. **Pick the types.** The list comes from the team's scope policy in `data/scope.py` (explained in `docs/scope_policy.md`): the 5 in-scope legal types, plus the 3 legal-adjacent ones while the team decides between 5 and 8 classes. If the policy changes, just rerun the script; no code changes needed.
2. **Use training documents only.** Schemas are built only from the training split, so the documents used to evaluate them never shape them. Until the team's split exists, the script uses a **provisional** stratified split: 70% of each type, seed 42.
3. **Merge the schemas** of each type's training documents:
   - keep every field any document has, and make a field required only if *every* document requires it;
   - combine types and allowed values (a field that's text in one document and a number in another accepts both);
   - keep other rules (`format`, `pattern`, `not`) only if every document agrees;
   - **close** each object, so fields never seen in training are rejected (for example, a field the extractor made up).
4. **Allow `null` for blank fields.** The dataset's answers use `null` when a field is empty on the page, and some field descriptions even say *"must return null if not signed"*. Yet the dataset's schemas never allow `null`. So every field accepts `null`, except the ones the dataset marks as never-null. This also lets the extractor answer "blank" instead of guessing.
5. **Check with code, not with a model.** Every score comes from `jsonschema` validation.

**What the merge guarantees:** if a training document's own schema accepts an answer, and the answer only uses fields that schema declares, the merged schema accepts it too. Every place where training schemas disagreed (on a type, allowed values, a constraint, or which fields are required) is listed under *Conflicts* in `REPORT.md`. Differences in description wording aren't listed; the most common wording is kept.

## Results

These are from the provisional split, where each type has 15 to 18 held-out documents; see `REPORT.md` for the full tables.

- **Fit:** all 8 merged schemas accept **100%** of held-out answers and cover 100% of their filled-in fields.
- **Other types rejected:** every schema rejects **100%** of other document types' held-out answers. This is mostly by design: objects are closed and different types use different field names. Treat it as a sanity check that nothing is wide open, not as proof of quality.
- **Where the gain comes from:** the dataset's own schemas accept only 13% of held-out `FORM_1040` answers, 40% for `PROXY_VOTING` and 53% for `CREDIT_CARD_STATEMENT`. The *Own + nulls* column shows that allowing `null` alone fixes this. So the merge doesn't raise validity: what it adds is **one reusable schema per type**, which the pipeline needs when a document arrives without its own.
- **Weak spot:** `PATENT`, `ACCOUNT_STATEMENT` and `FORM_1040` have no field that every document requires, so an empty answer `{}` still passes. The report flags this with ⚠️.

## Dataset issues found

- **Nulls vs. schemas** (above): anyone validating or scoring against the dataset's `json_schema` will hit this. The team's `docs/scope_policy.md` found the same thing across all 1,000 documents.
- **Outside the legal types**, two schema bugs:
  - `BANK_CHECK` uses `"type": "enum"` in all 52 schemas, which isn't valid JSON Schema.
  - `PATIENT_INTAKE` misspells `enum` as `emum` in 28 schemas. Validators silently ignore unknown words, so those allowed-value lists are never enforced.
  - `merge.py` refuses both with a clear error instead of guessing.
- **`PROXY_VOTING`:** all 50 schemas put `"required": ["name"]` on the `votingBreakdown` list itself, where it has no effect. The merge drops it and lists it as a conflict.

## Limitations and next steps

- Rerun with the team's official split once it exists (`--split-manifest`). The numbers may shift a little. The scope policy is already read automatically.
- Held-out sets are small (15 to 18 documents per type), so one surprising document can move a percentage a lot.
- Some schemas are wide (`FORM_1040` has 66 fields). If extraction struggles, rarely seen fields could be trimmed using the *Field support* tables in `REPORT.md`.
- Extraction scoring (November) needs a rule for fields a schema allows but a document's answer doesn't include.
