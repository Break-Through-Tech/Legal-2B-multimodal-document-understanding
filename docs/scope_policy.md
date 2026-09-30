# Document Scope Policy

**Tasks:** #2a (label parsing and EDA), #2b (scope policy and long-tail handling)
**Code:** [`data/scope.py`](../data/scope.py) applies this policy. [`notebooks/Label_EDA_and_Scope.ipynb`](../notebooks/Label_EDA_and_Scope.ipynb) reproduces every number below.

When this policy changes, update `data/scope.py` and this document in the same commit.

---

## Summary

| Scope | Formats | Documents | What it's used for |
|---|---|---|---|
| `in_scope` | 5 core legal types | 264 | Classifier train / val / test |
| `out_of_scope` | 13 other head classes | 661 | Negatives for the out-of-scope detector |
| `tail` | 18 rare classes (< 30 docs each) | 75 | Held-out novelty test for the detector only. Never trained on |
| **Total** | 36 | 1,000 | |

Every document gets exactly one `scope` value. No documents are dropped.

---

## 1. Labels (Task #2a)

The labels come from the `format` field inside the `metadata` JSON string. `data/load_data.py` parses them.

- **Normalization:** there are 37 raw `format` values. The only problem is `"SCANNED_TABLE "`, which has a trailing space. After `strip().upper()` there are **36 labels**. There are no case variants and no other whitespace issues.
- **Counts:** these match the table in `Challenge-Project-Overview.md` exactly. There are 18 head classes (925 docs) and 18 tail classes (75 docs).
- **Missing labels:** none. `format` and `documentQuality` are present on all 1,000 rows.
- **Near-duplicate names are not merged.** Pairs like `NUTRITION`/`PHOTO_NUTRITION` and `CHART`/`PHOTO_CHART`, and the four `*TABLE` labels, differ by how the document was captured. All of them are outside the legal label space, so merging them would not change the classifier.

## 2. Which types are in scope

**In scope (the classifier's 5 labels):** `REAL_ESTATE` (59), `COMMERCIAL_LEASE_AGREEMENT` (52), `PATENT` (52), `PETITION_FORM` (51), `PROXY_VOTING` (50).

Why:
- These are the "core legal" group in the project overview. They are the types a legal review queue would route to a legal extraction schema.
- They are also the hard case for the research question. They are all dense text with headers and signature blocks, so they test whether vision embeddings can separate documents whose layout is similar.
- Classes are balanced (50–59 each), so macro-F1 and accuracy won't diverge much. The majority-class baseline is about 22%.

**Legal-adjacent, currently out of scope:** `ACCOUNT_STATEMENT`, `CREDIT_CARD_STATEMENT`, `FORM_1040` (153 docs). These show up in discovery, but they are financial records rather than legal instruments. They are marked `out_of_scope` with `is_legal_adjacent = True`, so we can:
- report detector performance on them separately, since they are the most plausible "near-miss" negatives, and
- widen the label space to 8 classes later by moving them into `IN_SCOPE_FORMATS` in `data/scope.py`.

> **Team decision to confirm:** 5 classes (this policy) vs. 8 classes (add the legal-adjacent types). Five keeps the problem focused on confusable legal instruments. Eight gives the classifier more data and a harder label space. Changing it is a one-line edit.

**Out of scope (13 head classes, 661 docs):** `PATIENT_INTAKE`, `BANK_CHECK`, `SHIPPING_INVOICE`, `SHIFT_SCHEDULE`, `DELIVERY_NOTE`, `EQUIPMENT_INSPECTION`, `GLOSSARY`, `PAY_IN_SHEET`, `CHART`, `NUTRITION`, plus the 3 legal-adjacent types above. They have enough examples to split into train/val/test. The detector can use them as known negatives, or leave them unseen during training, depending on how it is designed.

## 3. The long tail: 75 documents in 18 classes (Task #2b)

**Policy: keep all 75, label them `tail`, never train on them, and use them only as a held-out test set for the out-of-scope detector.**

Why:
- **They can't be classes.** 11 of the 18 tail classes have 3 or fewer examples, and 4 have only one. None of them can be stratified into train/val/test.
- **They're the realistic case.** In production, the documents a review queue gets wrong are the unfamiliar ones. A detector that has never seen any tail type during training is the honest test of "flag what you don't recognize."
- **None of them is a legal type.** No tail class is one of the five in-scope formats, so none of them should be accepted by the classifier.
- **Dropping them would waste data.** They are the only truly novel documents we have.

We are **not** merging tail classes into head classes (for example `PHOTO_NUTRITION` into `NUTRITION`). Doing so would leak near-copies of known out-of-scope types into the "novel" test set.

### Confound: the tail is all photos

All **75 tail documents are `PHOTO` quality and have no `fontFamily`**, while only 25 of the 925 head documents are `PHOTO`. The head classes appear to be synthetic (rendered in known fonts, then degraded to CLEAN / HIGH_QUALITY / LOW_QUALITY). The tail appears to be real photos and scans from other sources.

| scope | CLEAN | HIGH_QUALITY | LOW_QUALITY | PHOTO |
|---|---|---|---|---|
| in_scope | 90 | 84 | 86 | 4 |
| out_of_scope | 248 | 219 | 173 | 21 |
| tail | 0 | 0 | 0 | 75 |

This means a detector could score well on the tail just by recognizing "this is a photo." To keep results honest:
1. **Report detector metrics on two negative sets separately.** One is held-out `out_of_scope` head documents (same rendering pipeline as in-scope documents). The other is `tail` documents (novel type and a different capture style).
2. **Check the 25 head `PHOTO` documents.** Four of them are in scope (2 `REAL_ESTATE`, 1 lease, 1 patent). If the detector flags those four, it has learned "photo" and not "legal type."
3. **Break down classifier and detector errors by `document_quality`.** November's error analysis already calls for this.

### Tail documents that look legal

Some tail documents contain legal correspondence even though their labels are generic. Examples are law-firm fax cover sheets (ids 30, 43, 52), a contract routing form (426) and a venue agreement (661). They stay `tail`, because they are not one of the five types. They are useful hard negatives, though, so check them first during error analysis.

## 4. Other data issues found during EDA

These don't change the scope, but later stages need to handle them.

| Issue | Affects | What to do |
|---|---|---|
| `true_markdown_output` is JSON-encoded (quoted, `\n` escaped) on **all 1,000** rows | OCR/text channel, TF-IDF baseline | `json.loads()` it before use |
| `fontFamily` missing on 124 rows (all tail + all `CHART`) | Only if used as a feature | Don't use metadata as model features. It leaks the source |
| `rotation` present on 43 rows (90/180/270°) | OCR, embeddings | Consider rotating before OCR; spot-check |
| 345 RGBA, 18 grayscale, 1 palette images | Embedding | `.convert("RGB")` |
| Image sizes 160×56 to ~6,900×7,000 px | Embedding, batching | Cap the long edge before batching |
| All 52 `BANK_CHECK` schemas are invalid JSON Schema (malformed `anyOf`) | Extraction validation | Out of scope. Note in limitations |
| 327 ground-truth outputs fail their own schema, always because of `null` in a non-nullable field. In scope: `PROXY_VOTING` 32/50 | Extraction scoring | Validate against a nullable-patched schema, or report both |
| Schemas per in-scope type: 1 each for `REAL_ESTATE`, lease and petition; 32 for `PATENT`; 50 for `PROXY_VOTING` | Extraction routing | Patent and proxy need the per-document schema or a canonical one |
| No duplicate images or transcriptions; ids are unique, 0–999 | — | — |

## 5. Next steps

- **Splits:** stratify `in_scope` by `format` into train/val/test with a fixed seed, and save the manifest (ids only) to git. Split `out_of_scope` the same way, so held-out negatives never overlap with training data. `tail` is test-only.
- **Confirm the 5- vs 8-class decision** with the team and Challenge Advisor.
