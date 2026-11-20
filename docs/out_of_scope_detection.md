# Out-of-Scope Detection: Confidence vs. Distance

**Tasks:** October milestone, workstream 3 (3a to 3f), and the detector part of issue #9.
**Code:** [`analysis/legal_classifier.py`](../analysis/legal_classifier.py), [`analysis/detection.py`](../analysis/detection.py), tests in [`tests/test_detection.py`](../tests/test_detection.py). Builds on the splits, embeddings and k-NN detector from PR #7.

```bash
python -m features.vision_embeddings   # GPU: about 15 minutes on an A6000, downloads included
python -m features.ocr                 # CPU: about 20 minutes with 24 workers
python -m features.text_embeddings
python -m analysis.legal_classifier    # results/classifier_*.csv
python -m analysis.detection           # results/detection_*.csv, results/figures/detection_confusion.png
```

All 8 channels from PR #7 are covered (4 vision, 4 text). The 2 `gt_` channels use the clean transcription, so they are an upper bound we won't have in production. Embeddings and OCR were computed on an A6000 machine with Tesseract 5.5.3, the version used in PR #7. As a check, rerunning `analysis/out_of_scope.py` on them gives the same ROC-AUC, precision, recall and F1 as PR #7's `results/out_of_scope_detection.csv`. Only the two OCR thresholds and a few photo scores differ, by less than 0.01, and no document's flag changes.

## 3a. What counts as in scope

- **In scope (should pass):** the 5 legal types in `IN_SCOPE_FORMATS` (`data/scope.py`): COMMERCIAL_LEASE_AGREEMENT, PATENT, PETITION_FORM, PROXY_VOTING, REAL_ESTATE.
- **Out of scope (should be flagged):** everything else. Reported in three groups, as `docs/scope_policy.md` asks:
  - `head_oos`: the 13 other common types (100 test documents)
  - `legal_adjacent`: account statements, credit card statements and Form 1040 (21 test documents, a subset of `head_oos`). These are the closest look-alikes.
  - `tail`: the 18 rare types (75 documents, all test). These are types the detector has never seen.
- **Splits:** `data/splits.csv`. Both detectors learn only from in-scope `train` documents. Every choice (classifier setting, threshold) is made on `val`. Test is used once.

In the metrics below, **positive = out of scope**, so recall is "share of out-of-scope documents caught" and precision is "share of flags that were right".

## 3b. Detector 1: classifier confidence

A logistic regression classifier sorts documents into the 5 legal types. A document's score is **1 minus the classifier's top probability**: if the classifier can't decide which legal type it is, it's probably not a legal document.

`route(model, X, threshold)` in `analysis/detection.py` returns the predicted type for each document, or `"REVIEW"` when the score is at or above the threshold. The threshold is a plain argument, so it can be tuned for how many documents a person is willing to review.

**The regularization strength C matters a lot here, more than for classification.** Almost every channel classifies the 5 types perfectly at every C, so classification accuracy can't choose C. C mostly changes how spread out the probabilities are, and this detector reads those. Two simpler rules were tried first, and both failed:

- **Strongest regularization (C = 0.001) on ties:** the TF-IDF probabilities became so flat that even legal documents scored a top probability of only 0.3 to 0.5. The detector then flagged all 39 legal test documents.
- **Best-calibrated on legal val documents (lowest log loss) on ties:** every channel except the pixel baseline picked C = 10, became overconfident, and SigLIP went from 0 errors to 5.

The rule used instead: among equally accurate settings, take the one whose confidence best separates legal from out-of-scope **val** documents (ROC-AUC, which needs no threshold). It's chosen on val only, like the threshold. The classifier script on its own still uses the log-loss rule, because its job is classification. The C each channel ended up with is in the `C` column of `results/detection_comparison.csv`.

## 3c. Detector 2: distance to legal documents

This is the k-NN detector from PR #7, reused unchanged: 1 minus the mean cosine similarity to the 5 nearest in-scope training documents. `results/detection_scores.csv` now saves each document's score, threshold, flag and predicted type for both detectors, which the October plan asked for.

## 3d. Thresholds

Each threshold is the score that **maximizes F1 for flagging on val**. That weighs missed out-of-scope documents and false alarms roughly equally, and it's the same rule PR #7 uses, so the results compare directly. Rare (tail) documents are never used to choose it.

How cleanly each threshold separates the **test** documents (production channels only):

| Channel | Detector | Threshold | Highest legal score | Lowest out-of-scope score |
|---|---|---|---|---|
| SigLIP | confidence | 0.223 | 0.160 | 0.265 |
| SigLIP | distance | 0.226 | 0.252 | 0.228 |
| ColQwen2 | confidence | 0.568 | 0.448 | 0.522 |
| ColQwen2 | distance | 0.311 | 0.125 | 0.146 |
| OCR + bge | confidence | 0.484 | 0.190 | 0.289 |
| OCR + bge | distance | 0.202 | 0.177 | 0.177 |
| DINOv2 | distance | 0.082 | 0.060 | 0.094 |

A clear gap means some threshold would get every test document right. SigLIP confidence, ColQwen2 (both detectors), OCR + bge confidence and DINOv2 distance all have one. For OCR + bge distance, the lowest out-of-scope score belongs to document 36, which is mislabeled (see 3f). Without it, that detector has a clear gap too.

For ColQwen2 (both detectors) and OCR + bge confidence, the threshold picked on val sits above the gap, so a few out-of-scope documents get through (see 3f). Val has no tail documents, so it can't show that the tail needs a lower threshold.

## 3e. Results on test

**The classifier.** Every learned channel classifies the 5 legal types perfectly on test (macro F1 1.00). The majority-class baseline gets macro F1 0.08 and accuracy 0.23.

**The detectors**, on all 214 test documents (39 legal, 175 out of scope):

| Channel | Detector | ROC-AUC | Precision | Recall | F1 | Legal flagged (of 39) | Out of scope accepted (of 175) |
|---|---|---|---|---|---|---|---|
| Pixel baseline | confidence | 0.84 | 0.95 | 0.79 | 0.86 | 7 | 37 |
| Pixel baseline | distance | 0.82 | 0.93 | 0.97 | 0.95 | 12 | 5 |
| DINOv2 | confidence | 0.99 | 1.00 | 0.98 | 0.99 | 0 | 3 |
| DINOv2 | distance | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| **SigLIP** | **confidence** | **1.00** | **1.00** | **1.00** | **1.00** | **0** | **0** |
| SigLIP | distance | 1.00 | 0.97 | 1.00 | 0.99 | 5 | 0 |
| ColQwen2 | confidence | 1.00 | 1.00 | 0.98 | 0.99 | 0 | 3 |
| ColQwen2 | distance | 1.00 | 1.00 | 0.97 | 0.98 | 0 | 6 |
| OCR + TF-IDF | confidence | 0.88 | 1.00 | 0.79 | 0.88 | 0 | 37 |
| OCR + TF-IDF | distance | 1.00 | 1.00 | 0.97 | 0.98 | 0 | 6 |
| OCR + bge | confidence | 1.00 | 1.00 | 0.98 | 0.99 | 0 | 3 |
| **OCR + bge** | **distance** | **1.00** | **1.00** | **0.994** | **0.997** | **0** | **1** |
| Clean text + bge (upper bound) | either | | 1.00 | 1.00 | 1.00 | 0 | 0 |

Per-group numbers (`head_oos`, `legal_adjacent`, `tail`) and confusion-matrix counts for every combination are in `results/detection_comparison.csv`. The matrices are plotted in `results/figures/detection_confusion.png`. Every learned detector catches **100% of the legal-adjacent documents**, except confidence on TF-IDF (86% on OCR text, 95% on clean text).

## 3f. What goes wrong, and why

I looked at every test document a learned detector got wrong, and at the 4 legal photos (scored leave-one-out, since all 4 are in train).

**1. Some detectors flag legal photos because they are photos.** The 4 legal photos are ids 32, 65, 135 and 760.

| Channel | Photos flagged by confidence (of 4) | Photos flagged by distance (of 4) |
|---|---|---|
| DINOv2 | 3 | 4 |
| SigLIP | 0 | 2 |
| ColQwen2 | 0 | 0 |
| OCR + TF-IDF | 3 | 1 |
| OCR + bge | 0 | 0 |

DINOv2 looks good on test only because test contains no legal photos. A real intake queue will get phone photos of leases, so DINOv2 is the riskiest choice, even though its distance detector is perfect on test.

**2. Charts get confidently called REAL_ESTATE.** DINOv2 confidence accepts two `CHART` pages (ids 18, 818) and a `PHOTO_CHART` (8) as REAL_ESTATE with over 99.9% confidence. The REAL_ESTATE documents in this dataset are chart pages, so any bar chart looks like one. Id 18 is a bank's liquidity slide. This is a known weak spot of the confidence detector: a document that looks like one legal type gets a confident answer.

**3. Tables and lists get mistaken for legal types.** Apart from document 36, the misses are table- or list-like pages. ColQwen2 confidence misses a `WEB_TABLE` (63) and a `NUTRITION` label (991), and OCR + bge confidence misses a `SCANNED_TABLE` (704) and a `LIST` (997). The classifier puts these in PATENT or REAL_ESTATE with a top probability of 0.43 to 0.63, which is low but not low enough to get flagged. ColQwen2 distance misses three `PAY_IN_SHEET` documents (130, 235, 308), the same 991, and 997. The confidence detector catches 130, 235, 308 and 997 (top probability 0.30 to 0.41), which is why combining the two helps (point 9). Id 130, for example, is a grid of names, ID numbers and amounts, much like a proxy form's vote tables.

**4. Document 36 is mislabeled.** It's labeled `PHOTO_TABLE`, a rare out-of-scope type, but the image is a photo of a **nominating petition**: party, candidate, office sought, and a signature table. Half of all the learned detectors accept it as legal, and the OCR + bge classifier calls it a PETITION_FORM with 71% confidence. It's the only miss of the best text detector (OCR + bge distance). It isn't in the list of legal-looking tail documents in `docs/scope_policy.md` (30, 43, 52, 426, 661), so it should be added there, or relabeled.

**5. SigLIP distance flags normal legal pages.** Its 5 false alarms (ids 83, 248 PROXY_VOTING; 215, 702, 705 REAL_ESTATE) are ordinary clean or low-quality pages with scores just above the threshold (0.227 to 0.252, threshold 0.226). SigLIP confidence has no such overlap.

**6. TF-IDF and the confidence detector don't mix.** Even with the best C, TF-IDF's top probabilities hardly differ between legal and out-of-scope documents (ROC-AUC 0.88 on OCR text, 0.71 on clean text). The same TF-IDF features work well with the distance detector (ROC-AUC 0.997 and 0.996). Confidence-based detection needs a classifier whose probabilities mean something. Dense neural embeddings (SigLIP, ColQwen2, bge) give that here, while TF-IDF with a linear model doesn't.

**7. OCR errors hurt on photos.** OCR + TF-IDF distance misses 6 tail documents, and all 6 are photos (ids 29, 30, 34, 36, 53, 407). One of them is the law-firm fax cover sheet (30). Noisy OCR text from photos falls closer to the legal documents than clean pages do. bge on the same OCR text misses none of these, apart from the mislabeled 36.

**8. The legal-looking rare documents are almost always caught.** Ids 43, 52 and 426 are flagged by 13 of the 14 learned detectors, 661 by 12, and 30 by 11. All of them are photos, so this dataset can't tell whether they're flagged for not being legal or for being photos.

**9. Combining the detectors helps only where they disagree.** Flagging when either detector fires cuts ColQwen2's misses from 3 and 6 to 2, and TF-IDF's from 37 and 6 to 2, with no new false alarms. Where one detector is already clean, it adds nothing, or inherits the other's false alarms (SigLIP).

## Recommendation

- **Vision: SigLIP with the confidence detector.** It's one of two production setups with zero test errors. The other one, DINOv2 distance, flags all 4 legal photos. SigLIP confidence flags none of them and reuses the classifier we need anyway.
- **Text: OCR + bge with the distance detector.** It has 1 error, the mislabeled document 36, and flags no legal photos. This matches PR #7's recommendation.
- Avoid DINOv2 (photos) and confidence on TF-IDF (flat probabilities).

The pipeline will run OCR anyway for extraction, so the text detector costs little extra. It could act as a second opinion on documents the vision detector is unsure about.

**Update (Nov 18): what the pipeline uses.** This replaces the vision pick above. At the Nov 10 lab, Grace picked the ColQwen2 embeddings for the classifier. So the pipeline ([`extraction/route.py`](../extraction/route.py)) uses ColQwen2 for the out-of-scope check too, with the `either` rule, and needs only one embedding model. This is the ColQwen2 detector PR #7 suggested, with the confidence check added. On test it sends no legal document to review and lets 2 of 175 out-of-scope documents through: 36 (the mislabeled petition photo) and 991 (a nutrition label). It also flags none of the 4 legal photos. SigLIP confidence has 0 test errors, but it would add a second embedding model. On val, all three ColQwen2 rules make no mistakes, so val could not choose between them. On test, confidence alone lets 3 through and distance alone lets 6 through. We picked `either` after seeing these test numbers, so "2 of 175" is a little optimistic.

## Caveats

- Test has only 39 legal documents, so each one moves precision or recall by about 2.5 points. The photo check rests on just 4 documents.
- One split and one seed, as in PR #7.
- The confidence detector depends on C (see 3b). C and the threshold are both chosen on the same 139 val documents, so the val-based choices could be slightly optimistic. Test is still untouched by both.
- The confidence detector has only seen 5 well-separated legal types. With more look-alike legal types, the top probability would drop for real legal documents too, and this detector would get weaker.
