# OCR Text Channel

**Task:** #3 (build and cache the OCR-based embedding channel). Related GitHub issue: #5.
**Code:** [`features/ocr.py`](../features/ocr.py) → [`features/text_embeddings.py`](../features/text_embeddings.py). OCR accuracy: [`analysis/ocr_quality.py`](../analysis/ocr_quality.py).

```bash
python -m features.ocr              # ~2 min on a 14-core laptop, resumable
python -m features.text_embeddings  # 4 channels into cache/embeddings/
python -m analysis.ocr_quality      # results/ocr_quality_*.csv
```

`cache/` is git-ignored. Each teammate rebuilds it with the commands above, or copies it from the shared Drive.

## Method

| Step | Choice | Why |
|---|---|---|
| OCR engine | **Tesseract 5.5.3**, LSTM engine (`--oem 1`), automatic page segmentation (`--psm 3`), English | Open source, runs locally (documents never leave the machine), and is the engine named in issue #5 |
| Image prep | RGB → grayscale; long edge under 1,000 px upscaled 2×; long edge over 3,500 px downscaled | Tesseract misses small text; giant scans are slow and gain nothing |
| Orientation | Tesseract's own OSD (`--psm 0`); rotate when confidence ≥ 2 | Fixes rotated scans **without** reading the dataset's `rotation` metadata, which a real intake queue wouldn't have |
| Cache | `cache/ocr/tesseract.jsonl`, one line per document, written as it goes | Survives an interrupted run or a Colab disconnect |
| Text embedder 1 | **TF-IDF** (word 1–2-grams, `min_df=2`, sublinear tf) → **TruncatedSVD 256-d** | The overview's "OCR + TF-IDF" baseline. Fitted on **train-split text only** |
| Text embedder 2 | **`BAAI/bge-base-en-v1.5`** (110M parameters), via sentence-transformers 6.2.0 | Strong for its size on the MTEB leaderboard, and fast on a laptop GPU. The 512-token window is shorter than most documents, so text is cut into 300-word chunks and the chunk vectors are averaged |
| Upper bound | The same two embedders on the **ground-truth transcription** (`gt_tfidf`, `gt_bge`) | Shows how much of the text channel's error comes from OCR mistakes |

All vectors are L2-normalized and saved as `cache/embeddings/{ocr,gt}_{tfidf,bge}.npy`. Row *i* is document id *i*.

## OCR accuracy vs. ground truth

Word recall is the share of ground-truth words that Tesseract also found (ignoring order). Word precision is the share of Tesseract's words that are real. See `results/ocr_quality_summary.csv`.

| Group | Docs | Word recall | Word precision |
|---|---|---|---|
| **All** | 1,000 | **0.77** | **0.85** |
| CLEAN | 338 | 0.77 | 0.85 |
| HIGH_QUALITY | 303 | 0.82 | 0.87 |
| LOW_QUALITY | 259 | 0.78 | 0.86 |
| PHOTO | 100 | 0.62 | 0.75 |
| COMMERCIAL_LEASE_AGREEMENT | 52 | 0.90 | 0.89 |
| PROXY_VOTING | 50 | 0.86 | 0.90 |
| PATENT | 52 | 0.80 | 0.75 |
| PETITION_FORM | 51 | 0.74 | 0.73 |
| **REAL_ESTATE** | 59 | **0.52** | 0.78 |
| Rotated (metadata says 90/180/270°) | 43 | 0.49 | 0.59 |

**Orientation detection** raised word recall on the 43 rotated documents from **0.27 to 0.49**. It also wrongly flipped 9 upright documents by 180°, so recall on upright pages fell from 0.792 to 0.786. Overall it's a small net gain (0.770 → 0.773). A stricter confidence threshold for 180° turns is an easy follow-up.

**Why REAL_ESTATE is low:** these aren't deeds. They're real-estate *market reports*, built from charts and dashboards. The ground-truth transcription includes data tables reconstructed from bar charts (for example "Q1-2017 | 5 | Retail"), and no OCR engine can read those numbers off bars. So the low recall is mostly a ground-truth effect, not Tesseract failing on legal text. It matters for extraction later: REAL_ESTATE fields come from charts.

**CLEAN is not the best tier.** CLEAN pages score below HIGH_QUALITY. The CLEAN group includes more chart-heavy and table-heavy types, so the type mix matters more than the noise level. Compare quality tiers within one type before drawing conclusions.

## Known limitations

- English only (`-l eng`). No layout or reading-order model, so tables come out row by row with broken columns.
- Mean-pooling bge chunks treats a 1-page form and a 20-page lease the same way.
- TF-IDF/SVD is fitted on train text, but every document is transformed with it, including unseen tail types. That's correct for evaluation, but rare-type vocabulary is mostly missing from it.
- Issue #5 (James) is building a pytesseract channel in Colab. When that's ready, compare it with this one by running `analysis/ocr_quality.py --ocr-file <his jsonl>` and keep the better settings.
