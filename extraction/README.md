# Extraction (issue #10)

The pipeline, from a page image to JSON:

1. **Classify and check scope** ([`route.py`](route.py)). A logistic regression on the ColQwen2 embeddings gives each document one of the 5 legal types. The document goes to `REVIEW` instead if the classifier is unsure or if the page is far from the legal training pages. This is the `either` rule from [`docs/out_of_scope_detection.md`](../docs/out_of_scope_detection.md).
2. **Extract** ([`run_vlm.py`](run_vlm.py)). A small open-source vision-language model gets the page image and the schema for the predicted type ([`schemas/classes/`](../schemas/classes/)) and replies with JSON. Documents in `REVIEW` are skipped.
3. **Score** (issue #14, Chinmaya's, described in #11). Validate the JSON against the schema and compare it with the ground truth.

## Model

| | |
|---|---|
| Model | [`Qwen/Qwen3-VL-4B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) (4B parameters, Apache-2.0) |
| Revision | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| Decoding | greedy (no random sampling), so a rerun on the same GPU and software should give the same replies |
| Image size | up to about 4.2 million pixels. A bit under half the pages are bigger and get scaled down, most of them only a little. |
| Max reply length | 4,096 tokens |
| Ran on | NVIDIA RTX A6000, torch 2.15.0, transformers 5.22.2 |

Issue #10 asked for a small (~3B) open-source Qwen-VL instruct model. We used Qwen3-VL-4B-Instruct (October 2025), which is newer than Qwen2.5-VL-3B. It runs on our own GPU, so the documents never leave our machine. We tried only this one model. Good ones to compare next are Qwen3-VL-2B-Instruct (closer to 3B, and easier to fit on a Colab GPU) and the newer Qwen3.5-4B (February 2026). The weights are about 9 GB in bfloat16, and `nvidia-smi` showed about 11 GB of GPU memory in use during our runs. The code has only been run on the A6000.

The prompt (in [`prompt.py`](prompt.py)) names the type, includes the whole schema, and asks for the JSON object only, with `null` for missing fields. A sample of the prompt, with the schema left out, is saved in the `_run.json` file next to each output.

## How to run

From the repository root:

```bash
python -m extraction.route                         # step 1, CPU, needs cache/embeddings/vision_colqwen2.npy
python -m extraction.run_vlm --split val --limit 5 # step 2, quick check on a GPU
python -m extraction.run_vlm --split test          # step 2, the test run
```

`routing.csv` is committed, so step 2 doesn't need the embeddings. `run_vlm.py` writes each reply as soon as it's ready. If the run stops, run the same command again and it skips the documents already done. A page that runs out of GPU memory is not saved, so the next run tries it again.

The committed output files already have every document, so step 2 skips them all. To rerun, first move `results/extraction/qwen3-vl-4b-instruct_<split>.jsonl` and its `_run.json` somewhere else. The run stops if the model or settings differ from the ones in `_run.json`, but it doesn't check package versions or the schema files. After changing those, start new files too.

## Output format (for scoring, #14)

`results/extraction/routing.csv`: one row per val and test document, with its true type (`format`), `predicted_type`, both detector scores and `route` (the type, or `REVIEW`).

`results/extraction/qwen3-vl-4b-instruct_<split>.jsonl`: one JSON object per line:

| Field | Meaning |
|---|---|
| `id` | document id (same as `data/documents.csv`) |
| `predicted_type` | the type whose schema the model got |
| `output` | the extracted JSON object, or `null` if the reply had no valid JSON or was removed |
| `raw` | the reply as text, or `null` if it was removed |
| `input_tokens`, `output_tokens` | prompt size (image + text) and reply size |
| `hit_token_limit` | `true` if the reply was cut off at 4,096 tokens |
| `seconds` | time to generate the reply |
| `error` | `null`, or why the reply was removed |

Notes for scoring:
- Match documents on `id`, never on row position. The dataset viewer on the Hugging Face website lists rows in a different order (its row 83 is document 173).
- Pick the documents to score by scope, not by `output`. The legal documents are the ones with `scope` = `in_scope` in `data/documents.csv`. 36 and 991 are out of scope: they got through step 1 by mistake.
- A legal document with `output: null` and `error: null` means the model gave no valid JSON. Count it as a failure. There are none in this run.
- A legal document missing from the file was sent to `REVIEW` in step 1 (see `routing.csv`). It counts as a miss in an end-to-end score. There are none in this run. If `routing.csv` gave it a type but it's still missing, the run isn't finished (a page ran out of GPU memory), so rerun step 2 before scoring.
- `output` comes from `parse_reply`, which also accepts JSON with text around it. In this run every reply is a bare JSON object, so a strict check on `raw` gives the same result.

The ground truth is `true_json_output` in the dataset. It follows each document's own schema (`json_schema`). For patents and proxy voting forms, a document's own schema has fewer fields than its type's schema, so the model fills fields that aren't in the ground truth. For the other 3 types the fields are the same. Filled-in values at fields that the document's own schema doesn't have:

| Type | Test | Val |
|---|---|---|
| Patent | 22 of 50 (44%) | 28 of 48 (58%) |
| Proxy voting | 85 of 713 (12%) | 109 of 1,104 (10%) |
| The other 3 types | 0 | 0 |

So we suggest scoring only the fields in each document's own schema, with the full-schema score next to it.

## Runs

**Test** (41 documents, about 38 minutes of generation time on the A6000):
- All 41 replies were valid JSON, and none hit the 4,096-token limit. We removed the reply for 36 afterwards (see below), so the file has 40 replies, and 36's line has `output` and `raw` set to `null`.
- Patents take about 10 seconds each. Proxy voting forms take 1.3 to 3.3 minutes, because their vote tables make long replies.
- 37 of the 39 legal replies pass their type's schema. We check dates too, like `make_validator` in `schemas/build_class_schemas.py`. Document 83 (proxy voting) puts `totalSharePositions` at the top level, but the schema only allows it inside each `votingBreakdown` item. Document 518 (lease) has the date `2025-10-0`.
- For 991 (the nutrition label sent to `PATENT`), the model returns all nulls.

**Val** (41 documents, all legal, about 42 minutes of generation time):
- All 41 replies are valid JSON, and none hit the token limit.
- Proxy voting forms take up to 4.6 minutes here.
- 37 of 41 pass their type's schema. The 4 that fail are proxy voting forms with the same `totalSharePositions` mistake.

The times are rough, because other jobs were running on the same server. Field-level scores come from #14.

## Things to know

- On test, 41 documents reach the model: all 39 legal ones, plus 2 out-of-scope documents that got through step 1 (36 and 991). 36 is a petition photo labeled `PHOTO_TABLE`. 991 is a nutrition label. For error analysis (#12) they are step 1 errors, not model errors.
- Step 1's `either` rule was picked after seeing test. On val, the confidence check, the distance check and `either` all make no mistakes, so val couldn't choose between them. On test, the confidence check alone lets 3 out-of-scope documents through, the distance check alone lets 6 through, and `either` lets 2 through. So "2 of 175" is a little optimistic.
- 36 is a photo of a real petition, and the model copied the names, addresses and birth dates of the people on it. The repo is public, so its reply is not saved: `output` and `raw` are `null`, and `error` says why. `REMOVED_IDS` in [`outputs.py`](outputs.py) does this on every run, so a rerun won't bring the data back. The other replies look made up (fake names and streets).
- The pipeline's classifier is the one inside the confidence detector (C = 0.001), not the one in `analysis/legal_classifier.py` and its confusion matrix (C = 10). Both are tuned on val, and both get all 41 val and 39 test legal documents right.
- The prompt and the model settings were not tuned on test: one model and one prompt, checked on 5 val documents before the runs.
- For #12: the 4 legal photos and the 8 rotated legal pages are all in train, so this run has no replies for them.
