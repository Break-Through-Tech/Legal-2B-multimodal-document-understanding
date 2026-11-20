"""Pipeline step 2: extract JSON from each routed document with a small free vision-language model (issue #10).

Every document that step 1 (extraction/route.py) gave a legal type goes to the model with
its page image and that type's schema (schemas/classes/<type>.json). Documents routed to
REVIEW are skipped: in the real pipeline a person would look at them.

Model: Qwen/Qwen3-VL-4B-Instruct (4B parameters, Apache-2.0, free on Hugging Face), pinned
to one revision. Greedy decoding (no random sampling), so a rerun on the same GPU and
software should give the same replies.

Each reply is added to the output file as soon as it is ready. If the run stops, running it
again skips the documents already done. A page that runs out of GPU memory is not saved, so
the next run tries it again. Pages of real people (extraction/outputs.py) are saved without
their reply.

Outputs (in results/extraction/):
    <model>_<split>.jsonl      one line per document:
        id, predicted_type, output (the JSON object, or null if the reply had none or was removed),
        raw (the reply as text), input_tokens, output_tokens, hit_token_limit, seconds,
        error (null, or why the reply was removed)
    <model>_<split>_run.json   model, revision, settings, a prompt example, GPU and package versions

Usage (from the repository root, on a GPU with about 12 GB free):
    python -m extraction.run_vlm --split val --limit 5    # quick look at a few replies
    python -m extraction.run_vlm --split test
"""

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForImageTextToText, AutoProcessor

from analysis.detection import REVIEW
from data.build_dataset import load_raw_dataset
from extraction.outputs import check_settings, done_ids, remove_personal_data
from extraction.prompt import build_prompt, load_schema, parse_reply, shrink_to
from extraction.route import OUT_DIR, load_routes

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"  # the Hugging Face commit we ran
MAX_PIXELS = 4096 * 32 * 32  # about 4.2 million pixels (about 4,096 image tokens): a bit under half the pages get scaled down, most only a little
MAX_NEW_TOKENS = 4096


def output_paths(split: str) -> tuple[Path, Path]:
    name = f"{MODEL_ID.split('/')[-1].lower()}_{split}"
    return OUT_DIR / f"{name}.jsonl", OUT_DIR / f"{name}_run.json"


def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, revision=REVISION, dtype=torch.bfloat16)
    return model.to("cuda").eval(), processor


@torch.inference_mode()
def extract(model, processor, image, prompt: str) -> dict:
    """Ask the model about one page. Returns its reply and a few counts."""
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt").to(model.device)
    n_input = inputs["input_ids"].shape[1]
    start = time.perf_counter()
    generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    seconds = time.perf_counter() - start
    new_tokens = generated[0, n_input:]
    return {
        "raw": processor.decode(new_tokens, skip_special_tokens=True),
        "input_tokens": n_input,
        "output_tokens": len(new_tokens),
        "hit_token_limit": len(new_tokens) >= MAX_NEW_TOKENS,
        "seconds": round(seconds, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract JSON from routed documents with a vision-language model.")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--limit", type=int, help="only the first N documents, for a quick check")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads for torch; the GPU does the real work")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    routes = load_routes()
    todo = routes[(routes["split"] == args.split) & (routes["route"] != REVIEW)].sort_values("id")
    if args.limit is not None:
        todo = todo.head(args.limit)

    out_path, run_path = output_paths(args.split)
    settings = {
        "model": MODEL_ID, "revision": REVISION, "decoding": "greedy",
        "max_pixels": MAX_PIXELS, "max_new_tokens": MAX_NEW_TOKENS,
        "prompt_example": build_prompt("PATENT", {"...": "the PATENT schema goes here"}),
    }
    check_settings(out_path, run_path, settings)
    done = done_ids(out_path)
    todo = todo[~todo["id"].isin(done)]
    print(f"{args.split}: {len(todo)} documents to do, {len(done)} already done", flush=True)
    if todo.empty:
        return

    model, processor = load_model()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not run_path.exists():
        run_path.write_text(json.dumps({
            **settings, "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "gpu": torch.cuda.get_device_name(0), "python": platform.python_version(),
            "torch": torch.__version__, "transformers": transformers.__version__,
        }, indent=2) + "\n")

    dataset = load_raw_dataset()
    with out_path.open("a") as f:
        for n, row in enumerate(todo.itertuples(), 1):
            record = dataset[int(row.id)]
            if record["id"] != row.id:
                raise SystemExit(f"Dataset row {row.id} has id {record['id']}: the row order must match the ids.")
            try:
                reply = extract(model, processor, shrink_to(record["image"], MAX_PIXELS),
                                build_prompt(row.route, load_schema(row.route)))
            except torch.cuda.OutOfMemoryError:
                # Not saved, so the next run tries this page again.
                torch.cuda.empty_cache()
                print(f"[{n}/{len(todo)}] id {row.id} {row.route}: out of GPU memory, not saved", flush=True)
                continue
            result = {"id": int(row.id), "predicted_type": row.route,
                      "output": parse_reply(reply["raw"]), **reply, "error": None}
            result = remove_personal_data(result)
            f.write(json.dumps(result) + "\n")
            f.flush()
            status = "JSON ok" if result["output"] is not None else (result["error"] or "no JSON")
            print(f"[{n}/{len(todo)}] id {row.id} {row.route}: {status}, "
                  f"{result['output_tokens']} tokens, {result['seconds']} s", flush=True)

    missing = len(todo) - len(done_ids(out_path) & set(todo["id"]))
    if missing:
        print(f"Not saved (out of GPU memory): {missing} of {len(todo)}. Run the same command again to retry them.")


if __name__ == "__main__":
    main()
