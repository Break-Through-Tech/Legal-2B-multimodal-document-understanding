"""OCR every document with Tesseract and cache the text (task #3, part 1).

Writes cache/ocr/tesseract.jsonl, one line per document: {"id", "text", "seconds"}.
The file is appended to after every document, so an interrupted run (or a Colab
disconnect) resumes where it stopped. Delete the file to OCR everything again.

Preprocessing, applied the same way to every image:
- convert to RGB (345 images are RGBA, 18 grayscale, 1 palette), then to grayscale
- upscale images whose long edge is under 1,000 px by 2x (Tesseract misses small text)
- downscale images whose long edge is over 3,500 px (speed; text stays legible)
- detect page orientation with Tesseract's own OSD (orientation and script detection) and
  rotate upright when it is confident. The dataset's `rotation` metadata is NOT used,
  because a real intake queue would not have it. Without this step, word recall on the
  43 rotated documents was 0.27 vs. 0.79 on upright ones.

Usage (from the repository root):
    python -m features.ocr               # all 1,000 documents, in parallel
    python -m features.ocr --limit 20    # quick test
Tesseract must be installed. Set TESSERACT_CMD if it is not on your PATH.
"""

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OCR_DIR = ROOT / "cache" / "ocr"
OCR_JSONL = OCR_DIR / "tesseract.jsonl"
TESSERACT_CONFIG = "--oem 1 --psm 3"  # LSTM engine, automatic page segmentation
MIN_LONG_EDGE = 1000
MAX_LONG_EDGE = 3500
OSD_MIN_CONFIDENCE = 2.0  # Tesseract's orientation confidence; below this, trust the page as is


def find_tesseract() -> str:
    candidates = [
        os.environ.get("TESSERACT_CMD"),
        shutil.which("tesseract"),
        "/opt/homebrew/bin/tesseract",
        "/opt/anaconda3/envs/tesseract/bin/tesseract",
        "/usr/bin/tesseract",
    ]
    for path in candidates:
        if path and Path(path).exists():
            return path
    raise SystemExit("Tesseract not found. Install it (brew install tesseract / apt install tesseract-ocr) or set TESSERACT_CMD.")


def prepare_for_ocr(image):
    from PIL import Image

    image = image.convert("RGB").convert("L")
    long_edge = max(image.size)
    if long_edge < MIN_LONG_EDGE:
        image = image.resize((image.width * 2, image.height * 2), Image.LANCZOS)
    elif long_edge > MAX_LONG_EDGE:
        image.thumbnail((MAX_LONG_EDGE, MAX_LONG_EDGE), Image.LANCZOS)
    return image


def _ocr_one(args):
    doc_id, tesseract_cmd = args
    os.environ["OMP_THREAD_LIMIT"] = "1"  # one thread per worker; the pool gives parallelism
    import pytesseract

    from data.build_dataset import load_raw_dataset

    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    image = load_raw_dataset()[doc_id]["image"]
    start = time.perf_counter()
    page, rotated_by = auto_rotate(prepare_for_ocr(image))
    text = pytesseract.image_to_string(page, lang="eng", config=TESSERACT_CONFIG)
    return {"id": doc_id, "text": text, "rotated_by": rotated_by, "seconds": round(time.perf_counter() - start, 2)}


def auto_rotate(image, min_confidence: float = OSD_MIN_CONFIDENCE):
    """Rotate the page upright using Tesseract OSD. Returns (image, degrees rotated)."""
    import pytesseract

    try:
        osd = pytesseract.image_to_osd(image, config="--psm 0", output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractError:  # too little text to decide; leave the page as is
        return image, 0
    degrees = int(osd.get("rotate", 0))
    if degrees and float(osd.get("orientation_conf", 0)) >= min_confidence:
        # OSD's "rotate" is clockwise; PIL rotates counter-clockwise.
        return image.rotate(-degrees, expand=True), degrees
    return image, 0


def load_ocr_text(path: Path = OCR_JSONL) -> dict[int, str]:
    """Return {id: text} for every document OCR'd so far."""
    if not path.exists():
        return {}
    with path.open() as f:
        return {row["id"]: row["text"] for row in map(json.loads, f)}


def tesseract_version(cmd: str) -> str:
    import subprocess

    out = subprocess.run([cmd, "--version"], capture_output=True, text=True)
    return (out.stdout or out.stderr).splitlines()[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="only OCR the first N documents")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()

    from tqdm import tqdm

    from data.build_dataset import load_raw_dataset

    cmd = find_tesseract()
    print(f"Using {tesseract_version(cmd)} at {cmd}")
    n_docs = len(load_raw_dataset())
    todo = [i for i in range(n_docs if args.limit is None else args.limit) if i not in load_ocr_text()]
    print(f"{n_docs - len(todo)} already cached, {len(todo)} to OCR with {args.workers} workers")

    OCR_DIR.mkdir(parents=True, exist_ok=True)
    with OCR_JSONL.open("a") as out, ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(_ocr_one, (i, cmd)) for i in todo]
        for future in tqdm(as_completed(futures), total=len(futures), desc="OCR"):
            out.write(json.dumps(future.result()) + "\n")
            out.flush()

    print(f"{len(load_ocr_text())} documents cached in {OCR_JSONL}")


if __name__ == "__main__":
    main()
