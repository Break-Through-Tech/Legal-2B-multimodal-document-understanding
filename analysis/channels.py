"""Load every cached embedding channel, so the analysis scripts compare them on equal terms."""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_DIR = ROOT / "cache" / "embeddings"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Display order and a short description for the results tables.
CHANNELS = {
    "vision_pixel_baseline": "Vision: 32x32 thumbnail (no model)",
    "vision_dinov2": "Vision: DINOv2-large",
    "vision_siglip": "Vision: SigLIP so400m",
    "vision_colqwen2": "Vision: ColQwen2 (mean-pooled)",
    "ocr_tfidf": "OCR text: Tesseract + TF-IDF/SVD",
    "ocr_bge": "OCR text: Tesseract + bge-base",
    "gt_tfidf": "Clean text (upper bound): TF-IDF/SVD",
    "gt_bge": "Clean text (upper bound): bge-base",
}


def channel_family(name: str) -> str:
    return {"vision": "vision", "ocr": "ocr_text", "gt": "clean_text"}[name.split("_")[0]]


def load_channels(names=None) -> dict[str, np.ndarray]:
    """{name: (1000, d) array} for every channel whose cache file exists, in CHANNELS order."""
    channels = {}
    for name in names or CHANNELS:
        path = EMBEDDINGS_DIR / f"{name}.npy"
        if path.exists():
            channels[name] = np.load(path)
    missing = [n for n in (names or CHANNELS) if n not in channels]
    if missing:
        print(f"Not cached yet, skipped: {', '.join(missing)}")
    return channels
