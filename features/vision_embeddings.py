"""Embed every document image with vision models and cache the vectors.

The model loaders are the ones from notebooks/Legal2B_Group_Notebook_embedding.ipynb
(branch data/preprocessing, issue #2), moved into a module so the clustering and
out-of-scope scripts can reuse the cached vectors next to the OCR channel.

| key            | checkpoint                         | one vector per page            |
|----------------|------------------------------------|--------------------------------|
| colqwen2       | vidore/colqwen2-v1.0-hf            | mean of ~750 patch vectors     |
| siglip         | google/siglip-so400m-patch14-384   | pooled image embedding         |
| dinov2         | facebook/dinov2-large              | CLS token, 448x448, no crop    |
| pixel_baseline | none                               | 32x32 grayscale thumbnail      |

Output: cache/embeddings/vision_{key}.npy, one L2-normalized row per document id.
Each model's file is written once; delete it to recompute.

Usage (from the repository root):
    python -m features.vision_embeddings                       # all four
    python -m features.vision_embeddings --models siglip dinov2
"""

import argparse
from pathlib import Path

import numpy as np

from data.build_dataset import load_raw_dataset

ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_DIR = ROOT / "cache" / "embeddings"
MAX_SIDE = 1536  # cap huge scans (up to ~7,000 px) before batching


def device():
    import torch

    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load_colqwen2(model_id="vidore/colqwen2-v1.0-hf"):
    import torch
    from transformers import ColQwen2ForRetrieval, ColQwen2Processor

    dev = device()
    dtype = torch.float32 if dev == "cpu" else torch.bfloat16
    processor = ColQwen2Processor.from_pretrained(model_id)
    model = ColQwen2ForRetrieval.from_pretrained(model_id, dtype=dtype).to(dev).eval()

    def embed(images):
        inputs = processor(images=images, return_tensors="pt").to(dev)
        token_vectors = model(**inputs).embeddings.float()
        mask = inputs["attention_mask"].unsqueeze(-1)
        return (token_vectors * mask).sum(dim=1) / mask.sum(dim=1)

    return embed


def load_siglip(model_id="google/siglip-so400m-patch14-384"):
    from transformers import AutoImageProcessor, AutoModel

    dev = device()
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(dev).eval()

    def embed(images):
        inputs = processor(images=images, return_tensors="pt").to(dev)
        out = model.get_image_features(**inputs)
        return getattr(out, "pooler_output", out)

    return embed


def load_dinov2(model_id="facebook/dinov2-large"):
    from transformers import AutoImageProcessor, AutoModel

    dev = device()
    processor = AutoImageProcessor.from_pretrained(
        model_id, do_center_crop=False, size={"height": 448, "width": 448}
    )
    model = AutoModel.from_pretrained(model_id).to(dev).eval()

    def embed(images):
        inputs = processor(images=images, return_tensors="pt").to(dev)
        return model(**inputs).pooler_output

    return embed


def load_pixel_baseline(size=32):
    import torch

    def embed(images):
        thumbs = [np.asarray(im.convert("L").resize((size, size)), dtype=np.float32) / 255 for im in images]
        return torch.from_numpy(np.stack(thumbs).reshape(len(images), -1))

    return embed


MODELS = {
    "pixel_baseline": load_pixel_baseline,
    "siglip": load_siglip,
    "dinov2": load_dinov2,
    "colqwen2": load_colqwen2,
}


def prepare(image):
    image = image.convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    return image


def embed_all(name: str, batch_size: int = 8) -> Path:
    import torch
    from tqdm import tqdm

    path = EMBEDDINGS_DIR / f"vision_{name}.npy"
    if path.exists():
        print(f"{path.name} cached, skipping")
        return path

    # Partial results are saved every batch, so an interrupted run resumes.
    partial = EMBEDDINGS_DIR / f"vision_{name}.partial.npy"
    done = list(np.load(partial)) if partial.exists() else []

    ds = load_raw_dataset()
    embed = MODELS[name]()
    for start in tqdm(range(len(done), len(ds), batch_size), desc=name, initial=len(done) // batch_size,
                      total=-(-len(ds) // batch_size)):
        images = [prepare(im) for im in ds[start : start + batch_size]["image"]]
        with torch.no_grad():
            done.extend(embed(images).float().cpu().numpy())
        np.save(partial, np.stack(done))

    X = np.stack(done)
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    np.save(path, X.astype(np.float32))
    partial.unlink()
    print(f"Wrote {path.name} {X.shape}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        embed_all(name, args.batch_size)


if __name__ == "__main__":
    main()
