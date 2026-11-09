"""Turn document text into embeddings and cache them (task #3, part 2).

Two text sources:
- `ocr`: Tesseract output from features/ocr.py. This is the real OCR channel.
- `gt`:  the benchmark's ground-truth transcription. A clean-OCR upper bound: how well a
         text channel could do if OCR made no mistakes. Not available in production.

Two encoders for each source:
- `tfidf`: TF-IDF over word 1-2grams, reduced to 256 dimensions with truncated SVD.
           The classic baseline from the project overview. Fitted on train-split
           documents only, so val/test text never shapes the vocabulary.
- `bge`:   BAAI/bge-base-en-v1.5, a 110M-parameter English text embedder near the top of
           the MTEB leaderboard for its size. Its window is 512 tokens and legal documents
           are longer, so each document is cut into ~300-word chunks, every chunk is
           embedded, and the chunk vectors are averaged.

Output: cache/embeddings/{source}_{encoder}.npy, one L2-normalized row per document id.

Usage (from the repository root, after `python -m features.ocr`):
    python -m features.text_embeddings
"""

import argparse
from pathlib import Path

import numpy as np

from data.build_dataset import decode_markdown, load_documents, load_raw_dataset
from features.ocr import load_ocr_text

ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_DIR = ROOT / "cache" / "embeddings"
BGE_MODEL = "BAAI/bge-base-en-v1.5"
CHUNK_WORDS = 300
SVD_DIM = 256
SEED = 42


def load_texts(source: str) -> list[str]:
    if source == "gt":
        return [decode_markdown(raw) for raw in load_raw_dataset()["true_markdown_output"]]
    if source == "ocr":
        ocr = load_ocr_text()
        n = len(load_documents())
        missing = [i for i in range(n) if i not in ocr]
        if missing:
            raise SystemExit(f"{len(missing)} documents have no OCR text yet. Run: python -m features.ocr")
        return [ocr[i] for i in range(n)]
    raise ValueError(source)


def embed_tfidf(texts: list[str], train_mask: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline

    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000, sublinear_tf=True),
        TruncatedSVD(n_components=SVD_DIM, random_state=SEED),
    )
    model.fit([t for t, keep in zip(texts, train_mask) if keep])
    return model.transform(texts)


def chunk(text: str, size: int = CHUNK_WORDS) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size)] or [""]


def embed_bge(texts: list[str]) -> np.ndarray:
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(BGE_MODEL, device=device)
    chunks, owner = [], []
    for doc_index, text in enumerate(texts):
        for piece in chunk(text):
            chunks.append(piece)
            owner.append(doc_index)
    vectors = model.encode(chunks, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    owner = np.array(owner)
    return np.stack([vectors[owner == i].mean(axis=0) for i in range(len(texts))])


def l2_normalize(X: np.ndarray) -> np.ndarray:
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=["ocr", "gt"])
    parser.add_argument("--encoders", nargs="+", default=["tfidf", "bge"])
    parser.add_argument("--force", action="store_true", help="recompute even if cached")
    args = parser.parse_args()

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    train_mask = (load_documents()["split"] == "train").to_numpy()
    for source in args.sources:
        texts = load_texts(source)
        for encoder in args.encoders:
            path = EMBEDDINGS_DIR / f"{source}_{encoder}.npy"
            if path.exists() and not args.force:
                print(f"{path.name} cached, skipping")
                continue
            X = embed_tfidf(texts, train_mask) if encoder == "tfidf" else embed_bge(texts)
            np.save(path, l2_normalize(X).astype(np.float32))
            print(f"Wrote {path.name} {X.shape}")


if __name__ == "__main__":
    main()
