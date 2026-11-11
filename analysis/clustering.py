"""Compare clustering algorithms across every embedding channel (task #5).

The question has two parts, and this script keeps them apart:
1. **Embedding quality:** does the channel put same-type documents together? Measured by
   the silhouette of the *true* labels, which needs no clustering algorithm at all.
2. **Algorithm preference:** given a channel, which algorithm recovers the types best?
   Measured by ARI and NMI between each algorithm's clusters and the true labels, plus the
   silhouette and Davies-Bouldin of the clusters it found.

Two populations:
- `legal`: the 264 in-scope documents, 5 types. The hard, look-alike case.
- `head`:  the 925 head-class documents, 18 types.
Clustering is unsupervised, so it uses every document in the population (no split).

Algorithms: k-means, agglomerative (Ward), agglomerative (average linkage, cosine),
Gaussian mixture (on 50 PCA components), spectral (k-NN graph), HDBSCAN (picks its own
number of clusters), and a random baseline averaged over 20 seeds. Every algorithm except
HDBSCAN is told the true number of types.

Outputs:
    results/clustering_comparison.csv
    results/figures/clustering_ari_{population}.png
    results/figures/tsne_legal.png

Usage (from the repository root):
    python -m analysis.clustering
"""

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture

from analysis.channels import CHANNELS, FIGURES_DIR, RESULTS_DIR, channel_family, load_channels
from data.build_dataset import load_documents

SEED = 42
RANDOM_BASELINE_SEEDS = 20


def algorithms(k: int):
    return {
        "kmeans": lambda X: KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(X),
        "agglomerative_ward": lambda X: AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X),
        "agglomerative_cosine": lambda X: AgglomerativeClustering(
            n_clusters=k, linkage="average", metric="cosine"
        ).fit_predict(X),
        "gaussian_mixture": lambda X: GaussianMixture(
            n_components=k, covariance_type="diag", n_init=3, random_state=SEED
        ).fit_predict(PCA(n_components=min(50, X.shape[1]), random_state=SEED).fit_transform(X)),
        "spectral": lambda X: SpectralClustering(
            n_clusters=k, affinity="nearest_neighbors", n_neighbors=10, random_state=SEED
        ).fit_predict(X),
        "hdbscan": lambda X: HDBSCAN(min_cluster_size=10, metric="cosine").fit_predict(X),
    }


def cluster_quality(X: np.ndarray, clusters: np.ndarray) -> dict:
    n_clusters = len(set(clusters) - {-1})
    out = {"n_clusters_found": n_clusters, "noise_share": float(np.mean(clusters == -1))}
    if 2 <= len(set(clusters)) < len(clusters):
        out["silhouette"] = silhouette_score(X, clusters, metric="cosine")
        out["davies_bouldin"] = davies_bouldin_score(X, clusters)
    return out


def evaluate_population(X: np.ndarray, y: np.ndarray) -> list[dict]:
    k = len(np.unique(y))
    rows = []
    for name, fit in algorithms(k).items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clusters = fit(X)
        rows.append(
            {
                "algorithm": name,
                "ARI": adjusted_rand_score(y, clusters),
                "NMI": normalized_mutual_info_score(y, clusters),
                **cluster_quality(X, clusters),
            }
        )

    rng = np.random.default_rng(SEED)
    random_runs = [rng.integers(0, k, size=len(y)) for _ in range(RANDOM_BASELINE_SEEDS)]
    rows.append(
        {
            "algorithm": "random_baseline",
            "ARI": np.mean([adjusted_rand_score(y, c) for c in random_runs]),
            "NMI": np.mean([normalized_mutual_info_score(y, c) for c in random_runs]),
            "n_clusters_found": k,
            "noise_share": 0.0,
            "silhouette": np.mean([silhouette_score(X, c, metric="cosine") for c in random_runs]),
            "davies_bouldin": np.mean([davies_bouldin_score(X, c) for c in random_runs]),
        }
    )
    label_silhouette = silhouette_score(X, y, metric="cosine")
    for row in rows:
        row["true_label_silhouette"] = label_silhouette
    return rows


def plot_ari(results: pd.DataFrame, population: str) -> None:
    table = results[results["population"] == population].pivot(index="channel", columns="algorithm", values="ARI")
    table = table.loc[[c for c in CHANNELS if c in table.index]]
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(table) + 1.8))
    im = ax.imshow(table.to_numpy(), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(table.columns)), table.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(table.index)), table.index)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = table.iat[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value < 0.6 else "black", fontsize=8)
    ax.set_title(f"Adjusted Rand Index vs. true types — {population} documents")
    fig.colorbar(im, ax=ax, label="ARI (1 = perfect, 0 = random)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"clustering_ari_{population}.png", dpi=150)
    plt.close(fig)


def plot_tsne(channels: dict, docs: pd.DataFrame) -> None:
    idx = docs.index[docs["scope"] == "in_scope"].to_numpy()
    y = docs.loc[idx, "format"].to_numpy()
    types = sorted(set(y))
    n = len(channels)
    cols = min(4, n)
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4 * rows), squeeze=False)
    for ax, (name, X) in zip(axes.flat, channels.items()):
        points = TSNE(metric="cosine", random_state=SEED, init="pca").fit_transform(X[idx])
        for t in types:
            m = y == t
            ax.scatter(points[m, 0], points[m, 1], s=8, label=t)
        ax.set_title(name, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(types), fontsize=8)
    fig.suptitle("t-SNE of the 264 in-scope legal documents, colored by true type (intuition only)")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(FIGURES_DIR / "tsne_legal.png", dpi=150)
    plt.close(fig)


def main() -> None:
    docs = load_documents()
    channels = load_channels()
    populations = {
        "legal": docs.index[docs["scope"] == "in_scope"].to_numpy(),
        "head": docs.index[docs["is_head_class"]].to_numpy(),
    }

    rows = []
    for channel, X in channels.items():
        for population, idx in populations.items():
            print(f"{channel:<24} {population:<6} ({len(idx)} docs)")
            for row in evaluate_population(X[idx], docs.loc[idx, "format"].to_numpy()):
                rows.append({"channel": channel, "family": channel_family(channel), "population": population, **row})

    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    results = pd.DataFrame(rows)
    results.round(4).to_csv(RESULTS_DIR / "clustering_comparison.csv", index=False)
    for population in populations:
        plot_ari(results, population)
    plot_tsne(channels, docs)

    print("\nARI by channel and algorithm")
    for population in populations:
        part = results[results["population"] == population]
        print(f"\n[{population}]")
        print(part.pivot(index="channel", columns="algorithm", values="ARI").round(3).to_string())
    print("\nTrue-label silhouette (embedding quality, no algorithm involved)")
    print(results.pivot_table(index="channel", columns="population", values="true_label_silhouette").round(3).to_string())


if __name__ == "__main__":
    main()
