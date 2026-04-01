from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .config import PipelineConfig
from .data import stratified_sample_indices

LOGGER = logging.getLogger(__name__)


def save_sample_grid(X: np.ndarray, y: np.ndarray, output_path: Path, random_state: int) -> None:
    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(y), size=25, replace=False)

    fig, axes = plt.subplots(5, 5, figsize=(8, 8))
    for axis, index in zip(axes.ravel(), indices, strict=True):
        axis.imshow(X[index].reshape(28, 28), cmap="gray_r")
        axis.set_title(f"Label: {int(y[index])}", fontsize=9)
        axis.axis("off")

    fig.suptitle("MNIST Sample Digits", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_embedding_plot(
    embedding: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        cmap="tab10",
        s=10,
        alpha=0.75,
    )
    plt.colorbar(scatter, ticks=range(10), label="Digit label")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def generate_embedding_artifacts(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: PipelineConfig,
) -> dict[str, float]:
    embedding_metadata: dict[str, float] = {}

    pca_indices = stratified_sample_indices(
        y_train,
        sample_size=config.pca_sample_size,
        random_state=config.random_state,
    )
    X_pca = X_train[pca_indices]
    y_pca = y_train[pca_indices]

    LOGGER.info("Generating PCA projection with %s samples", len(pca_indices))
    pca = PCA(n_components=2, random_state=config.random_state)
    pca_embedding = pca.fit_transform(X_pca)
    embedding_metadata["pca_explained_variance_ratio"] = float(pca.explained_variance_ratio_.sum())
    save_embedding_plot(
        pca_embedding,
        y_pca,
        config.figures_dir / "digits_pca_2d.png",
        title="MNIST Digits Projected with PCA",
        x_label="Principal Component 1",
        y_label="Principal Component 2",
    )

    tsne_indices = stratified_sample_indices(
        y_train,
        sample_size=config.tsne_sample_size,
        random_state=config.random_state,
    )
    X_tsne = X_train[tsne_indices]
    y_tsne = y_train[tsne_indices]

    LOGGER.info("Reducing %s samples to 50 dimensions before t-SNE", len(tsne_indices))
    tsne_preprocessor = PCA(n_components=50, random_state=config.random_state)
    X_tsne_reduced = tsne_preprocessor.fit_transform(X_tsne)

    LOGGER.info("Generating t-SNE embedding")
    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=30,
        random_state=config.random_state,
    )
    tsne_embedding = tsne.fit_transform(X_tsne_reduced)
    save_embedding_plot(
        tsne_embedding,
        y_tsne,
        config.figures_dir / "digits_tsne_2d.png",
        title="MNIST Digits Projected with t-SNE",
        x_label="t-SNE Component 1",
        y_label="t-SNE Component 2",
    )

    return embedding_metadata
