from __future__ import annotations

import argparse
import json
import logging
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

LOGGER = logging.getLogger(__name__)
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


@dataclass(frozen=True)
class PipelineConfig:
    project_root: Path
    raw_data_dir: Path
    processed_data_dir: Path
    raw_dataset_path: Path
    processed_dataset_path: Path
    artifacts_dir: Path
    figures_dir: Path
    results_dir: Path
    pca_sample_size: int = 5_000
    tsne_sample_size: int = 3_000
    random_state: int = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MNIST ETL and modeling pipeline")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root where data and artifacts will be stored.",
    )
    parser.add_argument(
        "--pca-sample-size",
        type=int,
        default=5_000,
        help="Number of training samples to use for the PCA plot.",
    )
    parser.add_argument(
        "--tsne-sample-size",
        type=int,
        default=3_000,
        help="Number of training samples to use for the t-SNE plot.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Redownload and regenerate cached datasets.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for sampling and model training.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    project_root = args.project_root.resolve()
    raw_data_dir = project_root / "data" / "raw"
    processed_data_dir = project_root / "data" / "processed"
    artifacts_dir = project_root / "artifacts"
    figures_dir = artifacts_dir / "figures"
    results_dir = artifacts_dir / "results"

    for directory in (raw_data_dir, processed_data_dir, figures_dir, results_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return PipelineConfig(
        project_root=project_root,
        raw_data_dir=raw_data_dir,
        processed_data_dir=processed_data_dir,
        raw_dataset_path=raw_data_dir / "mnist.npz",
        processed_dataset_path=processed_data_dir / "mnist_flattened.npz",
        artifacts_dir=artifacts_dir,
        figures_dir=figures_dir,
        results_dir=results_dir,
        pca_sample_size=args.pca_sample_size,
        tsne_sample_size=args.tsne_sample_size,
        random_state=args.random_state,
    )


def download_raw_dataset(config: PipelineConfig, force_refresh: bool) -> Path:
    if config.raw_dataset_path.exists() and not force_refresh:
        LOGGER.info("Using cached raw dataset at %s", config.raw_dataset_path)
        return config.raw_dataset_path

    LOGGER.info("Downloading MNIST dataset to %s", config.raw_dataset_path)
    urllib.request.urlretrieve(MNIST_URL, config.raw_dataset_path)
    return config.raw_dataset_path


def load_processed_dataset(config: PipelineConfig, force_refresh: bool) -> tuple[np.ndarray, ...]:
    if config.processed_dataset_path.exists() and not force_refresh:
        LOGGER.info("Using cached processed dataset at %s", config.processed_dataset_path)
        with np.load(config.processed_dataset_path) as dataset:
            return dataset["X_train"], dataset["X_test"], dataset["y_train"], dataset["y_test"]

    raw_dataset_path = download_raw_dataset(config, force_refresh=force_refresh)

    LOGGER.info("Transforming raw dataset into flat float32 arrays")
    with np.load(raw_dataset_path) as dataset:
        x_train = dataset["x_train"].astype(np.float32) / 255.0
        x_test = dataset["x_test"].astype(np.float32) / 255.0
        y_train = dataset["y_train"].astype(np.uint8)
        y_test = dataset["y_test"].astype(np.uint8)

    X_train = x_train.reshape((x_train.shape[0], -1))
    X_test = x_test.reshape((x_test.shape[0], -1))

    np.savez_compressed(
        config.processed_dataset_path,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    return X_train, X_test, y_train, y_test


def stratified_sample_indices(y: np.ndarray, sample_size: int, random_state: int) -> np.ndarray:
    if sample_size >= len(y):
        return np.arange(len(y))

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=sample_size,
        random_state=random_state,
    )
    sample_indices, _ = next(splitter.split(np.zeros((len(y), 1)), y))
    return sample_indices


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


def save_confusion_matrix_plot(
    matrix: np.ndarray,
    model_name: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def train_and_evaluate_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    config: PipelineConfig,
) -> pd.DataFrame:
    models = {
        "knn": KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1),
        "logistic_regression": LogisticRegression(
            max_iter=300,
            solver="lbfgs",
            random_state=config.random_state,
        ),
        "neural_network": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            batch_size=256,
            learning_rate_init=0.001,
            max_iter=40,
            early_stopping=True,
            n_iter_no_change=5,
            random_state=config.random_state,
        ),
    }

    metrics_records: list[dict[str, float | str]] = []
    predictions_frame = pd.DataFrame({"index": np.arange(len(y_test)), "true_label": y_test})

    for model_name, model in models.items():
        LOGGER.info("Training %s", model_name)
        train_start = perf_counter()
        model.fit(X_train, y_train)
        train_seconds = perf_counter() - train_start

        predict_start = perf_counter()
        predictions = model.predict(X_test)
        predict_seconds = perf_counter() - predict_start

        predictions_frame[model_name] = predictions

        metrics_records.append(
            {
                "model": model_name,
                "train_seconds": round(train_seconds, 3),
                "predict_seconds": round(predict_seconds, 3),
                "accuracy": round(accuracy_score(y_test, predictions), 5),
                "precision_macro": round(
                    precision_score(y_test, predictions, average="macro", zero_division=0),
                    5,
                ),
                "recall_macro": round(
                    recall_score(y_test, predictions, average="macro", zero_division=0),
                    5,
                ),
                "f1_macro": round(
                    f1_score(y_test, predictions, average="macro", zero_division=0),
                    5,
                ),
            }
        )

        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        report_frame = pd.DataFrame(report).transpose()
        report_frame.to_csv(config.results_dir / f"{model_name}_classification_report.csv")

        matrix = confusion_matrix(y_test, predictions)
        save_confusion_matrix_plot(
            matrix=matrix,
            model_name=model_name.replace("_", " ").title(),
            output_path=config.figures_dir / f"{model_name}_confusion_matrix.png",
        )

    predictions_frame.to_csv(config.results_dir / "test_predictions.csv", index=False)
    metrics_frame = pd.DataFrame(metrics_records).sort_values(by="accuracy", ascending=False)
    metrics_frame.to_csv(config.results_dir / "model_metrics.csv", index=False)
    return metrics_frame


def save_run_manifest(
    config: PipelineConfig,
    metrics_frame: pd.DataFrame,
    embedding_metadata: dict[str, float],
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> None:
    manifest = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "dataset": {
            "train_samples": int(len(y_train)),
            "test_samples": int(len(y_test)),
            "num_features": 784,
            "num_classes": 10,
        },
        "embedding": embedding_metadata,
        "top_model": json.loads(metrics_frame.iloc[0].to_json()),
    }
    with (config.results_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def main() -> None:
    configure_logging()
    args = parse_args()
    config = build_config(args)

    LOGGER.info("Starting MNIST pipeline in %s", config.project_root)
    X_train, X_test, y_train, y_test = load_processed_dataset(
        config=config,
        force_refresh=args.force_refresh,
    )

    save_sample_grid(
        X_train,
        y_train,
        config.figures_dir / "sample_digits.png",
        random_state=config.random_state,
    )
    embedding_metadata = generate_embedding_artifacts(X_train, y_train, config)
    metrics_frame = train_and_evaluate_models(X_train, X_test, y_train, y_test, config)
    save_run_manifest(config, metrics_frame, embedding_metadata, y_train, y_test)

    LOGGER.info("Pipeline complete. Metrics saved to %s", config.results_dir / "model_metrics.csv")
    print(metrics_frame.to_string(index=False))
