from __future__ import annotations

import argparse
import json
import logging
import urllib.request
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
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
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

LOGGER = logging.getLogger(__name__)
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
MODEL_DISPLAY_NAMES = {
    "knn": "k-Nearest Neighbors",
    "logistic_regression": "Logistic Regression",
    "neural_network": "Neural Network",
}
KNN_SEARCH_GRID = [
    {"n_neighbors": n_neighbors, "weights": weights}
    for n_neighbors, weights in product((1, 3, 5, 7, 9, 11), ("uniform", "distance"))
]
LOGISTIC_SEARCH_GRID = [{"C": c_value} for c_value in (0.1, 0.3, 1.0, 3.0)]
NEURAL_NETWORK_SEARCH_GRID = [
    {"hidden_layer_sizes": (128,), "alpha": 1e-4, "learning_rate_init": 1e-3},
    {"hidden_layer_sizes": (256,), "alpha": 1e-4, "learning_rate_init": 1e-3},
    {"hidden_layer_sizes": (128, 64), "alpha": 1e-4, "learning_rate_init": 1e-3},
    {"hidden_layer_sizes": (256, 128), "alpha": 1e-4, "learning_rate_init": 1e-3},
    {"hidden_layer_sizes": (128, 64), "alpha": 5e-4, "learning_rate_init": 1e-3},
    {"hidden_layer_sizes": (128, 64), "alpha": 1e-4, "learning_rate_init": 5e-4},
]


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
    validation_fraction: float = 0.15
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
        "--validation-fraction",
        type=float,
        default=0.15,
        help="Fraction of the training set reserved for hyperparameter validation.",
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
        validation_fraction=args.validation_fraction,
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


def build_knn_model(params: dict[str, Any], random_state: int) -> KNeighborsClassifier:
    del random_state
    return KNeighborsClassifier(
        n_neighbors=int(params["n_neighbors"]),
        weights=str(params["weights"]),
        n_jobs=-1,
    )


def build_logistic_model(params: dict[str, Any], random_state: int) -> LogisticRegression:
    return LogisticRegression(
        C=float(params["C"]),
        max_iter=400,
        solver="lbfgs",
        random_state=random_state,
    )


def build_neural_network_model(params: dict[str, Any], random_state: int) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
        activation="relu",
        solver="adam",
        batch_size=256,
        learning_rate_init=float(params["learning_rate_init"]),
        alpha=float(params["alpha"]),
        max_iter=35,
        early_stopping=True,
        n_iter_no_change=5,
        random_state=random_state,
    )


def format_hidden_layers(hidden_layer_sizes: tuple[int, ...] | list[int]) -> str:
    return "-".join(str(units) for units in hidden_layer_sizes)


def build_parameter_label(model_key: str, params: dict[str, Any]) -> str:
    if model_key == "knn":
        return f"k={params['n_neighbors']}, weights={params['weights']}"
    if model_key == "logistic_regression":
        return f"C={params['C']}"
    if model_key == "neural_network":
        hidden_layers = format_hidden_layers(tuple(params["hidden_layer_sizes"]))
        return (
            f"layers={hidden_layers}, alpha={params['alpha']}, "
            f"lr={params['learning_rate_init']}"
        )
    raise KeyError(f"Unsupported model key: {model_key}")


def serialise_params(model_key: str, params: dict[str, Any]) -> dict[str, Any]:
    serialised: dict[str, Any] = {"parameter_label": build_parameter_label(model_key, params)}
    for key, value in params.items():
        if isinstance(value, tuple):
            serialised[key] = format_hidden_layers(value)
        else:
            serialised[key] = value
    return serialised


def compute_metrics(y_true: np.ndarray, predictions: np.ndarray, prefix: str = "") -> dict[str, float]:
    return {
        f"{prefix}accuracy": round(accuracy_score(y_true, predictions), 5),
        f"{prefix}precision_macro": round(
            precision_score(y_true, predictions, average="macro", zero_division=0),
            5,
        ),
        f"{prefix}recall_macro": round(
            recall_score(y_true, predictions, average="macro", zero_division=0),
            5,
        ),
        f"{prefix}f1_macro": round(
            f1_score(y_true, predictions, average="macro", zero_division=0),
            5,
        ),
    }


def select_best_configuration(search_frame: pd.DataFrame) -> pd.Series:
    sorted_frame = search_frame.sort_values(
        by=["validation_accuracy", "validation_f1_macro", "fit_seconds"],
        ascending=[False, False, True],
    )
    return sorted_frame.iloc[0]


def run_validation_search(
    model_key: str,
    model_builder: Callable[[dict[str, Any], int], Any],
    param_grid: list[dict[str, Any]],
    X_search_train: np.ndarray,
    X_validation: np.ndarray,
    y_search_train: np.ndarray,
    y_validation: np.ndarray,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    search_records: list[dict[str, Any]] = []

    for params in param_grid:
        LOGGER.info("Validating %s with params %s", model_key, params)
        model = model_builder(params, config.random_state)

        fit_start = perf_counter()
        model.fit(X_search_train, y_search_train)
        fit_seconds = perf_counter() - fit_start

        predict_start = perf_counter()
        predictions = model.predict(X_validation)
        predict_seconds = perf_counter() - predict_start

        record = {
            "model_key": model_key,
            "model": MODEL_DISPLAY_NAMES[model_key],
            **serialise_params(model_key, params),
            "fit_seconds": round(fit_seconds, 3),
            "predict_seconds": round(predict_seconds, 3),
            **compute_metrics(y_validation, predictions, prefix="validation_"),
        }
        search_records.append(record)

    search_frame = pd.DataFrame(search_records)
    search_frame = search_frame.sort_values(
        by=["validation_accuracy", "validation_f1_macro", "fit_seconds"],
        ascending=[False, False, True],
    )
    best_configuration = select_best_configuration(search_frame)
    return search_frame, param_grid[int(best_configuration.name)]


def annotate_best_point(ax: plt.Axes, x_value: float, y_value: float, label: str) -> None:
    ax.scatter([x_value], [y_value], s=160, facecolors="none", edgecolors="black", linewidths=2)
    ax.annotate(
        label,
        (x_value, y_value),
        textcoords="offset points",
        xytext=(10, 12),
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#333333", "alpha": 0.9},
    )


def save_knn_search_plot(search_frame: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=search_frame,
        x="n_neighbors",
        y="validation_accuracy",
        hue="weights",
        marker="o",
    )
    best_row = select_best_configuration(search_frame)
    annotate_best_point(
        plt.gca(),
        float(best_row["n_neighbors"]),
        float(best_row["validation_accuracy"]),
        f"Best: k={int(best_row['n_neighbors'])}, {best_row['weights']}",
    )
    plt.title("k-NN Hyperparameter Sweep on Validation Set")
    plt.xlabel("Number of neighbors (k)")
    plt.ylabel("Validation accuracy")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_logistic_search_plot(search_frame: pd.DataFrame, output_path: Path) -> None:
    plot_frame = search_frame.melt(
        id_vars=["C"],
        value_vars=["validation_accuracy", "validation_f1_macro"],
        var_name="metric",
        value_name="score",
    )
    plot_frame["metric"] = plot_frame["metric"].str.replace("validation_", "", regex=False)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=plot_frame, x="C", y="score", hue="metric", marker="o")
    plt.xscale("log")
    best_row = select_best_configuration(search_frame)
    annotate_best_point(
        plt.gca(),
        float(best_row["C"]),
        float(best_row["validation_accuracy"]),
        f"Best: C={best_row['C']}",
    )
    plt.title("Logistic Regression Regularization Sweep")
    plt.xlabel("Inverse regularization strength (C)")
    plt.ylabel("Validation score")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_neural_network_search_plot(search_frame: pd.DataFrame, output_path: Path) -> None:
    plot_frame = search_frame.sort_values("validation_accuracy", ascending=False).copy()
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=plot_frame,
        x="parameter_label",
        y="validation_accuracy",
        hue="parameter_label",
        palette="viridis",
        legend=False,
    )
    plt.xticks(rotation=25, ha="right")
    best_row = select_best_configuration(search_frame)
    plt.axhline(float(best_row["validation_accuracy"]), linestyle="--", color="black", linewidth=1)
    plt.title("Neural Network Architecture and Training Sweep")
    plt.xlabel("Configuration")
    plt.ylabel("Validation accuracy")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_hyperparameter_search_plot(
    model_key: str,
    search_frame: pd.DataFrame,
    config: PipelineConfig,
) -> None:
    output_path = config.figures_dir / f"{model_key}_hyperparameter_search.png"
    if model_key == "knn":
        save_knn_search_plot(search_frame, output_path)
        return
    if model_key == "logistic_regression":
        save_logistic_search_plot(search_frame, output_path)
        return
    if model_key == "neural_network":
        save_neural_network_search_plot(search_frame, output_path)
        return
    raise KeyError(f"Unsupported model key: {model_key}")


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


def build_top_confusions_frame(
    y_true: np.ndarray,
    predictions: np.ndarray,
) -> pd.DataFrame:
    mistakes = pd.DataFrame(
        {
            "true_label": y_true,
            "predicted_label": predictions,
        }
    )
    mistakes = mistakes[mistakes["true_label"] != mistakes["predicted_label"]]
    if mistakes.empty:
        return pd.DataFrame(columns=["true_label", "predicted_label", "count", "pair_label"])

    summary = (
        mistakes.groupby(["true_label", "predicted_label"])
        .size()
        .reset_index(name="count")
        .sort_values(by="count", ascending=False)
    )
    summary["pair_label"] = summary.apply(
        lambda row: f"{int(row['true_label'])} → {int(row['predicted_label'])}",
        axis=1,
    )
    return summary


def save_confusing_samples_gallery(
    X_test: np.ndarray,
    y_true: np.ndarray,
    predictions: np.ndarray,
    model_key: str,
    config: PipelineConfig,
    top_n_pairs: int = 3,
    samples_per_pair: int = 4,
) -> None:
    confusion_summary = build_top_confusions_frame(y_true, predictions)
    confusion_summary.to_csv(config.results_dir / f"{model_key}_top_confusions.csv", index=False)

    if confusion_summary.empty:
        return

    top_pairs = confusion_summary.head(top_n_pairs).copy()
    rows = len(top_pairs)
    fig, axes = plt.subplots(
        rows,
        samples_per_pair,
        figsize=(samples_per_pair * 2.3, rows * 2.6),
    )
    axes_array = np.atleast_2d(axes)

    for row_index, pair in enumerate(top_pairs.itertuples(index=False), start=0):
        pair_mask = (y_true == pair.true_label) & (predictions == pair.predicted_label)
        pair_indices = np.flatnonzero(pair_mask)[:samples_per_pair]

        for column_index in range(samples_per_pair):
            axis = axes_array[row_index, column_index]
            axis.axis("off")

            if column_index < len(pair_indices):
                sample_index = int(pair_indices[column_index])
                axis.imshow(X_test[sample_index].reshape(28, 28), cmap="gray_r")
                axis.set_title(
                    f"true {int(pair.true_label)}\npred {int(pair.predicted_label)}",
                    fontsize=9,
                )

        left_axis = axes_array[row_index, 0]
        left_axis.text(
            -0.42,
            0.5,
            f"{pair.pair_label}\ncount={int(pair.count)}",
            transform=left_axis.transAxes,
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    fig.suptitle(
        f"{MODEL_DISPLAY_NAMES[model_key]}: Most Frequent Misclassified Samples",
        fontsize=15,
    )
    fig.tight_layout(rect=(0.08, 0.03, 1, 0.95))
    fig.savefig(
        config.figures_dir / f"{model_key}_confusing_samples.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_split_protocol_artifacts(
    y_train: np.ndarray,
    y_search_train: np.ndarray,
    y_validation: np.ndarray,
    y_test: np.ndarray,
    config: PipelineConfig,
) -> None:
    split_frame = pd.DataFrame(
        [
            {
                "split": "Official train set",
                "samples": int(len(y_train)),
                "purpose": "Source pool for tuning and final retraining",
            },
            {
                "split": "Selection train split",
                "samples": int(len(y_search_train)),
                "purpose": "Fit candidate hyperparameter settings",
            },
            {
                "split": "Validation split",
                "samples": int(len(y_validation)),
                "purpose": "Pick the best hyperparameters",
            },
            {
                "split": "Official test set",
                "samples": int(len(y_test)),
                "purpose": "Final unbiased evaluation only",
            },
        ]
    )
    split_frame["fraction_of_full_dataset"] = (
        split_frame["samples"] / float(len(y_train) + len(y_test))
    ).round(4)
    split_frame.to_csv(config.results_dir / "data_split_summary.csv", index=False)

    plt.figure(figsize=(11, 6))
    chart_frame = split_frame[split_frame["split"] != "Official train set"].copy()
    sns.barplot(data=chart_frame, x="samples", y="split", hue="split", palette="Blues", legend=False)
    for index, row in chart_frame.iterrows():
        plt.text(
            float(row["samples"]) + 500,
            chart_frame.index.get_loc(index),
            f"{int(row['samples']):,}  |  {row['purpose']}",
            va="center",
            fontsize=10,
        )
    plt.title("Dataset Split and Validation Protocol")
    plt.xlabel("Number of images")
    plt.ylabel("")
    plt.xlim(0, chart_frame["samples"].max() * 1.32)
    plt.figtext(
        0.5,
        0.01,
        "Hold-out validation was used for hyperparameter selection. After choosing the best setting, each model was retrained on all 60,000 official training images before the final 10,000-image test evaluation.",
        ha="center",
        fontsize=10,
    )
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig(config.figures_dir / "data_split_protocol.png", dpi=220, bbox_inches="tight")
    plt.close()


def save_model_comparison_dashboard(metrics_frame: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    metric_plot_frame = metrics_frame.melt(
        id_vars=["model"],
        value_vars=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
        var_name="metric",
        value_name="score",
    )
    sns.barplot(data=metric_plot_frame, x="model", y="score", hue="metric", ax=axes[0])
    axes[0].set_title("Final Test Metrics by Model")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Score")
    axes[0].tick_params(axis="x", rotation=15)

    runtime_plot_frame = metrics_frame.melt(
        id_vars=["model"],
        value_vars=["train_seconds", "predict_seconds"],
        var_name="runtime",
        value_name="seconds",
    )
    sns.barplot(data=runtime_plot_frame, x="model", y="seconds", hue="runtime", ax=axes[1])
    axes[1].set_title("Training and Prediction Time")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Seconds")
    axes[1].tick_params(axis="x", rotation=15)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_neural_network_architecture_diagram(
    hidden_layer_sizes: tuple[int, ...],
    output_path: Path,
) -> None:
    layer_specs = [("Input", 784, "28x28 pixels")]
    for index, units in enumerate(hidden_layer_sizes, start=1):
        layer_specs.append((f"Hidden {index}", units, "Dense + ReLU"))
    layer_specs.append(("Output", 10, "Softmax classes"))

    x_positions = np.linspace(0.12, 0.88, len(layer_specs))
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for x_position, (title, units, subtitle) in zip(x_positions, layer_specs, strict=True):
        box = FancyBboxPatch(
            (x_position - 0.09, 0.38),
            0.18,
            0.24,
            boxstyle="round,pad=0.03",
            linewidth=1.5,
            edgecolor="#224466",
            facecolor="#D8E8F7",
        )
        ax.add_patch(box)
        ax.text(
            x_position,
            0.50,
            f"{title}\n{units} units\n{subtitle}",
            ha="center",
            va="center",
            fontsize=11,
        )

    for left, right in zip(x_positions[:-1], x_positions[1:], strict=True):
        arrow = FancyArrowPatch(
            (left + 0.09, 0.50),
            (right - 0.09, 0.50),
            arrowstyle="->",
            mutation_scale=16,
            linewidth=1.8,
            color="#224466",
        )
        ax.add_patch(arrow)

    ax.text(
        0.5,
        0.18,
        "Training settings: Adam optimizer, batch size 256, early stopping, cross-entropy loss",
        ha="center",
        va="center",
        fontsize=11,
    )
    ax.set_title("Selected Neural Network Architecture", fontsize=16, pad=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_neural_network_training_curve(model: MLPClassifier, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    axes[0].plot(np.arange(1, len(model.loss_curve_) + 1), model.loss_curve_, marker="o")
    axes[0].set_title("Neural Network Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    validation_scores = getattr(model, "validation_scores_", None)
    if validation_scores:
        axes[1].plot(np.arange(1, len(validation_scores) + 1), validation_scores, marker="o")
        axes[1].set_title("Neural Network Validation Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Validation accuracy")
    else:
        axes[1].text(0.5, 0.5, "Validation curve not available", ha="center", va="center")
        axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def train_and_evaluate_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    config: PipelineConfig,
) -> pd.DataFrame:
    X_search_train, X_validation, y_search_train, y_validation = train_test_split(
        X_train,
        y_train,
        test_size=config.validation_fraction,
        stratify=y_train,
        random_state=config.random_state,
    )
    save_split_protocol_artifacts(
        y_train=y_train,
        y_search_train=y_search_train,
        y_validation=y_validation,
        y_test=y_test,
        config=config,
    )

    model_builders = {
        "knn": build_knn_model,
        "logistic_regression": build_logistic_model,
        "neural_network": build_neural_network_model,
    }
    search_grids = {
        "knn": KNN_SEARCH_GRID,
        "logistic_regression": LOGISTIC_SEARCH_GRID,
        "neural_network": NEURAL_NETWORK_SEARCH_GRID,
    }

    selection_records: list[dict[str, Any]] = []
    metrics_records: list[dict[str, Any]] = []
    predictions_frame = pd.DataFrame({"index": np.arange(len(y_test)), "true_label": y_test})
    trained_models: dict[str, Any] = {}

    for model_key, model_builder in model_builders.items():
        search_frame, best_params = run_validation_search(
            model_key=model_key,
            model_builder=model_builder,
            param_grid=search_grids[model_key],
            X_search_train=X_search_train,
            X_validation=X_validation,
            y_search_train=y_search_train,
            y_validation=y_validation,
            config=config,
        )

        search_frame.to_csv(config.results_dir / f"{model_key}_hyperparameter_search.csv", index=False)
        save_hyperparameter_search_plot(model_key, search_frame, config)

        best_search_row = select_best_configuration(search_frame)
        selection_records.append(
            {
                "model_key": model_key,
                "model": MODEL_DISPLAY_NAMES[model_key],
                "selected_configuration": best_search_row["parameter_label"],
                "validation_accuracy": best_search_row["validation_accuracy"],
                "validation_f1_macro": best_search_row["validation_f1_macro"],
            }
        )

        LOGGER.info("Training final %s model with params %s", model_key, best_params)
        model = model_builder(best_params, config.random_state)
        fit_start = perf_counter()
        model.fit(X_train, y_train)
        train_seconds = perf_counter() - fit_start

        predict_start = perf_counter()
        predictions = model.predict(X_test)
        predict_seconds = perf_counter() - predict_start

        trained_models[model_key] = model
        predictions_frame[model_key] = predictions

        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        pd.DataFrame(report).transpose().to_csv(
            config.results_dir / f"{model_key}_classification_report.csv"
        )

        matrix = confusion_matrix(y_test, predictions)
        save_confusion_matrix_plot(
            matrix=matrix,
            model_name=MODEL_DISPLAY_NAMES[model_key],
            output_path=config.figures_dir / f"{model_key}_confusion_matrix.png",
        )
        save_confusing_samples_gallery(
            X_test=X_test,
            y_true=y_test,
            predictions=predictions,
            model_key=model_key,
            config=config,
        )

        metrics_records.append(
            {
                "model_key": model_key,
                "model": MODEL_DISPLAY_NAMES[model_key],
                "selected_configuration": build_parameter_label(model_key, best_params),
                "train_seconds": round(train_seconds, 3),
                "predict_seconds": round(predict_seconds, 3),
                **compute_metrics(y_test, predictions),
            }
        )

    predictions_frame.to_csv(config.results_dir / "test_predictions.csv", index=False)

    selection_frame = pd.DataFrame(selection_records).sort_values(
        by=["validation_accuracy", "validation_f1_macro"],
        ascending=[False, False],
    )
    selection_frame.to_csv(config.results_dir / "model_selection_summary.csv", index=False)

    metrics_frame = pd.DataFrame(metrics_records).sort_values(by="accuracy", ascending=False)
    metrics_frame.to_csv(config.results_dir / "model_metrics.csv", index=False)
    save_model_comparison_dashboard(
        metrics_frame,
        config.figures_dir / "model_comparison_dashboard.png",
    )

    neural_network_model = trained_models["neural_network"]
    hidden_layers = tuple(neural_network_model.hidden_layer_sizes)
    save_neural_network_architecture_diagram(
        hidden_layer_sizes=hidden_layers,
        output_path=config.figures_dir / "neural_network_architecture.png",
    )
    save_neural_network_training_curve(
        model=neural_network_model,
        output_path=config.figures_dir / "neural_network_training_curve.png",
    )

    return metrics_frame


def load_model_selection_summary(config: PipelineConfig) -> list[dict[str, Any]]:
    summary_path = config.results_dir / "model_selection_summary.csv"
    if not summary_path.exists():
        return []
    return json.loads(pd.read_csv(summary_path).to_json(orient="records"))


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
        "validation_strategy": {
            "validation_fraction": config.validation_fraction,
            "selection_summary": load_model_selection_summary(config),
        },
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
