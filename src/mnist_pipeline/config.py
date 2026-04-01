from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
MODEL_DISPLAY_NAMES = {
    "knn": "k-Nearest Neighbors",
    "logistic_regression": "Logistic Regression",
    "neural_network": "Neural Network",
}
PCA_BENCHMARK_COMPONENTS = (25, 50, 75, 100, 150, 200)
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
