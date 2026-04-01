from __future__ import annotations

import logging

from .config import build_config, parse_args
from .data import load_processed_dataset
from .embeddings import generate_embedding_artifacts, save_sample_grid
from .reporting import save_run_manifest
from .training import train_and_evaluate_models

LOGGER = logging.getLogger(__name__)


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
