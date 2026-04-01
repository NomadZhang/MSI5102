from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import PCA_BENCHMARK_COMPONENTS, PipelineConfig


def load_model_selection_summary(config: PipelineConfig) -> list[dict[str, Any]]:
    summary_path = config.results_dir / "model_selection_summary.csv"
    if not summary_path.exists():
        return []
    return json.loads(pd.read_csv(summary_path).to_json(orient="records"))


def load_pca_selection_summary(config: PipelineConfig) -> list[dict[str, Any]]:
    summary_path = config.results_dir / "pca_selection_summary.csv"
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
        "pca_benchmark": {
            "component_grid": list(PCA_BENCHMARK_COMPONENTS),
            "selection_summary": load_pca_selection_summary(config),
        },
        "top_model": json.loads(metrics_frame.iloc[0].to_json()),
    }
    with (config.results_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
