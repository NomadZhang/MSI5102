from __future__ import annotations

from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from .config import MODEL_DISPLAY_NAMES

ModelBuilder = Callable[[dict[str, Any], int], Any]


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


def build_representation_label(n_components: int | None) -> str:
    if n_components is None:
        return "Raw (784)"
    return f"PCA ({n_components})"


def build_representation_key(n_components: int | None) -> str:
    if n_components is None:
        return "raw"
    return f"pca_{n_components}"


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


def select_best_pca_configuration(benchmark_frame: pd.DataFrame) -> pd.Series:
    pca_only = benchmark_frame[benchmark_frame["is_pca"]].copy()
    sorted_frame = pca_only.sort_values(
        by=["validation_accuracy", "validation_f1_macro", "n_components"],
        ascending=[False, False, True],
    )
    return sorted_frame.iloc[0]


def run_validation_search(
    model_key: str,
    model_builder: ModelBuilder,
    param_grid: list[dict[str, Any]],
    X_search_train: np.ndarray,
    X_validation: np.ndarray,
    y_search_train: np.ndarray,
    y_validation: np.ndarray,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    search_records: list[dict[str, Any]] = []

    for params in param_grid:
        model = model_builder(params, random_state)

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

    search_frame = pd.DataFrame(search_records).sort_values(
        by=["validation_accuracy", "validation_f1_macro", "fit_seconds"],
        ascending=[False, False, True],
    )
    best_configuration = select_best_configuration(search_frame)
    return search_frame, param_grid[int(best_configuration.name)]
