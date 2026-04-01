from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from .config import (
    KNN_SEARCH_GRID,
    LOGISTIC_SEARCH_GRID,
    MODEL_DISPLAY_NAMES,
    NEURAL_NETWORK_SEARCH_GRID,
    PCA_BENCHMARK_COMPONENTS,
    PipelineConfig,
)
from .charts import (
    save_hyperparameter_search_plot,
    save_model_comparison_dashboard,
    save_pca_component_benchmark_plot,
    save_raw_vs_best_pca_dashboard,
)
from .diagnostics import (
    save_confusing_samples_gallery,
    save_confusion_matrix_plot,
    save_neural_network_architecture_diagram,
    save_neural_network_training_curve,
    save_split_protocol_artifacts,
)
from .modeling import (
    ModelBuilder,
    build_knn_model,
    build_logistic_model,
    build_neural_network_model,
    build_parameter_label,
    build_representation_key,
    build_representation_label,
    compute_metrics,
    run_validation_search,
    select_best_configuration,
    select_best_pca_configuration,
)

LOGGER = logging.getLogger(__name__)


def run_pca_component_benchmark(
    X_search_train: np.ndarray,
    X_validation: np.ndarray,
    y_search_train: np.ndarray,
    y_validation: np.ndarray,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_builders: dict[str, ModelBuilder],
    best_params_by_model: dict[str, dict[str, Any]],
    metrics_frame: pd.DataFrame,
    config: PipelineConfig,
) -> None:
    validation_records: list[dict[str, Any]] = []

    representation_options: list[int | None] = [None, *PCA_BENCHMARK_COMPONENTS]
    for n_components in representation_options:
        if n_components is None:
            X_fit = X_search_train
            X_eval = X_validation
            explained_variance_ratio = 1.0
            transform_seconds = 0.0
        else:
            pca = PCA(n_components=n_components, random_state=config.random_state)
            transform_start = perf_counter()
            X_fit = pca.fit_transform(X_search_train)
            X_eval = pca.transform(X_validation)
            transform_seconds = perf_counter() - transform_start
            explained_variance_ratio = float(pca.explained_variance_ratio_.sum())

        for model_key, params in best_params_by_model.items():
            model = model_builders[model_key](params, config.random_state)

            fit_start = perf_counter()
            model.fit(X_fit, y_search_train)
            fit_seconds = perf_counter() - fit_start

            predict_start = perf_counter()
            predictions = model.predict(X_eval)
            predict_seconds = perf_counter() - predict_start

            validation_records.append(
                {
                    "model_key": model_key,
                    "model": MODEL_DISPLAY_NAMES[model_key],
                    "selected_configuration": build_parameter_label(model_key, params),
                    "representation_key": build_representation_key(n_components),
                    "representation_label": build_representation_label(n_components),
                    "is_pca": n_components is not None,
                    "n_components": 784 if n_components is None else n_components,
                    "explained_variance_ratio": round(explained_variance_ratio, 5),
                    "transform_seconds": round(transform_seconds, 3),
                    "fit_seconds": round(fit_seconds, 3),
                    "predict_seconds": round(predict_seconds, 3),
                    **compute_metrics(y_validation, predictions, prefix="validation_"),
                }
            )

    validation_frame = pd.DataFrame(validation_records)
    validation_frame.to_csv(config.results_dir / "pca_component_benchmark_validation.csv", index=False)
    save_pca_component_benchmark_plot(
        validation_frame,
        config.figures_dir / "pca_component_benchmark.png",
    )

    selection_records: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []
    for model_key, params in best_params_by_model.items():
        model_validation_frame = validation_frame[validation_frame["model_key"] == model_key].copy()
        best_pca_row = select_best_pca_configuration(model_validation_frame)
        selection_records.append(
            {
                "model_key": model_key,
                "model": MODEL_DISPLAY_NAMES[model_key],
                "selected_configuration": build_parameter_label(model_key, params),
                "selected_pca_components": int(best_pca_row["n_components"]),
                "validation_accuracy": float(best_pca_row["validation_accuracy"]),
                "validation_f1_macro": float(best_pca_row["validation_f1_macro"]),
                "explained_variance_ratio": float(best_pca_row["explained_variance_ratio"]),
            }
        )

        raw_row = metrics_frame[metrics_frame["model_key"] == model_key].iloc[0]
        summary_records.append(
            {
                "model_key": model_key,
                "model": MODEL_DISPLAY_NAMES[model_key],
                "selected_configuration": build_parameter_label(model_key, params),
                "representation_key": "raw",
                "representation_label": build_representation_label(None),
                "is_pca": False,
                "n_components": 784,
                "explained_variance_ratio": 1.0,
                "transform_seconds": 0.0,
                "train_seconds": float(raw_row["train_seconds"]),
                "predict_seconds": float(raw_row["predict_seconds"]),
                "accuracy": float(raw_row["accuracy"]),
                "precision_macro": float(raw_row["precision_macro"]),
                "recall_macro": float(raw_row["recall_macro"]),
                "f1_macro": float(raw_row["f1_macro"]),
            }
        )

        pca = PCA(
            n_components=int(best_pca_row["n_components"]),
            random_state=config.random_state,
        )
        transform_start = perf_counter()
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        transform_seconds = perf_counter() - transform_start

        model = model_builders[model_key](params, config.random_state)
        fit_start = perf_counter()
        model.fit(X_train_pca, y_train)
        train_seconds = perf_counter() - fit_start

        predict_start = perf_counter()
        predictions = model.predict(X_test_pca)
        predict_seconds = perf_counter() - predict_start

        summary_records.append(
            {
                "model_key": model_key,
                "model": MODEL_DISPLAY_NAMES[model_key],
                "selected_configuration": build_parameter_label(model_key, params),
                "representation_key": build_representation_key(int(best_pca_row["n_components"])),
                "representation_label": build_representation_label(int(best_pca_row["n_components"])),
                "is_pca": True,
                "n_components": int(best_pca_row["n_components"]),
                "explained_variance_ratio": round(float(pca.explained_variance_ratio_.sum()), 5),
                "transform_seconds": round(transform_seconds, 3),
                "train_seconds": round(train_seconds, 3),
                "predict_seconds": round(predict_seconds, 3),
                **compute_metrics(y_test, predictions),
            }
        )

    selection_frame = pd.DataFrame(selection_records).sort_values(
        by=["validation_accuracy", "validation_f1_macro"],
        ascending=[False, False],
    )
    selection_frame.to_csv(config.results_dir / "pca_selection_summary.csv", index=False)

    summary_frame = pd.DataFrame(summary_records)
    raw_accuracy_map = (
        summary_frame[~summary_frame["is_pca"]]
        .set_index("model_key")["accuracy"]
        .to_dict()
    )
    summary_frame["accuracy_delta_vs_raw"] = summary_frame["model_key"].map(raw_accuracy_map)
    summary_frame["accuracy_delta_vs_raw"] = (
        summary_frame["accuracy"] - summary_frame["accuracy_delta_vs_raw"]
    ).round(5)
    summary_frame = summary_frame.sort_values(by=["model_key", "is_pca"])
    summary_frame.to_csv(config.results_dir / "raw_vs_best_pca_test_summary.csv", index=False)
    save_raw_vs_best_pca_dashboard(
        summary_frame,
        config.figures_dir / "raw_vs_best_pca_dashboard.png",
    )


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

    model_builders: dict[str, ModelBuilder] = {
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
    best_params_by_model: dict[str, dict[str, Any]] = {}

    for model_key, model_builder in model_builders.items():
        LOGGER.info("Searching %s hyperparameters", model_key)
        search_frame, best_params = run_validation_search(
            model_key=model_key,
            model_builder=model_builder,
            param_grid=search_grids[model_key],
            X_search_train=X_search_train,
            X_validation=X_validation,
            y_search_train=y_search_train,
            y_validation=y_validation,
            random_state=config.random_state,
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
        best_params_by_model[model_key] = best_params
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
    run_pca_component_benchmark(
        X_search_train=X_search_train,
        X_validation=X_validation,
        y_search_train=y_search_train,
        y_validation=y_validation,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        model_builders=model_builders,
        best_params_by_model=best_params_by_model,
        metrics_frame=metrics_frame,
        config=config,
    )

    return metrics_frame
