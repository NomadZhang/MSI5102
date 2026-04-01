from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import PCA_BENCHMARK_COMPONENTS, PipelineConfig
from .modeling import build_representation_label, select_best_configuration


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


def save_pca_component_benchmark_plot(benchmark_frame: pd.DataFrame, output_path: Path) -> None:
    plot_frame = benchmark_frame.copy()
    order = [build_representation_label(None)] + [
        build_representation_label(n_components) for n_components in PCA_BENCHMARK_COMPONENTS
    ]
    plot_frame["representation_label"] = pd.Categorical(
        plot_frame["representation_label"],
        categories=order,
        ordered=True,
    )
    plot_frame = plot_frame.sort_values(["representation_label", "model"])

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    sns.lineplot(
        data=plot_frame,
        x="representation_label",
        y="validation_accuracy",
        hue="model",
        marker="o",
        linewidth=2.2,
        ax=axes[0],
    )
    axes[0].set_title("Validation Accuracy Across Raw and PCA Feature Sets")
    axes[0].set_xlabel("Feature representation")
    axes[0].set_ylabel("Validation accuracy")
    axes[0].tick_params(axis="x", rotation=25)

    for _, row in (
        plot_frame[plot_frame["is_pca"]]
        .sort_values(
            by=["validation_accuracy", "validation_f1_macro", "n_components"],
            ascending=[False, False, True],
        )
        .groupby("model_key", as_index=False)
        .first()
        .iterrows()
    ):
        axes[0].annotate(
            f"best {int(row['n_components'])}",
            (row["representation_label"], float(row["validation_accuracy"])),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#333333", "alpha": 0.9},
        )

    variance_frame = (
        plot_frame[plot_frame["is_pca"]][["n_components", "explained_variance_ratio"]]
        .drop_duplicates()
        .sort_values("n_components")
    )
    sns.lineplot(
        data=variance_frame,
        x="n_components",
        y="explained_variance_ratio",
        marker="o",
        linewidth=2.2,
        ax=axes[1],
    )
    axes[1].set_title("PCA Variance Retained by Component Count")
    axes[1].set_xlabel("PCA components")
    axes[1].set_ylabel("Explained variance ratio")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_raw_vs_best_pca_dashboard(summary_frame: pd.DataFrame, output_path: Path) -> None:
    plot_frame = summary_frame.copy()
    plot_frame["comparison_label"] = np.where(plot_frame["is_pca"], "Best PCA", "Raw")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    sns.barplot(
        data=plot_frame,
        x="model",
        y="accuracy",
        hue="comparison_label",
        ax=axes[0],
    )
    axes[0].set_title("Final Test Accuracy: Raw vs Best PCA")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Accuracy")
    axes[0].tick_params(axis="x", rotation=15)

    sns.barplot(
        data=plot_frame,
        x="model",
        y="predict_seconds",
        hue="comparison_label",
        ax=axes[1],
    )
    axes[1].set_title("Final Test Prediction Time: Raw vs Best PCA")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Seconds")
    axes[1].tick_params(axis="x", rotation=15)

    pca_rows = plot_frame[plot_frame["is_pca"]].reset_index(drop=True)
    for axis in axes:
        pca_patches = axis.patches[len(pca_rows) :]
        if len(pca_patches) != len(pca_rows):
            pca_patches = axis.patches[-len(pca_rows) :]
        for patch, row in zip(pca_patches, pca_rows.itertuples(index=False), strict=True):
            axis.annotate(
                f"{int(row.n_components)} PCs",
                (
                    patch.get_x() + (patch.get_width() / 2),
                    patch.get_height(),
                ),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


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
