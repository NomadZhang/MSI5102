from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.neural_network import MLPClassifier

from .config import MODEL_DISPLAY_NAMES, PipelineConfig


def save_confusion_matrix_plot(matrix: np.ndarray, model_name: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def build_top_confusions_frame(y_true: np.ndarray, predictions: np.ndarray) -> pd.DataFrame:
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
