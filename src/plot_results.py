import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


MODEL_LOCATIONS = {
    "Random Forest": ("ml", "random_forest"),
    "XGBoost": ("ml", "xgboost"),
    "CNN": ("deep", "cnn"),
    "ResNet": ("deep", "resnet"),
    "GCN": ("gnn", "gcn"),
}


def _add_bar_labels(ax, decimals=3):
    for container in ax.containers:
        labels = []
        for bar in container:
            height = bar.get_height()
            labels.append(f"{height:.{decimals}f}")
        ax.bar_label(container, labels=labels, padding=2, fontsize=8)


def _load_predictions(run_dir, model_name):
    location = MODEL_LOCATIONS.get(model_name)
    if not location:
        return None
    subdir, key = location
    path = os.path.join(run_dir, subdir, f"{key}_predictions.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return {name: data[name] for name in data.files}


def _plot_confusion_matrix(y_true, y_pred, model_name, output_path):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    image = plt.imshow(cm, aspect="auto")
    plt.colorbar(image, label="Count")
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion Matrix - {model_name}")
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     color="white" if cm[i, j] > threshold else "black")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _positive_scores(y_prob):
    if y_prob is None:
        return None
    if y_prob.ndim == 2 and y_prob.shape[1] >= 2:
        return y_prob[:, 1]
    return y_prob.ravel()


def plot_results(report_dir):
    result_path = os.path.join(report_dir, "final_results.csv")
    output_dir = os.path.join(report_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(result_path)
    run_dir = str(Path(report_dir).parent)

    metrics = [metric for metric in [
        "accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "mcc", "roc_auc"
    ] if metric in df.columns]

    # Biểu đồ cột riêng từng metric.
    for metric in metrics:
        plot_df = df.sort_values(metric, ascending=False)
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        ax.bar(plot_df["model"], plot_df[metric].fillna(0))
        ax.set_title(f"{metric.replace('_', ' ').title()} Comparison")
        ax.set_xlabel("Model")
        ax.set_ylabel(metric.replace("_", " ").title())
        if metric != "mcc":
            ax.set_ylim(0, 1.08)
        ax.tick_params(axis="x", rotation=25)
        _add_bar_labels(ax)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_comparison.png"), dpi=300)
        plt.close()

    # Biểu đồ cột nhóm tổng hợp, phần dễ so sánh nhất trong báo cáo.
    core_metrics = [m for m in ["accuracy", "precision", "recall", "f1", "roc_auc"] if m in df.columns]
    if core_metrics:
        metric_df = df[["model"] + core_metrics].set_index("model")
        ax = metric_df.plot(kind="bar", figsize=(14, 7))
        ax.set_title("Overall Model Performance Comparison")
        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.08)
        ax.tick_params(axis="x", rotation=20)
        ax.legend(title="Metric", ncol=min(5, len(core_metrics)), loc="lower center")
        _add_bar_labels(ax, decimals=3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "all_metrics_grouped_bar.png"), dpi=300)
        plt.close()

    # F1 xếp hạng theo model.
    if "f1" in df.columns:
        ranked = df.sort_values("f1", ascending=True)
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        ax.barh(ranked["model"], ranked["f1"])
        ax.set_title("Model Ranking by F1-score")
        ax.set_xlabel("F1-score")
        ax.set_xlim(0, 1.08)
        for patch, value in zip(ax.patches, ranked["f1"]):
            ax.text(value, patch.get_y() + patch.get_height() / 2, f" {value:.4f}", va="center")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "model_ranking_by_f1.png"), dpi=300)
        plt.close()

    if "train_time_sec" in df.columns:
        time_df = df.sort_values("train_time_sec", ascending=False)
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        ax.bar(time_df["model"], time_df["train_time_sec"])
        ax.set_title("Training Time Comparison")
        ax.set_ylabel("Seconds")
        ax.tick_params(axis="x", rotation=25)
        _add_bar_labels(ax, decimals=1)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "train_time_comparison.png"), dpi=300)
        plt.close()

    # Confusion matrix và dữ liệu ROC/PR.
    roc_series = []
    pr_series = []
    for model_name in df["model"]:
        predictions = _load_predictions(run_dir, model_name)
        if predictions is None:
            continue
        y_true = predictions["y_true"]
        y_pred = predictions["y_pred"]
        y_prob = predictions.get("y_prob")
        safe_model = model_name.lower().replace(" ", "_")
        _plot_confusion_matrix(
            y_true, y_pred, model_name,
            os.path.join(output_dir, f"confusion_matrix_{safe_model}.png"),
        )
        scores = _positive_scores(y_prob)
        if scores is not None and len(np.unique(y_true)) == 2:
            fpr, tpr, _ = roc_curve(y_true, scores)
            roc_series.append((model_name, fpr, tpr, auc(fpr, tpr)))
            precision, recall, _ = precision_recall_curve(y_true, scores)
            pr_series.append((model_name, recall, precision, auc(recall, precision)))

    if roc_series:
        plt.figure(figsize=(8, 7))
        for model_name, fpr, tpr, area in roc_series:
            plt.plot(fpr, tpr, label=f"{model_name} (AUC={area:.4f})")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves - All Models")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "roc_curves_all_models.png"), dpi=300)
        plt.close()

    if pr_series:
        plt.figure(figsize=(8, 7))
        for model_name, recall, precision, area in pr_series:
            plt.plot(recall, precision, label=f"{model_name} (AUC={area:.4f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curves - All Models")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "precision_recall_curves_all_models.png"), dpi=300)
        plt.close()

    # Learning curves CNN, ResNet, GCN nếu history tồn tại.
    history_paths = [
        ("CNN", os.path.join(run_dir, "deep", "cnn_history.csv")),
        ("ResNet", os.path.join(run_dir, "deep", "resnet_history.csv")),
        ("GCN", os.path.join(run_dir, "gnn", "gcn_history.csv")),
    ]
    for model_name, history_path in history_paths:
        if not os.path.exists(history_path):
            continue
        history = pd.read_csv(history_path)
        val_f1_col = "val_f1" if "val_f1" in history.columns else "f1" if "f1" in history.columns else None
        if "epoch" not in history.columns or "loss" not in history.columns:
            continue
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(history["epoch"], history["loss"], label="Training loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        lines, labels = ax1.get_legend_handles_labels()
        if val_f1_col:
            ax2 = ax1.twinx()
            ax2.plot(history["epoch"], history[val_f1_col], label="Validation F1", linestyle="--")
            ax2.set_ylabel("Validation F1")
            ax2.set_ylim(0, 1.05)
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines += lines2
            labels += labels2
        ax1.legend(lines, labels, loc="best")
        plt.title(f"Training History - {model_name}")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"learning_curve_{model_name.lower()}.png"), dpi=300)
        plt.close(fig)

    print("Saved figures to:", output_dir)
