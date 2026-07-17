import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data, path):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_").lower()


def classification_metrics(y_true, y_pred, y_prob=None):
    labels = np.unique(y_true)
    average = "binary" if len(labels) == 2 else "weighted"
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "specificity": None,
        "fpr": None,
        "fnr": None,
        "roc_auc": None,
    }
    if len(labels) == 2:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            result["specificity"] = float(tn / (tn + fp)) if (tn + fp) else 0.0
            result["fpr"] = float(fp / (fp + tn)) if (fp + tn) else 0.0
            result["fnr"] = float(fn / (fn + tp)) if (fn + tp) else 0.0
    if y_prob is not None:
        try:
            if len(labels) == 2:
                prob = y_prob[:, 1] if getattr(y_prob, "ndim", 1) == 2 else y_prob
                result["roc_auc"] = float(roc_auc_score(y_true, prob))
            else:
                result["roc_auc"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
                )
        except Exception:
            result["roc_auc"] = None
    return result


def save_prediction_artifacts(output_dir, model_key, y_true, y_pred, y_prob=None):
    """Lưu prediction để vẽ confusion matrix, ROC và PR curve sau khi train."""
    os.makedirs(output_dir, exist_ok=True)
    payload = {"y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred)}
    if y_prob is not None:
        payload["y_prob"] = np.asarray(y_prob)
    np.savez_compressed(os.path.join(output_dir, f"{model_key}_predictions.npz"), **payload)

    labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels]).to_csv(
        os.path.join(output_dir, f"{model_key}_confusion_matrix.csv")
    )
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(
        os.path.join(output_dir, f"{model_key}_classification_report.csv")
    )
