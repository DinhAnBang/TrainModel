import sys
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


PROJECT_DIR = Path(__file__).resolve().parent.parent
BEST_MODEL_DIR = PROJECT_DIR / "artifacts" / "best_models"
PREDICTION_DIR = PROJECT_DIR / "reports" / "predictions"


def clean_dataframe(X):
    X = X.copy()

    for col in X.columns:
        X[col] = X[col].astype(str)
        X[col] = X[col].str.replace(",", "", regex=False)
        X[col] = X[col].str.replace("%", "", regex=False)
        X[col] = X[col].str.strip()
        X[col] = X[col].str.lower()
        X[col] = X[col].replace(["nan", "none", "na", ""], np.nan)

    return X


def load_best_ml_model(dataset_name):
    model_dir = BEST_MODEL_DIR / dataset_name

    if not model_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy best model folder: {model_dir}")

    model_files = list(model_dir.glob("*.pkl"))

    model_files = [
        f for f in model_files
        if f.name not in [
            "scaler.pkl",
            "categorical_encoder.pkl",
            "preprocess_info.pkl"
        ]
    ]

    if len(model_files) == 0:
        raise FileNotFoundError("Không tìm thấy file model .pkl")

    model_path = model_files[0]

    model = joblib.load(model_path)
    scaler = joblib.load(model_dir / "scaler.pkl")
    preprocess_info = joblib.load(model_dir / "preprocess_info.pkl")

    encoder_path = model_dir / "categorical_encoder.pkl"
    encoder = joblib.load(encoder_path) if encoder_path.exists() else None

    return model, scaler, encoder, preprocess_info, model_path


def preprocess_new_data(df, scaler, encoder, preprocess_info):
    numeric_cols = preprocess_info["numeric_cols"]
    categorical_cols = preprocess_info["categorical_cols"]
    feature_names = preprocess_info["feature_names"]
    drop_cols = preprocess_info["drop_cols"]

    X = df.drop(columns=drop_cols, errors="ignore")
    X = clean_dataframe(X)

    parts = []

    if len(numeric_cols) > 0:
        X_num = pd.DataFrame(index=X.index)

        for col in numeric_cols:
            if col in X.columns:
                X_num[col] = pd.to_numeric(X[col], errors="coerce")
            else:
                X_num[col] = 0

        X_num = X_num.fillna(0)
        parts.append(X_num)

    if len(categorical_cols) > 0 and encoder is not None:
        X_cat = pd.DataFrame(index=X.index)

        for col in categorical_cols:
            if col in X.columns:
                X_cat[col] = X[col].fillna("missing")
            else:
                X_cat[col] = "missing"

        X_cat_encoded = encoder.transform(X_cat)
        cat_feature_names = [f"{col}_encoded" for col in categorical_cols]

        X_cat_df = pd.DataFrame(
            X_cat_encoded,
            columns=cat_feature_names,
            index=X.index
        )

        parts.append(X_cat_df)

    if len(parts) == 0:
        raise ValueError("Không có feature hợp lệ để predict")

    X_final = pd.concat(parts, axis=1)

    for col in feature_names:
        if col not in X_final.columns:
            X_final[col] = 0

    X_final = X_final[feature_names]

    X_scaled = scaler.transform(X_final)

    return X_scaled


def evaluate_if_label_exists(result, dataset_name, output_dir):
    if "label" not in result.columns:
        print("Không có cột label nên chỉ predict, không evaluate.")
        return None

    y_true = result["label"].astype(int).values
    y_pred = result["prediction"].astype(int).values

    metrics = {
        "dataset": dataset_name,
        "samples": int(len(result)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    }

    if "attack_probability" in result.columns:
        try:
            metrics["roc_auc"] = float(
                roc_auc_score(y_true, result["attack_probability"].values)
            )
        except Exception:
            metrics["roc_auc"] = None

    metrics_path = output_dir / "metrics.json"
    txt_path = output_dir / "report.txt"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("PREDICTION EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Samples: {len(result)}\n\n")
        f.write(f"Accuracy : {metrics['accuracy']:.6f}\n")
        f.write(f"Precision: {metrics['precision']:.6f}\n")
        f.write(f"Recall   : {metrics['recall']:.6f}\n")
        f.write(f"F1-score : {metrics['f1']:.6f}\n")

        if metrics.get("roc_auc") is not None:
            f.write(f"ROC-AUC  : {metrics['roc_auc']:.6f}\n")

        f.write("\nConfusion Matrix:\n")
        f.write(str(metrics["confusion_matrix"]))
        f.write("\n\nClassification Report:\n")
        f.write(metrics["classification_report"])

    print("\nPREDICTION EVALUATION")
    print("=" * 60)
    print(f"Accuracy : {metrics['accuracy']:.6f}")
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall   : {metrics['recall']:.6f}")
    print(f"F1-score : {metrics['f1']:.6f}")

    if metrics.get("roc_auc") is not None:
        print(f"ROC-AUC  : {metrics['roc_auc']:.6f}")

    print("Confusion Matrix:", metrics["confusion_matrix"])
    print("Saved metrics:", metrics_path)
    print("Saved report :", txt_path)

    return metrics


def create_prediction_output_dir(dataset_name, csv_path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = Path(csv_path).stem

    output_dir = (
        PREDICTION_DIR
        / dataset_name
        / "history"
        / f"{csv_name}_{timestamp}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    latest_file = PREDICTION_DIR / dataset_name / "latest_prediction.txt"
    latest_file.parent.mkdir(parents=True, exist_ok=True)

    with open(latest_file, "w", encoding="utf-8") as f:
        f.write(str(output_dir))

    return output_dir


def save_prediction_config(output_dir, dataset_name, csv_path, model_path):
    info = {
        "dataset": dataset_name,
        "input_csv": str(csv_path),
        "model_path": str(model_path),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(output_dir / "prediction_config.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)


def predict_csv(dataset_name, csv_path):
    model, scaler, encoder, preprocess_info, model_path = load_best_ml_model(dataset_name)

    df = pd.read_csv(csv_path, low_memory=False)

    X = preprocess_new_data(df, scaler, encoder, preprocess_info)

    pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
    else:
        prob = [None] * len(pred)

    result = df.copy()
    result["prediction"] = pred
    result["attack_probability"] = prob

    output_dir = create_prediction_output_dir(dataset_name, csv_path)

    prediction_path = output_dir / "prediction.csv"

    result.to_csv(prediction_path, index=False)

    save_prediction_config(output_dir, dataset_name, csv_path, model_path)

    print("Loaded model:", model_path)
    print("Saved prediction:", prediction_path)
    print("Output dir:", output_dir)
    print(result[["prediction", "attack_probability"]].head())

    evaluate_if_label_exists(result, dataset_name, output_dir)

    return result


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("python inference/predict_ml.py <dataset_name> <csv_path>")
        sys.exit(1)

    dataset_name = sys.argv[1]
    csv_path = sys.argv[2]

    predict_csv(dataset_name, csv_path)