import json
import os
import shutil

import pandas as pd

MODEL_FILES = {
    "Random Forest": ("ml", "random_forest.pkl"),
    "XGBoost": ("ml", "xgboost.pkl"),
    "CNN": ("deep", "cnn.pt"),
    "ResNet": ("deep", "resnet.pt"),
    "GCN": ("gnn", "gcn.pt"),
}


def select_best_model(experiment_name, run_dir, best_model_dir):
    report_path = os.path.join(run_dir, "reports", "final_results.csv")
    df = pd.read_csv(report_path).sort_values(["f1", "roc_auc", "accuracy"], ascending=False, na_position="last")
    best = df.iloc[0]
    model_name = best["model"]
    if model_name not in MODEL_FILES:
        raise ValueError(f"Chưa cấu hình file model cho: {model_name}")
    subdir, filename = MODEL_FILES[model_name]
    source_model = os.path.join(run_dir, subdir, filename)
    if not os.path.exists(source_model):
        raise FileNotFoundError(source_model)

    target_dir = os.path.join(best_model_dir, experiment_name)
    os.makedirs(target_dir, exist_ok=True)
    for existing in os.listdir(target_dir):
        path = os.path.join(target_dir, existing)
        if os.path.isfile(path):
            os.remove(path)
    shutil.copy2(source_model, os.path.join(target_dir, filename))

    processed_dir = os.path.join(run_dir, "processed")
    for file_name in ["scaler.pkl", "categorical_encoder.pkl", "label_encoder.pkl", "preprocess_info.pkl", "feature_names.txt", "preprocess_summary.json"]:
        source = os.path.join(processed_dir, file_name)
        if os.path.exists(source):
            shutil.copy2(source, os.path.join(target_dir, file_name))
    shutil.copy2(report_path, os.path.join(target_dir, "final_results.csv"))

    info = {
        "experiment": experiment_name,
        "model_group": best["model_group"],
        "model": model_name,
        "model_file": filename,
        "metrics": {key: (None if pd.isna(best.get(key)) else float(best.get(key))) for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "train_time_sec"]},
        "source_run": run_dir,
    }
    with open(os.path.join(target_dir, "best_model_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)
    print("Best model:", model_name)
    print("Saved to:", target_dir)
    return best
