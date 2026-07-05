import os
import shutil
import pandas as pd


def safe_name(name):
    return name.replace(" ", "_").lower()


def select_best_model(dataset_name, run_dir, best_model_dir):
    os.makedirs(best_model_dir, exist_ok=True)

    report_path = os.path.join(run_dir, "reports", "final_results.csv")

    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Không tìm thấy final_results.csv: {report_path}")

    df = pd.read_csv(report_path)
    df = df.sort_values(by="f1", ascending=False)

    best = df.iloc[0]

    model_group = best["model_group"]
    model_name = best["model"]
    model_file_name = None
    source_model_path = None

    if model_group == "Baseline ML":
        model_file_name = safe_name(model_name) + ".pkl"
        source_model_path = os.path.join(run_dir, "baseline", model_file_name)

    elif model_group == "GNN":
        model_file_name = model_name.lower() + ".pt"
        source_model_path = os.path.join(run_dir, "gnn", model_file_name)

    else:
        raise ValueError(f"Unknown model group: {model_group}")

    if not os.path.exists(source_model_path):
        raise FileNotFoundError(f"Không tìm thấy model file: {source_model_path}")

    dataset_best_dir = os.path.join(best_model_dir, dataset_name)
    os.makedirs(dataset_best_dir, exist_ok=True)

    target_model_path = os.path.join(dataset_best_dir, model_file_name)
    shutil.copy2(source_model_path, target_model_path)

    processed_dir = os.path.join(run_dir, "processed")

    for file_name in [
        "scaler.pkl",
        "categorical_encoder.pkl",
        "preprocess_info.pkl",
        "feature_names.txt",
        "preprocess_summary.json"
    ]:
        source = os.path.join(processed_dir, file_name)
        if os.path.exists(source):
            shutil.copy2(source, os.path.join(dataset_best_dir, file_name))

    shutil.copy2(report_path, os.path.join(dataset_best_dir, "final_results.csv"))

    with open(os.path.join(dataset_best_dir, "best_model_info.txt"), "w", encoding="utf-8") as f:
        f.write("BEST MODEL INFO\n")
        f.write("=" * 50 + "\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Model group: {model_group}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {best['accuracy']}\n")
        f.write(f"Precision: {best['precision']}\n")
        f.write(f"Recall: {best['recall']}\n")
        f.write(f"F1: {best['f1']}\n")
        f.write(f"Source run: {run_dir}\n")
        f.write(f"Model file: {model_file_name}\n")

    print("Best model selected:")
    print(best)
    print("Saved to:", dataset_best_dir)

    return best