import os

import pandas as pd


def compare_results(ml_dir, deep_dir, gnn_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    paths = [
        os.path.join(ml_dir, "ml_results.csv"),
        os.path.join(deep_dir, "deep_results.csv"),
        os.path.join(gnn_dir, "gnn_results.csv"),
    ]
    frames = [pd.read_csv(path) for path in paths if os.path.exists(path)]
    if not frames:
        raise FileNotFoundError("Không tìm thấy file kết quả model nào")
    final_df = pd.concat(frames, ignore_index=True)
    final_df = final_df.sort_values(["f1", "roc_auc", "accuracy"], ascending=False, na_position="last").reset_index(drop=True)
    final_df.insert(0, "rank", range(1, len(final_df) + 1))
    final_path = os.path.join(output_dir, "final_results.csv")
    final_df.to_csv(final_path, index=False)
    best = final_df.iloc[0]
    with open(os.path.join(output_dir, "final_results_summary.txt"), "w", encoding="utf-8") as f:
        f.write("FINAL MODEL COMPARISON\n" + "=" * 70 + "\n\n")
        f.write(final_df.to_string(index=False))
        f.write("\n\nBEST MODEL\n" + "-" * 40 + "\n")
        for key in ["model_group", "model", "accuracy", "precision", "recall", "f1", "roc_auc", "train_time_sec"]:
            f.write(f"{key}: {best.get(key)}\n")
    print("\n" + "=" * 110)
    print("FINAL MODEL RANKING (sorted by F1, ROC-AUC, Accuracy)")
    print("=" * 110)
    print(final_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("=" * 110 + "\n")
    return final_df
