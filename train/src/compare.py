import os
import pandas as pd


def compare_results(baseline_dir, gnn_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    baseline_path = os.path.join(baseline_dir, "baseline_results.csv")
    gnn_path = os.path.join(gnn_dir, "gnn_results.csv")

    baseline_df = pd.read_csv(baseline_path)
    gnn_df = pd.read_csv(gnn_path)

    baseline_df["model_group"] = "Baseline ML"
    gnn_df["model_group"] = "GNN"

    if "roc_auc" not in gnn_df.columns:
        gnn_df["roc_auc"] = None

    cols = [
        "model_group",
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "train_time_sec"
    ]

    final_df = pd.concat(
        [baseline_df[cols], gnn_df[cols]],
        ignore_index=True
    )

    final_df = final_df.sort_values(by="f1", ascending=False)

    final_df.to_csv(os.path.join(output_dir, "final_results.csv"), index=False)

    with open(os.path.join(output_dir, "final_results_summary.txt"), "w", encoding="utf-8") as f:
        f.write("FINAL MODEL COMPARISON\n")
        f.write("=" * 50 + "\n\n")
        f.write(final_df.to_string(index=False))
        f.write("\n\n")

        best = final_df.iloc[0]

        f.write("BEST MODEL\n")
        f.write("-" * 30 + "\n")
        f.write(f"Model group: {best['model_group']}\n")
        f.write(f"Model: {best['model']}\n")
        f.write(f"F1-score: {best['f1']:.4f}\n")
        f.write(f"Accuracy: {best['accuracy']:.4f}\n")
        f.write(f"Precision: {best['precision']:.4f}\n")
        f.write(f"Recall: {best['recall']:.4f}\n")

    print(final_df)

    return final_df