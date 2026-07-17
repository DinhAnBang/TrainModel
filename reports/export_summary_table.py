import os
import sys
import pandas as pd
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
OUTPUT_DIR = PROJECT_DIR / "reports" / "final_tables"


def export_summary_table():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for dataset_dir in ARTIFACT_DIR.iterdir():
        if not dataset_dir.is_dir():
            continue

        if dataset_dir.name == "best_models":
            continue

        history_dir = dataset_dir / "history"

        if not history_dir.exists():
            continue

        for run_dir in history_dir.iterdir():
            result_path = run_dir / "reports" / "final_results.csv"

            if result_path.exists():
                df = pd.read_csv(result_path)
                df["dataset"] = dataset_dir.name
                df["run_id"] = run_dir.name
                rows.append(df)

    if len(rows) == 0:
        print("Không tìm thấy final_results.csv nào.")
        return

    final_df = pd.concat(rows, ignore_index=True)

    cols = [
        "dataset",
        "run_id",
        "model_group",
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "train_time_sec"
    ]

    final_df = final_df[cols]
    final_df = final_df.sort_values(by=["dataset", "f1"], ascending=[True, False])

    output_path = OUTPUT_DIR / "all_experiment_results.csv"
    final_df.to_csv(output_path, index=False)

    print("Saved:", output_path)
    print(final_df)


if __name__ == "__main__":
    export_summary_table()