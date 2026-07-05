import pandas as pd
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_DIR / "reports" / "final_tables" / "all_experiment_results.csv"
OUTPUT_DIR = PROJECT_DIR / "reports" / "final_tables"


def build_final_report():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {INPUT_PATH}. Hãy chạy export_summary_table.py trước."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    best_rows = []

    for dataset in df["dataset"].unique():
        temp = df[df["dataset"] == dataset].sort_values(by="f1", ascending=False)
        best_rows.append(temp.iloc[0])

    best_df = pd.DataFrame(best_rows)

    best_output = OUTPUT_DIR / "best_model_by_dataset.csv"
    best_df.to_csv(best_output, index=False)

    summary_txt = OUTPUT_DIR / "final_report_summary.txt"

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("FINAL EXPERIMENT SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        for _, row in best_df.iterrows():
            f.write(f"Dataset: {row['dataset']}\n")
            f.write(f"Best model: {row['model']}\n")
            f.write(f"Model group: {row['model_group']}\n")
            f.write(f"Run ID: {row['run_id']}\n")
            f.write(f"Accuracy: {row['accuracy']:.4f}\n")
            f.write(f"Precision: {row['precision']:.4f}\n")
            f.write(f"Recall: {row['recall']:.4f}\n")
            f.write(f"F1-score: {row['f1']:.4f}\n")
            f.write("-" * 60 + "\n")

    print("Saved:", best_output)
    print("Saved:", summary_txt)
    print(best_df)


if __name__ == "__main__":
    build_final_report()