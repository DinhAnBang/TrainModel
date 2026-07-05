import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_results(report_dir):
    result_path = os.path.join(report_dir, "final_results.csv")
    output_dir = os.path.join(report_dir, "figures")

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(result_path)

    metrics = ["accuracy", "precision", "recall", "f1"]

    for metric in metrics:
        df_plot = df.sort_values(by=metric, ascending=False)

        plt.figure(figsize=(10, 6))
        plt.bar(df_plot["model"], df_plot[metric])
        plt.title(f"{metric.upper()} Comparison")
        plt.xlabel("Model")
        plt.ylabel(metric.upper())
        plt.ylim(0, 1.05)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_comparison.png"), dpi=300)
        plt.close()

    df_metric = df[["model"] + metrics].set_index("model")

    df_metric.plot(kind="bar", figsize=(12, 6))
    plt.title("Model Evaluation Metrics")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_metrics_comparison.png"), dpi=300)
    plt.close()

    df_time = df.sort_values(by="train_time_sec", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.bar(df_time["model"], df_time["train_time_sec"])
    plt.title("Training Time Comparison")
    plt.xlabel("Model")
    plt.ylabel("Seconds")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_time_comparison.png"), dpi=300)
    plt.close()

    print("Saved figures to:", output_dir)