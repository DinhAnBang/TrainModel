import json
import argparse
from pathlib import Path
from datetime import datetime

import torch

from config import DATASETS
from src.loader import load_dataset
from src.preprocess import preprocess_dataset
from src.baseline import train_baseline
from src.graph_builder import build_graph
from src.gnn_train import train_gnn
from src.compare import compare_results
from src.plot_results import plot_results
from src.select_best_model import select_best_model


PROJECT_DIR = Path(__file__).resolve().parent
BASE_ARTIFACT_DIR = PROJECT_DIR / "artifacts"


def get_device_info():
    return {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }


def save_run_config(run_dir, dataset_name, step, config, gnn_params):
    run_info = {
        "dataset": dataset_name,
        "step": step,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": config,
        "gnn_params": gnn_params,
        "device_info": get_device_info()
    }

    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=4, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True, choices=DATASETS.keys())

    parser.add_argument(
        "--step",
        default="all",
        choices=[
            "all",
            "preprocess",
            "baseline",
            "graph",
            "gnn",
            "compare",
            "plot",
            "select_best"
        ]
    )

    parser.add_argument("--run_id", default=None)

    args = parser.parse_args()

    dataset_name = args.dataset
    step = args.step
    config = DATASETS[dataset_name]

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    dataset_root_dir = BASE_ARTIFACT_DIR / dataset_name
    run_dir = dataset_root_dir / "history" / run_id

    processed_dir = run_dir / "processed"
    baseline_dir = run_dir / "baseline"
    graph_dir = run_dir / "graph"
    gnn_dir = run_dir / "gnn"
    report_dir = run_dir / "reports"

    gnn_params = {
        "epochs": config.get("epochs", 50),
        "hidden_dim": config.get("hidden_dim", 64),
        "lr": config.get("lr", 0.01)
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_root_dir.mkdir(parents=True, exist_ok=True)

    save_run_config(run_dir, dataset_name, step, config, gnn_params)

    with open(dataset_root_dir / "latest_run.txt", "w", encoding="utf-8") as f:
        f.write(str(run_dir))

    device_info = get_device_info()

    print("=" * 60)
    print("RUN DATASET:", dataset_name)
    print("STEP:", step)
    print("RUN ID:", run_id)
    print("OUTPUT DIR:", run_dir)
    print("Using device:", device_info["device"])

    if device_info["gpu_name"]:
        print("GPU name:", device_info["gpu_name"])

    print("=" * 60)

    if step in ["all", "preprocess"]:
        df = load_dataset(config)
        preprocess_dataset(df, config, str(processed_dir))

    if step in ["all", "baseline"]:
        train_baseline(str(processed_dir), str(baseline_dir))

    if step in ["all", "graph"]:
        build_graph(
            str(processed_dir),
            str(graph_dir),
            k_neighbors=config.get("k_neighbors", 5)
        )

    if step in ["all", "gnn"]:
        train_gnn(
            str(graph_dir),
            str(gnn_dir),
            epochs=gnn_params["epochs"],
            hidden_dim=gnn_params["hidden_dim"],
            lr=gnn_params["lr"]
        )

    if step in ["all", "compare"]:
        compare_results(str(baseline_dir), str(gnn_dir), str(report_dir))

    if step in ["all", "plot"]:
        plot_results(str(report_dir))

    if step in ["all", "select_best"]:
        select_best_model(
            dataset_name=dataset_name,
            run_dir=str(run_dir),
            best_model_dir=str(BASE_ARTIFACT_DIR / "best_models")
        )

    print("DONE:", dataset_name, step)
    print("RUN ID:", run_id)
    print("RESULT SAVED TO:", run_dir)


if __name__ == "__main__":
    main()