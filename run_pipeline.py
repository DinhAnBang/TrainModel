import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from config import DATASETS
from src.baseline import train_baseline
from src.compare import compare_results
from src.deep_train import train_deep_models
from src.gnn_train import train_gnn
from src.graph_builder import build_graph
from src.loader import list_dataset_experiments, load_csv_group
from src.plot_results import plot_results
from src.preprocess import preprocess_dataset
from src.select_best_model import select_best_model
from src.utils import safe_name

PROJECT_DIR = Path(__file__).resolve().parent
BASE_ARTIFACT_DIR = PROJECT_DIR / "artifacts"
ALL_MODELS = ["random_forest", "xgboost", "cnn", "resnet", "gcn"]
ML_MODELS = {"random_forest", "xgboost"}
DEEP_MODELS = {"cnn", "resnet"}


def save_run_config(run_dir, platform_name, experiment, step, config, models):
    info = {
        "platform": platform_name,
        "experiment": experiment["name"],
        "csv_files": [str(path) for path in experiment["files"]],
        "step": step,
        "models": models,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": config,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)


def run_one_experiment(platform_name, experiment, step, config, models, run_id=None):
    experiment_name = safe_name(experiment["name"])
    current_run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_root = BASE_ARTIFACT_DIR / "history" / platform_name / experiment_name
    run_dir = experiment_root / current_run_id
    processed_dir = run_dir / "processed"
    ml_dir = run_dir / "ml"
    deep_dir = run_dir / "deep"
    graph_dir = run_dir / "graph"
    gnn_dir = run_dir / "gnn"
    report_dir = run_dir / "reports"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(run_dir, platform_name, experiment, step, config, models)
    (experiment_root / "latest_run.txt").write_text(str(run_dir), encoding="utf-8")

    print("\n" + "=" * 100)
    print("PLATFORM:", platform_name)
    print("EXPERIMENT:", experiment_name)
    print("CSV FILES:")
    for path in experiment["files"]:
        print("  -", path)
    print("RUN DIR:", run_dir)
    print("MODELS:", models)
    print("=" * 100)

    requested = set(models)

    if step in ["all", "preprocess"]:
        df = load_csv_group(experiment["files"], config)
        preprocess_dataset(df, config, str(processed_dir), source_files=experiment["files"])

    if step in ["all", "ml"] and requested & ML_MODELS:
        train_baseline(str(processed_dir), str(ml_dir), config, models)

    if step in ["all", "deep"] and requested & DEEP_MODELS:
        train_deep_models(str(processed_dir), str(deep_dir), config, models)

    if step in ["all", "graph"] and "gcn" in requested:
        build_graph(str(processed_dir), str(graph_dir), config.get("k_neighbors", 10))

    if step in ["all", "gnn"] and "gcn" in requested:
        train_gnn(str(graph_dir), str(gnn_dir), config)

    if step in ["all", "compare"]:
        compare_results(str(ml_dir), str(deep_dir), str(gnn_dir), str(report_dir))

    if step in ["all", "plot"]:
        plot_results(str(report_dir))

    if step in ["all", "select_best"]:
        select_best_model(experiment_name, str(run_dir), str(BASE_ARTIFACT_DIR / "best_models"))

    print("DONE:", experiment_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASETS.keys())
    parser.add_argument(
        "--file",
        default=None,
        help="Chạy riêng một CSV. Khi bỏ trống, pipeline dùng chiến lược nhóm trong config.",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Chạy riêng một experiment, ví dụ: linux_process hoặc IoT_Fridge.",
    )
    parser.add_argument(
        "--step",
        default="all",
        choices=["all", "preprocess", "ml", "deep", "graph", "gnn", "compare", "plot", "select_best"],
    )
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument("--run_id", default=None)
    args = parser.parse_args()

    config = DATASETS[args.dataset]
    experiments = list_dataset_experiments(config, args.file)

    if args.experiment:
        target = safe_name(args.experiment)
        experiments = [exp for exp in experiments if safe_name(exp["name"]) == target]
        if not experiments:
            available = ", ".join(exp["name"] for exp in list_dataset_experiments(config))
            raise ValueError(
                f"Không tìm thấy experiment '{args.experiment}'. Có thể chọn: {available}"
            )

    for index, experiment in enumerate(experiments):
        run_id = args.run_id
        if run_id and len(experiments) > 1:
            run_id = f"{run_id}_{index + 1:02d}"
        run_one_experiment(args.dataset, experiment, args.step, config, args.models, run_id)


if __name__ == "__main__":
    main()
