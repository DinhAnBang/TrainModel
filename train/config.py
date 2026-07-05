from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR.parent / "Processed_datasets"

COMMON_NETWORK_DROP_COLS = [
    "type", "ts",
    "uid", "src_ip", "dst_ip",
    "http_uri", "dns_query", "http_user_agent"
]

DATASETS = {
    "windows": {
        "data_dir": str(DATASET_DIR / "Processed_Windows_dataset"),
        "files": ["windows7_dataset.csv", "windows10_dataset.csv"],
        "label_col": "label",
        "drop_cols": ["type", "ts"],
        "use_common_columns": True,
        "k_neighbors": 10,
        "sample_size": None,
        "epochs": 100,
        "hidden_dim": 64,
        "lr": 0.01
    },

    "network": {
        "data_dir": str(DATASET_DIR / "Processed_Network_dataset"),
        "files": None,
        "label_col": "label",
        "drop_cols": COMMON_NETWORK_DROP_COLS,
        "use_common_columns": True,
        "k_neighbors": 5,
        "epochs": 100,
        "hidden_dim": 64,
        "sample_size": 500000,
        "lr": 0.01
    },

    "network_sample": {
        "data_dir": str(DATASET_DIR / "Processed_Network_dataset"),
        "files": None,
        "label_col": "label",
        "drop_cols": COMMON_NETWORK_DROP_COLS,
        "use_common_columns": True,
        "k_neighbors": 5,
        "sample_size": 100000,
        "epochs": 100,
        "hidden_dim": 64,
        "lr": 0.01
    }
}