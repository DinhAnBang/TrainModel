from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "Processed_datasets"

COMMON_NETWORK_DROP_COLS = [
    "type", "ts", "uid",
    "src_ip", "dst_ip", "src_port", "dst_port", "sport", "dport",
    "http_uri", "dns_query", "http_user_agent"
]

COMMON_TRAINING = {
    "label_col": "label",
    "sample_size": None,
    "test_size": 0.20,
    "val_size": 0.15,
    "random_state": 42,
    "numeric_detection_threshold": 0.8,
    "feature_selection": True,
    "max_features": None,
    "rf_estimators": 200,
    "xgb_estimators": 300,
    "batch_size": 512,
    "epochs": 100,
    "hidden_dim": 64,
    "lr": 0.001,
    "k_neighbors": 10,
}

DATASETS = {
    "windows": {
        **COMMON_TRAINING,
        "data_dir": str(DATASET_DIR / "Processed_Windows_dataset"),
        "files": ["windows7_dataset.csv", "windows10_dataset.csv"],
        "experiment_strategy": "per_file",
        "drop_cols": ["type", "ts"],
    },

    # Linux: gộp đúng theo từng loại telemetry trước khi EDA/preprocess/train.
    "linux": {
        **COMMON_TRAINING,
        "data_dir": str(DATASET_DIR / "Processed_Linux_dataset"),
        "files": None,
        "epochs": 30,
        "batch_size": 1024,
        "experiment_strategy": "groups",
        "experiment_groups": [
            {
                "name": "linux_process",
                "files": ["Linux_process_1.csv", "Linux_process_2.csv"],
            },
            {
                "name": "linux_disk",
                "files": ["linux_disk_1.csv", "linux_disk_2.csv"],
            },
            {
                "name": "linux_memory",
                "files": ["linux_memory1.csv", "linux_memory2.csv"],
            },
        ],
        "include_ungrouped_files": False,
        "drop_cols": ["type", "ts"],
        "enable_linux_process_features": True,
        "linux_timestamp_col": "ts",
        "linux_frequency_cols": ["CMD"],
        "linux_add_pid_frequency": True,
        "linux_onehot_cols": ["Status", "State", "POLI"],
    },

    "iot": {
        **COMMON_TRAINING,
        "data_dir": str(DATASET_DIR / "Processed_IoT_dataset"),
        "files": None,
        "experiment_strategy": "per_file",
        "drop_cols": ["type", "date", "time", "ts"],
        "epochs": 30,
        "batch_size": 1024,
    },

    # Network 1..23 có cùng schema hữu ích nên gộp tất cả trước khi preprocess/train.
    "network": {
        **COMMON_TRAINING,
        "data_dir": str(DATASET_DIR / "Processed_Network_dataset"),
        "files": None,
        "experiment_strategy": "all",
        "combined_experiment_name": "network_all_1_23",
        "drop_cols": COMMON_NETWORK_DROP_COLS,
        "sample_size": None,
        "epochs": 50,
        "k_neighbors": 5,
    },

    # Bản chạy thử nhanh nhưng vẫn gộp toàn bộ Network trước khi stratified sample.
    "network_sample": {
        **COMMON_TRAINING,
        "data_dir": str(DATASET_DIR / "Processed_Network_dataset"),
        "files": None,
        "experiment_strategy": "all",
        "combined_experiment_name": "network_all_1_23_sample",
        "drop_cols": COMMON_NETWORK_DROP_COLS,
        "sample_size": 100000,
        "rf_estimators": 150,
        "xgb_estimators": 200,
        "epochs": 30,
        "patience": 6,
        "k_neighbors": 5,
    },
}