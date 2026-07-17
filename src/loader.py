import fnmatch
import os
import re
from pathlib import Path

import pandas as pd


def _natural_key(path: str):
    """Sắp xếp Network_dataset_2 trước Network_dataset_10."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", Path(path).name)]


def _all_csv_files(data_dir: str) -> list[str]:
    files = sorted((str(path) for path in Path(data_dir).glob("*.csv")), key=_natural_key)
    if not files:
        raise FileNotFoundError(f"Không tìm thấy CSV trong: {data_dir}")
    return files


def _match_patterns(files: list[str], patterns: list[str]) -> list[str]:
    matched = []
    for path in files:
        name = Path(path).name.lower()
        if any(fnmatch.fnmatch(name, pattern.lower()) for pattern in patterns):
            matched.append(path)
    return sorted(dict.fromkeys(matched), key=_natural_key)


def list_dataset_experiments(config: dict, only_file: str | None = None) -> list[dict]:
    """Trả về danh sách đơn vị thực nghiệm.

    Mỗi phần tử có dạng:
        {"name": "experiment_name", "files": [csv_1, csv_2, ...]}

    Hỗ trợ ba chiến lược trong config:
    - per_file: mỗi CSV là một experiment riêng.
    - groups: mỗi nhóm pattern là một experiment và các CSV trong nhóm được gộp.
    - all: gộp toàn bộ CSV thành một experiment.
    """
    data_dir = config["data_dir"]

    if only_file:
        path = only_file if os.path.isabs(only_file) else os.path.join(data_dir, only_file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy file:\n{path}")
        return [{"name": Path(path).stem, "files": [path]}]

    configured_files = config.get("files")
    if configured_files:
        all_files = [os.path.join(data_dir, name) for name in configured_files]
        missing = [path for path in all_files if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError("Không tìm thấy file:\n" + "\n".join(missing))
    else:
        all_files = _all_csv_files(data_dir)

    strategy = config.get("experiment_strategy", "per_file")

    if strategy == "per_file":
        return [{"name": Path(path).stem, "files": [path]} for path in all_files]

    if strategy == "all":
        return [{
            "name": config.get("combined_experiment_name", "combined_dataset"),
            "files": all_files,
        }]

    if strategy == "groups":
        experiments = []
        used_files = set()
        for group in config.get("experiment_groups", []):
            if group.get("files"):
                group_files = [os.path.join(data_dir, name) for name in group["files"]]
                missing = [path for path in group_files if not os.path.exists(path)]
                if missing:
                    raise FileNotFoundError(
                        f"Thiếu CSV cho nhóm '{group['name']}':\n" + "\n".join(missing)
                    )
            else:
                group_files = _match_patterns(all_files, group.get("patterns", []))

            if not group_files:
                if group.get("required", True):
                    raise FileNotFoundError(
                        f"Không tìm thấy CSV cho nhóm '{group['name']}' với patterns: "
                        f"{group.get('patterns', [])}\nTrong thư mục: {data_dir}"
                    )
                continue
            experiments.append({"name": group["name"], "files": group_files})
            used_files.update(group_files)

        if config.get("include_ungrouped_files", False):
            for path in all_files:
                if path not in used_files:
                    experiments.append({"name": Path(path).stem, "files": [path]})

        if not experiments:
            raise FileNotFoundError(f"Không tạo được experiment nào từ: {data_dir}")
        return experiments

    raise ValueError(f"experiment_strategy không hợp lệ: {strategy}")


def _stratified_sample(df: pd.DataFrame, n: int, label_col: str, random_state: int) -> pd.DataFrame:
    if n is None or len(df) <= n:
        return df
    if label_col not in df.columns or df[label_col].nunique() < 2:
        return df.sample(n=n, random_state=random_state).reset_index(drop=True)

    parts = []
    proportions = df[label_col].value_counts(normalize=True)
    allocated = 0
    for index, (label, proportion) in enumerate(proportions.items()):
        group = df[df[label_col] == label]
        take = n - allocated if index == len(proportions) - 1 else max(1, round(n * proportion))
        take = min(take, len(group), n - allocated)
        if take > 0:
            parts.append(group.sample(n=take, random_state=random_state))
            allocated += take
    sampled = pd.concat(parts, ignore_index=True)
    return sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)


def load_csv_group(file_paths: list[str], config: dict) -> pd.DataFrame:
    """Đọc một hoặc nhiều CSV và gộp theo hàng.

    pandas tự căn chỉnh theo tên cột. Cột chỉ xuất hiện ở một file sẽ nhận NaN ở
    các file còn lại và tiếp tục được xử lý bởi preprocess (drop_cols/all-NaN/constant).
    """
    frames = []
    schemas = {}
    for file_path in file_paths:
        print("Loading:", file_path)
        frame = pd.read_csv(file_path, low_memory=False)
        schemas[Path(file_path).name] = list(frame.columns)
        print(f"CSV: {Path(file_path).name} | shape={frame.shape}")
        frames.append(frame)

    if len(frames) == 1:
        df = frames[0]
    else:
        common_cols = set(frames[0].columns)
        union_cols = set(frames[0].columns)
        for frame in frames[1:]:
            common_cols &= set(frame.columns)
            union_cols |= set(frame.columns)
        print(
            f"Merging {len(frames)} CSV files | common_columns={len(common_cols)} | "
            f"union_columns={len(union_cols)}"
        )
        df = pd.concat(frames, ignore_index=True, sort=False)

    original_shape = df.shape
    sample_size = config.get("sample_size")
    if sample_size is not None:
        df = _stratified_sample(
            df,
            int(sample_size),
            config["label_col"],
            int(config.get("random_state", 42)),
        )
    print(f"Combined dataset | original={original_shape} | loaded={df.shape}")
    return df
