import os
import glob
import pandas as pd


def load_dataset(config):
    data_dir = config["data_dir"]
    files = config.get("files")
    sample_size = config.get("sample_size")

    if files is None:
        file_paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    else:
        file_paths = [os.path.join(data_dir, f) for f in files]

    if len(file_paths) == 0:
        raise FileNotFoundError(f"Không tìm thấy CSV trong: {data_dir}")

    dfs = []

    per_file_sample = None
    if sample_size is not None:
        per_file_sample = max(sample_size // len(file_paths) + 1, 1)

    for path in file_paths:
        name = os.path.basename(path).replace(".csv", "")
        print("Loading:", path)

        df = pd.read_csv(path, low_memory=False)

        if per_file_sample is not None and len(df) > per_file_sample:
            df = df.sample(per_file_sample, random_state=42)

        df["source_file"] = name
        dfs.append(df)

    if config.get("use_common_columns", False) and len(dfs) > 1:
        common_cols = set(dfs[0].columns)

        for df in dfs[1:]:
            common_cols = common_cols.intersection(set(df.columns))

        common_cols = sorted(list(common_cols))
        dfs = [df[common_cols].copy() for df in dfs]

        print("Common columns:", len(common_cols))

    df_all = pd.concat(dfs, ignore_index=True)

    if sample_size is not None and len(df_all) > sample_size:
        df_all = df_all.sample(sample_size, random_state=42)

    print("Final loaded shape:", df_all.shape)

    return df_all