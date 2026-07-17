import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler


def _is_iot_experiment(config: dict, source_files=None) -> bool:
    """Chỉ bật feature engineering thời gian cho dữ liệu IoT.

    Ưu tiên flag rõ ràng trong config:
        "enable_datetime_features": True

    Nếu chưa có flag, tự nhận diện qua platform/dataset name hoặc đường dẫn nguồn.
    """
    if bool(config.get("enable_datetime_features", False)):
        return True

    for key in ["platform", "dataset", "dataset_name", "name"]:
        value = str(config.get(key, "")).lower()
        if value == "iot" or value.startswith("iot_"):
            return True

    return any("processed_iot_dataset" in str(path).lower() for path in (source_files or []))


def _add_iot_datetime_features(df: pd.DataFrame, config: dict, source_files=None):
    """Tạo đặc trưng thời gian cho IoT trước khi drop date/time.

    Tạo:
    - hour, minute, second, weekday, day
    - hour_sin/hour_cos và weekday_sin/weekday_cos để biểu diễn tính chu kỳ

    Không thay đổi Windows/Linux/Network.
    """
    df = df.copy()
    created_features = []

    if not _is_iot_experiment(config, source_files):
        return df, created_features

    date_col = config.get("date_col", "date")
    time_col = config.get("time_col", "time")

    if date_col not in df.columns and time_col not in df.columns:
        return df, created_features

    if date_col in df.columns and time_col in df.columns:
        datetime_text = (
            df[date_col].astype(str).str.strip()
            + " "
            + df[time_col].astype(str).str.strip()
        )
        dt = pd.to_datetime(datetime_text, errors="coerce", dayfirst=True)
    elif date_col in df.columns:
        dt = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    else:
        dt = pd.to_datetime(df[time_col], errors="coerce")

    parse_ratio = float(dt.notna().mean())
    print(f"[IoT datetime] parse_success={parse_ratio:.2%}")

    if parse_ratio == 0:
        print("[IoT datetime] Không parse được date/time, bỏ qua feature engineering.")
        return df, created_features

    df["iot_hour"] = dt.dt.hour.fillna(0).astype(np.int16)
    df["iot_minute"] = dt.dt.minute.fillna(0).astype(np.int16)
    df["iot_second"] = dt.dt.second.fillna(0).astype(np.int16)
    df["iot_weekday"] = dt.dt.dayofweek.fillna(0).astype(np.int16)
    df["iot_day"] = dt.dt.day.fillna(0).astype(np.int16)

    # Đặc trưng chu kỳ: 23 giờ gần 0 giờ hơn so với 12 giờ.
    df["iot_hour_sin"] = np.sin(2 * np.pi * df["iot_hour"] / 24.0)
    df["iot_hour_cos"] = np.cos(2 * np.pi * df["iot_hour"] / 24.0)
    df["iot_weekday_sin"] = np.sin(2 * np.pi * df["iot_weekday"] / 7.0)
    df["iot_weekday_cos"] = np.cos(2 * np.pi * df["iot_weekday"] / 7.0)

    created_features = [
        "iot_hour",
        "iot_minute",
        "iot_second",
        "iot_weekday",
        "iot_day",
        "iot_hour_sin",
        "iot_hour_cos",
        "iot_weekday_sin",
        "iot_weekday_cos",
    ]
    print("[IoT datetime] Created features:", created_features)
    return df, created_features




def _is_linux_process_experiment(config: dict, source_files=None) -> bool:
    """Chỉ bật feature engineering chuyên biệt cho nhóm Linux Process."""
    if not bool(config.get("enable_linux_process_features", False)):
        return False

    source_names = [str(path).lower() for path in (source_files or [])]
    if source_names:
        return any("linux_process" in name for name in source_names)

    experiment_name = str(config.get("experiment_name", "")).lower()
    return not experiment_name or "linux_process" in experiment_name


def _add_linux_process_time_features(df: pd.DataFrame, config: dict, source_files=None):
    """Tạo đặc trưng thời gian từ ts cho Linux Process trước khi drop ts.

    Các feature chỉ phụ thuộc timestamp của từng bản ghi hoặc thứ tự thời gian,
    không sử dụng label.
    """
    df = df.copy()
    created_features = []

    if not _is_linux_process_experiment(config, source_files):
        return df, created_features

    ts_col = config.get("linux_timestamp_col", "ts")
    if ts_col not in df.columns:
        print(f"[Linux process datetime] Không tìm thấy cột {ts_col}, bỏ qua.")
        return df, created_features

    ts_numeric = pd.to_numeric(df[ts_col], errors="coerce")
    # ToN-IoT Linux Process dùng Unix timestamp theo giây.
    dt = pd.to_datetime(ts_numeric, unit="s", errors="coerce", utc=True)
    parse_ratio = float(dt.notna().mean())
    print(f"[Linux process datetime] parse_success={parse_ratio:.2%}")

    if parse_ratio == 0:
        print("[Linux process datetime] Không parse được ts, bỏ qua feature engineering.")
        return df, created_features

    df["linux_hour"] = dt.dt.hour.fillna(0).astype(np.int16)
    df["linux_minute"] = dt.dt.minute.fillna(0).astype(np.int16)
    df["linux_second"] = dt.dt.second.fillna(0).astype(np.int16)
    df["linux_weekday"] = dt.dt.dayofweek.fillna(0).astype(np.int16)

    df["linux_hour_sin"] = np.sin(2 * np.pi * df["linux_hour"] / 24.0)
    df["linux_hour_cos"] = np.cos(2 * np.pi * df["linux_hour"] / 24.0)
    df["linux_minute_sin"] = np.sin(2 * np.pi * df["linux_minute"] / 60.0)
    df["linux_minute_cos"] = np.cos(2 * np.pi * df["linux_minute"] / 60.0)
    df["linux_weekday_sin"] = np.sin(2 * np.pi * df["linux_weekday"] / 7.0)
    df["linux_weekday_cos"] = np.cos(2 * np.pi * df["linux_weekday"] / 7.0)

    valid_ts = ts_numeric.dropna()
    min_ts = float(valid_ts.min()) if not valid_ts.empty else 0.0
    df["linux_elapsed_time"] = (ts_numeric - min_ts).fillna(0).clip(lower=0)

    # Khoảng cách thời gian giữa hai bản ghi liên tiếp theo thứ tự timestamp.
    # Sau đó trả feature về đúng index gốc để không làm xáo trộn label/row.
    ordered = pd.DataFrame({"ts": ts_numeric, "original_index": df.index})
    ordered = ordered.sort_values(["ts", "original_index"], kind="mergesort")
    ordered["delta"] = ordered["ts"].diff().fillna(0).clip(lower=0)
    delta_by_index = ordered.set_index("original_index")["delta"]
    df["linux_time_since_previous_record"] = delta_by_index.reindex(df.index).fillna(0).to_numpy()

    created_features = [
        "linux_hour", "linux_minute", "linux_second", "linux_weekday",
        "linux_hour_sin", "linux_hour_cos",
        "linux_minute_sin", "linux_minute_cos",
        "linux_weekday_sin", "linux_weekday_cos",
        "linux_elapsed_time", "linux_time_since_previous_record",
    ]
    print("[Linux process datetime] Created features:", created_features)
    return df, created_features


def _apply_frequency_encoding(train, val, test, columns):
    """Fit frequency mapping trên train và áp dụng sang val/test."""
    encoded_parts = {"train": [], "val": [], "test": []}
    mappings = {}

    split_frames = {"train": train, "val": val, "test": test}
    for col in columns:
        if col not in train.columns:
            continue

        train_values = train[col].fillna("missing").astype(str)
        frequency = train_values.value_counts(normalize=True)
        mappings[col] = {str(k): float(v) for k, v in frequency.items()}

        for split_name, frame in split_frames.items():
            values = frame[col].fillna("missing").astype(str)
            encoded = values.map(frequency).fillna(0.0).astype(np.float32)
            encoded_parts[split_name].append(
                pd.DataFrame({f"{col}_frequency": encoded}, index=frame.index)
            )

    return encoded_parts, mappings


def clean_dataframe(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            continue
        X[col] = X[col].astype(str).str.replace(",", "", regex=False)
        X[col] = X[col].str.replace("%", "", regex=False).str.strip().str.lower()
        X[col] = X[col].replace(["nan", "none", "na", "null", ""], np.nan)
    return X


def detect_column_types(X_train: pd.DataFrame, numeric_threshold: float = 0.8):
    numeric_cols, categorical_cols = [], []
    for col in X_train.columns:
        converted = pd.to_numeric(X_train[col], errors="coerce")
        if converted.notna().mean() >= numeric_threshold:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols


def _split_data(X, y, test_size, val_size, random_state):
    stratify = y if len(np.unique(y)) > 1 else None
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    relative_val = val_size / (1.0 - test_size)
    stratify_train_val = y_train_val if len(np.unique(y_train_val)) > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_val,
        random_state=random_state,
        stratify=stratify_train_val,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_dataset(df: pd.DataFrame, config: dict, output_dir: str, source_files=None):
    os.makedirs(output_dir, exist_ok=True)
    eda_dir = os.path.join(output_dir, "eda")
    os.makedirs(eda_dir, exist_ok=True)

    label_col = config["label_col"]
    if label_col not in df.columns:
        raise ValueError(f"Không tìm thấy cột nhãn: {label_col}")

    rows_before = len(df)
    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    rows_after = len(df)

    # IoT: khai thác date/time thành feature trước khi drop các cột gốc.
    df, iot_datetime_features = _add_iot_datetime_features(
        df,
        config,
        source_files=source_files,
    )

    # Linux Process: khai thác ts thành feature trước khi drop ts.
    df, linux_datetime_features = _add_linux_process_time_features(
        df,
        config,
        source_files=source_files,
    )
    datetime_features_created = iot_datetime_features + linux_datetime_features

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[label_col].astype(str))

    drop_cols = [label_col]
    drop_cols.extend([col for col in config.get("drop_cols", []) if col in df.columns])
    drop_cols = list(dict.fromkeys(drop_cols))

    X = clean_dataframe(df.drop(columns=drop_cols, errors="ignore"))
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = _split_data(
        X,
        y,
        float(config.get("test_size", 0.2)),
        float(config.get("val_size", 0.15)),
        int(config.get("random_state", 42)),
    )

    numeric_cols, categorical_cols = detect_column_types(
        X_train_raw,
        numeric_threshold=float(config.get("numeric_detection_threshold", 0.8)),
    )
    all_nan_cols = []
    medians = {}
    parts = {"train": [], "val": [], "test": []}

    if numeric_cols:
        converted = {}
        for split_name, frame in [("train", X_train_raw), ("val", X_val_raw), ("test", X_test_raw)]:
            converted[split_name] = frame[numeric_cols].apply(pd.to_numeric, errors="coerce")
        all_nan_cols = [col for col in numeric_cols if converted["train"][col].isna().all()]
        usable_numeric = [col for col in numeric_cols if col not in all_nan_cols]
        train_num = converted["train"][usable_numeric]
        median_series = train_num.median().fillna(0)
        medians = {str(k): float(v) for k, v in median_series.items()}
        for split_name in parts:
            parts[split_name].append(converted[split_name][usable_numeric].fillna(median_series).fillna(0))
    else:
        usable_numeric = []

    encoder = None
    onehot_encoder = None
    frequency_mappings = {}
    linux_process_mode = _is_linux_process_experiment(config, source_files)

    if linux_process_mode:
        # CMD có cardinality cao: frequency encoding fit chỉ trên train.
        frequency_cols = [
            col for col in config.get("linux_frequency_cols", ["CMD"])
            if col in X_train_raw.columns
        ]
        if bool(config.get("linux_add_pid_frequency", True)) and "PID" in X_train_raw.columns:
            frequency_cols.append("PID")
        frequency_cols = list(dict.fromkeys(frequency_cols))

        frequency_parts, frequency_mappings = _apply_frequency_encoding(
            X_train_raw, X_val_raw, X_test_raw, frequency_cols
        )
        for split_name in parts:
            parts[split_name].extend(frequency_parts[split_name])

        # Category ít giá trị: one-hot để không tạo thứ tự giả như OrdinalEncoder.
        onehot_cols = [
            col for col in config.get("linux_onehot_cols", ["Status", "State", "POLI"])
            if col in X_train_raw.columns
        ]
        if onehot_cols:
            try:
                onehot_encoder = OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float32,
                )
            except TypeError:
                onehot_encoder = OneHotEncoder(
                    handle_unknown="ignore",
                    sparse=False,
                    dtype=np.float32,
                )

            train_onehot = X_train_raw[onehot_cols].fillna("missing").astype(str)
            val_onehot = X_val_raw[onehot_cols].fillna("missing").astype(str)
            test_onehot = X_test_raw[onehot_cols].fillna("missing").astype(str)
            onehot_encoder.fit(train_onehot)
            onehot_names = onehot_encoder.get_feature_names_out(onehot_cols).tolist()

            for split_name, frame in [
                ("train", train_onehot),
                ("val", val_onehot),
                ("test", test_onehot),
            ]:
                encoded = onehot_encoder.transform(frame)
                parts[split_name].append(
                    pd.DataFrame(encoded, columns=onehot_names, index=frame.index)
                )

        # Các category còn lại vẫn ordinal để giữ tương thích chung.
        excluded = set(frequency_cols) | set(onehot_cols)
        remaining_categorical = [col for col in categorical_cols if col not in excluded]
    else:
        remaining_categorical = categorical_cols

    if remaining_categorical:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        train_cat = X_train_raw[remaining_categorical].fillna("missing").astype(str)
        val_cat = X_val_raw[remaining_categorical].fillna("missing").astype(str)
        test_cat = X_test_raw[remaining_categorical].fillna("missing").astype(str)
        encoder.fit(train_cat)
        cat_names = [f"{col}_encoded" for col in remaining_categorical]
        for split_name, frame in [("train", train_cat), ("val", val_cat), ("test", test_cat)]:
            encoded = encoder.transform(frame)
            parts[split_name].append(pd.DataFrame(encoded, columns=cat_names, index=frame.index))

    if not parts["train"]:
        raise ValueError("Không còn feature hợp lệ sau preprocess")

    X_train = pd.concat(parts["train"], axis=1)
    X_val = pd.concat(parts["val"], axis=1)
    X_test = pd.concat(parts["test"], axis=1)

    constant_cols = X_train.columns[X_train.nunique(dropna=False) <= 1].tolist()
    X_train = X_train.drop(columns=constant_cols)
    X_val = X_val.drop(columns=constant_cols)
    X_test = X_test.drop(columns=constant_cols)

    selected_features = X_train.columns.tolist()
    feature_importance_df = pd.DataFrame(columns=["feature", "importance"])
    if config.get("feature_selection", True) and X_train.shape[1] > 1:
        selector = RandomForestClassifier(
            n_estimators=min(150, int(config.get("rf_estimators", 200))),
            class_weight="balanced",
            random_state=int(config.get("random_state", 42)),
            n_jobs=-1,
        )
        selector.fit(X_train, y_train)
        feature_importance_df = pd.DataFrame({
            "feature": X_train.columns,
            "importance": selector.feature_importances_,
        }).sort_values("importance", ascending=False)
        max_features = config.get("max_features")
        if max_features:
            selected_features = feature_importance_df.head(int(max_features))["feature"].tolist()
        else:
            positive = feature_importance_df[feature_importance_df["importance"] > 0]["feature"].tolist()
            selected_features = positive if positive else X_train.columns.tolist()
        X_train = X_train[selected_features]
        X_val = X_val[selected_features]
        X_test = X_test[selected_features]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    for name, array in {
        "X_train_raw.npy": X_train.to_numpy(dtype=np.float32),
        "X_val_raw.npy": X_val.to_numpy(dtype=np.float32),
        "X_test_raw.npy": X_test.to_numpy(dtype=np.float32),
        "X_train_scaled.npy": X_train_scaled,
        "X_val_scaled.npy": X_val_scaled,
        "X_test_scaled.npy": X_test_scaled,
        "y_train.npy": y_train.astype(np.int64),
        "y_val.npy": y_val.astype(np.int64),
        "y_test.npy": y_test.astype(np.int64),
    }.items():
        np.save(os.path.join(output_dir, name), array)

    # Tương thích tạm thời với code cũ.
    np.save(os.path.join(output_dir, "X_train.npy"), X_train_scaled)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test_scaled)

    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.pkl"))
    if encoder is not None:
        joblib.dump(encoder, os.path.join(output_dir, "categorical_encoder.pkl"))
    if onehot_encoder is not None:
        joblib.dump(onehot_encoder, os.path.join(output_dir, "linux_onehot_encoder.pkl"))
    if frequency_mappings:
        joblib.dump(frequency_mappings, os.path.join(output_dir, "linux_frequency_mappings.pkl"))

    preprocess_info = {
        "label_col": label_col,
        "drop_cols": drop_cols,
        "numeric_cols": usable_numeric,
        "categorical_cols": categorical_cols,
        "linux_process_mode": linux_process_mode,
        "linux_frequency_mappings": frequency_mappings,
        "all_nan_cols_removed": all_nan_cols,
        "constant_cols_removed": constant_cols,
        "feature_names": selected_features,
        "numeric_medians": medians,
        "classes": label_encoder.classes_.tolist(),
        "datetime_features_created": datetime_features_created,
    }
    joblib.dump(preprocess_info, os.path.join(output_dir, "preprocess_info.pkl"))

    with open(os.path.join(output_dir, "feature_names.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(selected_features))

    missing_df = df.isna().sum().sort_values(ascending=False).rename("missing_count").reset_index()
    missing_df.columns = ["column", "missing_count"]
    missing_df["missing_ratio"] = missing_df["missing_count"] / max(len(df), 1)
    missing_df.to_csv(os.path.join(eda_dir, "missing_values.csv"), index=False)

    class_df = pd.DataFrame({"class": label_encoder.classes_, "count": np.bincount(y)})
    class_df["ratio"] = class_df["count"] / class_df["count"].sum()
    class_df.to_csv(os.path.join(eda_dir, "class_distribution.csv"), index=False)

    numeric_eda = df.drop(columns=[label_col], errors="ignore").select_dtypes(include=np.number)
    if not numeric_eda.empty:
        stats = pd.DataFrame({
            "feature": numeric_eda.columns,
            "skewness": numeric_eda.skew(numeric_only=True).values,
            "kurtosis": numeric_eda.kurtosis(numeric_only=True).values,
        })
        stats.to_csv(os.path.join(eda_dir, "distribution_statistics.csv"), index=False)
        numeric_eda.corr(method="pearson").to_csv(os.path.join(eda_dir, "correlation_matrix.csv"))
    feature_importance_df.to_csv(os.path.join(eda_dir, "feature_importance.csv"), index=False)

    summary = {
        "source_files": [str(path) for path in (source_files or [])],
        "source_file_count": len(source_files or []),
        "rows_before_deduplication": rows_before,
        "duplicate_rows_removed": duplicate_count,
        "rows_after_deduplication": rows_after,
        "original_column_count": int(df.shape[1]),
        "feature_count_after_preprocess": int(len(selected_features)),
        "train_shape": list(X_train_scaled.shape),
        "val_shape": list(X_val_scaled.shape),
        "test_shape": list(X_test_scaled.shape),
        "class_distribution": {str(cls): int(count) for cls, count in zip(label_encoder.classes_, np.bincount(y))},
        "drop_cols": drop_cols,
        "all_nan_cols_removed": all_nan_cols,
        "constant_cols_removed": constant_cols,
        "selected_features": selected_features,
        "datetime_features_created": datetime_features_created,
    }
    with open(os.path.join(output_dir, "preprocess_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("Preprocess summary:", summary)
    return summary