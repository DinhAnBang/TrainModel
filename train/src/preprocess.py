import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder


def clean_dataframe(X):
    X = X.copy()

    for col in X.columns:
        X[col] = X[col].astype(str)
        X[col] = X[col].str.replace(",", "", regex=False)
        X[col] = X[col].str.replace("%", "", regex=False)
        X[col] = X[col].str.strip()
        X[col] = X[col].str.lower()
        X[col] = X[col].replace(["nan", "none", "na", ""], np.nan)

    return X


def detect_column_types(X_train, numeric_threshold=0.8):
    numeric_cols = []
    categorical_cols = []

    for col in X_train.columns:
        numeric_version = pd.to_numeric(X_train[col], errors="coerce")
        numeric_ratio = numeric_version.notna().mean()

        if numeric_ratio >= numeric_threshold:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def preprocess_dataset(df, config, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    label_col = config["label_col"]

    if label_col not in df.columns:
        raise ValueError(f"Không tìm thấy cột label: {label_col}")

    df = df.drop_duplicates()

    y = df[label_col].astype(int)

    drop_cols = [label_col, "source_file"]

    for col in config.get("drop_cols", []):
        if col in df.columns:
            drop_cols.append(col)

    X = df.drop(columns=drop_cols, errors="ignore")
    X = clean_dataframe(X)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y.values,
        test_size=0.2,
        random_state=42,
        stratify=y.values
    )

    numeric_cols, categorical_cols = detect_column_types(X_train_raw)

    print("Numeric columns:", numeric_cols)
    print("Categorical columns:", categorical_cols)

    X_train_parts = []
    X_test_parts = []
    all_nan_cols = []

    if len(numeric_cols) > 0:
        X_train_num = X_train_raw[numeric_cols].apply(
            lambda col: pd.to_numeric(col, errors="coerce")
        )
        X_test_num = X_test_raw[numeric_cols].apply(
            lambda col: pd.to_numeric(col, errors="coerce")
        )

        for col in numeric_cols:
            if X_train_num[col].isna().all():
                all_nan_cols.append(col)

        X_train_num = X_train_num.drop(columns=all_nan_cols, errors="ignore")
        X_test_num = X_test_num.drop(columns=all_nan_cols, errors="ignore")

        medians = X_train_num.median(numeric_only=True)

        X_train_num = X_train_num.fillna(medians).fillna(0)
        X_test_num = X_test_num.fillna(medians).fillna(0)

        X_train_parts.append(X_train_num)
        X_test_parts.append(X_test_num)

    encoder = None

    if len(categorical_cols) > 0:
        X_train_cat = X_train_raw[categorical_cols].fillna("missing")
        X_test_cat = X_test_raw[categorical_cols].fillna("missing")

        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )

        X_train_cat_encoded = encoder.fit_transform(X_train_cat)
        X_test_cat_encoded = encoder.transform(X_test_cat)

        cat_feature_names = [f"{col}_encoded" for col in categorical_cols]

        X_train_cat_df = pd.DataFrame(
            X_train_cat_encoded,
            columns=cat_feature_names,
            index=X_train_raw.index
        )

        X_test_cat_df = pd.DataFrame(
            X_test_cat_encoded,
            columns=cat_feature_names,
            index=X_test_raw.index
        )

        X_train_parts.append(X_train_cat_df)
        X_test_parts.append(X_test_cat_df)

    if len(X_train_parts) == 0:
        raise ValueError("Không còn feature nào sau preprocess")

    X_train_final = pd.concat(X_train_parts, axis=1)
    X_test_final = pd.concat(X_test_parts, axis=1)

    constant_cols = X_train_final.columns[
        X_train_final.nunique() <= 1
    ].tolist()

    X_train_final = X_train_final.drop(columns=constant_cols)
    X_test_final = X_test_final.drop(columns=constant_cols)

    feature_names = X_train_final.columns.tolist()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_final)
    X_test_scaled = scaler.transform(X_test_final)

    np.save(os.path.join(output_dir, "X_train.npy"), X_train_scaled)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test_scaled)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)

    with open(os.path.join(output_dir, "feature_names.txt"), "w", encoding="utf-8") as f:
        for col in feature_names:
            f.write(col + "\n")

    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

    if encoder is not None:
        joblib.dump(encoder, os.path.join(output_dir, "categorical_encoder.pkl"))

    preprocess_info = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "feature_names": feature_names,
        "drop_cols": drop_cols,
        "all_nan_cols_removed": all_nan_cols,
        "constant_cols_removed": constant_cols
    }

    joblib.dump(preprocess_info, os.path.join(output_dir, "preprocess_info.pkl"))

    summary = {
        "final_feature_shape": int(X_train_scaled.shape[0] + X_test_scaled.shape[0]),
        "final_feature_dim": int(X_train_scaled.shape[1]),
        "train_shape": list(X_train_scaled.shape),
        "test_shape": list(X_test_scaled.shape),
        "feature_count": int(len(feature_names)),
        "label_distribution": {
            str(k): int(v) for k, v in zip(*np.unique(y.values, return_counts=True))
        },
        "drop_cols": drop_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "all_nan_cols_removed": all_nan_cols,
        "constant_cols_removed": constant_cols
    }

    with open(os.path.join(output_dir, "preprocess_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("Preprocess summary:", summary)

    return X_train_scaled, X_test_scaled, y_train, y_test