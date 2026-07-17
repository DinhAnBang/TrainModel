import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


DEFAULT_FILES = [
    r"Processed_datasets\Processed_Linux_dataset\Linux_process_1.csv",
    r"Processed_datasets\Processed_Linux_dataset\Linux_process_2.csv",
]


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_to_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def entropy_from_counts(counts):
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    if len(counts) == 0:
        return np.nan
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def cramers_v(confusion):
    observed = confusion.to_numpy(dtype=float)
    n = observed.sum()
    if n == 0:
        return np.nan

    row_sum = observed.sum(axis=1, keepdims=True)
    col_sum = observed.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / n

    valid = expected > 0
    chi2 = (((observed - expected) ** 2) / np.where(valid, expected, 1))[valid].sum()

    r, k = observed.shape
    if min(r - 1, k - 1) <= 0:
        return 0.0

    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
    r_corr = r - ((r - 1) ** 2) / max(n - 1, 1)
    k_corr = k - ((k - 1) ** 2) / max(n - 1, 1)
    denom = min(k_corr - 1, r_corr - 1)
    return float(np.sqrt(phi2_corr / denom)) if denom > 0 else 0.0


def binary_numeric_auc(y, values):
    from sklearn.metrics import roc_auc_score

    mask = values.notna() & y.notna()
    if mask.sum() < 10 or y[mask].nunique() != 2:
        return np.nan

    x = values[mask].astype(float)
    if x.nunique() < 2:
        return 0.5

    try:
        auc = roc_auc_score(y[mask], x)
        return float(max(auc, 1.0 - auc))
    except Exception:
        return np.nan


def standardized_mean_difference(group0, group1):
    group0 = pd.to_numeric(group0, errors="coerce").dropna()
    group1 = pd.to_numeric(group1, errors="coerce").dropna()

    if len(group0) < 2 or len(group1) < 2:
        return np.nan

    pooled_var = (
        ((len(group0) - 1) * group0.var(ddof=1)
         + (len(group1) - 1) * group1.var(ddof=1))
        / max(len(group0) + len(group1) - 2, 1)
    )

    if pooled_var <= 0:
        return 0.0

    return float(abs(group1.mean() - group0.mean()) / np.sqrt(pooled_var))


def load_files(paths, sample_rows=None):
    frames = []
    file_summaries = []

    for file_path in paths:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {file_path.resolve()}")

        print(f"Loading: {file_path}")
        if sample_rows:
            df = pd.read_csv(file_path, nrows=sample_rows, low_memory=False)
        else:
            df = pd.read_csv(file_path, low_memory=False)

        df["source_file"] = file_path.name
        frames.append(df)

        file_summaries.append({
            "source_file": file_path.name,
            "rows": int(len(df)),
            "columns": int(df.shape[1] - 1),
        })

    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined, pd.DataFrame(file_summaries)


def basic_quality_report(df):
    rows = []

    for col in df.columns:
        s = df[col]
        unique_count = int(s.nunique(dropna=True))
        missing_count = int(s.isna().sum())
        missing_pct = missing_count / max(len(df), 1)

        rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "unique_count": unique_count,
            "unique_pct": unique_count / max(len(df), 1),
            "is_constant": unique_count <= 1,
            "is_id_like": unique_count / max(len(df), 1) >= 0.90,
        })

    return pd.DataFrame(rows)


def label_report(df, label_col):
    if label_col not in df.columns:
        raise KeyError(f"Không tìm thấy label column: {label_col}")

    counts = df[label_col].value_counts(dropna=False).sort_index()
    result = pd.DataFrame({
        "label": counts.index.astype(str),
        "count": counts.values,
        "ratio": counts.values / max(counts.sum(), 1),
    })
    return result


def source_label_report(df, label_col):
    if "source_file" not in df.columns:
        return pd.DataFrame()

    table = pd.crosstab(df["source_file"], df[label_col])
    table_ratio = pd.crosstab(df["source_file"], df[label_col], normalize="index")

    output = table.copy()
    output.columns = [f"count_label_{col}" for col in output.columns]

    for col in table_ratio.columns:
        output[f"ratio_label_{col}"] = table_ratio[col]

    return output.reset_index()


def duplicate_report(df):
    exact_duplicates = int(df.duplicated().sum())
    without_source = df.drop(columns=["source_file"], errors="ignore")
    content_duplicates = int(without_source.duplicated().sum())

    return {
        "rows": int(len(df)),
        "exact_duplicates_with_source": exact_duplicates,
        "duplicate_content_ignoring_source": content_duplicates,
        "duplicate_pct_ignoring_source": content_duplicates / max(len(df), 1),
    }


def numeric_feature_report(df, label_col):
    y = pd.to_numeric(df[label_col], errors="coerce")
    numeric_cols = []

    for col in df.columns:
        if col in {label_col, "source_file"}:
            continue

        converted = safe_to_numeric(df[col])
        valid_ratio = converted.notna().mean()

        if valid_ratio >= 0.80:
            numeric_cols.append((col, converted, valid_ratio))

    rows = []

    for col, values, valid_ratio in numeric_cols:
        g0 = values[y == 0]
        g1 = values[y == 1]

        rows.append({
            "feature": col,
            "numeric_valid_ratio": float(valid_ratio),
            "unique_count": int(values.nunique(dropna=True)),
            "missing_pct": float(values.isna().mean()),
            "mean_label_0": float(g0.mean()) if len(g0) else np.nan,
            "mean_label_1": float(g1.mean()) if len(g1) else np.nan,
            "median_label_0": float(g0.median()) if len(g0) else np.nan,
            "median_label_1": float(g1.median()) if len(g1) else np.nan,
            "std_label_0": float(g0.std()) if len(g0) else np.nan,
            "std_label_1": float(g1.std()) if len(g1) else np.nan,
            "standardized_mean_difference": standardized_mean_difference(g0, g1),
            "single_feature_auc": binary_numeric_auc(y, values),
            "zero_ratio": float((values == 0).mean()),
            "negative_ratio": float((values < 0).mean()),
        })

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            ["single_feature_auc", "standardized_mean_difference"],
            ascending=False
        )

    return result


def categorical_feature_report(df, label_col, max_cardinality=10000):
    rows = []

    for col in df.columns:
        if col in {label_col, "source_file"}:
            continue

        s = df[col]
        unique_count = s.nunique(dropna=True)

        converted = safe_to_numeric(s)
        numeric_valid_ratio = converted.notna().mean()

        looks_categorical = (
            s.dtype == "object"
            or unique_count <= 100
            or col.lower().endswith("_encoded")
        )

        if not looks_categorical:
            continue

        if unique_count > max_cardinality:
            rows.append({
                "feature": col,
                "unique_count": int(unique_count),
                "missing_pct": float(s.isna().mean()),
                "numeric_valid_ratio": float(numeric_valid_ratio),
                "cramers_v": np.nan,
                "label_purity_max": np.nan,
                "rare_category_ratio": np.nan,
                "note": f"Skipped association: cardinality>{max_cardinality}",
            })
            continue

        temp = pd.DataFrame({
            "feature_value": s.fillna("__MISSING__").astype(str),
            "label": df[label_col].astype(str),
        })

        confusion = pd.crosstab(temp["feature_value"], temp["label"])
        association = cramers_v(confusion)

        category_counts = temp["feature_value"].value_counts()
        rare_ratio = float(
            category_counts[category_counts < 10].sum() / max(len(temp), 1)
        )

        purity_by_category = confusion.max(axis=1) / confusion.sum(axis=1)
        weighted_purity = float(
            (purity_by_category * confusion.sum(axis=1)).sum()
            / max(confusion.to_numpy().sum(), 1)
        )

        rows.append({
            "feature": col,
            "unique_count": int(unique_count),
            "missing_pct": float(s.isna().mean()),
            "numeric_valid_ratio": float(numeric_valid_ratio),
            "cramers_v": association,
            "label_purity_max": weighted_purity,
            "rare_category_ratio": rare_ratio,
            "note": "",
        })

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            ["cramers_v", "label_purity_max"],
            ascending=False
        )

    return result


def encoded_column_diagnostics(df, label_col):
    rows = []

    encoded_cols = [
        col for col in df.columns
        if col.lower().endswith("_encoded")
    ]

    for col in encoded_cols:
        s = safe_to_numeric(df[col])
        y = pd.to_numeric(df[label_col], errors="coerce")
        unique_values = np.sort(s.dropna().unique())

        rows.append({
            "feature": col,
            "unique_count": int(len(unique_values)),
            "min_code": float(unique_values.min()) if len(unique_values) else np.nan,
            "max_code": float(unique_values.max()) if len(unique_values) else np.nan,
            "codes_contiguous": bool(
                len(unique_values) == 0
                or np.array_equal(
                    unique_values,
                    np.arange(unique_values.min(), unique_values.max() + 1)
                )
            ),
            "single_feature_auc": binary_numeric_auc(y, s),
            "warning": (
                "CNN/ResNet coi mã số là giá trị có thứ tự; nên kiểm tra one-hot, frequency encoding hoặc embedding."
            ),
        })

    return pd.DataFrame(rows)


def correlation_report(df, label_col, threshold=0.95):
    numeric = {}

    for col in df.columns:
        if col in {label_col, "source_file"}:
            continue

        values = safe_to_numeric(df[col])
        if values.notna().mean() >= 0.80 and values.nunique(dropna=True) > 1:
            numeric[col] = values

    if len(numeric) < 2:
        return pd.DataFrame()

    numeric_df = pd.DataFrame(numeric)
    corr = numeric_df.corr(method="spearman").abs()

    rows = []
    columns = corr.columns

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            value = corr.iloc[i, j]

            if pd.notna(value) and value >= threshold:
                rows.append({
                    "feature_1": columns[i],
                    "feature_2": columns[j],
                    "abs_spearman_correlation": float(value),
                })

    return pd.DataFrame(rows).sort_values(
        "abs_spearman_correlation",
        ascending=False
    ) if rows else pd.DataFrame()


def train_quick_models(df, label_col, random_state=42, max_rows=300000):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    work = df.dropna(subset=[label_col]).copy()

    if len(work) > max_rows:
        work, _ = train_test_split(
            work,
            train_size=max_rows,
            stratify=work[label_col],
            random_state=random_state,
        )

    y = pd.to_numeric(work[label_col], errors="coerce")
    valid = y.notna()
    work = work.loc[valid]
    y = y.loc[valid].astype(int)

    drop_cols = [
        label_col,
        "type",
        "ts",
        "source_file",
    ]

    X = work.drop(columns=drop_cols, errors="ignore")

    numeric_cols = []
    categorical_cols = []

    for col in X.columns:
        converted = safe_to_numeric(X[col])
        if converted.notna().mean() >= 0.80:
            X[col] = converted
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=10,
            max_categories=100,
        )),
    ])

    transformer = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])

    model = ExtraTreesClassifier(
        n_estimators=150,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        max_depth=None,
    )

    pipeline = Pipeline([
        ("preprocess", transformer),
        ("model", model),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=random_state,
    )

    print("Training quick ExtraTrees diagnostic model...")
    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)
    prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "sample_rows_used": int(len(work)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(
            y_test, pred, zero_division=0, output_dict=True
        ),
    }

    preprocess = pipeline.named_steps["preprocess"]
    feature_names = preprocess.get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    return metrics, importance_df


def pid_cmd_ablation(df, label_col, random_state=42, max_rows=300000):
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    scenarios = {
        "full_numeric": [],
        "without_PID": ["PID"],
        "without_CMD_encoded": ["CMD_encoded"],
        "without_PID_and_CMD_encoded": ["PID", "CMD_encoded"],
    }

    work = df.dropna(subset=[label_col]).copy()

    if len(work) > max_rows:
        work, _ = train_test_split(
            work,
            train_size=max_rows,
            stratify=work[label_col],
            random_state=random_state,
        )

    y = pd.to_numeric(work[label_col], errors="coerce")
    work = work[y.notna()].copy()
    y = y[y.notna()].astype(int)

    candidate_cols = []

    for col in work.columns:
        if col in {label_col, "type", "ts", "source_file"}:
            continue

        converted = safe_to_numeric(work[col])
        if converted.notna().mean() >= 0.80 and converted.nunique(dropna=True) > 1:
            work[col] = converted
            candidate_cols.append(col)

    results = []

    for scenario, drop_cols in scenarios.items():
        features = [col for col in candidate_cols if col not in drop_cols]

        if not features:
            continue

        X = work[features]
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            stratify=y,
            random_state=random_state,
        )

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            )),
        ])

        print(f"Running ablation: {scenario}")
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]

        results.append({
            "scenario": scenario,
            "feature_count": len(features),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_test, pred)
            ),
            "roc_auc": float(roc_auc_score(y_test, prob)),
            "features": ", ".join(features),
        })

    return pd.DataFrame(results).sort_values("f1", ascending=False)


def generate_recommendations(
    quality_df,
    numeric_df,
    categorical_df,
    encoded_df,
    correlation_df,
    ablation_df,
):
    recommendations = []

    high_missing = quality_df[quality_df["missing_pct"] >= 0.20]
    if not high_missing.empty:
        recommendations.append(
            "Có cột thiếu >=20%; cần xem xét bỏ hoặc xây dựng chiến lược imputation riêng."
        )

    id_like = quality_df[
        quality_df["is_id_like"]
        & ~quality_df["column"].isin(["ts", "source_file"])
    ]
    if not id_like.empty:
        recommendations.append(
            "Có cột gần như định danh: "
            + ", ".join(id_like["column"].astype(str).tolist())
            + ". Nên kiểm tra leakage và làm ablation."
        )

    if not numeric_df.empty:
        weak_numeric = numeric_df[
            numeric_df["single_feature_auc"].fillna(0.5) < 0.60
        ]

        if len(weak_numeric) >= max(1, int(len(numeric_df) * 0.60)):
            recommendations.append(
                "Phần lớn feature số có khả năng phân biệt đơn biến yếu; nên tạo feature hành vi theo thời gian/tần suất."
            )

    if not categorical_df.empty:
        suspicious = categorical_df[
            (categorical_df["cramers_v"].fillna(0) >= 0.80)
            | (categorical_df["label_purity_max"].fillna(0) >= 0.98)
        ]

        if not suspicious.empty:
            recommendations.append(
                "Các feature category có liên hệ gần như tuyệt đối với label: "
                + ", ".join(suspicious["feature"].tolist())
                + ". Cần kiểm tra label leakage."
            )

    if not encoded_df.empty:
        recommendations.append(
            "Các cột *_encoded đang là số nguyên. Với CNN/ResNet, nên đánh giá one-hot, frequency encoding hoặc embedding."
        )

    if not correlation_df.empty:
        recommendations.append(
            "Có cặp feature tương quan rất cao; có thể loại bớt để giảm dư thừa."
        )

    if not ablation_df.empty and len(ablation_df) >= 2:
        best = ablation_df.iloc[0]
        full = ablation_df[
            ablation_df["scenario"] == "full_numeric"
        ]

        if not full.empty and best["scenario"] != "full_numeric":
            if best["f1"] > float(full.iloc[0]["f1"]) + 0.01:
                recommendations.append(
                    f"Ablation tốt nhất là {best['scenario']}; nên cân nhắc bỏ PID/CMD_encoded trong pipeline Linux."
                )

    if not recommendations:
        recommendations.append(
            "Chưa thấy lỗi dữ liệu rõ ràng; bước tiếp theo là threshold tuning và feature engineering theo thời gian."
        )

    return recommendations


def main():
    parser = argparse.ArgumentParser(
        description="Chẩn đoán dữ liệu Linux Process của ToN-IoT."
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=DEFAULT_FILES,
        help="Danh sách CSV Linux.",
    )
    parser.add_argument(
        "--label-col",
        default="label",
        help="Tên cột nhãn.",
    )
    parser.add_argument(
        "--output-dir",
        default=r"artifacts\linux_dataset_diagnostics",
        help="Thư mục lưu báo cáo.",
    )
    parser.add_argument(
        "--sample-rows-per-file",
        type=int,
        default=None,
        help="Chỉ đọc N dòng mỗi file để chạy thử nhanh.",
    )
    parser.add_argument(
        "--model-sample-size",
        type=int,
        default=300000,
        help="Số dòng tối đa dùng cho diagnostic model và ablation.",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Bỏ qua quick model và feature importance.",
    )
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="Bỏ qua kiểm tra bỏ PID/CMD_encoded.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    print("=" * 100)
    print("LINUX PROCESS DATASET DIAGNOSTICS")
    print("=" * 100)

    df, file_summary = load_files(
        args.files,
        sample_rows=args.sample_rows_per_file,
    )

    print(f"Combined shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    quality_df = basic_quality_report(df)
    labels_df = label_report(df, args.label_col)
    source_labels_df = source_label_report(df, args.label_col)
    duplicates = duplicate_report(df)
    numeric_df = numeric_feature_report(df, args.label_col)
    categorical_df = categorical_feature_report(df, args.label_col)
    encoded_df = encoded_column_diagnostics(df, args.label_col)
    correlation_df = correlation_report(df, args.label_col)

    file_summary.to_csv(output_dir / "01_file_summary.csv", index=False)
    quality_df.to_csv(output_dir / "02_data_quality.csv", index=False)
    labels_df.to_csv(output_dir / "03_label_distribution.csv", index=False)
    source_labels_df.to_csv(
        output_dir / "04_label_distribution_by_source.csv",
        index=False,
    )
    numeric_df.to_csv(
        output_dir / "05_numeric_feature_separation.csv",
        index=False,
    )
    categorical_df.to_csv(
        output_dir / "06_categorical_feature_association.csv",
        index=False,
    )
    encoded_df.to_csv(
        output_dir / "07_encoded_feature_diagnostics.csv",
        index=False,
    )
    correlation_df.to_csv(
        output_dir / "08_high_correlation_pairs.csv",
        index=False,
    )

    with open(
        output_dir / "09_duplicate_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(duplicates, f, ensure_ascii=False, indent=2)

    quick_metrics = {}
    importance_df = pd.DataFrame()

    if not args.skip_model:
        quick_metrics, importance_df = train_quick_models(
            df,
            args.label_col,
            max_rows=args.model_sample_size,
        )

        importance_df.to_csv(
            output_dir / "10_quick_model_feature_importance.csv",
            index=False,
        )

        with open(
            output_dir / "11_quick_model_metrics.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                quick_metrics,
                f,
                ensure_ascii=False,
                indent=2,
            )

    ablation_df = pd.DataFrame()

    if not args.skip_ablation:
        ablation_df = pid_cmd_ablation(
            df,
            args.label_col,
            max_rows=args.model_sample_size,
        )
        ablation_df.to_csv(
            output_dir / "12_pid_cmd_ablation.csv",
            index=False,
        )

    recommendations = generate_recommendations(
        quality_df,
        numeric_df,
        categorical_df,
        encoded_df,
        correlation_df,
        ablation_df,
    )

    with open(
        output_dir / "13_recommendations.txt",
        "w",
        encoding="utf-8",
    ) as f:
        for index, recommendation in enumerate(
            recommendations,
            start=1,
        ):
            f.write(f"{index}. {recommendation}\n")

    print("\n" + "=" * 100)
    print("TÓM TẮT")
    print("=" * 100)
    print("\nLabel distribution:")
    print(labels_df.to_string(index=False))

    print("\nLabel distribution by source:")
    print(
        source_labels_df.to_string(index=False)
        if not source_labels_df.empty
        else "Không có"
    )

    print("\nDuplicate summary:")
    print(json.dumps(duplicates, ensure_ascii=False, indent=2))

    print("\nTop numeric feature separation:")
    print(
        numeric_df.head(15).to_string(index=False)
        if not numeric_df.empty
        else "Không có"
    )

    print("\nCategorical association:")
    print(
        categorical_df.head(15).to_string(index=False)
        if not categorical_df.empty
        else "Không có"
    )

    print("\nEncoded columns:")
    print(
        encoded_df.to_string(index=False)
        if not encoded_df.empty
        else "Không có"
    )

    print("\nHigh correlation pairs:")
    print(
        correlation_df.head(20).to_string(index=False)
        if not correlation_df.empty
        else "Không có"
    )

    if not importance_df.empty:
        print("\nTop quick-model feature importance:")
        print(importance_df.head(25).to_string(index=False))

    if quick_metrics:
        print("\nQuick-model metrics:")
        print(json.dumps(quick_metrics, ensure_ascii=False, indent=2))

    if not ablation_df.empty:
        print("\nPID/CMD ablation:")
        print(ablation_df.to_string(index=False))

    print("\nRecommendations:")
    for index, recommendation in enumerate(recommendations, start=1):
        print(f"{index}. {recommendation}")

    print("\nReports saved to:")
    print(output_dir.resolve())
    print("=" * 100)


if __name__ == "__main__":
    main()