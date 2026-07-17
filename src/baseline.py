import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .utils import classification_metrics, save_prediction_artifacts


def _build_xgboost(config, num_classes, scale_pos_weight=None):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("Thiếu xgboost. Cài bằng: pip install xgboost") from exc

    params = {
        "n_estimators": int(config.get("xgb_estimators", 300)),
        "max_depth": int(config.get("xgb_max_depth", 8)),
        "learning_rate": float(config.get("xgb_learning_rate", 0.05)),
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": int(config.get("random_state", 42)),
        "n_jobs": -1,
        "tree_method": "hist",
        "eval_metric": "logloss" if num_classes == 2 else "mlogloss",
    }
    if num_classes > 2:
        params.update(objective="multi:softprob", num_class=num_classes)
    else:
        params.update(objective="binary:logistic")
        if scale_pos_weight is not None:
            params["scale_pos_weight"] = float(scale_pos_weight)
    return XGBClassifier(**params)


def train_baseline(processed_dir, output_dir, config, models=None):
    os.makedirs(output_dir, exist_ok=True)
    X_train = np.load(os.path.join(processed_dir, "X_train_raw.npy"))
    X_test = np.load(os.path.join(processed_dir, "X_test_raw.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    num_classes = len(np.unique(y_train))
    requested = set(models or ["random_forest", "xgboost"])

    classes, class_counts = np.unique(y_train, return_counts=True)
    class_count_map = {int(cls): int(count) for cls, count in zip(classes, class_counts)}
    n_normal = class_count_map.get(0, 0)
    n_attack = class_count_map.get(1, 0)
    scale_pos_weight = None
    if num_classes == 2:
        scale_pos_weight = n_normal / max(n_attack, 1)

    print("\n" + "-" * 100)
    print("[CLASS WEIGHT - TRADITIONAL ML]")
    print(f"  Train class distribution : {class_count_map}")
    print('  Random Forest weight     : class_weight="balanced"')
    if scale_pos_weight is not None:
        print(f"  XGBoost scale_pos_weight : {scale_pos_weight:.6f}")
    else:
        print("  XGBoost class weighting  : not applied for multi-class mode")
    print("-" * 100)

    candidates = []
    if "random_forest" in requested:
        candidates.append(("Random Forest", RandomForestClassifier(
            n_estimators=int(config.get("rf_estimators", 200)),
            class_weight="balanced",
            random_state=int(config.get("random_state", 42)),
            n_jobs=-1,
        ), "random_forest.pkl"))
    if "xgboost" in requested:
        candidates.append(("XGBoost", _build_xgboost(config, num_classes, scale_pos_weight), "xgboost.pkl"))

    results = []
    for name, model, filename in candidates:
        print("Training:", name)
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        result = {"model_group": "Traditional ML", "model": name, **classification_metrics(y_test, y_pred, y_prob), "train_time_sec": train_time}
        results.append(result)
        model_key = "random_forest" if name == "Random Forest" else "xgboost"
        save_prediction_artifacts(output_dir, model_key, y_test, y_pred, y_prob)
        joblib.dump(model, os.path.join(output_dir, filename))

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "ml_results.csv"), index=False)
    print(df)
    return df