import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def train_baseline(processed_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))

    models = [
        (
            "Logistic Regression",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced"
            )
        ),
        (
            "Gradient Boosting",
            GradientBoostingClassifier(random_state=42)
        )
    ]

    results = []

    for name, model in models:
        print("Training baseline:", name)

        start = time.time()

        model.fit(X_train, y_train)

        train_time = time.time() - start

        y_pred = model.predict(X_test)

        result = {
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "train_time_sec": train_time
        }

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            result["roc_auc"] = roc_auc_score(y_test, y_prob)
        else:
            result["roc_auc"] = None

        results.append(result)

        model_name = name.replace(" ", "_").lower()
        joblib.dump(model, os.path.join(output_dir, f"{model_name}.pkl"))

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "baseline_results.csv"), index=False)

    print(df)

    return df