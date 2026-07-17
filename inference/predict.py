import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data

from src.deep_train import TabularCNN, TabularResNet
from src.gnn_train import GCN
from src.preprocess import clean_dataframe

PROJECT_DIR = Path(__file__).resolve().parent.parent
BEST_MODEL_DIR = PROJECT_DIR / "artifacts" / "best_models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_new_data(df, model_dir):
    info = joblib.load(model_dir / "preprocess_info.pkl")
    scaler = joblib.load(model_dir / "scaler.pkl")
    encoder_path = model_dir / "categorical_encoder.pkl"
    encoder = joblib.load(encoder_path) if encoder_path.exists() else None
    X = clean_dataframe(df.drop(columns=info["drop_cols"], errors="ignore"))
    parts = []
    if info["numeric_cols"]:
        num = pd.DataFrame(index=X.index)
        for col in info["numeric_cols"]:
            num[col] = pd.to_numeric(X[col], errors="coerce") if col in X.columns else np.nan
            num[col] = num[col].fillna(info["numeric_medians"].get(col, 0))
        parts.append(num)
    if info["categorical_cols"] and encoder is not None:
        cat = pd.DataFrame(index=X.index)
        for col in info["categorical_cols"]:
            cat[col] = X[col].fillna("missing").astype(str) if col in X.columns else "missing"
        encoded = encoder.transform(cat)
        parts.append(pd.DataFrame(encoded, columns=[f"{c}_encoded" for c in info["categorical_cols"]], index=X.index))
    final = pd.concat(parts, axis=1)
    for col in info["feature_names"]:
        if col not in final.columns:
            final[col] = 0
    final = final[info["feature_names"]]
    return final.to_numpy(np.float32), scaler.transform(final).astype(np.float32), info


def predict_csv(experiment_name, csv_path):
    model_dir = BEST_MODEL_DIR / experiment_name
    best_info = json.loads((model_dir / "best_model_info.json").read_text(encoding="utf-8"))
    df = pd.read_csv(csv_path, low_memory=False)
    X_raw, X_scaled, prep_info = preprocess_new_data(df, model_dir)
    model_name = best_info["model"]
    model_path = model_dir / best_info["model_file"]

    if model_name in ["Random Forest", "XGBoost"]:
        model = joblib.load(model_path)
        pred = model.predict(X_raw)
        prob = model.predict_proba(X_raw) if hasattr(model, "predict_proba") else None
    elif model_name in ["CNN", "ResNet"]:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        cls = TabularCNN if model_name == "CNN" else TabularResNet
        model = cls(checkpoint["input_dim"], checkpoint["num_classes"]).to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_scaled, dtype=torch.float32, device=DEVICE))
            pred = logits.argmax(1).cpu().numpy()
            prob = torch.softmax(logits, 1).cpu().numpy()
    else:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model = GCN(checkpoint["input_dim"], checkpoint["hidden_dim"], checkpoint["num_classes"]).to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        k = min(checkpoint.get("k_neighbors", 5), max(len(X_scaled) - 1, 1))
        if len(X_scaled) == 1:
            edge_index = np.array([[0], [0]], dtype=np.int64)
        else:
            adjacency = kneighbors_graph(X_scaled, n_neighbors=k, mode="connectivity", include_self=False)
            source, target = adjacency.nonzero()
            edge_index = np.vstack([source, target]).astype(np.int64)
        data = Data(x=torch.tensor(X_scaled, dtype=torch.float32), edge_index=torch.tensor(edge_index, dtype=torch.long)).to(DEVICE)
        model.eval()
        with torch.no_grad():
            logits = model(data)
            pred = logits.argmax(1).cpu().numpy()
            prob = torch.softmax(logits, 1).cpu().numpy()

    label_encoder = joblib.load(model_dir / "label_encoder.pkl")
    result = df.copy()
    result["prediction_encoded"] = pred
    result["prediction"] = label_encoder.inverse_transform(pred.astype(int))
    if prob is not None:
        result["prediction_probability"] = np.max(prob, axis=1)
    output = Path(f"prediction_{experiment_name}.csv")
    result.to_csv(output, index=False)
    print("Best model:", model_name)
    print("Saved:", output)
    return result


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python inference/predict.py <experiment_name> <csv_path>")
        raise SystemExit(1)
    predict_csv(sys.argv[1], sys.argv[2])
