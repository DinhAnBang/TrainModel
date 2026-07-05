import os
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn.functional as F

from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv


PROJECT_DIR = Path(__file__).resolve().parent.parent
BEST_MODEL_DIR = PROJECT_DIR / "artifacts" / "best_models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = F.relu(x)
        x = self.conv2(x, data.edge_index)
        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = F.relu(x)
        x = self.conv2(x, data.edge_index)
        return x


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


def load_gnn_artifacts(dataset_name):
    model_dir = BEST_MODEL_DIR / dataset_name

    if not model_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy best model folder: {model_dir}")

    pt_files = list(model_dir.glob("*.pt"))

    if len(pt_files) == 0:
        raise FileNotFoundError(
            f"Best model của {dataset_name} không phải GNN hoặc chưa có .pt"
        )

    model_path = pt_files[0]

    scaler = joblib.load(model_dir / "scaler.pkl")
    preprocess_info = joblib.load(model_dir / "preprocess_info.pkl")

    encoder_path = model_dir / "categorical_encoder.pkl"
    encoder = joblib.load(encoder_path) if encoder_path.exists() else None

    return model_path, scaler, encoder, preprocess_info


def preprocess_new_data(df, scaler, encoder, preprocess_info):
    numeric_cols = preprocess_info["numeric_cols"]
    categorical_cols = preprocess_info["categorical_cols"]
    feature_names = preprocess_info["feature_names"]
    drop_cols = preprocess_info["drop_cols"]

    X = df.drop(columns=drop_cols, errors="ignore")
    X = clean_dataframe(X)

    parts = []

    if len(numeric_cols) > 0:
        X_num = pd.DataFrame(index=X.index)

        for col in numeric_cols:
            if col in X.columns:
                X_num[col] = pd.to_numeric(X[col], errors="coerce")
            else:
                X_num[col] = 0

        X_num = X_num.fillna(0)
        parts.append(X_num)

    if len(categorical_cols) > 0 and encoder is not None:
        X_cat = pd.DataFrame(index=X.index)

        for col in categorical_cols:
            if col in X.columns:
                X_cat[col] = X[col].fillna("missing")
            else:
                X_cat[col] = "missing"

        X_cat_encoded = encoder.transform(X_cat)

        cat_feature_names = [f"{col}_encoded" for col in categorical_cols]

        X_cat_df = pd.DataFrame(
            X_cat_encoded,
            columns=cat_feature_names,
            index=X.index
        )

        parts.append(X_cat_df)

    if len(parts) == 0:
        raise ValueError("Không có feature hợp lệ cho inference")

    X_final = pd.concat(parts, axis=1)

    for col in feature_names:
        if col not in X_final.columns:
            X_final[col] = 0

    X_final = X_final[feature_names]

    X_scaled = scaler.transform(X_final)

    return X_scaled


def build_inference_graph(X, k=5):
    n = X.shape[0]

    if n <= 1:
        edge_index = np.array([[0], [0]], dtype=np.int64)
    else:
        k = min(k, n - 1)

        adjacency = kneighbors_graph(
            X,
            n_neighbors=k,
            mode="connectivity",
            include_self=False
        )

        source, target = adjacency.nonzero()
        edge_index = np.vstack([source, target]).astype(np.int64)

    data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long)
    )

    return data.to(DEVICE)


def load_model(model_path, input_dim):
    model_name = model_path.stem.lower()

    if "graphsage" in model_name:
        model = GraphSAGE(input_dim=input_dim, hidden_dim=64, output_dim=2)
    elif "gcn" in model_name:
        model = GCN(input_dim=input_dim, hidden_dim=64, output_dim=2)
    else:
        raise ValueError(f"Không nhận diện được loại GNN từ file: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    return model


def predict_gnn_csv(dataset_name, csv_path, k=5):
    model_path, scaler, encoder, preprocess_info = load_gnn_artifacts(dataset_name)

    df = pd.read_csv(csv_path)

    X = preprocess_new_data(df, scaler, encoder, preprocess_info)

    data = build_inference_graph(X, k=k)

    model = load_model(model_path, input_dim=X.shape[1])

    with torch.no_grad():
        out = model(data)
        prob = F.softmax(out, dim=1)[:, 1]
        pred = out.argmax(dim=1)

    result = df.copy()
    result["prediction"] = pred.detach().cpu().numpy()
    result["attack_probability"] = prob.detach().cpu().numpy()

    print("Loaded GNN model:", model_path)
    print("Device:", DEVICE)
    print(result[["prediction", "attack_probability"]].head())

    return result


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("python inference/predict_gnn.py <dataset_name> <csv_path>")
        sys.exit(1)

    dataset_name = sys.argv[1]
    csv_path = sys.argv[2]

    result = predict_gnn_csv(dataset_name, csv_path)

    output_path = f"gnn_prediction_{dataset_name}.csv"
    result.to_csv(output_path, index=False)

    print("Saved:", output_path)