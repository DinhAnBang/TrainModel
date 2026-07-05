import os
import time
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
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
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, data.edge_index)
        return x


def load_graph(graph_dir):
    X = np.load(os.path.join(graph_dir, "X_graph.npy"))
    y = np.load(os.path.join(graph_dir, "y_graph.npy"))
    edge_index = np.load(os.path.join(graph_dir, "edge_index.npy"))
    train_mask = np.load(os.path.join(graph_dir, "train_mask.npy"))
    test_mask = np.load(os.path.join(graph_dir, "test_mask.npy"))

    data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.long),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        train_mask=torch.tensor(train_mask, dtype=torch.bool),
        test_mask=torch.tensor(test_mask, dtype=torch.bool)
    )

    data = data.to(DEVICE)

    print(data)

    return data


def get_class_weights(data):
    train_labels = data.y[data.train_mask]
    class_counts = torch.bincount(train_labels)

    class_weights = train_labels.size(0) / (
        len(class_counts) * class_counts.float()
    )

    class_weights = class_weights.to(DEVICE)

    print("Class counts:", class_counts)
    print("Class weights:", class_weights)

    return class_weights


def evaluate_epoch(model, data, loss):
    model.eval()

    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)

        train_true = data.y[data.train_mask].detach().cpu().numpy()
        train_pred = pred[data.train_mask].detach().cpu().numpy()

        test_true = data.y[data.test_mask].detach().cpu().numpy()
        test_pred = pred[data.test_mask].detach().cpu().numpy()

        return {
            "loss": loss.item(),
            "train_accuracy": accuracy_score(train_true, train_pred),
            "test_accuracy": accuracy_score(test_true, test_pred),
            "train_precision": precision_score(train_true, train_pred, zero_division=0),
            "test_precision": precision_score(test_true, test_pred, zero_division=0),
            "train_recall": recall_score(train_true, train_pred, zero_division=0),
            "test_recall": recall_score(test_true, test_pred, zero_division=0),
            "train_f1": f1_score(train_true, train_pred, zero_division=0),
            "test_f1": f1_score(test_true, test_pred, zero_division=0)
        }


def train_one_model(model, data, model_name, output_dir, epochs=50, lr=0.01):
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    class_weights = get_class_weights(data)

    history = []

    start = time.time()

    print("\n" + "=" * 60)
    print("Training:", model_name)
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        out = model(data)

        loss = F.cross_entropy(
            out[data.train_mask],
            data.y[data.train_mask],
            weight=class_weights
        )

        loss.backward()
        optimizer.step()

        metrics = evaluate_epoch(model, data, loss)

        history.append({
            "model": model_name,
            "epoch": epoch,
            **metrics
        })

        print(
            f"{model_name} | "
            f"Epoch {epoch:03d}/{epochs} | "
            f"Loss={metrics['loss']:.4f} | "
            f"Train Acc={metrics['train_accuracy']:.4f} | "
            f"Test Acc={metrics['test_accuracy']:.4f} | "
            f"Train F1={metrics['train_f1']:.4f} | "
            f"Test F1={metrics['test_f1']:.4f} | "
            f"Test Recall={metrics['test_recall']:.4f}"
        )

    train_time = time.time() - start

    history_df = pd.DataFrame(history)
    history_df.to_csv(
        os.path.join(output_dir, f"{model_name.lower()}_training_history.csv"),
        index=False
    )

    model.eval()

    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)

    y_true = data.y[data.test_mask].detach().cpu().numpy()
    y_pred = pred[data.test_mask].detach().cpu().numpy()

    result = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "train_time_sec": train_time
    }

    torch.save(
        model.state_dict(),
        os.path.join(output_dir, f"{model_name.lower()}.pt")
    )

    return result


def train_gnn(graph_dir, output_dir, epochs=50, hidden_dim=64, lr=0.01):
    os.makedirs(output_dir, exist_ok=True)

    data = load_graph(graph_dir)

    input_dim = data.x.shape[1]
    output_dim = 2

    results = []

    gcn = GCN(input_dim, hidden_dim, output_dim)
    results.append(train_one_model(gcn, data, "GCN", output_dir, epochs, lr))

    sage = GraphSAGE(input_dim, hidden_dim, output_dim)
    results.append(train_one_model(sage, data, "GraphSAGE", output_dir, epochs, lr))

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "gnn_results.csv"), index=False)

    print("\nFINAL GNN RESULTS")
    print(df)

    return df