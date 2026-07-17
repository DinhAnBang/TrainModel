import copy
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from .utils import classification_metrics, save_prediction_artifacts

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        return self.conv2(x, data.edge_index)


def _load_split(graph_dir, split):
    return Data(
        x=torch.tensor(np.load(os.path.join(graph_dir, f"X_{split}.npy")), dtype=torch.float32),
        y=torch.tensor(np.load(os.path.join(graph_dir, f"y_{split}.npy")), dtype=torch.long),
        edge_index=torch.tensor(np.load(os.path.join(graph_dir, f"edge_index_{split}.npy")), dtype=torch.long),
    ).to(DEVICE)


def _evaluate(model, data, return_predictions=False):
    model.eval()
    with torch.no_grad():
        logits = model(data)
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        pred = logits.argmax(1).cpu().numpy()
    y_true = data.y.cpu().numpy()
    metrics = classification_metrics(y_true, pred, prob)
    if return_predictions:
        return metrics, y_true, pred, prob
    return metrics


def _print_header(config, input_dim, num_classes, train_data, val_data, test_data):
    print("\n" + "=" * 100)
    print("TRAINING MODEL: GCN")
    print("=" * 100)
    print(f"  input_dim={input_dim} | hidden_dim={config.get('hidden_dim', 64)} | num_classes={num_classes}")
    print(f"  train_nodes={train_data.x.shape[0]} | train_edges={train_data.edge_index.shape[1]} | "
          f"val_nodes={val_data.x.shape[0]} | test_nodes={test_data.x.shape[0]}")
    print(f"  epochs={config.get('epochs', 50)} | lr={config.get('lr', 0.001)} | "
          f"k_neighbors={config.get('k_neighbors', 10)} | device={DEVICE}")
    print("-" * 100)
    print(f"{'Epoch':>8} | {'Loss':>8} | {'Val Acc':>8} | {'Val Prec':>9} | "
          f"{'Val Rec':>8} | {'Val F1':>8} | {'Val AUC':>8} | {'Time(s)':>8} | Best")
    print("-" * 100)


def _print_epoch(epoch, total_epochs, loss, metrics, elapsed, is_best):
    auc = metrics.get("roc_auc")
    auc_str = f"{auc:.4f}" if auc is not None else "   n/a "
    marker = " <- best" if is_best else ""
    print(f"{epoch:>4}/{total_epochs:<3} | {loss:>8.4f} | {metrics['accuracy']:>8.4f} | "
          f"{metrics['precision']:>9.4f} | {metrics['recall']:>8.4f} | {metrics['f1']:>8.4f} | "
          f"{auc_str:>8} | {elapsed:>8.2f}{marker}")


def _print_footer(best_epoch, best_f1, test_result, train_time):
    print("-" * 100)
    print(f"Best epoch (theo val_f1): {best_epoch} | best val_f1={best_f1:.4f}")
    print("KET QUA TREN TEST SET - GCN")
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        val = test_result.get(key)
        val_str = f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
        print(f"  {key:>10}: {val_str}")
    print(f"  {'time(s)':>10}: {train_time:.2f}")
    print("=" * 100 + "\n")


def train_gnn(graph_dir, output_dir, config):
    os.makedirs(output_dir, exist_ok=True)
    train_data, val_data, test_data = [_load_split(graph_dir, split) for split in ["train", "val", "test"]]
    input_dim = train_data.x.shape[1]
    num_classes = int(torch.max(train_data.y).item()) + 1
    hidden_dim = int(config.get("hidden_dim", 64))
    model = GCN(input_dim, hidden_dim, num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("lr", 0.001)), weight_decay=5e-4)
    classes = np.unique(train_data.y.cpu().numpy())
    weights = compute_class_weight("balanced", classes=classes, y=train_data.y.cpu().numpy())
    criterion_weight = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

    train_labels_np = train_data.y.cpu().numpy()
    class_values, class_counts = np.unique(train_labels_np, return_counts=True)
    class_count_map = {
        int(cls): int(count)
        for cls, count in zip(class_values, class_counts)
    }
    weight_map = {
        int(cls): float(weight)
        for cls, weight in zip(classes, weights)
    }

    print("\n" + "-" * 100)
    print("[CLASS WEIGHT - GCN]")
    print(f"  Train class distribution : {class_count_map}")
    print(f"  CrossEntropyLoss weights  : {weight_map}")
    print("-" * 100)

    epochs = int(config.get("epochs", 50))

    best_f1, best_epoch, best_state = -1.0, 0, None
    history = []
    start = time.time()

    _print_header(config, input_dim, num_classes, train_data, val_data, test_data)

    # Luon chay du so epoch da config, KHONG early stop.
    # Van theo doi epoch co val_f1 tot nhat de phuc hoi (restore) trong so do cho model cuoi cung.
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad()
        logits = model(train_data)
        loss = F.cross_entropy(logits, train_data.y, weight=criterion_weight)
        loss.backward()
        optimizer.step()

        val_metrics = _evaluate(model, val_data)
        epoch_time = time.time() - epoch_start
        history.append({"epoch": epoch, "loss": float(loss.item()), **{f"val_{k}": v for k, v in val_metrics.items()}, "epoch_time_sec": epoch_time})

        is_best = val_metrics["f1"] > best_f1
        if is_best:
            best_f1, best_epoch, best_state = val_metrics["f1"], epoch, copy.deepcopy(model.state_dict())

        _print_epoch(epoch, epochs, float(loss.item()), val_metrics, epoch_time, is_best)

    model.load_state_dict(best_state)
    train_time = time.time() - start
    test_metrics, y_true, y_pred, y_prob = _evaluate(model, test_data, return_predictions=True)
    save_prediction_artifacts(output_dir, "gcn", y_true, y_pred, y_prob)
    result = {"model_group": "Graph Neural Network", "model": "GCN", **test_metrics, "train_time_sec": train_time}

    _print_footer(best_epoch, best_f1, test_metrics, train_time)

    torch.save({
        "model_state_dict": model.state_dict(), "input_dim": input_dim,
        "hidden_dim": hidden_dim, "num_classes": num_classes,
        "model_name": "GCN", "k_neighbors": int(config.get("k_neighbors", 10)),
        "best_epoch": best_epoch,
    }, os.path.join(output_dir, "gcn.pt"))
    pd.DataFrame(history).to_csv(os.path.join(output_dir, "gcn_history.csv"), index=False)
    df = pd.DataFrame([result])
    df.to_csv(os.path.join(output_dir, "gnn_results.csv"), index=False)
    return df