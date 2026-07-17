import copy
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight

from .utils import classification_metrics, save_prediction_artifacts

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TabularCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x.unsqueeze(1)))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1), nn.BatchNorm1d(channels), nn.ReLU(),
            nn.Conv1d(channels, channels, 3, padding=1), nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class TabularResNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU())
        self.body = nn.Sequential(ResidualBlock(64), ResidualBlock(64), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, num_classes))

    def forward(self, x):
        x = self.stem(x.unsqueeze(1))
        return self.head(self.body(x))


def _predict(model, loader):
    model.eval()
    all_true, all_pred, all_prob = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(DEVICE))
            prob = torch.softmax(logits, dim=1)
            all_true.append(yb.numpy())
            all_pred.append(logits.argmax(1).cpu().numpy())
            all_prob.append(prob.cpu().numpy())
    return np.concatenate(all_true), np.concatenate(all_pred), np.concatenate(all_prob)


def _print_header(model_name, config, input_dim, num_classes, train_size, val_size, test_size):
    print("\n" + "=" * 100)
    print(f"TRAINING MODEL: {model_name}")
    print("=" * 100)
    print(f"  input_dim={input_dim} | num_classes={num_classes} | "
          f"train={train_size} | val={val_size} | test={test_size}")
    print(f"  epochs={config.get('epochs', 50)} | lr={config.get('lr', 0.001)} | "
          f"batch_size={config.get('batch_size', 512)} | device={DEVICE}")
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


def _print_footer(model_name, best_epoch, best_f1, test_result, train_time):
    print("-" * 100)
    print(f"Best epoch (theo val_f1): {best_epoch} | best val_f1={best_f1:.4f}")
    print(f"KET QUA TREN TEST SET - {model_name}")
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        val = test_result.get(key)
        val_str = f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
        print(f"  {key:>10}: {val_str}")
    print(f"  {'time(s)':>10}: {train_time:.2f}")
    print("=" * 100 + "\n")


def _train_one(model, model_name, loaders, class_weights, output_dir, config):
    model = model.to(DEVICE)

    train_labels = loaders["train"].dataset.tensors[1].cpu().numpy()
    classes, class_counts = np.unique(train_labels, return_counts=True)
    class_count_map = {int(cls): int(count) for cls, count in zip(classes, class_counts)}
    weight_map = {
        int(cls): float(weight)
        for cls, weight in zip(classes, class_weights.detach().cpu().numpy())
    }

    print("\n" + "-" * 100)
    print(f"[CLASS WEIGHT - {model_name.upper()}]")
    print(f"  Train class distribution : {class_count_map}")
    print(f"  CrossEntropyLoss weights  : {weight_map}")
    print("-" * 100)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("lr", 0.001)), weight_decay=1e-4)
    epochs = int(config.get("epochs", 50))

    best_f1, best_epoch, best_state = -1.0, 0, None
    history = []
    start = time.time()

    _print_header(
        model_name, config,
        input_dim=loaders["train"].dataset.tensors[0].shape[1],
        num_classes=len(torch.unique(loaders["train"].dataset.tensors[1])),
        train_size=len(loaders["train"].dataset),
        val_size=len(loaders["val"].dataset),
        test_size=len(loaders["test"].dataset),
    )

    # Luon chay du so epoch da config, KHONG early stop.
    # Van theo doi epoch co val_f1 tot nhat de phuc hoi (restore) trong so do cho model cuoi cung,
    # tranh viec lay dung epoch cuoi (co the dang overfit/kem hon) lam ket qua bao cao.
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        for xb, yb in loaders["train"]:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)

        avg_loss = total_loss / len(loaders["train"].dataset)
        y_true, y_pred, y_prob = _predict(model, loaders["val"])
        metrics = classification_metrics(y_true, y_pred, y_prob)
        epoch_time = time.time() - epoch_start
        history.append({"epoch": epoch, "loss": avg_loss, **metrics, "epoch_time_sec": epoch_time})

        is_best = metrics["f1"] > best_f1
        if is_best:
            best_f1, best_epoch, best_state = metrics["f1"], epoch, copy.deepcopy(model.state_dict())

        _print_epoch(epoch, epochs, avg_loss, metrics, epoch_time, is_best)

    model.load_state_dict(best_state)
    train_time = time.time() - start
    y_true, y_pred, y_prob = _predict(model, loaders["test"])
    test_metrics = classification_metrics(y_true, y_pred, y_prob)
    model_key = model_name.lower()
    save_prediction_artifacts(output_dir, model_key, y_true, y_pred, y_prob)
    result = {"model_group": "Deep Learning", "model": model_name, **test_metrics, "train_time_sec": train_time}

    _print_footer(model_name, best_epoch, best_f1, test_metrics, train_time)

    filename = "cnn.pt" if model_name == "CNN" else "resnet.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": loaders["train"].dataset.tensors[0].shape[1],
        "num_classes": y_prob.shape[1],
        "model_name": model_name,
        "best_epoch": best_epoch,
    }, os.path.join(output_dir, filename))
    pd.DataFrame(history).to_csv(os.path.join(output_dir, f"{model_name.lower()}_history.csv"), index=False)
    return result


def train_deep_models(processed_dir, output_dir, config, models=None):
    os.makedirs(output_dir, exist_ok=True)
    arrays = {name: np.load(os.path.join(processed_dir, f"{name}.npy")) for name in [
        "X_train_scaled", "X_val_scaled", "X_test_scaled", "y_train", "y_val", "y_test"
    ]}
    batch_size = int(config.get("batch_size", 512))
    loaders = {}
    for split in ["train", "val", "test"]:
        ds = TensorDataset(torch.tensor(arrays[f"X_{split}_scaled"], dtype=torch.float32), torch.tensor(arrays[f"y_{split}"], dtype=torch.long))
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"), num_workers=0)

    classes = np.unique(arrays["y_train"])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=arrays["y_train"])
    class_weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    input_dim, num_classes = arrays["X_train_scaled"].shape[1], len(classes)
    requested = set(models or ["cnn", "resnet"])
    results = []
    if "cnn" in requested:
        results.append(_train_one(TabularCNN(input_dim, num_classes), "CNN", loaders, class_weights, output_dir, config))
    if "resnet" in requested:
        results.append(_train_one(TabularResNet(input_dim, num_classes), "ResNet", loaders, class_weights, output_dir, config))
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "deep_results.csv"), index=False)
    print("\nTONG HOP DEEP LEARNING MODELS:")
    print(df.to_string(index=False))
    return df   