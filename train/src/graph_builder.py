import os
import json
import numpy as np
import pandas as pd

from sklearn.neighbors import kneighbors_graph


def build_graph(processed_dir, output_dir, k_neighbors=10):
    os.makedirs(output_dir, exist_ok=True)

    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))

    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    train_mask = np.zeros(len(y), dtype=bool)
    test_mask = np.zeros(len(y), dtype=bool)

    train_mask[:len(y_train)] = True
    test_mask[len(y_train):] = True

    print("Building kNN graph...")
    print("Nodes:", X.shape[0])
    print("Features:", X.shape[1])
    print("k:", k_neighbors)

    adjacency = kneighbors_graph(
        X,
        n_neighbors=k_neighbors,
        mode="connectivity",
        include_self=False,
        n_jobs=-1
    )

    source, target = adjacency.nonzero()
    edge_index = np.vstack([source, target]).astype(np.int64)

    np.save(os.path.join(output_dir, "X_graph.npy"), X)
    np.save(os.path.join(output_dir, "y_graph.npy"), y)
    np.save(os.path.join(output_dir, "edge_index.npy"), edge_index)
    np.save(os.path.join(output_dir, "train_mask.npy"), train_mask)
    np.save(os.path.join(output_dir, "test_mask.npy"), test_mask)

    pd.DataFrame({
        "node_id": np.arange(len(y)),
        "label": y,
        "split": np.where(train_mask, "train", "test")
    }).to_csv(os.path.join(output_dir, "node_table.csv"), index=False)

    pd.DataFrame({
        "source": edge_index[0],
        "target": edge_index[1]
    }).to_csv(os.path.join(output_dir, "edge_table.csv"), index=False)

    summary = {
        "num_nodes": int(len(y)),
        "num_edges": int(edge_index.shape[1]),
        "feature_dim": int(X.shape[1]),
        "k_neighbors": int(k_neighbors),
        "train_nodes": int(train_mask.sum()),
        "test_nodes": int(test_mask.sum())
    }

    with open(os.path.join(output_dir, "graph_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("Graph summary:", summary)

    return summary