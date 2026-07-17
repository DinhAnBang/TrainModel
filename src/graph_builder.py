import json
import os
import time

import numpy as np
from sklearn.neighbors import kneighbors_graph

# Ngưỡng số node để quyết định dùng sklearn (brute-force/kd-tree) hay faiss (ANN, nhanh hơn nhiều với n lớn).
# Có thể chỉnh nếu máy bạn RAM nhiều/ít.
FAISS_THRESHOLD = 150_000

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


def _symmetrize(source, target, num_nodes):
    """Đối xứng hóa graph: nếu có cạnh A->B thì thêm cả B->A, rồi loại cạnh trùng.
    Cần thiết cho GCNConv để message truyền được 2 chiều giữa các node."""
    edge_index = np.vstack([source, target])
    edge_index_sym = np.hstack([edge_index, edge_index[[1, 0]]])
    edge_index_sym = np.unique(edge_index_sym, axis=1)
    return edge_index_sym.astype(np.int64)


def _knn_sklearn(X, k):
    adjacency = kneighbors_graph(X, n_neighbors=k, mode="connectivity", include_self=False, n_jobs=-1)
    source, target = adjacency.nonzero()
    return source, target


def _knn_faiss(X, k):
    """kNN xấp xỉ bằng faiss (CPU). Nhanh hơn nhiều so với sklearn khi n lớn (vd. ~500k dòng).
    IndexFlatL2 vẫn là exact search (không mất độ chính xác), chỉ tối ưu tốc độ hơn sklearn."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    n, dim = X.shape
    index = faiss.IndexFlatL2(dim)
    index.add(X)
    # k+1 vì kết quả gần nhất luôn bao gồm chính nó (khoảng cách = 0)
    _, neighbor_idx = index.search(X, k + 1)

    source_list, target_list = [], []
    for i in range(n):
        neighbors = neighbor_idx[i][neighbor_idx[i] != i][:k]
        source_list.append(np.full(len(neighbors), i, dtype=np.int64))
        target_list.append(neighbors.astype(np.int64))
    source = np.concatenate(source_list)
    target = np.concatenate(target_list)
    return source, target


def _build_split_graph(X, k_neighbors, verbose_label=""):
    n = len(X)
    if n <= 1:
        return np.array([[0], [0]], dtype=np.int64)

    k = min(int(k_neighbors), n - 1)
    start = time.time()

    use_faiss = FAISS_AVAILABLE and n >= FAISS_THRESHOLD
    if use_faiss:
        print(f"  [{verbose_label}] n={n} >= {FAISS_THRESHOLD} -> dùng faiss (k={k})")
        source, target = _knn_faiss(X, k)
    else:
        if n >= FAISS_THRESHOLD and not FAISS_AVAILABLE:
            print(f"  [{verbose_label}] Cảnh báo: n={n} lớn nhưng chưa cài faiss "
                  f"(pip install faiss-cpu) -> đang dùng sklearn, có thể chậm/tốn RAM")
        else:
            print(f"  [{verbose_label}] n={n} -> dùng sklearn (k={k})")
        source, target = _knn_sklearn(X, k)

    edge_index = _symmetrize(source, target, n)
    elapsed = time.time() - start
    print(f"  [{verbose_label}] Xong trong {elapsed:.1f}s, {edge_index.shape[1]} cạnh (sau đối xứng hóa)")
    return edge_index


def build_graph(processed_dir, output_dir, k_neighbors=10):
    os.makedirs(output_dir, exist_ok=True)
    summary = {"k_neighbors": int(k_neighbors), "faiss_available": FAISS_AVAILABLE, "splits": {}}

    for split in ["train", "val", "test"]:
        X = np.load(os.path.join(processed_dir, f"X_{split}_scaled.npy"))
        y = np.load(os.path.join(processed_dir, f"y_{split}.npy"))

        edge_index = _build_split_graph(X, k_neighbors, verbose_label=split)

        np.save(os.path.join(output_dir, f"X_{split}.npy"), X)
        np.save(os.path.join(output_dir, f"y_{split}.npy"), y)
        np.save(os.path.join(output_dir, f"edge_index_{split}.npy"), edge_index)

        summary["splits"][split] = {
            "nodes": int(len(y)),
            "edges": int(edge_index.shape[1]),
            "features": int(X.shape[1]),
        }

    with open(os.path.join(output_dir, "graph_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    print("Graph summary:", summary)
    return summary