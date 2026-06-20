# ============================================================
# scripts/run_step8_build_clusters.py
#
# Fits document clusters on the Word2Vec vectors, then reports:
#   - cluster sizes
#   - silhouette score (on a sample)
#   - top terms per cluster (interpretability)
#   - charts: cluster-size bar + 2D cluster map (PCA)  -> PNG for report
#
# Run from the project root:
#   python -m scripts.run_step8_build_clusters
# ============================================================
import sys
import os
import json
import time
import random
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import config
from services.representation_embedding import Word2VecEmbedding
from services.clustering import DocumentClustering


def top_terms_per_cluster(clustering, processed_path, k_clusters, top_n=8):
    counters = [Counter() for _ in range(k_clusters)]
    with open(processed_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            lbl = clustering.label_of(rec["doc_id"])
            if lbl is not None:
                counters[lbl].update(rec.get("lexical", "").split())
    return {c: [w for w, _ in counters[c].most_common(top_n)] for c in range(k_clusters)}


def make_charts(clustering, matrix, sizes, sil, out_dir, n_scatter, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # cluster sizes
    cids = sorted(sizes)
    ax1.bar([str(c) for c in cids], [sizes[c] for c in cids], color="#4C78A8")
    ax1.set_title("Cluster sizes")
    ax1.set_xlabel("cluster id"); ax1.set_ylabel("# documents")
    ax1.tick_params(axis="x", rotation=90, labelsize=7)

    # 2D map (PCA on a sample)
    rng = np.random.default_rng(seed)
    n = min(n_scatter, matrix.shape[0])
    idx = rng.choice(matrix.shape[0], size=n, replace=False)
    pts = PCA(n_components=2, random_state=seed).fit_transform(matrix[idx])
    sc = ax2.scatter(pts[:, 0], pts[:, 1], c=clustering.labels[idx],
                     cmap="tab20", s=6, alpha=0.6)
    ax2.set_title(f"2D cluster map (PCA, {n} sampled docs)")
    ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")

    fig.suptitle(f"{config.DATASET_NAME}: {clustering.n_clusters} clusters  |  "
                 f"silhouette = {sil:.3f}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, "clusters_overview.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    if not os.path.isdir(config.EMBEDDING_DIR):
        print("ERROR: embeddings not found. Run Step 4 first.")
        return
    cfg = config.CLUSTER

    print("Loading Word2Vec document vectors ...")
    emb = Word2VecEmbedding.load(config.EMBEDDING_DIR)
    matrix = emb.doc_matrix
    print(f"  matrix: {matrix.shape}")

    print(f"\nClustering into {cfg['n_clusters']} clusters (MiniBatchKMeans) ...")
    t0 = time.time()
    clustering = DocumentClustering.fit(
        matrix, emb.doc_ids, n_clusters=cfg["n_clusters"],
        seed=cfg["seed"], batch_size=cfg["batch_size"])
    clustering.save(config.CLUSTER_DIR)
    print(f"  done in {time.time()-t0:.1f}s   saved -> {config.CLUSTER_DIR}")

    sizes = clustering.cluster_sizes()
    print("\nCluster sizes:", dict(sorted(sizes.items())))

    # silhouette on a sample
    print("\nComputing silhouette score (sample) ...")
    from sklearn.metrics import silhouette_score
    rng = np.random.default_rng(cfg["seed"])
    n = min(cfg["sample_silhouette"], matrix.shape[0])
    sidx = rng.choice(matrix.shape[0], size=n, replace=False)
    sil = silhouette_score(matrix[sidx], clustering.labels[sidx])
    print(f"  silhouette = {sil:.4f}  (on {n} sampled docs; range -1..1, higher = better)")

    # top terms per cluster
    print("\nTop terms per cluster:")
    tops = top_terms_per_cluster(clustering, config.PROCESSED_DOCS_PATH, cfg["n_clusters"])
    for c in range(cfg["n_clusters"]):
        print(f"  cluster {c:>2} (n={sizes.get(c,0):>6}): {', '.join(tops[c])}")
    with open(os.path.join(config.CLUSTER_DIR, "top_terms.json"), "w", encoding="utf-8") as f:
        json.dump(tops, f, indent=2)

    # charts
    try:
        path = make_charts(clustering, matrix, sizes, sil, config.CLUSTER_DIR,
                           cfg["sample_scatter"], cfg["seed"])
        print("\nSaved chart ->", path)
    except Exception as e:
        print("[warning] could not draw charts:", repr(e))

    print("\nClustering part 1 complete. Next: cluster-based search (part 2).")


if __name__ == "__main__":
    main()
