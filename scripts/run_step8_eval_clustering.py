# ============================================================
# scripts/run_step8_eval_clustering.py
#
# "Before vs after" evaluation for the clustering feature (req 8):
# compares full semantic search against cluster-pruned search on
# MAP / nDCG@10 / Recall@100 and average query time, and saves a
# before/after chart.
#
# Run from the project root:
#   python -m scripts.run_step8_eval_clustering
# ============================================================
import sys
import os
import random
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from services import data_loader as dl
from services.preprocessing import Preprocessor
from services.representation_embedding import Word2VecEmbedding
from services.clustering import DocumentClustering
from services.search import EmbeddingSearcher, ClusterSearcher
from services.evaluation import evaluate


def run_and_time(search_fn, sampled, cutoff):
    run, total = {}, 0.0
    try:
        from tqdm import tqdm
        it = tqdm(sampled, leave=False)
    except Exception:
        it = sampled
    for qid, qtext in it:
        t0 = time.time()
        results = search_fn(qtext, cutoff)
        total += (time.time() - t0)
        run[qid] = [r["doc_id"] for r in results]
    avg_ms = (total / max(len(sampled), 1)) * 1000
    return run, avg_ms


def main():
    cfg = config.EVAL
    ccfg = config.CLUSTER
    cutoff = cfg["cutoff"]

    print("Loading queries + qrels ...")
    ds = dl.load_dataset(config.DATASET_ID)
    qrels = dl.load_qrels(ds)
    queries = {qid: text for qid, text in dl.iter_queries(ds)}
    judged = [q for q in queries if q in qrels and qrels[q]]
    random.seed(cfg["seed"]); random.shuffle(judged)
    sampled = [(q, queries[q]) for q in judged[:cfg["sample_queries"]]]
    print(f"  evaluating on {len(sampled)} queries")

    print("Loading embeddings + clusters ...")
    prep = Preprocessor(**config.PREP)
    emb = Word2VecEmbedding.load(config.EMBEDDING_DIR)
    clustering = DocumentClustering.load(config.CLUSTER_DIR)
    full = EmbeddingSearcher(emb, prep)
    clustered = ClusterSearcher(emb, clustering, prep)
    probe = ccfg["probe_clusters"]

    configs = {
        "W2V (no clustering)": lambda q, k: full.search(q, top_k=k),
        f"W2V + clustering (probe={probe})": lambda q, k: clustered.search(q, top_k=k, probe=probe),
    }

    results = {}
    ndcg_key = f"nDCG@{cfg['ndcg_k']}"
    recall_key = f"Recall@{cfg['recall_k']}"
    for name, fn in configs.items():
        print(f"\nEvaluating: {name}")
        run, avg_ms = run_and_time(fn, sampled, cutoff)
        m = evaluate(run, qrels, p_k=cfg["p_k"], ndcg_k=cfg["ndcg_k"], recall_k=cfg["recall_k"])
        m["avg_ms"] = round(avg_ms, 1)
        results[name] = m
        print(f"  MAP={m['MAP']:.4f}  {ndcg_key}={m[ndcg_key]:.4f}  "
              f"{recall_key}={m[recall_key]:.4f}  avg={m['avg_ms']}ms")

    # table
    cols = ["MAP", f"nDCG@{cfg['ndcg_k']}", f"Recall@{cfg['recall_k']}", "avg_ms"]
    print("\n" + "=" * 74)
    print("BEFORE vs AFTER clustering")
    print("=" * 74)
    print(f"{'Config':<30}" + "".join(f"{c:>11}" for c in cols))
    for name, m in results.items():
        print(f"{name:<30}" + "".join(f"{m[c]:>11.4f}" for c in cols))

    # chart
    try:
        make_chart(results, cfg, config.EVAL_DIR)
        print("\nSaved chart ->", os.path.join(config.EVAL_DIR, "clustering_before_after.png"))
    except Exception as e:
        print("[warning] chart failed:", repr(e))


def make_chart(results, cfg, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    names = list(results.keys())
    metrics = ["MAP", f"nDCG@{cfg['ndcg_k']}", f"Recall@{cfg['recall_k']}", "avg_ms"]
    titles = ["MAP", f"nDCG@{cfg['ndcg_k']}", f"Recall@{cfg['recall_k']}", "avg query time (ms)"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    colors = ["#4C78A8", "#F58518"]
    for ax, metric, title in zip(axes, metrics, titles):
        vals = [results[n][metric] for n in names]
        bars = ax.bar(names, vals, color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    fig.suptitle("Clustering feature: before vs after", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out_dir, "clustering_before_after.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
