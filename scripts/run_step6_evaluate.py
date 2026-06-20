# ============================================================
# scripts/run_step6_evaluate.py
#
# Evaluates every representation model on a sample of the test queries
# and reports MAP, nDCG@10, P@10, Recall@100. Saves a CSV/JSON table
# and comparison charts (PNG) for the report.
#
# Run from the project root:
#   python -m scripts.run_step6_evaluate
# ============================================================
import sys
import os
import json
import argparse
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from services import data_loader as dl
from services.preprocessing import Preprocessor
from services.indexing import InvertedIndex
from services.representation_tfidf import TfidfRepresentation
from services.representation_bm25 import BM25
from services.representation_embedding import Word2VecEmbedding
from services.search import TfidfSearcher, BM25Searcher, EmbeddingSearcher
from services.hybrid import HybridSearcher
from services.evaluation import evaluate


def build_run(search_fn, sampled, cutoff):
    """search_fn(query_text) -> list of result dicts. Returns {qid: [doc_ids]}."""
    run = {}
    try:
        from tqdm import tqdm
        it = tqdm(sampled, leave=False)
    except Exception:
        it = sampled
    for qid, qtext in it:
        results = search_fn(qtext, cutoff)
        run[qid] = [r["doc_id"] for r in results]
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0,
                    help="evaluate on only N queries (0 = ALL judged queries, the default)")
    args = ap.parse_args()

    cfg = config.EVAL
    cutoff = cfg["cutoff"]
    ndcg_key = f"nDCG@{cfg['ndcg_k']}"
    p_key = f"P@{cfg['p_k']}"
    recall_key = f"Recall@{cfg['recall_k']}"

    # ---------- load queries + qrels ----------
    print("Loading dataset queries + qrels:", config.DATASET_ID)
    ds = dl.load_dataset(config.DATASET_ID)
    qrels = dl.load_qrels(ds)
    queries = {qid: text for qid, text in dl.iter_queries(ds)}

    total_judgments = sum(len(v) for v in qrels.values())
    judged = [qid for qid in queries if qid in qrels and qrels[qid]]
    judged.sort()                       # deterministic order (no random sampling)

    if args.sample and args.sample > 0:
        eval_ids = judged[:args.sample]
    else:
        eval_ids = judged               # ALL judged queries (required for grading)
    sampled = [(qid, queries[qid]) for qid in eval_ids]

    # ---------- print the counts the TA will check ----------
    print("\n" + "=" * 70)
    print("EVALUATION QUERY COUNT")
    print("=" * 70)
    print(f"  Test queries in dataset      : {len(queries):,}")
    print(f"  Unique queries in qrels      : {len(judged):,}")
    print(f"  Total relevance judgments    : {total_judgments:,}")
    print(f"  >>> QUERIES USED IN THIS RUN : {len(sampled):,}", 
          "(ALL judged queries)" if not args.sample else f"(--sample {args.sample})")
    print("=" * 70)

    # ---------- load models ----------
    print("Loading models ...")
    prep = Preprocessor(**config.PREP)
    inv = InvertedIndex.load(config.INVERTED_INDEX_PATH)
    bm25 = BM25(inv, k1=config.BM25_K1, b=config.BM25_B)
    tfidf = TfidfRepresentation.load(config.TFIDF_DIR)
    emb = Word2VecEmbedding.load(config.EMBEDDING_DIR)

    tfidf_s = TfidfSearcher(tfidf, prep)
    bm25_s = BM25Searcher(bm25, prep)
    emb_s = EmbeddingSearcher(emb, prep)
    hybrid = HybridSearcher(bm25, emb, prep)

    methods = {
        "TF-IDF":           lambda q, k: tfidf_s.search(q, top_k=k),
        "BM25":             lambda q, k: bm25_s.search(q, top_k=k),
        "Word2Vec":         lambda q, k: emb_s.search(q, top_k=k),
        "Hybrid-Serial":    lambda q, k: hybrid.serial(q, top_k=k, candidates=100),
        "Hybrid-Par-RRF":   lambda q, k: hybrid.parallel(q, top_k=k, fusion="rrf"),
        "Hybrid-Par-Weight": lambda q, k: hybrid.parallel(q, top_k=k, fusion="weighted",
                                                          alpha=cfg["alpha"]),
    }

    # ---------- evaluate ----------
    results = {}
    for name, fn in methods.items():
        print(f"\nEvaluating: {name}")
        t0 = time.time()
        run = build_run(fn, sampled, cutoff)
        metrics = evaluate(run, qrels, p_k=cfg["p_k"],
                           ndcg_k=cfg["ndcg_k"], recall_k=cfg["recall_k"])
        metrics["seconds"] = round(time.time() - t0, 1)
        results[name] = metrics
        print(f"  MAP={metrics['MAP']:.4f}  {ndcg_key}={metrics[ndcg_key]:.4f}  "
              f"{p_key}={metrics[p_key]:.4f}  {recall_key}={metrics[recall_key]:.4f}  "
              f"({metrics['seconds']}s)")

    # ---------- print table ----------
    print("\n" + "=" * 78)
    print("RESULTS  (dataset: %s, %d queries)" % (config.DATASET_NAME, len(sampled)))
    print("=" * 78)
    cols = ["MAP", f"nDCG@{cfg['ndcg_k']}", f"P@{cfg['p_k']}", f"Recall@{cfg['recall_k']}"]
    header = f"{'Method':<20}" + "".join(f"{c:>12}" for c in cols)
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:<20}" + "".join(f"{m[c]:>12.4f}" for c in cols))

    # ---------- save table ----------
    os.makedirs(config.EVAL_DIR, exist_ok=True)
    json_path = os.path.join(config.EVAL_DIR, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    csv_path = os.path.join(config.EVAL_DIR, "results.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Method," + ",".join(cols) + "\n")
        for name, m in results.items():
            f.write(name + "," + ",".join(f"{m[c]:.4f}" for c in cols) + "\n")
    print("\nSaved table ->", csv_path)

    # ---------- charts ----------
    try:
        make_charts(results, cols, config.EVAL_DIR)
        print("Saved charts ->", config.EVAL_DIR)
    except Exception as e:
        print("[warning] could not draw charts:", repr(e))
        print("  (install matplotlib:  python -m pip install matplotlib)")


def make_charts(results, cols, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric in zip(axes.ravel(), cols):
        vals = [results[n][metric] for n in names]
        bars = ax.bar(names, vals, color="#4C78A8")
        ax.set_title(metric)
        ax.set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 1)
        ax.tick_params(axis="x", rotation=45)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"Retrieval metrics on {config.DATASET_NAME} "
                 f"({results[names[0]]['queries']} queries)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(out_dir, "metrics_comparison.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
