# ============================================================
# scripts/run_step8_cluster_search.py
#
# Demonstrates the additional feature being INDEPENDENTLY testable:
# the same query is run WITHOUT clustering (full semantic search) and
# WITH clustering (only the nearest clusters), showing results, the
# probed clusters, and the speed difference.
#
#   :probe 3     -> how many nearest clusters to search
#   exit
#
# Run from the project root:
#   python -m scripts.run_step8_cluster_search
# ============================================================
import sys
import os
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from services.representation_embedding import Word2VecEmbedding
from services.clustering import DocumentClustering
from services.preprocessing import Preprocessor
from services.search import EmbeddingSearcher, ClusterSearcher


def load_id_to_text(path):
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            d[rec["doc_id"]] = rec.get("text", "")
    return d


def show(title, results, dt):
    print(f"  {title}  ({dt:.0f} ms)")
    for rank, r in enumerate(results[:5], 1):
        extra = f" [c{r['cluster']}]" if "cluster" in r else ""
        print(f"    {rank}. [{r['score']:.3f}]{extra} {r['text'][:75]}")


def main():
    if not os.path.isdir(config.EMBEDDING_DIR) or not os.path.isdir(config.CLUSTER_DIR):
        print("ERROR: need embeddings (Step 4) and clusters (Step 8 part 1).")
        return

    print("Loading embeddings + clusters + documents ...")
    t0 = time.time()
    emb = Word2VecEmbedding.load(config.EMBEDDING_DIR)
    clustering = DocumentClustering.load(config.CLUSTER_DIR)
    id_to_text = load_id_to_text(config.PROCESSED_DOCS_PATH)
    prep = Preprocessor(**config.PREP)
    full = EmbeddingSearcher(emb, prep, id_to_text)
    clustered = ClusterSearcher(emb, clustering, prep, id_to_text)
    print(f"  ready in {time.time()-t0:.1f}s  ({clustering.n_clusters} clusters)")

    probe = config.CLUSTER["probe_clusters"]
    print(f"\nprobe = {probe} nearest clusters.  Commands: :probe N   exit\n")

    while True:
        try:
            line = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line or line.lower() in ("exit", "quit"):
            break
        if line.startswith(":probe"):
            parts = line.split()
            if len(parts) == 2 and parts[1].isdigit():
                probe = int(parts[1]); print(f"  probe = {probe}\n")
            continue

        t0 = time.time(); r_full = full.search(line, top_k=5); dt_full = (time.time()-t0)*1000
        t0 = time.time(); r_clu = clustered.search(line, top_k=5, probe=probe); dt_clu = (time.time()-t0)*1000

        print()
        show("WITHOUT clustering (full search):", r_full, dt_full)
        show(f"WITH clustering (probe={probe}):", r_clu, dt_clu)
        if dt_clu > 0:
            print(f"  speedup: {dt_full/dt_clu:.1f}x\n")
        else:
            print()


if __name__ == "__main__":
    main()
