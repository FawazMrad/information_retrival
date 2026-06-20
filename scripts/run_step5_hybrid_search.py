# ============================================================
# scripts/run_step5_hybrid_search.py
#
# Interactive HYBRID search (BM25 + Word2Vec).
#   :mode serial          -> BM25 retrieve, embeddings re-rank
#   :mode parallel        -> both score, then fuse
#   :fusion rrf           -> reciprocal rank fusion (parallel)
#   :fusion weighted      -> weighted sum of normalized scores (parallel)
#   :alpha 0.6            -> weight on BM25 in weighted fusion (0..1)
#   :show                 -> show current settings
#   exit                  -> quit
#
# Run from the project root:
#   python -m scripts.run_step5_hybrid_search
# ============================================================
import sys
import os
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from services.indexing import InvertedIndex
from services.representation_bm25 import BM25
from services.representation_embedding import Word2VecEmbedding
from services.preprocessing import Preprocessor
from services.hybrid import HybridSearcher


def load_id_to_text(path):
    id_to_text = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            id_to_text[rec["doc_id"]] = rec.get("text", "")
    return id_to_text


def main():
    if not os.path.exists(config.INVERTED_INDEX_PATH) or not os.path.isdir(config.EMBEDDING_DIR):
        print("ERROR: need both the inverted index (Step 2) and embeddings (Step 4).")
        return

    print("Loading BM25 index + Word2Vec embeddings + documents ...")
    t0 = time.time()
    inv = InvertedIndex.load(config.INVERTED_INDEX_PATH)
    bm25 = BM25(inv, k1=config.BM25_K1, b=config.BM25_B)
    emb = Word2VecEmbedding.load(config.EMBEDDING_DIR)
    id_to_text = load_id_to_text(config.PROCESSED_DOCS_PATH)
    prep = Preprocessor(**config.PREP)
    hybrid = HybridSearcher(bm25, emb, prep, id_to_text)
    print(f"  ready in {time.time()-t0:.1f}s  ({inv.N:,} documents)")

    settings = {"mode": "parallel", "fusion": "rrf", "alpha": 0.5}
    print(f"\nStart settings: {settings}")
    print("Commands: :mode serial|parallel  :fusion rrf|weighted  :alpha 0..1  :show  exit\n")

    while True:
        try:
            line = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line or line.lower() in ("exit", "quit"):
            break

        if line.startswith(":"):
            parts = line[1:].split()
            cmd = parts[0].lower() if parts else ""
            if cmd == "mode" and len(parts) == 2 and parts[1] in ("serial", "parallel"):
                settings["mode"] = parts[1]
            elif cmd == "fusion" and len(parts) == 2 and parts[1] in ("rrf", "weighted"):
                settings["fusion"] = parts[1]
            elif cmd == "alpha" and len(parts) == 2:
                settings["alpha"] = float(parts[1])
            elif cmd == "show":
                pass
            else:
                print("  usage: :mode serial|parallel  :fusion rrf|weighted  :alpha 0..1\n")
                continue
            print(f"  settings: {settings}\n")
            continue

        t0 = time.time()
        results = hybrid.search(line, top_k=config.TOP_K, **settings)
        dt = (time.time() - t0) * 1000
        if not results:
            print("  (no matching documents)\n"); continue
        tag = settings["mode"]
        if settings["mode"] == "parallel":
            tag += f"/{settings['fusion']}"
            if settings["fusion"] == "weighted":
                tag += f" alpha={settings['alpha']}"
        print(f"  top {len(results)} [{tag}] in {dt:.0f} ms:")
        for rank, r in enumerate(results, 1):
            snippet = r["text"][:90].replace("\n", " ")
            print(f"   {rank:>2}. [{r['score']:.4f}] (doc {r['doc_id']}) {snippet}")
        print()


if __name__ == "__main__":
    main()
