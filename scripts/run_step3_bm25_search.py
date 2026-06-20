# ============================================================
# scripts/run_step3_bm25_search.py
#
# Interactive BM25 search with LIVE parameter control (req 2, note 2).
#   - just type a query to search
#   - type   :k1 2.0    to change k1
#   - type   :b 0.5     to change b
#   - type   :show      to see current parameters
#   - type   exit       to quit
#
# Run from the project root:
#   python -m scripts.run_step3_bm25_search
# ============================================================
import sys
import os
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from services.indexing import InvertedIndex
from services.representation_bm25 import BM25
from services.preprocessing import Preprocessor
from services.search import BM25Searcher


def load_id_to_text(path):
    id_to_text = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            id_to_text[rec["doc_id"]] = rec.get("text", "")
    return id_to_text


def main():
    if not os.path.exists(config.INVERTED_INDEX_PATH):
        print("ERROR: inverted index not found. Run:  python -m scripts.run_step2_build")
        return

    print("Loading inverted index + documents ...")
    t0 = time.time()
    inv = InvertedIndex.load(config.INVERTED_INDEX_PATH)
    bm25 = BM25(inv, k1=config.BM25_K1, b=config.BM25_B)
    id_to_text = load_id_to_text(config.PROCESSED_DOCS_PATH)
    prep = Preprocessor(**config.PREP)
    searcher = BM25Searcher(bm25, prep, id_to_text)
    print(f"  ready in {time.time()-t0:.1f}s  ({inv.N:,} documents, "
          f"{inv.vocab_size():,} terms)")
    print(f"  current parameters: k1={bm25.k1}, b={bm25.b}")

    print("\nCommands:  :k1 <value>   :b <value>   :show   exit")
    print("Otherwise, type a query.\n")

    while True:
        try:
            line = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line or line.lower() in ("exit", "quit"):
            break

        # --- parameter commands ---
        if line.startswith(":"):
            parts = line[1:].split()
            cmd = parts[0].lower() if parts else ""
            if cmd == "k1" and len(parts) == 2:
                bm25.set_params(k1=parts[1]); print(f"  k1 set to {bm25.k1}\n")
            elif cmd == "b" and len(parts) == 2:
                bm25.set_params(b=parts[1]); print(f"  b set to {bm25.b}\n")
            elif cmd == "show":
                print(f"  k1={bm25.k1}, b={bm25.b}\n")
            else:
                print("  usage: :k1 <value> | :b <value> | :show\n")
            continue

        # --- search ---
        t0 = time.time()
        results = searcher.search(line, top_k=config.TOP_K)
        dt = (time.time() - t0) * 1000
        if not results:
            print("  (no matching documents)\n"); continue
        print(f"  top {len(results)} results in {dt:.0f} ms  (k1={bm25.k1}, b={bm25.b}):")
        for rank, r in enumerate(results, 1):
            snippet = r["text"][:90].replace("\n", " ")
            print(f"   {rank:>2}. [{r['score']:.3f}] (doc {r['doc_id']}) {snippet}")
        print()


if __name__ == "__main__":
    main()
