# ============================================================
# scripts/run_step2_search.py
#
# Interactive TF-IDF search. Type a question, get the 10 most
# relevant documents ranked by cosine similarity.
#
# Run from the project root:
#   python -m scripts.run_step2_search
# ============================================================
import sys
import os
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from services.preprocessing import Preprocessor
from services.representation_tfidf import TfidfRepresentation
from services.search import TfidfSearcher


def load_id_to_text(path):
    """Map doc_id -> original text, for displaying results."""
    id_to_text = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            id_to_text[rec["doc_id"]] = rec.get("text", "")
    return id_to_text


def main():
    if not os.path.isdir(config.TFIDF_DIR):
        print("ERROR: TF-IDF artifacts not found. Run:  python -m scripts.run_step2_build")
        return

    print("Loading TF-IDF model and documents ...")
    t0 = time.time()
    tfidf = TfidfRepresentation.load(config.TFIDF_DIR)
    id_to_text = load_id_to_text(config.PROCESSED_DOCS_PATH)
    prep = Preprocessor(**config.PREP)
    searcher = TfidfSearcher(tfidf, prep, id_to_text)
    print(f"  ready in {time.time()-t0:.1f}s  ({tfidf.shape[0]:,} documents indexed)")

    print("\nType a query and press Enter. Type 'exit' to quit.\n")
    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in ("exit", "quit", ""):
            break

        t0 = time.time()
        results = searcher.search(query, top_k=config.TOP_K)
        dt = (time.time() - t0) * 1000

        if not results:
            print("  (no matching documents)\n")
            continue
        print(f"  top {len(results)} results in {dt:.0f} ms:")
        for rank, r in enumerate(results, 1):
            snippet = r["text"][:90].replace("\n", " ")
            print(f"   {rank:>2}. [{r['score']:.4f}] (doc {r['doc_id']}) {snippet}")
        print()


if __name__ == "__main__":
    main()
