# ============================================================
# scripts/run_step7_query_refinement.py
#
# Demonstrates query refinement (req 5): spell-correction, synonym
# expansion (via Word2Vec), and session-history weighting. Shows the
# search results BEFORE and AFTER refinement so the effect is visible.
#
# Commands:  :spell on|off   :expand on|off   :hist on|off   :clear   exit
#
# Run from the project root:
#   python -m scripts.run_step7_query_refinement
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
from services.search import BM25Searcher
from services.query_refinement import QueryRefinement


def load_id_to_text(path):
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            d[rec["doc_id"]] = rec.get("text", "")
    return d


def show(title, results):
    print(f"  {title}")
    if not results:
        print("    (none)")
        return
    for rank, r in enumerate(results[:5], 1):
        print(f"    {rank}. [{r['score']:.3f}] {r['text'][:80]}")


def main():
    if not os.path.exists(config.INVERTED_INDEX_PATH) or not os.path.isdir(config.EMBEDDING_DIR):
        print("ERROR: need the inverted index (Step 2) and embeddings (Step 4).")
        return

    print("Loading index + embeddings + documents ...")
    t0 = time.time()
    inv = InvertedIndex.load(config.INVERTED_INDEX_PATH)
    bm25 = BM25(inv, k1=config.BM25_K1, b=config.BM25_B)
    emb = Word2VecEmbedding.load(config.EMBEDDING_DIR)
    id_to_text = load_id_to_text(config.PROCESSED_DOCS_PATH)
    prep = Preprocessor(**config.PREP)
    searcher = BM25Searcher(bm25, prep, id_to_text)
    qr = QueryRefinement(prep, vocab=inv.index.keys(), w2v=emb, df=inv.df)
    print(f"  ready in {time.time()-t0:.1f}s")

    opts = {"spell": True, "expand": True, "hist": True}
    history = []
    print("\nCommands: :spell on|off  :expand on|off  :hist on|off  :clear  exit")
    print("Try a typo, e.g.:  how to invset in indian stock markett\n")

    while True:
        try:
            line = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line or line.lower() in ("exit", "quit"):
            break
        if line.startswith(":"):
            p = line[1:].split()
            if p and p[0] in opts and len(p) == 2:
                opts[p[0]] = (p[1].lower() == "on")
                print(f"  {p[0]} = {opts[p[0]]}\n")
            elif p and p[0] == "clear":
                history.clear(); print("  history cleared\n")
            else:
                print("  usage: :spell on|off  :expand on|off  :hist on|off  :clear\n")
            continue

        refined, info = qr.refine(
            line, use_spell=opts["spell"], use_expand=opts["expand"],
            expand_n=2, history_tokens=history if opts["hist"] else None,
        )
        print("  original :", info["original"])
        if info["corrections"]:
            print("  corrected:", info["corrections"])
        if info["expansions"]:
            print("  expanded :", info["expansions"])
        print("  suggestion:", qr.suggest(line))

        base = searcher.search(" ".join(info["original"]), top_k=5)
        ref = searcher.search(" ".join(refined), top_k=5)
        print()
        show("BEFORE refinement:", base)
        show("AFTER refinement :", ref)
        print()

        # update session history with this query's (corrected) terms
        for t in info["original"]:
            if t not in history:
                history.append(t)
        history[:] = history[-12:]   # keep it short


if __name__ == "__main__":
    main()
