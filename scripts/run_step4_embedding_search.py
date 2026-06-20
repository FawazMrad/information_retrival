# ============================================================
# scripts/run_step4_embedding_search.py
#
# Interactive semantic search using Word2Vec mean-vector cosine.
# Try paraphrases that share NO words with the documents to see the
# difference from lexical search (BM25/TF-IDF).
#
# Run from the project root:
#   python -m scripts.run_step4_embedding_search
# ============================================================
import sys
import os
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from services.representation_embedding import Word2VecEmbedding
from services.preprocessing import Preprocessor
from services.search import EmbeddingSearcher


def load_id_to_text(path):
    id_to_text = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            id_to_text[rec["doc_id"]] = rec.get("text", "")
    return id_to_text


def main():
    if not os.path.isdir(config.EMBEDDING_DIR):
        print("ERROR: embeddings not found. Run:  python -m scripts.run_step4_build_embeddings")
        return

    print("Loading Word2Vec model + document vectors ...")
    t0 = time.time()
    emb = Word2VecEmbedding.load(config.EMBEDDING_DIR)
    id_to_text = load_id_to_text(config.PROCESSED_DOCS_PATH)
    prep = Preprocessor(**config.PREP)
    searcher = EmbeddingSearcher(emb, prep, id_to_text)
    print(f"  ready in {time.time()-t0:.1f}s  "
          f"({emb.doc_matrix.shape[0]:,} documents, dim {emb.dim})")

    print("\nType a query and press Enter. Type 'exit' to quit.\n")
    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in ("exit", "quit"):
            break

        t0 = time.time()
        results = searcher.search(query, top_k=config.TOP_K)
        dt = (time.time() - t0) * 1000
        if not results:
            print("  (no matching documents)\n"); continue
        print(f"  top {len(results)} results in {dt:.0f} ms:")
        for rank, r in enumerate(results, 1):
            snippet = r["text"][:90].replace("\n", " ")
            print(f"   {rank:>2}. [{r['score']:.4f}] (doc {r['doc_id']}) {snippet}")
        print()


if __name__ == "__main__":
    main()
