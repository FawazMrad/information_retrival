# ============================================================
# scripts/run_step4_build_embeddings.py
#
# Trains Word2Vec on the preprocessed corpus and builds the document
# embedding matrix, then saves both.
#
# Run from the project root:
#   python -m scripts.run_step4_build_embeddings
# ============================================================
import sys
import os
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from services.representation_embedding import Word2VecEmbedding


def load_tokens(path):
    doc_ids, token_lists = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            doc_ids.append(rec["doc_id"])
            token_lists.append(rec.get("lexical", "").split())
    return doc_ids, token_lists


def main():
    path = config.PROCESSED_DOCS_PATH
    if not os.path.exists(path):
        print("ERROR: processed file not found. Run Step 1 first.")
        return

    print("Loading preprocessed corpus ...")
    t0 = time.time()
    doc_ids, token_lists = load_tokens(path)
    print(f"  loaded {len(doc_ids):,} documents in {time.time()-t0:.1f}s")

    print("\nTraining Word2Vec ...", config.W2V)
    t0 = time.time()
    emb = Word2VecEmbedding.train(token_lists, **config.W2V)
    print(f"  trained in {time.time()-t0:.1f}s  "
          f"(vocabulary: {len(emb.model.wv):,} words, dim: {emb.dim})")

    print("\nBuilding document embedding matrix ...")
    t0 = time.time()
    emb.build_doc_matrix(doc_ids, token_lists)
    print(f"  built in {time.time()-t0:.1f}s  matrix shape: {emb.doc_matrix.shape}  "
          f"(~{emb.doc_matrix.nbytes/1e6:.0f} MB)")

    emb.save(config.EMBEDDING_DIR)
    print("\nSaved ->", config.EMBEDDING_DIR)
    print("Next: python -m scripts.run_step4_embedding_search")


if __name__ == "__main__":
    main()
