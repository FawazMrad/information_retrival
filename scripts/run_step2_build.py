# ============================================================
# scripts/run_step2_build.py
#
# Builds the inverted index + the TF-IDF representation from the
# preprocessed corpus produced in Step 1, and saves both to disk.
#
# Run from the project root:
#   python -m scripts.run_step2_build
# ============================================================
import sys
import os
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from services.indexing import InvertedIndex
from services.representation_tfidf import TfidfRepresentation


def load_processed(path):
    """Yield records from the Step-1 jsonl file."""
    doc_ids, lexical_texts, token_lists = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            lex = rec.get("lexical", "")
            doc_ids.append(rec["doc_id"])
            lexical_texts.append(lex)
            token_lists.append(lex.split())
    return doc_ids, lexical_texts, token_lists


def main():
    path = config.PROCESSED_DOCS_PATH
    if not os.path.exists(path):
        print("ERROR: processed file not found:", path)
        print("Run Step 1 first:  python -m scripts.run_step1_preprocessing --full --skip-samples")
        return

    print("Loading preprocessed corpus:", path)
    t0 = time.time()
    doc_ids, lexical_texts, token_lists = load_processed(path)
    print(f"  loaded {len(doc_ids):,} documents in {time.time()-t0:.1f}s")

    # ---------- Inverted index (requirement 3) ----------
    print("\nBuilding inverted index ...")
    t0 = time.time()
    inv = InvertedIndex().build(zip(doc_ids, token_lists))
    inv.save(config.INVERTED_INDEX_PATH)
    print(f"  done in {time.time()-t0:.1f}s")
    print(f"  documents     : {inv.N:,}")
    print(f"  vocabulary    : {inv.vocab_size():,} unique terms")
    print(f"  postings      : {inv.num_postings():,}")
    print(f"  avg doc length: {inv.avg_doc_len:.2f} tokens")
    # show a sample posting list so you can screenshot it for the report
    for sample_term in ("invest", "india", "python"):
        if sample_term in inv.index:
            postings = inv.postings(sample_term)[:5]
            print(f"  e.g. term '{sample_term}': df={inv.df[sample_term]:,}, "
                  f"first postings (doc_idx, tf) = {postings}")
            break
    print("  saved ->", config.INVERTED_INDEX_PATH)

    # ---------- TF-IDF / VSM (requirement 2) ----------
    print("\nBuilding TF-IDF representation ...")
    t0 = time.time()
    tfidf = TfidfRepresentation().build(doc_ids, lexical_texts)
    tfidf.save(config.TFIDF_DIR)
    print(f"  done in {time.time()-t0:.1f}s")
    print(f"  matrix shape  : {tfidf.shape}  (documents x terms)")
    print("  saved ->", config.TFIDF_DIR)

    print("\nStep 2 build complete. Next: python -m scripts.run_step2_search")


if __name__ == "__main__":
    main()
