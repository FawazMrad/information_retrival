# ============================================================
# scripts/run_step0_offline_check.py
#
# Verifies the whole system can run WITHOUT internet (requirement 4).
# Run it ONCE with internet to warm the local caches (dataset + NLTK),
# then turn the internet OFF and run it again: everything must pass.
#
#   python -m scripts.run_step0_offline_check
# ============================================================
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


def check(label, fn):
    try:
        info = fn()
        print(f"  [OK]   {label}{(' - ' + info) if info else ''}")
        return True
    except Exception as e:
        print(f"  [FAIL] {label}  ->  {type(e).__name__}: {e}")
        return False


def main():
    print("Offline readiness check (run once online to cache, then offline to verify)\n")
    ok = True

    # 1) NLTK data (stopwords) is available locally
    def _nltk():
        from services.preprocessing import Preprocessor
        p = Preprocessor(**config.PREP)
        toks = p.process("How to invest in the stock market?", mode="lexical")
        return f"sample tokens: {toks}"
    ok &= check("NLTK preprocessing", _nltk)

    # 2) dataset + qrels load from the local ir_datasets cache
    def _ds():
        from services import data_loader as dl
        ds = dl.load_dataset(config.DATASET_ID)
        qrels = dl.load_qrels(ds)
        return f"{len(qrels):,} queries have judgments"
    ok &= check("Dataset + qrels (ir_datasets cache)", _ds)

    # 3) built artifacts exist on disk
    def _artifacts():
        missing = [pth for pth in [
            config.PROCESSED_DOCS_PATH, config.INVERTED_INDEX_PATH,
            config.TFIDF_DIR, config.EMBEDDING_DIR] if not os.path.exists(pth)]
        if missing:
            raise FileNotFoundError("missing: " + ", ".join(missing))
        return "all core artifacts present"
    ok &= check("Built artifacts (index / tfidf / embeddings)", _artifacts)

    print("\n" + ("ALL CHECKS PASSED - system is offline-ready."
                  if ok else "SOME CHECKS FAILED - run once WITH internet first."))


if __name__ == "__main__":
    main()
