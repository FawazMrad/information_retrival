# ============================================================
# scripts/run_step1_preprocessing.py
#
# Usage (run from the project root folder ir_project/):
#   python -m scripts.run_step1_preprocessing                 # stats + samples
#   python -m scripts.run_step1_preprocessing --full          # + save whole corpus
#   python -m scripts.run_step1_preprocessing --full --skip-samples
# ============================================================
import sys
import os
import json
import time
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from services import data_loader as dl
from services.preprocessing import Preprocessor


def p(*a, **k):
    """print that always flushes (so output shows up immediately in PowerShell)."""
    k["flush"] = True
    print(*a, **k)


def show_doc_samples(ds, prep, n=3):
    p("\n" + "=" * 70)
    p("SAMPLE DOCUMENTS  (raw -> lexical -> semantic)")
    p("=" * 70)
    try:
        for doc_id, text in dl.iter_docs(ds, limit=n):
            p(f"\n[doc_id={doc_id}]")
            p("  RAW     :", text[:160])
            p("  LEXICAL :", prep.process(text, mode="lexical")[:25])
            p("  SEMANTIC:", prep.clean_for_embeddings(text)[:160])
    except Exception as e:
        p("[warning] could not show document samples:", repr(e))


def show_query_samples(ds, prep, n=3):
    p("\n" + "=" * 70)
    p("SAMPLE QUERIES  (raw -> lexical)")
    p("=" * 70)
    p("(a short 'opening zip file' pause here is normal the first time)")
    try:
        for qid, text in dl.iter_queries(ds, limit=n):
            p(f"\n[query_id={qid}]")
            p("  RAW    :", text)
            p("  LEXICAL:", prep.process(text, mode="lexical"))
    except Exception as e:
        p("[warning] could not show query samples:", repr(e))


def run_full(ds, prep, out_path):
    p("\n" + "=" * 70)
    p("FULL PREPROCESSING -> saving to:", out_path)
    p("=" * 70)
    t0 = time.time()
    n = 0
    try:
        from tqdm import tqdm
        iterator = tqdm(dl.iter_docs(ds), total=ds.docs_count(), desc="preprocessing")
    except Exception:
        iterator = dl.iter_docs(ds)

    with open(out_path, "w", encoding="utf-8") as f:
        for doc_id, text in iterator:
            rec = {
                "doc_id": doc_id,
                "lexical": prep.process_to_string(text, mode="lexical"),
                "text": text,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    dt = time.time() - t0
    p(f"\nDone. Preprocessed {n:,} documents in {dt:.1f}s "
      f"({n / max(dt, 1e-9):.0f} docs/sec).")
    p("Output file:", out_path)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="preprocess and save the entire corpus")
    ap.add_argument("--skip-samples", action="store_true",
                    help="skip the sample printouts and go straight to processing")
    args = ap.parse_args()

    p("Loading dataset:", config.DATASET_ID)
    ds = dl.load_dataset(config.DATASET_ID)

    stats = dl.dataset_stats(ds)
    p("\nDATASET STATS")
    p("  documents :", f"{stats['num_docs']:,}" if stats["num_docs"] else "n/a")
    p("  queries   :", f"{stats['num_queries']:,}" if stats["num_queries"] else "n/a")
    p("  qrels     :", f"{stats['num_qrels']:,}" if stats["num_qrels"] else "n/a")

    prep = Preprocessor(**config.PREP)

    if not args.skip_samples:
        show_doc_samples(ds, prep, n=3)
        show_query_samples(ds, prep, n=3)

    # Full processing runs no matter what happened in the sample steps above.
    if args.full:
        run_full(ds, prep, config.PROCESSED_DOCS_PATH)
    else:
        p("\n(Run again with --full to preprocess and save the whole corpus.)")


if __name__ == "__main__":
    main()
