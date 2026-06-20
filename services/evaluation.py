# ============================================================
# services/evaluation.py  -  EVALUATION  (Step 6, requirement 8)
#
# Standard IR metrics implemented from scratch (no external eval lib):
#   - Precision@k
#   - Recall@k
#   - Average Precision  -> MAP (mean over queries)
#   - nDCG@k  (graded: gain = 2^rel - 1, so it also works for binary qrels)
#
# run   : {query_id: [ranked doc_id, ...]}
# qrels : {query_id: {doc_id: relevance}}
# ============================================================
import numpy as np


def precision_at_k(ranked, relevant_set, k):
    if k <= 0:
        return 0.0
    hits = sum(1 for d in ranked[:k] if d in relevant_set)
    return hits / k


def recall_at_k(ranked, relevant_set, k):
    if not relevant_set:
        return 0.0
    hits = sum(1 for d in ranked[:k] if d in relevant_set)
    return hits / len(relevant_set)


def average_precision(ranked, relevant_set):
    if not relevant_set:
        return 0.0
    hits, running = 0, 0.0
    for i, d in enumerate(ranked, start=1):
        if d in relevant_set:
            hits += 1
            running += hits / i
    return running / len(relevant_set)


def _dcg(gains):
    return sum(g / np.log2(i + 1) for i, g in enumerate(gains, start=1))


def ndcg_at_k(ranked, rel_map, k):
    gains = [(2 ** rel_map.get(d, 0) - 1) for d in ranked[:k]]
    ideal = sorted((2 ** r - 1) for r in rel_map.values())
    ideal = ideal[::-1][:k]
    idcg = _dcg(ideal)
    return (_dcg(gains) / idcg) if idcg > 0 else 0.0


def evaluate(run, qrels, p_k=10, ndcg_k=10, recall_k=100):
    """Aggregate metrics (means) over all queries that have judgments."""
    aps, precs, recs, ndcgs = [], [], [], []
    for qid, ranked in run.items():
        rel_map = qrels.get(qid, {})
        relevant_set = {d for d, r in rel_map.items() if r > 0}
        if not relevant_set:
            continue
        aps.append(average_precision(ranked, relevant_set))
        precs.append(precision_at_k(ranked, relevant_set, p_k))
        recs.append(recall_at_k(ranked, relevant_set, recall_k))
        ndcgs.append(ndcg_at_k(ranked, rel_map, ndcg_k))
    n = len(aps)
    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    return {
        "queries": n,
        "MAP": mean(aps),
        f"P@{p_k}": mean(precs),
        f"Recall@{recall_k}": mean(recs),
        f"nDCG@{ndcg_k}": mean(ndcgs),
    }
