# ============================================================
# services/hybrid.py  -  HYBRID REPRESENTATION  (Step 5, requirement 2)
#
# Combines a LEXICAL model (BM25) with a SEMANTIC model (Word2Vec):
#
#   SERIAL   : BM25 retrieves top-N candidates, embeddings RE-RANK them.
#   PARALLEL : both score independently, then a FUSION METHOD merges:
#                - "weighted" : alpha*minmax(bm25) + (1-alpha)*minmax(emb)
#                - "rrf"      : Reciprocal Rank Fusion (scale-free)
#
# Documents are referenced by doc_id throughout, so the BM25 index and
# the embedding matrix stay correctly aligned even if their internal
# ordering ever differs.
# ============================================================
import numpy as np


def _minmax(score_dict):
    """Scale a {id: score} dict into [0, 1]."""
    if not score_dict:
        return {}
    vals = list(score_dict.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {k: 1.0 for k in score_dict}
    return {k: (v - lo) / (hi - lo) for k, v in score_dict.items()}


class HybridSearcher:
    def __init__(self, bm25_model, emb_model, preprocessor, id_to_text=None):
        self.bm25 = bm25_model
        self.emb = emb_model
        self.prep = preprocessor
        self.id_to_text = id_to_text or {}
        self.emb_id2row = {did: i for i, did in enumerate(emb_model.doc_ids)}

    # ---------- helpers ----------
    def _result(self, doc_id, score):
        return {"doc_id": doc_id, "score": round(float(score), 4),
                "text": self.id_to_text.get(doc_id, "")}

    def _bm25_by_id(self, q_tokens):
        raw = self.bm25.get_scores(q_tokens)              # {doc_idx: score}
        return {self.bm25.inv.doc_ids[i]: s for i, s in raw.items()}

    def _emb_scores(self, q_tokens):
        q = self.emb.transform(q_tokens)                  # normalized
        if not np.any(q):
            return None
        return self.emb.doc_matrix @ q                    # (N,) cosine scores

    # ---------- SERIAL: BM25 -> embedding re-rank ----------
    def serial(self, query, top_k=10, candidates=100):
        q_tokens = self.prep.process(query, mode="lexical")
        if not q_tokens:
            return []
        bm = self._bm25_by_id(q_tokens)
        if not bm:
            return []
        cand = sorted(bm.items(), key=lambda kv: kv[1], reverse=True)[:candidates]

        emb_scores = self._emb_scores(q_tokens)
        if emb_scores is None:                            # no embeddable terms
            return [self._result(d, s) for d, s in cand[:top_k]]

        reranked = []
        for doc_id, _ in cand:
            row = self.emb_id2row.get(doc_id)
            es = float(emb_scores[row]) if row is not None else 0.0
            reranked.append((doc_id, es))
        reranked.sort(key=lambda kv: kv[1], reverse=True)
        return [self._result(d, s) for d, s in reranked[:top_k]]

    # ---------- PARALLEL: both score -> fuse ----------
    def parallel(self, query, top_k=10, fusion="rrf", alpha=0.5, pool=1000, rrf_k=60):
        q_tokens = self.prep.process(query, mode="lexical")
        if not q_tokens:
            return []
        bm = self._bm25_by_id(q_tokens)
        emb_scores = self._emb_scores(q_tokens)

        bm_top = sorted(bm.items(), key=lambda kv: kv[1], reverse=True)[:pool]
        if emb_scores is not None:
            top_idx = np.argsort(emb_scores)[::-1][:pool]
            emb_top = [(self.emb.doc_ids[i], float(emb_scores[i])) for i in top_idx]
        else:
            emb_top = []

        if fusion == "weighted":
            bm_n, emb_n = _minmax(dict(bm_top)), _minmax(dict(emb_top))
            ids = set(bm_n) | set(emb_n)
            fused = {d: alpha * bm_n.get(d, 0.0) + (1 - alpha) * emb_n.get(d, 0.0)
                     for d in ids}
        else:  # reciprocal rank fusion
            fused = {}
            for rank, (d, _) in enumerate(bm_top):
                fused[d] = fused.get(d, 0.0) + 1.0 / (rrf_k + rank + 1)
            for rank, (d, _) in enumerate(emb_top):
                fused[d] = fused.get(d, 0.0) + 1.0 / (rrf_k + rank + 1)

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [self._result(d, s) for d, s in ranked]

    # ---------- single entry point (used by the UI later) ----------
    def search(self, query, mode="parallel", **kw):
        if mode == "serial":
            return self.serial(query, top_k=kw.get("top_k", 10),
                                candidates=kw.get("candidates", 100))
        return self.parallel(query, top_k=kw.get("top_k", 10),
                             fusion=kw.get("fusion", "rrf"),
                             alpha=kw.get("alpha", 0.5),
                             pool=kw.get("pool", 1000),
                             rrf_k=kw.get("rrf_k", 60))
