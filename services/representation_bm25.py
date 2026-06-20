# ============================================================
# services/representation_bm25.py  -  BM25  (Step 3, requirement 2)
#
# Probabilistic ranking built directly on the inverted index from
# Step 2. Only documents that contain a query term are scored, so it
# is fast. k1 and b are live parameters (req 2, note 2): IDF does NOT
# depend on them, so you can change them at any time without rebuilding.
#
# score(d, q) = SUM over query terms t of
#     idf(t) * ( tf(t,d) * (k1 + 1) )
#               / ( tf(t,d) + k1 * (1 - b + b * |d| / avgdl) )
#
# idf(t) = ln( 1 + (N - df(t) + 0.5) / (df(t) + 0.5) )   # always positive
# ============================================================
import math
from collections import defaultdict


class BM25:
    def __init__(self, inverted_index, k1=1.5, b=0.75):
        self.inv = inverted_index
        self.k1 = float(k1)
        self.b = float(b)
        self.N = inverted_index.N
        self.avgdl = inverted_index.avg_doc_len or 1.0
        self.doc_len = inverted_index.doc_len
        self.idf = self._calc_idf(inverted_index.df, self.N)

    @staticmethod
    def _calc_idf(df_map, N):
        idf = {}
        for term, df in df_map.items():
            idf[term] = math.log(1.0 + (N - df + 0.5) / (df + 0.5))
        return idf

    def set_params(self, k1=None, b=None):
        """Change parameters at runtime (no rebuild needed)."""
        if k1 is not None:
            self.k1 = float(k1)
        if b is not None:
            self.b = float(b)

    def get_scores(self, query_terms):
        """Return {doc_idx: bm25_score} for docs containing >=1 query term."""
        scores = defaultdict(float)
        k1, b, avgdl = self.k1, self.b, self.avgdl
        for t in query_terms:
            postings = self.inv.index.get(t)
            if not postings:
                continue
            idf = self.idf.get(t, 0.0)
            for doc_idx, tf in postings:
                dl = self.doc_len[doc_idx]
                denom = tf + k1 * (1.0 - b + b * dl / avgdl)
                scores[doc_idx] += idf * (tf * (k1 + 1.0)) / denom
        return scores

    def rank(self, query_terms, top_k=10):
        """Return [(doc_id, score), ...] sorted by score descending."""
        scores = self.get_scores(query_terms)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [(self.inv.doc_ids[idx], float(sc)) for idx, sc in ranked]
