# ============================================================
# services/indexing.py  -  INDEXING SERVICE  (Step 2, requirement 3)
#
# Builds an INVERTED INDEX from the preprocessed (lexical) tokens:
#   term -> [(doc_idx, tf), ...]
# Also stores everything BM25 will need later: document frequency
# (df) per term, document lengths, and the average document length.
# ============================================================
import pickle
from collections import Counter


class InvertedIndex:
    def __init__(self):
        self.index = {}        # term -> list of (doc_idx, term_frequency)
        self.df = {}           # term -> document frequency (in how many docs)
        self.doc_ids = []      # doc_idx -> original doc_id
        self.doc_len = []      # doc_idx -> number of tokens in that doc
        self.N = 0             # total number of documents
        self.avg_doc_len = 0.0

    def build(self, doc_token_iter):
        """doc_token_iter yields (doc_id, [token, token, ...])."""
        index, df, doc_ids, doc_len = {}, {}, [], []
        for doc_id, tokens in doc_token_iter:
            idx = len(doc_ids)
            doc_ids.append(doc_id)
            doc_len.append(len(tokens))
            for term, tf in Counter(tokens).items():
                index.setdefault(term, []).append((idx, tf))
                df[term] = df.get(term, 0) + 1
        self.index = index
        self.df = df
        self.doc_ids = doc_ids
        self.doc_len = doc_len
        self.N = len(doc_ids)
        self.avg_doc_len = (sum(doc_len) / self.N) if self.N else 0.0
        return self

    # --- lookups ---
    def postings(self, term):
        """Return [(doc_idx, tf), ...] for a term."""
        return self.index.get(term, [])

    def candidate_docs(self, query_terms):
        """All doc indices that contain at least one query term (fast retrieval)."""
        cands = set()
        for t in query_terms:
            for doc_idx, _ in self.index.get(t, []):
                cands.add(doc_idx)
        return cands

    # --- stats (useful for the report) ---
    def vocab_size(self):
        return len(self.index)

    def num_postings(self):
        return sum(len(p) for p in self.index.values())

    # --- persistence ---
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({
                "index": self.index, "df": self.df, "doc_ids": self.doc_ids,
                "doc_len": self.doc_len, "N": self.N, "avg_doc_len": self.avg_doc_len,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        obj = cls()
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj.__dict__.update(d)
        return obj
