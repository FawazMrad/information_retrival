# ============================================================
# services/search.py  -  MATCHING & RANKING  (Step 2, requirement 6)
#
# Takes a raw query, preprocesses it with the SAME Preprocessor used
# on documents (requirement 4 -> query/doc compatibility), turns it
# into a TF-IDF vector, then ranks all documents by cosine similarity.
# ============================================================
import numpy as np


class TfidfSearcher:
    def __init__(self, tfidf_rep, preprocessor, id_to_text=None):
        self.rep = tfidf_rep
        self.prep = preprocessor
        self.id_to_text = id_to_text or {}

    def search(self, query, top_k=10):
        # 1) same preprocessing as documents
        q_tokens = self.prep.process(query, mode="lexical")
        q_str = " ".join(q_tokens)
        if not q_str.strip():
            return []

        # 2) represent the query in the same TF-IDF space
        q_vec = self.rep.transform(q_str)            # 1 x V, L2-normalized

        # 3) cosine similarity = dot product (rows are L2-normalized)
        scores = (self.rep.matrix @ q_vec.T).toarray().ravel()   # shape: (N,)

        # 4) rank and take the top-k positive scores
        order = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in order:
            s = float(scores[i])
            if s <= 0:
                break
            doc_id = self.rep.doc_ids[i]
            results.append({
                "doc_id": doc_id,
                "score": round(s, 4),
                "text": self.id_to_text.get(doc_id, ""),
            })
        return results


class BM25Searcher:
    """Same interface as TfidfSearcher, but ranks with the BM25 model."""

    def __init__(self, bm25_model, preprocessor, id_to_text=None):
        self.model = bm25_model
        self.prep = preprocessor
        self.id_to_text = id_to_text or {}

    def search(self, query, top_k=10):
        q_tokens = self.prep.process(query, mode="lexical")   # same preprocessing
        if not q_tokens:
            return []
        ranked = self.model.rank(q_tokens, top_k=top_k)
        results = []
        for doc_id, score in ranked:
            if score <= 0:
                continue
            results.append({
                "doc_id": doc_id,
                "score": round(score, 4),
                "text": self.id_to_text.get(doc_id, ""),
            })
        return results


class EmbeddingSearcher:
    """Same interface, but ranks by Word2Vec mean-vector cosine similarity."""

    def __init__(self, emb_model, preprocessor, id_to_text=None):
        self.emb = emb_model
        self.prep = preprocessor
        self.id_to_text = id_to_text or {}

    def search(self, query, top_k=10):
        q_tokens = self.prep.process(query, mode="lexical")   # same preprocessing
        q_vec = self.emb.transform(q_tokens)                  # L2-normalized
        if not np.any(q_vec):
            return []
        # cosine similarity (rows are normalized) = dot product
        scores = self.emb.doc_matrix @ q_vec                  # shape: (N,)
        order = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in order:
            s = float(scores[i])
            if s <= 0:
                break
            doc_id = self.emb.doc_ids[i]
            results.append({
                "doc_id": doc_id,
                "score": round(s, 4),
                "text": self.id_to_text.get(doc_id, ""),
            })
        return results


class ClusterSearcher:
    """Cluster-pruned semantic search: only score documents that live in the
    query's nearest clusters (the 'cluster hypothesis'). Faster than scanning
    the whole corpus; toggled on/off to compare with full search."""

    def __init__(self, emb_model, clustering, preprocessor, id_to_text=None):
        self.emb = emb_model
        self.cl = clustering
        self.prep = preprocessor
        self.id_to_text = id_to_text or {}

    def search(self, query, top_k=10, probe=3):
        q_tokens = self.prep.process(query, mode="lexical")
        q_vec = self.emb.transform(q_tokens)
        if not np.any(q_vec):
            return []
        cluster_ids = self.cl.nearest_clusters(q_vec, c=probe)
        idxs = self.cl.docs_in_clusters(cluster_ids)          # candidate doc rows
        if len(idxs) == 0:
            return []
        sub_scores = self.emb.doc_matrix[idxs] @ q_vec        # cosine on candidates
        order = np.argsort(sub_scores)[::-1][:top_k]
        results = []
        for j in order:
            s = float(sub_scores[j])
            if s <= 0:
                break
            doc_id = self.emb.doc_ids[idxs[j]]
            results.append({
                "doc_id": doc_id,
                "score": round(s, 4),
                "text": self.id_to_text.get(doc_id, ""),
                "cluster": int(self.cl.labels[idxs[j]]),
            })
        return results
