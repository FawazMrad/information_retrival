# ============================================================
# services/representation_embedding.py  -  EMBEDDINGS  (Step 4, req 2)
#
# Word2Vec trained on the corpus (gensim). A document/query vector is
# the L2-normalized MEAN of its word vectors -> cosine similarity = dot
# product. No external model download: it learns from your own corpus.
# (A BERT/sentence-transformers variant can be added later as a second
#  embedding model for the parallel hybrid - req 2, note 4.)
# ============================================================
import os
import numpy as np
import joblib


class Word2VecEmbedding:
    def __init__(self, model=None, doc_matrix=None, doc_ids=None):
        self.model = model            # gensim Word2Vec
        self.doc_matrix = doc_matrix  # (N, D) float32, rows L2-normalized
        self.doc_ids = doc_ids or []
        self.dim = (model.vector_size if model is not None
                    else (doc_matrix.shape[1] if doc_matrix is not None else 0))

    # ---------- training ----------
    @classmethod
    def train(cls, token_lists, vector_size=100, window=5, min_count=2,
              epochs=5, workers=4, sg=0):
        from gensim.models import Word2Vec
        model = Word2Vec(
            sentences=token_lists, vector_size=vector_size, window=window,
            min_count=min_count, epochs=epochs, workers=workers, sg=sg,
        )
        return cls(model=model)

    # ---------- vectorizing ----------
    def embed_tokens(self, tokens):
        wv = self.model.wv
        vecs = [wv[t] for t in tokens if t in wv]
        if not vecs:
            return np.zeros(self.dim, dtype=np.float32)
        v = np.mean(vecs, axis=0)
        n = np.linalg.norm(v)
        return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)

    def transform(self, tokens):
        return self.embed_tokens(tokens)

    def build_doc_matrix(self, doc_ids, token_lists):
        mat = np.zeros((len(doc_ids), self.dim), dtype=np.float32)
        for i, toks in enumerate(token_lists):
            mat[i] = self.embed_tokens(toks)
        self.doc_matrix = mat
        self.doc_ids = list(doc_ids)
        return self

    # ---------- persistence ----------
    def save(self, dirpath):
        os.makedirs(dirpath, exist_ok=True)
        self.model.save(os.path.join(dirpath, "w2v.model"))
        np.save(os.path.join(dirpath, "doc_matrix.npy"), self.doc_matrix)
        joblib.dump(self.doc_ids, os.path.join(dirpath, "doc_ids.joblib"))

    @classmethod
    def load(cls, dirpath):
        from gensim.models import Word2Vec
        model = Word2Vec.load(os.path.join(dirpath, "w2v.model"))
        doc_matrix = np.load(os.path.join(dirpath, "doc_matrix.npy"))
        doc_ids = joblib.load(os.path.join(dirpath, "doc_ids.joblib"))
        return cls(model=model, doc_matrix=doc_matrix, doc_ids=doc_ids)
