# ============================================================
# services/representation_tfidf.py  -  TF-IDF / VSM  (Step 2, requirement 2)
#
# Represents every document as an L2-normalized TF-IDF vector.
# Because the corpus is ALREADY preprocessed (stemmed, stopwords
# removed, space-joined), we tell the vectorizer to just split on
# whitespace -> its vocabulary matches the inverted index exactly.
# ============================================================
import os
import joblib
from scipy.sparse import save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfRepresentation:
    def __init__(self):
        self.vectorizer = None
        self.matrix = None      # sparse (N_docs x V), L2-normalized rows
        self.doc_ids = []

    def build(self, doc_ids, lexical_texts):
        # input is already tokenized text -> just split on spaces
        self.vectorizer = TfidfVectorizer(
            tokenizer=str.split,
            preprocessor=None,
            token_pattern=None,
            lowercase=False,
            norm="l2",          # so dot product == cosine similarity
        )
        self.matrix = self.vectorizer.fit_transform(lexical_texts)
        self.doc_ids = list(doc_ids)
        return self

    def transform(self, lexical_text):
        """Vectorize a query (already preprocessed + space-joined)."""
        return self.vectorizer.transform([lexical_text])

    @property
    def shape(self):
        return None if self.matrix is None else self.matrix.shape

    # --- persistence ---
    def save(self, dirpath):
        os.makedirs(dirpath, exist_ok=True)
        joblib.dump(self.vectorizer, os.path.join(dirpath, "vectorizer.joblib"))
        save_npz(os.path.join(dirpath, "matrix.npz"), self.matrix)
        joblib.dump(self.doc_ids, os.path.join(dirpath, "doc_ids.joblib"))

    @classmethod
    def load(cls, dirpath):
        obj = cls()
        obj.vectorizer = joblib.load(os.path.join(dirpath, "vectorizer.joblib"))
        obj.matrix = load_npz(os.path.join(dirpath, "matrix.npz"))
        obj.doc_ids = joblib.load(os.path.join(dirpath, "doc_ids.joblib"))
        return obj
