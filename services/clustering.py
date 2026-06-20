# ============================================================
# services/clustering.py  -  DOCUMENT CLUSTERING  (additional feature)
#
# Groups documents with MiniBatchKMeans on the Word2Vec document
# vectors (dense + fast, scales to the whole corpus). Also supports
# cluster-based retrieval (part 2): find the nearest clusters to a
# query and search only within them.
# ============================================================
import os
import numpy as np
import joblib
from sklearn.cluster import MiniBatchKMeans


class DocumentClustering:
    def __init__(self, model=None, labels=None, doc_ids=None):
        self.model = model                       # fitted MiniBatchKMeans
        self.labels = labels                     # np.array (N,) cluster id per doc
        self.doc_ids = list(doc_ids) if doc_ids is not None else []
        self.id2idx = {d: i for i, d in enumerate(self.doc_ids)}

    @classmethod
    def fit(cls, matrix, doc_ids, n_clusters=20, seed=42, batch_size=4096):
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed,
                             batch_size=batch_size, n_init=3)
        labels = km.fit_predict(matrix)
        return cls(model=km, labels=labels, doc_ids=doc_ids)

    @property
    def n_clusters(self):
        return self.model.n_clusters

    def cluster_sizes(self):
        vals, counts = np.unique(self.labels, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, counts)}

    def label_of(self, doc_id):
        i = self.id2idx.get(doc_id)
        return int(self.labels[i]) if i is not None else None

    # ---------- cluster-based retrieval (part 2) ----------
    def nearest_clusters(self, vec, c=1):
        """Cluster ids whose centroids are closest to a (query) vector."""
        dists = self.model.transform(vec.reshape(1, -1))[0]
        return list(np.argsort(dists)[:c])

    def docs_in_clusters(self, cluster_ids):
        """Indices of all documents in the given clusters."""
        mask = np.isin(self.labels, list(cluster_ids))
        return np.nonzero(mask)[0]

    # ---------- persistence ----------
    def save(self, dirpath):
        os.makedirs(dirpath, exist_ok=True)
        joblib.dump(self.model, os.path.join(dirpath, "kmeans.joblib"))
        np.save(os.path.join(dirpath, "labels.npy"), self.labels)
        joblib.dump(self.doc_ids, os.path.join(dirpath, "doc_ids.joblib"))

    @classmethod
    def load(cls, dirpath):
        model = joblib.load(os.path.join(dirpath, "kmeans.joblib"))
        labels = np.load(os.path.join(dirpath, "labels.npy"))
        doc_ids = joblib.load(os.path.join(dirpath, "doc_ids.joblib"))
        return cls(model=model, labels=labels, doc_ids=doc_ids)
