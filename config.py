# ============================================================
# config.py  -  central configuration for the whole project
# Change the dataset here ONCE and every service follows.
# ============================================================
import os

# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------
# We use one ir_datasets id that exposes docs + queries + qrels together.
# beir/quora/test  ->  522,931 docs | 10,000 test queries | qrels available
# To switch datasets later, just change this line, e.g.:
#   "beir/trec-covid"      (CORD-19, graded qrels, ~171K docs)
#   "wikir/en1k/test"      (~369K docs)
DATASET_ID = "beir/quora/test"

# A short human-friendly name used for output folders/files
DATASET_NAME = "quora"

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DOCS_PATH = os.path.join(DATA_DIR, f"{DATASET_NAME}_processed_docs.jsonl")
os.makedirs(DATA_DIR, exist_ok=True)

# ------------------------------------------------------------
# PREPROCESSING OPTIONS
# ------------------------------------------------------------
PREP = {
    "language": "english",
    "remove_stopwords": True,
    "use_lemmatization": False,  # False -> Snowball stemming (recommended for IR)
    "min_token_len": 2,
    "tokenizer": "split",        # "split" (fast) or "nltk"
}

# ------------------------------------------------------------
# INDEX / REPRESENTATION ARTIFACTS  (Step 2+)
# ------------------------------------------------------------
INVERTED_INDEX_PATH = os.path.join(DATA_DIR, f"{DATASET_NAME}_inverted_index.pkl")
TFIDF_DIR = os.path.join(DATA_DIR, f"{DATASET_NAME}_tfidf")
TOP_K = 10

# ------------------------------------------------------------
# BM25 parameters (adjustable at runtime / via the UI) - req 2, note 2
#   k1 -> term-frequency saturation (typical 1.2 - 2.0)
#   b  -> document-length normalization (0 = none, 1 = full; typical 0.75)
# ------------------------------------------------------------
BM25_K1 = 1.5
BM25_B = 0.75

# ------------------------------------------------------------
# EMBEDDINGS - Word2Vec (Step 4)
# vector_size 100 keeps the doc matrix small (~210 MB for this corpus).
# Raise it (e.g. 200/300) for a bit more quality if your RAM allows.
# ------------------------------------------------------------
EMBEDDING_DIR = os.path.join(DATA_DIR, f"{DATASET_NAME}_w2v")
W2V = {
    "vector_size": 100,
    "window": 5,
    "min_count": 2,
    "epochs": 5,
    "workers": 4,
    "sg": 0,          # 0 = CBOW (faster), 1 = skip-gram (slower, better on rare words)
}

# ------------------------------------------------------------
# CLUSTERING - additional feature (Document Clustering)
# Clusters are fit on the Word2Vec document vectors (dense, fast).
# ------------------------------------------------------------
CLUSTER_DIR = os.path.join(DATA_DIR, f"{DATASET_NAME}_clusters")
CLUSTER = {
    "n_clusters": 20,
    "sample_silhouette": 5000,   # silhouette is O(n^2) -> compute on a sample
    "sample_scatter": 4000,      # points to draw in the 2D cluster map
    "batch_size": 4096,
    "probe_clusters": 3,         # nearest clusters to search within (part 2)
    "seed": 42,
}

# ------------------------------------------------------------
# EVALUATION (Step 6, requirement 8)
# Evaluating every test query against every model is slow, so we
# evaluate on a random SAMPLE (raise sample_queries for the final run).
# ------------------------------------------------------------
EVAL_DIR = os.path.join(DATA_DIR, f"{DATASET_NAME}_eval")
EVAL = {
    "sample_queries": 300,   # number of test queries to evaluate on
    "cutoff": 100,           # retrieval depth per query
    "p_k": 10,               # Precision@k
    "ndcg_k": 10,            # nDCG@k
    "recall_k": 100,         # Recall@k
    "alpha": 0.7,            # weighted-fusion weight on BM25
    "seed": 42,
}

# ------------------------------------------------------------
# SOA - service registry (Step 9 / requirement 7)
# Each service runs as an independent FastAPI app on its own port.
# ------------------------------------------------------------
SERVICE_HOST = "127.0.0.1"
SERVICES = {
    "gateway": 8000,
    "preprocessing": 8001,
    "indexing": 8002,
    "refinement": 8003,
    "retrieval": 8004,
    "evaluation": 8005,
}

# ------------------------------------------------------------
# SOA - service URLs (requirement 7)
# Each service runs as its own process on its own port; the gateway
# and evaluation service talk to the others over REST.
# ------------------------------------------------------------
SOA = {
    "preprocessing": "http://127.0.0.1:8001",
    "search":        "http://127.0.0.1:8002",
    "refinement":    "http://127.0.0.1:8003",
    "evaluation":    "http://127.0.0.1:8004",
    "gateway":       "http://127.0.0.1:8000",
}