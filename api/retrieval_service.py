# ============================================================
# api/retrieval_service.py  -  RETRIEVAL & RANKING SERVICE
# Run independently:
#   uvicorn api.retrieval_service:app --port 8004
# ============================================================
import time
import functools
from fastapi import FastAPI
from pydantic import BaseModel

import config
from api.common import (get_preprocessor, get_index, get_bm25, get_tfidf,
                        get_embedding, get_clustering, get_id_to_text)
from services.search import TfidfSearcher, BM25Searcher, EmbeddingSearcher
from services.hybrid import HybridSearcher

app = FastAPI(title="Retrieval & Ranking Service", docs_url=None)

from api.common import enable_offline_docs
enable_offline_docs(app)


@functools.lru_cache(maxsize=1)
def get_searchers():
    prep = get_preprocessor()
    bm25 = get_bm25()
    id2t = get_id_to_text()
    emb = get_embedding()
    return {
        "prep": prep, "bm25": bm25, "emb": emb,
        "tfidf_s": TfidfSearcher(get_tfidf(), prep, id2t),
        "bm25_s": BM25Searcher(bm25, prep, id2t),
        "emb_s": EmbeddingSearcher(emb, prep, id2t),
        "hybrid": HybridSearcher(bm25, emb, prep, id2t),
    }


class SearchRequest(BaseModel):
    query: str
    model: str = "BM25"      # TF-IDF | BM25 | Word2Vec | Hybrid-Serial | Hybrid-Parallel
    top_k: int = 10
    k1: float = config.BM25_K1
    b: float = config.BM25_B
    fusion: str = "rrf"
    alpha: float = 0.7
    cluster: bool = False
    probe: int = 3


@app.get("/health")
def health():
    return {"service": "retrieval", "status": "ok"}


@app.post("/search")
def search(req: SearchRequest):
    S = get_searchers()
    S["bm25"].set_params(k1=req.k1, b=req.b)
    t0 = time.time()

    k = 100 if req.cluster else req.top_k
    if req.model == "TF-IDF":
        res = S["tfidf_s"].search(req.query, top_k=k)
    elif req.model == "Word2Vec":
        res = S["emb_s"].search(req.query, top_k=k)
    elif req.model == "Hybrid-Serial":
        res = S["hybrid"].serial(req.query, top_k=k)
    elif req.model == "Hybrid-Parallel":
        res = S["hybrid"].parallel(req.query, top_k=k, fusion=req.fusion, alpha=req.alpha)
    else:  # BM25 (default)
        res = S["bm25_s"].search(req.query, top_k=k)

    if req.cluster:
        cl = get_clustering()
        if cl is not None:
            qv = S["emb"].transform(S["prep"].process(req.query, mode="lexical"))
            near = set(cl.nearest_clusters(qv, c=req.probe))
            res = [x for x in res if cl.label_of(x["doc_id"]) in near][:req.top_k]
        else:
            res = res[:req.top_k]
    else:
        res = res[:req.top_k]

    return {"query": req.query, "model": req.model,
            "ms": round((time.time() - t0) * 1000, 1), "results": res}