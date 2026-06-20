# ============================================================
# api/evaluation_service.py  -  EVALUATION SERVICE
# Demonstrates LOOSE COUPLING: it does not import the retrieval
# code, it calls the Retrieval Service over REST.
# Run independently (Retrieval Service must be up):
#   uvicorn api.evaluation_service:app --port 8005
# ============================================================
import random
import functools
import requests
from fastapi import FastAPI
from pydantic import BaseModel

import config
from api.common import service_url
from services import data_loader as dl
from services.evaluation import evaluate

app = FastAPI(title="Evaluation Service", docs_url=None)

from api.common import enable_offline_docs
enable_offline_docs(app)


@functools.lru_cache(maxsize=1)
def get_qrels_queries():
    ds = dl.load_dataset(config.DATASET_ID)
    qrels = dl.load_qrels(ds)
    queries = {qid: text for qid, text in dl.iter_queries(ds)}
    return qrels, queries


class EvalRequest(BaseModel):
    model: str = "BM25"
    sample_queries: int = 100
    cutoff: int = 100


@app.get("/health")
def health():
    return {"service": "evaluation", "status": "ok"}


@app.post("/evaluate")
def do_evaluate(req: EvalRequest):
    qrels, queries = get_qrels_queries()
    judged = [q for q in queries if q in qrels and qrels[q]]
    random.seed(config.EVAL["seed"])
    random.shuffle(judged)
    sample = judged[:req.sample_queries]

    retrieval = service_url("retrieval")
    run = {}
    for qid in sample:
        resp = requests.post(retrieval + "/search",
                             json={"query": queries[qid], "model": req.model,
                                   "top_k": req.cutoff}, timeout=60)
        run[qid] = [x["doc_id"] for x in resp.json()["results"]]

    metrics = evaluate(run, qrels, p_k=config.EVAL["p_k"],
                       ndcg_k=config.EVAL["ndcg_k"], recall_k=config.EVAL["recall_k"])
    return {"model": req.model, "queries": len(sample), "metrics": metrics}