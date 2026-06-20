# ============================================================
# api/gateway.py  -  API GATEWAY
# The single entry point. Orchestrates the pipeline:
#   (optional) Refinement Service -> Retrieval Service.
# Run:
#   uvicorn api.gateway:app --port 8000
# ============================================================
import requests
from fastapi import FastAPI
from pydantic import BaseModel

import config
from api.common import service_url

app = FastAPI(title="API Gateway", docs_url=None)

from api.common import enable_offline_docs
enable_offline_docs(app)


class GatewaySearch(BaseModel):
    query: str
    model: str = "BM25"
    top_k: int = 10
    k1: float = config.BM25_K1
    b: float = config.BM25_B
    fusion: str = "rrf"
    alpha: float = 0.7
    refine: bool = False      # enhancement: query refinement
    cluster: bool = False     # enhancement: clustering
    probe: int = 3


@app.get("/health")
def health():
    out = {}
    for name in config.SERVICES:
        if name == "gateway":
            continue
        try:
            r = requests.get(service_url(name) + "/health", timeout=2)
            out[name] = r.json().get("status", "?")
        except Exception:
            out[name] = "down"
    return {"gateway": "ok", "services": out}


@app.post("/search")
def search(req: GatewaySearch):
    query = req.query
    suggestion = None

    # enhancement: query refinement (delegated to the Refinement Service)
    if req.refine:
        try:
            rr = requests.post(service_url("refinement") + "/refine",
                               json={"query": query}, timeout=30).json()
            query = " ".join(rr["refined"])
            suggestion = rr["info"]
        except Exception as e:
            suggestion = {"error": str(e)}

    # delegate retrieval + ranking to the Retrieval Service
    payload = {
        "query": query, "model": req.model, "top_k": req.top_k,
        "k1": req.k1, "b": req.b, "fusion": req.fusion, "alpha": req.alpha,
        "cluster": req.cluster, "probe": req.probe,
    }
    resp = requests.post(service_url("retrieval") + "/search", json=payload, timeout=60).json()
    resp["refinement"] = suggestion
    resp["original_query"] = req.query
    return resp