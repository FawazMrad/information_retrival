# ============================================================
# api/indexing_service.py  -  INDEXING SERVICE
# Run independently:
#   uvicorn api.indexing_service:app --port 8002
# ============================================================
from fastapi import FastAPI

from api.common import get_index

app = FastAPI(title="Indexing Service")


@app.get("/health")
def health():
    return {"service": "indexing", "status": "ok"}


@app.get("/stats")
def stats():
    inv = get_index()
    return {
        "num_docs": inv.N,
        "vocab_size": inv.vocab_size(),
        "num_postings": inv.num_postings(),
        "avg_doc_len": round(inv.avg_doc_len, 3),
    }


@app.get("/postings/{term}")
def postings(term: str, limit: int = 10):
    inv = get_index()
    return {
        "term": term,
        "df": inv.df.get(term, 0),
        "postings_sample": inv.postings(term)[:limit],
    }
