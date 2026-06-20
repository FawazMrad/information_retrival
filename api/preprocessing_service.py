# ============================================================
# api/preprocessing_service.py  -  PREPROCESSING SERVICE
# Run independently:
#   uvicorn api.preprocessing_service:app --port 8001
# ============================================================
from fastapi import FastAPI
from pydantic import BaseModel

from api.common import get_preprocessor

app = FastAPI(title="Preprocessing Service")


class PrepRequest(BaseModel):
    text: str
    mode: str = "lexical"   # "lexical" or "semantic"


@app.get("/health")
def health():
    return {"service": "preprocessing", "status": "ok"}


@app.post("/preprocess")
def preprocess(req: PrepRequest):
    prep = get_preprocessor()
    if req.mode == "semantic":
        return {"mode": "semantic", "result": prep.clean_for_embeddings(req.text)}
    return {"mode": "lexical", "tokens": prep.process(req.text, mode="lexical")}
