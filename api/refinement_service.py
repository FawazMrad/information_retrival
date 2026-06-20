# ============================================================
# api/refinement_service.py  -  QUERY REFINEMENT SERVICE
# Run independently:
#   uvicorn api.refinement_service:app --port 8003
# ============================================================
import functools
from fastapi import FastAPI
from pydantic import BaseModel

from api.common import get_preprocessor, get_index, get_embedding
from services.query_refinement import QueryRefinement

app = FastAPI(title="Query Refinement Service", docs_url=None)

from api.common import enable_offline_docs
enable_offline_docs(app)


@functools.lru_cache(maxsize=1)
def get_qr():
    inv = get_index()
    return QueryRefinement(get_preprocessor(), vocab=inv.index.keys(),
                           w2v=get_embedding(), df=inv.df)


class RefineRequest(BaseModel):
    query: str
    use_spell: bool = True
    use_expand: bool = True
    expand_n: int = 2


@app.get("/health")
def health():
    return {"service": "refinement", "status": "ok"}


@app.post("/refine")
def refine(req: RefineRequest):
    refined, info = get_qr().refine(
        req.query, use_spell=req.use_spell,
        use_expand=req.use_expand, expand_n=req.expand_n)
    return {"refined": refined, "info": info}


@app.post("/suggest")
def suggest(req: RefineRequest):
    return {"suggestion": get_qr().suggest(req.query)}