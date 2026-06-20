# ============================================================
# api/common.py  -  shared, lazily-cached model loaders
# Each service imports ONLY the loaders it needs, so a service
# never loads artifacts it doesn't use.
# ============================================================
import os
import json
import functools

from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html

import config
from services.preprocessing import Preprocessor
from services.indexing import InvertedIndex
from services.representation_tfidf import TfidfRepresentation
from services.representation_bm25 import BM25
from services.representation_embedding import Word2VecEmbedding
from services.clustering import DocumentClustering


_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static", "swagger-ui")


def enable_offline_docs(app):
    """Serve the interactive /docs page from local files instead of a CDN.

    By default FastAPI's Swagger UI loads its JS/CSS from
    https://cdn.jsdelivr.net, which fails with no internet (requirement 4).
    This mounts the bundled swagger-ui assets locally and re-points /docs
    at them, so the docs page also works fully offline.
    Call this right after creating the FastAPI() app, with docs_url=None.
    """
    app.mount("/static-docs", StaticFiles(directory=_STATIC_DIR), name="static-docs")

    @app.get("/docs", include_in_schema=False)
    def custom_swagger_ui_html():
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=f"{app.title} - Swagger UI",
            swagger_js_url="/static-docs/swagger-ui-bundle.js",
            swagger_css_url="/static-docs/swagger-ui.css",
            swagger_favicon_url="/static-docs/favicon-32x32.png",
        )


@functools.lru_cache(maxsize=1)
def get_preprocessor():
    return Preprocessor(**config.PREP)


@functools.lru_cache(maxsize=1)
def get_index():
    return InvertedIndex.load(config.INVERTED_INDEX_PATH)


@functools.lru_cache(maxsize=1)
def get_bm25():
    return BM25(get_index(), k1=config.BM25_K1, b=config.BM25_B)


@functools.lru_cache(maxsize=1)
def get_tfidf():
    return TfidfRepresentation.load(config.TFIDF_DIR)


@functools.lru_cache(maxsize=1)
def get_embedding():
    return Word2VecEmbedding.load(config.EMBEDDING_DIR)


@functools.lru_cache(maxsize=1)
def get_clustering():
    if os.path.isdir(config.CLUSTER_DIR):
        return DocumentClustering.load(config.CLUSTER_DIR)
    return None


@functools.lru_cache(maxsize=1)
def get_id_to_text():
    d = {}
    with open(config.PROCESSED_DOCS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            d[rec["doc_id"]] = rec.get("text", "")
    return d


def service_url(name):
    return f"http://{config.SERVICE_HOST}:{config.SERVICES[name]}"