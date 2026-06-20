# ============================================================
# app.py  -  USER INTERFACE  (requirement 9)
#
# A simple Streamlit search UI that exposes the whole system:
#   - dataset selector
#   - representation model selector (TF-IDF / BM25 / Word2Vec / Hybrid)
#   - BM25 parameter sliders (k1, b)  -> probabilistic model controls
#   - hybrid mode + fusion + alpha selectors
#   - "Basic only" vs "Basic + Enhancements" execution
#   - enhancements: query refinement + clustering (the additional feature)
#
# Run from the project root:
#   streamlit run app.py
# ============================================================
import json
import time
import streamlit as st

import config
from services.preprocessing import Preprocessor
from services.indexing import InvertedIndex
from services.representation_tfidf import TfidfRepresentation
from services.representation_bm25 import BM25
from services.representation_embedding import Word2VecEmbedding
from services.clustering import DocumentClustering
from services.query_refinement import QueryRefinement
from services.search import (TfidfSearcher, BM25Searcher,
                             EmbeddingSearcher)
from services.hybrid import HybridSearcher

import os

st.set_page_config(page_title="IR Search Engine", layout="wide")


@st.cache_resource(show_spinner="Loading models (first time only) ...")
def load_system():
    prep = Preprocessor(**config.PREP)
    inv = InvertedIndex.load(config.INVERTED_INDEX_PATH)
    bm25 = BM25(inv, k1=config.BM25_K1, b=config.BM25_B)
    tfidf = TfidfRepresentation.load(config.TFIDF_DIR)
    emb = Word2VecEmbedding.load(config.EMBEDDING_DIR)
    clustering = (DocumentClustering.load(config.CLUSTER_DIR)
                  if os.path.isdir(config.CLUSTER_DIR) else None)

    id_to_text = {}
    with open(config.PROCESSED_DOCS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            id_to_text[rec["doc_id"]] = rec.get("text", "")

    qr = QueryRefinement(prep, vocab=inv.index.keys(), w2v=emb, df=inv.df)
    return {
        "prep": prep, "inv": inv, "bm25": bm25, "emb": emb,
        "clustering": clustering, "id_to_text": id_to_text, "qr": qr,
        "tfidf_s": TfidfSearcher(tfidf, prep, id_to_text),
        "bm25_s": BM25Searcher(bm25, prep, id_to_text),
        "emb_s": EmbeddingSearcher(emb, prep, id_to_text),
        "hybrid": HybridSearcher(bm25, emb, prep, id_to_text),
    }


S = load_system()

# ------------------------------------------------------------ sidebar
st.sidebar.header("Search settings")
dataset = st.sidebar.selectbox("Dataset", [config.DATASET_NAME])

model = st.sidebar.selectbox(
    "Representation model",
    ["TF-IDF", "BM25", "Word2Vec", "Hybrid (serial)", "Hybrid (parallel)"])

top_k = st.sidebar.slider("Number of results", 5, 20, 10)

# BM25 params (probabilistic model controls)
k1, b = config.BM25_K1, config.BM25_B
if model in ("BM25", "Hybrid (serial)", "Hybrid (parallel)"):
    st.sidebar.markdown("**BM25 parameters**")
    k1 = st.sidebar.slider("k1 (term saturation)", 0.0, 3.0, config.BM25_K1, 0.1)
    b = st.sidebar.slider("b (length norm.)", 0.0, 1.0, config.BM25_B, 0.05)

fusion, alpha = "rrf", 0.7
if model == "Hybrid (parallel)":
    st.sidebar.markdown("**Fusion**")
    fusion = st.sidebar.selectbox("Fusion method", ["rrf", "weighted"])
    if fusion == "weighted":
        alpha = st.sidebar.slider("alpha (BM25 weight)", 0.0, 1.0, 0.7, 0.05)

# execution mode
st.sidebar.markdown("---")
mode = st.sidebar.radio("Execution", ["Basic only", "Basic + Enhancements"])
use_refine = use_cluster = False
probe = config.CLUSTER["probe_clusters"]
if mode == "Basic + Enhancements":
    use_refine = st.sidebar.checkbox("Query refinement (spell + expand)", True)
    if S["clustering"] is not None:
        use_cluster = st.sidebar.checkbox("Clustering (search nearest clusters)", True)
        probe = st.sidebar.slider("Clusters to probe", 1, 8, probe)

# ------------------------------------------------------------ main
st.title("Information Retrieval Search Engine")
st.caption(f"Dataset: {dataset}  ·  {S['inv'].N:,} documents  ·  model: {model}")

query = st.text_input("Enter your query", placeholder="e.g. how to invest in the indian stock market")


def run_model(q, k):
    S["bm25"].set_params(k1=k1, b=b)
    if model == "TF-IDF":
        return S["tfidf_s"].search(q, top_k=k)
    if model == "BM25":
        return S["bm25_s"].search(q, top_k=k)
    if model == "Word2Vec":
        return S["emb_s"].search(q, top_k=k)
    if model == "Hybrid (serial)":
        return S["hybrid"].serial(q, top_k=k)
    return S["hybrid"].parallel(q, top_k=k, fusion=fusion, alpha=alpha)


if st.button("Search", type="primary") and query.strip():
    t0 = time.time()
    search_q = query

    # enhancement 1: query refinement
    if use_refine:
        refined, info = S["qr"].refine(query)
        search_q = " ".join(refined)
        with st.expander("Query refinement", expanded=True):
            st.write("**Suggestion:**", S["qr"].suggest(query))
            if info["corrections"]:
                st.write("**Corrections:**", info["corrections"])
            if info["expansions"]:
                st.write("**Expansions:**", info["expansions"])

    # retrieve (grab a larger pool if we will cluster-filter)
    pool_k = 100 if use_cluster else top_k
    results = run_model(search_q, pool_k)

    # enhancement 2: clustering (restrict to the query's nearest clusters)
    if use_cluster and S["clustering"] is not None:
        qvec = S["emb"].transform(S["prep"].process(search_q, mode="lexical"))
        near = set(S["clustering"].nearest_clusters(qvec, c=probe))
        results = [r for r in results
                   if S["clustering"].label_of(r["doc_id"]) in near][:top_k]
        st.caption(f"Clustering on · searching clusters {sorted(near)}")
    else:
        results = results[:top_k]

    dt = (time.time() - t0) * 1000
    st.write(f"**{len(results)} results** in {dt:.0f} ms")
    st.caption("Each result shows the document's original ID and its original (un-cleaned) text from the dataset.")
    if not results:
        st.warning("No matching documents.")
    for rank, r in enumerate(results, 1):
        cl = f" · cluster {S['clustering'].label_of(r['doc_id'])}" if S["clustering"] else ""
        st.markdown(f"**{rank}.  Document ID: `{r['doc_id']}`**  ·  score {r['score']:.3f}{cl}")
        st.markdown(r["text"] if r.get("text") else "_(original text unavailable)_")
        st.divider()
