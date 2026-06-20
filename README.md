# Information Retrieval System — IR Project 2026

A search engine over the **BEIR/Quora** dataset (522,931 documents) built with
classical and neural IR methods, a Service-Oriented Architecture, a Streamlit
UI, and full evaluation. Language: **Python**.

## Pipeline / requirements covered
1. **Preprocessing** — normalize, stopword removal, Snowball stemming (`services/preprocessing.py`)
2. **Representations** — TF-IDF (VSM), BM25, Word2Vec, Hybrid (serial + parallel with RRF / weighted fusion)
3. **Indexing** — inverted index (`services/indexing.py`)
4. **Query processing** — same preprocessing applied to queries
5. **Query refinement** — spell correction + Word2Vec synonym expansion + session history
6. **Matching & ranking** — cosine similarity / BM25 / fusion
7. **SOA** — independent FastAPI services behind a gateway (`api/`)
8. **Evaluation** — MAP, nDCG@10, P@10, Recall@100 with charts
9. **UI** — Streamlit app (`app.py`)
- **Additional feature** — Document Clustering (KMeans) with cluster-based search

## Setup
```bash
python -m venv venv
venv\Scripts\activate          # Windows  (source venv/bin/activate on mac/linux)
pip install -r requirements.txt
```

## Build the artifacts (run once, in order)
```bash
python -m scripts.run_step1_preprocessing --full --skip-samples   # processed corpus
python -m scripts.run_step2_build                                 # inverted index + TF-IDF
python -m scripts.run_step4_build_embeddings                      # Word2Vec vectors
python -m scripts.run_step8_build_clusters                        # clusters (feature)
```

## Run
```bash
streamlit run app.py            # the UI
python -m scripts.run_step6_evaluate          # evaluation table + charts
python -m scripts.run_step8_eval_clustering   # before/after clustering
```

## SOA (services)
```bash
python run_services.py          # starts all services
python test_services.py         # smoke test
# gateway docs: http://127.0.0.1:8000/docs
```
Each service is independent, e.g. `uvicorn api.retrieval_service:app --port 8004`.

## Layout
```
config.py            central configuration (dataset, paths, parameters)
services/            core logic (one responsibility per file)
api/                 FastAPI services + gateway (SOA)
scripts/             build + search + evaluation entry points
app.py               Streamlit UI
data/                generated artifacts (not committed)
```

