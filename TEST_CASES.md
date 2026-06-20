# Test Cases — IR Search Engine (BEIR/Quora, 522,931 docs)

Each case lists **what it checks**, the **steps/input**, and the **expected result**.
Expected numbers come from the built system; small variations are normal.

Legend: ✅ pass criteria.

---

## 1. Environment & data

| ID | Checks | Steps | Expected |
|----|--------|-------|----------|
| ENV-1 | Dependencies install | `pip install -r requirements.txt` | completes with no errors |
| ENV-2 | Dataset loads + has qrels | `python -m scripts.run_step1_preprocessing` | DATASET STATS shows **documents 522,931 · queries 10,000 · qrels 15,675** ✅ |
| ENV-3 | Full preprocessing | `python -m scripts.run_step1_preprocessing --full --skip-samples` | prints `Done. Preprocessed 522,931 documents`; creates `data/quora_processed_docs.jsonl` |

## 2. Preprocessing (req 1)

| ID | Checks | Input | Expected |
|----|--------|-------|----------|
| PRE-1 | Normalization + stemming | "What are the best programming languages to learn in 2026?" | `['best','program','languag','learn','2026']` |
| PRE-2 | Stopword removal | any sentence | "what/the/to/in/is" removed |
| PRE-3 | URL / punctuation stripped | "See http://run.com !!!" | url + `!!!` removed |
| PRE-4 | Semantic mode keeps word forms | mode="semantic" | lower-cased text, stopwords & forms intact |
| PRE-5 | Query == doc preprocessing | same text as query and doc | identical token output (req 4) |

## 3. Indexing + TF-IDF (req 2, 3)

| ID | Checks | Steps | Expected |
|----|--------|-------|----------|
| IDX-1 | Build index + TF-IDF | `python -m scripts.run_step2_build` | inverted index ~**64,000 terms**, matrix shape `(522931, ~64000)` |
| IDX-2 | Postings correctness | inspect term "invest" | df > 0, postings list of `(doc_idx, tf)` |
| TFIDF-1 | Relevant ranking | search `how to invest in the indian stock market` | top results about investing in Indian stock market; cosine 0.2–0.95 |
| TFIDF-2 | Semantic-free matching | search `learn python programming` | python-learning questions on top |

## 4. BM25 + parameter control (req 2, note 2)

| ID | Checks | Steps | Expected |
|----|--------|-------|----------|
| BM25-1 | Ranking quality | `run_step3_bm25_search` → `how to invest in indian stock market` | top hits are Indian-stock-market questions |
| BM25-2 | k1/b live change | `:k1 2.0` then `:b 0.4`, re-search | ranking visibly shifts; longer/more-specific docs rise as `b` drops |
| BM25-3 | Speed | any query | results in < ~150 ms |

## 5. Word2Vec (req 2)

| ID | Checks | Steps | Expected |
|----|--------|-------|----------|
| W2V-1 | Build vectors | `run_step4_build_embeddings` | matrix `(522931, 100)`, ~210 MB |
| W2V-2 | Semantic match | `run_step4_embedding_search` → `coding lessons for beginners` | bootcamp / learn-to-code results (no shared words needed) |
| W2V-3 | Known weakness | `putting savings into shares` | noisier results — expected (averaging short text) |

## 6. Hybrid (req 2, notes 1 & 3)

| ID | Checks | Steps | Expected |
|----|--------|-------|----------|
| HYB-1 | Serial mode | `:mode serial` + query | runs; BM25 candidates re-ranked by embedding |
| HYB-2 | Parallel + RRF | `:mode parallel` `:fusion rrf` | fused ranking; small RRF scores (~0.01–0.03) |
| HYB-3 | Parallel + weighted | `:fusion weighted` `:alpha 0.7` | normalized scores [0,1]; favors BM25 |
| HYB-4 | Mode switch in UI | dropdown serial/parallel | both selectable and produce results |

## 7. Query refinement (req 5)

| ID | Checks | Input | Expected |
|----|--------|-------|----------|
| QR-1 | Spell correction | `how to invset in indian stock markett` | `invset→invest`, `markett→market` |
| QR-2 | Frequency-aware correction | `invset` | maps to common `invest`, not rare look-alikes |
| QR-3 | Synonym expansion | `stock market` | adds `bse, nse, trade, broker` (relevant) |
| QR-4 | Noise filtered | any query | weak expansions (low similarity) dropped |
| QR-5 | Toggles | `:spell off` / `:expand off` | each technique can be disabled independently |
| QR-6 | Suggestion string | typo query | "did you mean: … | related terms: …" |

## 8. Clustering — additional feature

| ID | Checks | Steps | Expected |
|----|--------|-------|----------|
| CL-1 | Build clusters | `run_step8_build_clusters` | 20 clusters, sizes ~16k–46k, **silhouette ≈ 0.06** (low but positive — normal) |
| CL-2 | Interpretable topics | inspect top terms | clear themes: relationships, programming, money, politics, health, etc. |
| CL-3 | Charts produced | check `data/quora_clusters/` | `clusters_overview.png` (sizes + 2D map), `top_terms.json` |
| CL-4 | With/without search | `run_step8_cluster_search` → a query | both result sets shown; clustering is faster; speedup printed |
| CL-5 | Before/after eval | `run_step8_eval_clustering` | MAP ≈ unchanged (~0.48), Recall slightly lower (~0.77→0.75), avg time lower (~69→50 ms) |

## 9. Evaluation (req 8)

| ID | Checks | Steps | Expected |
|----|--------|-------|----------|
| EV-1 | Metric correctness | unit example (rel={A,C}, ranked=[A,B,C,D]) | AP=0.833, nDCG@4=0.920, P@10=0.2, Recall@4=1.0 |
| EV-2 | Full comparison | `run_step6_evaluate` | table for 6 methods; **BM25 MAP ≈ 0.73**, Word2Vec ≈ 0.48 |
| EV-3 | Chart | check `data/quora_eval/` | `metrics_comparison.png` (MAP, nDCG, P@10, Recall) |
| EV-4 | P@10 sanity | any | ~0.11–0.12 (Quora has ~1–2 relevant/query — expected ceiling) |

## 10. UI (req 9)

| ID | Checks | Steps | Expected |
|----|--------|-------|----------|
| UI-1 | Launch | `streamlit run app.py` | opens at `localhost:8501`; loads models once |
| UI-2 | Dataset selector | sidebar | shows the dataset before querying |
| UI-3 | Model selector | pick TF-IDF/BM25/Word2Vec/Hybrid | results update accordingly |
| UI-4 | BM25 sliders | move k1 / b, search | ranking changes live |
| UI-5 | Hybrid controls | parallel + fusion + alpha | selectable, affects results |
| UI-6 | Basic vs Enhanced | radio toggle | "Basic only" hides refinement/clustering; "Enhanced" shows them |
| UI-7 | Enhancements | enable refinement + clustering | suggestion shown; results restricted to nearest clusters |
| UI-8 | Results display | search | ranked list with score + text (+ cluster id) + query time |

## 11. SOA services (req 7)

| ID | Checks | Steps | Expected |
|----|--------|-------|----------|
| SOA-1 | Start all services | `python run_services.py` | 6 services start on ports 8000–8005 |
| SOA-2 | Gateway health | `GET :8000/health` | every service reports `ok` |
| SOA-3 | Preprocessing service | `POST :8001/preprocess {"text":"How to INVEST?"}` | `{"tokens":["invest"]}` |
| SOA-4 | Indexing service | `GET :8002/stats` | num_docs 522931, vocab_size ~64000 |
| SOA-5 | Refinement service | `POST :8003/suggest {"query":"invset in stock"}` | suggestion with `invset→invest` |
| SOA-6 | Retrieval service | `POST :8004/search {"query":"...","model":"BM25"}` | ranked results + `ms` |
| SOA-7 | Evaluation service (REST→retrieval) | `POST :8005/evaluate {"model":"BM25","sample_queries":50}` | metrics dict; demonstrates loose coupling |
| SOA-8 | Gateway orchestration | `POST :8000/search {"query":"how to invset...","refine":true}` | refinement applied, then retrieval; combined response |
| SOA-9 | Independent run | start only `uvicorn api.retrieval_service:app --port 8004` | service works alone |
| SOA-10 | Smoke test | `python test_services.py` | all 6 checks print sensible output |

## 12. Edge cases & robustness

| ID | Checks | Input | Expected |
|----|--------|-------|----------|
| EDGE-1 | Empty query | "" | no crash; no results / prompt ignored |
| EDGE-2 | All-stopword query | "the of in a" | empty after preprocessing; graceful "no results" |
| EDGE-3 | Out-of-vocabulary query | "zzzqxqz" | no matches; no crash |
| EDGE-4 | Very long query | a paragraph | processed + ranked normally |
| EDGE-5 | Non-ASCII / symbols | "¿qué? 😀 #stocks" | symbols stripped; "stocks" stemmed and searched |
| EDGE-6 | Repeated startup | re-run any search script | NLTK not re-downloaded (fast startup) |

---

### Suggested demo script (12–15 min)
1. ENV-2 (qrels proof) → 2. BM25-1 + BM25-2 (param control) → 3. W2V-2 vs W2V-3 (semantic strength + weakness) →
4. HYB-2/HYB-3 (fusion) → 5. QR-1/QR-3 (refinement) → 6. CL-4 + CL-5 (clustering with/without + before/after) →
7. EV-2 + charts (the key section) → 8. UI walkthrough → 9. SOA-1/SOA-2/SOA-8 (services + gateway).
