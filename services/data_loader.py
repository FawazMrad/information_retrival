# ============================================================
# services/data_loader.py  -  DATA LOADING SERVICE
# Wraps ir_datasets so the rest of the system never depends on
# the specific dataset format. Same API for any dataset id.
# ============================================================
import ir_datasets


def load_dataset(dataset_id):
    """Load an ir_datasets dataset object (corpus + queries + qrels)."""
    return ir_datasets.load(dataset_id)


def get_doc_text(doc):
    """Return the full text of a document, joining title + text if present."""
    parts = []
    if hasattr(doc, "title") and doc.title:
        parts.append(str(doc.title))
    if hasattr(doc, "text") and doc.text:
        parts.append(str(doc.text))
    if not parts and hasattr(doc, "body") and doc.body:  # some datasets use 'body'
        parts.append(str(doc.body))
    return " ".join(parts).strip()


def iter_docs(ds, limit=None):
    """Yield (doc_id, text) pairs. Streams - safe for large corpora."""
    for i, doc in enumerate(ds.docs_iter()):
        if limit is not None and i >= limit:
            break
        yield doc.doc_id, get_doc_text(doc)


def iter_queries(ds, limit=None):
    """Yield (query_id, text) pairs."""
    for i, q in enumerate(ds.queries_iter()):
        if limit is not None and i >= limit:
            break
        yield q.query_id, q.text


def load_qrels(ds):
    """Return qrels as {query_id: {doc_id: relevance}} for evaluation later."""
    qrels = {}
    for qr in ds.qrels_iter():
        qrels.setdefault(qr.query_id, {})[qr.doc_id] = int(qr.relevance)
    return qrels


def dataset_stats(ds):
    """Return basic counts. Falls back to iteration if metadata is missing."""
    stats = {}
    # docs
    try:
        stats["num_docs"] = ds.docs_count()
    except Exception:
        stats["num_docs"] = sum(1 for _ in ds.docs_iter())
    # queries
    try:
        stats["num_queries"] = ds.queries_count()
    except Exception:
        try:
            stats["num_queries"] = sum(1 for _ in ds.queries_iter())
        except Exception:
            stats["num_queries"] = None
    # qrels
    try:
        stats["num_qrels"] = ds.qrels_count()
    except Exception:
        try:
            stats["num_qrels"] = sum(1 for _ in ds.qrels_iter())
        except Exception:
            stats["num_qrels"] = None
    return stats
