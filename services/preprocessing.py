# ============================================================
# services/preprocessing.py  -  PREPROCESSING SERVICE  (Step 1)
#
# ONE class is used for BOTH documents and queries, which
# guarantees query/document compatibility (Step 4).
#
# Two output modes:
#   - "lexical"  : list of cleaned tokens (normalize + stopword
#                  removal + stemming/lemmatization).
#                  -> used by TF-IDF, BM25, inverted index.
#   - "semantic" : lightly cleaned string (keeps word forms &
#                  stopwords) -> used by BERT/Word2Vec embeddings,
#                  because transformers handle raw text better.
# ============================================================
import re
import unicodedata
from functools import lru_cache
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer


def _have(resource):
    """True if an NLTK resource is already on disk (no network call)."""
    try:
        nltk.data.find(resource)
        return True
    except LookupError:
        return False


def ensure_nltk(need_wordnet=False, force=False):
    """Download only what's needed, and SKIP the network if already present.
    By default we only need 'stopwords' (we use the 'split' tokenizer and a
    stemmer, which need no extra data). WordNet is fetched only for lemmatization.
    """
    wanted = [("stopwords", "corpora/stopwords")]
    if need_wordnet:
        wanted += [("wordnet", "corpora/wordnet"), ("omw-1.4", "corpora/omw-1.4")]
    for pkg, resource in wanted:
        if not force and _have(resource):
            continue  # already on disk -> no download, no network
        try:
            nltk.download(pkg, quiet=True, force=force)
        except Exception:
            pass


def _load_stopwords(language, remove_stopwords):
    """Load stopwords, self-healing only if the data file is actually corrupt."""
    if not remove_stopwords:
        return set()
    ensure_nltk()
    try:
        return set(stopwords.words(language))
    except Exception:
        # Corrupt file -> force ONE clean re-download, then retry.
        ensure_nltk(force=True)
        return set(stopwords.words(language))


class Preprocessor:
    def __init__(self, language="english", remove_stopwords=True,
                 use_lemmatization=False, min_token_len=2, tokenizer="split"):
        if use_lemmatization:
            ensure_nltk(need_wordnet=True)
        self.language = language
        self.remove_stopwords = remove_stopwords
        self.use_lemmatization = use_lemmatization
        self.min_token_len = min_token_len
        self.tokenizer = tokenizer

        self.stop = _load_stopwords(language, remove_stopwords)
        self.stemmer = SnowballStemmer(language)
        self.lemmatizer = WordNetLemmatizer()
        # Cache word->stem / word->lemma. Vocabulary is bounded, so repeated
        # words (very common in real corpora) are processed only once. Huge speedup.
        self._stem = lru_cache(maxsize=None)(self.stemmer.stem)
        self._lem = lru_cache(maxsize=None)(self.lemmatizer.lemmatize)
        self._url_re = re.compile(r"http\S+|www\.\S+")
        self._keep_re = re.compile(r"[^a-z0-9\s]")
        self._ws_re = re.compile(r"\s+")

    # ---------- shared cleaning ----------
    def normalize(self, text):
        text = text.lower()
        text = unicodedata.normalize("NFKD", text)
        text = self._url_re.sub(" ", text)
        text = self._keep_re.sub(" ", text)     # drop punctuation/symbols
        text = self._ws_re.sub(" ", text).strip()
        return text

    def _tokenize(self, text):
        if self.tokenizer == "nltk":
            from nltk.tokenize import word_tokenize
            return word_tokenize(text)
        return text.split()  # text is already normalized -> split is enough

    # ---------- lexical mode (TF-IDF / BM25 / index) ----------
    def process(self, text, mode="lexical"):
        if mode == "semantic":
            return self.clean_for_embeddings(text)

        text = self.normalize(text)
        tokens = []
        for tok in self._tokenize(text):
            if tok in self.stop:
                continue
            if len(tok) < self.min_token_len:
                continue
            if self.use_lemmatization:
                tok = self._lem(tok)
            else:
                tok = self._stem(tok)
            tokens.append(tok)
        return tokens

    def process_to_string(self, text, mode="lexical"):
        """Same as process() but returns a space-joined string."""
        if mode == "semantic":
            return self.clean_for_embeddings(text)
        return " ".join(self.process(text, mode="lexical"))

    # ---------- semantic mode (embeddings) ----------
    def clean_for_embeddings(self, text):
        text = text.lower()
        text = self._url_re.sub(" ", text)
        text = self._ws_re.sub(" ", text).strip()
        return text
