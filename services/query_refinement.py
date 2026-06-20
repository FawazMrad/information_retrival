# ============================================================
# services/query_refinement.py  -  QUERY REFINEMENT  (req 5)
#
# Three refinement techniques, all reusing what we already built
# (no new downloads):
#   1. Spell correction  -> fix out-of-vocabulary query terms to the
#                           closest term in the index vocabulary,
#                           preferring the most FREQUENT candidate.
#   2. Synonym expansion -> add the most similar words from the trained
#                           Word2Vec model, above a similarity threshold.
#   3. Session history   -> lightly fold in terms from the user's
#                           previous queries in the same session.
# ============================================================
import difflib


class QueryRefinement:
    def __init__(self, preprocessor, vocab=None, w2v=None, df=None):
        self.prep = preprocessor
        self.vocab = set(vocab) if vocab else set()
        self.vocab_list = list(self.vocab)
        self.w2v = w2v                       # Word2VecEmbedding (or None)
        self.df = df or {}                   # term -> document frequency

    # ---------- 1) spell correction ----------
    def correct_term(self, stem):
        if not self.vocab or stem in self.vocab:
            return stem
        matches = difflib.get_close_matches(stem, self.vocab_list, n=6, cutoff=0.8)
        if not matches:
            return stem
        # Among look-alike candidates, prefer the most frequent term -
        # a common word is far more likely to be the intended correction
        # (so "invset" -> "invest", not the rare "invet").
        if self.df:
            matches.sort(key=lambda w: self.df.get(w, 0), reverse=True)
        return matches[0]

    # ---------- 2) synonym / semantic expansion ----------
    def expand_term(self, stem, n=2, min_sim=0.55):
        if self.w2v is None:
            return []
        wv = self.w2v.model.wv
        if stem not in wv:
            return []
        # keep only sufficiently-similar words -> drops noisy expansions
        return [w for w, s in wv.most_similar(stem, topn=n) if s >= min_sim]

    # ---------- combined ----------
    def refine(self, query, use_spell=True, use_expand=True,
               expand_n=2, expand_min_sim=0.55, history_tokens=None):
        """Return (refined_tokens, info) where info explains what changed."""
        base = self.prep.process(query, mode="lexical")     # stemmed tokens
        info = {"original": list(base), "corrections": {}, "expansions": {}}

        tokens = []
        for t in base:
            c = self.correct_term(t) if use_spell else t
            if c != t:
                info["corrections"][t] = c
            tokens.append(c)

        refined = list(tokens)
        if use_expand:
            for t in tokens:
                exp = self.expand_term(t, n=expand_n, min_sim=expand_min_sim)
                exp = [w for w in exp if w not in refined]
                if exp:
                    info["expansions"][t] = exp
                    refined.extend(exp)

        if history_tokens:
            for t in history_tokens:
                if t not in refined:
                    refined.append(t)
            info["history_added"] = list(history_tokens)

        info["refined"] = refined
        return refined, info

    def suggest(self, query, expand_n=3):
        """Human-readable suggestion string for the UI."""
        _, info = self.refine(query, expand_n=expand_n)
        bits = []
        if info["corrections"]:
            fixes = ", ".join(f"{a}->{b}" for a, b in info["corrections"].items())
            bits.append(f"did you mean: {fixes}")
        related = sorted({w for ws in info["expansions"].values() for w in ws})
        if related:
            bits.append("related terms: " + ", ".join(related[:6]))
        return " | ".join(bits) if bits else "(no suggestions)"
