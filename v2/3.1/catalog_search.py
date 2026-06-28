"""Lightweight semantic index over the feature CATALOG (metadata only).

Lets an LLM/agent answer "do we already have X / what is adjacent / where is the
white-space" over hundreds of thousands of features WITHOUT reading 75k+ LOC of
source. Implements TF-IDF + cosine similarity BY HAND on numpy + stdlib only --
no sklearn, no embeddings, no network. Search by meaning, surface near-duplicate
blocks, and score how well an idea is already covered.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

# English + domain stopwords: noise that should never drive a match.
_STOP: frozenset[str] = frozenset(
    {
        "the", "of", "a", "an", "and", "or", "to", "in", "on", "for", "per",
        "by", "with", "as", "at", "is", "are", "be", "from", "this", "that",
        "it", "its", "into", "over", "each", "via", "no", "not", "but", "if",
        "then", "than", "so", "all", "any", "we", "you", "our", "your",
        # domain filler that adds no discriminative signal across feature docs
        "feature", "features", "block", "blocks", "column", "columns",
        "compute", "value", "values", "signal", "df",
    }
)

# snake_case boundary handled by the non-alnum split; these catch camelCase and
# digit<->letter transitions inside an already-lowercased-source token.
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])")
_DIGIT_LETTER = re.compile(r"(?<=[0-9])(?=[A-Za-z])|(?<=[A-Za-z])(?=[0-9])")
_NON_ALNUM = re.compile(r"[^0-9A-Za-z]+")


def tokenize(text: str) -> list[str]:
    """Split text into normalized tokens.

    Splits on non-alphanumerics, snake_case, camelCase and digit<->letter
    boundaries; lowercases; drops a small stoplist and length<2 tokens -- except
    single-char window/param markers (a lone letter like "w" or a lone digit),
    which carry meaning. e.g. "roll_zscore_w20" -> ["roll","zscore","w","20"];
    "xsRank" -> ["xs","rank"].
    """
    if not text:
        return []
    # split camelCase and digit/letter boundaries BEFORE lowercasing
    text = _CAMEL.sub(" ", text)
    text = _DIGIT_LETTER.sub(" ", text)
    raw = _NON_ALNUM.sub(" ", text).lower().split()
    out: list[str] = []
    for tok in raw:
        if tok in _STOP:
            continue
        # keep len>=2 always; keep len-1 only when it is a window/param marker
        # produced by the digit<->letter split (a bare letter or bare digit).
        if len(tok) < 2 and not tok.isalnum():
            continue
        out.append(tok)
    return out


def _doc_text(block: dict) -> str:
    """Join name + description + tags + family + produces into one searchable text."""
    parts: list[str] = [str(block.get("name", "")), str(block.get("description", ""))]
    parts.extend(str(t) for t in (block.get("tags") or []))
    fam = block.get("family")
    if fam:
        parts.append(str(fam))
    parts.extend(str(p) for p in (block.get("produces") or []))
    return " ".join(parts)


class CatalogIndex:
    """A TF-IDF + cosine index over catalog["blocks"] metadata."""

    def __init__(self, catalog: dict) -> None:
        blocks = list(catalog.get("blocks", []))
        self.names: list[str] = [str(b.get("name", f"block_{i}")) for i, b in enumerate(blocks)]
        token_lists = [tokenize(_doc_text(b)) for b in blocks]

        # vocabulary + document frequency
        df: Counter[str] = Counter()
        for toks in token_lists:
            df.update(set(toks))
        self.vocab: dict[str, int] = {tok: i for i, tok in enumerate(sorted(df))}
        n_docs = max(len(token_lists), 1)
        # smoothed idf, >= 0 so weights never go negative
        self.idf: dict[str, float] = {
            tok: math.log((1.0 + n_docs) / (1.0 + df[tok])) + 1.0 for tok in self.vocab
        }

        # per-document L2-normalized sparse TF-IDF vectors: dict[token_id]->weight
        self.docs: list[dict[int, float]] = [self._vectorize(toks) for toks in token_lists]

    def _vectorize(self, tokens: list[str]) -> dict[int, float]:
        """Sparse, L2-normalized TF-IDF vector keyed by vocab id."""
        if not tokens:
            return {}
        tf: Counter[str] = Counter(tokens)
        vec: dict[int, float] = {}
        for tok, count in tf.items():
            idx = self.vocab.get(tok)
            if idx is None:
                continue  # OOV (query-side only)
            # sublinear tf damps repeated tokens within one short doc
            vec[idx] = (1.0 + math.log(count)) * self.idf[tok]
        norm = math.sqrt(sum(w * w for w in vec.values()))
        if norm > 0.0:
            for idx in vec:
                vec[idx] /= norm
        return vec

    @staticmethod
    def _cosine(a: dict[int, float], b: dict[int, float]) -> float:
        """Dot product of two already-L2-normalized sparse vectors == cosine."""
        if not a or not b:
            return 0.0
        # iterate the shorter vector for speed
        if len(b) < len(a):
            a, b = b, a
        dot = 0.0
        for idx, w in a.items():
            other = b.get(idx)
            if other is not None:
                dot += w * other
        # clamp to [0,1] against tiny float drift
        if dot < 0.0:
            return 0.0
        return 1.0 if dot > 1.0 else dot

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Top-k (block_name, cosine_score) for the query, sorted desc."""
        qvec = self._vectorize(tokenize(query))
        if not qvec:
            return []
        scored = [(self.names[i], self._cosine(qvec, dv)) for i, dv in enumerate(self.docs)]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return [(n, s) for n, s in scored[: max(k, 0)] if s > 0.0]

    def near_duplicates(
        self, threshold: float = 0.85, max_pairs: int = 50
    ) -> list[tuple[str, str, float]]:
        """All block pairs with cosine >= threshold (redundant features), desc."""
        # invert: token_id -> docs containing it, so we only compare docs that
        # share at least one token instead of all O(n^2) pairs.
        postings: dict[int, list[int]] = defaultdict(list)
        for di, vec in enumerate(self.docs):
            for idx in vec:
                postings[idx].append(di)
        seen: set[tuple[int, int]] = set()
        pairs: list[tuple[str, str, float]] = []
        for doc_ids in postings.values():
            for a in range(len(doc_ids)):
                for b in range(a + 1, len(doc_ids)):
                    i, j = doc_ids[a], doc_ids[b]
                    key = (i, j) if i < j else (j, i)
                    if key in seen:
                        continue
                    seen.add(key)
                    sim = self._cosine(self.docs[i], self.docs[j])
                    if sim >= threshold:
                        pairs.append((self.names[i], self.names[j], sim))
        pairs.sort(key=lambda x: (-x[2], x[0], x[1]))
        return pairs[: max(max_pairs, 0)]

    def coverage(self, query: str) -> float:
        """Max cosine of the query against any doc. ~0 == white-space."""
        hits = self.search(query, k=1)
        return hits[0][1] if hits else 0.0

    def whitespace(
        self, queries: list[str], max_cov: float = 0.2
    ) -> list[tuple[str, float]]:
        """Candidate topics with coverage <= max_cov (gaps worth researching)."""
        gaps = [(q, self.coverage(q)) for q in queries]
        gaps = [(q, c) for q, c in gaps if c <= max_cov]
        gaps.sort(key=lambda x: (x[1], x[0]))
        return gaps


def build_index(catalog: dict) -> CatalogIndex:
    """Convenience constructor."""
    return CatalogIndex(catalog)


def _demo_catalog() -> dict:
    """A small synthetic catalog shaped like registry.build_catalog() output."""
    blocks = [
        # --- volatility family (two intentionally near-duplicate blocks) ---
        {
            "name": "vol_realized_w20",
            "description": "Rolling 20-day realized volatility regime of close-to-close returns.",
            "tags": ["volatility", "rolling", "regime", "risk"],
            "family": "volatility",
            "produces": ["vol_realized_w20"],
        },
        {
            "name": "vol_realized_window20",
            "description": "Realized volatility over a rolling 20 day window of returns; regime risk.",
            "tags": ["volatility", "rolling", "regime", "risk"],
            "family": "volatility",
            "produces": ["vol_realized_window20"],
        },
        {
            "name": "vol_garch_proxy",
            "description": "GARCH-style conditional volatility proxy capturing variance clustering.",
            "tags": ["volatility", "garch", "regime"],
            "family": "volatility",
            "produces": ["vol_garch_proxy"],
        },
        # --- momentum / rolling family ---
        {
            "name": "mom_roll_zscore_w20",
            "description": "Rolling z-score of 20-day price momentum within ticker.",
            "tags": ["momentum", "rolling", "zscore"],
            "family": "momentum",
            "produces": ["mom_roll_zscore_w20"],
        },
        {
            "name": "mom_xsRank_60",
            "description": "Cross-sectional rank of 60-day return momentum across the universe.",
            "tags": ["momentum", "cross_sectional", "rank"],
            "family": "momentum",
            "produces": ["mom_xs_rank_60"],
        },
        {
            "name": "trend_slope_w50",
            "description": "Slope of a 50-day rolling linear trend on log price.",
            "tags": ["momentum", "trend", "rolling"],
            "family": "momentum",
            "produces": ["trend_slope_w50"],
        },
        # --- volume family ---
        {
            "name": "vol_dollar_turnover",
            "description": "Daily dollar turnover, price times share volume traded.",
            "tags": ["volume", "liquidity", "turnover"],
            "family": "volume",
            "produces": ["dollar_turnover"],
        },
        {
            "name": "obv_accumulation",
            "description": "On-balance volume accumulation distribution flow over 30 days.",
            "tags": ["volume", "flow", "accumulation"],
            "family": "volume",
            "produces": ["obv_30"],
        },
        {
            "name": "amihud_illiquidity_w20",
            "description": "Amihud illiquidity ratio of absolute return to dollar volume.",
            "tags": ["volume", "liquidity", "illiquidity"],
            "family": "volume",
            "produces": ["amihud_w20"],
        },
    ]
    return {"n_blocks": len(blocks), "blocks": blocks}


if __name__ == "__main__":
    catalog = _demo_catalog()
    index = build_index(catalog)

    print("=== catalog_search smoke test ===")
    print(f"indexed {len(index.names)} blocks, vocab size {len(index.vocab)}")

    # (a) search ranks volatility features on top
    print("\n[search] query='volatility regime'")
    hits = index.search("volatility regime", k=5)
    for name, score in hits:
        print(f"  {score:0.3f}  {name}")
    top3 = {n for n, _ in hits[:3]}
    ok_search = bool(hits) and all("vol" in n for n in top3)
    print("PASS: vol features ranked top" if ok_search else "FAIL: search did not rank vol features top")

    # (b) near-duplicates surfaces the two intentionally-similar vol blocks
    print("\n[near_duplicates] threshold=0.6")
    dups = index.near_duplicates(threshold=0.6, max_pairs=10)
    for a, b, sim in dups:
        print(f"  {sim:0.3f}  {a}  <->  {b}")
    ok_dup = any(
        {"vol_realized_w20", "vol_realized_window20"} == {a, b} for a, b, _ in dups
    )
    print("PASS: duplicate vol blocks surfaced" if ok_dup else "FAIL: duplicates not surfaced")

    # (c) whitespace flags topics absent from the catalog
    print("\n[whitespace] candidate topics")
    candidates = ["insider trading signal", "options skew", "earnings drift"]
    gaps = index.whitespace(candidates, max_cov=0.2)
    for topic, cov in gaps:
        print(f"  cov={cov:0.3f}  {topic}")
    ok_gap = {t for t, _ in gaps} == set(candidates)
    print("PASS: all white-space topics flagged" if ok_gap else "FAIL: white-space detection wrong")

    all_ok = ok_search and ok_dup and ok_gap
    print("\n=== ALL PASS ===" if all_ok else "\n=== SOME CHECKS FAILED ===")
