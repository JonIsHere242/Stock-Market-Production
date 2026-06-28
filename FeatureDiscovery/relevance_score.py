"""
relevance_score.py  --  Quick-and-dirty triage of fetched arXiv papers.

WHAT THIS IS (AND IS NOT)
-------------------------
A CHEAP, deterministic, zero-cost pre-filter. Its only jobs are:
  1. Throw out papers that obviously can't become a FeatureFramework block, the
     biggest win being papers that need data we don't have. A feature block is just
     compute(df) where df is OHLCV (+ index/VIX via the _indexes helper). Anything that
     needs a limit order book, tick/microstructure data, news/sentiment text,
     fundamentals, the full options surface, or alt-data CANNOT be implemented -- so it
     is hard-capped regardless of how clever it is.
  2. Rank the survivors so a human (or a stronger model, later) only looks at the top.

It is explicitly NOT the acceptance gate. It does not measure predictive power, leakage,
or marginal value -- those come later and are the parts that actually decide if a feature
is real. Treat rel_score as "worth a closer look", nothing more.

Deliberately NOT scored: "claimed performance / Sharpe / outperformance". Heuristically
rewarding those just selects for the most overfit, cherry-picked papers. We stay neutral
on claims and let the downstream IC/leakage gates do the judging.

USAGE
-----
  # Score the store in place and print the top 30
  python FeatureDiscovery/relevance_score.py --top 30

  # Custom threshold for the rel_keep flag (coarse pre-filter only)
  python FeatureDiscovery/relevance_score.py --threshold 5.5 --top 50

LLM SEAM (later)
----------------
score_paper_llm() is a stub. When you want a sharper triage on the survivors, call a
local model (Ollama) or a frontier model on ONLY the papers that clear the heuristic
gate -- that is the cost-optimal split (pennies, because volume is already cut ~90%).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT       = Path(__file__).resolve().parent.parent
STORE_PATH = ROOT / "Data" / "PaperFeed" / "papers.parquet"

# ---------------------------------------------------------------------------
# Keyword rubric  (edit freely -- all matching is whole-word, case-insensitive)
# ---------------------------------------------------------------------------

# Axis 1: finance focus.  NOTE: overloaded tokens removed on purpose -- "alpha"
# (Greek/learning-rate), "trade" (matches "trade-off"), "index" (array index) caused
# generic ML papers to read as finance. Keep terms that are unambiguously markets.
FINANCE_TERMS = [
    "stock", "stocks", "equity", "equities", "market", "markets", "trading",
    "price", "prices", "stock return", "returns", "volatility", "portfolio", "asset",
    "financial", "finance", "investment", "factor model", "momentum", "sharpe",
    "backtest", "cross-sectional", "securities", "exchange", "limit order",
]

# Axis 2: time-series / predictive relevance
TS_TERMS = [
    "time series", "time-series", "forecast", "forecasting", "prediction", "predictive",
    "predict", "signal", "feature", "features", "regime", "trend", "mean reversion",
    "mean-reversion", "autocorrelation", "nonstationary", "non-stationary",
    "sequence", "temporal", "lead-lag", "anomaly",
]

# Axis 3: methods that ARE computable from OHLCV (boosts implementability)
METHOD_TERMS = [
    "technical indicator", "moving average", "entropy", "wavelet", "fourier", "spectral",
    "fractal", "hurst", "visibility graph", "network", "graph", "kalman", "hilbert",
    "rolling", "z-score", "zscore", "quantile", "kurtosis", "skewness", "garch",
    "stochastic", "hawkes", "permutation entropy", "complexity", "detrended", "dfa",
    "matrix", "eigenvalue", "lyapunov", "recurrence", "phase space", "embedding",
]

# Axis 3 (negative): needs data a compute(df) cannot see -> HARD CAP.
# NOTE: SEC fundamentals USED to be red-flagged here, but we now serve point-in-time XBRL
# fundamentals through FeatureTemplates/_fundamentals.py (built by build_fundamentals_panel.py).
# So fundamental/balance-sheet/10-K/10-Q/filing terms moved OUT of the cap and INTO the positive
# FUNDAMENTALS_TERMS axis below. Data we STILL can't serve stays capped. ("analyst" stays: we have
# filed fundamentals, NOT analyst estimates/recommendations; macro stays: no _macro helper yet.)
EXTERNAL_DATA_REDFLAGS = [
    "limit order book", "order book", "order flow", "microstructure", "tick data",
    "high-frequency", "high frequency", "bid-ask", "bid ask", "quote", "level-2", "level 2",
    "news", "sentiment", "twitter", "reddit", "social media", "text", "nlp",
    "language model", "llm", "transformer text", "headlines", "earnings call",
    "analyst", "options surface", "implied volatility surface", "option chain", "greeks",
    "satellite", "alternative data", "alt-data", "credit card", "supply chain",
    "on-chain", "blockchain", "crypto", "cryptocurrency", "bitcoin",
    "macroeconomic", "interest rate", "central bank", "esg",
]

# Axis 4 (positive): signals built from SEC fundamentals we NOW serve point-in-time. A hit marks
# the paper as finance-relevant AND implementable-with-our-data (it lifts the finance gate and
# implementability, and is no longer hard-capped).
FUNDAMENTALS_TERMS = [
    "fundamental", "fundamentals", "balance sheet", "income statement", "cash flow statement",
    "financial statement", "accounting", "accruals", "10-k", "10-q", "filing", "xbrl",
    "earnings", "earnings surprise", "book-to-market", "book to market", "price-to-earnings",
    "price to earnings", "valuation ratio", "profitability", "value factor", "quality factor",
    "gross profitability", "net income", "free cash flow", "return on equity", "leverage",
]

# "Is this a computable time-series method?" weights (sum to 1.0). Finance is NOT here --
# it is applied multiplicatively as a gate below. Novelty/performance intentionally
# excluded (heuristically rewarding claims just selects for overfit papers).
METHOD_WEIGHTS = {
    "ts":     0.45,
    "impl":   0.35,
    "method": 0.20,
}

# Saturation targets: this many distinct hits -> subscore ~1.0 for that axis.
_SAT = {"finance": 3, "ts": 3, "method": 3}
_TITLE_BONUS = 2          # a title hit counts as this many body hits
_EXTERNAL_CAP = 3.0       # max rel_score if a hard external-data red flag is present
_FIN_GATE_FLOOR = 0.20    # non-finance papers keep this fraction of their method score


def _matches(terms: list[str], title: str, abstract: str) -> list[str]:
    """Return the distinct terms that appear (whole-word) in title/abstract."""
    hits: list[str] = []
    for t in terms:
        pat = r"\b" + re.escape(t) + r"\b"
        in_title = re.search(pat, title) is not None
        in_abs   = re.search(pat, abstract) is not None
        if in_title or in_abs:
            hits.append(t)
    return hits


def _weighted_count(terms: list[str], title: str, abstract: str) -> tuple[float, list[str]]:
    """Count hits, weighting title hits higher; also return the matched terms."""
    hits = _matches(terms, title, abstract)
    score = 0.0
    for t in hits:
        pat = r"\b" + re.escape(t) + r"\b"
        score += _TITLE_BONUS if re.search(pat, title) else 1.0
    return score, hits


def score_paper(title: str, abstract: str, primary_category: str = "") -> dict:
    """
    Heuristic 0-10 relevance score with per-axis breakdown and matched-term reasons.
    Pure function -- no I/O, deterministic.
    """
    title = (title or "").lower()
    abstract = (abstract or "").lower()
    pcat = (primary_category or "").lower()

    fin_n,  fin_hits = _weighted_count(FINANCE_TERMS, title, abstract)
    ts_n,   ts_hits  = _weighted_count(TS_TERMS, title, abstract)
    meth_n, meth_hits = _weighted_count(METHOD_TERMS, title, abstract)
    fund_n, fund_hits = _weighted_count(FUNDAMENTALS_TERMS, title, abstract)
    redflags = _matches(EXTERNAL_DATA_REDFLAGS, title, abstract)

    # Subscores in [0, 1]. Fundamentals terms count as finance signal (they ARE markets) AND
    # now-implementable, since we serve PIT XBRL via _fundamentals.py.
    s_fin  = min((fin_n + fund_n) / _SAT["finance"], 1.0)
    if pcat.startswith("q-fin"):
        s_fin = max(s_fin, 0.7)                      # q-fin primary => finance by construction
    s_ts   = min(ts_n / _SAT["ts"], 1.0)
    s_meth = min(meth_n / _SAT["method"], 1.0)

    # Implementability: start full, decay with each external-data red flag.
    needs_external = len(redflags) > 0
    s_impl = max(0.0, 1.0 - 0.5 * len(redflags))
    if not needs_external:
        s_impl = max(s_impl, 0.3 * s_meth)           # a computable method nudges impl up
        if fund_n > 0:
            s_impl = max(s_impl, 0.6)                # fundamentals signals are implementable now

    # "Is this a computable time-series method?"  (0-10, finance-agnostic)
    base_method = 10.0 * (
        METHOD_WEIGHTS["ts"]     * s_ts
        + METHOD_WEIGHTS["impl"]   * s_impl
        + METHOD_WEIGHTS["method"] * s_meth
    )
    # Finance relevance is a GATE, not an additive axis: a paper must be about markets
    # to matter to us. Floor (_FIN_GATE_FLOOR) demotes -- not zeroes -- a strongly
    # transferable pure-method paper, so a great cs.LG entropy method still surfaces low.
    finance_gate = _FIN_GATE_FLOOR + (1.0 - _FIN_GATE_FLOOR) * s_fin
    total = base_method * finance_gate
    if needs_external:
        total = min(total, _EXTERNAL_CAP)

    reasons = []
    if fin_hits:  reasons.append("fin:" + ",".join(fin_hits[:5]))
    if ts_hits:   reasons.append("ts:" + ",".join(ts_hits[:5]))
    if meth_hits: reasons.append("method:" + ",".join(meth_hits[:5]))
    if fund_hits: reasons.append("fundamentals:" + ",".join(fund_hits[:5]))
    if redflags:  reasons.append("EXTERNAL:" + ",".join(redflags[:5]))

    return {
        "rel_score":          round(total, 3),
        "rel_finance":        round(s_fin, 3),
        "rel_ts":             round(s_ts, 3),
        "rel_impl":           round(s_impl, 3),
        "rel_method":         round(s_meth, 3),
        "needs_external_data": needs_external,
        "rel_reasons":        " | ".join(reasons),
    }


def score_paper_llm(title: str, abstract: str, backend: str = "ollama") -> dict:
    """
    STUB for a later sharper pass. Run this ONLY on papers that already clear the
    heuristic gate (volume is ~90% smaller by then, so even a frontier model costs
    pennies). Wire to Ollama (local, free) or the Anthropic API here when ready.
    """
    raise NotImplementedError(
        "LLM triage not wired yet. Run the heuristic gate first, then score survivors."
    )


# ---------------------------------------------------------------------------
# Batch / CLI
# ---------------------------------------------------------------------------

def score_store(df: pd.DataFrame) -> pd.DataFrame:
    """Apply score_paper to every row, returning df with rel_* columns added/overwritten."""
    scored = df.apply(
        lambda r: score_paper(r.get("title", ""), r.get("abstract", ""),
                              r.get("primary_category", "")),
        axis=1, result_type="expand",
    )
    for col in scored.columns:
        df[col] = scored[col]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Cheap heuristic triage of fetched papers")
    parser.add_argument("--in",  dest="in_path",  default=str(STORE_PATH),
                        help="Input papers parquet (from arxiv_fetch.py)")
    parser.add_argument("--out", dest="out_path", default=None,
                        help="Output parquet (default: overwrite input in place)")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="rel_keep flag threshold (coarse pre-filter, default 5.0)")
    parser.add_argument("--top", type=int, default=25,
                        help="Print this many top-ranked papers (default 25)")
    args = parser.parse_args()

    in_path  = Path(args.in_path)
    out_path = Path(args.out_path) if args.out_path else in_path
    if not in_path.exists():
        raise SystemExit(f"No paper store at {in_path} -- run arxiv_fetch.py first.")

    df = pd.read_parquet(in_path)
    if df.empty:
        raise SystemExit("Paper store is empty.")

    df = score_store(df)
    df["rel_keep"] = df["rel_score"] >= args.threshold
    df = df.sort_values("rel_score", ascending=False).reset_index(drop=True)
    df.to_parquet(out_path, index=False)

    n_keep = int(df["rel_keep"].sum())
    n_ext  = int(df["needs_external_data"].sum())
    print(f"Scored {len(df)} papers  |  keep (>= {args.threshold}): {n_keep}  "
          f"|  external-data (hard-capped): {n_ext}")
    print(f"Saved -> {out_path}\n")

    show = df.head(args.top)
    print(f"{'Score':>5}  {'Ext':>3}  {'Cat':<10}  Title")
    print("-" * 100)
    for _, r in show.iterrows():
        ext = "!" if r["needs_external_data"] else ""
        title = r["title"][:70]
        print(f"{r['rel_score']:>5.2f}  {ext:>3}  {r['primary_category']:<10}  {title}")
    print()
    print("Tip: inspect reasons with  "
          "pd.read_parquet(store)[['rel_score','rel_reasons','abs_url']].head(40)")


if __name__ == "__main__":
    main()
