"""
_lens_fundamentals.py  --  CONTEXTUAL RANK NORMALIZATION (trailing time-series percentile) of
raw fundamental LEVEL/RATIO columns from the "fundamentals" cluster.

HIDDEN CANDIDATE block (single-underscore file name): excluded from production builds, included
only under --include_candidates. Inert until promoted.

Lens
----
A global monotone transform of a single column is a NO-OP for a gradient-boosted tree (trees split
on rank order). A trailing TIME-SERIES PERCENTILE RANK (WorldQuant ts_rank) injects NEW information
because the comparison set slides each day -- it is the rank of today's value among that SAME
ticker's own trailing window. It is leak-free (uses only past+present) and distribution-free, which
matters here because raw valuation yields and fundamental margins are heavy-tailed and regime-drifting
(plain z-scores break on them).

Transform (the blessed pattern from price_differential_signal_pack.py):
    win = <window>
    mp  = max(20, win // 2)
    df[out] = df[src].rolling(win, min_periods=mp).rank(pct=True)
-> output in (0, 1]; NaN only for the leading mp-1 rows (expected, NOT filled).

Source selection (surgical -- only RAW level/ratio columns; already-ranked/already-z-scored columns
are deliberately SKIPPED to avoid near-redundant noise):
  fvn_earnings_yield      raw E/P yield (clip-only)        -> tsr63, tsr252
  fvn_fcf_yield           raw FCF/P yield (clip-only)      -> tsr63, tsr252
  fqg_fcf_margin          raw fcf_ttm/revenue ratio        -> tsr252
  fqg_accruals_to_assets  raw Sloan accrual ratio          -> tsr252

SKIPPED: fvn_valuation_dynamics (its *_pctl_250 are ALREADY rolling ranks; *_chg / composite are
diffs / z-scores), the *_chg / *_yoy / *_trend / *_momentum columns across fqg blocks (already
differences), and marketcap_dynamics (szc_*_z_* already supply within-ticker context for size).

Contract: adds ONLY the columns in produces; never sorts/reindexes/fills/drops; stateless; pandas
+ numpy only; no helper imports (we only re-rank existing produced columns); always returns df.
"""

from __future__ import annotations

import numpy as np  # noqa: F401  (kept for parity / future use; transform is pure pandas)
import pandas as pd

# (source_col, window) pairs. window 63 ~= quarter, 252 ~= year.
_TARGETS = [
    ("fvn_earnings_yield", 63),
    ("fvn_earnings_yield", 252),
    ("fvn_fcf_yield", 63),
    ("fvn_fcf_yield", 252),
    ("fqg_fcf_margin", 252),
    ("fqg_accruals_to_assets", 252),
]

_REQUIRES = sorted({src for src, _ in _TARGETS})
_PRODUCES = [f"{src}_tsr{win}" for src, win in _TARGETS]

METADATA = {
    "name":        "lens_fundamentals",
    "description": (
        "Trailing time-series percentile rank (ts_rank, pct=True) of raw fundamental level/ratio "
        "columns -- earnings & FCF yields, FCF margin, Sloan accruals -- over each ticker's own "
        "63d / 252d trailing history. Contextual rank normalization (non-monotone, leak-free)."
    ),
    "requires":    _REQUIRES,
    "produces":    _PRODUCES,
    "tags":        ["experimental", "lens", "rank", "fundamentals", "valuation"],
    "version":     "1.0",
    "author":      "lens-batch",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    for src, win in _TARGETS:
        mp = max(20, win // 2)
        df[f"{src}_tsr{win}"] = (
            pd.to_numeric(df[src], errors="coerce")
            .rolling(win, min_periods=mp)
            .rank(pct=True)
        )
    return df
