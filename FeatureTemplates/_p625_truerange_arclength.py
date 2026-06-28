import numpy as np
import pandas as pd

METADATA = {
    "name":        "_p625_truerange_arclength",
    "description": "Range-inflated path arc-length inefficiency: cumulative true-range travel vs net displacement, capturing intrabar whip the close-to-close Kaufman efficiency ratio misses (Sevcik 1998; Kaufman 1995; Wilder 1978).",
    "requires":    [],
    "produces":    ["arcl_eff_20", "arcl_eff_60", "arcl_logpathlen_60"],
    "tags":        ["volatility", "trend", "efficiency", "experimental"],
    "version":     "1.0",
    "author":      "paper:Sevcik (1998) waveform fractal dimension; Kaufman (1995); Wilder (1978) true range",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)

    prev_close = close.shift(1)

    # Wilder (1978) true range: max(H-L, |H-prevC|, |L-prevC|)
    tr = pd.concat(
        [(high - low).abs(),
         (high - prev_close).abs(),
         (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    eps = 1e-8
    for w in (20, 60):
        pathlen = tr.rolling(w, min_periods=w // 2).sum()
        disp = (close - close.shift(w)).abs()
        eff = disp / (pathlen + eps)
        df[f"arcl_eff_{w}"] = eff.clip(lower=0.0, upper=1.0)

    # Normalized (return-scale) cumulative true-range travel, log-compressed.
    tr_norm = tr / prev_close.replace(0, np.nan)
    df["arcl_logpathlen_60"] = np.log1p(tr_norm.rolling(60, min_periods=30).sum())

    return df
