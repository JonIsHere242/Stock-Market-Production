import numpy as np
import pandas as pd

METADATA = {
    "name":        "_p625_drawdown_norms",
    "description": "Drawdown-curve L1/L2 norms: Ulcer Index, Pain Index, and Martin ratio (Martin & McCann 1989; Becker/Zephyr Pain Index 2006; Chekhlov, Uryasev & Zabarankin 2005).",
    "requires":    [],
    "produces":    ["dnm_ulcer_63", "dnm_ulcer_126", "dnm_pain_63", "dnm_pain_126", "dnm_martin_63"],
    "tags":        ["volatility", "risk", "drawdown", "experimental"],
    "version":     "1.0",
    "author":      "paper:Martin & McCann (1989); Becker/Zephyr (2006); Chekhlov, Uryasev & Zabarankin (2005)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)

    for W in (63, 126):
        mp = W // 2
        # Trailing rolling peak (only rows <= t).
        peak = close.rolling(W, min_periods=mp).max()

        # Drawdown from peak, in percent. Guard zero/negative peaks -> NaN.
        peak_safe = peak.where(peak > 0, np.nan)

        dd_pct = 100.0 * (close / peak_safe - 1.0)
        df[f"dnm_ulcer_{W}"] = np.sqrt((dd_pct ** 2).rolling(W, min_periods=mp).mean())

        dd_pos = (peak_safe - close) / peak_safe
        df[f"dnm_pain_{W}"] = dd_pos.rolling(W, min_periods=mp).mean()

    # Martin ratio: 63d log-return (in %) divided by Ulcer Index, clipped.
    log_close = np.log(close.where(close > 0, np.nan))
    ret63 = (log_close - log_close.shift(63)) * 100.0
    martin = ret63 / (df["dnm_ulcer_63"] + 1e-6)
    df["dnm_martin_63"] = martin.clip(-50.0, 50.0)

    return df
