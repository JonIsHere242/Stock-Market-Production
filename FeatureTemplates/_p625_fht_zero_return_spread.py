import numpy as np
import pandas as pd
from scipy.stats import norm

METADATA = {
    "name":        "_p625_fht_zero_return_spread",
    "description": "Fong-Holden-Trzcinka closed-form effective-spread proxy from zero-return "
                   "frequency and realized vol, plus spread term-structure. FHT(2017) RoF 21(4):1355-1401; "
                   "Lesmond-Ogden-Trzcinka(1999) RFS 12(5).",
    "requires":    [],
    "produces":    ["fht_spread_21", "fht_spread_63", "fht_spread_126", "fht_spread_trend"],
    "tags":        ["liquidity", "spread", "microstructure", "experimental"],
    "version":     "1.0",
    "author":      "paper:Fong-Holden-Trzcinka(2017) RoF 21(4):1355-1401; Lesmond-Ogden-Trzcinka(1999) RFS 12(5)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)

    # daily simple returns; first element is NaN (no prior bar)
    ret = close.pct_change()

    # near-zero return indicator; mask the leading NaN so it never counts as a zero-return day
    abs_ret = ret.abs()
    z = (abs_ret <= 1e-4).astype(float)
    z = z.where(ret.notna(), np.nan)

    for w in (21, 63, 126):
        mp = w // 2
        zero_freq = z.rolling(w, min_periods=mp).mean()
        sigma = ret.rolling(w, min_periods=mp).std()

        # FHT closed form: 2 * sigma * Phi^{-1}((1 + ZeroFreq) / 2)
        zf = zero_freq.clip(0.0, 0.999999)
        ppf = pd.Series(norm.ppf((1.0 + zf.to_numpy()) / 2.0), index=close.index)
        fht = 2.0 * sigma * ppf
        df[f"fht_spread_{w}"] = fht.clip(0.0, 0.5)

    # spread term-structure: log ratio of short- to long-window spread
    num = df["fht_spread_21"] + 1e-9
    den = df["fht_spread_126"] + 1e-9
    df["fht_spread_trend"] = np.log(num / den).clip(-10.0, 10.0)

    return df
