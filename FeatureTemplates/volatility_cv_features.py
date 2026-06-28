import pandas as pd
import numpy as np

METADATA = {
    "name":        "volatility_cv_features",
    "description": "Coefficient of variation features across multiple time windows with percentile ranking and regime classification",
    "requires":    ["Close"],
    "produces": [
        "cv_10d",
        "cv_10d_percentile",
        "cv_10d_regime_high",
        "cv_10d_regime_low",
        "cv_20d",
        "cv_20d_percentile",
        "cv_20d_regime_high",
        "cv_20d_regime_low",
        "cv_50d",
        "cv_50d_percentile",
        "cv_50d_regime_high",
        "cv_50d_regime_low",
    ],
    "tags":        ["volatility"],
    "version":     "1.0",
    "author":      "migration from monolith",
}


def _rolling_percentile(cv_values: np.ndarray) -> np.ndarray:
    """
    Vectorized replacement for the original per-window apply.

    Original (faithful) semantics, per row t:
      - cv_shifted = cv.shift(1)
      - window = cv_shifted.rolling(252, min_periods=50) ending at t,
        i.e. the slice cv_shifted[max(0, t-251) .. t]. pandas keeps NaN
        entries inside the window; it only requires >=50 NON-NaN values
        before the lambda is invoked (otherwise the result is NaN).
      - lambda: (x <= cv[t]).mean() * 100
        where the comparison treats NaN window entries as False (not <=)
        and `.mean()` divides by the FULL window length (NaNs included).
        A NaN cv[t] makes every comparison False -> 0.0.
      - the `len(x) == 0` branch never fires (rolling yields NaN when
        min_periods isn't met, rather than calling the lambda).

    So per valid row t:
        percentile[t] = count_le / window_len * 100
    where
        window_len = number of elements in the window (incl. NaN),
        count_le   = number of NON-NaN window entries <= cv[t]
                     (0 if cv[t] is NaN).

    We maintain an incrementally-sorted list of the non-NaN window
    values; count_le is a binary search. The denominator is the raw
    window length, NOT the valid count -- matching pandas' `.mean()`.
    """
    n = len(cv_values)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out

    shifted = np.empty(n, dtype=np.float64)
    shifted[0] = np.nan
    shifted[1:] = cv_values[:-1]

    import bisect
    from collections import deque

    window = []        # sorted list of non-NaN shifted values in the window
    buf = deque()      # raw shifted values (incl. NaN), in window order
    count_valid = 0    # number of non-NaN values currently in the window

    for t in range(n):
        v = shifted[t]
        buf.append(v)
        if v == v:  # not NaN
            bisect.insort(window, v)
            count_valid += 1
        # evict the element leaving the 252-wide window
        if len(buf) > 252:
            old = buf.popleft()
            if old == old:  # not NaN
                del window[bisect.bisect_left(window, old)]
                count_valid -= 1

        if count_valid >= 50:
            cvt = cv_values[t]
            window_len = len(buf)  # full window length, incl. NaN entries
            if cvt == cvt:  # not NaN
                cnt_le = bisect.bisect_right(window, cvt)
                out[t] = cnt_le / window_len * 100.0
            else:
                out[t] = 0.0

    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract coefficient of variation features for different windows,
    including historical percentile ranking and regime classification.
    """

    close = df["Close"]

    new_cols = {}

    # Coefficient of variation for different windows
    for window in [10, 20, 50]:
        close_window = close.rolling(window, min_periods=window // 2)
        mean_price = close_window.mean()
        std_price = close_window.std()
        cv = std_price / (mean_price + 1e-8)
        new_cols[f"cv_{window}d"] = cv

        # Historical CV comparison (vectorized rolling percentile rank)
        pct = _rolling_percentile(cv.to_numpy(dtype=np.float64))
        cv_pct = pd.Series(pct, index=df.index)
        new_cols[f"cv_{window}d_percentile"] = cv_pct

        # CV regime classification
        new_cols[f"cv_{window}d_regime_high"] = (cv_pct >= 80).astype(float)
        new_cols[f"cv_{window}d_regime_low"] = (cv_pct <= 20).astype(float)

    result_df = pd.concat(
        [df, pd.DataFrame(new_cols, index=df.index)], axis=1
    )

    return result_df
