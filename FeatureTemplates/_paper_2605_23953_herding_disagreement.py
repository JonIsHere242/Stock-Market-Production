"""
_paper_2605_23953_herding_disagreement.py
------------------------------------------
Per-ticker OHLCV-only proxy features for herding and investor disagreement
dynamics, inspired by:

  "Game-Theoretic Modeling of Heterogeneous Investor Interactions for Stock
  Price Forecasting", arXiv 2605.23953

True cross-sectional herding metrics require multi-stock data.  This block
implements the closest causal, per-ticker approximations using only OHLCV:

  hrd_vol_disagree_20     : intraday range / volume-z-score ratio  (high =>
                             disagreement: lots of volume, wide price dispersion)
  hrd_dir_consensus_10    : fraction of last 10 days all moving same direction
  hrd_vw_dir_agree_20     : volume-weighted directional agreement over 20 days
  hrd_churn_20            : high volume / small net-move ratio (two-sided churn)
  hrd_body_ratio_10       : candle body / total range (low => indecision;
                             high => directional conviction)
  hrd_conviction_score_30 : composite herding/conviction z-score (causal, 30-day)
  hrd_vol_price_impact_20 : volume-normalised absolute return (consensus signal)
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
METADATA = {
    "name":        "herding_disagreement",
    "description": (
        "Per-ticker herding / investor-disagreement proxy features derived from "
        "OHLCV price-volume relationships (causal rolling windows only); inspired "
        "by arXiv:2605.23953 game-theoretic heterogeneous-investor model."
    ),
    "requires":    ["Open", "High", "Low", "Close", "Volume"],
    "produces": [
        "hrd_vol_disagree_20",
        "hrd_dir_consensus_10",
        "hrd_vw_dir_agree_20",
        "hrd_churn_20",
        "hrd_body_ratio_10",
        "hrd_conviction_score_30",
        "hrd_vol_price_impact_20",
    ],
    "tags":    ["volume", "volatility", "market_regime", "experimental"],
    "version": "1.0",
    "author":  "paper block — arXiv:2605.23953 (herding/disagreement per-ticker proxy)",
}

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Causal z-score: (x - past_mean) / past_std using shift(1) so today is excluded."""
    past = series.shift(1)
    mu  = past.rolling(window, min_periods=max(5, window // 4)).mean()
    sd  = past.rolling(window, min_periods=max(5, window // 4)).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (series - mu) / sd.replace(0.0, np.nan)
    return z


def _safe_div(num: pd.Series, den: pd.Series, fill: float = np.nan) -> pd.Series:
    """Element-wise division; replaces zero/nan denominators with *fill*."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = num / den.replace(0.0, np.nan)
    if not np.isnan(fill):
        result = result.fillna(fill)
    return result


# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute herding/disagreement proxy features.  All windows are CAUSAL
    (past-only).  Leading NaNs are expected and left as-is.

    Contract: ONLY add the columns listed in METADATA['produces'].
              NEVER modify, drop, or reindex existing columns.
              Always return df.
    """
    idx    = df.index
    close  = df["Close"]
    open_  = df["Open"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    new: dict[str, pd.Series] = {}

    # ------------------------------------------------------------------
    # 1. hrd_vol_disagree_20
    #    Disagreement signal: intraday range relative to volume activity.
    #    High value  => lots of volume AND wide intraday range => disagreement.
    #    Low value   => lots of volume AND tight range         => herding/consensus.
    #
    #    vol_z is the causal 20-day z-score of volume (shift(1) baseline).
    #    range_pct = (High - Low) / Close normalises for price level.
    #    We divide range_pct by (vol_z + 3) so high volume *reduces* the ratio,
    #    but we guard against extreme vol_z signs by clipping.
    # ------------------------------------------------------------------
    range_pct = _safe_div(high - low, close)

    vol_z_20 = _rolling_zscore(volume, 20)
    # Denominator: shift vol_z so it is always positive (clip at -2.5 → floor 0.5)
    vol_z_denom = (vol_z_20.clip(lower=-2.5) + 3.0)   # range [0.5, ∞)

    new["hrd_vol_disagree_20"] = (range_pct / vol_z_denom).replace(
        [np.inf, -np.inf], np.nan
    )

    # ------------------------------------------------------------------
    # 2. hrd_dir_consensus_10
    #    Directional-run consensus: fraction of the past 10 days where
    #    daily returns share the SAME sign as the majority direction.
    #    Pure agreement => 1.0; perfect split => 0.5.
    #    Uses shift(1) so today's close is not included.
    # ------------------------------------------------------------------
    daily_ret = close.pct_change(fill_method=None)          # today's return, no fill
    direction = np.sign(daily_ret)                           # -1, 0, +1

    # Past-only: shift(1) so row t sees rows t-1 … t-10
    dir_past = direction.shift(1)

    def _consensus(arr: np.ndarray) -> float:
        valid = arr[~np.isnan(arr)]
        if len(valid) < 3:
            return np.nan
        pos_frac = (valid > 0).mean()
        neg_frac = (valid < 0).mean()
        return float(max(pos_frac, neg_frac))

    new["hrd_dir_consensus_10"] = dir_past.rolling(10, min_periods=5).apply(
        _consensus, raw=True
    )

    # ------------------------------------------------------------------
    # 3. hrd_vw_dir_agree_20
    #    Volume-weighted directional agreement over 20 days.
    #    Each day contributes its volume × direction sign.
    #    Normalised by total volume so result is in [-1, +1].
    #    +1 = all volume flowed in one direction (herding).
    #     0 = equal two-way flow (maximum disagreement / churn).
    #    Uses shift(1) on both volume and direction for causality.
    # ------------------------------------------------------------------
    signed_vol = volume.shift(1) * dir_past   # volume-weighted direction (past)
    vol_past   = volume.shift(1)

    sv_sum  = signed_vol.rolling(20, min_periods=10).sum()
    vol_sum = vol_past.rolling(20, min_periods=10).sum()

    new["hrd_vw_dir_agree_20"] = _safe_div(sv_sum, vol_sum)

    # ------------------------------------------------------------------
    # 4. hrd_churn_20
    #    Churn / two-sided disagreement:
    #    high volume that produces little net price move.
    #    = cumulative volume / abs(net price change) over 20 days, normalised.
    #    A high ratio means lots of shares changed hands but price barely moved:
    #    buyers and sellers are in rough balance (disagreement / churn).
    #    Both series shifted so bar t uses bars t-1 … t-20.
    # ------------------------------------------------------------------
    net_move_20  = close.shift(1).diff(20).abs().replace(0.0, np.nan)
    vol_sum_20   = vol_past.rolling(20, min_periods=10).sum()

    churn_raw = _safe_div(vol_sum_20, net_move_20)
    # Z-score causal normalisation so values are comparable across stocks/time
    new["hrd_churn_20"] = _rolling_zscore(churn_raw, 60).replace(
        [np.inf, -np.inf], np.nan
    )

    # ------------------------------------------------------------------
    # 5. hrd_body_ratio_10
    #    Candle body-to-range ratio (10-day rolling mean).
    #    body  = |Close - Open|  (directional commitment)
    #    range = High - Low      (total intraday uncertainty)
    #    low ratio => Doji-like indecision / disagreement
    #    high ratio => conviction / herding toward one direction
    #    Uses current bar's OHLC — no look-ahead because open/high/low/close
    #    are all realised within the same day.
    # ------------------------------------------------------------------
    body  = (close - open_).abs()
    range_ = (high - low).replace(0.0, np.nan)
    body_ratio = _safe_div(body, range_)   # [0, 1] per bar

    new["hrd_body_ratio_10"] = body_ratio.shift(1).rolling(10, min_periods=5).mean()

    # ------------------------------------------------------------------
    # 6. hrd_vol_price_impact_20
    #    Volume-normalised absolute return.
    #    High impact per unit volume => consensus / herding: price moves easily.
    #    Low impact per unit volume  => churn / disagreement: volume absorbed.
    #    vol_z used as denominator after clipping to keep it positive.
    # ------------------------------------------------------------------
    abs_ret    = daily_ret.abs()
    vol_z_clp  = vol_z_20.clip(lower=-1.5) + 2.5   # floor at 1.0

    impact_raw = abs_ret / vol_z_clp
    # Smooth with a 20-day rolling mean (past-only via shift(1))
    new["hrd_vol_price_impact_20"] = impact_raw.shift(1).rolling(
        20, min_periods=10
    ).mean().replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # 7. hrd_conviction_score_30
    #    Composite causal z-score combining:
    #      + hrd_body_ratio_10   (conviction from candle bodies)
    #      + hrd_vw_dir_agree_20 (volume-weighted directional agreement)
    #      - hrd_vol_disagree_20 (disagreement indicator, inverted)
    #    Each component is z-scored over a 30-day causal window, then averaged.
    #    Positive => herding/consensus; Negative => disagreement/churn.
    # ------------------------------------------------------------------
    comp_body  =  _rolling_zscore(new["hrd_body_ratio_10"],    30)
    comp_agree =  _rolling_zscore(new["hrd_vw_dir_agree_20"],  30)
    comp_disag = -_rolling_zscore(new["hrd_vol_disagree_20"],  30)  # inverted

    # Stack components and mean across available (ignores NaN columns per row)
    stack = pd.concat(
        [comp_body.rename("b"), comp_agree.rename("a"), comp_disag.rename("d")],
        axis=1,
    )
    new["hrd_conviction_score_30"] = stack.mean(axis=1, skipna=True).replace(
        [np.inf, -np.inf], np.nan
    )

    # ------------------------------------------------------------------
    # Attach all produced columns at once (single concat, index-aligned)
    # ------------------------------------------------------------------
    new_df = pd.DataFrame(new, index=idx)
    return pd.concat([df, new_df], axis=1)
