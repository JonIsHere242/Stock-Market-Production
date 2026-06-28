"""OOF target-ensemble stacking — the variance-reduction win, validated.

This operationalizes the project's first real predictor-level lift: a RANK-average of
predictions trained on several DIVERSE target arms (e.g. next-1d, next-5d, downside)
beats any single arm and, crucially, lowers the variance of the edge. Here it is done
the honest way:

  1. for each target arm, train a PURGED out-of-fold model (no look-ahead);
  2. rank-average the per-arm OOF predictions cross-sectionally each date
     (RANK-avg, not prob-avg -- prob-avg scrambles the top-1% selection);
  3. evaluate every arm and the ensemble by cross-sectional tail-lift vs the primary
     forward return, reporting both the lift AND its day-to-day SPREAD.

The ensemble should match/beat the best arm's lift while cutting the spread.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from eval import tail_lift_xs
from model import auc, oof_predict


def _per_date_lift(panel: pd.DataFrame, score: np.ndarray, primary: str, date_col: str, q: float) -> np.ndarray:
    df = panel[[date_col, primary]].copy()
    df["_s"] = score
    lifts = []
    for _, g in df.groupby(date_col):
        x = g["_s"].to_numpy(float)
        y = g[primary].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 10:
            lifts.append(tail_lift_xs(x[m], y[m], q))
    return np.asarray(lifts)


def _xs_rank(panel: pd.DataFrame, score: np.ndarray, date_col: str) -> np.ndarray:
    s = pd.Series(score, index=panel.index)
    return s.groupby(panel[date_col]).rank(pct=True).to_numpy()


def target_ensemble(
    panel: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    *,
    date_col: str = "Date",
    horizon: int = 5,
    n_splits: int = 5,
    params: dict | None = None,
    q: float = 0.1,
) -> dict:
    """Train a purged-OOF model per target arm, rank-average them, and score the lot.

    panel must be time-ordered by date_col. target_cols[0] is the PRIMARY return used
    for evaluation; each arm is binarized (up-move) as its training label.
    """
    panel = panel.sort_values([date_col]).reset_index(drop=True)
    event_time = panel[date_col].to_numpy()
    X = panel[feature_cols].to_numpy(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    primary = target_cols[0]

    per_arm, rank_stack = {}, []
    for arm in target_cols:
        y = (panel[arm].to_numpy(float) > 0).astype(float)
        oof = oof_predict(X, y, event_time, horizon=horizon, n_splits=n_splits, params=params)
        lift = _per_date_lift(panel, oof, primary, date_col, q)
        per_arm[arm] = {
            "mean_lift": float(np.nanmean(lift)),
            "spread": float(np.nanstd(lift)),
            "oof_auc": auc((panel[arm].to_numpy(float) > 0).astype(float), oof),
        }
        rank_stack.append(_xs_rank(panel, oof, date_col))

    ens_score = np.nanmean(np.vstack(rank_stack), axis=0)
    ens_lift = _per_date_lift(panel, ens_score, primary, date_col, q)
    ensemble = {"mean_lift": float(np.nanmean(ens_lift)), "spread": float(np.nanstd(ens_lift))}

    best_arm = max(per_arm.values(), key=lambda d: d["mean_lift"])
    mean_arm_spread = float(np.mean([d["spread"] for d in per_arm.values()]))
    return {
        "per_arm": per_arm,
        "ensemble": ensemble,
        "lift_improvement": ensemble["mean_lift"] - best_arm["mean_lift"],
        "spread_reduction": mean_arm_spread - ensemble["spread"],
        "mean_arm_spread": mean_arm_spread,
    }
