"""Model-in-the-loop — purged out-of-fold XGBoost.

Closes the loop from feature -> actual prediction. Screen statistics are proxies;
this trains the real model the way it must be validated for financial ML: with
PURGED + EMBARGOED folds (see validation.py) so overlapping forward-return labels
cannot leak train into test. Produces honest out-of-fold predictions and purged
gain-importance. The OOF predictions are the substrate the target ensemble stacks.
"""
from __future__ import annotations

import numpy as np

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

from validation import purged_kfold

_DEFAULT = dict(
    tree_method="hist", max_depth=4, eta=0.1, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=5, verbosity=0,
)


def _objective(y: np.ndarray) -> str:
    u = np.unique(y[np.isfinite(y)])
    return "binary:logistic" if set(np.round(u, 6).tolist()) <= {0.0, 1.0} else "reg:squarederror"


def auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUC, no sklearn. y_true treated as positive iff > 0.5."""
    y = np.asarray(y_true, float)
    s = np.asarray(score, float)
    m = np.isfinite(y) & np.isfinite(s)
    y, s = y[m], s[m]
    pos = y > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = np.empty(len(s))
    ranks[np.argsort(s, kind="mergesort")] = np.arange(1, len(s) + 1)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def oof_predict(
    X: np.ndarray,
    y: np.ndarray,
    event_time: np.ndarray,
    *,
    horizon: int = 1,
    n_splits: int = 5,
    embargo_frac: float = 0.01,
    params: dict | None = None,
    num_boost_round: int = 80,
) -> np.ndarray:
    """Purged out-of-fold predictions aligned to X's rows (NaN if never tested).

    NOTE: rows must be in time order (sort by date before calling). Set horizon=0,
    embargo_frac=0 to get the NAIVE (leaky) k-fold for comparison.
    """
    if not XGB_AVAILABLE:
        raise RuntimeError("xgboost is not installed")
    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.asarray(y, float)
    p = dict(_DEFAULT, objective=_objective(y))
    if params:
        p.update(params)
    oof = np.full(len(y), np.nan)
    for tr, te in purged_kfold(event_time, n_splits=n_splits, horizon=horizon, embargo_frac=embargo_frac):
        mtr = tr[np.isfinite(y[tr])]
        if len(mtr) < 20 or len(te) == 0:
            continue
        dtr = xgb.DMatrix(X[mtr], label=y[mtr])
        bst = xgb.train(p, dtr, num_boost_round=num_boost_round)
        oof[te] = bst.predict(xgb.DMatrix(X[te]))
    return oof


def purged_importance(
    X: np.ndarray,
    y: np.ndarray,
    event_time: np.ndarray,
    feature_names: list[str],
    *,
    horizon: int = 1,
    n_splits: int = 5,
    embargo_frac: float = 0.01,
    params: dict | None = None,
    num_boost_round: int = 80,
) -> dict[str, float]:
    """Mean gain importance over purged folds (leakage-free feature ranking)."""
    if not XGB_AVAILABLE:
        raise RuntimeError("xgboost is not installed")
    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.asarray(y, float)
    p = dict(_DEFAULT, objective=_objective(y))
    if params:
        p.update(params)
    agg: dict[str, float] = {n: 0.0 for n in feature_names}
    folds = 0
    for tr, te in purged_kfold(event_time, n_splits=n_splits, horizon=horizon, embargo_frac=embargo_frac):
        mtr = tr[np.isfinite(y[tr])]
        if len(mtr) < 20:
            continue
        dtr = xgb.DMatrix(X[mtr], label=y[mtr], feature_names=feature_names)
        bst = xgb.train(p, dtr, num_boost_round=num_boost_round)
        for k, v in bst.get_score(importance_type="gain").items():
            agg[k] = agg.get(k, 0.0) + float(v)
        folds += 1
    if folds:
        agg = {k: v / folds for k, v in agg.items()}
    return dict(sorted(agg.items(), key=lambda kv: -kv[1]))
