from __future__ import annotations

"""Purged + embargoed cross-validation for forward-return labels (Lopez de Prado).

A sample at time t carries a label that spans [event_time, event_time + horizon],
so its information leaks `horizon` steps into the future. Naive K-fold leaks because
train labels whose window overlaps the test span share future data with the test set.
PURGE drops any train sample whose label window overlaps the test fold's time span.
EMBARGO additionally drops train samples whose event_time falls just AFTER the test
fold (serial correlation bleed). CPCV tests every C(n_groups, k) group combination,
yielding many overlapping backtest PATHS instead of a single train/test partition.
"""

import itertools
import math
from typing import List, Tuple

import numpy as np


def embargo_times(
    event_time: np.ndarray,
    test_end_time: float,
    embargo_frac: float,
) -> Tuple[float, float]:
    """Embargo window (start, end) applied AFTER a test fold's end time.

    Width is `embargo_frac * (max_time - min_time)`. The window is the half-open
    span (test_end_time, test_end_time + width] in event-time units; train samples
    landing inside it are dropped to kill serial-correlation bleed past the fold.
    """
    et = np.asarray(event_time)
    span = float(et.max() - et.min()) if et.size else 0.0
    width = max(0.0, float(embargo_frac)) * span
    start = float(test_end_time)
    end = float(test_end_time) + width
    return (start, end)


def purge_overlap(
    train_event_time: np.ndarray,
    test_span: Tuple[float, float],
    horizon: int,
) -> np.ndarray:
    """Boolean KEEP-mask for train samples (True = no label-window overlap).

    Each train sample's label spans [t, t + horizon]; the test span is
    [test_start, test_end] (the caller passes test_end already padded by horizon).
    Two intervals overlap iff t <= test_end and t + horizon >= test_start.
    Returns True where they do NOT overlap, i.e. the sample is safe to keep.
    """
    t = np.asarray(train_event_time, dtype=np.float64)
    test_start, test_end = float(test_span[0]), float(test_span[1])
    h = float(horizon)
    overlaps = (t <= test_end) & ((t + h) >= test_start)
    return ~overlaps


def _apply_purge_embargo(
    event_time: np.ndarray,
    candidate_train_idx: np.ndarray,
    test_start_time: float,
    test_end_time: float,
    horizon: int,
    embargo_frac: float,
) -> np.ndarray:
    """Filter a candidate train index by purge (label overlap) then embargo."""
    cand = np.asarray(candidate_train_idx)
    if cand.size == 0:
        return cand
    train_et = event_time[cand]
    # Test span padded by horizon so a label ending inside the fold is caught.
    test_span = (float(test_start_time), float(test_end_time) + float(horizon))
    keep = purge_overlap(train_et, test_span, horizon)
    emb_start, emb_end = embargo_times(event_time, test_end_time, embargo_frac)
    if emb_end > emb_start:
        in_embargo = (train_et > emb_start) & (train_et <= emb_end)
        keep &= ~in_embargo
    return cand[keep]


def purged_kfold(
    event_time: np.ndarray,
    n_splits: int = 5,
    horizon: int = 1,
    embargo_frac: float = 0.01,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """K contiguous test folds over time-ordered samples, with purge + embargo.

    Samples are assumed pre-sorted by time (index 0..n-1). Each fold's test block
    is a contiguous slice; train is everything else minus samples whose label
    window overlaps the fold (purge) or that fall in the post-fold embargo window.
    """
    et = np.asarray(event_time)
    n = et.shape[0]
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if n < n_splits:
        raise ValueError("fewer samples than splits")

    all_idx = np.arange(n)
    bounds = np.linspace(0, n, n_splits + 1).astype(int)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for f in range(n_splits):
        lo, hi = bounds[f], bounds[f + 1]
        test_idx = all_idx[lo:hi]
        if test_idx.size == 0:
            continue
        test_start_time = float(et[test_idx].min())
        test_end_time = float(et[test_idx].max())
        candidate = np.concatenate([all_idx[:lo], all_idx[hi:]])
        train_idx = _apply_purge_embargo(
            et, candidate, test_start_time, test_end_time, horizon, embargo_frac
        )
        splits.append((train_idx, test_idx))
    return splits


def cpcv(
    event_time: np.ndarray,
    n_groups: int = 6,
    k_test_groups: int = 2,
    horizon: int = 1,
    embargo_frac: float = 0.01,
) -> dict:
    """Combinatorial Purged CV over C(n_groups, k_test_groups) group combinations.

    Split samples into `n_groups` contiguous groups. For every combination of
    `k_test_groups` groups used as test (rest as train), purge + embargo the train
    side against each test group's span. n_paths = C(n,k) * k / n is the number of
    distinct backtest paths reconstructable from the per-group out-of-sample slices.
    """
    et = np.asarray(event_time)
    n = et.shape[0]
    if n_groups < 2:
        raise ValueError("n_groups must be >= 2")
    if not (1 <= k_test_groups < n_groups):
        raise ValueError("k_test_groups must satisfy 1 <= k < n_groups")
    if n < n_groups:
        raise ValueError("fewer samples than groups")

    all_idx = np.arange(n)
    bounds = np.linspace(0, n, n_groups + 1).astype(int)
    groups = [all_idx[bounds[g]:bounds[g + 1]] for g in range(n_groups)]

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for combo in itertools.combinations(range(n_groups), k_test_groups):
        test_idx = np.concatenate([groups[g] for g in combo])
        test_idx = np.sort(test_idx)
        train_groups = [g for g in range(n_groups) if g not in combo]
        candidate = (
            np.sort(np.concatenate([groups[g] for g in train_groups]))
            if train_groups
            else np.array([], dtype=int)
        )
        # Purge/embargo train against EACH test group span (groups are disjoint blocks).
        for g in combo:
            g_et = et[groups[g]]
            g_start, g_end = float(g_et.min()), float(g_et.max())
            candidate = _apply_purge_embargo(
                et, candidate, g_start, g_end, horizon, embargo_frac
            )
        splits.append((candidate, test_idx))

    n_splits = math.comb(n_groups, k_test_groups)
    n_paths = (n_splits * k_test_groups) // n_groups
    return {"splits": splits, "n_splits": n_splits, "n_paths": n_paths}


if __name__ == "__main__":
    # --- synthetic time-ordered index: 1000 daily samples, label horizon = 5 ---
    n = 1000
    horizon = 5
    event_time = np.arange(n)

    # ---- purged_kfold ----
    n_splits = 5
    folds = purged_kfold(event_time, n_splits=n_splits, horizon=horizon, embargo_frac=0.01)
    assert len(folds) == n_splits, f"expected {n_splits} folds, got {len(folds)}"

    total_purged = 0
    naive_unpurged_overlaps = 0
    for train_idx, test_idx in folds:
        # Partition sanity: train and test are disjoint.
        assert np.intersect1d(train_idx, test_idx).size == 0, "train/test overlap"
        test_start = float(event_time[test_idx].min())
        test_end = float(event_time[test_idx].max())

        # Naive (un-purged) train = everything not in test.
        naive_train = np.setdiff1d(np.arange(n), test_idx)
        purged_count = naive_train.size - train_idx.size
        total_purged += purged_count

        # Count how many naive train samples WOULD have leaked (label overlaps test span).
        naive_et = event_time[naive_train]
        span = (test_start, test_end + horizon)
        leaks = (~purge_overlap(naive_et, span, horizon)).sum()
        naive_unpurged_overlaps += int(leaks)

        # Verify purge worked: NO surviving train label window overlaps the test span.
        train_et = event_time[train_idx]
        keep_mask = purge_overlap(train_et, span, horizon)
        assert keep_mask.all(), "purge failed: a train label window still overlaps test span"

    avg_purged = total_purged / len(folds)
    assert avg_purged > 0, "expected > 0 purged per fold with horizon=5"
    print("PASS purged_kfold: %d folds, disjoint train/test, no surviving overlaps" % len(folds))
    print("     avg samples purged+embargoed per fold = %.1f" % avg_purged)
    print("     naive (un-purged) leaking train samples that purge removed = %d total"
          % naive_unpurged_overlaps)

    # ---- cpcv ----
    res = cpcv(event_time, n_groups=6, k_test_groups=2, horizon=horizon, embargo_frac=0.01)
    assert res["n_splits"] == math.comb(6, 2) == 15, "expected C(6,2)=15 splits"
    assert res["n_paths"] == 5, "expected n_paths=5"
    n_groups = 6
    g_bounds = np.linspace(0, n, n_groups + 1).astype(int)
    groups = [np.arange(n)[g_bounds[g]:g_bounds[g + 1]] for g in range(n_groups)]
    cpcv_purged = 0
    for train_idx, test_idx in res["splits"]:
        assert np.intersect1d(train_idx, test_idx).size == 0, "cpcv train/test overlap"
        test_set = set(test_idx.tolist())
        train_et = event_time[train_idx]
        # Purge must hold against EACH contiguous test group span (test groups may be
        # non-adjacent, so the union span would wrongly cover the train gap between them).
        for g in groups:
            if not test_set.issuperset(g.tolist()):
                continue  # not a test group of this split
            g_et = event_time[g]
            span = (float(g_et.min()), float(g_et.max()) + horizon)
            assert purge_overlap(train_et, span, horizon).all(), \
                "cpcv purge failed against a test group span"
        # Tally purge effect vs naive (all non-test) train.
        naive_train = np.setdiff1d(np.arange(n), test_idx)
        cpcv_purged += naive_train.size - train_idx.size
    print("PASS cpcv: n_splits=%d (C(6,2)=15), n_paths=%d (expected 5)"
          % (res["n_splits"], res["n_paths"]))
    print("     total samples purged+embargoed across 15 cpcv splits = %d" % cpcv_purged)

    # ---- embargo_times sanity ----
    es, ee = embargo_times(event_time, test_end_time=500.0, embargo_frac=0.01)
    assert abs((ee - es) - 0.01 * (n - 1)) < 1e-9, "embargo width mismatch"
    print("PASS embargo_times: window width = %.1f time units after fold end" % (ee - es))

    print("ALL VALIDATION CHECKS PASSED")
