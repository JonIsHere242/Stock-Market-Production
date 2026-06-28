from __future__ import annotations

"""Honest multiple-testing / overfitting statistics for feature screening.

When we screen ~10,000 candidate features and keep ~300, the survivors are
contaminated by selection bias: the best-looking strategy out of many trials
will look good by luck alone. This module deflates that luck. It implements the
expected maximum Sharpe under the null, the Deflated Sharpe Ratio, the
Romano-Wolf step-down family-wise-error procedure, and the Probability of
Backtest Overfitting via Combinatorially-Symmetric Cross-Validation (CSCV).
"""

import itertools
import math
from typing import Iterator

import numpy as np
from scipy.stats import norm

EULER_MASCHERONI: float = 0.5772156649015329


def expected_max_sharpe(sr_std: float, n_trials: int) -> float:
    """Expected maximum estimated Sharpe under the null (true SR = 0).

    Bailey & Lopez de Prado approximation for the expectation of the maximum of
    n_trials i.i.d. standard-normal-scaled Sharpe estimates with cross-trial
    std `sr_std`. Returns 0.0 when fewer than 2 trials (no selection effect).
    """
    n = int(n_trials)
    if n < 2:
        return 0.0
    g = EULER_MASCHERONI
    z1 = norm.ppf(1.0 - 1.0 / n)
    z2 = norm.ppf(1.0 - 1.0 / (n * math.e))
    return float(sr_std * ((1.0 - g) * z1 + g * z2))


def deflated_sharpe_ratio(
    observed_sr: float,
    sr_std_across_trials: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio: P(true SR > 0) after correcting for selection.

    SR0 is the expected maximum Sharpe under the null across n_trials. The DSR
    is the probability that the observed Sharpe exceeds that benchmark given the
    sample length and the return distribution's higher moments. Returns a
    probability in [0, 1].
    """
    sr0 = expected_max_sharpe(sr_std_across_trials, n_trials)
    denom_sq = 1.0 - skew * observed_sr + ((kurt - 1.0) / 4.0) * observed_sr ** 2
    # Guard against a non-positive variance term (pathological moments).
    if denom_sq <= 0.0 or n_obs <= 1:
        return float("nan")
    z = ((observed_sr - sr0) * math.sqrt(n_obs - 1)) / math.sqrt(denom_sq)
    return float(np.clip(norm.cdf(z), 0.0, 1.0))


def romano_wolf(
    t_obs: np.ndarray, t_boot: np.ndarray, alpha: float = 0.05
) -> np.ndarray:
    """Romano-Wolf step-down procedure controlling family-wise error rate.

    `t_obs` shape (k,): observed (two-sided -> absolute) test statistics, larger
    is more significant. `t_boot` shape (B, k): bootstrap/null draws of the same
    statistics, centered under the null. Returns a boolean (k,) array, True =
    reject (significant). Iteratively rejects features whose |t_obs| exceeds the
    (1 - alpha) quantile of the bootstrap max over the surviving set, until no
    new rejections occur.
    """
    t_obs = np.abs(np.asarray(t_obs, dtype=float))
    t_boot = np.abs(np.asarray(t_boot, dtype=float))
    k = t_obs.shape[0]
    rejected = np.zeros(k, dtype=bool)

    while True:
        active = ~rejected
        if not active.any():
            break
        # Bootstrap maximum statistic over the not-yet-rejected set.
        boot_max = t_boot[:, active].max(axis=1)
        crit = float(np.quantile(boot_max, 1.0 - alpha))
        new_rej = active & (t_obs > crit)
        if not new_rej.any():
            break
        rejected |= new_rej
    return rejected


def _iter_is_block_sets(n_splits: int, cap: int = 1000) -> Iterator[tuple[int, ...]]:
    """Deterministically yield up to `cap` choices of half the blocks as IS."""
    half = n_splits // 2
    combos = itertools.combinations(range(n_splits), half)
    if math.comb(n_splits, half) > cap:
        combos = itertools.islice(combos, cap)
    yield from combos


def pbo_cscv(perf: np.ndarray, n_splits: int = 16) -> float:
    """Probability of Backtest Overfitting via CSCV (Bailey et al.).

    `perf` shape (T, N): performance metric (returns/IC) for T time-periods x N
    configurations. T is split into n_splits contiguous blocks; for each way of
    assigning half the blocks to in-sample (IS) and the complement to
    out-of-sample (OOS), the IS-best config's OOS rank fraction w gives
    logit = ln(w/(1-w)). PBO is the fraction of splits where the IS-best config
    lands in the bottom OOS half (logit <= 0). Returns PBO in [0, 1].
    """
    perf = np.asarray(perf, dtype=float)
    t, n = perf.shape
    if n < 2 or n_splits < 2:
        return float("nan")
    n_splits = min(n_splits, t)
    if n_splits % 2 == 1:  # CSCV needs an even number of blocks to halve.
        n_splits -= 1
    if n_splits < 2:
        return float("nan")

    # Contiguous (near-equal) blocks of row indices.
    blocks = np.array_split(np.arange(t), n_splits)
    all_blocks = set(range(n_splits))

    logits: list[float] = []
    for is_blocks in _iter_is_block_sets(n_splits):
        oos_blocks = tuple(sorted(all_blocks - set(is_blocks)))
        is_rows = np.concatenate([blocks[b] for b in is_blocks])
        oos_rows = np.concatenate([blocks[b] for b in oos_blocks])

        is_mean = perf[is_rows].mean(axis=0)
        oos_mean = perf[oos_rows].mean(axis=0)

        best = int(np.argmax(is_mean))
        # OOS rank of the IS-best config: fraction of configs it beats OOS.
        rank = float((oos_mean <= oos_mean[best]).sum())  # 1..N (includes self)
        w = rank / (n + 1.0)  # in (0,1), avoids exact 0/1
        w = float(np.clip(w, 1e-6, 1.0 - 1e-6))
        logits.append(math.log(w / (1.0 - w)))

    if not logits:
        return float("nan")
    arr = np.asarray(logits)
    return float((arr <= 0.0).mean())


if __name__ == "__main__":
    rng = np.random.default_rng(7)

    def _line(name: str, ok: bool, detail: str) -> None:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    # 1) expected_max_sharpe grows with n_trials; degenerate guard returns 0.
    e_few = expected_max_sharpe(1.0, 10)
    e_many = expected_max_sharpe(1.0, 10000)
    _line("expected_max_sharpe monotone",
          0.0 == expected_max_sharpe(1.0, 1) and 0.0 < e_few < e_many,
          f"N=1->{0.0}, N=10->{e_few:.3f}, N=10000->{e_many:.3f}")

    # 2) DSR ~ 1 for an obviously good Sharpe with few trials...
    dsr_good = deflated_sharpe_ratio(observed_sr=2.0, sr_std_across_trials=0.5,
                                     n_trials=5, n_obs=1000)
    # ...and ~ 0 for a 'lucky' Sharpe selected among many trials.
    dsr_lucky = deflated_sharpe_ratio(observed_sr=0.6, sr_std_across_trials=0.5,
                                      n_trials=10000, n_obs=1000)
    _line("deflated_sharpe_ratio discriminates",
          dsr_good > 0.95 and dsr_lucky < 0.10,
          f"good={dsr_good:.4f}, lucky={dsr_lucky:.4f}")

    # 3) Romano-Wolf: a few strong signals among noise should be rejected,
    #    and pure noise should (almost) never be.
    k = 50
    b = 2000
    t_obs = rng.standard_normal(k)
    t_obs[:3] += 6.0  # three genuine signals
    t_boot = rng.standard_normal((b, k))  # null draws, centered at 0
    rej = romano_wolf(t_obs, t_boot, alpha=0.05)
    _line("romano_wolf finds signals, controls FWER",
          rej[:3].all() and rej[3:].sum() <= 2,
          f"true rejected={int(rej[:3].sum())}/3, false={int(rej[3:].sum())}/{k - 3}")

    # 4a) Pure noise (all configs same true mean): IS-best overfits IS noise and
    #     has no real OOS edge, so PBO is high-ish (not low) -- NOT a safe config.
    perf_rand = rng.standard_normal((500, 30))
    pbo_rand = pbo_cscv(perf_rand, n_splits=12)
    # 4b) A real spread of edges: the IS-best config is genuinely best OOS too,
    #     so PBO collapses toward 0 -- selection is trustworthy.
    perf_edge = rng.standard_normal((500, 30)) + np.linspace(0.0, 1.0, 30)
    pbo_edge = pbo_cscv(perf_edge, n_splits=12)
    _line("pbo_cscv: not-low on pure noise, ~0 on real spread",
          pbo_rand > 0.25 and pbo_edge < 0.10,
          f"pure_noise={pbo_rand:.3f}, real_spread={pbo_edge:.3f}")

    print("smoke test complete")
