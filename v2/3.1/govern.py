"""Compute-cost governance for the feature-engineering framework.

As the library grows toward millions of features, raw breadth stops being free:
slow features quietly tax every nightly build. This module sits on top of the
framework's per-block wall-clock timings and (1) detects p95 runtime regressions
against a saved baseline, (2) ranks features by value-per-CPU-second so the cost
is amortized against the edge, and (3) raises the promotion bar for expensive
features. Pure numpy/pandas + stdlib; no network, no sklearn.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

_EPS_MS = 1e-3  # tiny floor so missing/zero costs never divide-by-zero


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))


class CostBaseline:
    """Persisted per-block runtime statistics used to flag regressions."""

    def __init__(self, stats: dict | None = None) -> None:
        # block_name -> {"p50_ms", "p95_ms", "mean_ms", "n"}
        self.stats: dict[str, dict[str, float]] = dict(stats or {})

    @classmethod
    def load(cls, path) -> "CostBaseline":
        p = Path(path)
        if not p.exists():
            return cls({})
        with p.open("r", encoding="utf-8") as fh:
            return cls(json.load(fh))

    def save(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            json.dump(self.stats, fh, indent=2, sort_keys=True)

    def update(self, timings_ms: dict[str, list[float]]) -> None:
        """Recompute and store p50/p95/mean/n per block (merge/replace)."""
        for block, calls in timings_ms.items():
            calls = [float(c) for c in calls]
            if not calls:
                continue
            self.stats[block] = {
                "p50_ms": _percentile(calls, 50.0),
                "p95_ms": _percentile(calls, 95.0),
                "mean_ms": float(np.mean(calls)),
                "n": int(len(calls)),
            }

    def regressions(
        self, current_ms: dict[str, list[float]], rel_threshold: float = 0.2
    ) -> list[dict]:
        """Flag blocks whose current p95 exceeds the baseline by > rel_threshold.

        Blocks absent from the baseline are reported with severity "new".
        Returns a list sorted worst-first (largest relative change first).
        """
        out: list[dict] = []
        for block, calls in current_ms.items():
            if not calls:
                continue
            cur_p95 = _percentile([float(c) for c in calls], 95.0)
            base = self.stats.get(block)
            if base is None:
                out.append(
                    {
                        "block": block,
                        "baseline_p95_ms": None,
                        "current_p95_ms": cur_p95,
                        "rel_change": float("inf"),
                        "severity": "new",
                    }
                )
                continue
            base_p95 = float(base.get("p95_ms", 0.0))
            if base_p95 <= 0.0:
                continue
            rel = (cur_p95 - base_p95) / base_p95
            if rel <= rel_threshold:
                continue
            severity = "crit" if rel > 1.0 else "warn"  # > 2x baseline == crit
            out.append(
                {
                    "block": block,
                    "baseline_p95_ms": base_p95,
                    "current_p95_ms": cur_p95,
                    "rel_change": rel,
                    "severity": severity,
                }
            )

        def _key(d: dict) -> float:
            rc = d["rel_change"]
            return float("inf") if rc == float("inf") else float(rc)

        out.sort(key=_key, reverse=True)
        return out


def value_per_cpu_second(
    screen: pd.DataFrame,
    cost_ms: dict[str, float],
    feature_col: str = "feature",
    lift_col: str = "lift",
) -> pd.DataFrame:
    """Rank features by edge earned per CPU-second of compute.

    `screen` may have multiple rows per feature (one per target arm); we take the
    BEST lift. lift_excess = best_lift - 1.0; value = lift_excess / cost_seconds.
    """
    if screen.empty:
        return pd.DataFrame(
            columns=["feature", "best_lift", "cost_ms", "value_per_cpu_s"]
        )
    best = (
        screen.groupby(feature_col)[lift_col]
        .max()
        .reset_index()
        .rename(columns={feature_col: "feature", lift_col: "best_lift"})
    )
    best["cost_ms"] = best["feature"].map(lambda f: float(cost_ms.get(f, _EPS_MS)))
    cost_s = (best["cost_ms"] / 1000.0).clip(lower=_EPS_MS / 1000.0)
    best["value_per_cpu_s"] = (best["best_lift"] - 1.0) / cost_s
    best = best.sort_values("value_per_cpu_s", ascending=False).reset_index(drop=True)
    return best[["feature", "best_lift", "cost_ms", "value_per_cpu_s"]]


def cost_weighted_threshold(
    base_threshold: float,
    cost_ms: float,
    ref_cost_ms: float = 1.0,
    elasticity: float = 0.5,
) -> float:
    """Raise the promotion bar for expensive features.

    A feature at (or below) ref cost faces base_threshold; cost above ref inflates
    the bar logarithmically, so a 100x-cost feature must clear a meaningfully
    higher one without the bar exploding.
    """
    excess = max(cost_ms - ref_cost_ms, 0.0)
    return base_threshold * (1.0 + elasticity * math.log1p(excess / ref_cost_ms))


def governance_report(
    current_ms: dict[str, list[float]],
    baseline: CostBaseline,
    screen: pd.DataFrame,
    cost_means: dict[str, float],
) -> dict:
    """Bundle regressions, top value-per-CPU-second features, and total wall time."""
    total_wall_ms = float(
        sum(float(s.get("mean_ms", 0.0)) for s in baseline.stats.values())
    )
    return {
        "regressions": baseline.regressions(current_ms),
        "top_value": value_per_cpu_second(screen, cost_means)
        .head(10)
        .to_dict("records"),
        "total_wall_ms": total_wall_ms,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(7)

    # Baseline timings: three well-behaved blocks.
    baseline_timings = {
        "returns": list(rng.normal(5.0, 0.3, 64)),
        "momentum": list(rng.normal(20.0, 1.0, 64)),
        "spectral": list(rng.normal(50.0, 2.0, 64)),
    }
    base = CostBaseline()
    base.update(baseline_timings)

    tmp = Path(
        "C:/Users/Masam/AppData/Local/Temp/claude/"
        "c--Users-Masam-Desktop-Stock-Market/_govern_baseline.json"
    )
    base.save(tmp)
    reloaded = CostBaseline.load(tmp)
    assert reloaded.stats.keys() == base.stats.keys()
    print("PASS save/load round-trip: %d blocks" % len(reloaded.stats))

    # Current run: momentum mildly slower (warn), spectral 3x slower (crit),
    # plus a brand-new block.
    current = {
        "returns": list(rng.normal(5.1, 0.3, 32)),  # within threshold
        "momentum": list(rng.normal(26.0, 1.0, 32)),  # ~+30% -> warn
        "spectral": list(rng.normal(160.0, 5.0, 32)),  # ~3x -> crit
        "fundamentals": list(rng.normal(8.0, 0.5, 32)),  # new
    }
    regs = reloaded.regressions(current, rel_threshold=0.2)
    sev = {r["block"]: r["severity"] for r in regs}
    assert sev.get("momentum") == "warn", sev
    assert sev.get("spectral") == "crit", sev
    assert sev.get("fundamentals") == "new", sev
    assert "returns" not in sev, sev
    print(
        "PASS regressions: momentum=%s spectral=%s fundamentals=%s (returns clean)"
        % (sev["momentum"], sev["spectral"], sev["fundamentals"])
    )

    # Value-per-CPU-second: cheap-high-lift should beat expensive-high-lift.
    screen = pd.DataFrame(
        {
            "feature": ["cheap_gold", "cheap_gold", "pricey_gold", "meh"],
            "arm": ["topq", "vol", "topq", "topq"],
            "lift": [1.35, 1.20, 1.40, 1.05],
        }
    )
    costs = {"cheap_gold": 2.0, "pricey_gold": 200.0, "meh": 1.0}
    ranked = value_per_cpu_second(screen, costs)
    assert list(ranked["feature"])[0] == "cheap_gold", ranked
    assert (
        ranked.set_index("feature").loc["cheap_gold", "value_per_cpu_s"]
        > ranked.set_index("feature").loc["pricey_gold", "value_per_cpu_s"]
    )
    print(
        "PASS value_per_cpu_s ranks cheap_gold(%.1f) > pricey_gold(%.1f)"
        % (
            ranked.set_index("feature").loc["cheap_gold", "value_per_cpu_s"],
            ranked.set_index("feature").loc["pricey_gold", "value_per_cpu_s"],
        )
    )

    # Cost-weighted threshold rises with cost.
    t_ref = cost_weighted_threshold(1.0, cost_ms=1.0)
    t_10 = cost_weighted_threshold(1.0, cost_ms=10.0)
    t_100 = cost_weighted_threshold(1.0, cost_ms=100.0)
    assert t_ref < t_10 < t_100, (t_ref, t_10, t_100)
    print(
        "PASS cost_weighted_threshold rises: ref=%.3f 10x=%.3f 100x=%.3f"
        % (t_ref, t_10, t_100)
    )

    report = governance_report(current, reloaded, screen, costs)
    print(
        "PASS governance_report: %d regressions, %d top-value rows, total_wall=%.1f ms"
        % (
            len(report["regressions"]),
            len(report["top_value"]),
            report["total_wall_ms"],
        )
    )
    print("ALL PASS")
