"""
__feature_lab.py  --  Systematic feature-improvement test battery.

Auto-skipped by the framework (leading __ keeps it out of block discovery),
exactly like __diagnostics.py.

WHAT THIS IS
------------
__diagnostics.py tells you WHICH produced columns are GREAT / ok / meh.
This tool tells you HOW TO RESCUE the meh ones: it throws a battery of simple,
*reordering* transforms (rolling z-score, cross-sectional rank, gating by a
condition, interactions, ...) at a feature and reports which transform actually
improves its predictive edge -- so you can promote the winner into a real
FeatureTemplates block (and even codegen it with --emit).

TWO FACTS THIS TOOL IS BUILT AROUND
-----------------------------------
1. MONOTONIC TRANSFORMS ARE NO-OPS FOR THE MODEL.
   4__Predictor.py is XGBoost (trees). A strictly monotonic transform of a
   feature (log, sqrt, sigmoid, any fixed power) produces the IDENTICAL splits,
   hence the IDENTICAL model and IDENTICAL IC. This is your own observation:
   "unless something reorders the outputs its IC is the same."  So this battery
   ONLY contains transforms that REORDER values across rows. Pure-monotonic
   transforms are deliberately excluded (they would waste your time).

2. TAIL DISCRIMINATION > GLOBAL IC.
   The strategy trades only the top ~1%/day (top 4 of a 12-name pool). A feature
   is valuable if, on a given day, its TOP slice concentrates next-day winners --
   not if it has a nice global rank-IC. So every transform is scored on BOTH:
     - IC      : pooled per-ticker Spearman vs next-day log-return
                 (same definition as __diagnostics.py, for direct comparison)
     - lift    : cross-sectional TAIL lift -- among the names in the feature's
                 top (or bottom) decile each day, how many times more likely are
                 they to be next-day top-decile winners vs the base rate.
                 lift = 1.0 means no edge; lift = 2.0 means 2x the base rate.
   The "best" transform is chosen by tail lift; IC is shown alongside.

USAGE
-----
  # Battery on one feature (fast -- full transform set)
  python FeatureTemplates/__feature_lab.py --feature vvg_max_betweenness

  # Battery on every column a block produces
  python FeatureTemplates/__feature_lab.py --block vvg

  # Sweep: find the meh features with the biggest rescue opportunity
  python FeatureTemplates/__feature_lab.py --all_meh --max_features 60

  # Cross-feature trick: try feature x other and feature / other
  python FeatureTemplates/__feature_lab.py --feature vvg_max_betweenness \
         --interact_with atr_percentile_rank

  # Emit a ready-to-paste FeatureTemplates block for a chosen transform
  python FeatureTemplates/__feature_lab.py --feature vvg_max_betweenness \
         --emit ts_z_60
  #   (cross-feature: --emit "xf_mul:atr_percentile_rank")

  # List the transform registry (and which are model-meaningful)
  python FeatureTemplates/__feature_lab.py --list_transforms

KNOBS
-----
  --n            tickers to load for the cross-section (default 200)
  --seed         sampling seed (default 42)
  --winner_pct   next-day return percentile that counts as a "winner" (def 0.10)
  --q            feature percentile slice used for tail lift (default 0.10)
  --min_names    skip days whose cross-section is smaller than this (default 15)
  --full         use the FULL transform set in --all_meh sweep (slower)
  --max_features cap features scanned in a sweep (default 60)

EXTENDING THE BATTERY
---------------------
Add an entry to the TRANSFORMS dict below. Each entry is:
    "name": dict(fn=<callable(panel, col)->Series>, desc=str,
                 kind="ts"|"xs"|"interact", core=bool, emittable=bool)
`fn` must be POINT-IN-TIME (no lookahead). Per-ticker ("ts") and interaction
transforms are emittable into a real block; cross-sectional ("xs") ones belong
in the framework's cross-sectional stage and are flagged emittable=False.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap: import the framework (same pattern __diagnostics.py uses)
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

_spec = importlib.util.spec_from_file_location("framework", ROOT / "3__FeatureFramework.py")
_fw = importlib.util.module_from_spec(_spec)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _spec.loader.exec_module(_fw)

discover_blocks = _fw.discover_blocks
PROCESSED_DIR = ROOT / "Data" / "ProcessedData_v2"

# ---------------------------------------------------------------------------
# Color helpers (compact subset mirrored from __diagnostics.py / Util.py)
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    os.system("")

RESET = "\033[0m"
BOLD = "\033[1m"


def _rgb(r: int, g: int, b: int) -> str:
    return f"\033[38;2;{r};{g};{b}m"


GREEN_BRIGHT = _rgb(0, 235, 0)
GREEN = _rgb(0, 200, 0)
YELLOW = _rgb(220, 220, 0)
ORANGE = _rgb(255, 180, 0)
RED = _rgb(220, 0, 0)
DIM = _rgb(150, 150, 150)


def _c(text: str, code: str) -> str:
    return f"{code}{text}{RESET}"


def _lift_color(lift: float) -> str:
    """Color a tail-lift value: >=1.5 bright, >=1.2 green, >=1.05 yellow, else dim."""
    if lift >= 1.5:
        return GREEN_BRIGHT
    if lift >= 1.2:
        return GREEN
    if lift >= 1.05:
        return YELLOW
    return DIM


def _ic_color(ic: float) -> str:
    a = abs(ic)
    if a >= 0.05:
        return GREEN_BRIGHT
    if a >= 0.01:
        return YELLOW
    return DIM


W = 92
SEP = "=" * W


def _section(title: str) -> None:
    print()
    print(_c(SEP, DIM))
    print(BOLD + f"  {title}" + RESET)
    print(_c(SEP, DIM))


def _rule() -> None:
    print("  " + _c("-" * (W - 4), DIM))


# ===========================================================================
# SECTION 1 -- Panel loading + context signals
# ===========================================================================

def _produced_columns() -> tuple[dict[str, str], list[str]]:
    """Return (col -> owning block) and the ordered list of produced columns."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        blocks = discover_blocks()
    block_for_col: dict[str, str] = {}
    for bname, info in blocks.items():
        for col in info["meta"].get("produces", []):
            block_for_col[col] = bname
    return block_for_col, list(block_for_col.keys())


def load_panel(n: int, seed: int, feature_cols: list[str] | None) -> pd.DataFrame:
    """
    Load a sample of n tickers from ProcessedData_v2 into one long panel.

    Only the columns we actually need are read (Date, Ticker, OHLCV + the
    requested feature columns) to keep memory sane for large sweeps.
    """
    paths = sorted(PROCESSED_DIR.glob("*.parquet"))
    if not paths:
        sys.exit(f"No parquet files in {PROCESSED_DIR} -- run 3__FeatureFramework.py --all first.")

    rng = random.Random(seed)
    sample = rng.sample(paths, min(n, len(paths)))

    base = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
    want = base + (feature_cols or [])

    frames: list[pd.DataFrame] = []
    for p in sample:
        try:
            df = pd.read_parquet(p, engine="pyarrow")
        except Exception:
            continue
        if feature_cols is not None:
            df = df[[c for c in want if c in df.columns]]
        if "Ticker" not in df.columns:
            df["Ticker"] = p.stem
        frames.append(df)

    if not frames:
        sys.exit("Failed to load any tickers.")

    panel = pd.concat(frames, ignore_index=True)
    panel["Date"] = pd.to_datetime(panel["Date"])
    panel = panel.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return panel


def add_context(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add point-in-time context columns used as the target, as gates, and as
    interaction partners. All computed per-ticker (no cross-ticker leakage),
    using only trailing data.

    Added columns (all prefixed _ so they never collide with features):
      _logc       log close
      _fwd_ret    next-day log return  (THE target; uses Close[t+1] -> only as label)
      _fwd_rank   cross-sectional pct-rank of _fwd_ret within each Date
      _vol_z      within-ticker z-score of 20d realized vol  (regime interactor)
      _trend      Close / 50d MA - 1                          (trend interactor)
      _dvol_z     within-ticker z-score of dollar volume      (liquidity interactor)
      _gate_hivol 1.0 when 20d vol >= its trailing 120d median, else 0.0
      _gate_up    1.0 when Close > 50d MA, else 0.0
    """
    panel["_logc"] = np.log(panel["Close"].clip(lower=1e-10))

    # target: next-day log return, strictly within ticker
    panel["_fwd_ret"] = panel.groupby("Ticker")["_logc"].shift(-1) - panel["_logc"]

    # 20d realized vol (computed once, reused for the regime gate + interactor)
    panel["_ret"] = panel.groupby("Ticker")["Close"].pct_change()
    vol20 = panel.groupby("Ticker")["_ret"].transform(lambda s: s.rolling(20, min_periods=10).std())
    panel["_vol_z"] = _within_z(panel, vol20, 120)

    ma50 = panel.groupby("Ticker")["Close"].transform(lambda s: s.rolling(50, min_periods=20).mean())
    panel["_trend"] = panel["Close"] / ma50 - 1.0
    panel["_gate_up"] = (panel["Close"] > ma50).astype(float)

    dvol = panel["Close"] * panel["Volume"]
    panel["_dvol_z"] = _within_z(panel, dvol, 60)

    # high-vol regime gate: today's vol vs its own trailing 120d median
    vmed = vol20.groupby(panel["Ticker"]).transform(
        lambda s: s.rolling(120, min_periods=30).median()
    )
    panel["_gate_hivol"] = (vol20 >= vmed).astype(float)

    # cross-sectional winner rank
    panel["_fwd_rank"] = panel.groupby("Date")["_fwd_ret"].rank(pct=True)

    # valid-day mask: enough names in the cross-section to rank meaningfully
    day_counts = panel.groupby("Date")["_fwd_ret"].transform("count")
    panel["_day_n"] = day_counts
    return panel


def _within_z(panel: pd.DataFrame, series: pd.Series, win: int) -> pd.Series:
    """Within-ticker rolling z-score of an arbitrary series aligned to panel."""
    tmp = series.copy()
    grp = panel["Ticker"]
    mean = tmp.groupby(grp).transform(lambda s: s.rolling(win, min_periods=max(10, win // 4)).mean())
    std = tmp.groupby(grp).transform(lambda s: s.rolling(win, min_periods=max(10, win // 4)).std())
    return (tmp - mean) / std.replace(0, np.nan)


# ===========================================================================
# SECTION 2 -- Transform registry  (ONLY reordering transforms live here)
# ===========================================================================

def _ts_roll_z(panel, col, w):
    return _within_z(panel, panel[col], w)


def _ts_rank(panel, col, w):
    grp = panel["Ticker"]
    return panel[col].groupby(grp).transform(
        lambda s: s.rolling(w, min_periods=max(10, w // 4)).rank(pct=True)
    )


def _ema(panel, col, w):
    grp = panel["Ticker"]
    return panel[col].groupby(grp).transform(lambda s: s.ewm(span=w, min_periods=max(5, w // 2)).mean())


def _diff(panel, col, w):
    grp = panel["Ticker"]
    return panel[col].groupby(grp).transform(lambda s: s.diff(w))


def _roc(panel, col, w):
    grp = panel["Ticker"]
    return panel[col].groupby(grp).transform(
        lambda s: s.diff(w) / s.shift(w).abs().replace(0, np.nan)
    )


def _xs_rank(panel, col):
    return panel.groupby("Date")[col].rank(pct=True)


def _xs_z(panel, col):
    grp = panel["Date"]
    mean = panel[col].groupby(grp).transform("mean")
    std = panel[col].groupby(grp).transform("std")
    return (panel[col] - mean) / std.replace(0, np.nan)


def _interact(panel, col, ctx):
    return panel[col] * panel[ctx]


def _gate(panel, col, gate):
    """Conditional feature: keep value where gate is on, else NaN.

    NaN (not 0) is the right "off" value: it doesn't conflate 'gate off' with
    'feature == 0' for signed features, avoids rank-ties, and is exactly how
    XGBoost consumes a conditional feature (NaN -> learned default direction).
    """
    return panel[col].where(panel[gate] > 0)


def _abs_tsz(panel, col, w):
    """FOLDING / extremeness: |within-ticker z|. NON-monotone -> a tree cannot
    recover this from the raw feature with one split. Captures 'unusual in
    EITHER direction' -- the salvage trick a monotone transform can't give you."""
    return _within_z(panel, panel[col], w).abs()


def _gate_extreme(panel, col, w, thr):
    """Self-conditioning ('threshold it', done right): keep the feature only when
    it is itself > thr stdevs from its own recent mean, else NaN. This is what a
    manual 'threshold the feature' actually wants -- a conditional, not a clip."""
    z = _within_z(panel, panel[col], w)
    return panel[col].where(z.abs() > thr)


def _diff2(panel, col, w):
    """Acceleration: change of the change (2nd difference). Reorders."""
    grp = panel["Ticker"]
    return panel[col].groupby(grp).transform(lambda s: s.diff(w).diff(w))


def _xfeat(panel, col, partner, op):
    """Cross-feature trick: feature * partner ('mul') or feature / partner ('div')."""
    if op == "mul":
        return panel[col] * panel[partner]
    return panel[col] / panel[partner].replace(0, np.nan)


def _xs_neutralize(panel, col, factor):
    """WorldQuant `vector_neut`: per-day residual of feature regressed on `factor`.

    Isolates the part of the feature ORTHOGONAL to a risk factor (e.g. trend) --
    i.e. the edge that ISN'T just that factor. This is the direct antidote when
    the battery shows a transform's lift is really the context's own edge.
    NON-monotone, cross-sectional -> reorders, real new info for a tree.
    """
    f = panel[col].to_numpy(dtype=float)
    x = panel[factor].to_numpy(dtype=float)
    out = np.full(len(panel), np.nan)
    codes, _ = pd.factorize(panel["Date"].to_numpy())
    for g in np.unique(codes):
        idx = np.where(codes == g)[0]
        ff, xx = f[idx], x[idx]
        m = np.isfinite(ff) & np.isfinite(xx)
        if m.sum() < 10:
            continue
        xc = xx[m] - xx[m].mean()
        denom = float((xc * xc).sum())
        beta = float((xc * (ff[m] - ff[m].mean())).sum()) / denom if denom > 0 else 0.0
        out[idx] = ff - (ff[m].mean() + beta * (xx - xx[m].mean()))
    return pd.Series(out, index=panel.index)


def _grp_rank(panel, col, group_factor, n_buckets=5):
    """WorldQuant `group_rank`: rank the feature WITHIN its peer bucket each day.

    Buckets are formed by cross-sectional rank of `group_factor` (e.g. vol). Soft
    neutralization -- a high rank now means 'high vs peers of similar vol', which
    strips the bucket's own level out of the signal. Cross-sectional, reorders.
    """
    grank = panel.groupby("Date")[group_factor].rank(pct=True)
    bucket = np.clip((grank.to_numpy() * n_buckets), 0, n_buckets - 1)
    bucket = np.where(np.isfinite(bucket), bucket.astype("float"), -1.0)
    tmp = pd.DataFrame({"Date": panel["Date"].to_numpy(), "b": bucket,
                        "f": panel[col].to_numpy(dtype=float)})
    return tmp.groupby(["Date", "b"])["f"].rank(pct=True)


def _ts_argmax(panel, col, w):
    """WorldQuant `ts_arg_max`: recency of the window max, 0..1 (1 = peak is today).
    Non-monotone in the level -- encodes WHEN the extreme happened, not its size."""
    grp = panel["Ticker"]
    return panel[col].groupby(grp).transform(
        lambda s: s.rolling(w, min_periods=max(10, w // 4)).apply(
            lambda a: (int(np.argmax(a)) + 1) / len(a), raw=True)
    )


def _ts_corr_ret(panel, col, w):
    """`ts_corr` of the feature with contemporaneous daily return, per ticker.
    A 'is this feature currently co-moving with price' meta-signal. Reorders."""
    out = pd.Series(np.nan, index=panel.index)
    for _tic, idx in panel.groupby("Ticker").groups.items():
        s = panel.loc[idx, col]
        r = panel.loc[idx, "_ret"]
        out.loc[idx] = s.rolling(w, min_periods=max(10, w // 4)).corr(r)
    return out


# --------------------------------------------------------------------------
# tsfresh-style rolling SHAPE / DISTRIBUTION / COMPLEXITY operators.
# All summarize a trailing window of the FEATURE itself -> leakage-safe, and
# all are non-monotone (a tree cannot recover them from the raw level).
# --------------------------------------------------------------------------
def _ts_roll(panel, col, w, fn, min_frac=4):
    """Apply a within-ticker trailing rolling reduction `fn` to the feature."""
    mp = max(10, w // min_frac)
    grp = panel["Ticker"]
    return panel[col].groupby(grp).transform(lambda s: s.rolling(w, min_periods=mp).apply(fn, raw=True))


def _ts_std(panel, col, w):
    grp = panel["Ticker"]
    return panel[col].groupby(grp).transform(lambda s: s.rolling(w, min_periods=max(10, w // 4)).std())


def _ts_skew(panel, col, w):
    grp = panel["Ticker"]
    return panel[col].groupby(grp).transform(lambda s: s.rolling(w, min_periods=max(10, w // 4)).skew())


def _ts_kurt(panel, col, w):
    grp = panel["Ticker"]
    return panel[col].groupby(grp).transform(lambda s: s.rolling(w, min_periods=max(10, w // 4)).kurt())


def _cid_ce(panel, col, w):
    """tsfresh complexity estimate: sqrt(sum of squared consecutive diffs). Jaggedness."""
    grp = panel["Ticker"]
    d2 = panel[col].groupby(grp).diff() ** 2
    return d2.groupby(grp).transform(lambda s: s.rolling(w, min_periods=max(10, w // 4)).sum()) ** 0.5


def _minmax_pos(panel, col, w):
    """Stochastic-oscillator position of the feature in its trailing [min,max] range."""
    grp = panel["Ticker"]
    def f(s):
        lo = s.rolling(w, min_periods=max(10, w // 4)).min()
        hi = s.rolling(w, min_periods=max(10, w // 4)).max()
        return (s - lo) / (hi - lo).replace(0, np.nan)
    return panel[col].groupby(grp).transform(f)


def _ts_argmin(panel, col, w):
    return _ts_roll(panel, col, w, lambda a: (int(np.argmin(a)) + 1) / len(a))


def _count_above_mean(panel, col, w):
    return _ts_roll(panel, col, w, lambda a: float(np.mean(a > np.mean(a))))


def _longest_strike_above(panel, col, w):
    def strike(a):
        above = a > np.mean(a)
        best = cur = 0
        for v in above:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        return best / len(a)
    return _ts_roll(panel, col, w, strike)


def _crossings_mean(panel, col, w):
    def cross(a):
        s = np.sign(a - np.mean(a))
        return float(np.sum(np.abs(np.diff(s)) > 0)) / len(a)
    return _ts_roll(panel, col, w, cross)


def _autocorr1(panel, col, w):
    def ac(a):
        if np.std(a) == 0:
            return 0.0
        return float(np.corrcoef(a[:-1], a[1:])[0, 1])
    return _ts_roll(panel, col, w, ac)


def _ratio_beyond(panel, col, w, r=2.0):
    def rb(a):
        sd = np.std(a)
        return float(np.mean(np.abs(a - np.mean(a)) > r * sd)) if sd > 0 else 0.0
    return _ts_roll(panel, col, w, rb)


def _decay_linear(panel, col, w):
    """Alpha101 decay_linear: linearly-weighted MA (recent weighted most)."""
    def dl(a):
        wt = np.arange(1, len(a) + 1, dtype=float)
        return float(np.dot(a, wt) / wt.sum())
    grp = panel["Ticker"]
    return panel[col].groupby(grp).transform(
        lambda s: s.rolling(w, min_periods=max(5, w // 2)).apply(dl, raw=True))


# Interaction / gate transforms inject an EXTERNAL signal (vol, trend, ...).
# Their tail edge is only real if it beats that signal ALONE -- otherwise the
# "improvement" is just the context's own cross-sectional edge leaking in
# (e.g. "high-vol names win more" has nothing to do with your feature). Each
# such transform is therefore judged against max(raw_feature, context_alone).
TRANSFORM_CONTEXT: dict[str, str] = {
    "x_vol": "_vol_z", "x_dvol": "_dvol_z", "x_trend": "_trend",
    "gate_hivol": "_vol_z", "gate_up": "_trend",
}
CONTEXT_LABEL = {"_vol_z": "vol regime", "_dvol_z": "dollar-vol", "_trend": "trend"}


# Each entry: fn(panel, col) -> Series.  kind drives codegen; core drives sweep.
TRANSFORMS: dict[str, dict] = {
    "ts_z_20":    dict(fn=lambda p, c: _ts_roll_z(p, c, 20),  desc="within-ticker 20d rolling z-score",   kind="ts", core=False, emittable=True),
    "ts_z_60":    dict(fn=lambda p, c: _ts_roll_z(p, c, 60),  desc="within-ticker 60d rolling z-score",   kind="ts", core=True,  emittable=True),
    "ts_z_120":   dict(fn=lambda p, c: _ts_roll_z(p, c, 120), desc="within-ticker 120d rolling z-score",  kind="ts", core=False, emittable=True),
    "ts_rank_60": dict(fn=lambda p, c: _ts_rank(p, c, 60),    desc="within-ticker 60d rolling pct-rank",  kind="ts", core=False, emittable=True),
    "ts_rank_120":dict(fn=lambda p, c: _ts_rank(p, c, 120),   desc="within-ticker 120d rolling pct-rank", kind="ts", core=True,  emittable=True),
    "ema_10":     dict(fn=lambda p, c: _ema(p, c, 10),        desc="10-span EMA smoothing",               kind="ts", core=False, emittable=True),
    "diff_5":     dict(fn=lambda p, c: _diff(p, c, 5),        desc="5-day change (momentum of feature)",  kind="ts", core=False, emittable=True),
    "roc_10":     dict(fn=lambda p, c: _roc(p, c, 10),        desc="10-day rate-of-change of feature",    kind="ts", core=False, emittable=True),
    "xs_rank":    dict(fn=lambda p, c: _xs_rank(p, c),        desc="cross-sectional pct-rank within day", kind="xs", core=True,  emittable=False),
    "xs_z":       dict(fn=lambda p, c: _xs_z(p, c),           desc="cross-sectional z-score within day",  kind="xs", core=False, emittable=False),
    "x_vol":      dict(fn=lambda p, c: _interact(p, c, "_vol_z"),    desc="feature x vol-regime z",       kind="interact", core=True,  emittable=True),
    "x_dvol":     dict(fn=lambda p, c: _interact(p, c, "_dvol_z"),   desc="feature x dollar-volume z",    kind="interact", core=False, emittable=True),
    "x_trend":    dict(fn=lambda p, c: _interact(p, c, "_trend"),    desc="feature x trend",              kind="interact", core=False, emittable=True),
    "gate_hivol": dict(fn=lambda p, c: _gate(p, c, "_gate_hivol"),   desc="feature gated to high-vol days", kind="interact", core=True,  emittable=True),
    "gate_up":    dict(fn=lambda p, c: _gate(p, c, "_gate_up"),      desc="feature gated to uptrend days",  kind="interact", core=False, emittable=True),
    "abs_tsz_60": dict(fn=lambda p, c: _abs_tsz(p, c, 60),    desc="|60d z| -- FOLDING / extremeness (U-shape)", kind="ts", core=True,  emittable=True),
    "gate_extreme":dict(fn=lambda p, c: _gate_extreme(p, c, 60, 1.5), desc="feature only when |60d z| > 1.5",   kind="ts", core=False, emittable=True),
    "diff2_5":    dict(fn=lambda p, c: _diff2(p, c, 5),       desc="5d acceleration (2nd difference)",     kind="ts", core=False, emittable=True),
    # --- WorldQuant-style operators added from web research (2026-06-14) ---
    "neut_trend": dict(fn=lambda p, c: _xs_neutralize(p, c, "_trend"),  desc="vector_neut: residual orthogonal to trend", kind="xs", core=False, emittable=False),
    "neut_vol":   dict(fn=lambda p, c: _xs_neutralize(p, c, "_vol_z"),  desc="vector_neut: residual orthogonal to vol",   kind="xs", core=False, emittable=False),
    "grp_rank_vol":dict(fn=lambda p, c: _grp_rank(p, c, "_vol_z"),      desc="group_rank: rank within vol peer bucket",   kind="xs", core=False, emittable=False),
    "ts_argmax_60":dict(fn=lambda p, c: _ts_argmax(p, c, 60),  desc="ts_argmax: recency of 60d peak (0..1)", kind="ts", core=False, emittable=True),
    "ts_corr_ret_20":dict(fn=lambda p, c: _ts_corr_ret(p, c, 20), desc="ts_corr: 20d corr(feature, return)", kind="ts", core=False, emittable=True),
    # --- tsfresh-style rolling shape/complexity ops (web research, 2026-06-14) ---
    "ts_std_60":     dict(fn=lambda p, c: _ts_std(p, c, 60),    desc="rolling 60d std of the feature (its own vol)", kind="ts", core=False, emittable=True),
    "ts_skew_60":    dict(fn=lambda p, c: _ts_skew(p, c, 60),   desc="rolling 60d skewness",                  kind="ts", core=False, emittable=True),
    "ts_kurt_60":    dict(fn=lambda p, c: _ts_kurt(p, c, 60),   desc="rolling 60d kurtosis (tail heaviness)", kind="ts", core=False, emittable=True),
    "cid_ce_20":     dict(fn=lambda p, c: _cid_ce(p, c, 20),    desc="complexity: sqrt(sum 20d sq-diffs)",    kind="ts", core=False, emittable=True),
    "minmax_pos_60": dict(fn=lambda p, c: _minmax_pos(p, c, 60),desc="stochastic position in 60d [min,max]",  kind="ts", core=True,  emittable=True),
    "ts_argmin_60":  dict(fn=lambda p, c: _ts_argmin(p, c, 60), desc="ts_argmin: recency of 60d trough",      kind="ts", core=False, emittable=True),
    "count_above_mean_60":dict(fn=lambda p, c: _count_above_mean(p, c, 60), desc="frac of 60d window above its mean", kind="ts", core=False, emittable=True),
    "longest_strike_60":dict(fn=lambda p, c: _longest_strike_above(p, c, 60), desc="longest run above mean / 60",  kind="ts", core=False, emittable=True),
    "crossings_mean_60":dict(fn=lambda p, c: _crossings_mean(p, c, 60),  desc="mean-crossing rate over 60d (choppiness)", kind="ts", core=False, emittable=True),
    "autocorr1_60":  dict(fn=lambda p, c: _autocorr1(p, c, 60), desc="lag-1 autocorrelation over 60d",        kind="ts", core=False, emittable=True),
    "ratio_beyond_2s_60":dict(fn=lambda p, c: _ratio_beyond(p, c, 60, 2.0), desc="frac of 60d beyond 2 sigma (tail freq)", kind="ts", core=False, emittable=True),
    "decay_linear_10":dict(fn=lambda p, c: _decay_linear(p, c, 10), desc="linearly-weighted 10d MA",          kind="ts", core=False, emittable=True),
    # --- more cross-sectional neutralization / group ops ---
    "neut_dvol":     dict(fn=lambda p, c: _xs_neutralize(p, c, "_dvol_z"), desc="vector_neut: residual orthogonal to size", kind="xs", core=False, emittable=False),
    "grp_rank_dvol": dict(fn=lambda p, c: _grp_rank(p, c, "_dvol_z"),  desc="group_rank: rank within size peer bucket",   kind="xs", core=False, emittable=False),
}

# NOTE on transforms deliberately NOT here -- they are NO-OPS for a tree model
# fed a single feature (confirmed by the XGBoost + WorldQuant feature-engineering
# literature), so they would only waste your time:
#   * negate / invert sign        -> tree handles direction; the tool already
#                                    reports BOTH tails (the 'side' column).
#   * any single threshold / bin   -> the tree finds its own split point.
#   * clip / winsorize             -> monotone (with ties) -> same splits.
#   * log / sqrt / pow / sigmoid   -> strictly monotone -> identical model & IC.
#   * signedpower sign(x)|x|^a     -> monotone in x -> no-op (it's an IC/linear
#                                    trick, useless for trees).
#   * scale (L1 normalize per day) -> monotone within day -> same ordering.
# What DOES add information a tree can't recover on its own (and IS here):
#   - non-monotone reshaping     : abs_tsz (folding), diff/diff2, ts_argmax
#   - conditioning / gates       : gate_*, gate_extreme
#   - cross-sectional reshaping  : xs_rank, xs_z, grp_rank (group_rank)
#   - neutralization             : neut_* (vector_neut -- strips a confound)
#   - feature x feature          : x_*, xf_* (--interact_with)


# ===========================================================================
# SECTION 3 -- Scoring
# ===========================================================================

try:
    from scipy import stats as _scipy_stats
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


def compute_ic(values: pd.Series, fwd: np.ndarray) -> float:
    """Pooled Spearman IC of a transform's values vs next-day log-return."""
    v = values.to_numpy(dtype=float)
    m = np.isfinite(v) & np.isfinite(fwd)
    if m.sum() < 100:
        return np.nan
    if _HAVE_SCIPY:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr, _ = _scipy_stats.spearmanr(v[m], fwd[m])
        return float(corr) if np.isfinite(corr) else np.nan
    # fallback: pearson on ranks
    a = pd.Series(v[m]).rank().to_numpy()
    b = pd.Series(fwd[m]).rank().to_numpy()
    c = np.corrcoef(a, b)[0, 1]
    return float(c) if np.isfinite(c) else np.nan


def compute_tail_lift(
    panel: pd.DataFrame,
    values: pd.Series,
    is_winner: np.ndarray,
    valid_day: np.ndarray,
    q: float,
    winner_pct: float,
) -> dict:
    """
    Cross-sectional tail lift.

    Each day, rank `values` across the names traded that day. Among the top-q
    slice, what fraction land in the next-day winner set?  lift = that fraction
    / winner_pct (the base rate). Symmetric bottom-q slice is also measured so
    the metric is sign-agnostic. Returns the better (by |lift-1|) side.
    """
    tmp = pd.DataFrame({"Date": panel["Date"].values, "v": values.to_numpy(dtype=float)})
    rank = tmp.groupby("Date")["v"].rank(pct=True).to_numpy()
    finite = np.isfinite(rank) & valid_day

    top = finite & (rank >= 1.0 - q)
    bot = finite & (rank <= q)

    def _lift(sel):
        if sel.sum() < 30:
            return np.nan, int(sel.sum())
        prec = is_winner[sel].mean()
        return float(prec / winner_pct), int(sel.sum())

    lift_top, n_top = _lift(top)
    lift_bot, n_bot = _lift(bot)

    cand = []
    if np.isfinite(lift_top):
        cand.append(("top", lift_top, n_top))
    if np.isfinite(lift_bot):
        cand.append(("bot", lift_bot, n_bot))
    if not cand:
        return dict(lift=np.nan, side="-", n=0, lift_top=lift_top, lift_bot=lift_bot)

    side, lift, n = max(cand, key=lambda x: abs(x[1] - 1.0))
    return dict(lift=lift, side=side, n=n, lift_top=lift_top, lift_bot=lift_bot)


def compute_context_lifts(panel, is_winner, valid_day, q, winner_pct) -> dict[str, float]:
    """Standalone tail lift of each context signal -- the bar interactions must clear."""
    out: dict[str, float] = {}
    for ctx in ("_vol_z", "_dvol_z", "_trend"):
        if ctx in panel.columns:
            out[ctx] = compute_tail_lift(panel, panel[ctx], is_winner, valid_day, q, winner_pct)["lift"]
    return out


def run_battery(panel, col, transforms, is_winner, valid_day, q, winner_pct, ctx_lifts,
                transform_context=None) -> list[dict]:
    """Score raw + every transform for one feature column. Returns rows for printing."""
    tctx = transform_context if transform_context is not None else TRANSFORM_CONTEXT
    fwd = panel["_fwd_ret"].to_numpy(dtype=float)
    rows: list[dict] = []

    # baseline (raw)
    raw_vals = panel[col]
    raw_ic = compute_ic(raw_vals, fwd)
    raw_tail = compute_tail_lift(panel, raw_vals, is_winner, valid_day, q, winner_pct)
    rows.append(dict(name="raw", desc="untransformed feature", ic=raw_ic,
                     lift=raw_tail["lift"], side=raw_tail["side"], kind="-",
                     emittable=False, gain=0.0, ctx_lift=np.nan))

    raw_lift = raw_tail["lift"] if np.isfinite(raw_tail["lift"]) else 1.0
    raw_edge = abs(raw_lift - 1.0)

    for tname, spec in transforms.items():
        try:
            vals = spec["fn"](panel, col)
        except Exception as exc:  # noqa
            rows.append(dict(name=tname, desc=spec["desc"], ic=np.nan, lift=np.nan,
                             side="-", kind=spec["kind"], emittable=spec["emittable"],
                             gain=np.nan, ctx_lift=np.nan, error=str(exc)[:40]))
            continue
        ic = compute_ic(vals, fwd)
        tail = compute_tail_lift(panel, vals, is_winner, valid_day, q, winner_pct)
        lift = tail["lift"]

        # interactions/gates must beat their context signal alone, not just raw
        ctx = tctx.get(tname)
        ctx_lift = ctx_lifts.get(ctx, np.nan) if ctx else np.nan
        base_edge = raw_edge
        if np.isfinite(ctx_lift):
            base_edge = max(base_edge, abs(ctx_lift - 1.0))

        gain = (abs(lift - 1.0) - base_edge) if np.isfinite(lift) else np.nan
        rows.append(dict(name=tname, desc=spec["desc"], ic=ic, lift=lift,
                         side=tail["side"], kind=spec["kind"],
                         emittable=spec["emittable"], gain=gain, ctx_lift=ctx_lift))
    return rows


# ===========================================================================
# SECTION 4 -- Reporting
# ===========================================================================

def print_battery(col: str, block: str, rows: list[dict], ctx_lifts: dict[str, float] | None = None) -> None:
    _section(f"Battery: {col}   {_c('[' + block + ']', DIM)}")
    print(
        f"  {_c('lift', DIM)} = cross-sectional tail edge (1.0 = none, 2.0 = 2x base rate)   "
        f"{_c('IC', DIM)} = pooled Spearman vs next-day return"
    )
    if ctx_lifts:
        baseline = "   ".join(
            f"{CONTEXT_LABEL.get(k, k)} {_c(f'{v:.2f}', _lift_color(v))}"
            for k, v in ctx_lifts.items() if np.isfinite(v)
        )
        print(f"  {_c('context-alone lift (interactions must beat this):', DIM)}  {baseline}")
    print()
    print(BOLD + f"  {'Transform':<14}  {'kind':<9}  {'IC':>8}  {'lift':>6}  {'side':>4}  "
          f"{'+edge':>7}  {'emit':>4}  Description" + RESET)
    _rule()

    raw = next((r for r in rows if r["name"] == "raw"), None)
    # sort: raw first, then by descending lift edge
    body = [r for r in rows if r["name"] != "raw"]
    body.sort(key=lambda r: (abs(r["lift"] - 1.0) if np.isfinite(r.get("lift", np.nan)) else -1), reverse=True)
    ordered = ([raw] if raw else []) + body

    best = max((r for r in body if np.isfinite(r.get("gain", np.nan))),
               key=lambda r: r["gain"], default=None)

    for r in ordered:
        ic = r["ic"]
        lift = r["lift"]
        ic_s = f"{ic:>+8.4f}" if np.isfinite(ic) else f"{'nan':>8}"
        lift_s = f"{lift:>6.2f}" if np.isfinite(lift) else f"{'nan':>6}"
        gain = r.get("gain", np.nan)
        gain_s = f"{gain:>+7.2f}" if np.isfinite(gain) else f"{'-':>7}"
        emit_s = "yes" if r.get("emittable") else "-"

        name = r["name"]
        marker = ""
        if best and name == best["name"]:
            marker = _c(" <- best", GREEN_BRIGHT)
        name_disp = _c(f"{name:<14}", GREEN_BRIGHT) if (best and name == best["name"]) else f"{name:<14}"

        desc = r["desc"]
        ctx_lift = r.get("ctx_lift", np.nan)
        if np.isfinite(ctx_lift):
            desc += f"  (vs ctx {ctx_lift:.2f})"

        print(
            f"  {name_disp}  {r['kind']:<9}  {_c(ic_s, _ic_color(ic) if np.isfinite(ic) else DIM)}  "
            f"{_c(lift_s, _lift_color(lift) if np.isfinite(lift) else DIM)}  {r['side']:>4}  "
            f"{_c(gain_s, GREEN if (np.isfinite(gain) and gain > 0) else DIM)}  "
            f"{emit_s:>4}  {_c(desc, DIM)}{marker}"
        )

    _rule()
    if best and np.isfinite(best["gain"]) and best["gain"] > 0:
        verdict = (f"  Best: {_c(best['name'], GREEN_BRIGHT)}  "
                   f"(lift {raw['lift']:.2f} -> {best['lift']:.2f}, IC {raw['ic']:+.4f} -> {best['ic']:+.4f})")
        if best["emittable"]:
            verdict += _c(f"   ->  --emit {best['name']}", YELLOW)
        else:
            verdict += _c("   (cross-sectional: add in the XS stage, not a per-ticker block)", DIM)
        print(verdict)
    else:
        print(_c("  No transform improves tail edge -- this feature may be noise, leave it.", DIM))
    print()


def print_sweep(results: list[dict], winner_pct: float, q: float) -> None:
    """results: one dict per feature with raw + best transform summary."""
    _section(f"Rescue sweep  (winner=top {winner_pct:.0%},  slice=top/bot {q:.0%})")
    # rank features by improvement opportunity (gain in tail edge)
    results.sort(key=lambda r: (r["gain"] if np.isfinite(r["gain"]) else -1), reverse=True)

    cw = min(max((len(r["col"]) for r in results), default=10), 40)
    bw = min(max((len(r["block"]) for r in results), default=8), 22)

    print(BOLD + f"  {'Feature':<{cw}}  {'Block':<{bw}}  {'raw IC':>7}  {'raw lift':>8}  "
          f"{'best transform':<14}  {'new lift':>8}  {'+edge':>7}" + RESET)
    _rule()

    n_rescued = 0
    for r in results:
        gain = r["gain"]
        if np.isfinite(gain) and gain > 0.10:
            n_rescued += 1
        col = r["col"] if len(r["col"]) <= cw else r["col"][:cw - 2] + ".."
        blk = r["block"] if len(r["block"]) <= bw else r["block"][:bw - 2] + ".."
        rawl = f"{r['raw_lift']:.2f}" if np.isfinite(r["raw_lift"]) else "nan"
        newl = f"{r['best_lift']:.2f}" if np.isfinite(r["best_lift"]) else "nan"
        gain_s = f"{gain:>+7.2f}" if np.isfinite(gain) else f"{'-':>7}"
        gc = GREEN_BRIGHT if (np.isfinite(gain) and gain > 0.25) else (GREEN if (np.isfinite(gain) and gain > 0.10) else DIM)
        blk_cell = _c(f"{blk:<{bw}}", DIM)
        best_cell = _c(f"{r['best_name']:<14}", gc)
        print(
            f"  {col:<{cw}}  {blk_cell}  {r['raw_ic']:>+7.4f}  {rawl:>8}  "
            f"{best_cell}  {newl:>8}  {_c(gain_s, gc)}"
        )

    _rule()
    print(f"  {BOLD}{n_rescued}{RESET} / {len(results)} meh features have a transform that lifts tail edge by > 0.10")
    print(_c(f"  Re-run a single one with --feature <name> to see its full battery, then --emit <transform>.", DIM))
    print()


# ===========================================================================
# SECTION 5 -- Codegen
# ===========================================================================

_EMIT_HEADER = '''"""
{name}.py  --  auto-drafted by __feature_lab.py

Transform "{transform}" applied to base feature "{base}".
Battery result on sample: tail lift {raw_lift:.2f} -> {new_lift:.2f},
IC {raw_ic:+.4f} -> {new_ic:+.4f}.

REVIEW before committing: confirm the gain holds on a larger --n and a
walk-forward split; tighten the window if it overfits.
"""
import numpy as np
import pandas as pd

METADATA = {{
    "name":        "{name}",
    "description": "{desc}",
    "requires":    ["{base}"],
    "produces":    ["{out_col}"],
    "tags":        ["transform", "experimental"],
    "version":     "1.0",
    "author":      "__feature_lab.py codegen",
}}


def compute(df: pd.DataFrame) -> pd.DataFrame:
{body}
    return df
'''

# Per-ticker bodies. The framework calls compute() on ONE ticker at a time,
# so these are plain trailing-window ops -- no groupby needed.
_EMIT_BODIES: dict[str, str] = {
    "ts_z": (
        "    w = {w}\n"
        "    s = df[\"{base}\"]\n"
        "    mean = s.rolling(w, min_periods=max(10, w // 4)).mean()\n"
        "    std = s.rolling(w, min_periods=max(10, w // 4)).std().replace(0, np.nan)\n"
        "    df[\"{out_col}\"] = (s - mean) / std"
    ),
    "ts_rank": (
        "    w = {w}\n"
        "    df[\"{out_col}\"] = df[\"{base}\"].rolling(w, min_periods=max(10, w // 4)).rank(pct=True)"
    ),
    "ema": (
        "    df[\"{out_col}\"] = df[\"{base}\"].ewm(span={w}, min_periods=max(5, {w} // 2)).mean()"
    ),
    "diff": (
        "    df[\"{out_col}\"] = df[\"{base}\"].diff({w})"
    ),
    "roc": (
        "    w = {w}\n"
        "    s = df[\"{base}\"]\n"
        "    df[\"{out_col}\"] = s.diff(w) / s.shift(w).abs().replace(0, np.nan)"
    ),
    "x_vol": (
        "    ret = df[\"Close\"].pct_change()\n"
        "    vol20 = ret.rolling(20, min_periods=10).std()\n"
        "    vol_z = (vol20 - vol20.rolling(120, min_periods=30).mean()) / vol20.rolling(120, min_periods=30).std().replace(0, np.nan)\n"
        "    df[\"{out_col}\"] = df[\"{base}\"] * vol_z"
    ),
    "x_dvol": (
        "    dvol = df[\"Close\"] * df[\"Volume\"]\n"
        "    dvol_z = (dvol - dvol.rolling(60, min_periods=15).mean()) / dvol.rolling(60, min_periods=15).std().replace(0, np.nan)\n"
        "    df[\"{out_col}\"] = df[\"{base}\"] * dvol_z"
    ),
    "x_trend": (
        "    ma50 = df[\"Close\"].rolling(50, min_periods=20).mean()\n"
        "    df[\"{out_col}\"] = df[\"{base}\"] * (df[\"Close\"] / ma50 - 1.0)"
    ),
    "gate_hivol": (
        "    ret = df[\"Close\"].pct_change()\n"
        "    vol20 = ret.rolling(20, min_periods=10).std()\n"
        "    vmed = vol20.rolling(120, min_periods=30).median()\n"
        "    # conditional feature: value on high-vol days, NaN otherwise (XGBoost handles NaN)\n"
        "    df[\"{out_col}\"] = df[\"{base}\"].where(vol20 >= vmed)"
    ),
    "gate_up": (
        "    ma50 = df[\"Close\"].rolling(50, min_periods=20).mean()\n"
        "    # conditional feature: value on uptrend days, NaN otherwise (XGBoost handles NaN)\n"
        "    df[\"{out_col}\"] = df[\"{base}\"].where(df[\"Close\"] > ma50)"
    ),
    "abs_tsz": (
        "    w = {w}\n"
        "    s = df[\"{base}\"]\n"
        "    mean = s.rolling(w, min_periods=max(10, w // 4)).mean()\n"
        "    std = s.rolling(w, min_periods=max(10, w // 4)).std().replace(0, np.nan)\n"
        "    # folding / extremeness: |z| -- a U-shape a tree can't build from one split\n"
        "    df[\"{out_col}\"] = ((s - mean) / std).abs()"
    ),
    "gate_extreme": (
        "    w = {w}\n"
        "    s = df[\"{base}\"]\n"
        "    mean = s.rolling(w, min_periods=max(10, w // 4)).mean()\n"
        "    std = s.rolling(w, min_periods=max(10, w // 4)).std().replace(0, np.nan)\n"
        "    z = (s - mean) / std\n"
        "    # value only when the feature itself is extreme (|z| > 1.5), NaN otherwise\n"
        "    df[\"{out_col}\"] = s.where(z.abs() > 1.5)"
    ),
    "diff2": (
        "    df[\"{out_col}\"] = df[\"{base}\"].diff({w}).diff({w})"
    ),
    "ts_argmax": (
        "    w = {w}\n"
        "    # recency of the window max, 0..1 (1 = the peak is today)\n"
        "    df[\"{out_col}\"] = df[\"{base}\"].rolling(w, min_periods=max(10, w // 4)).apply(\n"
        "        lambda a: (int(np.argmax(a)) + 1) / len(a), raw=True)"
    ),
    "ts_corr_ret": (
        "    w = {w}\n"
        "    ret = df[\"Close\"].pct_change()\n"
        "    df[\"{out_col}\"] = df[\"{base}\"].rolling(w, min_periods=max(10, w // 4)).corr(ret)"
    ),
    "ts_std": (
        "    w = {w}\n"
        "    df[\"{out_col}\"] = df[\"{base}\"].rolling(w, min_periods=max(10, w // 4)).std()"
    ),
    "ts_skew": (
        "    w = {w}\n"
        "    df[\"{out_col}\"] = df[\"{base}\"].rolling(w, min_periods=max(10, w // 4)).skew()"
    ),
    "ts_kurt": (
        "    w = {w}\n"
        "    df[\"{out_col}\"] = df[\"{base}\"].rolling(w, min_periods=max(10, w // 4)).kurt()"
    ),
    "cid_ce": (
        "    w = {w}\n"
        "    d2 = df[\"{base}\"].diff() ** 2\n"
        "    df[\"{out_col}\"] = d2.rolling(w, min_periods=max(10, w // 4)).sum() ** 0.5"
    ),
    "minmax_pos": (
        "    w = {w}\n"
        "    s = df[\"{base}\"]\n"
        "    lo = s.rolling(w, min_periods=max(10, w // 4)).min()\n"
        "    hi = s.rolling(w, min_periods=max(10, w // 4)).max()\n"
        "    df[\"{out_col}\"] = (s - lo) / (hi - lo).replace(0, np.nan)"
    ),
    "ts_argmin": (
        "    w = {w}\n"
        "    df[\"{out_col}\"] = df[\"{base}\"].rolling(w, min_periods=max(10, w // 4)).apply(\n"
        "        lambda a: (int(np.argmin(a)) + 1) / len(a), raw=True)"
    ),
    "count_above_mean": (
        "    w = {w}\n"
        "    df[\"{out_col}\"] = df[\"{base}\"].rolling(w, min_periods=max(10, w // 4)).apply(\n"
        "        lambda a: float(np.mean(a > np.mean(a))), raw=True)"
    ),
    "longest_strike": (
        "    w = {w}\n"
        "    def _strike(a):\n"
        "        above = a > np.mean(a); best = cur = 0\n"
        "        for v in above:\n"
        "            cur = cur + 1 if v else 0\n"
        "            best = max(best, cur)\n"
        "        return best / len(a)\n"
        "    df[\"{out_col}\"] = df[\"{base}\"].rolling(w, min_periods=max(10, w // 4)).apply(_strike, raw=True)"
    ),
    "crossings_mean": (
        "    w = {w}\n"
        "    def _cross(a):\n"
        "        s = np.sign(a - np.mean(a))\n"
        "        return float(np.sum(np.abs(np.diff(s)) > 0)) / len(a)\n"
        "    df[\"{out_col}\"] = df[\"{base}\"].rolling(w, min_periods=max(10, w // 4)).apply(_cross, raw=True)"
    ),
    "autocorr1": (
        "    w = {w}\n"
        "    def _ac(a):\n"
        "        return float(np.corrcoef(a[:-1], a[1:])[0, 1]) if np.std(a) > 0 else 0.0\n"
        "    df[\"{out_col}\"] = df[\"{base}\"].rolling(w, min_periods=max(10, w // 4)).apply(_ac, raw=True)"
    ),
    "ratio_beyond": (
        "    w = {w}\n"
        "    def _rb(a):\n"
        "        sd = np.std(a)\n"
        "        return float(np.mean(np.abs(a - np.mean(a)) > 2.0 * sd)) if sd > 0 else 0.0\n"
        "    df[\"{out_col}\"] = df[\"{base}\"].rolling(w, min_periods=max(10, w // 4)).apply(_rb, raw=True)"
    ),
    "decay_linear": (
        "    w = {w}\n"
        "    def _dl(a):\n"
        "        wt = np.arange(1, len(a) + 1, dtype=float)\n"
        "        return float(np.dot(a, wt) / wt.sum())\n"
        "    df[\"{out_col}\"] = df[\"{base}\"].rolling(w, min_periods=max(5, w // 2)).apply(_dl, raw=True)"
    ),
}

# transform name -> (body key, window arg, out-suffix)
_EMIT_MAP = {
    "ts_z_20": ("ts_z", 20, "tsz20"), "ts_z_60": ("ts_z", 60, "tsz60"), "ts_z_120": ("ts_z", 120, "tsz120"),
    "ts_rank_60": ("ts_rank", 60, "tsrank60"), "ts_rank_120": ("ts_rank", 120, "tsrank120"),
    "ema_10": ("ema", 10, "ema10"), "diff_5": ("diff", 5, "diff5"), "roc_10": ("roc", 10, "roc10"),
    "x_vol": ("x_vol", None, "xvol"), "x_dvol": ("x_dvol", None, "xdvol"), "x_trend": ("x_trend", None, "xtrend"),
    "gate_hivol": ("gate_hivol", None, "ghivol"), "gate_up": ("gate_up", None, "gup"),
    "abs_tsz_60": ("abs_tsz", 60, "abstsz60"), "gate_extreme": ("gate_extreme", 60, "gext"),
    "diff2_5": ("diff2", 5, "accel5"),
    "ts_argmax_60": ("ts_argmax", 60, "argmax60"), "ts_corr_ret_20": ("ts_corr_ret", 20, "corret20"),
    "ts_std_60": ("ts_std", 60, "tsstd60"), "ts_skew_60": ("ts_skew", 60, "tsskew60"),
    "ts_kurt_60": ("ts_kurt", 60, "tskurt60"), "cid_ce_20": ("cid_ce", 20, "cidce20"),
    "minmax_pos_60": ("minmax_pos", 60, "mmpos60"), "ts_argmin_60": ("ts_argmin", 60, "argmin60"),
    "count_above_mean_60": ("count_above_mean", 60, "cam60"), "longest_strike_60": ("longest_strike", 60, "lstrk60"),
    "crossings_mean_60": ("crossings_mean", 60, "xmean60"), "autocorr1_60": ("autocorr1", 60, "ac1_60"),
    "ratio_beyond_2s_60": ("ratio_beyond", 60, "rb2s60"), "decay_linear_10": ("decay_linear", 10, "dlin10"),
}


def emit_xfeat_block(base: str, partner: str, op: str, stats: dict) -> None:
    """Codegen for a cross-feature trick (feature * or / another feature)."""
    suffix = ("x_" if op == "mul" else "div_") + partner
    out_col = f"{base}_{suffix}"
    if op == "mul":
        body = f'    df["{out_col}"] = df["{base}"] * df["{partner}"]'
    else:
        body = f'    df["{out_col}"] = df["{base}"] / df["{partner}"].replace(0, np.nan)'
    code = _EMIT_HEADER.format(
        name=out_col, transform=f"{op} with {partner}", base=base, out_col=out_col,
        desc=f"{base} {'times' if op == 'mul' else 'over'} {partner}",
        raw_lift=stats.get("raw_lift", float("nan")), new_lift=stats.get("new_lift", float("nan")),
        raw_ic=stats.get("raw_ic", float("nan")), new_ic=stats.get("new_ic", float("nan")),
        body=body,
    )
    # cross-feature block needs BOTH columns present
    code = code.replace(f'"requires":    ["{base}"],', f'"requires":    ["{base}", "{partner}"],')
    print()
    print(_c(SEP, DIM))
    print(BOLD + f"  Drafted block -> FeatureTemplates/{out_col}.py" + RESET)
    print(_c(SEP, DIM))
    print(code)
    print(_c("  (not written to disk -- copy it into the file above, review, then run", DIM))
    print(_c("   python 3__FeatureFramework.py --ticker AAPL  to smoke-test it.)", DIM))


def emit_block(base: str, transform: str, stats: dict) -> None:
    if transform not in _EMIT_MAP:
        spec = TRANSFORMS.get(transform, {})
        if not spec.get("emittable", False):
            sys.exit(f"Transform '{transform}' is cross-sectional -- it belongs in the framework's "
                     f"cross-sectional stage, not a per-ticker block. Not emittable.")
        sys.exit(f"Unknown transform '{transform}'.")

    body_key, w, suffix = _EMIT_MAP[transform]
    out_col = f"{base}_{suffix}"
    name = out_col
    body_tmpl = _EMIT_BODIES[body_key]
    body = body_tmpl.format(base=base, out_col=out_col, w=w)

    code = _EMIT_HEADER.format(
        name=name, transform=transform, base=base, out_col=out_col,
        desc=f"{TRANSFORMS[transform]['desc']} of {base}",
        raw_lift=stats.get("raw_lift", float("nan")), new_lift=stats.get("new_lift", float("nan")),
        raw_ic=stats.get("raw_ic", float("nan")), new_ic=stats.get("new_ic", float("nan")),
        body=body,
    )
    out_path = ROOT / "FeatureTemplates" / f"{name}.py"
    print()
    print(_c(SEP, DIM))
    print(BOLD + f"  Drafted block -> FeatureTemplates/{name}.py" + RESET)
    print(_c(SEP, DIM))
    print(code)
    print(_c("  (not written to disk -- copy it into the file above, review, then run", DIM))
    print(_c("   python 3__FeatureFramework.py --ticker AAPL  to smoke-test it.)", DIM))


# ===========================================================================
# SECTION 6 -- Main
# ===========================================================================

def main() -> None:
    p = argparse.ArgumentParser(description="Feature-improvement test battery")
    p.add_argument("--feature", help="Single feature column to run the full battery on")
    p.add_argument("--block", help="Run the battery on every column a block produces")
    p.add_argument("--all_meh", action="store_true", help="Sweep all meh (|IC|<0.01) features")
    p.add_argument("--include_ok", action="store_true", help="In --all_meh, also include ok (0.01<=|IC|<0.05) features")
    p.add_argument("--emit", metavar="TRANSFORM", help="Codegen a FeatureTemplates block for --feature + this transform")
    p.add_argument("--interact_with", metavar="COL", help="(with --feature) also try feature x COL and feature / COL")
    p.add_argument("--only", metavar="T1,T2", help="Comma-list of transforms to test (skip the rest) -- fast all-universe vetting")
    p.add_argument("--list_transforms", action="store_true", help="Print the transform registry and exit")
    p.add_argument("--n", type=int, default=200, help="Tickers to load (default 200)")
    p.add_argument("--seed", type=int, default=42, help="Sampling seed (default 42)")
    p.add_argument("--winner_pct", type=float, default=0.10, help="Next-day winner percentile (default 0.10)")
    p.add_argument("--q", type=float, default=0.10, help="Feature top/bottom slice (default 0.10)")
    p.add_argument("--min_names", type=int, default=15, help="Skip days with fewer names (default 15)")
    p.add_argument("--full", action="store_true", help="Use full transform set in sweep (slower)")
    p.add_argument("--max_features", type=int, default=60, help="Cap features in a sweep (default 60)")
    args = p.parse_args()

    if args.list_transforms:
        _section("Transform registry  (ONLY reordering transforms -- monotonic ones are no-ops for trees)")
        print(BOLD + f"  {'Name':<14}  {'kind':<9}  {'core':>4}  {'emit':>4}  Description" + RESET)
        _rule()
        for name, spec in TRANSFORMS.items():
            core = "yes" if spec["core"] else "-"
            emit = "yes" if spec["emittable"] else "-"
            print(f"  {name:<14}  {spec['kind']:<9}  {core:>4}  {emit:>4}  {_c(spec['desc'], DIM)}")
        print()
        print(_c("  kind: ts=per-ticker time-series  xs=cross-sectional (not per-ticker emittable)  interact=feature x context", DIM))
        print(_c("  core: included in the fast --all_meh sweep (use --full for all).", DIM))
        print()
        return

    if not _HAVE_SCIPY:
        print(_c("  [warn] scipy not installed -- IC falls back to rank-pearson (pip install scipy).", ORANGE))

    block_for_col, all_produced = _produced_columns()

    # -------- which feature columns do we need? --------
    if args.feature:
        feature_cols = [args.feature]
        if args.interact_with:
            feature_cols.append(args.interact_with)
    elif args.block:
        feature_cols = [c for c, b in block_for_col.items() if b == args.block]
        if not feature_cols:
            sys.exit(f"Block '{args.block}' produces no known columns. Try --list (in 3__FeatureFramework.py).")
    elif args.all_meh:
        feature_cols = list(all_produced)  # classify after loading
    else:
        p.print_help()
        return

    # -------- load panel + context --------
    print(_c(f"  Loading {args.n} tickers from {PROCESSED_DIR.name} ...", DIM))
    load_cols = None if args.all_meh else feature_cols
    panel = load_panel(args.n, args.seed, load_cols)
    panel = add_context(panel)

    valid_day = (panel["_day_n"] >= args.min_names).to_numpy()
    is_winner = (panel["_fwd_rank"] >= (1.0 - args.winner_pct)).to_numpy()
    ctx_lifts = compute_context_lifts(panel, is_winner, valid_day, args.q, args.winner_pct)

    # restrict feature_cols to those actually present
    present = set(panel.columns)
    feature_cols = [c for c in feature_cols if c in present]
    if not feature_cols:
        sys.exit("None of the requested feature columns are present in the panel.")

    transforms = dict(TRANSFORMS)

    # cross-feature tricks: inject feature x COL and feature / COL on request
    tctx = None
    if args.feature and args.interact_with:
        pc = args.interact_with
        if pc not in present:
            sys.exit(f"--interact_with column '{pc}' not found in the panel.")
        transforms[f"xf_mul:{pc}"] = dict(
            fn=lambda p, c, _pc=pc: _xfeat(p, c, _pc, "mul"),
            desc=f"feature x {pc}", kind="xfeat", core=False, emittable=True)
        transforms[f"xf_div:{pc}"] = dict(
            fn=lambda p, c, _pc=pc: _xfeat(p, c, _pc, "div"),
            desc=f"feature / {pc}", kind="xfeat", core=False, emittable=True)
        # the product's edge must beat the PARTNER alone too, not just the base
        ctx_lifts[pc] = compute_tail_lift(panel, panel[pc], is_winner, valid_day,
                                          args.q, args.winner_pct)["lift"]
        CONTEXT_LABEL[pc] = pc
        tctx = dict(TRANSFORM_CONTEXT)
        tctx[f"xf_mul:{pc}"] = pc
        tctx[f"xf_div:{pc}"] = pc

    # transform selection: --only (shortlist, overrides) > core-only sweep > full
    if args.only:
        wanted = [t.strip() for t in args.only.split(",") if t.strip()]
        unknown = [t for t in wanted if t not in transforms]
        if unknown:
            sys.exit(f"--only: unknown transform(s) {unknown}.  See --list_transforms.")
        transforms = {k: transforms[k] for k in wanted}
    elif args.all_meh and not args.full:
        transforms = {k: v for k, v in transforms.items() if v["core"]}

    # ============ MODE: single feature ============
    if args.feature:
        col = args.feature
        rows = run_battery(panel, col, transforms, is_winner, valid_day, args.q, args.winner_pct,
                           ctx_lifts, transform_context=tctx)
        print_battery(col, block_for_col.get(col, "?"), rows, ctx_lifts)

        if args.emit:
            raw = next(r for r in rows if r["name"] == "raw")
            chosen = next((r for r in rows if r["name"] == args.emit), None)
            stats = dict(
                raw_lift=raw["lift"], new_lift=(chosen["lift"] if chosen else float("nan")),
                raw_ic=raw["ic"], new_ic=(chosen["ic"] if chosen else float("nan")),
            )
            if args.emit.startswith("xf_mul:") or args.emit.startswith("xf_div:"):
                op = "mul" if args.emit.startswith("xf_mul:") else "div"
                partner = args.emit.split(":", 1)[1]
                emit_xfeat_block(col, partner, op, stats)
            else:
                emit_block(col, args.emit, stats)
        return

    # ============ MODE: block ============
    if args.block:
        for col in feature_cols:
            rows = run_battery(panel, col, transforms, is_winner, valid_day, args.q, args.winner_pct, ctx_lifts)
            print_battery(col, args.block, rows, ctx_lifts)
        return

    # ============ MODE: sweep ============
    fwd = panel["_fwd_ret"].to_numpy(dtype=float)
    # classify by raw IC first (cheap), keep only meh (+ ok if asked)
    print(_c(f"  Classifying {len(feature_cols)} features by raw IC ...", DIM))
    classified: list[tuple[str, float]] = []
    for col in feature_cols:
        ic = compute_ic(panel[col], fwd)
        if not np.isfinite(ic):
            continue
        a = abs(ic)
        if a < 0.01 or (args.include_ok and a < 0.05):
            classified.append((col, ic))

    classified.sort(key=lambda x: abs(x[1]))  # mehest first
    classified = classified[: args.max_features]
    print(_c(f"  Running battery on {len(classified)} features "
             f"({'core' if not args.full else 'full'} transforms) ...", DIM))

    try:
        from tqdm import tqdm
        iterator = tqdm(classified, desc="battery", unit="feat")
    except ImportError:
        iterator = classified

    results: list[dict] = []
    for col, ic in iterator:
        rows = run_battery(panel, col, transforms, is_winner, valid_day, args.q, args.winner_pct, ctx_lifts)
        raw = next(r for r in rows if r["name"] == "raw")
        body = [r for r in rows if r["name"] != "raw" and np.isfinite(r.get("gain", np.nan))]
        best = max(body, key=lambda r: r["gain"], default=None)
        results.append(dict(
            col=col, block=block_for_col.get(col, "?"),
            raw_ic=raw["ic"], raw_lift=raw["lift"],
            best_name=(best["name"] if best else "-"),
            best_lift=(best["lift"] if best else float("nan")),
            gain=(best["gain"] if best else float("nan")),
        ))

    print_sweep(results, args.winner_pct, args.q)


if __name__ == "__main__":
    main()
