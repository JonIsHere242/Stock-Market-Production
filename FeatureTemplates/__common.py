"""
__common.py  --  the ONE implementation of the primitives every screening tool shares.

WHY THIS EXISTS
---------------
`__diagnostics`, `__feature_lab`, `__tail_screen`, `__live_xcheck` and
`__relatedness_map` grew by copy-paste. An AST scan on 2026-07-24 found the same
names defined in several files with bodies that had already drifted:

    _c              x5   (3 variants)
    build_panel     x3   (3 variants)
    add_context     x2   (DIVERGED IN SUBSTANCE -- see the warning below)
    compute_ic      x2
    _xs_neutralize  x2
    _within_z       x2
    _rgb/_section/_rule  x2 each

The scoring primitives had only drifted cosmetically (warning suppression, ndarray
vs Series, a faster groupby) -- the maths still agreed. That made extraction safe
*at the time of writing*, and every future edit lands in one place instead of
silently forking a second definition of "IC".

*** WHAT IS DELIBERATELY **NOT** UNIFIED HERE ***
------------------------------------------------
`add_context` is NOT in this module, on purpose. The two implementations are
genuinely different functions that happen to share a name:

    __feature_lab.add_context   -> gates + interaction partners
                                   (_vol_z, _gate_hivol, _gate_up, _dvol_z)
    __tail_screen.add_context   -> alt-targets + the neutralization basis
                                   (_volabs, _logprice, _logdvol, _ret1, _ret5, _tgt_*)

They therefore NEUTRALIZE AGAINST DIFFERENT FACTOR SETS:

    __feature_lab  neut_vol   -> residual vs _vol_z
    __tail_screen  neut       -> residual vs NEUT_FACTORS
                                 (_ret1, _ret5, _volabs, _logprice, _logdvol)

So a `neut` number printed by one tool IS NOT COMPARABLE to a `neut` number printed
by the other. __tail_screen's module docstring used to claim it "reuses the EXACT
tail-lift definition from __feature_lab.py" -- true of the lift, false of the
neutralization. Merging them silently would have changed screening results, so each
tool keeps its own and states its basis in its output header instead.

Everything here is import-safe (no side effects, no data loaded at import time) and
the module is auto-skipped by the framework (leading __).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

try:
    from scipy import stats as _scipy_stats
    HAVE_SCIPY = True
except Exception:                                    # pragma: no cover
    _scipy_stats = None
    HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# ANSI (24-bit gradient + quality tiers, matching Util.py's scheme)
# ---------------------------------------------------------------------------
def rgb(r: int, g: int, b: int) -> str:
    """Truecolor SGR parameter string, e.g. rgb(255,0,0) -> '38;2;255;0;0'."""
    return f"38;2;{r};{g};{b}"


def c(text, code: str) -> str:
    """Wrap `text` in an SGR code. Single definition -- there were 5, in 3 variants."""
    return f"\033[{code}m{text}\033[0m"


def ic_color(ic: float) -> str:
    """GREAT |IC|>=0.05, ok |IC|>=0.01, else meh."""
    a = abs(ic)
    if not np.isfinite(a):
        return "38;5;240"
    return "1;32" if a >= 0.05 else "32" if a >= 0.02 else "33" if a >= 0.01 else "38;5;240"


def lift_color(v: float) -> str:
    """Colour a tail-lift by distance from 1.0 (no edge)."""
    if not np.isfinite(v):
        return "38;5;240"
    e = abs(v - 1.0)
    return "1;32" if e >= 0.30 else "32" if e >= 0.15 else "33" if e >= 0.07 else "38;5;240"


def section(title: str, width: int = 96) -> None:
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def rule(width: int = 96) -> None:
    print("-" * width)


def fmt_ms(ms: float) -> str:
    return f"{ms / 1000:.2f}s" if ms >= 1_000 else f"{ms:.1f}ms"


# ---------------------------------------------------------------------------
# Block loading  (stems vs METADATA names)
# ---------------------------------------------------------------------------
def load_block(stem: str):
    """Import a FeatureTemplates block by FILE STEM (not METADATA name)."""
    path = HERE / f"{stem}.py" if not str(stem).endswith(".py") else Path(stem)
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_framework():
    """Import 3__FeatureFramework for discover_blocks / resolve_order."""
    spec = importlib.util.spec_from_file_location("_ff", ROOT / "3__FeatureFramework.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def block_alias_map(universe: dict) -> dict:
    """
    {file_stem OR METADATA name} -> METADATA name.

    discover_blocks() keys by METADATA["name"] but every caller passes file stems,
    and 488 of 723 blocks have name != stem. Resolve both.
    """
    alias = {}
    for name, blk in universe.items():
        alias[name] = name
        alias.setdefault(blk["path"].stem, name)
    return alias


# ---------------------------------------------------------------------------
# Scoring primitives
# ---------------------------------------------------------------------------
def compute_ic(values, fwd) -> float:
    """Pooled Spearman IC of a feature's values vs next-day log-return."""
    v = np.asarray(values, dtype=float)
    fwd = np.asarray(fwd, dtype=float)
    m = np.isfinite(v) & np.isfinite(fwd)
    if m.sum() < 100:
        return np.nan
    if HAVE_SCIPY:
        corr, _ = _scipy_stats.spearmanr(v[m], fwd[m])
        return float(corr) if np.isfinite(corr) else np.nan
    a = pd.Series(v[m]).rank().to_numpy()
    b = pd.Series(fwd[m]).rank().to_numpy()
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def within_z(panel: pd.DataFrame, series: pd.Series, win: int) -> pd.Series:
    """Within-ticker rolling z-score of an arbitrary series aligned to `panel`."""
    tmp = series.copy()
    grp = panel["Ticker"]
    mp = max(10, win // 4)
    mean = tmp.groupby(grp).transform(lambda s: s.rolling(win, min_periods=mp).mean())
    std = tmp.groupby(grp).transform(lambda s: s.rolling(win, min_periods=mp).std())
    return (tmp - mean) / std.replace(0, np.nan)


def xs_neutralize(panel: pd.DataFrame, col: str, factor: str) -> np.ndarray:
    """
    WorldQuant `vector_neut`: per-day residual of `col` regressed on `factor`.

    Isolates the part of the feature ORTHOGONAL to a risk factor -- the edge that
    ISN'T just "high-vol/high-trend names win". Non-monotone and cross-sectional,
    so it genuinely reorders rows (a monotone transform would be a no-op for a tree).

    Returns a plain ndarray aligned to panel's row order.
    """
    return xs_neutralize_multi(panel, col, [factor])


BASE_COLS = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]


def plan_blocks(requested, *, verbose: bool = True):
    """
    Resolve `requested` block names into a dependency-correct execution plan.

    THE BUG THIS EXISTS TO KILL: every build_panel in this directory used to run
    blocks in whatever order the caller listed them, with no `requires` check and
    an `except Exception: pass`. A block that depends on a column another block
    produces would fail, contribute nothing, and score as "no edge" -- a
    false-negative generator sitting in front of the whole screening funnel.
    (Measured: _lens_liquidity screened 0 of its 7 features that way.)

    Resolves against the FULL block universe (promoted + candidates), pulls in
    prerequisite blocks transitively, and returns them topologically ordered.

    Returns (order, universe, scaffold): `scaffold` is the set of prerequisite
    blocks that were pulled in but that the caller did NOT ask to screen.
    """
    fw = load_framework()
    universe = fw.discover_blocks(include_candidates=True)
    alias = block_alias_map(universe)

    resolved, unknown = [], []
    for r in requested:
        (resolved.append(alias[r]) if r in alias else unknown.append(r))
    if unknown and verbose:
        print(c(f"  [WARN] unknown block(s), NOT screened: {', '.join(unknown)}", "1;31"))

    produced_by = {}
    for name, blk in universe.items():
        for col in blk["meta"].get("produces", []):
            produced_by.setdefault(col, name)

    wanted, queue, missing_dep = set(), list(resolved), {}
    while queue:
        name = queue.pop()
        if name in wanted:
            continue
        blk = universe.get(name)
        if blk is None:
            continue
        wanted.add(name)
        for req in blk["meta"].get("requires", []):
            if req in BASE_COLS:
                continue
            producer = produced_by.get(req)
            if producer is None:
                missing_dep.setdefault(name, []).append(req)
            elif producer != name:
                queue.append(producer)

    order = fw.resolve_order({k: v for k, v in universe.items() if k in wanted})
    scaffold = wanted - set(resolved)

    if verbose:
        for name, cols in missing_dep.items():
            print(c(f"  [WARN] {name} requires {cols} which NO block produces", "1;31"))
        if scaffold:
            print(f"  [deps] pulled in {len(scaffold)} prerequisite block(s): "
                  f"{', '.join(sorted(scaffold)[:8])}{' ...' if len(scaffold) > 8 else ''}")
    return order, universe, scaffold


def build_panel(n, seed, blocks, *, price_dir: Path, sort_panel: bool = True,
                verbose: bool = True):
    """
    Build a fresh feature panel straight from PriceData, in dependency order.

    Shared by __tail_screen and __live_xcheck (which had byte-for-byte copies of
    the broken version). Failures are COUNTED AND REPORTED -- a block that never
    ran must never be mistaken for a block with no edge.

    sort_panel : __tail_screen sorts by (Ticker, Date); __live_xcheck historically
                 did not. Keep each caller's behaviour so results don't shift.
    """
    import random as _random
    import time as _time

    paths = sorted(Path(price_dir).glob("*.parquet"))
    sample = _random.Random(seed).sample(paths, min(n, len(paths)))
    order, universe, scaffold = plan_blocks(list(blocks), verbose=verbose)

    feat_cols, family = [], {}
    for b in order:
        if b in scaffold:
            continue                       # scaffolding, not under test
        for col in universe[b]["meta"].get("produces", []):
            feat_cols.append(col)
            family[col] = b

    fail_counts, fail_msg, skip_counts = {}, {}, {}
    frames = []
    t0 = _time.perf_counter()
    for p in sample:
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        if "Ticker" not in df.columns:
            df["Ticker"] = p.stem
        df = df.sort_values("Date").reset_index(drop=True)

        for b in order:
            blk = universe[b]
            missing = [r for r in blk["meta"].get("requires", []) if r not in df.columns]
            if missing:
                skip_counts[b] = skip_counts.get(b, 0) + 1
                continue
            cols_before = list(df.columns)
            try:
                df = blk["fn"](df)
                if df.columns.duplicated().any():
                    df = df.loc[:, ~df.columns.duplicated(keep="last")]
            except Exception as exc:
                keep_c = [cc for cc in cols_before if cc in df.columns]
                if len(keep_c) != len(df.columns):
                    df = df[keep_c]        # discard partial mutation
                fail_counts[b] = fail_counts.get(b, 0) + 1
                fail_msg.setdefault(b, f"{type(exc).__name__}: {exc}")

        frames.append(df[BASE_COLS + [cc for cc in feat_cols if cc in df.columns]])

    panel = pd.concat(frames, ignore_index=True)
    panel["Date"] = pd.to_datetime(panel["Date"])
    if sort_panel:
        panel = panel.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    if verbose:
        ns = len(sample)
        for b, cnt in sorted(fail_counts.items(), key=lambda kv: -kv[1]):
            print(c(f"  [FAIL] {b}: raised on {cnt}/{ns} tickers -- {fail_msg[b][:90]}", "1;31"))
        for b, cnt in sorted(skip_counts.items(), key=lambda kv: -kv[1]):
            miss = [r for r in universe[b]["meta"].get("requires", []) if r not in panel.columns]
            print(c(f"  [SKIP] {b}: requires missing on {cnt}/{ns} tickers "
                    f"({', '.join(miss[:4])})", "33"))
        ghost = [cc for cc in feat_cols if cc not in panel.columns]
        if ghost:
            print(c(f"  [GHOST] {len(ghost)} declared column(s) never materialized and are NOT "
                    f"being screened: {', '.join(ghost[:8])}"
                    f"{' ...' if len(ghost) > 8 else ''}", "1;31"))

    feat_cols = [cc for cc in feat_cols if cc in panel.columns]
    if verbose:
        print(f"  built panel: {len(sample)} tickers, {len(panel):,} rows, "
              f"{len(feat_cols)} features, {_time.perf_counter() - t0:.1f}s")
    return panel, feat_cols, family


def xs_neutralize_multi(panel: pd.DataFrame, col: str, factors: list[str]) -> np.ndarray:
    """Per-day residual of `col` jointly regressed on `factors` (multi-factor vector_neut)."""
    f = panel[col].to_numpy(dtype=float)
    X = np.column_stack([panel[fc].to_numpy(dtype=float) for fc in factors])
    out = np.full(len(panel), np.nan)

    codes, _ = pd.factorize(panel["Date"].to_numpy())
    order = np.argsort(codes, kind="stable")
    cs = codes[order]
    bounds = np.flatnonzero(np.diff(cs)) + 1
    kf = X.shape[1]

    for idx in np.split(order, bounds):
        ff, XX = f[idx], X[idx]
        m = np.isfinite(ff) & np.isfinite(XX).all(axis=1)
        if m.sum() < max(20, kf * 5):
            continue
        Xm = XX[m]
        mu = Xm.mean(axis=0)
        Xc = Xm - mu
        yc = ff[m] - ff[m].mean()
        beta, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
        out[idx] = ff - (ff[m].mean() + (XX - mu) @ beta)
    return out
