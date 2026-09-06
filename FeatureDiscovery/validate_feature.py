"""
validate_feature.py  --  Post-build gate for a candidate feature block.

The check that runs right after generate_feature.py writes a block. It does NOT decide a
feature is "good" -- only whether it is SAFE, DURABLE, and NOVEL enough to be worth a real
(model-level, multi-seed backtest) evaluation. Four stages, cheapest-and-most-decisive first:

  1. LEAKAGE  -- static scan for look-ahead smells + CAUSALITY TEST: recompute on the price
     series TRUNCATED at several cut points; a causal feature's past value[t] must be identical
     with or without future bars. Catches full-sample normalisation / future leakage. For
     machine-generated code a high IC is usually a leak, not alpha -- this catches it.

  2. SIGNAL + OOS DURABILITY  -- pooled across a SAMPLE OF TICKERS (single-ticker IC is noise).
     The history is split IN-SAMPLE (older) vs OUT-OF-SAMPLE (recent). We report IC on each and
     require the OOS sign to SURVIVE and the OOS t-stat to clear the bar. This directly targets
     the project's #1 failure mode: features that look great in-sample and decay out-of-sample.

  3. REDUNDANCY  -- correlate the candidate against the EXISTING feature set (the framework's
     own blocks, cached). A column that is ~a duplicate of something you already compute adds no
     marginal value and only bloats the model. Near-duplicates are rejected.

  4. VERDICT
       FAIL  -- leak, broken, or redundant (>= hard threshold). Reject (quarantine).
       WEAK  -- safe + novel but no column is durable OOS at the bar. Keep, but unproven.
       PASS  -- safe + novel + a column with durable, sign-stable OOS signal. Promote to the
                real gate (marginal contribution in the model + multi-seed backtest).

Deliberately conservative and UNCONDITIONAL: it does not measure marginal value on top of the
existing set beyond a correlation screen, and it is NOT a backtest -- those are the next gates.

USAGE
-----
  python FeatureDiscovery/validate_feature.py FeatureTemplates/tda_delay_embedding_anomaly.py
  python FeatureDiscovery/validate_feature.py <block_name> --n 50 --oos_frac 0.25
  python FeatureDiscovery/validate_feature.py <block_name> --no_redundancy
  python FeatureDiscovery/validate_feature.py <block_name> --refresh_incumbents
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))  # book_target.py

# The report uses ✓/~/✗ badges. On Windows cp1252 (default when stdout is a pipe/redirect) those
# raise UnicodeEncodeError and crash the gate -- including when generate_feature.py runs it with
# verbose=True. Reconfigure to UTF-8 once, mirroring fetchers/common.py.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, Exception):
    pass

ROOT      = Path(__file__).resolve().parent.parent
# repo root must be importable: 3__FeatureFramework.py imports repo helper
# modules (e.g. auxiliary._quiet_progress); without this the redundancy check dies silently
# and every candidate reports maxcorr 0.00.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TEMPLATES = ROOT / "FeatureTemplates"
PRICE_DIR = ROOT / "Data" / "PriceData"
FW_PATH   = ROOT / "3__FeatureFramework.py"
CACHE_PARQUET = ROOT / "Data" / "PaperFeed" / "_incumbent_cache.parquet"
CACHE_META    = ROOT / "Data" / "PaperFeed" / "_incumbent_cache.json"
OHLCV     = {"Date", "Ticker", "Open", "High", "Low", "Close", "Volume"}

LEAK_PATTERNS = [
    (r"\.shift\(\s*-", "negative .shift() (pulls future rows back)"),
    (r"\.iloc\[\s*i\s*\+", "iloc[i+...] forward indexing"),
    (r"\[\s*i\s*\+\s*1", "[i+1] forward indexing"),
    (r"\.diff\(\s*-", "negative .diff()"),
]

# SEC fundamentals can leak in a way the causality test CANNOT see: it only truncates the price
# df, not external data, so a fundamentals value merged on the wrong reference date is identical
# with/without future bars. Fundamentals are lookahead-safe ONLY when reached through
# _fundamentals.as_of (a backward merge on filed_date). These patterns flag a block that bypasses
# that contract -- reads the raw panel, references period_start/period_end (the period the number
# DESCRIBES, not when it was filed), or forward-merges (peeks at the next filing). Hard-fail.
# Code-level signatures (not mere name mentions): a block must reach fundamentals ONLY through
# _fundamentals.as_of. Doing its own file IO, or touching period_start/period_end (which the helper
# hides precisely because they are NOT the date the number became public), or forward-merging, is a
# bypass the causality test cannot see.
SEC_BYPASS_PATTERNS = [
    (r"\.read_parquet\s*\(",  "direct read_parquet (use _fundamentals.as_of, the only sanctioned reader)"),
    (r"\.read_csv\s*\(",      "direct read_csv (use _fundamentals.as_of, the only sanctioned reader)"),
    (r"\bperiod_end\b",   "references period_end (use filed_date via _fundamentals.as_of)"),
    (r"\bperiod_start\b", "references period_start (the period described, not the filing date)"),
    (r"direction\s*=\s*['\"]forward['\"]", "forward merge_asof (peeks at a future filing)"),
]
CAUSALITY_TOL_ATOL = 1e-6
CAUSALITY_TOL_RTOL = 1e-4

# Verdict thresholds
OOS_T_PASS   = 3.0   # |t| on the OUT-OF-SAMPLE slice to PASS (deflated for multi-col + selection)
OOS_T_WEAK   = 2.0
SIGN_AGREE   = 0.6
REDUN_NOVEL  = 0.95  # max |corr| above this == not novel (can't count toward PASS)
REDUN_FAIL   = 0.97  # all columns above this == redundant block, FAIL/quarantine
REDUN_ROWS   = 4000  # subsample rows for the correlation screen (corr is stable; keeps it fast)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def resolve_path(arg: str) -> Path:
    p = Path(arg)
    if p.exists():
        return p
    cand = TEMPLATES / (arg if arg.endswith(".py") else f"{arg}.py")
    if cand.exists():
        return cand
    sys.exit(f"Block not found: {arg}")


def load_block(path: Path):
    source = path.read_text(encoding="utf-8", errors="replace")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "METADATA") or not hasattr(mod, "compute"):
        sys.exit("Block missing METADATA or compute().")
    return mod.METADATA, mod.compute, source


def sample_paths(n: int, seed: int) -> list[Path]:
    paths = sorted(PRICE_DIR.glob("*.parquet"))
    rng = random.Random(seed)
    return rng.sample(paths, min(n, len(paths)))


def load_frames(paths: list[Path]) -> list[pd.DataFrame]:
    out = []
    for p in paths:
        try:
            df = pd.read_parquet(p)
            if len(df) >= 300:
                out.append(df)
        except Exception:
            pass
    return out


def forward_logret(df: pd.DataFrame) -> pd.Series:
    c = df["Close"].astype(float)
    return np.log(c.shift(-1) / c)   # fwd_ret[t] = log(close[t+1]/close[t]); causal target


# Configurable gate target (2026-09-06). Default is the historical next-day log return so every
# ledger row stays comparable; --target touch_win / win_touch / book_ret score the block against
# the event the 3-slot trigger book actually pays for. See FeatureDiscovery/book_target.py.
TARGET = "logret"


def gate_target(df: pd.DataFrame) -> pd.Series:
    if TARGET == "logret":
        return forward_logret(df)
    from book_target import make_target
    return make_target(df, TARGET)


def _spearman(a, b) -> float:
    from scipy.stats import spearmanr
    try:
        return float(spearmanr(a, b)[0])
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# 1. leakage
# ---------------------------------------------------------------------------

def static_scan(source: str) -> list[str]:
    return [desc for pat, desc in LEAK_PATTERNS if re.search(pat, source)]


def sec_bypass_scan(source: str) -> list[str]:
    """Flag fundamentals access that bypasses the lookahead-safe _fundamentals.as_of contract."""
    return [desc for pat, desc in SEC_BYPASS_PATTERNS if re.search(pat, source)]


# ---------------------------------------------------------------------------
# 1b. structural / garbage-code checks (catch bad generated code beyond leakage)
# ---------------------------------------------------------------------------

# Contract libs + a small safe-stdlib whitelist. Anything else (sklearn, statsmodels, torch, ta,
# pandas_ta, requests, ...) means training / heavy deps / external data / non-portable -> FAIL.
ALLOWED_TOP_IMPORTS = {
    "pandas", "numpy", "scipy", "networkx", "math", "typing", "importlib", "pathlib",
    "__future__", "warnings", "collections", "functools", "itertools", "dataclasses", "enum",
}


def import_scan(source: str) -> list[str]:
    """Return imported top-level modules outside the allowlist.

    AST-based: only REAL import statements count. The previous line-regex matched any line
    starting with 'from'/'import' -- including docstring/comment prose like 'from the stock's
    mean ...' or 'from month t-12 ...' -- and wrongly failed valid blocks. We parse the source
    and inspect Import/ImportFrom nodes; if the source doesn't parse, fall back to the regex.
    """
    bad: set[str] = set()
    try:
        import ast as _ast
        tree = _ast.parse(source)
    except Exception:
        for line in source.splitlines():
            m = re.match(r"\s*(?:import|from)\s+([a-zA-Z0-9_\.]+)", line)
            if m:
                top = m.group(1).split(".")[0]
                if top and top not in ALLOWED_TOP_IMPORTS:
                    bad.add(top)
        return sorted(bad)
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Import):
            for n in node.names:
                top = n.name.split(".")[0]
                if top not in ALLOWED_TOP_IMPORTS:
                    bad.add(top)
        elif isinstance(node, _ast.ImportFrom):
            if node.level and not node.module:
                bad.add("." * node.level)          # relative import (none allowed in a block)
            else:
                top = (node.module or "").split(".")[0]
                if top and top not in ALLOWED_TOP_IMPORTS:
                    bad.add(top)
    return sorted(bad)


def _mode_frac(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) == 0:
        return 1.0
    vc = s.value_counts(normalize=True)
    return float(vc.iloc[0]) if len(vc) else 1.0


def structural_check(compute, source, frames, produces) -> tuple[list[str], list[str]]:
    """
    Catch garbage-code edge cases the leakage/signal stages miss. Returns (fail_reasons, warn_notes).
    Hard fails: disallowed imports, compute() crash, row-count/order change, mutated 'sacred' input
    columns, non-numeric or inf output, non-deterministic output. Soft warns: undeclared output columns.
    """
    fails, warns = [], []
    bad = import_scan(source)
    if bad:
        fails.append(f"disallowed import(s) {bad} (allowed: pandas,numpy,scipy,networkx,math,typing)")
    if not frames:
        return fails, warns

    df0 = frames[0]
    base = list(df0.columns)
    try:
        o1 = compute(df0.copy())
    except Exception as exc:
        fails.append(f"compute() raised: {exc}")
        return fails, warns

    if len(o1) != len(df0):
        fails.append(f"row count changed ({len(df0)} -> {len(o1)}) -- must not add/drop rows")
    for c in base:
        if c not in o1.columns:
            fails.append(f"dropped/renamed input column '{c}'")
        elif len(o1) == len(df0) and not df0[c].reset_index(drop=True).equals(
                o1[c].reset_index(drop=True)):
            fails.append(f"modified sacred input column '{c}'")
    for c in produces:
        if c not in o1.columns:
            continue
        s = o1[c]
        if not pd.api.types.is_numeric_dtype(s):
            fails.append(f"non-numeric output column '{c}'")
        elif np.isinf(pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64")).any():
            fails.append(f"produces inf/-inf in '{c}' (unguarded division?)")
    extra = [c for c in o1.columns if c not in base and c not in produces]
    if extra:
        warns.append(f"undeclared output columns not in METADATA.produces: {extra[:6]}")

    # determinism: identical input -> identical output (catches unseeded randomness)
    try:
        o2 = compute(df0.copy())
        for c in produces:
            if c in o1.columns and c in o2.columns and not \
                    o1[c].reset_index(drop=True).equals(o2[c].reset_index(drop=True)):
                fails.append(f"non-deterministic output '{c}' (differs across identical runs)")
                break
    except Exception:
        pass
    return fails, warns


def causality_test(compute, produces, frames, cuts=(0.6, 0.75, 0.9), n_tickers=4) -> list[dict]:
    leaks = []
    for df in frames[:n_tickers]:
        try:
            full = compute(df.copy())
        except Exception as exc:
            leaks.append({"col": "*", "detail": f"compute failed: {exc}", "ticker": "?"})
            continue
        for f in cuts:
            k = int(len(df) * f)
            if k < 60:
                continue
            try:
                trunc = compute(df.iloc[:k].copy())
            except Exception:
                continue
            for col in produces:
                if col not in full.columns or col not in trunc.columns:
                    continue
                a = np.asarray(full[col].iloc[:k], dtype=float)
                b = np.asarray(trunc[col], dtype=float)
                m = min(len(a), len(b))
                a, b = a[:m], b[:m]
                fin = np.isfinite(a) & np.isfinite(b)
                if not fin.any():
                    continue
                ok = np.isclose(a[fin], b[fin], atol=CAUSALITY_TOL_ATOL, rtol=CAUSALITY_TOL_RTOL)
                if not ok.all():
                    leaks.append({"ticker": str(df["Ticker"].iloc[0]) if "Ticker" in df else "?",
                                 "col": col, "cut": f, "max_diff": float(np.max(np.abs(a[fin] - b[fin])))})
    return leaks


# ---------------------------------------------------------------------------
# 2. signal + OOS durability
# ---------------------------------------------------------------------------

def build_panel(compute, frames, produces) -> pd.DataFrame:
    parts = []
    for df in frames:
        try:
            r = compute(df.copy())
        except Exception:
            continue
        r = r.copy()
        r["__fwd"] = gate_target(r)
        cols = ["Date", "Ticker", "__fwd"] + [c for c in produces if c in r.columns]
        parts.append(r[[c for c in cols if c in r.columns]])
    if not parts:
        return pd.DataFrame()
    panel = pd.concat(parts, ignore_index=True)
    panel["Date"] = pd.to_datetime(panel["Date"], errors="coerce")
    return panel


def topdecile_lift(sub: pd.DataFrame, col: str, q: float = 0.9, min_per_day: int = 10):
    spreads = []
    for _, g in sub.groupby("Date"):
        if len(g) < min_per_day:
            continue
        top = g.loc[g[col] >= g[col].quantile(q), "__fwd"].mean()
        bot = g.loc[g[col] <= g[col].quantile(1 - q), "__fwd"].mean()
        if np.isfinite(top) and np.isfinite(bot):
            spreads.append(top - bot)
    if len(spreads) < 20:
        return None
    s = np.array(spreads)
    return {"daily_spread_pct": float(s.mean()) * 100,
            "t": float(s.mean() / (s.std(ddof=1) / math.sqrt(len(s)) + 1e-12))}


def column_signal(panel: pd.DataFrame, col: str, cutoff) -> dict:
    sub = panel[["Date", "Ticker", col, "__fwd"]].dropna(subset=[col, "__fwd"])
    out = {"col": col, "n": len(sub)}
    if len(sub) < 300:
        out["note"] = "too few obs"
        return out

    is_mask  = sub["Date"] < cutoff
    oos_mask = ~is_mask
    ic_is  = _spearman(sub.loc[is_mask, col],  sub.loc[is_mask, "__fwd"])  if is_mask.sum()  > 100 else float("nan")
    ic_oos = _spearman(sub.loc[oos_mask, col], sub.loc[oos_mask, "__fwd"]) if oos_mask.sum() > 100 else float("nan")
    n_oos  = int(oos_mask.sum())

    out["ic_is"]  = ic_is
    out["ic_oos"] = ic_oos
    out["t_oos"]  = ic_oos * math.sqrt(n_oos) if np.isfinite(ic_oos) else float("nan")
    out["ic_all"] = _spearman(sub[col], sub["__fwd"])
    out["durable"] = bool(np.isfinite(ic_is) and np.isfinite(ic_oos)
                          and np.sign(ic_is) == np.sign(ic_oos))

    per = []
    for _, g in sub.groupby("Ticker"):
        if len(g) >= 50:
            per.append(_spearman(g[col], g["__fwd"]))
    per = np.array([x for x in per if np.isfinite(x)])
    ref = ic_oos if np.isfinite(ic_oos) else out["ic_all"]
    out["sign_agree"] = float(np.mean(np.sign(per) == np.sign(ref))) if len(per) else float("nan")
    out["topdecile"] = topdecile_lift(sub, col)
    return out


# ---------------------------------------------------------------------------
# 3. redundancy vs the existing feature set
# ---------------------------------------------------------------------------

def _load_framework():
    spec = importlib.util.spec_from_file_location("framework", FW_PATH)
    fw = importlib.util.module_from_spec(spec)
    import warnings, contextlib, io
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        spec.loader.exec_module(fw)
    return fw


def incumbent_matrix(inc_paths: list[Path], refresh: bool) -> pd.DataFrame:
    """Pooled (Date,Ticker)+all framework feature columns for a small fixed ticker set. Cached."""
    key = hashlib.md5("|".join(sorted(p.stem for p in inc_paths)).encode()).hexdigest()[:10]
    if (not refresh) and CACHE_PARQUET.exists() and CACHE_META.exists():
        try:
            meta = json.loads(CACHE_META.read_text())
            if meta.get("key") == key:
                return pd.read_parquet(CACHE_PARQUET)
        except Exception:
            pass

    fw = _load_framework()
    parts = []
    for p in inc_paths:
        try:
            df = pd.read_parquet(p)
            if "Date" not in df.columns and df.index.name == "Date":
                df = df.reset_index()
            import warnings, contextlib, io
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                warnings.simplefilter("ignore")
                res, _ = fw.run_pipeline_timed(df, verbose=False)
            res = res.copy()
            res["Date"] = pd.to_datetime(res["Date"], errors="coerce")
            num = res.select_dtypes(include="number")
            feat = [c for c in num.columns if c not in OHLCV]
            parts.append(pd.concat([res[["Date", "Ticker"]], num[feat]], axis=1))
        except Exception:
            continue
    inc = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    CACHE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    if not inc.empty:
        inc.to_parquet(CACHE_PARQUET, index=False)
        CACHE_META.write_text(json.dumps(
            {"key": key, "tickers": [p.stem for p in inc_paths], "n_cols": inc.shape[1],
             "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
    return inc


def redundancy(compute, produces, inc_paths, inc: pd.DataFrame) -> dict:
    """Max |spearman| of each produced column vs every incumbent column (+ which one)."""
    if inc.empty:
        return {}
    # candidate on the SAME tickers, pooled + aligned to incumbent rows
    cparts = []
    for p in inc_paths:
        try:
            df = pd.read_parquet(p)
            if "Date" not in df.columns and df.index.name == "Date":
                df = df.reset_index()
            r = compute(df.copy())
            r["Date"] = pd.to_datetime(r["Date"], errors="coerce")
            cparts.append(r[["Date", "Ticker"] + [c for c in produces if c in r.columns]])
        except Exception:
            continue
    if not cparts:
        return {}
    cand = pd.concat(cparts, ignore_index=True)
    merged = cand.merge(inc, on=["Date", "Ticker"], how="inner", suffixes=("", "_inc"))
    inc_cols = [c for c in inc.columns if c not in ("Date", "Ticker") and c not in produces]
    if not inc_cols or merged.empty:
        return {}
    if len(merged) > REDUN_ROWS:
        merged = merged.sample(REDUN_ROWS, random_state=0)

    ranks_inc = merged[inc_cols].rank()
    out = {}
    for col in produces:
        if col not in merged.columns:
            continue
        rc = merged[col].rank()
        best_corr, best_col = 0.0, None
        for ic_col in inc_cols:
            pair = pd.concat([rc, ranks_inc[ic_col]], axis=1).dropna()
            if len(pair) < 200:
                continue
            c = pair.iloc[:, 0].corr(pair.iloc[:, 1])   # pearson on ranks == spearman
            if np.isfinite(c) and abs(c) > abs(best_corr):
                best_corr, best_col = float(c), ic_col
        out[col] = {"max_corr": best_corr, "match": best_col}
    return out


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------

def decide(leaks, healthy, signals, redun, bypass=None, structural=None) -> tuple[str, list[str]]:
    reasons = []
    if structural:
        reasons.append("GARBAGE: " + structural[0]
                       + (f"  (+{len(structural)-1} more)" if len(structural) > 1 else ""))
        return "FAIL", reasons
    if bypass:
        reasons.append("SEC-BYPASS: fundamentals reached outside the lookahead-safe "
                       f"_fundamentals.as_of contract -- {bypass[0]}")
        return "FAIL", reasons
    if leaks:
        ex = leaks[0]
        reasons.append(f"LEAK: causality test failed ({ex.get('col')} @cut {ex.get('cut')}, "
                       f"max_diff {ex.get('max_diff', float('nan')):.2e} on {ex.get('ticker')})")
        return "FAIL", reasons
    if not healthy:
        reasons.append("no produced column is populated and non-constant")
        return "FAIL", reasons

    def corr(col):
        return abs(redun.get(col, {}).get("max_corr", 0.0))

    # redundant block: every produced column duplicates an existing feature
    if redun and all(corr(c) >= REDUN_FAIL for c in healthy):
        worst = max(healthy, key=corr)
        m = redun.get(worst, {}).get("match")
        reasons.append(f"REDUNDANT: all columns ~duplicate existing features "
                       f"(e.g. {worst} vs {m}, corr {corr(worst):.2f})")
        return "FAIL", reasons

    novel = [s for s in signals if corr(s["col"]) < REDUN_NOVEL]
    promising = [s for s in novel
                 if abs(s.get("t_oos", 0) or 0) >= OOS_T_PASS and s.get("durable")
                 and (s.get("sign_agree") or 0) >= SIGN_AGREE]
    if promising:
        best = max(promising, key=lambda s: abs(s["t_oos"]))
        reasons.append(f"PASS: '{best['col']}' OOS IC={best['ic_oos']:+.4f} t_oos={best['t_oos']:+.1f} "
                       f"(IS {best['ic_is']:+.4f}, sign-agree {best['sign_agree']:.0%}, "
                       f"maxcorr {corr(best['col']):.2f}) -- durable, sign-stable, novel")
        return "PASS", reasons

    near = [s for s in novel if abs(s.get("t_oos", 0) or 0) >= OOS_T_WEAK and s.get("durable")]
    if near:
        best = max(near, key=lambda s: abs(s["t_oos"]))
        reasons.append(f"WEAK: best novel '{best['col']}' OOS t={best['t_oos']:+.1f}, durable but "
                       f"below the |t|>={OOS_T_PASS:.0f} bar -- raise --n or treat as unproven")
    else:
        flips = [s for s in novel if not s.get("durable") and np.isfinite(s.get("ic_is", np.nan))]
        if flips and any(abs(s.get("ic_is", 0)) > 0.02 for s in flips):
            reasons.append("WEAK: in-sample signal present but sign FLIPS / dies out-of-sample "
                           "(the decay trap) -- not durable")
        else:
            reasons.append("WEAK: no novel column clears |t_oos|>=2 -- indistinguishable from noise here")
    return "WEAK", reasons


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def validate_block(path: Path, n: int = 40, seed: int = 13, oos_frac: float = 0.25,
                   do_redundancy: bool = True, inc_n: int = 6, refresh: bool = False,
                   verbose: bool = True) -> dict:
    metadata, compute, source = load_block(path)
    produces = list(metadata.get("produces", []))
    requires = set(metadata.get("requires", []))
    report: dict = {"block": path.stem, "produces": produces, "verdict": "FAIL", "reasons": []}

    if not requires.issubset(OHLCV):
        report["reasons"].append(f"requires non-OHLCV columns {sorted(requires - OHLCV)} -- "
                                 "validate inside the framework instead")
        if verbose:
            _print(report, [], [], [], {})
        return report

    frames = load_frames(sample_paths(n, seed))
    if not frames:
        report["reasons"].append("no usable price frames")
        return report

    # speed probe: single-core compute time on one ticker (~700 rows). Budget is <100ms/file.
    ms = float("nan")
    try:
        _tc = time.perf_counter(); compute(frames[0].copy()); ms = (time.perf_counter() - _tc) * 1000.0
    except Exception:
        pass

    t0 = time.perf_counter()
    static = static_scan(source)
    bypass = sec_bypass_scan(source)
    struct_fail, struct_warn = structural_check(compute, source, frames, produces)
    leaks = causality_test(compute, produces, frames)
    panel = build_panel(compute, frames, produces)

    cutoff = None
    if not panel.empty:
        cutoff = panel["Date"].quantile(1 - oos_frac)

    healthy, signals = [], []
    if not panel.empty:
        for col in produces:
            if col not in panel.columns:
                continue
            tail = panel[col].iloc[200:]
            # non-constant AND not near-constant (a 99.9%-one-value flag carries ~no info)
            if tail.nunique(dropna=True) > 1 and _mode_frac(tail) < 0.99:
                healthy.append(col)
                signals.append(column_signal(panel, col, cutoff))

    redun = {}
    if do_redundancy and healthy:
        try:
            inc_paths = sample_paths(inc_n, seed=999)   # fixed set -> shared incumbent cache
            inc = incumbent_matrix(inc_paths, refresh)
            redun = redundancy(compute, produces, inc_paths, inc)
        except Exception as exc:
            report.setdefault("notes", []).append(f"redundancy skipped: {exc}")

    notes = list(struct_warn) + report.get("notes", [])   # keep any earlier note (e.g. redundancy skipped)
    susp = [s for s in signals if abs(s.get("ic_all", 0) or 0) > 0.25]
    if susp:
        notes.append(f"suspicious |IC|>0.25 ({susp[0]['col']} {susp[0].get('ic_all', 0):+.3f}) -- "
                     "verify it is not a leak the causality test missed")

    verdict, reasons = decide(leaks, healthy, signals, redun, bypass, struct_fail)
    report.update({"verdict": verdict, "reasons": reasons, "static_smells": static,
                   "sec_bypass": bypass, "structural": struct_fail, "ms": ms,
                   "leaks": leaks, "signals": signals, "redun": redun, "notes": notes,
                   "n_tickers": len(frames), "oos_cutoff": str(cutoff)[:10] if cutoff is not None else "?",
                   "secs": time.perf_counter() - t0})
    if verbose:
        _print(report, static, leaks, signals, redun)
    return report


def _print(report, static, leaks, signals, redun):
    W = 92
    badge = {"PASS": "PASS  ✓", "WEAK": "WEAK  ~", "FAIL": "FAIL  ✗"}.get(report["verdict"], report["verdict"])
    print("\n" + "=" * W)
    print(f"  VALIDATE  {report['block']}   ->   {badge}"
          f"        (target {TARGET}, OOS cutoff {report.get('oos_cutoff','?')})")
    print("=" * W)
    if report.get("structural"):
        print("  [GARBAGE] " + "; ".join(report["structural"][:4]))
    if static:
        print("  [static look-ahead smells] " + "; ".join(static))
    if report.get("sec_bypass"):
        print("  [SEC bypass] " + "; ".join(report["sec_bypass"]))
    if leaks:
        print(f"  [LEAK] causality failed on {len(leaks)} combos: "
              + ", ".join(f"{lk.get('col')}@{lk.get('cut')}" for lk in leaks[:5]))
    else:
        print("  [leakage] causality PASSED (past values stable under truncation)")
    if signals:
        print(f"  {'column':<30} {'IC_is':>8} {'IC_oos':>8} {'t_oos':>7} {'sign%':>6} "
              f"{'maxcorr':>8} {'~dup of':<18}")
        print("  " + "-" * (W - 2))
        for s in signals:
            if "ic_all" not in s:
                print(f"  {s['col']:<30}  {s.get('note','-')}")
                continue
            rc = redun.get(s["col"], {})
            mc = rc.get("max_corr", float("nan"))
            match = (rc.get("match") or "")[:18]
            print(f"  {s['col']:<30} {s.get('ic_is', float('nan')):>+8.4f} "
                  f"{s.get('ic_oos', float('nan')):>+8.4f} {s.get('t_oos', float('nan')):>+7.1f} "
                  f"{(s.get('sign_agree') or float('nan'))*100:>5.0f}% {mc:>+8.2f}  {match:<18}")
    print("  " + "-" * (W - 2))
    for r in report["reasons"]:
        print(f"  -> {r}")
    for note in report.get("notes", []):
        print(f"  (note) {note}")
    _ms = report.get("ms", float("nan"))
    print(f"  ({report.get('n_tickers','?')} tickers, {report.get('secs',0):.1f}s gate, "
          f"compute {_ms:.0f}ms/ticker)")
    print("=" * W + "\n")


# ---------------------------------------------------------------------------
# batch  --  gate a whole glob of candidates, rank by IC, quarantine FAIL/SLOW
# ---------------------------------------------------------------------------

def batch(glob_pat: str, n: int, oos_frac: float, inc_n: int,
          max_ms: float = 150.0, keep_fail: bool = False) -> None:
    import csv as _csv
    from collections import Counter
    # rebuild incumbents against the CURRENT (e.g. post-cull) feature set
    CACHE_PARQUET.unlink(missing_ok=True)
    CACHE_META.unlink(missing_ok=True)

    files = sorted(TEMPLATES.glob(glob_pat))
    if not files:
        sys.exit(f"no candidates match FeatureTemplates/{glob_pat}")
    reject = ROOT / "FeatureDiscovery" / "_rejected"
    print(f"Batch gate: {len(files)} candidates ({glob_pat})\n")

    rows = []
    for i, f in enumerate(files, 1):
        try:
            rep = validate_block(f, n=n, oos_frac=oos_frac, inc_n=inc_n, verbose=False)
        except Exception as exc:
            print(f"  [{i:>2}/{len(files)}] ERROR  {f.name}: {str(exc)[:70]}")
            rows.append({"block": f.stem.lstrip("_"), "verdict": "ERROR", "ic": "", "oos_ic": "",
                         "maxcorr": "", "ms": "", "note": str(exc)[:90]})
            continue
        sigs = rep.get("signals", [])
        ic   = max((abs(s.get("ic_all", 0) or 0) for s in sigs), default=0.0)
        oos  = max((abs(s.get("ic_oos", 0) or 0) for s in sigs
                    if abs(s.get("t_oos", 0) or 0) >= OOS_T_WEAK), default=0.0)
        maxc = max((abs(v.get("max_corr", 0) or 0) for v in rep.get("redun", {}).values()), default=0.0)
        msv  = rep.get("ms", float("nan"))
        verdict = rep["verdict"]
        too_slow = (msv == msv) and msv > max_ms          # NaN-safe
        quarantined = False
        if (verdict == "FAIL" or too_slow) and not keep_fail:
            reject.mkdir(parents=True, exist_ok=True)
            f.replace(reject / f.name)
            quarantined = True
        tag = "SLOW" if (too_slow and verdict != "FAIL") else verdict
        star = " *" if ic >= 0.05 else ""
        print(f"  [{i:>2}/{len(files)}] {tag:<5} {f.stem.lstrip('_'):<44} "
              f"IC={ic:+.4f}{star}  {msv:.0f}ms" + ("  [quarantined]" if quarantined else ""))
        rows.append({"block": f.stem.lstrip("_"), "verdict": tag, "ic": round(ic, 4),
                     "oos_ic": round(oos, 4), "maxcorr": round(maxc, 3),
                     "ms": round(msv, 1) if msv == msv else "",
                     "note": (rep.get("reasons") or [""])[0][:90]})

    led = ROOT / "Data" / "PaperFeed" / "battery_results.csv"
    with open(led, "w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    surv = sorted([r for r in rows if r["verdict"] in ("PASS", "WEAK")],
                  key=lambda r: -(r["ic"] if isinstance(r["ic"], float) else 0.0))
    cnt = Counter(r["verdict"] for r in rows)
    W = 86
    print("\n" + "=" * W)
    print(f"  BATTERY   {dict(cnt)}")
    print("=" * W)
    print(f"  {'#':>3} {'verdict':<6} {'IC':>8} {'OOS_IC':>8} {'maxcorr':>8} {'ms':>6}  block")
    print("  " + "-" * (W - 2))
    for j, r in enumerate(surv, 1):
        st = "*" if (isinstance(r["ic"], float) and r["ic"] >= 0.05) else " "
        print(f"  {j:>3} {r['verdict']:<6} {r['ic']:>+8} {r['oos_ic']:>+8} "
              f"{r['maxcorr']:>8} {str(r['ms']):>6} {st} {r['block']}")
    over = [r for r in surv if isinstance(r["ic"], float) and r["ic"] >= 0.05]
    print("  " + "-" * (W - 2))
    print(f"  survivors {len(surv)}  |  IC>=0.05: {len(over)}  |  "
          f"quarantined {sum(v for k, v in cnt.items() if k in ('FAIL', 'SLOW', 'ERROR'))}")
    print(f"  ledger -> {led}")
    print("=" * W + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Post-build validation gate for a feature block")
    ap.add_argument("block", nargs="?", help="block file/name (omit when using --batch)")
    ap.add_argument("--batch", metavar="GLOB",
                    help="gate every FeatureTemplates/<GLOB> candidate, rank by IC, quarantine FAIL/SLOW")
    ap.add_argument("--max_ms", type=float, default=150.0,
                    help="quarantine candidates whose compute exceeds this (default 150ms)")
    ap.add_argument("--keep_fail", action="store_true", help="report but do not quarantine FAIL/SLOW")
    ap.add_argument("--n", type=int, default=40, help="ticker sample for signal/OOS (default 40)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--oos_frac", type=float, default=0.25, help="fraction of recent history held out (default 0.25)")
    ap.add_argument("--inc_n", type=int, default=6, help="tickers for the incumbent redundancy matrix")
    ap.add_argument("--no_redundancy", action="store_true")
    ap.add_argument("--refresh_incumbents", action="store_true", help="rebuild the incumbent cache")
    ap.add_argument("--target", default="logret",
                    choices=["logret", "ret5", "hit8", "touch", "touch_win", "win_touch", "book_ret", "book_ret_all"],
                    help="gate target: logret (default, historical ledger basis) or a book-rule outcome "
                         "from FeatureDiscovery/book_target.py (touch_win = P(touch AND win), the book label)")
    args = ap.parse_args()
    global TARGET
    TARGET = args.target

    if args.batch:
        batch(args.batch, n=args.n, oos_frac=args.oos_frac, inc_n=args.inc_n,
              max_ms=args.max_ms, keep_fail=args.keep_fail)
        return

    if not args.block:
        sys.exit("provide a block name/path, or use --batch GLOB")
    report = validate_block(resolve_path(args.block), n=args.n, seed=args.seed,
                            oos_frac=args.oos_frac, do_redundancy=not args.no_redundancy,
                            inc_n=args.inc_n, refresh=args.refresh_incumbents)
    sys.exit(0 if report["verdict"] in ("PASS", "WEAK") else 1)


if __name__ == "__main__":
    main()
