"""
__diagnostics.py  --  Feature pipeline diagnostics tool.

Auto-skipped by the framework (leading __ keeps it out of block discovery).

Usage
-----
  python FeatureTemplates/__diagnostics.py [--n 10] [--seed 42] [--exclude block1 block2]

Reports
-------
1. LOC per template file (non-__ files only)
2. Per-block speed benchmarks across N random tickers
   Key metric: µs per feature value  (block_ms * 1000 / (n_rows * n_produces))
               tells you the true cost of each output cell, independent of
               how many features the block happens to produce.
3. IC analysis — Spearman rank IC of every produced column vs next-day log-return
   GREAT: |IC| >= 0.05   ok: |IC| >= 0.01   meh: |IC| < 0.01
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap: import the framework from the parent directory
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

_spec = importlib.util.spec_from_file_location(
    "framework", ROOT / "3__FeatureFramework.py"
)
_fw = importlib.util.module_from_spec(_spec)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _spec.loader.exec_module(_fw)

run_pipeline_timed = _fw.run_pipeline_timed
discover_blocks    = _fw.discover_blocks
PRICE_DATA_DIR     = _fw.PRICE_DATA_DIR
TEMPLATES_DIR      = _fw.TEMPLATES_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _loc_table() -> list[tuple[str, int]]:
    """(filename, line_count) for every non-__ .py in FeatureTemplates."""
    rows = []
    for p in sorted(TEMPLATES_DIR.glob("*.py")):
        if p.name.startswith("_"):
            continue
        lines = sum(1 for _ in p.open(encoding="utf-8", errors="ignore"))
        rows.append((p.name, lines))
    return rows


def _fmt_ms(ms: float) -> str:
    if ms >= 1_000:
        return f"{ms / 1_000:.2f}s"
    if ms >= 1:
        return f"{ms:.1f}ms"
    return f"{ms * 1_000:.0f}us"


def _bar(frac: float, width: int = 20) -> str:
    filled = round(frac * width)
    return "#" * filled + "." * (width - filled)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Feature pipeline diagnostics")
    parser.add_argument("--n",       type=int, default=10,
                        help="Number of random tickers to benchmark (default 10)")
    parser.add_argument("--seed",    type=int, default=42,
                        help="Random seed for ticker sampling")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="Block names to exclude from the benchmark run")
    parser.add_argument("--top_n",   type=int, default=25,
                        help="Top-N columns to show in IC ranking (default 25)")
    args = parser.parse_args()

    W   = 84
    SEP = "=" * W

    # -----------------------------------------------------------------------
    # Section 1 — LOC
    # -----------------------------------------------------------------------
    print()
    print(SEP)
    print("  Feature Template LOC  (non-__ files)")
    print(SEP)

    loc_rows   = _loc_table()
    total_loc  = sum(n for _, n in loc_rows)
    name_width = max((len(r[0]) for r in loc_rows), default=10)

    for fname, n in loc_rows:
        print(f"  {fname:<{name_width}}  {n:>5} lines")

    print("  " + "-" * (W - 2))
    print(f"  {'TOTAL':<{name_width}}  {total_loc:>5} lines  "
          f"({len(loc_rows)} active template files)")

    # -----------------------------------------------------------------------
    # Section 2 — Benchmark
    # -----------------------------------------------------------------------
    all_paths = sorted(PRICE_DATA_DIR.glob("*.parquet"))
    if not all_paths:
        print(f"\n[!] No parquet files found in {PRICE_DATA_DIR}")
        return

    rng = random.Random(args.seed)
    sample_paths = rng.sample(all_paths, min(args.n, len(all_paths)))
    tickers = [p.stem for p in sample_paths]

    # Discover blocks once (suppress the duplicate-column [WARN] spam)
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        blocks = discover_blocks()

    # n_produces per block (from METADATA)
    n_produces = {
        name: len(info["meta"].get("produces", []))
        for name, info in blocks.items()
    }

    print()
    print(SEP)
    print(f"  Benchmark  (N={len(tickers)}, seed={args.seed})")
    print(SEP)
    print(f"  Tickers : {' '.join(tickers)}")
    print()

    # Accumulate per-block: [elapsed_ms, ...]
    block_times:   dict[str, list[float]]     = {}
    block_rows:    dict[str, list[int]]       = {}   # rows processed per call
    results_store: dict[str, pd.DataFrame]   = {}   # ticker -> full feature df (for IC)
    total_rows   = 0
    total_fvals  = 0
    wall_start   = time.perf_counter()
    errors: list[str] = []

    for path in sample_paths:
        ticker = path.stem
        try:
            df = pd.read_parquet(path)
            if "Date" not in df.columns and df.index.name == "Date":
                df = df.reset_index()
        except Exception as exc:
            errors.append(f"{ticker}: load error — {exc}")
            continue

        n_rows = len(df)

        try:
            # suppress per-ticker [WARN] repeats (stdout) and stderr noise
            with warnings.catch_warnings(), \
                 contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                warnings.simplefilter("ignore")
                result_df, timing = run_pipeline_timed(df, exclude=args.exclude, verbose=False)
        except Exception as exc:
            errors.append(f"{ticker}: pipeline error - {exc}")
            continue

        total_rows += n_rows
        results_store[ticker] = result_df

        for name, elapsed_s in timing["blocks"].items():
            ms = elapsed_s * 1_000
            block_times.setdefault(name, []).append(ms)
            block_rows.setdefault(name, []).append(n_rows)

        # feature values = rows × number of feature columns produced this run
        fv = sum(
            n_rows * n_produces.get(name, 0)
            for name in timing["blocks"]
        )
        total_fvals += fv

    wall_ms = (time.perf_counter() - wall_start) * 1_000

    if errors:
        for e in errors:
            print(f"  [WARN] {e}")
        print()

    if not block_times:
        print("  No results collected — all tickers failed.")
        return

    # ---- build ranked table -------------------------------------------------
    # rank by mean time descending (slowest first)
    sorted_blocks = sorted(
        block_times.keys(),
        key=lambda n: np.mean(block_times[n]),
        reverse=True,
    )

    total_block_ms = sum(np.sum(v) for v in block_times.values())

    # column widths
    bname_w = max(len(n) for n in sorted_blocks)
    bname_w = max(bname_w, 5)   # min "Block"

    header = (
        f"  {'Block':<{bname_w}}  {'Calls':>5}  {'Avg':>8}  "
        f"{'Std':>8}  {'us/fval':>8}  {'Share':>6}  Hotness"
    )
    print(header)
    print("  " + "-" * (W - 2))

    for name in sorted_blocks:
        times = block_times[name]
        rows  = block_rows[name]
        calls = len(times)
        avg_ms   = float(np.mean(times))
        std_ms   = float(np.std(times))
        avg_rows = float(np.mean(rows))
        np_    = n_produces.get(name, 1)

        # µs per feature value: (avg_ms * 1000) / (avg_rows * np_)
        us_per_fval = (avg_ms * 1_000) / max(avg_rows * np_, 1)

        share = np.sum(times) / max(total_block_ms, 1)
        bar   = _bar(share)

        print(
            f"  {name:<{bname_w}}  {calls:>5}  {_fmt_ms(avg_ms):>8}  "
            f"{_fmt_ms(std_ms):>8}  {us_per_fval:>7.2f}us  "
            f"{share * 100:>5.1f}%  {bar}"
        )

    print("  " + "-" * (W - 2))

    # total row
    total_sum_ms = sum(np.sum(v) for v in block_times.values())
    print(
        f"  {'ALL BLOCKS':<{bname_w}}  "
        f"  {'':8}  {'':8}  "
        f"  {'100.0%':>6}  wall: {_fmt_ms(wall_ms)}"
    )

    # ---- summary ------------------------------------------------------------
    throughput = total_fvals / max(wall_ms / 1_000, 1e-9)

    print()
    print(f"  Summary")
    print(f"  {'-' * 40}")
    print(f"  Tickers tested   : {len(block_times.get(sorted_blocks[0], [1])):>6}  "
          f"(of {len(sample_paths)} sampled)")
    print(f"  Total rows       : {total_rows:>9,}")
    print(f"  Total fvals      : {total_fvals:>9,}   "
          f"(rows x features, summed across tickers)")
    print(f"  Pipeline wall    : {_fmt_ms(wall_ms):>9}   "
          f"(incl. parquet load)")
    print(f"  Throughput       : {throughput:>9,.0f} fval/s")
    print(f"  Avg per ticker   : {_fmt_ms(wall_ms / max(len(sample_paths), 1)):>9}")

    # slowest 3
    top3 = sorted_blocks[:3]
    pcts = [
        np.sum(block_times[n]) / max(total_sum_ms, 1) * 100
        for n in top3
    ]
    print()
    print(f"  Hottest blocks:")
    for name, pct in zip(top3, pcts):
        avg_ms = float(np.mean(block_times[name]))
        np_ = n_produces.get(name, 1)
        avg_rows = float(np.mean(block_rows[name]))
        us_pf = (avg_ms * 1_000) / max(avg_rows * np_, 1)
        print(f"    {name:<{bname_w}}  {pct:5.1f}% of block time  "
              f"avg {_fmt_ms(avg_ms)}  {us_pf:.2f}us/fval")

    print()
    print(SEP)
    print()

    # -----------------------------------------------------------------------
    # Section 3 — IC Analysis
    # IC = Spearman rank correlation between feature[t] and next-day log-return[t]
    # Pooled across all sampled tickers.
    # |IC| >= 0.05  -> GREAT   |IC| >= 0.01 -> ok   |IC| < 0.01 -> meh
    # -----------------------------------------------------------------------
    try:
        from scipy import stats as scipy_stats
    except ImportError:
        print("  [skip] IC analysis — scipy not available (pip install scipy)")
        print()
        print(SEP)
        print()
        return

    print()
    print(SEP)
    print(f"  IC Analysis  (target: next-day log-return, Spearman, N={len(results_store)} tickers pooled)")
    print(SEP)
    print("  Thresholds:  |IC| >= 0.05 -> GREAT   |IC| >= 0.01 -> ok   |IC| < 0.01 -> meh")
    print()

    # Build per-ticker target series (next-day log-return, aligned to feature date)
    feat_frames: list[pd.DataFrame] = []
    targ_series: list[pd.Series]    = []

    for _ticker, rdf in results_store.items():
        if "Close" not in rdf.columns:
            continue
        rdf = rdf.loc[:, ~rdf.columns.duplicated(keep="last")]
        close  = rdf["Close"].astype(float)
        logret = np.log(close / close.shift(1))
        target = logret.shift(-1)          # feature[t] predicts return[t+1]
        feat_frames.append(rdf)
        targ_series.append(target)

    if not feat_frames:
        print("  [!] No Close column found in results — cannot compute IC.")
        print()
        print(SEP)
        print()
        return

    # Map each produced column to its owning block (last writer wins)
    produced_cols: set[str]   = set()
    block_for_col: dict[str, str] = {}
    for _bname, info in blocks.items():
        for col in info["meta"].get("produces", []):
            produced_cols.add(col)
            block_for_col[col] = _bname

    # Compute Spearman IC per column across pooled observations
    ic_results: dict[str, tuple[float, int]] = {}   # col -> (ic, n_obs)

    for col in produced_cols:
        f_parts: list[np.ndarray] = []
        t_parts: list[np.ndarray] = []
        for fdf, tgt in zip(feat_frames, targ_series):
            if col not in fdf.columns:
                continue
            pair = pd.DataFrame({"f": fdf[col], "t": tgt}).dropna()
            if len(pair) < 20:
                continue
            f_parts.append(pair["f"].values)
            t_parts.append(pair["t"].values)
        if not f_parts:
            continue
        f_all = np.concatenate(f_parts)
        t_all = np.concatenate(t_parts)
        if len(f_all) < 30:
            continue
        try:
            corr, _ = scipy_stats.spearmanr(f_all, t_all)
            if np.isfinite(corr):
                ic_results[col] = (float(corr), len(f_all))
        except Exception:
            pass

    if not ic_results:
        print("  No IC results computed.")
        print()
        print(SEP)
        print()
        return

    def _grade(ic: float) -> str:
        a = abs(ic)
        if a >= 0.05:
            return "GREAT"
        if a >= 0.01:
            return "ok"
        return "meh"

    # ---- Per-block summary --------------------------------------------------
    block_abs_ics: dict[str, list[float]] = {}
    for col, (ic, _) in ic_results.items():
        bn = block_for_col.get(col, "?")
        block_abs_ics.setdefault(bn, []).append(abs(ic))

    sorted_blocks_ic = sorted(
        block_abs_ics.items(),
        key=lambda x: float(np.median(x[1])),
        reverse=True,
    )

    bw2 = max((len(n) for n in block_abs_ics), default=5)
    bw2 = max(bw2, 5)

    print(f"  {'Block':<{bw2}}  {'Cols':>4}  {'GREAT':>5}  {'ok':>4}  {'meh':>4}  "
          f"{'Med|IC|':>7}  {'Max|IC|':>7}")
    print("  " + "-" * (W - 2))

    for bn, abs_ics in sorted_blocks_ic:
        n_g = sum(1 for x in abs_ics if x >= 0.05)
        n_o = sum(1 for x in abs_ics if 0.01 <= x < 0.05)
        n_m = sum(1 for x in abs_ics if x < 0.01)
        print(
            f"  {bn:<{bw2}}  {len(abs_ics):>4}  {n_g:>5}  {n_o:>4}  {n_m:>4}  "
            f"{float(np.median(abs_ics)):>7.4f}  {float(np.max(abs_ics)):>7.4f}"
        )

    print()

    # ---- Top-N individual columns -------------------------------------------
    sorted_cols = sorted(ic_results.items(), key=lambda x: abs(x[1][0]), reverse=True)
    top_n = args.top_n

    cw = max((len(c) for c in ic_results), default=10)
    cw = min(max(cw, 6), 58)   # cap width for readability

    print(f"  Top {top_n} columns by |IC|:")
    print(f"  {'Rank':>4}  {'Column':<{cw}}  {'Block':<{bw2}}  {'IC':>8}  {'|IC|':>5}  Grade")
    print("  " + "-" * (W - 2))

    for rank, (col, (ic, n_obs)) in enumerate(sorted_cols[:top_n], 1):
        bn    = block_for_col.get(col, "?")
        grade = _grade(ic)
        disp  = col if len(col) <= cw else col[:cw - 2] + ".."
        print(
            f"  {rank:>4}  {disp:<{cw}}  {bn:<{bw2}}  "
            f"{ic:>+8.4f}  {abs(ic):>5.3f}  {grade}"
        )

    # ---- Distribution summary -----------------------------------------------
    n_great = sum(1 for _, (ic, _) in ic_results.items() if abs(ic) >= 0.05)
    n_ok    = sum(1 for _, (ic, _) in ic_results.items() if 0.01 <= abs(ic) < 0.05)
    n_meh   = sum(1 for _, (ic, _) in ic_results.items() if abs(ic) < 0.01)
    total_c = len(ic_results)

    print()
    print(f"  Distribution ({total_c} columns):  "
          f"{n_great} GREAT ({n_great * 100 // total_c}%)  "
          f"  {n_ok} ok ({n_ok * 100 // total_c}%)  "
          f"  {n_meh} meh ({n_meh * 100 // total_c}%)")
    if total_c > top_n:
        print(f"  (use --top_n {total_c} to see all columns)")

    print()
    print(SEP)
    print()


if __name__ == "__main__":
    main()
