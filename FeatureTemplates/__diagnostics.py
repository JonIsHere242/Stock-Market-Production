"""
__diagnostics.py  --  Feature pipeline diagnostics tool.

Auto-skipped by the framework (leading __ keeps it out of block discovery).

Usage
-----
  python FeatureTemplates/__diagnostics.py [--n 10] [--seed 42] [--exclude block1 block2]
  python FeatureTemplates/__diagnostics.py --candidates   # also benchmark single-_ candidates

Reports
-------
1. LOC per template file (non-__ files only)
2. Per-block speed benchmarks across N random tickers
   Key metric: us per feature value  (block_ms * 1000 / (n_rows * n_produces))
               tells you the true cost of each output cell, independent of
               how many features the block happens to produce.
3. IC analysis — Spearman rank IC of every produced column vs next-day log-return
   GREAT: |IC| >= 0.05   ok: |IC| >= 0.01   meh: |IC| < 0.01

Color scheme mirrors Util.py (24-bit ANSI gradient + SQN-style quality tiers).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import os
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
# Color helpers  (palette + SQN-style quality tiers, mirrored from Util.py)
# ---------------------------------------------------------------------------

# Enable VT100 escape processing on Windows terminals.
if sys.platform == "win32":
    os.system("")

RESET = "\033[0m"
BOLD  = "\033[1m"


def _rgb(r: int, g: int, b: int) -> str:
    return f"\033[38;2;{r};{g};{b}m"


# Fixed tier colors — same RGB values Util.py uses for its SQN print-out.
GREEN_BRIGHT = _rgb(0, 235, 0)
GREEN        = _rgb(0, 200, 0)
GREEN_LIGHT  = _rgb(180, 255, 180)
YELLOW       = _rgb(220, 220, 0)
ORANGE       = _rgb(255, 180, 0)
RED          = _rgb(220, 0, 0)
DIM          = _rgb(150, 150, 150)


def _c(text: str, code: str) -> str:
    """Wrap an already-width-padded string in a color so alignment is kept."""
    return f"{code}{text}{RESET}"


def _grad(frac: float) -> str:
    """Green(good=0) -> Red(bad=1) gradient — same stops as Util.colorize_output."""
    frac = max(0.0, min(1.0, frac))
    colors = [
        (0, 235, 0), (0, 180, 0), (220, 220, 0),
        (220, 140, 0), (220, 0, 0), (240, 0, 0),
    ]
    i = min(int(frac * (len(colors) - 1)), len(colors) - 2)
    t = frac * (len(colors) - 1) - i
    r = int(colors[i][0] * (1 - t) + colors[i + 1][0] * t)
    g = int(colors[i][1] * (1 - t) + colors[i + 1][1] * t)
    b = int(colors[i][2] * (1 - t) + colors[i + 1][2] * t)
    return _rgb(r, g, b)


def _ic_color(ic: float) -> str:
    """SQN-style quality tier for an IC value: GREAT/ok/meh."""
    a = abs(ic)
    if a >= 0.05:
        return GREEN_BRIGHT
    if a >= 0.01:
        return YELLOW
    return DIM


def _grade_color(grade: str) -> str:
    return {"GREAT": GREEN_BRIGHT, "ok": YELLOW, "meh": DIM}.get(grade, DIM)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

W   = 84
SEP = "=" * W


def _section(title: str) -> None:
    print()
    print(_c(SEP, DIM))
    print(BOLD + f"  {title}" + RESET)
    print(_c(SEP, DIM))


def _rule() -> None:
    print("  " + _c("-" * (W - 4), DIM))


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


def _hotbar(frac: float, code: str, width: int = 20) -> str:
    """Colored hotness bar — filled portion in `code`, remainder dimmed."""
    frac = max(0.0, min(1.0, frac))
    filled = round(frac * width)
    return _c("#" * filled, code) + _c("." * (width - filled), DIM)


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
    parser.add_argument("--ic_min_names", type=int, default=20,
                        help="Min names in a day's cross-section to count its IC (default 20). "
                             "Per-day cross-sectional IC needs a WIDE universe -- run --n 300+ "
                             "for meaningful numbers; at the default --n 10 most days are skipped.")
    parser.add_argument("--no_ledger", action="store_true",
                        help="do not append IC results to the trial ledger")
    parser.add_argument("--candidates", action="store_true",
                        help="ALSO benchmark single-underscore candidate blocks (e.g. _paper_*) "
                             "alongside the promoted blocks, to check they're healthy. Helpers "
                             "without METADATA/compute are auto-skipped.")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Section 1 — LOC
    # -----------------------------------------------------------------------
    _section("Feature Template LOC  (non-__ files)")

    loc_rows   = _loc_table()
    total_loc  = sum(n for _, n in loc_rows)
    name_width = max((len(r[0]) for r in loc_rows), default=10)

    for fname, n in loc_rows:
        print(f"  {fname:<{name_width}}  {n:>5} {_c('lines', DIM)}")

    _rule()
    print(f"  {BOLD}{'TOTAL':<{name_width}}{RESET}  {BOLD}{total_loc:>5}{RESET} lines  "
          f"{_c(f'({len(loc_rows)} active template files)', DIM)}")

    # -----------------------------------------------------------------------
    # Section 2 — Benchmark
    # -----------------------------------------------------------------------
    all_paths = sorted(PRICE_DATA_DIR.glob("*.parquet"))
    if not all_paths:
        print(_c(f"\n[!] No parquet files found in {PRICE_DATA_DIR}", RED))
        return

    rng = random.Random(args.seed)
    sample_paths = rng.sample(all_paths, min(args.n, len(all_paths)))
    tickers = [p.stem for p in sample_paths]

    # Discover blocks once (suppress the duplicate-column [WARN] spam)
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        blocks = discover_blocks(include_candidates=args.candidates)

    # n_produces per block (from METADATA)
    n_produces = {
        name: len(info["meta"].get("produces", []))
        for name, info in blocks.items()
    }

    _cand_tag = f", +{len(blocks)} blocks incl. candidates" if args.candidates else ""
    _section(f"Benchmark  (N={len(tickers)}, seed={args.seed}{_cand_tag})")
    print(f"  {_c('Tickers', DIM)} : {' '.join(tickers)}")
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
                result_df, timing = run_pipeline_timed(
                    df, exclude=args.exclude, verbose=False,
                    include_candidates=args.candidates,
                )
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
            print(f"  {_c('[WARN]', ORANGE)} {e}")
        print()

    if not block_times:
        print(_c("  No results collected — all tickers failed.", RED))
        return

    # ---- build ranked table -------------------------------------------------
    # rank by mean time descending (slowest first)
    sorted_blocks = sorted(
        block_times.keys(),
        key=lambda n: np.mean(block_times[n]),
        reverse=True,
    )

    total_block_ms = sum(np.sum(v) for v in block_times.values())
    max_share = max(
        (np.sum(block_times[n]) / max(total_block_ms, 1) for n in sorted_blocks),
        default=1.0,
    )

    # column widths
    bname_w = max(len(n) for n in sorted_blocks)
    bname_w = max(bname_w, 5)   # min "Block"

    header = (
        f"  {'Block':<{bname_w}}  {'Calls':>5}  {'Avg':>8}  "
        f"{'Std':>8}  {'us/fval':>9}  {'Share':>6}  Hotness"
    )
    print(BOLD + header + RESET)
    _rule()

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
        # hotness normalized so the slowest block reads full-red
        hot   = share / max(max_share, 1e-9)
        code  = _grad(hot)

        us_str    = f"{us_per_fval:>7.2f}us"
        share_str = f"{share * 100:>5.1f}%"

        print(
            f"  {name:<{bname_w}}  {calls:>5}  {_fmt_ms(avg_ms):>8}  "
            f"{_fmt_ms(std_ms):>8}  {us_str}  {_c(share_str, code)}  "
            f"{_hotbar(hot, code)}"
        )

    _rule()

    # total row
    print(
        f"  {BOLD}{'ALL BLOCKS':<{bname_w}}{RESET}  {'':>5}  {'':>8}  "
        f"{'':>8}  {'':>9}  {'100.0%':>6}  {_c('wall ' + _fmt_ms(wall_ms), DIM)}"
    )

    # ---- summary ------------------------------------------------------------
    throughput = total_fvals / max(wall_ms / 1_000, 1e-9)

    print()
    print(f"  {BOLD}Summary{RESET}")
    print("  " + _c("-" * 40, DIM))
    print(f"  Tickers tested   : {len(block_times.get(sorted_blocks[0], [1])):>6}  "
          f"{_c(f'(of {len(sample_paths)} sampled)', DIM)}")
    print(f"  Total rows       : {total_rows:>9,}")
    print(f"  Total fvals      : {total_fvals:>9,}   "
          f"{_c('(rows x features, summed across tickers)', DIM)}")
    print(f"  Pipeline wall    : {_fmt_ms(wall_ms):>9}   "
          f"{_c('(incl. parquet load)', DIM)}")
    print(f"  Throughput       : {_c(f'{throughput:>9,.0f}', GREEN)} fval/s")
    print(f"  Avg per ticker   : {_fmt_ms(wall_ms / max(len(sample_paths), 1)):>9}")

    # slowest 3
    top3 = sorted_blocks[:3]
    print()
    print(f"  {BOLD}Hottest blocks:{RESET}")
    for name in top3:
        pct = np.sum(block_times[name]) / max(total_block_ms, 1) * 100
        avg_ms = float(np.mean(block_times[name]))
        np_ = n_produces.get(name, 1)
        avg_rows = float(np.mean(block_rows[name]))
        us_pf = (avg_ms * 1_000) / max(avg_rows * np_, 1)
        code = _grad((pct / 100) / max(max_share, 1e-9))
        print(f"    {_c(f'{name:<{bname_w}}', code)}  {pct:5.1f}% of block time  "
              f"avg {_fmt_ms(avg_ms)}  {us_pf:.2f}us/fval")

    # -----------------------------------------------------------------------
    # Section 3 — IC Analysis  (PER-DAY CROSS-SECTIONAL, not pooled)
    #
    # The model consumes a daily CROSS-SECTION (rank names against each other each
    # day), so the honest descriptor is a per-day cross-sectional Spearman IC:
    #   ic[d] = spearman(feature, next-day-logret) across the names trading on day d
    # We then summarize the IC time-series:
    #   IC mean : average daily cross-sectional rank correlation
    #   IC-IR   : mean / std  (Grinold information ratio -- the STABILITY of the edge)
    #   t       : IC-IR * sqrt(n_days)  [descriptive only; daily ICs autocorrelate]
    #   pos%    : fraction of days with IC > 0  (sign consistency)
    # The old POOLED IC (concatenate every ticker's whole history, one correlation)
    # conflated cross-sectional and time-series variance -- strictly worse than what
    # the model sees, and not how features are actually selected. This replaces it.
    #
    # NOTE: per-day cross-sectional IC needs a WIDE universe. At the default --n 10
    # most days have too few names and are skipped -- run --n 300+ for real numbers.
    # IC mean is still a whole-universe LINEAR descriptor: use it as triage, never as
    # a ranking gate (the strategy trades the tail -- use __tail_screen.py for that).
    # -----------------------------------------------------------------------
    _section(
        f"IC Analysis  (per-day cross-sectional Spearman vs next-day log-return, "
        f"min {args.ic_min_names} names/day)"
    )
    print(
        f"  Thresholds (|IC mean|):  {_c('>= 0.05 GREAT', GREEN_BRIGHT)}   "
        f"{_c('>= 0.01 ok', YELLOW)}   {_c('< 0.01 meh', DIM)}"
        f"   {_c('| IC-IR = stability', DIM)}"
    )
    print()

    # Build ONE long panel: Date, Ticker, every produced column, next-day target.
    panel_parts: list[pd.DataFrame] = []
    for ticker, rdf in results_store.items():
        if "Close" not in rdf.columns or "Date" not in rdf.columns:
            continue
        rdf = rdf.loc[:, ~rdf.columns.duplicated(keep="last")].copy()
        close = rdf["Close"].astype(float)
        logret = np.log(close / close.shift(1))
        rdf["__target"] = logret.shift(-1)        # feature[t] predicts return[t+1]
        rdf["__ticker"] = ticker
        panel_parts.append(rdf)

    if not panel_parts:
        print(_c("  [!] No usable results (need Date + Close) — cannot compute IC.", RED))
        return

    panel = pd.concat(panel_parts, ignore_index=True)
    panel["Date"] = pd.to_datetime(panel["Date"])
    n_dates_total = panel["Date"].nunique()
    avg_xs = panel.groupby("Date")["__ticker"].size().mean()
    print(f"  {_c('panel', DIM)}: {len(panel):,} rows, {n_dates_total} dates, "
          f"avg {avg_xs:.0f} names/day  {_c(f'(of {len(results_store)} tickers)', DIM)}")
    if avg_xs < args.ic_min_names:
        print(_c(f"  [WARN] avg cross-section ({avg_xs:.0f}) < --ic_min_names "
                 f"({args.ic_min_names}); most days skipped. Re-run with larger --n.", ORANGE))
    print()

    # Map each produced column to its owning block (last writer wins)
    produced_cols: set[str] = set()
    block_for_col: dict[str, str] = {}
    for _bname, info in blocks.items():
        for col in info["meta"].get("produces", []):
            produced_cols.add(col)
            block_for_col[col] = _bname

    date_codes_all = panel["Date"].to_numpy()
    targ_all = panel["__target"].to_numpy(dtype=float)
    min_names = args.ic_min_names

    def _per_day_ic(fvals: np.ndarray) -> np.ndarray:
        """Vectorized per-day cross-sectional Spearman IC series (no python day-loop)."""
        s = pd.DataFrame({"d": date_codes_all, "f": fvals, "t": targ_all}).dropna()
        if s.empty:
            return np.array([])
        cnt = s.groupby("d")["f"].transform("size")
        s = s[cnt >= min_names]
        if s.empty:
            return np.array([])
        s["fr"] = s.groupby("d")["f"].rank()
        s["tr"] = s.groupby("d")["t"].rank()
        g = s.groupby("d")
        fc = s["fr"] - g["fr"].transform("mean")
        tc = s["tr"] - g["tr"].transform("mean")
        num = (fc * tc).groupby(s["d"]).sum()
        den = np.sqrt((fc * fc).groupby(s["d"]).sum() * (tc * tc).groupby(s["d"]).sum())
        ic = (num / den).replace([np.inf, -np.inf], np.nan).dropna()
        return ic.to_numpy()

    # col -> dict(mean, ir, t, pos, n_days)
    ic_results: dict[str, dict] = {}
    for col in produced_cols:
        if col not in panel.columns:
            continue
        ser = _per_day_ic(panel[col].to_numpy(dtype=float))
        if len(ser) < 10:                 # need a few days to mean anything
            continue
        m = float(np.mean(ser))
        sd = float(np.std(ser))
        ir = m / sd if sd > 1e-12 else np.nan
        ic_results[col] = dict(
            mean=m, ir=ir,
            t=(ir * np.sqrt(len(ser)) if np.isfinite(ir) else np.nan),
            pos=float(np.mean(ser > 0)), n_days=len(ser),
        )

    if not ic_results:
        print(_c("  No IC results computed (cross-section too small? try larger --n).", RED))
        return

    def _grade(ic: float) -> str:
        a = abs(ic)
        if a >= 0.05:
            return "GREAT"
        if a >= 0.01:
            return "ok"
        return "meh"

    # ---- Per-block summary (by median |IC mean|) ----------------------------
    block_abs_ics: dict[str, list[float]] = {}
    for col, r in ic_results.items():
        block_abs_ics.setdefault(block_for_col.get(col, "?"), []).append(abs(r["mean"]))

    sorted_blocks_ic = sorted(block_abs_ics.items(),
                              key=lambda x: float(np.median(x[1])), reverse=True)
    bw2 = max((len(n) for n in block_abs_ics), default=5)
    bw2 = max(bw2, 5)

    print(BOLD + f"  {'Block':<{bw2}}  {'Cols':>4}  {'GREAT':>5}  {'ok':>4}  {'meh':>4}  "
          f"{'Med|IC|':>7}  {'Max|IC|':>7}" + RESET)
    _rule()
    for bn, abs_ics in sorted_blocks_ic:
        n_g = sum(1 for x in abs_ics if x >= 0.05)
        n_o = sum(1 for x in abs_ics if 0.01 <= x < 0.05)
        n_m = sum(1 for x in abs_ics if x < 0.01)
        med = float(np.median(abs_ics)); mx = float(np.max(abs_ics))
        g_cell = _c(f"{n_g:>5}", GREEN_BRIGHT) if n_g else f"{n_g:>5}"
        o_cell = _c(f"{n_o:>4}", YELLOW) if n_o else f"{n_o:>4}"
        m_cell = _c(f"{n_m:>4}", DIM) if n_m else f"{n_m:>4}"
        print(f"  {bn:<{bw2}}  {len(abs_ics):>4}  {g_cell}  {o_cell}  {m_cell}  "
              f"{_c(f'{med:>7.4f}', _ic_color(med))}  {_c(f'{mx:>7.4f}', _ic_color(mx))}")
    print()

    # ---- Top-N individual columns (by |IC mean|) ----------------------------
    sorted_cols = sorted(ic_results.items(), key=lambda x: abs(x[1]["mean"]), reverse=True)
    top_n = args.top_n
    cw = max((len(c) for c in ic_results), default=10)
    cw = min(max(cw, 6), 48)

    print(f"  {BOLD}Top {top_n} columns by |IC mean|:{RESET}")
    print(BOLD + f"  {'Rank':>4}  {'Column':<{cw}}  {'Block':<{bw2}}  "
          f"{'ICmean':>8}  {'IC-IR':>6}  {'pos%':>5}  {'days':>5}  Grade" + RESET)
    _rule()
    for rank, (col, r) in enumerate(sorted_cols[:top_n], 1):
        bn = block_for_col.get(col, "?")
        grade = _grade(r["mean"]); code = _grade_color(grade)
        disp = col if len(col) <= cw else col[:cw - 2] + ".."
        ir_s = f"{r['ir']:>+6.2f}" if np.isfinite(r["ir"]) else f"{'nan':>6}"
        mean_s = f"{r['mean']:>+8.4f}"
        pos_s = f"{r['pos'] * 100:>4.0f}%"
        print(f"  {rank:>4}  {_c(f'{disp:<{cw}}', code)}  {_c(f'{bn:<{bw2}}', DIM)}  "
              f"{_c(mean_s, code)}  {_c(ir_s, code)}  "
              f"{pos_s:>5}  {r['n_days']:>5}  {_c(grade, code)}")

    # ---- Distribution summary -----------------------------------------------
    n_great = sum(1 for r in ic_results.values() if abs(r["mean"]) >= 0.05)
    n_ok    = sum(1 for r in ic_results.values() if 0.01 <= abs(r["mean"]) < 0.05)
    n_meh   = sum(1 for r in ic_results.values() if abs(r["mean"]) < 0.01)
    total_c = len(ic_results)
    print()
    print(f"  {BOLD}Distribution{RESET} ({total_c} columns):   "
          f"{_c(f'{n_great} GREAT ({n_great * 100 // total_c}%)', GREEN_BRIGHT)}    "
          f"{_c(f'{n_ok} ok ({n_ok * 100 // total_c}%)', YELLOW)}    "
          f"{_c(f'{n_meh} meh ({n_meh * 100 // total_c}%)', DIM)}")
    if total_c > top_n:
        print(_c(f"  (use --top_n {total_c} to see all columns)", DIM))

    # ---- Trial ledger -------------------------------------------------------
    if not args.no_ledger:
        try:
            sys.path.insert(0, str(HERE))
            from importlib import import_module
            _tl = import_module("__trial_ledger")
            led = _tl.TrialLedger(tool="diagnostics", seed=args.seed,
                                  params={"n": args.n, "ic_min_names": args.ic_min_names,
                                          "n_dates": int(n_dates_total)})
            for col, r in ic_results.items():
                led.add_metrics(col, {"ic_perday_mean": r["mean"], "ic_ir": r["ir"],
                                      "ic_pos_frac": r["pos"]},
                                family=block_for_col.get(col, "?"), n_obs=r["n_days"])
            p = led.flush()
            print(_c(f"  [ledger] {total_c} columns logged -> {p}", DIM))
        except Exception as exc:
            print(_c(f"  [ledger off: {exc}]", DIM))

    print()
    print(_c(SEP, DIM))
    print()


if __name__ == "__main__":
    main()
