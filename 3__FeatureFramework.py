"""
3__FeatureFramework.py  --  Modular feature engineering pipeline

USAGE
-----
  # Process one ticker in-process with per-block timing printed live
  python 3__FeatureFramework.py --ticker AAPL

  # Process all tickers in parallel, write to Data/ProcessedData_v2/
  python 3__FeatureFramework.py --all

  # Control parallelism and scope
  python 3__FeatureFramework.py --all --workers 8 --runpercent 50

  # Skip specific blocks
  python 3__FeatureFramework.py --all --exclude rsi momentum_score

  # List every registered block in execution order, then exit
  python 3__FeatureFramework.py --list

HOW TO ADD A NEW FEATURE
------------------------
  1. Copy FeatureTemplates/__example_template.py to a new file,
     e.g. FeatureTemplates/my_new_feature.py
  2. Fill in METADATA (name, description, requires, produces, tags).
  3. Implement compute(df) -> df.
  4. Run -- the new file is auto-discovered immediately.
  No other files need to be touched.

HOW TO REMOVE / DISABLE A FEATURE
----------------------------------
  Permanent : delete the file from FeatureTemplates/
  One-off   : pass --exclude <name> at the command line
"""

import argparse
import importlib.util
import os
import statistics
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths  (all relative to this file so workers in subprocesses find them too)
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).parent
TEMPLATES_DIR = ROOT / "FeatureTemplates"
PRICE_DATA_DIR = ROOT / "Data" / "PriceData"
OUT_DIR       = ROOT / "Data" / "ProcessedData_v2"

# These columns are always placed first in the output, in this fixed order.
# Feature blocks must NEVER overwrite or drop any of these.
OHLCV_COLS = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]


# ===========================================================================
# SECTION 1 -- Block discovery
# ===========================================================================

def discover_blocks() -> dict:
    """
    Import every .py file in FeatureTemplates/ whose name does NOT start with
    an underscore.  Files starting with _ or __ are skipped (they are
    templates, helpers, or documentation).

    Returns
    -------
    dict[str, dict]
        {block_name: {"meta": METADATA dict, "fn": compute fn, "path": Path}}
    """
    blocks: dict = {}

    for path in sorted(TEMPLATES_DIR.glob("*.py")):
        if path.stem.startswith("_"):
            continue

        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod  = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as exc:
            print(f"  [WARN] Cannot import {path.name}: {exc}", file=sys.stderr)
            continue

        if not hasattr(mod, "METADATA") or not hasattr(mod, "compute"):
            print(f"  [WARN] {path.name} missing METADATA or compute() -- skipped",
                  file=sys.stderr)
            continue

        meta = mod.METADATA
        name = meta.get("name", path.stem)

        if name in blocks:
            print(f"  [WARN] Duplicate block name '{name}' in {path.name} -- skipped",
                  file=sys.stderr)
            continue

        blocks[name] = {"meta": meta, "fn": mod.compute, "path": path}

    return blocks


# ===========================================================================
# SECTION 2 -- Dependency resolution (topological sort)
# ===========================================================================

def resolve_order(blocks: dict) -> list[str]:
    """
    Return block names in the execution order that satisfies all
    requires / produces dependencies.

    Algorithm: Kahn's BFS topological sort.
    Raises ValueError on circular dependencies.
    """
    # produced_by[column] = block_name
    produced_by: dict[str, str] = {}
    for name, block in blocks.items():
        for col in block["meta"].get("produces", []):
            if col in produced_by:
                print(
                    f"  [WARN] Column '{col}' claimed by both "
                    f"'{produced_by[col]}' and '{name}' -- keeping '{name}'",
                    file=sys.stderr,
                )
            produced_by[col] = name

    in_degree:  dict[str, int]        = {n: 0 for n in blocks}
    adjacency:  dict[str, list[str]]  = defaultdict(list)

    for name, block in blocks.items():
        for req in block["meta"].get("requires", []):
            producer = produced_by.get(req)
            if producer and producer != name and producer in blocks:
                adjacency[producer].append(name)
                in_degree[name] += 1

    queue  = deque(n for n in blocks if in_degree[n] == 0)
    order: list[str] = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in adjacency[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(order) != len(blocks):
        cycle = [n for n in blocks if n not in order]
        raise ValueError(f"Circular dependency detected among blocks: {cycle}")

    return order


# ===========================================================================
# SECTION 3 -- Per-ticker pipeline runner (timed)
# ===========================================================================

def run_pipeline_timed(
    df:      pd.DataFrame,
    exclude: list[str] | None = None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Run all active feature blocks on df, timing each block individually.

    Parameters
    ----------
    df      : Per-ticker OHLCV DataFrame, ascending by date.
    exclude : Block names to skip this run.
    verbose : If True, print a live line per block with its ms cost.

    Returns
    -------
    (output_df, timing_dict)

    timing_dict schema
    ------------------
    {
        "n_rows":   int,            # rows in input df
        "total_s":  float,          # sum of all block wall-clock seconds
        "blocks":   {               # one entry per block that actually ran
            block_name: float,      #   seconds
            ...
        },
        "skipped":  [str, ...],     # blocks that were missing required columns
    }
    """
    exclude_set = set(exclude or [])
    all_blocks  = discover_blocks()
    blocks      = {k: v for k, v in all_blocks.items() if k not in exclude_set}
    order       = resolve_order(blocks)

    timing: dict = {
        "n_rows":       len(df),
        "total_s":      0.0,
        "blocks":       {},
        "skipped":      [],
        # extras used by _print_ticker_report
        "order":        order,
        "meta":         {n: {"produces": blocks[n]["meta"].get("produces", []),
                             "requires": blocks[n]["meta"].get("requires", [])}
                         for n in blocks},
        "skip_reasons": {},   # name -> [missing col, ...]
    }

    for name in order:
        block    = blocks[name]
        requires = block["meta"].get("requires", [])
        missing  = [r for r in requires if r not in df.columns]

        if missing:
            timing["skip_reasons"][name] = missing
            timing["skipped"].append(name)
            if verbose:
                print(f"  [skip]  {name:<26}  needs: {', '.join(missing)}")
            continue

        t0 = time.perf_counter()
        try:
            df = block["fn"](df)
            if df.columns.duplicated().any():
                df = df.loc[:, ~df.columns.duplicated(keep="last")]
            df = df.copy()   # defragment between blocks
        except Exception as exc:
            print(f"  [ERROR] {name}: {exc}", file=sys.stderr)
            timing["skipped"].append(name)
            continue
        elapsed = time.perf_counter() - t0

        timing["blocks"][name]  = elapsed
        timing["total_s"]      += elapsed

        if verbose:
            produces = block["meta"].get("produces", [])
            preview  = "  ".join(produces[:5])
            if len(produces) > 5:
                preview += f"  (+{len(produces) - 5})"
            print(f"  {name:<26}  {elapsed * 1000:7.1f}ms  +{len(produces)}  {preview}")

    return _standardize_columns(df), timing


# ===========================================================================
# SECTION 4 -- Column ordering
# ===========================================================================

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Date, Ticker, Open, High, Low, Close, Volume first -- then A-Z."""
    prefix = [c for c in OHLCV_COLS if c in df.columns]
    rest   = sorted(c for c in df.columns if c not in set(OHLCV_COLS))
    return df[prefix + rest]


# ===========================================================================
# SECTION 5 -- Worker function  (top-level so ProcessPoolExecutor can pickle it)
# ===========================================================================

def _worker_fn(
    file_path: str,
    out_dir:   str,
    exclude:   list[str],
) -> tuple[bool, str, dict | None, str | None]:
    """
    Full pipeline for one ticker file.  Runs in a subprocess.

    Returns (success, ticker, timing_dict | None, error_msg | None).
    timing_dict has the same schema as run_pipeline_timed() above plus a
    "ticker" key added here so the main process can identify it.
    """
    path   = Path(file_path)
    ticker = path.stem

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        return False, ticker, None, f"read error: {exc}"

    if "Date" not in df.columns and df.index.name == "Date":
        df = df.reset_index()

    try:
        result, timing = run_pipeline_timed(df, exclude=exclude, verbose=False)
    except Exception as exc:
        return False, ticker, None, f"pipeline error: {exc}"

    try:
        result.to_parquet(Path(out_dir) / path.name, index=False)
    except Exception as exc:
        return False, ticker, None, f"write error: {exc}"

    timing["ticker"] = ticker
    return True, ticker, timing, None


# ===========================================================================
# SECTION 6 -- Timing report
# ===========================================================================

def _fmt_ms(ms: float) -> str:
    """Human-readable time: show as seconds if >= 1000 ms, else ms."""
    if ms >= 1_000:
        return f"{ms / 1000:.2f}s"
    return f"{ms:.1f}ms"


def _fmt_val(val) -> str:
    """Format a scalar cell value for the last-row debug view."""
    if val is None:
        return "None"
    try:
        if pd.isna(val):
            return "NaN"
    except (TypeError, ValueError):
        pass
    if isinstance(val, (int,)):
        return str(val)
    if isinstance(val, float):
        return f"{val:.6f}"
    return str(val)


def _print_ticker_report(
    ticker:  str,
    result:  pd.DataFrame,
    timing:  dict,
    wall_ms: float,
) -> None:
    """
    Two-section debug report for --ticker mode.

    Section 1  Block run table: name | time | +N cols | first few col names
    Section 2  Last-row values grouped by the block that produced them
    """
    W           = 80
    SEP         = "=" * W
    order       = timing.get("order", list(timing["blocks"].keys()))
    meta        = timing.get("meta", {})
    skip_reason = timing.get("skip_reasons", {})
    n_ran       = len(timing["blocks"])
    n_skip      = len(timing["skipped"])
    n_feat_cols = result.shape[1] - sum(1 for c in OHLCV_COLS if c in result.columns)

    # ---- section 1: block run table ------------------------------------------
    print()
    print(SEP)
    print(
        f"  {ticker}  |  {timing['n_rows']:,} rows in  |  "
        f"{result.shape[1]} cols out (+{n_feat_cols} features)  |  "
        f"{n_ran} ran  {n_skip} skipped"
    )
    print(SEP)
    print(f"  {'Block':<26}  {'Time':>8}  {'Added':>6}  Columns produced")
    print("  " + "-" * (W - 2))

    for name in order:
        produces = meta.get(name, {}).get("produces", [])
        if name in timing["blocks"]:
            t_ms    = timing["blocks"][name] * 1_000
            preview = "  ".join(produces[:6])
            if len(produces) > 6:
                preview += f"  (+{len(produces) - 6})"
            print(
                f"  {name:<26}  {_fmt_ms(t_ms):>8}  "
                f"{'+' + str(len(produces)):>6}  {preview}"
            )
        else:
            reason     = skip_reason.get(name, [])
            reason_str = f"needs: {', '.join(reason)}" if reason else "excluded"
            print(f"  {'[skip] ' + name:<26}  {'':>8}  {'':>6}  {reason_str}")

    print("  " + "-" * (W - 2))
    print(
        f"  {'TOTAL':<26}  {_fmt_ms(timing['total_s'] * 1_000):>8}  "
        f"  wall: {_fmt_ms(wall_ms)}"
    )

    if len(result) == 0:
        return

    # ---- section 2: last-row values grouped by block -------------------------
    last     = result.iloc[-1]
    date_val = last["Date"] if "Date" in result.columns else result.index[-1]

    # When two blocks claim the same column, the later block in execution order
    # wins (the framework deduplicates keeping the last).  Pre-compute ownership
    # so each column is printed exactly once, under its winning block.
    col_owner: dict[str, str] = {}
    for name in order:
        if name not in timing["blocks"]:
            continue
        for col in meta.get(name, {}).get("produces", []):
            if col in result.columns:
                col_owner[col] = name   # last writer wins

    print()
    print(SEP)
    print(f"  LAST ROW  |  {date_val}")
    print(SEP)

    # OHLCV baseline
    ohlcv_present = [c for c in OHLCV_COLS if c in result.columns]
    if ohlcv_present:
        print(f"  [OHLCV]")
        for col in ohlcv_present:
            print(f"    {col:<38}  {_fmt_val(last[col])}")

    # per-block features
    for name in order:
        if name not in timing["blocks"]:
            reason     = skip_reason.get(name, [])
            reason_str = f"needs: {', '.join(reason)}" if reason else "excluded"
            print(f"  [skip: {name}]  {reason_str}")
            continue
        produces  = meta.get(name, {}).get("produces", [])
        col_vals  = [(c, last[c]) for c in produces
                     if c in result.columns and col_owner.get(c) == name]
        if not col_vals:
            continue
        print(f"  [{name}]")
        for col, val in col_vals:
            print(f"    {col:<38}  {_fmt_val(val)}")

    print(SEP)
    print()


def _print_timing_report(
    timings:      list[dict],
    wall_clock_s: float,
    n_workers:    int,
    top_n:        int = 10,
) -> None:
    """
    Aggregate per-ticker timing dicts and print two sections:

    SECTION A -- Block summary table
        Block / Calls / Total / Mean / p50 / p95 / Max
        Serial-equivalent vs wall-clock speedup.

    SECTION B -- Slow ticker detail (top_n slowest per block)
        For each block: the top_n slowest individual tickers, their row count,
        their ms cost, and how many times the mean they are (xMEAN).
        Use this to find pathological tickers and investigate why they're slow.
    """
    if not timings:
        print("[timing report] No data collected.")
        return

    W = 85  # report width

    # ---- Build per-block data structures ------------------------------------
    # block_samples[bname] = [(ms, ticker, n_rows), ...]
    block_entries: dict[str, list[tuple[float, str, int]]] = defaultdict(list)
    total_rows = 0
    for t in timings:
        nr = t.get("n_rows", 0)
        total_rows += nr
        ticker = t.get("ticker", "?")
        for bname, secs in t.get("blocks", {}).items():
            block_entries[bname].append((secs * 1_000.0, ticker, nr))

    # Preserve execution order
    seen: set[str] = set()
    order: list[str] = []
    for t in timings:
        for k in t.get("blocks", {}):
            if k not in seen:
                order.append(k)
                seen.add(k)
    for k in block_entries:
        if k not in seen:
            order.append(k)
            seen.add(k)

    n_tickers     = len(timings)
    block_samples = {b: [e[0] for e in v] for b, v in block_entries.items()}
    serial_equiv_s = sum(sum(ms) / 1_000.0 for ms in block_samples.values())

    # =========================================================================
    # SECTION A -- aggregate summary
    # =========================================================================
    print()
    print("=" * W)
    print(
        f"TIMING REPORT  --  {n_tickers} tickers  |  {total_rows:,} rows  |  "
        f"{n_workers} workers"
    )
    print(
        f"Wall clock : {wall_clock_s:.1f}s   |   "
        f"Serial equiv : {serial_equiv_s:.1f}s   |   "
        f"Speedup : {serial_equiv_s / max(wall_clock_s, 0.001):.1f}x"
    )
    print("=" * W)
    print(f"  {'Block':<28}  {'Calls':>6}  {'Total':>8}  {'Mean':>8}  {'p50':>8}  {'p95':>8}  {'Max':>8}")
    print("-" * W)

    grand_total_ms = 0.0
    block_means: dict[str, float] = {}
    for bname in order:
        if bname not in block_samples:
            continue
        ms          = sorted(block_samples[bname])
        calls       = len(ms)
        total_ms    = sum(ms)
        grand_total_ms += total_ms
        mean_ms     = statistics.mean(ms)
        p50_ms      = statistics.median(ms)
        idx95       = max(0, int(calls * 0.95) - 1)
        p95_ms      = ms[idx95] if calls >= 5 else ms[-1]
        max_ms      = ms[-1]
        block_means[bname] = mean_ms

        print(
            f"  {bname:<28}  {calls:>6}  {_fmt_ms(total_ms):>8}  "
            f"{_fmt_ms(mean_ms):>8}  {_fmt_ms(p50_ms):>8}  "
            f"{_fmt_ms(p95_ms):>8}  {_fmt_ms(max_ms):>8}"
        )

    print("-" * W)
    print(f"  {'TOTAL (all blocks)':<28}  {n_tickers:>6}  {_fmt_ms(grand_total_ms):>8}")
    print(
        f"  Throughput: {n_tickers / max(wall_clock_s, 0.001):.1f} tickers/s  |  "
        f"{total_rows / max(wall_clock_s, 0.001):,.0f} rows/s"
    )
    print("=" * W)

    # =========================================================================
    # SECTION B -- slow ticker detail
    # =========================================================================
    print()
    print("=" * W)
    print(f"SLOW TICKER DETAIL  --  top {top_n} per block  (xMEAN = multiple of mean cost)")
    print("=" * W)

    for bname in order:
        if bname not in block_entries:
            continue
        entries = block_entries[bname]
        mean_ms = block_means.get(bname, 1.0) or 1.0
        # sort descending by ms so slowest is first
        slowest = sorted(entries, key=lambda e: e[0], reverse=True)[:top_n]

        print(f"\n  {bname}  (mean {_fmt_ms(mean_ms)})")
        print(f"  {'#':<4} {'Ticker':<10} {'Rows':>6}  {'Time':>8}  {'xMEAN':>6}")
        print("  " + "-" * 40)
        for rank, (ms, ticker, n_rows) in enumerate(slowest, 1):
            mult = ms / mean_ms
            print(
                f"  {rank:<4} {ticker:<10} {n_rows:>6}  {_fmt_ms(ms):>8}  {mult:>5.1f}x"
            )

    print()
    print("=" * W)
    print()


def _save_timing_csv(timings: list[dict], path: str) -> None:
    """
    Write the full per-ticker timing matrix to a CSV file.

    Columns: ticker, n_rows, total_ms, <block>_ms, <block>_ms, ...

    Load in pandas for custom analysis:
        df = pd.read_csv("timing.csv").sort_values("total_ms", ascending=False)
    """
    if not timings:
        return

    # Collect all block names (union across all tickers)
    all_blocks: list[str] = []
    seen: set[str] = set()
    for t in timings:
        for k in t.get("blocks", {}):
            if k not in seen:
                all_blocks.append(k)
                seen.add(k)

    rows = []
    for t in timings:
        row: dict = {
            "ticker":   t.get("ticker", ""),
            "n_rows":   t.get("n_rows", 0),
            "total_ms": round(t.get("total_s", 0.0) * 1_000, 4),
        }
        for bname in all_blocks:
            secs = t.get("blocks", {}).get(bname)
            row[f"{bname}_ms"] = round(secs * 1_000, 4) if secs is not None else ""
        rows.append(row)

    # Sort by total_ms descending so the slowest tickers are at the top
    rows.sort(key=lambda r: r["total_ms"], reverse=True)

    import csv
    fieldnames = ["ticker", "n_rows", "total_ms"] + [f"{b}_ms" for b in all_blocks]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Timing matrix saved -> {path}  ({len(rows)} tickers, {len(all_blocks)} blocks)")


# ===========================================================================
# SECTION 7 -- Parallel batch processor
# ===========================================================================

def process_all(
    paths:     list[Path],
    out_dir:   Path,
    exclude:   list[str],
    n_workers: int,
) -> list[dict]:
    """
    Process a list of ticker parquet files in parallel.

    Uses ProcessPoolExecutor (CPU-bound compute, one process per core).
    Returns list of timing dicts for all successful tickers.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = fail = 0
    all_timings: list[dict] = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_worker_fn, str(p), str(out_dir), exclude): p
            for p in paths
        }

        with tqdm(total=len(futures), desc="Features", unit="ticker") as pbar:
            for future in as_completed(futures):
                success, ticker, timing, err = future.result()
                if success:
                    ok += 1
                    if timing:
                        all_timings.append(timing)
                else:
                    fail += 1
                    tqdm.write(f"  [FAIL] {ticker}: {err}")

                pbar.update(1)
                pbar.set_description(f"ok={ok} fail={fail}")

    print(f"\nCompleted: {ok} ok, {fail} failed, {len(paths)} total")
    return all_timings


# ===========================================================================
# SECTION 8 -- CLI helpers
# ===========================================================================

def _list_blocks() -> None:
    """Print every registered block in dependency-resolved execution order."""
    blocks = discover_blocks()
    if not blocks:
        print("No blocks found in", TEMPLATES_DIR)
        return

    order = resolve_order(blocks)
    w = 85
    print(f"\n{'#':>2}  {'Name':<22} {'Tags':<26} {'Produces':<28} Description")
    print("-" * w)
    for i, name in enumerate(order, 1):
        meta     = blocks[name]["meta"]
        tags     = ", ".join(meta.get("tags", []))
        produces = ", ".join(meta.get("produces", []))
        desc     = meta.get("description", "")
        print(f"{i:>2}  {name:<22} {tags:<26} {produces:<28} {desc}")
    print()


# ===========================================================================
# Entry point  --  MUST stay under if __name__ == "__main__" on Windows
#                  (ProcessPoolExecutor uses spawn mode, which re-imports this
#                  module in every worker; the guard prevents re-running main)
# ===========================================================================

def main() -> None:
    cpu_count = os.cpu_count() or 4

    parser = argparse.ArgumentParser(
        description="Modular feature engineering pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ticker",      help="Process a single ticker (verbose, in-process)")
    parser.add_argument("--all",         action="store_true",
                        help="Process all tickers in Data/PriceData/ in parallel")
    parser.add_argument("--exclude",     nargs="*", default=[], metavar="BLOCK",
                        help="Block names to skip")
    parser.add_argument("--list",        action="store_true",
                        help="List registered blocks in execution order, then exit")
    parser.add_argument("--out_dir",     default=str(OUT_DIR),
                        help="Output directory for --all mode")
    parser.add_argument("--workers",      type=int, default=min(32, cpu_count),
                        help=f"Parallel worker processes (default: {min(32, cpu_count)})")
    parser.add_argument("--runpercent",  type=int, default=100,
                        help="Percentage of tickers to process in --all mode (default: 100)")
    parser.add_argument("--top_n",       type=int, default=10,
                        help="Slow-ticker detail: how many tickers to show per block (default: 10)")
    parser.add_argument("--save_timing", metavar="PATH",
                        help="Save full per-ticker timing matrix to this CSV path")
    args = parser.parse_args()

    # ---- list ---------------------------------------------------------------
    if args.list:
        _list_blocks()
        return

    # ---- single ticker (in-process, live timing) ----------------------------
    if args.ticker:
        path = PRICE_DATA_DIR / f"{args.ticker}.parquet"
        if not path.exists():
            sys.exit(f"File not found: {path}")

        df = pd.read_parquet(path)
        if "Date" not in df.columns and df.index.name == "Date":
            df = df.reset_index()

        print(f"\nRunning pipeline on {args.ticker} ({len(df):,} rows) ...")

        t_wall = time.perf_counter()
        result, timing = run_pipeline_timed(df, exclude=args.exclude, verbose=True)
        wall_ms = (time.perf_counter() - t_wall) * 1_000

        _print_ticker_report(args.ticker, result, timing, wall_ms)
        return

    # ---- batch (parallel) ---------------------------------------------------
    if args.all:
        all_paths = sorted(PRICE_DATA_DIR.glob("*.parquet"))
        if not all_paths:
            sys.exit(f"No parquets found in {PRICE_DATA_DIR}")

        n      = max(1, int(len(all_paths) * args.runpercent / 100))
        paths  = all_paths[:n]
        n_work = min(args.workers, len(paths))
        out    = Path(args.out_dir)

        active_blocks = [b for b in discover_blocks() if b not in args.exclude]
        print(f"FeatureFramework  --  {len(paths)} tickers | {n_work} workers | -> {out}")
        print(f"Active blocks ({len(active_blocks)}): {active_blocks}")
        print()

        t0         = time.perf_counter()
        all_timing = process_all(paths, out, args.exclude, n_work)
        wall_s     = time.perf_counter() - t0

        _print_timing_report(all_timing, wall_s, n_work, top_n=args.top_n)

        if args.save_timing:
            _save_timing_csv(all_timing, args.save_timing)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
