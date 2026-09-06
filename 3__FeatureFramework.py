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

  # Full per-block timing tables and slow-ticker detail (default output is compact)
  python 3__FeatureFramework.py --all --report full

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
import random
import statistics
import sys
import time
import warnings
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from auxiliary._quiet_progress import tqdm

# Diagnostics sidecars (diagnostics/hooks.py). No-op unless DIAG_OUT is set; only
# main() calls it, never the pool workers. The stand-in keeps this file independent.
try:
    from diagnostics import hooks as _diag
except Exception:
    class _diag:
        enabled = staticmethod(lambda: False)
        outdir = staticmethod(lambda: None)
        dump_json = dump_parquet = append_jsonl = stamp = staticmethod(lambda *a, **k: None)

# ---------------------------------------------------------------------------
# Paths  (all relative to this file so workers in subprocesses find them too)
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).parent
TEMPLATES_DIR = ROOT / "FeatureTemplates"
PRICE_DATA_DIR = Path(os.environ.get("FF_PRICE_DIR", str(ROOT / "Data" / "PriceData")))
OUT_DIR       = ROOT / "Data" / "ProcessedData_v2"

# These columns are always placed first in the output, in this fixed order.
# Feature blocks must NEVER overwrite or drop any of these.
OHLCV_COLS = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]

# Defragment the frame only once pandas' block manager has actually fragmented.
# Blocks add columns one at a time, so nblocks climbs toward the column count;
# copying at every block costs more than it saves (measured: 3.23s every-block vs
# 2.91s with this trigger vs 3.14s never).
_DEFRAG_NBLOCKS = 200


# ===========================================================================
# SECTION 1 -- Block discovery
# ===========================================================================

# Discovery is memoized per PROCESS. Block modules load via exec_module, which
# deliberately bypasses sys.modules -- so without this cache every call re-executes
# all ~190 module bodies. run_pipeline_timed() calls discover once per ticker, which
# made that ~0.09s of pure re-import PER TICKER, and is why helpers like
# _marketcap.py resort to stashing their panel on the `sys` module to survive.
_BLOCK_CACHE: dict[bool, dict] = {}

# False inside worker processes. Discovery/dedup warnings are identical in every
# worker (same block files), and per-ticker block [ERROR] lines are carried back in
# the timing dict and tallied once by _print_integrity_report -- so with 16-32
# workers x 4250 tickers, printing them here too is pure duplicate spam.
_IS_MAIN_PROCESS = True


def _warn(msg: str) -> None:
    if _IS_MAIN_PROCESS:
        print(msg, file=sys.stderr)


def _silence_noisy_warnings() -> None:
    """
    Suppress the warning classes that blocks emit BY DESIGN, which otherwise
    repeat per ticker per worker (a full --all run printed tens of thousands
    of them, drowning the [FAIL]/[ERROR] lines that matter):

      - pandas PerformanceWarning (fragmentation): blocks insert columns one at
        a time on purpose; the framework already defragments via _DEFRAG_NBLOCKS.
      - numpy nan/degenerate-slice RuntimeWarnings: rolling windows are EXPECTED
        to hit all-NaN and short slices during warm-up rows.

    Real failures still surface -- block exceptions are caught, tallied and
    printed by _print_integrity_report.
    """
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    for msg in ("Mean of empty slice", "All-NaN slice encountered",
                "Degrees of freedom <= 0", "invalid value encountered",
                "divide by zero encountered"):
        warnings.filterwarnings("ignore", message=f".*{msg}.*",
                                category=RuntimeWarning)


def discover_blocks(include_candidates: bool = False, *, refresh: bool = False) -> dict:
    """
    Import every .py file in FeatureTemplates/ whose name does NOT start with
    an underscore.  Files starting with _ or __ are skipped (they are
    templates, helpers, or documentation).

    Parameters
    ----------
    include_candidates : if True, ALSO import single-underscore CANDIDATE blocks
        (e.g. ``_paper_*``) so tooling can check them. Pure helpers (``_marketcap``,
        ``_indexes``, ...) lack METADATA/compute and are still skipped by the check
        below; double-underscore tooling files are never imported. Default False
        keeps the production build to promoted blocks only.
    refresh : bypass the per-process cache and re-import from disk. Only tooling that
        edits block files inside a live session needs this.

    Returns
    -------
    dict[str, dict]
        {block_name: {"meta": METADATA dict, "fn": compute fn, "path": Path}}
        The SAME dict object comes back on repeat calls -- treat it as read-only.
    """
    if not refresh and include_candidates in _BLOCK_CACHE:
        return _BLOCK_CACHE[include_candidates]

    blocks: dict = {}

    for path in sorted(TEMPLATES_DIR.glob("*.py")):
        stem = path.stem
        if stem.startswith("__"):
            continue                                  # tooling/templates -- never blocks
        if stem.startswith("_") and not include_candidates:
            continue                                  # single-_ candidates/helpers: opt-in only

        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod  = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as exc:
            _warn(f"  [WARN] Cannot import {path.name}: {exc}")
            continue

        if not hasattr(mod, "METADATA") or not hasattr(mod, "compute"):
            _warn(f"  [WARN] {path.name} missing METADATA or compute() -- skipped")
            continue

        meta = mod.METADATA
        name = meta.get("name", path.stem)

        if name in blocks:
            _warn(f"  [WARN] Duplicate block name '{name}' in {path.name} -- skipped")
            continue

        blocks[name] = {"meta": meta, "fn": mod.compute, "path": path}

    _BLOCK_CACHE[include_candidates] = blocks
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
                _warn(
                    f"  [WARN] Column '{col}' claimed by both "
                    f"'{produced_by[col]}' and '{name}' -- keeping '{name}'"
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
    include_candidates: bool = False,
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
    all_blocks  = discover_blocks(include_candidates=include_candidates)
    blocks      = {k: v for k, v in all_blocks.items() if k not in exclude_set}
    order       = resolve_order(blocks)

    timing: dict = {
        "n_rows":       len(df),
        "total_s":      0.0,
        "blocks":       {},
        "skipped":      [],   # requires not satisfied -- an EXPECTED, benign outcome
        "errors":       {},   # name -> "ExcType: msg"  -- a BUG; never conflate with skipped
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

        # Blocks assign columns in place (df["x"] = ...), so a mid-compute raise
        # leaves PARTIAL columns behind and those get written to parquet -- silent
        # schema drift across tickers. `before = df` would be a mere alias (in-place
        # writes hit it too), so snapshot the COLUMN LIST and restore that on failure.
        cols_before = list(df.columns)
        t0 = time.perf_counter()
        try:
            df = block["fn"](df)
            if df.columns.duplicated().any():
                df = df.loc[:, ~df.columns.duplicated(keep="last")]
            # Defragment only once fragmentation has actually built up. Copying after
            # EVERY block costs more than it saves; the nblocks trigger measured
            # fastest of the strategies tried (3.23s every-block -> 2.91s here).
            if getattr(df, "_mgr", None) is not None and df._mgr.nblocks > _DEFRAG_NBLOCKS:
                df = df.copy()
        except Exception as exc:
            keep = [c for c in cols_before if c in df.columns]
            if len(keep) != len(df.columns):
                df = df[keep]                          # discard partial mutation
            _warn(f"  [ERROR] {name}: {type(exc).__name__}: {exc}")
            timing["errors"][name] = f"{type(exc).__name__}: {exc}"
            continue
        elapsed = time.perf_counter() - t0

        timing["blocks"][name]  = elapsed
        timing["total_s"]      += elapsed

        # The OHLCV contract ("blocks must NEVER overwrite or drop these") was
        # documented but unenforced. Catch the violation at the block that caused
        # it rather than discovering a mangled Close downstream.
        dropped = [c for c in cols_before if c in OHLCV_COLS and c not in df.columns]
        if dropped:
            raise RuntimeError(f"block '{name}' dropped protected column(s): {dropped}")

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

# Shared external-data helpers. Each exposes available(), which forces its panel
# load; warming them here means the cost is paid once per worker at startup instead
# of being billed to whichever ticker happens to touch them first.
_SHARED_HELPERS = ("_indexes", "_marketcap", "_fundamentals", "_insider")


def _worker_init(exclude: list[str]) -> None:
    """
    Runs ONCE per worker process, before any ticker.

    Two jobs:
      1. Warm the block-discovery cache so the first ticker doesn't pay the ~4.5s
         cold import of every block module.
      2. Warm the shared external panels (market caps, indexes, fundamentals,
         insider). _marketcap's panel alone takes ~1.7s. Without this it lands on
         whichever ticker got there first, which is why amihud_size_illiquidity
         reported 1682ms on one ticker and 6ms on all the others -- a permanent
         false entry at the top of the SLOW TICKER DETAIL table.

    Warm-up failures are swallowed: a worker must still start, and any genuine
    problem will resurface (reported properly) on the real run.
    """
    global _IS_MAIN_PROCESS
    _IS_MAIN_PROCESS = False   # silence per-worker duplicates of main-process warnings
    _silence_noisy_warnings()

    try:
        blocks = discover_blocks()
    except Exception:
        return

    # Warm helpers that stash their panel process-globally (e.g. _marketcap on the
    # `sys` module) -- these are shared by every block instance.
    for helper in _SHARED_HELPERS:
        path = TEMPLATES_DIR / f"{helper}.py"
        if not path.exists():
            continue
        try:
            spec = importlib.util.spec_from_file_location(helper, path)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "available"):
                mod.available()
        except Exception:
            pass

    # Helpers that cache per module INSTANCE (e.g. _indexes._FRAME_CACHE) can only be
    # warmed through the block that owns the instance. Run those blocks once on a
    # TRUNCATED frame: panel loads are O(1) in rows, so a 150-row slice pays the load
    # without paying the compute.
    try:
        blocks = {k: v for k, v in blocks.items() if k not in set(exclude or [])}
        seed_paths = sorted(PRICE_DATA_DIR.glob("*.parquet"))
        if not seed_paths:
            return
        df = pd.read_parquet(seed_paths[0])
        if "Date" not in df.columns and df.index.name == "Date":
            df = df.reset_index()
        df = df.head(150)

        for name in resolve_order(blocks):
            blk = blocks[name]
            src = blk["path"].read_text(encoding="utf-8", errors="replace")
            if not any(h in src for h in _SHARED_HELPERS):
                continue                      # block touches no shared panel
            if any(r not in df.columns for r in blk["meta"].get("requires", [])):
                continue
            try:
                blk["fn"](df.copy())
            except Exception:
                pass                          # warm-up only
    except Exception:
        pass


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
    # Carry the output schema back so the parent can detect drift across tickers
    # without re-reading 4250 parquets.
    timing["schema"] = tuple(result.columns)
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
    full:    bool = False,
) -> None:
    """
    Debug report for --ticker mode.

    Section 1  Block run table: name | time | +N cols | first few col names
    Section 2  (--report full only) Last-row values grouped by producing block --
               one line per output column, several hundred lines on a full build.
    """
    W           = 80
    SEP         = "=" * W
    order       = timing.get("order", list(timing["blocks"].keys()))
    meta        = timing.get("meta", {})
    skip_reason = timing.get("skip_reasons", {})
    errors      = timing.get("errors", {})
    n_ran       = len(timing["blocks"])
    n_skip      = len(timing["skipped"])
    n_err       = len(errors)
    n_feat_cols = result.shape[1] - sum(1 for c in OHLCV_COLS if c in result.columns)

    # ---- section 1: block run table ------------------------------------------
    print()
    print(SEP)
    err_bit = f"  {n_err} ERRORED" if n_err else ""
    print(
        f"  {ticker}  |  {timing['n_rows']:,} rows in  |  "
        f"{result.shape[1]} cols out (+{n_feat_cols} features)  |  "
        f"{n_ran} ran  {n_skip} skipped{err_bit}"
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
            if name in errors:
                reason_str = f"ERROR: {errors[name][:60]}"
            else:
                reason_str = f"needs: {', '.join(reason)}" if reason else "excluded"
            print(f"  {'[skip] ' + name:<26}  {'':>8}  {'':>6}  {reason_str}")

    print("  " + "-" * (W - 2))
    print(
        f"  {'TOTAL':<26}  {_fmt_ms(timing['total_s'] * 1_000):>8}  "
        f"  wall: {_fmt_ms(wall_ms)}"
    )

    if len(result) == 0:
        return
    if not full:
        print("  (--report full for the last-row value dump)")
        print()
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
            if name in errors:
                reason_str = f"ERROR: {errors[name][:60]}"
            else:
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
    full:         bool = False,
) -> None:
    """
    Aggregate per-ticker timing dicts.

    Default (compact): one header plus the top blocks by total cost. A full run
    has ~190 blocks; printing a row for each, then a slow-ticker table for each,
    was ~3000 lines of terminal per run -- almost all of it about blocks costing
    single-digit ms.

    full=True restores both original sections:

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

    W = 100  # report width (block names run to ~40 chars)

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

    # Per-block stats, computed once for both report shapes
    grand_total_ms = 0.0
    block_stats: dict[str, tuple] = {}   # name -> (calls, total, mean, p50, p95, max)
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
        block_stats[bname] = (calls, total_ms, mean_ms, p50_ms, p95_ms, ms[-1])

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
        f"Speedup : {serial_equiv_s / max(wall_clock_s, 0.001):.1f}x   |   "
        f"{n_tickers / max(wall_clock_s, 0.001):.1f} tickers/s"
    )
    print("=" * W)
    print(f"  {'Block':<42}  {'Calls':>6}  {'Total':>8}  {'Mean':>8}  {'p50':>8}  {'p95':>8}  {'Max':>8}")
    print("-" * W)

    if full:
        shown = [b for b in order if b in block_stats]
    else:
        shown = sorted(block_stats, key=lambda b: -block_stats[b][1])[:top_n]

    for bname in shown:
        calls, total_ms, mean_ms, p50_ms, p95_ms, max_ms = block_stats[bname]
        print(
            f"  {bname:<42}  {calls:>6}  {_fmt_ms(total_ms):>8}  "
            f"{_fmt_ms(mean_ms):>8}  {_fmt_ms(p50_ms):>8}  "
            f"{_fmt_ms(p95_ms):>8}  {_fmt_ms(max_ms):>8}"
        )

    n_rest = len(block_stats) - len(shown)
    if n_rest > 0:
        rest_ms = grand_total_ms - sum(block_stats[b][1] for b in shown)
        print(f"  {f'... {n_rest} more blocks':<42}  {'':>6}  {_fmt_ms(rest_ms):>8}"
              f"   (--report full for all)")

    print("-" * W)
    print(f"  {'TOTAL (all blocks)':<42}  {n_tickers:>6}  {_fmt_ms(grand_total_ms):>8}")
    print("=" * W)

    if not full:
        return

    # =========================================================================
    # SECTION B -- slow ticker detail (--report full only)
    # =========================================================================
    print()
    print("=" * W)
    print(f"SLOW TICKER DETAIL  --  top {top_n} per block  (xMEAN = multiple of mean cost)")
    print("=" * W)

    for bname in order:
        if bname not in block_entries:
            continue
        entries = block_entries[bname]
        mean_ms = block_stats[bname][2] or 1.0
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

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init,
        initargs=(exclude,),
    ) as executor:
        futures = {
            executor.submit(_worker_fn, str(p), str(out_dir), exclude): p
            for p in paths
        }

        # _quiet_progress.tqdm never redraws: one plain line per 5% of tickers,
        # so a full run costs ~20 progress lines whether the output is a live
        # terminal or a captured nightly log.
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

                if fail:
                    pbar.set_postfix(ok=ok, fail=fail)
                pbar.update(1)

    print(f"\nCompleted: {ok} ok, {fail} failed, {len(paths)} total")
    _print_integrity_report(all_timings)
    return all_timings


def _integrity_summary(timings: list[dict]):
    """The two tallies _print_integrity_report prints, as data, so the diagnostics
    sidecar and the printed report cannot disagree."""
    err_tickers: dict[str, list[str]] = defaultdict(list)
    err_first:   dict[str, str] = {}
    for t in timings:
        for bname, msg in t.get("errors", {}).items():
            err_tickers[bname].append(t.get("ticker", "?"))
            err_first.setdefault(bname, msg)
    schemas: dict[tuple, list[str]] = defaultdict(list)
    for t in timings:
        sc = t.get("schema")
        if sc:
            schemas[sc].append(t.get("ticker", "?"))
    return err_tickers, err_first, schemas


def _integrity_payload(timings: list[dict]) -> dict:
    err_tickers, err_first, schemas = _integrity_summary(timings)
    ranked = sorted(schemas.items(), key=lambda kv: -len(kv[1]))
    majority = set(ranked[0][0]) if ranked else set()
    skip_reasons: dict[str, int] = defaultdict(int)
    for t in timings:
        for bname in t.get("skipped", []) or []:
            skip_reasons[str(bname)] += 1
    return {
        "errors": {b: {"n": len(tks), "example": tks[0], "first_msg": err_first[b][:200]}
                   for b, tks in err_tickers.items()},
        "schemas": [{"n_tickers": len(tks), "n_cols": len(cols), "example": tks[0],
                     "missing_vs_majority": sorted(majority - set(cols))[:40],
                     "extra_vs_majority": sorted(set(cols) - majority)[:40]}
                    for cols, tks in ranked],
        "skip_reasons": dict(skip_reasons),
    }


def _print_integrity_report(timings: list[dict]) -> None:
    """
    Two things that used to fail silently across a 4250-ticker run:

      BLOCK ERRORS -- a block raising on a subset of tickers printed to a worker's
      stderr under the tqdm bar and was otherwise indistinguishable from a
      deliberate --exclude. Here every erroring block is tallied with its ticker
      count and first message.

      SCHEMA DRIFT -- when a block errors on some tickers and not others, those
      parquets end up with different column sets. Nothing checked. Downstream that
      surfaces in 4__Predictor, far from the cause.
    """
    if not timings:
        return

    err_tickers, err_first, schemas = _integrity_summary(timings)

    if err_tickers:
        n = len(timings)
        print()
        print("=" * 85)
        print(f"BLOCK ERRORS  --  {len(err_tickers)} block(s) raised on at least one ticker")
        print("=" * 85)
        for bname, tks in sorted(err_tickers.items(), key=lambda kv: -len(kv[1])):
            print(f"  {bname:<40} {len(tks):>5}/{n} tickers   e.g. {tks[0]}")
            print(f"  {'':<40} {err_first[bname][:100]}")

    if len(schemas) > 1:
        ranked = sorted(schemas.items(), key=lambda kv: -len(kv[1]))
        majority_cols, majority_tks = ranked[0]
        print()
        print("=" * 85)
        print(f"SCHEMA DRIFT  --  {len(schemas)} distinct column sets across {len(timings)} tickers")
        print("=" * 85)
        print(f"  majority ({len(majority_tks)} tickers): {len(majority_cols)} columns")
        for cols, tks in ranked[1:6]:
            missing = sorted(set(majority_cols) - set(cols))
            extra   = sorted(set(cols) - set(majority_cols))
            print(f"  {len(tks):>5} ticker(s) differ ({len(cols)} cols) e.g. {tks[0]}")
            if missing:
                print(f"        missing: {', '.join(missing[:8])}{' ...' if len(missing) > 8 else ''}")
            if extra:
                print(f"        extra  : {', '.join(extra[:8])}{' ...' if len(extra) > 8 else ''}")
        print("  -> downstream concat/predict will see ragged columns. Fix the erroring block above.")
    elif schemas:
        print(f"Schema: uniform across all {len(timings)} tickers "
              f"({len(next(iter(schemas)))} columns).")


# ===========================================================================
# SECTION 8 -- CLI helpers
# ===========================================================================

def _verify_metadata(ticker: str | None, exclude: list[str]) -> None:
    """
    Compare each block's declared METADATA["produces"] against the columns it
    actually emits, on one real ticker.

    Why this matters beyond tidiness: __tail_screen.py builds its feature list from
    `produces`, so a column that is declared but never emitted silently drops out of
    the screen -- the feature looks like it was tested when it never was. The reverse
    (emitted but undeclared) means a column nothing will ever screen.
    """
    ticker = ticker or "AAPL"
    path = PRICE_DATA_DIR / f"{ticker}.parquet"
    if not path.exists():
        candidates = sorted(PRICE_DATA_DIR.glob("*.parquet"))
        if not candidates:
            sys.exit(f"No parquets in {PRICE_DATA_DIR}")
        path = candidates[0]
        ticker = path.stem

    df = pd.read_parquet(path)
    if "Date" not in df.columns and df.index.name == "Date":
        df = df.reset_index()

    blocks = {k: v for k, v in discover_blocks().items() if k not in set(exclude or [])}
    order  = resolve_order(blocks)

    print(f"\nVerifying METADATA against actual output on {ticker} ({len(df):,} rows)\n")
    ghost: dict[str, list[str]] = {}
    undeclared: dict[str, list[str]] = {}

    for name in order:
        blk = blocks[name]
        if any(r not in df.columns for r in blk["meta"].get("requires", [])):
            continue
        cols_before = set(df.columns)
        try:
            out = blk["fn"](df.copy())
        except Exception as exc:
            print(f"  [ERROR] {name}: {type(exc).__name__}: {exc}")
            continue
        emitted  = set(out.columns) - cols_before
        declared = set(blk["meta"].get("produces", []))
        if declared - emitted:
            ghost[name] = sorted(declared - emitted)
        if emitted - declared:
            undeclared[name] = sorted(emitted - declared)
        df = out

    w = 85
    print("=" * w)
    print(f"DECLARED BUT NEVER EMITTED  --  {sum(len(v) for v in ghost.values())} column(s) "
          f"across {len(ghost)} block(s)")
    print("=" * w)
    for name, cols in sorted(ghost.items()):
        print(f"  {name:<34} {', '.join(cols)}")
    print()
    print("=" * w)
    print(f"EMITTED BUT NOT DECLARED  --  {sum(len(v) for v in undeclared.values())} column(s) "
          f"across {len(undeclared)} block(s)")
    print("=" * w)
    for name, cols in sorted(undeclared.items()):
        print(f"  {name:<34} {', '.join(cols[:8])}{' ...' if len(cols) > 8 else ''}")
    if not ghost and not undeclared:
        print("\nAll blocks: METADATA matches emitted columns exactly.")
    print()


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
    _silence_noisy_warnings()
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
                        help="Percentage of tickers to process in --all mode (default: 100). "
                             "Taken as a seeded RANDOM sample, not an alphabetical prefix.")
    parser.add_argument("--sample_seed", type=int, default=42,
                        help="Seed for --runpercent ticker sampling (default: 42)")
    parser.add_argument("--verify_metadata", action="store_true",
                        help="Run one ticker and report blocks whose METADATA['produces'] "
                             "does not match the columns they actually emit, then exit")
    parser.add_argument("--report",      choices=["compact", "full"], default="compact",
                        help="Report size. compact (default): top blocks by total cost only. "
                             "full: every block, slow-ticker detail, last-row dump in --ticker mode.")
    parser.add_argument("--top_n",       type=int, default=10,
                        help="How many blocks (compact) / slow tickers per block (full) to show "
                             "(default: 10)")
    parser.add_argument("--save_timing", metavar="PATH",
                        help="Save full per-ticker timing matrix to this CSV path")
    args = parser.parse_args()

    # ---- list ---------------------------------------------------------------
    if args.list:
        _list_blocks()
        return

    # ---- metadata verification ----------------------------------------------
    if args.verify_metadata:
        _verify_metadata(args.ticker, args.exclude)
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

        # Live per-block lines duplicate the block table printed right after, so
        # they are full-report-only; compact mode prints each block exactly once.
        t_wall = time.perf_counter()
        result, timing = run_pipeline_timed(df, exclude=args.exclude,
                                            verbose=(args.report == "full"))
        wall_ms = (time.perf_counter() - t_wall) * 1_000

        _print_ticker_report(args.ticker, result, timing, wall_ms,
                             full=(args.report == "full"))
        return

    # ---- batch (parallel) ---------------------------------------------------
    if args.all:
        all_paths = sorted(PRICE_DATA_DIR.glob("*.parquet"))
        if not all_paths:
            sys.exit(f"No parquets found in {PRICE_DATA_DIR}")

        n = max(1, int(len(all_paths) * args.runpercent / 100))
        if n >= len(all_paths):
            paths = all_paths
        else:
            # A prefix slice (all_paths[:n]) gives you tickers A-B, not a 10% sample.
            # Every screening tool in FeatureTemplates/ uses a seeded random sample;
            # match that so partial runs are representative and reproducible.
            paths = sorted(random.Random(args.sample_seed).sample(all_paths, n))
            print(f"Sampling {n}/{len(all_paths)} tickers (seed {args.sample_seed})")
        n_work = min(args.workers, len(paths))
        out    = Path(args.out_dir)

        active_blocks = [b for b in discover_blocks() if b not in args.exclude]
        print(f"FeatureFramework  --  {len(paths)} tickers | {n_work} workers | -> {out}")
        excluded = [b for b in args.exclude] if args.exclude else []
        print(f"Active blocks: {len(active_blocks)}"
              + (f"  |  excluded: {', '.join(excluded)}" if excluded else "")
              + "   (--list for the full roster)")
        print()

        t0         = time.perf_counter()
        all_timing = process_all(paths, out, args.exclude, n_work)
        wall_s     = time.perf_counter() - t0

        if _diag.enabled():
            try:
                _diag.stamp("3__FeatureFramework")
                _payload = _integrity_payload(all_timing)
                _payload.update({"n_paths": len(paths), "n_workers": n_work, "exclude": list(args.exclude or []),
                                 "wall_s": wall_s, "active_blocks": active_blocks,
                                 "n_ok": sum(1 for t in all_timing if t), "out_dir": str(out),
                                 "runpercent": args.runpercent})
                _diag.dump_json("F_run", _payload)
                _save_timing_csv(all_timing, os.path.join(_diag.outdir(), "F_timing.csv"))
            except Exception:
                pass

        _print_timing_report(all_timing, wall_s, n_work, top_n=args.top_n,
                             full=(args.report == "full"))

        if args.save_timing:
            _save_timing_csv(all_timing, args.save_timing)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
