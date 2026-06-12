"""
run_pipeline.py — Smart end-to-end pipeline runner.

Checks whether each stage's output is "fresh" (produced within FRESHNESS_HOURS)
and offers to skip already-done stages or run only what's needed.

Usage:
    python run_pipeline.py                   # interactive — shows state, prompts
    python run_pipeline.py --auto            # run stale stages without prompting
    python run_pipeline.py --force           # run all stages regardless of freshness
    python run_pipeline.py --only 4 5        # run only predictor + backtester
    python run_pipeline.py --skip 2 3        # skip price + alpha
    python run_pipeline.py --dry-run         # show state without executing
    python run_pipeline.py --retrain         # full retrain for stage 4 (slow, ~hours)
    python run_pipeline.py --freshness 12    # treat outputs older than 12h as stale
"""

import argparse
import datetime
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE = Path(__file__).parent
PYTHON = BASE / "stock_env" / "Scripts" / "python.exe"
FRESHNESS_HOURS = 20   # within this many hours = "fresh" (covers overnight cycle)

# Stage 4 args: "--predict_only" for fast nightly inference (uses saved model).
# Pass --retrain to the pipeline runner to do a full train + tune instead.
_STAGE4_ARGS_DAILY = "--predict_only"
_STAGE4_ARGS_RETRAIN = "--runpercent 75"   # full Optuna train (~hours)

STAGES = [
    {
        "id": 2,
        "name": "Price Downloader",
        "script": "2__PriceDownloader.py",
        "args": "--RefreshMode",
        "log":  "Data/logging/2__BulkPriceDownloader.log",   # what the script actually writes
        "output": "Data/PriceData",                            # dir with .parquet files
    },
    {
        "id": 3,
        "name": "Alpha Sensitivity",
        "script": "3__AlphaSensitivity.py",
        "args": "--runpercent 100",
        "log":  "Data/logging/3__Indicators.log",
        "output": "Data/ProcessedData",
    },
    {
        "id": 4,
        "name": "Predictor",
        "script": "4__Predictor.py",
        "args": _STAGE4_ARGS_DAILY,   # overridden below if --retrain
        "log":  None,                  # 4__Predictor.py logs to stdout only
        "output": "Data/RFpredictions",
    },
    {
        "id": 5,
        "name": "Nightly BackTester",
        "script": "5__NightlyBackTester.py",
        "args": "--force",
        "log":  "Data/logging/5__NightlyBackTester.log",
        "output": "Z_signals.parquet",   # single file (not a dir)
    },
]

# ---------------------------------------------------------------------------
# Freshness helpers
# ---------------------------------------------------------------------------

def _newest_mtime(path: Path) -> datetime.datetime | None:
    """Return the newest mtime in a directory (parquets) or a single file."""
    if not path.exists():
        return None
    if path.is_file():
        return datetime.datetime.fromtimestamp(path.stat().st_mtime)
    # directory — look at .parquet files only (fast, avoids subdirs)
    mtimes = [
        f.stat().st_mtime
        for f in path.iterdir()
        if f.suffix == ".parquet" and f.is_file()
    ]
    if not mtimes:
        return None
    return datetime.datetime.fromtimestamp(max(mtimes))


def check_stage(stage: dict, freshness_hours: int) -> tuple[bool, str]:
    """Return (is_fresh, human-readable detail string)."""
    now = datetime.datetime.now()
    cutoff = now - datetime.timedelta(hours=freshness_hours)

    log_time = _newest_mtime(BASE / stage["log"]) if stage["log"] else None
    out_time = _newest_mtime(BASE / stage["output"])

    candidates = [(t, src) for t, src in [(log_time, "log"), (out_time, "output")] if t]
    if not candidates:
        return False, "no output found — never run?"

    best_time, best_src = max(candidates, key=lambda x: x[0])
    age_h = (now - best_time).total_seconds() / 3600
    ts = best_time.strftime("%Y-%m-%d %H:%M")
    is_fresh = best_time > cutoff
    return is_fresh, f"{ts}  ({age_h:.1f}h ago, {best_src})"

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

SEP = "-" * 62
SEP2 = "=" * 62

def print_state(stages: list[dict], freshness_hours: int) -> list[dict]:
    """Print the status table. Returns the list of stale stages."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{SEP2}")
    print(f"  PIPELINE STATE  --  {now}  (window: {freshness_hours}h)")
    print(SEP2)

    stale = []
    for s in stages:
        fresh, detail = check_stage(s, freshness_hours)
        badge = " FRESH " if fresh else " STALE "
        print(f"  [{s['id']}]  {badge}  {s['name']:<22}  {detail}")
        if not fresh:
            stale.append(s)

    print(SEP2)
    n_fresh = len(stages) - len(stale)
    print(f"  Fresh: {n_fresh}  |  Stale: {len(stale)}")
    print(f"{SEP2}\n")
    return stale


def _stage_label(stages: list[dict]) -> str:
    return ", ".join(str(s["id"]) for s in stages)

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def run_stage(stage: dict, dry_run: bool = False) -> bool:
    """Run one pipeline stage. Returns True on success."""
    script = BASE / stage["script"]
    args = stage["args"].split() if stage["args"].strip() else []
    cmd = [str(PYTHON), str(script)] + args

    print(f"\n{SEP}")
    print(f"  STAGE [{stage['id']}]  {stage['name']}")
    print(f"  CMD:  {' '.join(cmd)}")
    print(f"{SEP}\n")

    if dry_run:
        print("  [DRY RUN - skipping execution]\n")
        return True

    t0 = time.monotonic()
    try:
        result = subprocess.run(cmd, cwd=str(BASE))
        elapsed = time.monotonic() - t0
        ok = result.returncode == 0
        label = "SUCCESS" if ok else f"FAILED (exit {result.returncode})"
        print(f"\n  [{stage['id']}] {stage['name']}: {label}  ({elapsed:.0f}s)")
        return ok
    except Exception as exc:
        print(f"\n  [{stage['id']}] {stage['name']}: ERROR — {exc}")
        return False


def prompt_choice(stages: list[dict], stale: list[dict]) -> list[dict] | None:
    """
    Interactive prompt. Returns list of stages to run, or None to abort.
    """
    if not stale:
        resp = input("  All stages appear fresh. Run anyway? [y/N] ").strip().lower()
        return stages if resp == "y" else None

    ids_str = _stage_label(stale)
    print(f"  Stale stages: {ids_str}")
    print("  [Enter] run stale  |  [a] run all  |  [2 3 4 5] run specific  |  [q] quit")
    resp = input("  > ").strip().lower()

    if resp in ("q", "quit"):
        return None
    if resp == "a":
        return stages
    if resp == "":
        return stale
    # parse stage IDs
    try:
        chosen_ids = {int(x) for x in resp.split()}
        chosen = [s for s in stages if s["id"] in chosen_ids]
        if not chosen:
            print("  No matching stages. Aborting.")
            return None
        return chosen
    except ValueError:
        print("  Could not parse input — running stale stages.")
        return stale

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Smart pipeline runner — detect and run stale stages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--auto", "-a", action="store_true",
                        help="Auto-run stale stages without prompting")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Run all stages regardless of freshness")
    parser.add_argument("--only", type=int, nargs="+", metavar="N",
                        help="Run only these stage IDs (e.g. --only 4 5)")
    parser.add_argument("--skip", type=int, nargs="+", metavar="N",
                        help="Skip these stage IDs (e.g. --skip 2 3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without executing")
    parser.add_argument("--retrain", action="store_true",
                        help="Stage 4: full Optuna retrain instead of --predict_only (~hours)")
    parser.add_argument("--freshness", type=int, default=FRESHNESS_HOURS, metavar="H",
                        help=f"Hours within which output is considered fresh (default {FRESHNESS_HOURS})")
    args = parser.parse_args()

    # Apply --retrain to stage 4
    if args.retrain:
        for s in STAGES:
            if s["id"] == 4:
                s["args"] = _STAGE4_ARGS_RETRAIN
                break

    # Apply --only / --skip to select stages
    stages = list(STAGES)
    if args.only:
        ids = set(args.only)
        stages = [s for s in stages if s["id"] in ids]
        unknown = ids - {s["id"] for s in STAGES}
        if unknown:
            print(f"  WARNING: unknown stage IDs: {sorted(unknown)}")
    if args.skip:
        stages = [s for s in stages if s["id"] not in set(args.skip)]

    if not stages:
        print("  No stages selected. Exiting.")
        return

    # Show state
    stale = print_state(stages, args.freshness)

    # Decide what to run
    if args.force:
        to_run = stages
        print(f"  --force: running all {len(to_run)} stage(s).")
    elif args.auto:
        to_run = stale
        if not to_run:
            print("  All stages fresh — nothing to do. Use --force to override.")
            return
        print(f"  --auto: running {len(to_run)} stale stage(s): {_stage_label(to_run)}")
    elif args.dry_run:
        to_run = stale if stale else stages
        print(f"  --dry-run: would run stages: {_stage_label(to_run)}")
    else:
        to_run = prompt_choice(stages, stale)
        if to_run is None:
            print("  Aborted.")
            return
        print(f"  Running stages: {_stage_label(to_run)}")

    # Execute
    results: dict[int, bool] = {}
    t_pipeline = time.monotonic()

    for i, stage in enumerate(to_run):
        ok = run_stage(stage, dry_run=args.dry_run)
        results[stage["id"]] = ok

        if not ok and i < len(to_run) - 1:
            resp = input(f"\n  Stage {stage['id']} failed. Continue to next stage? [y/N] ").strip().lower()
            if resp != "y":
                print("  Pipeline aborted.")
                break

        # Brief RAM-cleanup pause between stages (skip after last)
        if not args.dry_run and i < len(to_run) - 1:
            print("  Waiting 10s (RAM cleanup)...")
            time.sleep(10)

    # Summary
    total = time.monotonic() - t_pipeline
    print(f"\n{SEP2}")
    print(f"  PIPELINE SUMMARY  ({total:.0f}s total)")
    print(SEP2)
    for stage in to_run:
        sid = stage["id"]
        ok = results.get(sid)
        if ok is None:
            mark = " - "
        elif ok:
            mark = " OK"
        else:
            mark = "ERR"
        print(f"  [{mark}]  [{sid}] {stage['name']}")
    print(f"{SEP2}\n")


if __name__ == "__main__":
    main()
