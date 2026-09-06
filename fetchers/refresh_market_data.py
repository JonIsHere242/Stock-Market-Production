"""Refresh every market/macro data lake. The missing nightly stage.

WHY THIS FILE EXISTS
--------------------
The nightly (`trading_system.ps1`) had no stage that refreshed market or macro
data. Its "Data Panels" stage runs `build_data_panels.py all --refresh-all`,
which delegates to exactly three SEC fetchers (companyfacts, insider,
submissions). Everything else in `fetchers/` was only ever run by hand -- so on
2026-07-28 a census found every one of those lakes frozen on 2026-05-28, the day
they were last invoked manually. `Data/Indexes` alone had been dead for ~17
trading sessions, silently taking ~88 index-relative panel columns to all-NaN
and leaving the predictor's neutralization beta leg inert.

This is the orchestrator that closes that gap. Pair it with `lake_freshness.py`,
which is the fail-closed gate that catches it if this ever stops working.

DESIGN
------
* SKIP WHEN FRESH. Each lake is asked (via `lake_freshness`) whether it is
  already current before its fetcher is run, matching the convention
  `--refresh-all` already uses ("fetchers self-skip files <20h old"). `--force`
  overrides. This keeps a nightly invocation cheap and idempotent.

* CONTINUE ON FAILURE, THEN REPORT. One dead upstream endpoint must not stop
  the other eight lakes from refreshing. Every failure is collected and printed
  in a summary, and the exit code reflects whether anything CRITICAL is still
  stale afterwards -- so the nightly's own stage-failure alerting fires.

* FRED NEEDS --force. `fetch_fred.py` returns early for any series already on
  disk unless `--force` is passed, even though its docstring claims re-running
  refreshes. A plain re-run is a silent no-op -- one of the two reasons the FRED
  lake sat 60 days stale with 30 of its 52 series missing entirely. So the FRED
  entry here always passes --force when it decides a refresh is due.

* VERIFY, DON'T ASSUME. After refreshing, the lakes are re-checked and the
  post-state is printed. "The fetcher exited 0" is not evidence the data moved;
  the price-downloader stage in the nightly already carries a guard for exactly
  that failure mode ("exited 0 but downloaded NOTHING").

USAGE
-----
    python fetchers/refresh_market_data.py              # skip-when-fresh
    python fetchers/refresh_market_data.py --force      # refresh everything
    python fetchers/refresh_market_data.py --only Indexes,FRED
    python fetchers/refresh_market_data.py --dry-run    # show the plan
    python fetchers/refresh_market_data.py --strict     # exit 1 if any lake
                                                        # (not just critical)
                                                        # is stale afterwards
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

FETCHERS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FETCHERS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common import log  # noqa: E402  (fetchers/common.py)

from auxiliary import lake_freshness as lf  # noqa: E402


# lake name -> argv for its fetcher. Order is deliberate: cheap/critical first,
# so a long tail (FINRA walks 17k+ files) cannot delay the lake that actually
# breaks the panel when stale.
PLAN: list[tuple[str, list[str]]] = [
    ("Indexes",     ["fetch_indexes.py", "--lake", "recent"]),
    ("IndexesFull", ["fetch_indexes.py", "--lake", "full"]),
    # --force is MANDATORY here, not optional -- see module docstring.
    ("FRED",        ["fetch_fred.py", "--force"]),
    ("Treasury",    ["fetch_treasury.py"]),
    ("Shiller",     ["fetch_shiller.py"]),
    ("CFTC_COT",    ["fetch_cftc_cot.py"]),
    ("KenFrench",   ["fetch_kenfrench.py"]),
    ("FINRA",       ["fetch_finra_shorts.py"]),
]

# Lakes with no fetcher in this repo. Listed so they are visible rather than
# quietly absent from the plan.
UNMANAGED = {
    "News": "no fetcher; Data/News last written 2025-11-30 (239d). "
            "Not consumed by production.",
    "ShortInterest": "directory is EMPTY and no fetcher writes it. "
                     "FINRA short VOLUME (Data/FINRA) is the live source.",
    # Corrected 2026-09-03: build_data_panels.py does NOT rebuild this. It has
    # no `marketcaps` subcommand and `all` does not build it either, so nothing
    # in the nightly refreshes it at all -- it was 41 days stale when this was
    # checked. The only writer is auxiliary/0__ApproximateMarketCaps.py, by hand.
    "MarketCaps": "NO automated writer. Rebuild by hand: python "
                  "auxiliary/0__ApproximateMarketCaps.py. Backs the micro-cap "
                  "gate's PIT fallback; known large per-name error.",
}


def _run(argv: list[str], timeout: int) -> tuple[bool, str]:
    try:
        p = subprocess.run([sys.executable, *argv], cwd=str(FETCHERS_DIR),
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {timeout}s"
    if p.returncode != 0:
        tail = (p.stderr or p.stdout or "").strip().splitlines()
        return False, f"exit {p.returncode}: " + (tail[-1] if tail else "no output")
    tail = (p.stdout or "").strip().splitlines()
    return True, (tail[-1] if tail else "ok")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true",
                    help="refresh every lake even if it looks current")
    ap.add_argument("--only", help="comma-separated subset of lake names")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if ANY lake is stale afterwards, not just critical ones")
    ap.add_argument("--timeout", type=int, default=1800,
                    help="per-fetcher timeout in seconds (default 1800)")
    a = ap.parse_args()

    only = {s.strip() for s in a.only.split(",")} if a.only else None
    plan = [(n, c) for n, c in PLAN if only is None or n in only]
    if only:
        unknown = only - {n for n, _ in PLAN}
        if unknown:
            log(f"unknown lake(s) {sorted(unknown)}; known: {[n for n, _ in PLAN]}")
            return 2

    log("=== pre-refresh state ===")
    pre = {r["lake"]: r for r in lf.check_all([n for n, _ in plan])}
    lf._print_table(list(pre.values()))

    results: list[tuple[str, str, str]] = []  # (lake, outcome, detail)
    log("")
    log("=== refreshing ===")
    for name, argv in plan:
        rep = pre.get(name, {})
        # SKIP ONLY ON A DEFINITE "caught up to the last completed session".
        #
        # This used to read `if st == "OK"`, i.e. it reused check_lake's
        # staleness verdict as a currency check. Those are different questions:
        # check_lake asks "is this ROTTEN?" and tolerates max(floor_days,
        # slack*gap) -- FIVE DAYS for Indexes. So the one critical lake could be
        # three sessions behind and still be skipped, every night, forever.
        # Observed 2026-08-31 (a Monday, after the close):
        #     Indexes  YES  OK  2026-08-28  3  5.0  data-date
        #       Indexes      SKIP (already current)
        # and the Feature Framework then built its ~88 alpha_*/beta_*/corr_*
        # columns on a three-session-stale basis with no error anywhere.
        #
        # `None` means "cannot tell" (an mtime-basis lake: KenFrench, CFTC_COT,
        # Shiller, FINRA). For those we fall back to the old rotten-check,
        # because mtime cannot answer the session question and re-pulling a
        # monthly series nightly is pure waste. That fallback is still weak for
        # FINRA, which is daily; it is fixed properly in 2__MacroFetcher.py,
        # where skipping is opt-in per source.
        current = lf.is_current_for_session(rep)
        if not a.force and (current is True
                            or (current is None and rep.get("status") == "OK")):
            why = ("caught up to last session" if current
                   else "not stale (mtime basis)")
            log(f"  {name:12s} SKIP ({why})")
            results.append((name, "SKIP", why))
            continue
        if a.dry_run:
            log(f"  {name:12s} WOULD RUN  {' '.join(argv)}")
            results.append((name, "DRY", " ".join(argv)))
            continue
        log(f"  {name:12s} RUN   {' '.join(argv)}")
        ok, detail = _run(argv, a.timeout)
        log(f"  {name:12s} {'ok' if ok else 'FAIL'}  {detail}")
        results.append((name, "OK" if ok else "FAIL", detail))

    if a.dry_run:
        return 0

    log("")
    log("=== post-refresh state ===")
    post = lf.check_all([n for n, _ in plan])
    lf._print_table(post)

    # A fetcher exiting 0 is not evidence the data moved. Report lakes whose
    # newest date did not advance despite a successful run.
    log("")
    log("=== summary ===")
    failed = [n for n, o, _ in results if o == "FAIL"]
    post_by = {r["lake"]: r for r in post}
    not_moved = []
    for name, outcome, _ in results:
        if outcome != "OK":
            continue
        b, af = pre.get(name, {}).get("newest"), post_by.get(name, {}).get("newest")
        if b is not None and af is not None and af <= b:
            not_moved.append(f"{name} (still {af})")
    for name, outcome, detail in results:
        log(f"  {name:12s} {outcome:5s} {detail}")
    if failed:
        log(f"  FAILED: {failed}")
    if not_moved:
        log(f"  RAN BUT DATA DID NOT ADVANCE: {not_moved}")
        log("    (upstream may simply have no new observation yet -- but if this "
            "persists, the fetcher is silently no-opping.)")
    for name, why in UNMANAGED.items():
        log(f"  {name:12s} UNMANAGED  {why}")

    stale = [r["lake"] for r in post if r["status"] != "OK"
             and (a.strict or r["critical"])]
    if stale:
        log("")
        log(f"  STALE AFTER REFRESH: {stale} -> exiting non-zero so the nightly "
            f"reports this stage failed.")
        return 1
    log("  all checked lakes current.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
