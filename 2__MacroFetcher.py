#!/usr/bin/env python
"""2__MacroFetcher.py -- the single entry point for every NON-PRICE data source.

WHY THIS FILE EXISTS
--------------------
Data acquisition was spread across four pipeline stages and nothing said they
belonged together:

    1__TickerDownloader.py --ImmediateDownload      SEC ticker/CIK universe
    2__PriceDownloader.py --RefreshMode             IBKR daily OHLCV
    fetchers/refresh_market_data.py                 8 market/macro lakes
    build_data_panels.py all --refresh-all          3 SEC pulls + 5 panels

To answer "where does Data/FRED come from" you had to know the third stage
existed in a subdirectory. To answer "where does Data/Insider come from" you had
to know the fourth hid three fetcher subprocesses behind a --refresh-all flag.
The only place the full source list was written down was a PowerShell array.

This file collapses the last two into one declarative MANIFEST. The `N__` prefix
is a PHASE MARKER, not a sequence index: `2__` is the acquisition phase and owns
more than one file. `2__PriceDownloader.py` keeps its own, because prices are
the one source with a hard external dependency (a live IBKR session) and a
30-minute pre-flight wait in the runner.

WHAT THIS IS NOT
----------------
It is not a code merge. Every source is still a subprocess running the same
script with the same argv it ran before. That is deliberate: subprocess
isolation is load-bearing here. 2__PriceDownloader.py calls os.chdir() and
nest_asyncio.apply() at module level, 4__Predictor.py calls parse_args() at
module level, and the numbered filenames are not importable identifiers anyway.

DESIGN
------
* SKIP ONLY WHAT IS PROVABLY CURRENT. Skipping is opt-in per step
  (`skip_when_fresh`), and the predicate is `lake_freshness.is_current_for_session`
  -- "has this caught up to the last completed NYSE session" -- NOT check_lake's
  "is this rotten", which tolerates five days. Conflating the two is what let
  the old orchestrator skip the one critical lake while it sat three sessions
  behind, every night. See lake_freshness.is_current_for_session for the
  evidence.

* STREAM, NEVER CAPTURE. refresh_market_data.py used
  subprocess.run(capture_output=True) and reported only the last stdout line.
  Applied to the 26-minute panel build that is 26 minutes of silent console and
  a log that appears only at the end -- and the SEC insider 404 was visible
  nowhere else. Every step is teed line by line to its own transcript.

* CONTINUE ON FAILURE, THEN REPORT. One dead endpoint must not cost the other
  ten sources. A step whose dependency failed is BLOCKED, not silently passed.

* VERIFY, DON'T ASSUME. "The fetcher exited 0" is not evidence the data moved.
  fetch_sec_insider.py exits 0 on a 404, and the insider lake has been five
  months stale because of it. Lakes are re-checked afterwards and any step that
  succeeded without advancing its data is reported.

USAGE
-----
    python 2__MacroFetcher.py                     # nightly
    python 2__MacroFetcher.py --dry-run           # print the plan, run nothing
    python 2__MacroFetcher.py --list              # steps + last known durations
    python 2__MacroFetcher.py --group market      # market | sec | panel
    python 2__MacroFetcher.py --only FRED,Indexes # named steps, ignores skips
    python 2__MacroFetcher.py --force             # refresh everything
    python 2__MacroFetcher.py --strict            # fail on ANY stale lake

A full run is 30-45 minutes (the companyfacts pull and the panel rebuild
dominate), so it is an unattended/nightly job. Use --group or --only for
anything interactive.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FETCHERS = ROOT / "fetchers"
LOGS = ROOT / "Data" / "logging"
TIMING_CSV = LOGS / "_macrofetch_timing.csv"

sys.path.insert(0, str(ROOT))
from auxiliary import lake_freshness as lf  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# ===========================================================================
# The manifest
# ===========================================================================

@dataclass(frozen=True)
class Step:
    """One acquisition step.

    name             transcript stem and --only key. NO SPACES: it becomes a
                     filename, and trading_system.ps1 splits stage args on " ".
    argv             argv after sys.executable; argv[0] is relative to `cwd`.
    cwd              load-bearing. Several scripts write RELATIVE output paths,
                     so the wrong cwd silently writes into the wrong tree.
    writes           lake_freshness.LAKES keys this step is expected to advance.
    depends_on       step names that must have succeeded first.
    critical         a failure here makes the whole stage exit non-zero.
    timeout          seconds. Set from observed history, not a global default:
                     refresh_market_data's 1800 would kill the real panel build.
    skip_when_fresh  opt-IN. Only ever consulted via is_current_for_session.
    """
    name: str
    argv: tuple[str, ...]
    cwd: Path
    group: str
    writes: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    critical: bool = False
    timeout: int = 1800
    skip_when_fresh: bool = False
    notes: str = ""


MANIFEST: tuple[Step, ...] = (
    # -- market / macro ----------------------------------------------------
    # Order lifted verbatim from refresh_market_data.PLAN and deliberate:
    # cheap and critical first, so FINRA's long tail cannot delay the lake that
    # actually breaks the feature panel when it is stale.
    Step("Indexes", ("fetch_indexes.py", "--lake", "recent"), FETCHERS, "market",
         writes=("Indexes",), critical=True,
         notes="~88 alpha_*/beta_*/corr_* panel columns and the predictor's "
               "spy200v2 neutralization beta leg read this."),
    Step("IndexesFull", ("fetch_indexes.py", "--lake", "full"), FETCHERS, "market",
         writes=("IndexesFull",)),
    # --force is MANDATORY, not optional: fetch_fred.py returns early for any
    # series already on disk without it, so a plain re-run is a silent no-op.
    # That is one of the two reasons the FRED lake once sat 60 days stale with
    # 30 of its 52 series missing outright.
    Step("FRED", ("fetch_fred.py", "--force"), FETCHERS, "market",
         writes=("FRED",)),
    Step("Treasury", ("fetch_treasury.py",), FETCHERS, "market",
         writes=("Treasury",)),
    Step("Shiller", ("fetch_shiller.py",), FETCHERS, "market",
         writes=("Shiller",), skip_when_fresh=True),
    Step("CFTC_COT", ("fetch_cftc_cot.py",), FETCHERS, "market",
         writes=("CFTC_COT",), skip_when_fresh=True),
    Step("KenFrench", ("fetch_kenfrench.py",), FETCHERS, "market",
         writes=("KenFrench",), skip_when_fresh=True),
    Step("FINRA", ("fetch_finra_shorts.py",), FETCHERS, "market",
         writes=("FINRA",), timeout=3600),

    # -- SEC raw -----------------------------------------------------------
    # Hoisted OUT of build_data_panels.py --refresh-all, whose _cmd_all catches
    # every fetcher exception into a print() and builds panels from whatever
    # stale raw is on disk, exiting 0. As real steps they get exit codes,
    # transcripts, and the did-not-advance check. argv matches what
    # build_data_panels.py passed, exactly.
    Step("SecInsiderRaw", ("fetch_sec_insider.py",), FETCHERS, "sec",
         writes=("SEC_insider",),
         notes="!! 2026q2/q3 currently 404 and the fetcher logs 'skip' and "
               "exits 0. Expect a did-not-advance report until the endpoint "
               "is fixed."),
    Step("SecSubmissionsRaw", ("fetch_sec_submissions.py", "--extract"),
         FETCHERS, "sec", writes=("SEC_submissions",), timeout=5400),
    Step("SecCompanyfactsRaw", ("fetch_sec_companyfacts.py", "--extract"),
         FETCHERS, "sec", writes=("SEC_companyfacts",), timeout=5400),

    # -- derived panels ----------------------------------------------------
    # `all` WITHOUT --refresh-all: the flag's only effect was the three fetches
    # above. Kept as ONE step on purpose -- _cmd_all fixes the internal order
    # sector -> insider -> fundamentals -> filing_meta -> filing_calendar, and
    # the last two read the ticker<->cik map out of fundamentals_panel.parquet,
    # so they cannot be split apart.
    Step("DataPanels", ("build_data_panels.py", "all"), ROOT, "panel",
         writes=("SectorMap", "Insider", "Fundamentals"),
         depends_on=("SecInsiderRaw", "SecSubmissionsRaw", "SecCompanyfactsRaw"),
         critical=True, timeout=7200,
         notes="also needs Data/PriceData populated by the 2__PriceDownloader "
               "stage: tradeable_universe() reads the PriceData stems."),
)

# Lakes with no fetcher anywhere in the live tree. Listed so they are visible
# rather than quietly absent from the plan.
UNMANAGED = {
    "MarketCaps": "NO automated writer -- build_data_panels.py has no "
                  "`marketcaps` subcommand and `all` does not build it. "
                  "Rebuild by hand: python auxiliary/0__ApproximateMarketCaps.py",
    "PriceDataFull": "NO live writer; downloader is in _ATTIC. Read by "
                     "7__MacroFilter.py, 10__FinalSolution.py and several "
                     "FeatureTemplates.",
    "PhxPanels": "NO live writer; builder is in _ATTIC. Read by "
                 "FeatureTemplates/_tickermeta.py.",
    "News": "no fetcher; not consumed by production.",
    "ShortInterest": "directory is EMPTY and no fetcher writes it. FINRA short "
                     "VOLUME (Data/FINRA) is the live source.",
}


# ===========================================================================
# Execution
# ===========================================================================

# Fetchers echo request URLs on error, and some carry credentials in the query
# string -- fetch_fred.py prints the whole URL including api_key= on a non-200.
# That was merely on-screen before; now that every step gets a persisted
# transcript it would be written to disk, so redact on the way through.
_SECRET_RE = re.compile(
    r"((?:api_key|apikey|token|access_token|key|password|secret)=)[^&\s\"']+",
    re.IGNORECASE)

# The fetchers stamp their own "[YYYY-MM-DD HH:MM:SS] " prefix; stripping it off
# the one-line summary keeps the dispatcher's own report readable.
_TS_PREFIX_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]\s*")


def _redact(text: str) -> str:
    return _SECRET_RE.sub(r"\1<redacted>", text)


def run_step(step: Step, transcript: Path) -> tuple[bool, str, float]:
    """Run one step, streaming its output to stdout AND its own transcript.

    Never capture_output: a 26-minute step must show progress live and must
    leave a readable log even if it is killed. Returns (ok, detail, seconds).
    """
    argv = [sys.executable, str(step.argv[0]), *step.argv[1:]]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    t0 = time.perf_counter()
    tail: deque[str] = deque(maxlen=40)
    timed_out = threading.Event()

    proc = subprocess.Popen(
        argv, cwd=str(step.cwd), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1)

    # Watchdog rather than a deadline check inside the read loop: a process that
    # hangs silently never emits another line, so a read-loop check never fires.
    def _kill() -> None:
        timed_out.set()
        try:
            proc.kill()
        except Exception:
            pass

    timer = threading.Timer(step.timeout, _kill)
    timer.daemon = True
    timer.start()
    try:
        with open(transcript, "w", encoding="utf-8", newline="\n") as fh:
            for raw in proc.stdout:                     # type: ignore[union-attr]
                line = _redact(raw)
                sys.stdout.write(line)
                fh.write(line)
                fh.flush()                              # survive a kill
                s = _TS_PREFIX_RE.sub("", line.strip())
                if s:
                    tail.append(s)
        code = proc.wait()
    finally:
        timer.cancel()
        sys.stdout.flush()

    secs = time.perf_counter() - t0
    if timed_out.is_set():
        return False, f"TIMEOUT after {step.timeout}s", secs
    if code != 0:
        return False, f"exit {code}: " + (tail[-1] if tail else "no output"), secs
    return True, (tail[-1] if tail else "ok"), secs


def _record_timing(rows: list[dict]) -> None:
    """Append per-step timings. The only long-run history of how slow this is."""
    try:
        LOGS.mkdir(parents=True, exist_ok=True)
        new = not TIMING_CSV.exists()
        with open(TIMING_CSV, "a", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["ts", "step", "outcome", "secs",
                                               "newest_before", "newest_after"])
            if new:
                w.writeheader()
            w.writerows(rows)
    except Exception as exc:
        log(f"  (could not write {TIMING_CSV.name}: {exc})")


def _known_durations() -> dict[str, float]:
    """Median observed seconds per step, for --list."""
    out: dict[str, list[float]] = {}
    try:
        with open(TIMING_CSV, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("outcome") == "OK":
                    try:
                        out.setdefault(row["step"], []).append(float(row["secs"]))
                    except (ValueError, KeyError):
                        pass
    except Exception:
        return {}
    return {k: sorted(v)[len(v) // 2] for k, v in out.items() if v}


# ===========================================================================
# CLI
# ===========================================================================

def select(only: str | None, group: str | None) -> list[Step]:
    steps = list(MANIFEST)
    if group:
        want = {g.strip() for g in group.split(",")}
        unknown = want - {s.group for s in MANIFEST}
        if unknown:
            sys.exit(f"unknown group(s) {sorted(unknown)}; "
                     f"known: {sorted({s.group for s in MANIFEST})}")
        steps = [s for s in steps if s.group in want]
    if only:
        want = {n.strip() for n in only.split(",")}
        unknown = want - {s.name for s in MANIFEST}
        if unknown:
            sys.exit(f"unknown step(s) {sorted(unknown)}; "
                     f"known: {[s.name for s in MANIFEST]}")
        steps = [s for s in steps if s.name in want]
    return steps


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true",
                    help="run every step even if its lakes look current")
    ap.add_argument("--only", help="comma-separated step names (implies --force "
                                   "for those steps)")
    ap.add_argument("--group", help="comma-separated groups: market, sec, panel")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    ap.add_argument("--list", action="store_true", help="list steps and exit")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if ANY lake is stale afterwards, not just critical ones")
    ap.add_argument("--interstep-sleep", type=int, default=0,
                    help="seconds between steps (RAM release); the runner already "
                         "pauses 20s between stages")
    a = ap.parse_args()

    steps = select(a.only, a.group)
    if not steps:
        log("no steps selected")
        return 2

    if a.list:
        known = _known_durations()
        print(f"{'step':20s} {'group':7s} {'crit':5s} {'skip?':6s} "
              f"{'~secs':>7s}  writes")
        print("-" * 84)
        for s in steps:
            d = known.get(s.name)
            print(f"{s.name:20s} {s.group:7s} {'YES' if s.critical else '':5s} "
                  f"{'opt-in' if s.skip_when_fresh else '':6s} "
                  f"{('%.0f' % d) if d else '?':>7s}  {','.join(s.writes)}")
        return 0

    LOGS.mkdir(parents=True, exist_ok=True)
    lakes = sorted({lk for s in steps for lk in s.writes if lk in lf.LAKES})

    log("=== pre-refresh state ===")
    pre = {r["lake"]: r for r in lf.check_all(lakes)}
    lf._print_table(list(pre.values()))
    session = lf.last_completed_session()
    log(f"last completed NYSE session: {None if session is None else session.date()}")

    # --only means "I am fixing this one thing by hand". It must never silently
    # do nothing because the lake looked current.
    force = a.force or bool(a.only)

    log("")
    log("=== running ===")
    results: list[tuple[str, str, str]] = []
    timing: list[dict] = []
    succeeded: set[str] = set()
    stamp = time.strftime("%Y%m%d_%H%M%S")

    for i, step in enumerate(steps):
        blocked = [d for d in step.depends_on
                   if d in {s.name for s in steps} and d not in succeeded]
        if blocked:
            log(f"  {step.name:20s} BLOCKED (needs {', '.join(blocked)})")
            results.append((step.name, "BLOCKED", f"needs {', '.join(blocked)}"))
            continue

        if not force and step.skip_when_fresh:
            reps = [pre.get(lk) for lk in step.writes if lk in pre]
            verdicts = [lf.is_current_for_session(r) for r in reps if r]
            # Skip ONLY on a definite yes for every lake this step writes.
            if verdicts and all(v is True for v in verdicts):
                log(f"  {step.name:20s} SKIP (caught up to last session)")
                results.append((step.name, "SKIP", "caught up to last session"))
                succeeded.add(step.name)
                continue
            if verdicts and all(v is None for v in verdicts):
                # mtime-basis lake: the session question is unanswerable, so
                # fall back to the rotten check rather than re-pulling a
                # monthly series every night.
                if all((r or {}).get("status") == "OK" for r in reps):
                    log(f"  {step.name:20s} SKIP (not stale; mtime basis)")
                    results.append((step.name, "SKIP", "not stale (mtime basis)"))
                    succeeded.add(step.name)
                    continue

        if a.dry_run:
            log(f"  {step.name:20s} WOULD RUN  {' '.join(step.argv)}")
            results.append((step.name, "DRY", " ".join(step.argv)))
            succeeded.add(step.name)
            continue

        transcript = LOGS / f"stage_{step.name}_{stamp}.log"
        log(f"  {step.name:20s} RUN   {' '.join(step.argv)}  -> {transcript.name}")
        ok, detail, secs = run_step(step, transcript)
        log(f"  {step.name:20s} {'ok' if ok else 'FAIL'} ({secs:.1f}s)  {detail}")
        results.append((step.name, "OK" if ok else "FAIL", detail))
        timing.append({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "step": step.name,
                       "outcome": "OK" if ok else "FAIL", "secs": round(secs, 1),
                       "newest_before": (pre.get(step.writes[0], {}).get("newest")
                                         if step.writes else None),
                       "newest_after": None})
        if ok:
            succeeded.add(step.name)
        if a.interstep_sleep and i < len(steps) - 1:
            time.sleep(a.interstep_sleep)

    if a.dry_run:
        return 0

    log("")
    log("=== post-refresh state ===")
    post = lf.check_all(lakes)
    lf._print_table(post)
    post_by = {r["lake"]: r for r in post}
    for row in timing:
        step = next((s for s in steps if s.name == row["step"]), None)
        if step and step.writes:
            row["newest_after"] = post_by.get(step.writes[0], {}).get("newest")
    _record_timing(timing)

    # ---- summary + machine-parseable lines for the runner's alert fan-out ----
    log("")
    log("=== summary ===")
    for name, outcome, detail in results:
        log(f"  {name:20s} {outcome:8s} {detail}")

    failed = [n for n, o, _ in results if o in ("FAIL", "BLOCKED")]

    # "Ran but the data did not advance" is only evidence of a silent no-op when
    # the lake is ALSO still behind. A lake that is already caught up to the last
    # completed session has nothing to advance to, and reporting that as a
    # warning every healthy night is how a real alert gets tuned out -- the same
    # way the exit-0-on-404 in fetch_sec_insider.py stopped being noticed.
    not_moved = []
    for step in steps:
        if next((o for n, o, _ in results if n == step.name), None) != "OK":
            continue
        for lk in step.writes:
            b = pre.get(lk, {}).get("newest")
            post_rep = post_by.get(lk, {})
            af = post_rep.get("newest")
            if b is None or af is None or af > b:
                continue
            current = lf.is_current_for_session(post_rep)
            if current is True:
                continue                       # caught up: nothing to advance to
            if current is None and post_rep.get("status") == "OK":
                continue                       # mtime basis and not rotten
            not_moved.append(f"{lk} (still {af})")

    if not_moved:
        log("")
        log(f"  RAN BUT DATA DID NOT ADVANCE: {not_moved}")
        log("    Upstream may simply have no new observation yet. If it "
            "persists, the fetcher is silently no-opping -- exactly how the SEC "
            "insider lake went five months stale while exiting 0 every night.")

    for lk, why in UNMANAGED.items():
        log(f"  {lk:20s} UNMANAGED  {why}")

    # The runner greps these. Keep the step name inside the first 40 characters:
    # trading_system.ps1 dedups alert popups on message[:40].
    log("")
    for name, outcome, detail in results:
        if outcome in ("FAIL", "BLOCKED"):
            clean = re.sub(r"\s+", " ", detail)[:120]
            log(f'MACROFETCH FAIL {name} reason="{clean}"')
    for item in not_moved:
        log(f"MACROFETCH STALE-AFTER {item}")
    stale = [r["lake"] for r in post
             if r["status"] != "OK" and (a.strict or r["critical"])]
    for lk in stale:
        log(f"MACROFETCH CRITICAL-STALE {lk}")
    ok_n = sum(1 for _, o, _ in results if o in ("OK", "SKIP"))
    log(f"MACROFETCH SUMMARY {ok_n}/{len(results)} ok")

    if stale:
        log("")
        log(f"  STALE AFTER REFRESH: {stale} -> exiting non-zero so the nightly "
            f"reports this stage failed.")
        return 1
    if failed:
        log("")
        log(f"  FAILED: {failed} (no critical lake left stale, so exit 0)")
    log("  all checked lakes current.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
