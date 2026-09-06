"""Fail-closed staleness gate for the raw data lakes.

WHY THIS FILE EXISTS
--------------------
On 2026-07-28 a census found that EVERY lake owned by a `fetchers/` script had
frozen on 2026-05-28 -- the day the fetchers were last run by hand:

    Indexes       2026-07-02   (~17 trading sessions stale)
    IndexesFull   2026-05-21   (67 days)
    FRED          2026-05-27   (60 days, and 30 of 52 series were MISSING outright)
    Treasury      2026-05-28   (60 days)
    FINRA         2026-05-28   (60 days)
    KenFrench     2026-05-28   (60 days)
    CFTC_COT      2026-05-28   (60 days)
    Shiller       2026-05-28   (60 days)
    News          2025-11-30   (239 days)

Meanwhile PriceData / SEC / Fundamentals / Insider / ProcessedData_v2 were all
current, because the nightly (`trading_system.ps1`) refreshes those and only
those: its "Data Panels" stage runs `build_data_panels.py all --refresh-all`,
which delegates to exactly THREE SEC fetchers (companyfacts, insider,
submissions). Nothing in the pipeline ever ran the market/macro fetchers.

The cost was silent. A frozen `Data/Indexes` alone took ~88 index-relative panel
columns (alpha_*/beta_*/corr_* and the whole cross-asset / beta_risk vein) to
all-NaN, and left the predictor's neutralization beta leg inert, with no error
anywhere. This module exists so that class of failure is LOUD instead.

DESIGN
------
1. DATA DATE, NOT FILE MTIME. An mtime says when we last *wrote*, not how fresh
   the contents are. A fetcher that runs nightly against a dead endpoint keeps
   mtime current forever while the newest observation rots. So every lake is
   judged on the newest DATE INSIDE its representative file; mtime is only a
   fallback for lakes too large to open (FINRA is 17k+ files).

2. SELF-CALIBRATING CADENCE. The lakes mix daily market data with weekly COT,
   monthly Shiller and quarterly GDP. A single hardcoded threshold would either
   miss real staleness or cry wolf every quarter. Instead the allowed lag is
   derived from the series' OWN median inter-observation gap:

       allowed_lag = max(floor_days, slack * median_observation_gap)

   A daily series tolerates a long weekend; a quarterly series tolerates a
   quarter. Nothing to maintain when a cadence changes.

3. FAIL CLOSED. `assert_fresh()` RAISES. The whole point is that a stale lake
   must stop the pipeline rather than quietly emit NaN columns. `--strict`
   exits non-zero for shell callers.

4. DELIBERATE BYPASS. Historical rebuilds legitimately run against pinned old
   data, so `LAKE_FRESHNESS_BYPASS=1` downgrades every raise to a warning. It
   is an env var and not a default argument so that bypassing is a visible,
   auditable act rather than something a caller can forget it did.

USAGE
-----
    python auxiliary/lake_freshness.py                      # census table, always exit 0
    python auxiliary/lake_freshness.py --strict             # exit 1 if anything is stale
    python auxiliary/lake_freshness.py --lakes Indexes,FRED # only these
    python auxiliary/lake_freshness.py --verify-bt-columns  # backtester line-set regression check

    from auxiliary.lake_freshness import assert_fresh, check_all
    assert_fresh(["Indexes", "FRED"])             # raises LakeStaleError
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import re
import sys
from dataclasses import dataclass

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # lives in auxiliary/
DATA = os.path.join(ROOT, "Data")

BYPASS_ENV = "LAKE_FRESHNESS_BYPASS"

# Slack multiplier applied to a series' own median observation gap. 3.0 means a
# daily series may lag 3 days before we complain -- enough for Fri->Mon plus a
# holiday, tight enough that a fortnight of silence is caught.
DEFAULT_SLACK = 3.0


_MARKET_TZ = "America/New_York"
# Grace after the close before we expect a fetcher to have landed the session.
_SETTLE_MINUTES = 30
_CAL = None


class LakeStaleError(RuntimeError):
    """Raised when a required lake is stale or missing. Fail-closed by design."""


def _nyse():
    global _CAL
    if _CAL is None:
        import exchange_calendars as ec
        _CAL = ec.get_calendar("XNYS")
    return _CAL


def last_completed_session(now: pd.Timestamp | None = None) -> pd.Timestamp | None:
    """The newest NYSE session whose data should already be on disk.

    Today counts only once it is `_SETTLE_MINUTES` past its own close, so a
    17:00 evening run expects today's bar while an 09:00 morning run does not.
    Returns None when the calendar is unavailable -- callers must treat that as
    "cannot tell", never as "fresh".
    """
    try:
        cal = _nyse()
    except Exception:
        return None
    now_et = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz=_MARKET_TZ)
    now_et = (now_et.tz_localize(_MARKET_TZ) if now_et.tzinfo is None
              else now_et.tz_convert(_MARKET_TZ))

    day = now_et.normalize().tz_localize(None)
    for _ in range(15):          # 15 days covers any holiday run
        try:
            if cal.is_session(day):
                close = pd.Timestamp(cal.session_close(day)).tz_convert(_MARKET_TZ)
                if now_et >= close + pd.Timedelta(minutes=_SETTLE_MINUTES):
                    return day
        except Exception:
            return None
        day -= pd.Timedelta(days=1)
    return None


def is_current_for_session(rep: dict, now: pd.Timestamp | None = None) -> bool | None:
    """Has this lake caught up to the last completed session?

    THIS IS NOT `check_lake`'s question. `check_lake` asks "is this ROTTEN?" --
    age <= max(floor_days, slack*gap), which for Indexes is FIVE DAYS. Treating
    that OK as "already current" is what let refresh_market_data skip the one
    critical lake while it sat three sessions behind (proved on 2026-08-31:
    `Indexes YES OK 2026-08-28 3 5.0` immediately followed by
    `Indexes SKIP (already current)`), silently taking ~88 alpha_*/beta_*/corr_*
    panel columns to a stale basis.

    Only THIS predicate may ever cause a fetcher to be skipped. `check_lake`
    stays what it is: the gate and the alert.

    Returns True/False, or None when it cannot be judged (an mtime-basis lake, a
    missing probe, or no calendar). None means "cannot tell" -- run the fetcher.
    """
    if rep.get("newest") is None:
        return False
    if str(rep.get("basis") or "").startswith("mtime"):
        return None          # mtime says when we wrote, not what we hold
    session = last_completed_session(now)
    if session is None:
        return None
    return pd.Timestamp(rep["newest"]).normalize() >= session


@dataclass
class LakeSpec:
    """How to judge one lake's freshness.

    probe        glob (relative to Data/) of the file whose newest date is
                 authoritative for the lake. See `pick` for which match wins.
                 A probe ending in .json is read as a _BUILD_STAMP (below).
    pick         which glob match is authoritative: "first" (lexicographically
                 lowest, the historical behaviour) or "last". Per-year and
                 per-quarter lakes MUST use "last" -- a *_form345.zip probe
                 under "first" resolves to 2006q1 and reports the lake
                 permanently 20 years stale.
    name_date_re regex over the resolved file's BASENAME, for archives whose
                 contents cannot be dated cheaply but whose filename states the
                 period they cover. Two capture groups are read as
                 (year, quarter) and resolve to that quarter's end; one group is
                 parsed as a date. This is what separates "when did we last
                 download" (mtime, which stays green against a dead endpoint)
                 from "what period does the newest archive actually cover".
    date_col     name of the column carrying the data date, when it is not one
                 of the Date/date/DATE/Datetime/datetime candidates. Every
                 SEC-derived panel keys on `filed_date`; without this they fall
                 through to the mtime branch and report the BUILD date, which
                 looks fresh no matter how stale the data underneath is.
    floor_days   minimum tolerated lag regardless of cadence. Guards against a
                 degenerate median gap (e.g. a series with 2 observations).
    slack        multiplier on the median observation gap.
    mtime_only   True for lakes too large/heterogeneous to open; judged on the
                 newest file mtime in the tree instead of a data date.
                 !! Never point an mtime_only probe at a large tree: the walk is
                 recursive and Data/SEC/submissions is ~988k files (~65 s), paid
                 twice per refresh because the pre- and post-state are both
                 checked. Probe the single zip artifact instead.
    refresher    the fetcher command that fixes this lake, quoted in the error
                 so the message is actionable rather than merely alarming.
    critical     True = a production consumer silently degrades when this is
                 stale, so it belongs in the nightly's fail-closed set.
    """
    probe: str
    refresher: str
    pick: str = "first"
    date_col: str | None = None
    name_date_re: str | None = None
    floor_days: float = 5.0
    slack: float = DEFAULT_SLACK
    mtime_only: bool = False
    critical: bool = False
    notes: str = ""


# The `critical` set is deliberately narrow: a lake earns it by having a
# PRODUCTION consumer that degrades SILENTLY when the lake is stale. Everything
# else is reported but does not stop the pipeline.
LAKES: dict[str, LakeSpec] = {
    "Indexes": LakeSpec(
        probe="Indexes/SPY.parquet",
        refresher="python fetchers/fetch_indexes.py --lake recent",
        floor_days=5.0, critical=True,
        notes="~88 alpha_*/beta_*/corr_* panel cols go all-NaN when stale; "
              "also the predictor's spy200v2 neutralization beta leg.",
    ),
    "IndexesFull": LakeSpec(
        probe="IndexesFull/SPY.parquet",
        refresher="python fetchers/fetch_indexes.py --lake full",
        floor_days=5.0, critical=False,
        notes="Deep-history lake; used by long-window/asof forks, not the nightly.",
    ),
    "FRED": LakeSpec(
        # DGS10 is daily and effectively never revised, which makes it the
        # honest freshness probe for this lake. Do NOT probe a revised
        # aggregate (GDPC1, UMCSENT) -- their publication lag is structural.
        probe="FRED/DGS10.parquet",
        refresher="python fetchers/fetch_fred.py --force",
        floor_days=6.0, critical=False,
        notes="!! fetch_fred.py SKIPS every series already on disk unless "
              "--force is passed, despite its docstring claiming re-running "
              "refreshes. A plain re-run is a no-op. Always pass --force.",
    ),
    "Treasury": LakeSpec(
        probe="Treasury/daily_par_yield_2026.parquet",
        refresher="python fetchers/fetch_treasury.py",
        floor_days=6.0, critical=False,
        notes="Per-year files; the current-year file is the probe. Prior years "
              "are immutable and cached.",
    ),
    "KenFrench": LakeSpec(
        probe="KenFrench/F-F_Research_Data_5_Factors_2x3_daily.csv",
        refresher="python fetchers/fetch_kenfrench.py",
        floor_days=45.0, mtime_only=True, critical=False,
        notes="Academic factor files land in monthly-ish batches with a "
              "multi-week publication lag; mtime is the honest signal here.",
    ),
    "CFTC_COT": LakeSpec(
        probe="CFTC_COT",
        refresher="python fetchers/fetch_cftc_cot.py",
        floor_days=14.0, mtime_only=True, critical=False,
        notes="Weekly release (Friday), zipped per year.",
    ),
    "Shiller": LakeSpec(
        probe="Shiller/shiller_data.parquet",
        refresher="python fetchers/fetch_shiller.py",
        floor_days=45.0, mtime_only=True, critical=False,
        notes="Monthly series. mtime_only because Shiller's 'Date' column is a "
              "FRACTIONAL YEAR float (2023.09 = Sep 2023), not a parseable date "
              "-- reading it as a date yields 1970-01-01 and a false STALE. "
              "!! SEPARATE UNFIXED ISSUE: the parsed series ends at 2023.09 even "
              "on a fresh download, so fetch_shiller.py's xls sheet parse looks "
              "truncated. Not consumed by production, so reported not gated.",
    ),
    "FINRA": LakeSpec(
        probe="FINRA",
        refresher="python fetchers/fetch_finra_shorts.py",
        floor_days=6.0, mtime_only=True, critical=False,
        notes="17k+ daily short-volume files; too many to open, judged on mtime.",
    ),
    "MarketCaps": LakeSpec(
        probe="MarketCaps/historical_market_caps.parquet",
        refresher="python auxiliary/0__ApproximateMarketCaps.py",
        floor_days=8.0, critical=False,
        notes="Backs the micro-cap gate's PIT fallback when FinViz is down. "
              "Known to carry large per-name error (static share-count anchor). "
              "!! NO AUTOMATED WRITER: no nightly stage rebuilds this. The "
              "refresher above is the only writer and must be run by hand. "
              "(Corrected 2026-09-03: this field used to read 'python "
              "build_data_panels.py marketcaps', a subcommand that does not "
              "exist -- the fix-it string itself failed with an argparse "
              "invalid-choice error.)",
    ),

    # -----------------------------------------------------------------------
    # SEC raw archives + the panels derived from them. Added 2026-09-03 with
    # 2__MacroFetcher.py, which owns these sources. Before that they were
    # invisible here: build_data_panels.py --refresh-all fetched them, caught
    # every exception into a print(), and exited 0 regardless.
    #
    # PROBE THE ZIPS, NOT THE EXTRACTED TREES. Data/SEC/submissions is ~988k
    # files and Data/SEC/companyfacts ~20k; an mtime_only probe walks the tree
    # recursively and the census reads every probe twice per refresh.
    # -----------------------------------------------------------------------
    "SEC_companyfacts": LakeSpec(
        probe="SEC/raw/companyfacts.zip",
        refresher="python fetchers/fetch_sec_companyfacts.py --extract",
        floor_days=8.0, mtime_only=True, critical=False,
        notes="1.3 GB bulk XBRL archive; the fetcher self-skips files <20h old. "
              "mtime is the only cheap signal for a zip, so this answers 'did we "
              "download recently', NOT 'is the data current'. Data/Fundamentals "
              "below is the honest data-date check.",
    ),
    "SEC_submissions": LakeSpec(
        probe="SEC/raw/submissions.zip",
        refresher="python fetchers/fetch_sec_submissions.py --extract",
        floor_days=8.0, mtime_only=True, critical=False,
        notes="1.5 GB filing-metadata archive. Feeds Data/SectorMap.parquet "
              "(SIC codes) and Data/SEC/filing_calendar/.",
    ),
    "SEC_insider": LakeSpec(
        probe="SEC/insider/raw/*_form345.zip",
        refresher="python fetchers/fetch_sec_insider.py",
        pick="last", name_date_re=r"(\d{4})q(\d)",
        # One quarter of cadence plus the SEC's own publication lag.
        floor_days=135.0, critical=False,
        notes="!! KNOWN BROKEN 2026-09-03: the newest archive on disk is "
              "2026q1 (data through 2026-03-31). 2026q2 and 2026q3 return 404 "
              "and fetch_sec_insider.py:85-88 catches that, logs 'skip' on the "
              "assumption it is a not-yet-published quarter, and exits 0. The "
              "SEC appears to have moved the quarterly URL. Every "
              "FeatureTemplates/_insider*.py column has been decaying since "
              "March. Probed on the FILENAME quarter, not mtime, so this stays "
              "STALE until real new data lands.",
    ),
    "Fundamentals": LakeSpec(
        probe="Fundamentals/by_ticker/AAPL.parquet",
        refresher="python build_data_panels.py fundamentals",
        date_col="filed_date",
        # AAPL is a quarterly filer; the median-gap slack lifts this anyway.
        floor_days=120.0, critical=True,
        notes="CRITICAL: the sfn_*/sfd_*/sfb_* feature columns decay silently "
              "toward zero when this is stale rather than erroring. Probed on a "
              "single 48 KB by_ticker file, NOT the 225 MB panel.",
    ),
    "Insider": LakeSpec(
        probe="Insider/by_ticker/AAPL.parquet",
        refresher="python build_data_panels.py insider",
        date_col="filed_date",
        floor_days=135.0, critical=False,
        notes="Derived from SEC_insider, so it inherits that lake's staleness. "
              "Probed on a 99 KB by_ticker file, not the 76 MB panel.",
    ),
    "SectorMap": LakeSpec(
        probe="SectorMap.parquet",
        refresher="python build_data_panels.py sector",
        floor_days=30.0, mtime_only=True, critical=False,
        notes="Columns are Ticker/CIK/SIC/SICDescription/Sector/IndustryGroup/"
              "Source -- there is NO date column of any kind, so mtime is the "
              "only honest basis. Feeds 7__MacroFilter.py's concentration cap.",
    ),

    # -----------------------------------------------------------------------
    # Consumed by production, NO LIVE WRITER. Declared so the gap is loud
    # rather than absent from the census entirely. Nothing refreshes these;
    # their builders were moved to _ATTIC.
    # -----------------------------------------------------------------------
    "PriceDataFull": LakeSpec(
        probe="PriceDataFull/AAPL.parquet",
        refresher="(NO LIVE WRITER: _ATTIC/research-loose/experimental/"
                  "2.1_DeepPriceDownloader.py)",
        floor_days=5.0, critical=False,
        notes="Deep 60Y daily history. Read by 7__MacroFilter.py, "
              "10__FinalSolution.py, auxiliary/signal_autopsy.py and several "
              "FeatureTemplates, but nothing in the live tree can refresh it.",
    ),
    "PhxPanels": LakeSpec(
        probe="PhxPanels/neighbor_panel.parquet",
        refresher="(NO LIVE WRITER: _ATTIC .../phonetics_lab/build_phx_panels.py)",
        floor_days=30.0, mtime_only=True, critical=False,
        notes="Read by FeatureTemplates/_tickermeta.py. Builder is atticked.",
    ),
}


_DATE_CANDIDATES = ("Date", "date", "DATE", "Datetime", "datetime")


def _stamp_date(path: str) -> tuple[pd.Timestamp | None, float | None]:
    """Read a builder's _BUILD_STAMP.json: {"built_utc", "n_tickers", "max_filed_date"}.

    The derived panels already write this, and it is the cheapest honest
    freshness signal available -- it states the newest DATA date directly, so
    there is no file to open and no column to guess.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            stamp = json.load(fh)
    except Exception:
        return None, None
    for key in ("max_filed_date", "max_date", "newest_date"):
        if stamp.get(key):
            d = pd.to_datetime(stamp[key], errors="coerce")
            if pd.notna(d):
                return pd.Timestamp(d).normalize(), None
    return None, None


def _date_from_name(path: str, pattern: str) -> pd.Timestamp | None:
    """Derive a data date from a filename, e.g. 2026q1_form345.zip -> 2026-03-31."""
    m = re.search(pattern, os.path.basename(path))
    if not m:
        return None
    try:
        if len(m.groups()) >= 2:
            year, quarter = int(m.group(1)), int(m.group(2))
            if not 1 <= quarter <= 4:
                return None
            # Quarter END: the archive covers through the last day of it.
            return (pd.Timestamp(year=year, month=quarter * 3, day=1)
                    + pd.offsets.MonthEnd(0)).normalize()
        d = pd.to_datetime(m.group(1), errors="coerce")
        return None if pd.isna(d) else pd.Timestamp(d).normalize()
    except Exception:
        return None


def _parquet_date_column(path: str, date_col: str | None) -> str | None:
    """Pick the date column from the parquet SCHEMA, without reading the data.

    Column projection matters: fundamentals_panel.parquet is 225 MB / 27.9M
    rows, and the census reads every probe twice per refresh.
    """
    try:
        import pyarrow.parquet as pq
        names = set(pq.read_schema(path).names)
    except Exception:
        return date_col
    if date_col and date_col in names:
        return date_col
    for cand in _DATE_CANDIDATES:
        if cand in names:
            return cand
    return None


def _newest_data_date(path: str,
                      date_col: str | None = None) -> tuple[pd.Timestamp | None, float | None]:
    """Return (newest date, median observation gap in days) for one file."""
    if path.lower().endswith(".json"):
        return _stamp_date(path)
    try:
        if path.lower().endswith(".csv"):
            df = pd.read_csv(path, nrows=200_000)
        else:
            col = _parquet_date_column(path, date_col)
            # Project to the single date column when we can identify one. Falls
            # back to a full read so an unreadable schema still degrades to the
            # old behaviour rather than to a false MISSING.
            df = (pd.read_parquet(path, columns=[col]) if col
                  else pd.read_parquet(path))
    except Exception:
        try:
            df = pd.read_parquet(path)
        except Exception:
            return None, None

    idx = None
    if date_col and date_col in df.columns:
        idx = pd.to_datetime(df[date_col], errors="coerce")
    elif isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
    else:
        for cand in _DATE_CANDIDATES:
            if cand in df.columns:
                idx = pd.to_datetime(df[cand], errors="coerce")
                break
    if idx is None:
        return None, None

    s = pd.Series(pd.to_datetime(idx, errors="coerce")).dropna().sort_values()
    if s.empty:
        return None, None
    # Median gap over the recent tail: cadence can change over a long history
    # (Shiller is monthly throughout, but several FRED series switched from
    # weekly to daily), and it is the CURRENT cadence we must tolerate.
    tail = s.tail(60)
    gaps = tail.diff().dropna().dt.total_seconds() / 86400.0
    gap = float(gaps.median()) if len(gaps) else None
    return pd.Timestamp(s.iloc[-1]).normalize(), gap


def _newest_mtime(path: str) -> pd.Timestamp | None:
    """Newest file mtime in a file or directory tree (local time)."""
    if os.path.isfile(path):
        return pd.Timestamp(datetime.datetime.fromtimestamp(os.path.getmtime(path)))
    if not os.path.isdir(path):
        return None
    newest = 0.0
    for dirpath, _dirs, files in os.walk(path):
        for f in files:
            try:
                m = os.path.getmtime(os.path.join(dirpath, f))
            except OSError:
                continue
            if m > newest:
                newest = m
    if newest == 0.0:
        return None
    return pd.Timestamp(datetime.datetime.fromtimestamp(newest))


def check_lake(name: str, spec: LakeSpec | None = None,
               today: pd.Timestamp | None = None) -> dict:
    """Judge one lake. Never raises -- returns a status dict."""
    spec = spec or LAKES[name]
    today = (today or pd.Timestamp.now()).normalize()
    target = os.path.join(DATA, spec.probe)

    # `pick` decides which glob match is authoritative. Defaults to "first" for
    # backwards compatibility, but any per-year/per-quarter lake needs "last":
    # sorted()[0] on Data/SEC/insider/raw/*_form345.zip is 2006q1.
    matches = sorted(glob.glob(target))
    if matches:
        resolved = matches[-1] if spec.pick == "last" else matches[0]
    else:
        resolved = target if os.path.exists(target) else None

    rep: dict = {
        "lake": name, "critical": spec.critical, "probe": spec.probe,
        "refresher": spec.refresher, "notes": spec.notes,
        "newest": None, "age_days": None, "allowed_days": None,
        "basis": "mtime" if spec.mtime_only else "data-date",
        "status": "MISSING",
    }
    if resolved is None:
        return rep

    if spec.name_date_re:
        # The filename states the period covered. Preferred over mtime for
        # archive lakes: mtime reports when we last downloaded, which stays
        # green forever against an endpoint that has started 404ing.
        newest = _date_from_name(resolved, spec.name_date_re)
        gap = None
        rep["basis"] = "data-date (from filename)"
        if newest is None:
            newest = _newest_mtime(resolved)
            rep["basis"] = "mtime (fallback: filename did not match)"
    elif spec.mtime_only:
        newest = _newest_mtime(resolved)
        gap = None
    else:
        newest, gap = _newest_data_date(resolved, spec.date_col)
        if newest is None:
            # Unreadable or no date column: degrade to mtime rather than
            # reporting a false OK. Degrading silently to "fine" is the exact
            # failure mode this module exists to prevent.
            newest = _newest_mtime(resolved)
            rep["basis"] = "mtime (fallback: no date column found)"

    if newest is None:
        return rep

    allowed = spec.floor_days
    if gap and gap > 0:
        allowed = max(allowed, spec.slack * gap)

    age = (today - newest.normalize()).days
    rep.update(newest=newest.normalize(), age_days=age, allowed_days=round(allowed, 1),
               median_gap_days=None if gap is None else round(gap, 2),
               status="OK" if age <= allowed else "STALE")
    return rep


def check_all(names: list[str] | None = None,
              today: pd.Timestamp | None = None) -> list[dict]:
    names = names or list(LAKES)
    return [check_lake(n, today=today) for n in names]


def assert_fresh(names: list[str] | None = None, *, critical_only: bool = False,
                 today: pd.Timestamp | None = None) -> list[dict]:
    """Fail-closed gate. Raises LakeStaleError on any STALE/MISSING lake.

    `critical_only` restricts the raise to lakes whose production consumers
    degrade silently; non-critical staleness is still reported to stderr so it
    cannot rot unnoticed either.
    """
    reps = check_all(names, today=today)
    considered = [r for r in reps if r["critical"]] if critical_only else reps
    bad = [r for r in considered if r["status"] != "OK"]

    for r in reps:
        if r["status"] != "OK" and r not in bad:
            print(f"[lake_freshness] WARNING {r['lake']} is {r['status']} "
                  f"(newest {r['newest']}, age {r['age_days']}d > "
                  f"{r['allowed_days']}d). Fix: {r['refresher']}", file=sys.stderr)

    if not bad:
        return reps

    lines = [f"{len(bad)} data lake(s) STALE or MISSING -- refusing to run on rotten data:"]
    for r in bad:
        lines.append(
            f"  {r['lake']:16s} {r['status']:7s} newest={r['newest']} "
            f"age={r['age_days']}d allowed={r['allowed_days']}d ({r['basis']})")
        if r["notes"]:
            lines.append(f"               why it matters: {r['notes']}")
        lines.append(f"               fix: {r['refresher']}")
    lines.append("  Refresh everything: python fetchers/refresh_market_data.py")
    lines.append(f"  Deliberate historical run? set {BYPASS_ENV}=1 to downgrade to a warning.")
    msg = "\n".join(lines)

    if os.environ.get(BYPASS_ENV) == "1":
        print(f"[lake_freshness] {BYPASS_ENV}=1 -- downgrading to WARNING:\n{msg}",
              file=sys.stderr)
        return reps
    raise LakeStaleError(msg)


def verify_bt_columns() -> bool:
    """Regression check: adding files to Data/Indexes must not change the
    backtester's dynamic line set.

    `get_dynamic_signal_columns()` in 5__NightlyBackTester.py enumerates the
    lake DIRECTORY and greps each parquet's COLUMNS for corr/alpha/beta. Since
    the lake files carry only OHLCV + Adj Close, it returns ([],[],[]) -- which
    is why the 5 rate/credit ETFs added on 2026-07-28 are safe. If a future
    change ever puts a corr/alpha/beta COLUMN into a lake parquet, the
    backtester would silently gain feed lines and results would move. This
    asserts that has not happened.
    """
    corr, alpha, beta = set(), set(), set()
    for f in sorted(glob.glob(os.path.join(DATA, "Indexes", "*.parquet"))):
        try:
            cols = list(pd.read_parquet(f).columns)
        except Exception as e:
            print(f"  ! unreadable {os.path.basename(f)}: {e}")
            continue
        corr |= {c for c in cols if "corr" in str(c).lower()}
        alpha |= {c for c in cols if "alpha" in str(c).lower()}
        beta |= {c for c in cols if "beta" in str(c).lower()}
    n = len(glob.glob(os.path.join(DATA, "Indexes", "*.parquet")))
    ok = not (corr or alpha or beta)
    print(f"Data/Indexes: {n} parquet file(s)")
    print(f"  corr cols : {sorted(corr) or '[]'}")
    print(f"  alpha cols: {sorted(alpha) or '[]'}")
    print(f"  beta cols : {sorted(beta) or '[]'}")
    print("  => backtester dynamic line set is EMPTY; lake file count cannot "
          "change backtest results." if ok else
          "  => !! backtester WOULD gain feed lines from the lake. Adding or "
          "removing lake files now CHANGES backtest results. Investigate before running.")
    return ok


def _print_table(reps: list[dict]) -> None:
    w = max(13, max((len(r["lake"]) for r in reps), default=13) + 1)
    print(f"{'lake':{w}s} {'crit':5s} {'status':7s} {'newest':12s} "
          f"{'age':>5s} {'allow':>6s}  basis")
    print("-" * (w + 65))
    for r in reps:
        newest = "-" if r["newest"] is None else str(r["newest"])[:10]
        age = "-" if r["age_days"] is None else str(r["age_days"])
        allow = "-" if r["allowed_days"] is None else str(r["allowed_days"])
        print(f"{r['lake']:{w}s} {'YES' if r['critical'] else '':5s} "
              f"{r['status']:7s} {newest:12s} {age:>5s} {allow:>6s}  {r['basis']}")
    bad = [r for r in reps if r["status"] != "OK"]
    if bad:
        print()
        for r in bad:
            print(f"  {r['lake']}: {r['status']}  fix: {r['refresher']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lakes", help="comma-separated subset (default: all)")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any checked lake is STALE/MISSING")
    ap.add_argument("--critical-only", action="store_true",
                    help="with --strict, only fail on lakes whose consumers "
                         "degrade silently")
    ap.add_argument("--verify-bt-columns", action="store_true",
                    help="assert the lake cannot change the backtester line set")
    a = ap.parse_args()

    if a.verify_bt_columns:
        return 0 if verify_bt_columns() else 1

    names = [s.strip() for s in a.lakes.split(",")] if a.lakes else None
    if names:
        unknown = [n for n in names if n not in LAKES]
        if unknown:
            print(f"unknown lake(s): {unknown}. known: {sorted(LAKES)}", file=sys.stderr)
            return 2

    reps = check_all(names)
    _print_table(reps)

    if not a.strict:
        return 0
    try:
        assert_fresh(names, critical_only=a.critical_only)
    except LakeStaleError as e:
        print(f"\n{e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
