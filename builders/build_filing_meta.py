"""
build_filing_meta.py  --  DISCLOSURE-BEHAVIOUR panel from the SEC companyfacts dump.

WHAT THIS IS
------------
`build_fundamentals_panel.py` extracts the NUMBERS a filing reports (revenue, net income, ...).
This builds the orthogonal half: the META of the filing act itself -- WHEN the firm reported,
HOW MUCH it tagged, WHICH concepts it used, and WHETHER it quietly rewrote numbers it had
already published. None of that is in `Data/Fundamentals/by_ticker/`, and none of it is a
transform of price.

WHY IT READS THE RAW JSONs AND NOT fundamentals_panel.parquet
--------------------------------------------------------------
`fundamentals_panel.parquet` is DEDUPED to one row per (concept, unit, period_end), keeping the
LATEST filing that reported it. That is fine for "what is revenue" but it destroys exactly what
this file needs: the per-accession concept set, and the earlier-vs-later value of the same fact
(which IS the revision signal). So we go back to Data/SEC/companyfacts/CIK*.json, where every
accession is preserved.

RESEARCH VEINS OPERATIONALISED  (see analysis_output/SEC_DISCLOSURE_META_RESEARCH_2026_07_27.md)
  1. REPORTING LAG            filed - period_end, and its abnormality vs the firm's own history.
                              Long/abnormal lag => withheld bad news, lower accrual quality,
                              higher restatement probability.
  2. OVERDUE (live)           how far past the firm's OWN cadence we are with no filing yet.
                              Fires BEFORE the late filing exists (computed in the feature block
                              from sfl_expected_gap, not here).
  3. TAG CHURN / COMPLEXITY   how many distinct XBRL concepts a filing tags and how far the
                              concept SET moved vs the last comparable filing. Poorly-performing
                              firms file more complex, less-comparable XBRL.
  4. CONCEPT RARITY DRIFT     migration from concepts everyone reports toward concepts almost
                              nobody reports -- the machine-readable trace of "our income line is
                              now called something weird". Cousin of McVay (2006) classification
                              shifting, measured on the TAG instead of the value.
  5. SILENT REVISION          the same (concept, unit, start, end) coming back with a DIFFERENT
                              value under a later accession = a "little r" revision. Disclosed
                              without an 8-K; the market prices them WITH A DELAY.

FORM-AWARE CHURN
----------------
A 10-K tags roughly twice as many concepts as a 10-Q, so comparing a 10-K's concept set to the
preceding 10-Q's manufactures enormous fake churn every fourth quarter. Churn is therefore always
measured against the previous filing of the SAME class (annual-vs-annual, interim-vs-interim).

POINT-IN-TIME DISCIPLINE
------------------------
Every row is stamped with `filed_date` -- the day the fact became public. All firm-relative
statistics (z-scores, expected cadence) are EXPANDING over STRICTLY PRIOR filings only, so a row
never embeds anything unknowable on its own filed_date. Feature blocks consume this through
FeatureTemplates/_filingmeta.py, a backward merge_asof on filed_date exactly like
_fundamentals.py. Never merge on period_end -- that is future leakage.

INPUT   Data/SEC/companyfacts/CIK##########.json   (raw, per-accession)
        Data/Fundamentals/fundamentals_panel.parquet  (ticker <-> cik map only)
        Data/Fundamentals/concept_dictionary.parquet  (concept -> n_tickers, for rarity)
OUTPUT  Data/Fundamentals/filing_meta/{TICKER}.parquet (one row per periodic filing event)

USAGE   stock_env\\Scripts\\python.exe builders/build_filing_meta.py
        ...\\python.exe builders/build_filing_meta.py --limit 40 --jobs 1     # smoke run
        ...\\python.exe builders/build_filing_meta.py --tickers AAPL,KO
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent   # this script lives in builders/
CF_DIR = ROOT / "Data" / "SEC" / "companyfacts"
PANEL = ROOT / "Data" / "Fundamentals" / "fundamentals_panel.parquet"
CONCEPT_DICT = ROOT / "Data" / "Fundamentals" / "concept_dictionary.parquet"
PRICE_DIR = ROOT / "Data" / "PriceData"
OUT_DIR = ROOT / "Data" / "Fundamentals" / "filing_meta"

# Periodic reports only. 8-K / 6-K / DEF 14A / S-1 tag a handful of concepts and would swamp the
# churn measures with meaningless set-distance; they are not "the quarterly report".
PERIODIC_PREFIXES = ("10-K", "10-Q", "20-F", "40-F")

# A value must move by more than this (relative) to count as a revision rather than a
# rounding/units artefact.
REVISION_TOL = 1e-3

# Filings tagging fewer concepts than this are cover-page-only stubs (an amendment that just
# refiles an exhibit). They are not comparable disclosures -- excluding them keeps the churn
# measures from being dominated by a set-distance against a 2-element set.
MIN_CONCEPTS = 10

_RARITY: dict | None = None
_RARITY_DEFAULT: float = 8.0


# --------------------------------------------------------------------------------------
# concept rarity table
# --------------------------------------------------------------------------------------
def load_rarity() -> tuple[dict, float]:
    """
    concept -> rarity score, where rarity = -log(share of filers that ever report it).

    "Assets" (every firm) ~ 0.0; a concept used by 3 firms out of 3461 ~ 7.0. Averaging this over
    a filing's concept set gives one "how exotic is this firm's chart of accounts" number that is
    comparable across firms and over time.
    """
    d = pd.read_parquet(CONCEPT_DICT, columns=["concept", "n_tickers"])
    d = d.groupby("concept", as_index=False)["n_tickers"].max()
    total = float(max(d["n_tickers"].max(), 1))
    share = (d["n_tickers"].astype(float) / total).clip(lower=1.0 / total, upper=1.0)
    rarity = -np.log(share)
    return dict(zip(d["concept"], rarity)), float(rarity.max())


def _init_worker(rarity: dict, default: float) -> None:
    global _RARITY, _RARITY_DEFAULT
    _RARITY, _RARITY_DEFAULT = rarity, default


# --------------------------------------------------------------------------------------
# one company
# --------------------------------------------------------------------------------------
def _form_class(form: str) -> str:
    """Annual vs interim. 10-K/20-F/40-F are annual; 10-Q is interim. Amendments keep the class."""
    f = form.upper()
    return "Q" if f.startswith("10-Q") else "K"


def build_one(path: Path, ticker: str) -> pd.DataFrame:
    """Parse one CIK's companyfacts JSON -> one row per periodic filing event."""
    rarity, rar_default = _RARITY, _RARITY_DEFAULT
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)

    # accn -> filing header;  accn -> {concept};  accn -> [(key, value)]
    hdr: dict[str, dict] = {}
    cset: dict[str, set] = defaultdict(set)
    vals: dict[str, dict] = defaultdict(dict)

    for taxonomy, concepts in (doc.get("facts") or {}).items():
        # `dei` cover-page facts (EntityCommonStockSharesOutstanding) are stamped "as of" the
        # COVER date, days before filing and AFTER the fiscal period end. Letting them set the
        # period end collapses every reporting lag to ~2 weeks (AAPL: 14d instead of the true
        # 34d). They may join the concept set; they may never define the period.
        dates_period = taxonomy != "dei"
        for concept, body in (concepts or {}).items():
            for unit, arr in (body.get("units") or {}).items():
                for r in arr:
                    form = str(r.get("form") or "")
                    if not form.startswith(PERIODIC_PREFIXES):
                        continue
                    accn = r.get("accn")
                    filed = r.get("filed")
                    end = r.get("end")
                    if not accn or not filed or not end:
                        continue
                    if accn not in hdr:
                        hdr[accn] = {"filed": filed, "form": form, "fy": r.get("fy"),
                                     "fp": r.get("fp"), "end": end if dates_period else None}
                    elif dates_period and (hdr[accn]["end"] is None or end > hdr[accn]["end"]):
                        hdr[accn]["end"] = end          # current period = latest us-gaap end
                    cset[accn].add(concept)
                    v = r.get("val")
                    if v is not None:
                        # (concept, unit, start, end) -- start matters: the same `end` carries
                        # both the quarter and the year-to-date version of an income-statement line.
                        vals[accn][(concept, unit, r.get("start"), end)] = v

    accns = [a for a in hdr if len(cset[a]) >= MIN_CONCEPTS and hdr[a]["end"]]
    if len(accns) < 2:
        return pd.DataFrame()
    # Chronological. accn tiebreaks same-day filings deterministically.
    accns.sort(key=lambda a: (hdr[a]["filed"], a))

    # ---- silent revisions: walk forward carrying the last published value of every fact ----
    seen: dict[tuple, float] = {}
    rev_frac, rev_max, rev_n = [], [], []
    for a in accns:
        own_end = hdr[a]["end"]
        n_rep = n_rev = 0
        worst = np.nan
        for k, v in vals[a].items():
            if k[3] >= own_end:          # the period this filing is itself closing: first print
                continue
            prev = seen.get(k)
            if prev is None:
                continue
            n_rep += 1
            # SYMMETRIC relative change, bounded in [0, 1]. A plain |new-prev|/|prev| explodes
            # on near-zero denominators (a fact moving 0 -> 1e8 scored 1e8 and owned the whole
            # distribution); this scores that as 1.0 and a 0.2% move as ~0.001.
            a, b = abs(float(v)), abs(float(prev))
            denom = a + b
            rel = abs(float(v) - float(prev)) / denom if denom > 0 else 0.0
            if rel > REVISION_TOL:
                n_rev += 1
                worst = rel if (np.isnan(worst) or rel > worst) else worst
        rev_frac.append(n_rev / n_rep if n_rep else np.nan)
        rev_max.append(worst)
        rev_n.append(float(n_rev))
        seen.update(vals[a])             # only AFTER scoring, so a filing never revises itself

    # ---- per-event frame -------------------------------------------------------------
    ev = pd.DataFrame({
        "filed_date": [hdr[a]["filed"] for a in accns],
        "form":       [hdr[a]["form"] for a in accns],
        "period_end": [hdr[a]["end"] for a in accns],
        "fy":         [hdr[a]["fy"] for a in accns],
        "fp":         [hdr[a]["fp"] for a in accns],
        "sfl_n_concepts":      [len(cset[a]) for a in accns],
        "sfr_revision_frac":   rev_frac,
        "sfr_revision_max_rel": rev_max,
        "sfr_revision_count":  rev_n,
    })
    ev["filed_date"] = pd.to_datetime(ev["filed_date"], errors="coerce")
    ev["period_end"] = pd.to_datetime(ev["period_end"], errors="coerce")
    ev = ev.dropna(subset=["filed_date"]).reset_index(drop=True)
    if len(ev) < 2:
        return pd.DataFrame()

    ev["_cls"] = ev["form"].map(_form_class)
    ev["sfo_concept_rarity"] = [
        float(np.mean([rarity.get(c, rar_default) for c in cset[a]])) for a in accns]

    # ---- form-aware concept-set churn -------------------------------------------------
    jac, newf, dropf = [], [], []
    last_by_cls: dict[str, set] = {}
    for i, a in enumerate(accns):
        cs = cset[a]
        prev = last_by_cls.get(ev["_cls"].iloc[i])
        if prev is None:
            jac.append(np.nan); newf.append(np.nan); dropf.append(np.nan)
        else:
            union = len(cs | prev)
            jac.append(len(cs & prev) / union if union else np.nan)
            newf.append(len(cs - prev) / len(cs) if cs else np.nan)
            dropf.append(len(prev - cs) / len(prev) if prev else np.nan)
        last_by_cls[ev["_cls"].iloc[i]] = cs
    ev["sfo_concept_jaccard"] = jac
    ev["sfo_concept_new_frac"] = newf
    ev["sfo_concept_drop_frac"] = dropf

    # ---- timing -----------------------------------------------------------------------
    ev["sfl_rep_lag_days"] = (ev["filed_date"] - ev["period_end"]).dt.days.astype(float)
    ev.loc[(ev["sfl_rep_lag_days"] < 0) | (ev["sfl_rep_lag_days"] > 400), "sfl_rep_lag_days"] = np.nan
    ev["sfl_gap_days"] = ev["filed_date"].diff().dt.days.astype(float)
    ev["sfl_is_amended"] = ev["form"].str.upper().str.contains("/A").astype(float)

    # Expanding stats over STRICTLY PRIOR filings: shift(1) first, so row t never sees itself.
    # Lag is compared within form class -- an annual report is structurally slower than a 10-Q.
    ev["sfl_rep_lag_z"] = np.nan
    ev["sfl_rep_lag_excess"] = np.nan
    for cls, idx in ev.groupby("_cls").groups.items():
        idx = list(idx)
        lag = ev.loc[idx, "sfl_rep_lag_days"]
        prior = lag.shift(1)
        med = prior.expanding(min_periods=3).median()
        sd = prior.expanding(min_periods=3).std().replace(0, np.nan)
        ev.loc[idx, "sfl_rep_lag_excess"] = (lag - med).to_numpy()
        ev.loc[idx, "sfl_rep_lag_z"] = ((lag - med) / sd).to_numpy()

    # Breadth and rarity baselines are ALSO form-class-relative: a 10-K tags ~2x the concepts of
    # a 10-Q, so a pooled baseline turns the ordinary annual report into a permanent +2-sigma
    # "complexity shock" every fourth quarter. Compare like with like.
    ev["sfo_n_concepts_z"] = np.nan
    ev["sfo_rarity_drift"] = np.nan
    for cls, idx in ev.groupby("_cls").groups.items():
        idx = list(idx)
        nc = ev.loc[idx, "sfl_n_concepts"].astype(float)
        nc_prior = nc.shift(1)
        ev.loc[idx, "sfo_n_concepts_z"] = (
            (nc - nc_prior.expanding(min_periods=3).mean())
            / nc_prior.expanding(min_periods=3).std().replace(0, np.nan)).to_numpy()
        rare = ev.loc[idx, "sfo_concept_rarity"]
        ev.loc[idx, "sfo_rarity_drift"] = (
            rare - rare.shift(1).expanding(min_periods=3).mean()).to_numpy()

    # Expected cadence from prior gaps only; blocks turn this into "days overdue".
    ev["sfl_expected_gap"] = ev["sfl_gap_days"].shift(1).expanding(min_periods=3).median()

    for c in ("sfl_rep_lag_z", "sfo_n_concepts_z"):
        ev[c] = ev[c].clip(-8, 8)

    ev.insert(1, "ticker", ticker)
    keep = ["filed_date", "ticker", "form", "period_end", "fy", "fp",
            "sfl_rep_lag_days", "sfl_rep_lag_z", "sfl_rep_lag_excess", "sfl_gap_days",
            "sfl_expected_gap", "sfl_is_amended", "sfl_n_concepts",
            "sfo_concept_jaccard", "sfo_concept_new_frac", "sfo_concept_drop_frac",
            "sfo_concept_rarity", "sfo_rarity_drift", "sfo_n_concepts_z",
            "sfr_revision_frac", "sfr_revision_max_rel", "sfr_revision_count"]
    return ev[keep]


def _job(task: tuple[str, str]) -> tuple[str, int, str]:
    ticker, path = task
    try:
        ev = build_one(Path(path), ticker)
    except Exception as exc:              # one bad filer must not kill a 4000-ticker build
        return ticker, -1, f"{type(exc).__name__}: {exc}"
    if ev.empty:
        return ticker, 0, ""
    ev.to_parquet(OUT_DIR / f"{ticker}.parquet", index=False)
    return ticker, len(ev), ""


# --------------------------------------------------------------------------------------
def ticker_cik_map() -> dict[str, str]:
    """ticker -> zero-padded CIK, taken from the fundamentals panel (one row group per ticker)."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(PANEL)
    out: dict[str, str] = {}
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=["ticker", "cik"]).to_pandas().head(1)
        if t.empty:
            continue
        out[str(t["ticker"].iloc[0])] = str(int(t["cik"].iloc[0])).zfill(10)
    return out


def write_stamp(out_dir: Path, n_tickers: int) -> None:
    """
    Record when this lake was built and how current its newest filing is.

    A derived lake that silently stops updating is this project's most expensive known failure
    mode: Data/Indexes froze and ~88 panel columns went all-NaN for ~17 trading sessions with
    nothing anywhere to catch it. Feature blocks reading this lake check the stamp and warn.
    """
    import json
    latest = ""
    for q in sorted(out_dir.glob("*.parquet"))[:200]:
        try:
            d = pd.read_parquet(q, columns=["filed_date"])
        except Exception:
            continue
        if len(d):
            m = str(pd.to_datetime(d["filed_date"]).max().date())
            latest = max(latest, m)
    (out_dir / "_BUILD_STAMP.json").write_text(json.dumps({
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_tickers": n_tickers,
        "max_filed_date": latest,
    }, indent=1), encoding="utf-8")
    print(f"[stamp] {out_dir.name}: {n_tickers} tickers, newest filing {latest}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N tickers (smoke run)")
    ap.add_argument("--tickers", type=str, default="", help="comma-separated subset")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    args = ap.parse_args()

    if not CF_DIR.is_dir():
        print(f"FAIL: missing {CF_DIR} (run fetchers/fetch_sec_companyfacts.py --extract)")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rarity, rar_default = load_rarity()
    print(f"[rarity] {len(rarity)} concepts, default={rar_default:.2f}")

    t0 = time.time()
    cmap = ticker_cik_map()
    print(f"[map] {len(cmap)} ticker->cik ({time.time() - t0:.0f}s)")

    want = {t.strip().upper() for t in args.tickers.split(",") if t.strip()}
    have_price = {p.stem.upper() for p in PRICE_DIR.glob("*.parquet")} if PRICE_DIR.is_dir() else set()

    tasks: list[tuple[str, str]] = []
    for ticker, cik in sorted(cmap.items()):
        if want and ticker.upper() not in want:
            continue
        if not want and have_price and ticker.upper() not in have_price:
            continue                       # no price series -> the features can never be used
        p = CF_DIR / f"CIK{cik}.json"
        if p.exists():
            tasks.append((ticker, str(p)))
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"[tasks] {len(tasks)} tickers, jobs={args.jobs}")

    t0 = time.time()
    written = empty = failed = 0
    if args.jobs <= 1:
        _init_worker(rarity, rar_default)
        results = (_job(t) for t in tasks)
        for ticker, n, err in results:
            if n > 0: written += 1
            elif n == 0: empty += 1
            else: failed += 1; print(f"  !! {ticker}: {err}")
    else:
        with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init_worker,
                                 initargs=(rarity, rar_default)) as pool:
            futs = [pool.submit(_job, t) for t in tasks]
            for k, f in enumerate(as_completed(futs), 1):
                ticker, n, err = f.result()
                if n > 0: written += 1
                elif n == 0: empty += 1
                else: failed += 1; print(f"  !! {ticker}: {err}")
                if k % 500 == 0:
                    print(f"  {k}/{len(tasks)}  ({time.time() - t0:.0f}s)")

    write_stamp(OUT_DIR, written)
    print(f"[done] wrote {written}, empty {empty}, failed {failed}, "
          f"{time.time() - t0:.0f}s -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
