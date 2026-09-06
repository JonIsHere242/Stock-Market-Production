#!/usr/bin/env python
"""
build_data_panels.py  --  ONE general "data downloader" for every NON-OHLCV data panel the
feature framework and the macro filter consume. Merges the three former standalone builders:

    build_fundamentals_panel.py   ->  Data/Fundamentals/   (SEC XBRL companyfacts)
    build_insider_panel.py        ->  Data/Insider/        (SEC Form 4 insider transactions)
    build_sector_map.py           ->  Data/SectorMap.parquet (ticker -> Sector/IndustryGroup)

WHY MERGE
---------
All three share the same plumbing (tradeable universe == Data/PriceData/*.parquet stems, the
ticker<->CIK map from Data/TickerCikData/TickerCIKs_*.parquet, the same PIT discipline) and all
three feed the SAME nightly prediction. Keeping them as one module lets a single pipeline stage
(`python build_data_panels.py all --refresh-all`) keep every non-price feature fresh for the
latest prediction -- the SEC-fundamentals blocks (FeatureTemplates/_fundamentals*.py), the
insider blocks (FeatureTemplates/_insider*.py), and the macro-filter concentration cap
(7__MacroFilter.py reads Data/SectorMap.parquet) -- alongside the price downloader.

The output paths are UNCHANGED from the three originals, so every existing feature block and the
macro filter keep working without edits to them. The sector classifier (group_from_text,
coarse_sector, COARSE_FROM_GROUP) is still importable -- 7__MacroFilter.py now does
`from build_data_panels import group_from_text, coarse_sector, COARSE_FROM_GROUP`.

THE LINCHPIN (fundamentals + insider): POINT-IN-TIME ON THE FILING DATE
----------------------------------------------------------------------
Every SEC fact carries the date it became PUBLIC (`filed` / FILING_DATE) AND the date it describes
(`end` / TRANS_DATE). We key EVERYTHING on the filing date. A backward merge_asof on the filing
date can never see a filing before it was public, so the downstream features pass the causality
test in FeatureDiscovery/validate_feature.py. Merging on the period/transaction date is leakage.

USAGE
-----
  # one panel at a time (each accepts --refresh to pull its raw source first)
  python build_data_panels.py fundamentals [--refresh] [--runpercent 5] [--ticker AAPL]
  python build_data_panels.py insider      [--refresh] [--runpercent 5] [--ticker AAPL]
  python build_data_panels.py sector       [--refresh] [--finviz --traded]

  # build all three (the nightly pipeline entry point)
  python build_data_panels.py all                   # rebuild all from on-disk raw (no downloads)
  python build_data_panels.py all --refresh-insider  # + re-pull only the cheap insider raw (~15 MB)
  python build_data_panels.py all --refresh-all       # + re-pull ALL raw (companyfacts ~1-2 GB +
                                                       #   ~10 GB extract, submissions ~600-900 MB)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from auxiliary._quiet_progress import tqdm

# ===========================================================================
# Shared paths
# ===========================================================================
ROOT          = Path(__file__).resolve().parent
PRICE_DIR     = ROOT / "Data" / "PriceData"
CIK_MAP_DIR   = ROOT / "Data" / "TickerCikData"
FETCHERS_DIR  = ROOT / "fetchers"

# -- fundamentals --
CF_DIR        = ROOT / "Data" / "SEC" / "companyfacts"
FUND_OUT_DIR  = ROOT / "Data" / "Fundamentals"
FUND_BY_TICKER = FUND_OUT_DIR / "by_ticker"
FUND_PANEL    = FUND_OUT_DIR / "fundamentals_panel.parquet"
FUND_CONCEPT  = FUND_OUT_DIR / "concept_dictionary.parquet"

# -- insider --
INS_RAW_DIR   = ROOT / "Data" / "SEC" / "insider" / "raw"
INS_OUT_DIR   = ROOT / "Data" / "Insider"
INS_BY_TICKER = INS_OUT_DIR / "by_ticker"
INS_PANEL     = INS_OUT_DIR / "insider_panel.parquet"

# -- sector --
SUBMISSIONS_DIR = ROOT / "Data" / "SEC" / "submissions"
SECTOR_OUT      = ROOT / "Data" / "SectorMap.parquet"
FINVIZ_CACHE    = ROOT / "Data" / "finviz_sector_cache.parquet"


# ===========================================================================
# Shared universe + ticker<->CIK mapping  (was duplicated across all three builders)
# ===========================================================================

def _norm(t: str) -> str:
    """Loose ticker key for matching across sources (AAPL == aapl, BRK.B ~ BRK-B ~ BRKB)."""
    return "".join(ch for ch in str(t).upper() if ch.isalnum())


def tradeable_universe() -> list[str]:
    """Tradeable tickers == stems of Data/PriceData/*.parquet (what 3__FeatureFramework --all uses)."""
    return sorted(p.stem for p in PRICE_DIR.glob("*.parquet"))


def latest_cik_map() -> Path:
    cands = sorted(CIK_MAP_DIR.glob("TickerCIKs_*.parquet"))
    if not cands:
        sys.exit(f"No TickerCIKs_*.parquet in {CIK_MAP_DIR}")
    return cands[-1]


def _ticker_cik_pairs() -> list[tuple[str, int]]:
    """[(ticker_str, cik_int)] from the latest TickerCIKs parquet, cik coerced to int."""
    m = pd.read_parquet(latest_cik_map())
    out: list[tuple[str, int]] = []
    for cik, tkr in zip(m["cik"], m["ticker"]):
        try:
            out.append((str(tkr), int(cik)))
        except (TypeError, ValueError):
            continue
    return out


def ticker_cik_map() -> dict[str, int]:
    """{PriceData-stem -> CIK int}, matching exact ticker first then a normalized fallback.
    Restricted to the tradeable universe -- used by the fundamentals builder (per-CIK lookup)."""
    exact: dict[str, int] = {}
    norm: dict[str, int] = {}
    for tkr, c in _ticker_cik_pairs():
        exact.setdefault(tkr.upper(), c)
        norm.setdefault(_norm(tkr), c)
    out: dict[str, int] = {}
    for stem in tradeable_universe():
        c = exact.get(stem.upper()) or norm.get(_norm(stem))
        if c is not None:
            out[stem] = c
    return out


def cik_to_ticker() -> dict[int, str]:
    """{issuer CIK int -> tradeable ticker}. Restricted to the tradeable universe so we only carry
    insider rows we can actually trade -- used by the insider builder (issuer-CIK -> ticker)."""
    exact: dict[str, int] = {}
    norm: dict[str, int] = {}
    for tkr, c in _ticker_cik_pairs():
        exact.setdefault(tkr.upper(), c)
        norm.setdefault(_norm(tkr), c)
    out: dict[int, str] = {}
    for stem in tradeable_universe():
        c = exact.get(stem.upper()) or norm.get(_norm(stem))
        if c is not None:
            out.setdefault(c, stem)   # first tradeable ticker wins a shared CIK
    return out


def load_ticker_ciks() -> pd.DataFrame:
    """Full [ticker, cik, name] universe from the latest TickerCIKs parquet (NOT restricted to the
    tradeable stems -- the sector map classifies every name the macro filter might look up)."""
    df = pd.read_parquet(latest_cik_map())
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["cik"] = pd.to_numeric(df["cik"], errors="coerce").astype("Int64")
    return df.dropna(subset=["cik"]).drop_duplicates("ticker")


# ===========================================================================
# Raw-source refresh helpers (delegate to the existing fetchers; each self-skips when fresh)
# ===========================================================================

def _run_fetcher(script: str, *flags: str) -> None:
    """Run fetchers/<script> with cwd=fetchers so its `from common import ...` resolves."""
    subprocess.run([sys.executable, str(FETCHERS_DIR / script), *flags],
                   cwd=str(FETCHERS_DIR), check=True)


def refresh_fundamentals_raw() -> None:
    print("[refresh] companyfacts.zip via fetchers/fetch_sec_companyfacts.py --extract ...")
    _run_fetcher("fetch_sec_companyfacts.py", "--extract")


def refresh_insider_raw() -> None:
    print("[refresh] insider archives via fetchers/fetch_sec_insider.py ...")
    _run_fetcher("fetch_sec_insider.py")


def _run_builder(script: str, *flags: str) -> None:
    """Run a sibling top-level builder script with the repo root as cwd."""
    subprocess.run([sys.executable, str(ROOT / script), *flags], cwd=str(ROOT), check=True)


def build_filing_meta_panel(workers: int = 8) -> None:
    """Disclosure-behaviour panel (reporting lag / tag churn / silent revisions) from companyfacts."""
    _run_builder("builders/build_filing_meta.py", "--jobs", str(workers))


def build_filing_calendar_panel(workers: int = 8) -> None:
    """Every filing event incl. non-XBRL forms (NT 12b-25, 8-K item codes) from submissions."""
    _run_builder("builders/build_filing_calendar.py", "--jobs", str(workers))


def refresh_submissions_raw() -> None:
    print("[refresh] submissions.zip via fetchers/fetch_sec_submissions.py --extract ...")
    _run_fetcher("fetch_sec_submissions.py", "--extract")


# ###########################################################################
# ###########################################################################
# ##  PANEL 1 -- SEC FUNDAMENTALS  (was build_fundamentals_panel.py)        ##
# ###########################################################################
# ###########################################################################

# The tall-panel column order.
PANEL_COLS = ["ticker", "cik", "taxonomy", "concept", "unit",
              "period_start", "period_end", "filed_date", "fy", "fp", "form", "value"]

_STR_COLS = ["ticker", "taxonomy", "concept", "unit",
             "period_start", "period_end", "filed_date", "fp", "form"]


def _coerce_tall(tall: pd.DataFrame) -> pd.DataFrame:
    """Force a stable dtype per column so every per-ticker shard shares ONE pyarrow schema
    (the streaming ParquetWriter rejects shard-to-shard schema drift)."""
    for c in _STR_COLS:
        tall[c] = tall[c].astype("string")
    tall["cik"] = pd.to_numeric(tall["cik"], errors="coerce").astype("int64")
    tall["fy"] = pd.array(pd.to_numeric(tall["fy"], errors="coerce"), dtype="Int64")
    tall["value"] = pd.to_numeric(tall["value"], errors="coerce").astype("float64")
    return tall


# ---------------------------------------------------------------------------
# CONCEPT_MAP  --  canonical field  ->  ordered candidate XBRL tags (taxonomy, concept)
# ---------------------------------------------------------------------------
# Coalesced in priority order: the first tag a filer actually reports wins. Synonyms exist
# because XBRL has churned tags over the years (e.g. Revenues -> RevenueFromContractWith...).
# FLOW fields are income-statement / cash-flow flows -> they get a trailing-4-quarter TTM.
# STOCK fields are balance-sheet instants -> used as-is (latest known).

FLOW_FIELDS = {
    "revenue": [
        ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
        ("us-gaap", "Revenues"),
        ("us-gaap", "SalesRevenueNet"),
        ("us-gaap", "RevenueFromContractWithCustomerIncludingAssessedTax"),
    ],
    "cost_of_revenue": [
        ("us-gaap", "CostOfRevenue"),
        ("us-gaap", "CostOfGoodsAndServicesSold"),
        ("us-gaap", "CostOfGoodsSold"),
    ],
    "gross_profit": [("us-gaap", "GrossProfit")],
    "operating_income": [
        ("us-gaap", "OperatingIncomeLoss"),
    ],
    "net_income": [
        ("us-gaap", "NetIncomeLoss"),
        ("us-gaap", "ProfitLoss"),
    ],
    "rnd_expense": [("us-gaap", "ResearchAndDevelopmentExpense")],
    "interest_expense": [
        ("us-gaap", "InterestExpense"),
        ("us-gaap", "InterestExpenseNonoperating"),
    ],
    "operating_cash_flow": [
        ("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
        ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
    ],
    "capex": [
        ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"),
        ("us-gaap", "PaymentsToAcquireProductiveAssets"),
    ],
    "dividends_paid": [
        ("us-gaap", "PaymentsOfDividendsCommonStock"),
        ("us-gaap", "PaymentsOfDividends"),
    ],
    "eps_basic": [("us-gaap", "EarningsPerShareBasic")],
    "eps_diluted": [
        ("us-gaap", "EarningsPerShareDiluted"),
        ("us-gaap", "EarningsPerShareBasicAndDiluted"),
    ],
}

STOCK_FIELDS = {
    "assets": [("us-gaap", "Assets")],
    "assets_current": [("us-gaap", "AssetsCurrent")],
    "liabilities": [("us-gaap", "Liabilities")],
    "liabilities_current": [("us-gaap", "LiabilitiesCurrent")],
    "equity": [
        ("us-gaap", "StockholdersEquity"),
        ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    ],
    "cash": [
        ("us-gaap", "CashAndCashEquivalentsAtCarryingValue"),
        ("us-gaap", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
    ],
    "long_term_debt": [
        ("us-gaap", "LongTermDebtNoncurrent"),
        ("us-gaap", "LongTermDebt"),
    ],
    "total_debt": [("us-gaap", "DebtLongtermAndShorttermCombinedAmount")],
    "inventory": [("us-gaap", "InventoryNet")],
    "receivables": [
        ("us-gaap", "AccountsReceivableNetCurrent"),
        ("us-gaap", "ReceivablesNetCurrent"),
    ],
    "ppe_net": [("us-gaap", "PropertyPlantAndEquipmentNet")],
    "goodwill": [("us-gaap", "Goodwill")],
    "shares_outstanding": [
        ("dei", "EntityCommonStockSharesOutstanding"),
        ("us-gaap", "CommonStockSharesOutstanding"),
        ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"),
    ],
}

CANON_FIELDS = {**FLOW_FIELDS, **STOCK_FIELDS}


def cf_path(cik: int) -> Path:
    return CF_DIR / f"CIK{cik:010d}.json"


def parse_companyfacts(path: Path, ticker: str, cik: int):
    """Return (tall_rows: list[dict], concept_meta: dict[(tax,concept)] -> meta).

    Walks facts.{taxonomy}.{concept}.units.{unit}[]. Dedups amendments by keeping, for each
    (concept, unit, period_end), the row with the latest `filed`.
    """
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)

    facts = doc.get("facts", {})
    # best[(tax, concept, unit, end)] = (filed, row_dict)
    best: dict[tuple, tuple] = {}
    concept_meta: dict[tuple, dict] = {}

    for tax, concepts in facts.items():
        for concept, body in concepts.items():
            units = body.get("units", {})
            if not units:
                continue
            meta = concept_meta.setdefault((tax, concept), {
                "label": body.get("label"), "description": body.get("description"),
                "units": set(), "n_obs": 0, "first_filed": None, "last_filed": None})
            for unit, rows in units.items():
                meta["units"].add(unit)
                for r in rows:
                    end = r.get("end")
                    filed = r.get("filed")
                    val = r.get("val")
                    if end is None or filed is None or val is None:
                        continue
                    meta["n_obs"] += 1
                    if meta["first_filed"] is None or filed < meta["first_filed"]:
                        meta["first_filed"] = filed
                    if meta["last_filed"] is None or filed > meta["last_filed"]:
                        meta["last_filed"] = filed
                    key = (tax, concept, unit, end)
                    prev = best.get(key)
                    if prev is None or filed > prev[0]:
                        best[key] = (filed, {
                            "ticker": ticker, "cik": cik, "taxonomy": tax, "concept": concept,
                            "unit": unit, "period_start": r.get("start"), "period_end": end,
                            "filed_date": filed, "fy": r.get("fy"), "fp": r.get("fp"),
                            "form": r.get("form"), "value": val})

    tall_rows = [row for _, row in best.values()]
    return tall_rows, concept_meta


def _period_days(start, end) -> float:
    if start is None or pd.isna(start):
        return float("nan")
    return (pd.Timestamp(end) - pd.Timestamp(start)).days


def _unit_pref(field: str) -> str:
    """Preferred XBRL unit for a canonical field (foreign filers also report in EUR/JPY/...)."""
    if field == "shares_outstanding":
        return "shares"
    if field in ("eps_basic", "eps_diluted"):
        return "USD/shares"
    return "USD"   # monetary -> keep USD so by_ticker values line up with the USD price


def _canonical_series(tall: pd.DataFrame, candidates: list[tuple], unit_pref: str) -> pd.DataFrame:
    """Coalesce candidate tags in priority order -> tidy [filed_date, period_start, period_end, value].

    For each (period_end) we take the highest-priority tag present, latest filing already deduped.
    Within a tag we prefer rows reported in `unit_pref` (USD etc.); fall back to other units only
    if the preferred unit is absent for that tag (keeps a pure-foreign filer from going empty).
    """
    sub = None
    have = {(t, c) for t, c in zip(tall["taxonomy"], tall["concept"])}
    for tax, concept in candidates:
        if (tax, concept) not in have:
            continue
        part = tall[(tall["taxonomy"] == tax) & (tall["concept"] == concept)]
        pref = part[part["unit"] == unit_pref]
        part = pref if not pref.empty else part
        part = part[["filed_date", "period_start", "period_end", "value"]]
        if sub is None:
            sub = part
        else:
            # only fill period_ends the higher-priority tag did not cover
            missing = part[~part["period_end"].isin(sub["period_end"])]
            sub = pd.concat([sub, missing], ignore_index=True)
    if sub is None or sub.empty:
        return pd.DataFrame(columns=["filed_date", "period_start", "period_end", "value"])
    return sub.sort_values(["period_end", "filed_date"]).reset_index(drop=True)


def _ttm(series: pd.DataFrame) -> pd.DataFrame:
    """Trailing-twelve-month value for a FLOW field -> [filed_date, value].

    Quarterly periods (~90d) are summed over a trailing 4; annual periods (~365d) used directly.
    Indexed by the filed_date of the most recent component (point-in-time).
    """
    if series.empty:
        return pd.DataFrame(columns=["filed_date", "value"])
    s = series.copy()
    s["days"] = [_period_days(st, en) for st, en in zip(s["period_start"], s["period_end"])]
    out = []
    quarterly = s[(s["days"] >= 60) & (s["days"] <= 100)].sort_values("period_end")
    annual = s[(s["days"] >= 330) & (s["days"] <= 400)].sort_values("period_end")
    if len(quarterly) >= 4:
        vals = quarterly["value"].astype(float).to_numpy()
        ends = quarterly["period_end"].to_numpy()
        fileds = quarterly["filed_date"].to_numpy()
        for i in range(3, len(quarterly)):
            out.append({"filed_date": fileds[i], "period_end": ends[i],
                        "value": float(vals[i - 3:i + 1].sum())})
    # annual rows give a TTM directly (covers names that only file 10-Ks, or early history)
    for _, r in annual.iterrows():
        out.append({"filed_date": r["filed_date"], "period_end": r["period_end"],
                    "value": float(r["value"])})
    if not out:
        return pd.DataFrame(columns=["filed_date", "value"])
    o = pd.DataFrame(out).sort_values(["period_end", "filed_date"])
    # if both quarterly-TTM and an annual exist for the same period_end, prefer quarterly-TTM
    o = o.drop_duplicates(subset="period_end", keep="first")
    return o[["filed_date", "value"]]


def build_wide(tall: pd.DataFrame) -> pd.DataFrame:
    """Wide point-in-time frame: one row per filed_date, canonical fields + TTM + ratios, ffilled."""
    if tall.empty:
        return pd.DataFrame()
    tall = tall.copy()
    tall["filed_date"] = pd.to_datetime(tall["filed_date"], errors="coerce")
    tall["period_end"] = pd.to_datetime(tall["period_end"], errors="coerce")
    tall["period_start"] = pd.to_datetime(tall["period_start"], errors="coerce")

    # Collect each canonical field as a (filed_date -> value) series, taking the latest value
    # per filing event. Stock fields: as-reported. Flow fields: keep raw quarterly AND a _ttm.
    pieces: dict[str, pd.Series] = {}

    for field, cands in CANON_FIELDS.items():
        ser = _canonical_series(tall, cands, _unit_pref(field))
        if ser.empty:
            continue
        ser = ser.copy()
        ser["filed_date"] = pd.to_datetime(ser["filed_date"], errors="coerce")
        latest = (ser.sort_values("filed_date")
                     .dropna(subset=["filed_date"])
                     .groupby("filed_date")["value"].last())
        pieces[field] = latest.astype(float)

        if field in FLOW_FIELDS:
            ttm = _ttm(ser)
            if not ttm.empty:
                ttm["filed_date"] = pd.to_datetime(ttm["filed_date"], errors="coerce")
                pieces[f"{field}_ttm"] = (ttm.dropna(subset=["filed_date"])
                                             .groupby("filed_date")["value"].last().astype(float))

    if not pieces:
        return pd.DataFrame()

    wide = pd.concat(pieces, axis=1).sort_index()
    # As-of carry-forward: at each filing event, hold the most recent value of every field.
    wide = wide.ffill()
    wide.index.name = "filed_date"
    wide = wide.reset_index()

    # ---- fundamentals-only derived ratios (PIT-safe; price-relative multiples are left to the
    #      feature block, which has the daily Close). Guard divide-by-zero -> NaN. ----
    def col(name):
        return wide[name] if name in wide.columns else pd.Series(index=wide.index, dtype=float)

    def safe_div(a, b):
        a = pd.to_numeric(a, errors="coerce")
        b = pd.to_numeric(b, errors="coerce").replace(0.0, np.nan)
        return (a / b).astype("float64")

    rev_ttm = col("revenue_ttm")
    ni_ttm  = col("net_income_ttm")
    wide["gross_margin"]     = safe_div(col("gross_profit_ttm"), rev_ttm)
    wide["operating_margin"] = safe_div(col("operating_income_ttm"), rev_ttm)
    wide["net_margin"]       = safe_div(ni_ttm, rev_ttm)
    wide["roe"]              = safe_div(ni_ttm, col("equity"))
    wide["roa"]              = safe_div(ni_ttm, col("assets"))
    wide["current_ratio"]    = safe_div(col("assets_current"), col("liabilities_current"))
    wide["debt_to_equity"]   = safe_div(col("long_term_debt"), col("equity"))
    wide["asset_turnover"]   = safe_div(rev_ttm, col("assets"))
    wide["fcf_ttm"]          = col("operating_cash_flow_ttm") - col("capex_ttm")
    wide["book_value_per_share"] = safe_div(col("equity"), col("shares_outstanding"))
    wide["sales_per_share"]      = safe_div(rev_ttm, col("shares_outstanding"))

    return wide.sort_values("filed_date").reset_index(drop=True)


def _fund_worker(ticker: str, cik: int, out_by_ticker: str):
    """Parse one CIK -> write its by_ticker wide parquet -> return (tall_df, concept_meta).

    Top-level so ProcessPoolExecutor can pickle it.
    """
    path = cf_path(cik)
    if not path.exists():
        return ticker, None, None, "no companyfacts file"
    try:
        tall_rows, concept_meta = parse_companyfacts(path, ticker, cik)
    except Exception as exc:
        return ticker, None, None, f"parse error: {exc}"
    if not tall_rows:
        return ticker, None, None, "no facts"

    tall = _coerce_tall(pd.DataFrame(tall_rows, columns=PANEL_COLS))
    try:
        wide = build_wide(tall)
        if not wide.empty:
            wide.insert(1, "ticker", ticker)
            wide.to_parquet(Path(out_by_ticker) / f"{ticker}.parquet", index=False)
    except Exception as exc:
        return ticker, tall, concept_meta, f"wide error: {exc}"
    return ticker, tall, concept_meta, None


def _merge_concept_meta(acc: dict, meta: dict) -> None:
    for key, m in meta.items():
        a = acc.get(key)
        if a is None:
            acc[key] = {"label": m["label"], "description": m["description"],
                        "units": set(m["units"]), "n_obs": m["n_obs"], "n_tickers": 1,
                        "first_filed": m["first_filed"], "last_filed": m["last_filed"]}
        else:
            a["units"].update(m["units"])
            a["n_obs"] += m["n_obs"]
            a["n_tickers"] += 1
            if m["first_filed"] and (a["first_filed"] is None or m["first_filed"] < a["first_filed"]):
                a["first_filed"] = m["first_filed"]
            if m["last_filed"] and (a["last_filed"] is None or m["last_filed"] > a["last_filed"]):
                a["last_filed"] = m["last_filed"]
            if not a["label"]:
                a["label"] = m["label"]
            if not a["description"]:
                a["description"] = m["description"]


def run_fundamentals(tickers_ciks: dict[str, int], workers: int) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    # One fixed schema for every shard so the streaming writer never sees drift.
    panel_schema = pa.schema([
        ("ticker", pa.string()), ("cik", pa.int64()), ("taxonomy", pa.string()),
        ("concept", pa.string()), ("unit", pa.string()), ("period_start", pa.string()),
        ("period_end", pa.string()), ("filed_date", pa.string()), ("fy", pa.int64()),
        ("fp", pa.string()), ("form", pa.string()), ("value", pa.float64())])

    FUND_OUT_DIR.mkdir(parents=True, exist_ok=True)
    FUND_BY_TICKER.mkdir(parents=True, exist_ok=True)

    concept_acc: dict[tuple, dict] = {}
    writer = None          # streaming ParquetWriter for the tall panel (bounded memory)
    n_ok = n_fail = n_rows = 0
    fails: list[str] = []

    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_fund_worker, t, c, str(FUND_BY_TICKER)): t
                    for t, c in tickers_ciks.items()}
            with tqdm(total=len(futs), desc="Fundamentals", unit="tic") as bar:
                for fut in as_completed(futs):
                    ticker, tall, meta, err = fut.result()
                    if tall is not None and meta is not None:
                        if meta:
                            _merge_concept_meta(concept_acc, meta)
                        table = pa.Table.from_pandas(
                            tall[PANEL_COLS], schema=panel_schema, preserve_index=False
                        ).replace_schema_metadata(None)
                        if writer is None:
                            writer = pq.ParquetWriter(FUND_PANEL, panel_schema)
                        writer.write_table(table)
                        n_rows += len(tall)
                    if err:
                        n_fail += 1
                        fails.append(f"{ticker}: {err}")
                    else:
                        n_ok += 1
                    bar.update(1)
                    bar.set_description(f"ok={n_ok} fail={n_fail}")
    finally:
        if writer is not None:
            writer.close()

    # ---- concept dictionary ----
    if concept_acc:
        rows = []
        for (tax, concept), m in concept_acc.items():
            rows.append({"taxonomy": tax, "concept": concept, "label": m["label"],
                         "description": m["description"], "units": ", ".join(sorted(m["units"])),
                         "n_tickers": m["n_tickers"], "n_obs": m["n_obs"],
                         "first_filed": m["first_filed"], "last_filed": m["last_filed"]})
        cd = pd.DataFrame(rows).sort_values("n_tickers", ascending=False).reset_index(drop=True)
        cd.to_parquet(FUND_CONCEPT, index=False)

    print(f"\nFundamentals done. {n_ok} ok, {n_fail} failed (no-file/no-facts included).")
    print(f"  tall panel : {FUND_PANEL}  ({n_rows:,} rows)")
    print(f"  concepts   : {FUND_CONCEPT}  ({len(concept_acc):,} concepts)")
    print(f"  by_ticker  : {FUND_BY_TICKER}/  ({len(list(FUND_BY_TICKER.glob('*.parquet'))):,} files)")
    if fails[:10]:
        print("  first failures:", "; ".join(fails[:10]))


def build_fundamentals(runpercent: int = 100, only_ticker: str | None = None,
                       workers: int | None = None) -> None:
    """Orchestrate the fundamentals panel. `--ticker` runs one CIK verbose, in-process (debug)."""
    workers = workers or min(16, (os.cpu_count() or 4))
    mapping = ticker_cik_map()
    print(f"Fundamentals universe: {len(tradeable_universe())} tradeable tickers, "
          f"{len(mapping)} mapped to a CIK.")

    if only_ticker:
        cik = mapping.get(only_ticker)
        if cik is None:
            sys.exit(f"{only_ticker} not in the ticker->CIK map.")
        FUND_BY_TICKER.mkdir(parents=True, exist_ok=True)
        ticker, tall, meta, err = _fund_worker(only_ticker, cik, str(FUND_BY_TICKER))
        if err:
            print(f"[{ticker}] note: {err}")
        if tall is not None:
            print(f"[{ticker}] CIK {cik}  tall rows={len(tall):,}  concepts={tall['concept'].nunique()}")
            wide = build_wide(tall)
            print(f"[{ticker}] wide: {wide.shape[0]} filing events x {wide.shape[1]} cols")
            if not wide.empty:
                show = [c for c in ("filed_date", "revenue_ttm", "net_income_ttm", "eps_diluted",
                                    "assets", "equity", "shares_outstanding", "net_margin", "roe")
                        if c in wide.columns]
                print(wide[show].tail(4).to_string(index=False))
        return

    items = list(mapping.items())
    if runpercent < 100:
        items = items[: max(1, len(items) * runpercent // 100)]
        mapping = dict(items)
    run_fundamentals(mapping, workers)


# ###########################################################################
# ###########################################################################
# ##  PANEL 2 -- SEC INSIDER (Form 4)  (was build_insider_panel.py)         ##
# ###########################################################################
# ###########################################################################

KEEP_DOC_TYPES = {"4", "4/A"}   # Form 4 ownership-change filings (+ amendments)

INS_OUT_COLS = ["ticker", "cik", "filed_date", "trans_date", "code", "ad",
                "shares", "price", "value", "shares_after",
                "owner_cik", "is_officer", "is_director", "is_tenpct",
                "is_ceo", "is_cfo", "is_deriv"]

_CEO_RE = re.compile(r"\bC\.?E\.?O\b|CHIEF\s+EXEC", re.I)
_CFO_RE = re.compile(r"\bC\.?F\.?O\b|CHIEF\s+FINANC", re.I)


def _read_tsv(zf: zipfile.ZipFile, name: str, usecols: list[str]) -> pd.DataFrame:
    with zf.open(name) as fh:
        return pd.read_csv(fh, sep="\t", dtype=str, usecols=lambda c: c in usecols,
                           encoding="latin-1", on_bad_lines="skip")


def _owner_flags(own: pd.DataFrame) -> pd.DataFrame:
    """One row per ACCESSION_NUMBER with boolean role flags (avg ~1.04 owners/filing -> OR them)."""
    rel = own["RPTOWNER_RELATIONSHIP"].fillna("")
    title = own["RPTOWNER_TITLE"].fillna("")
    own = own.assign(
        is_officer=rel.str.contains("Officer", case=False, na=False),
        is_director=rel.str.contains("Director", case=False, na=False),
        is_tenpct=rel.str.contains("TenPercent", case=False, na=False),
        is_ceo=title.str.contains(_CEO_RE, na=False),
        is_cfo=title.str.contains(_CFO_RE, na=False),
        owner_cik=pd.to_numeric(own["RPTOWNERCIK"], errors="coerce"),
    )
    agg = own.groupby("ACCESSION_NUMBER").agg(
        owner_cik=("owner_cik", "first"),
        is_officer=("is_officer", "any"),
        is_director=("is_director", "any"),
        is_tenpct=("is_tenpct", "any"),
        is_ceo=("is_ceo", "any"),
        is_cfo=("is_cfo", "any"),
    )
    return agg.reset_index()


def parse_quarter(zip_path: Path, cik2tic: dict[int, str]) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        sub = _read_tsv(zf, "SUBMISSION.tsv",
                        ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK"])
        sub = sub[sub["DOCUMENT_TYPE"].isin(KEEP_DOC_TYPES)].copy()
        sub["issuer_cik"] = pd.to_numeric(sub["ISSUERCIK"], errors="coerce")
        sub["ticker"] = sub["issuer_cik"].map(lambda c: cik2tic.get(int(c)) if pd.notna(c) else None)
        sub = sub.dropna(subset=["ticker"])
        if sub.empty:
            return pd.DataFrame(columns=INS_OUT_COLS)
        sub["filed_date"] = pd.to_datetime(sub["FILING_DATE"], format="%d-%b-%Y", errors="coerce")
        sub = sub[["ACCESSION_NUMBER", "ticker", "issuer_cik", "filed_date"]]

        own = _read_tsv(zf, "REPORTINGOWNER.tsv",
                        ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"])
        owner = _owner_flags(own)

        tcols = ["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES",
                 "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD", "SHRS_OWND_FOLWNG_TRANS"]
        parts = []
        for tbl, is_deriv in [("NONDERIV_TRANS.tsv", False), ("DERIV_TRANS.tsv", True)]:
            if tbl not in names:
                continue
            t = _read_tsv(zf, tbl, tcols)
            if t.empty:
                continue
            t["is_deriv"] = is_deriv
            parts.append(t)
        if not parts:
            return pd.DataFrame(columns=INS_OUT_COLS)
        trans = pd.concat(parts, ignore_index=True)

    # Only transactions whose filing we kept (Form 4, tradeable issuer)
    df = trans.merge(sub, on="ACCESSION_NUMBER", how="inner")
    df = df.merge(owner, on="ACCESSION_NUMBER", how="left")

    df["shares"] = pd.to_numeric(df["TRANS_SHARES"], errors="coerce")
    df["price"] = pd.to_numeric(df["TRANS_PRICEPERSHARE"], errors="coerce")
    df["shares_after"] = pd.to_numeric(df["SHRS_OWND_FOLWNG_TRANS"], errors="coerce")
    df["value"] = df["shares"] * df["price"]
    df["trans_date"] = pd.to_datetime(df["TRANS_DATE"], format="%d-%b-%Y", errors="coerce")
    df = df.rename(columns={"TRANS_CODE": "code", "TRANS_ACQUIRED_DISP_CD": "ad",
                            "issuer_cik": "cik"})

    for c in ["is_officer", "is_director", "is_tenpct", "is_ceo", "is_cfo"]:
        df[c] = df[c].fillna(False).astype(bool)
    df["is_deriv"] = df["is_deriv"].astype(bool)
    df = df.dropna(subset=["filed_date"])
    return df[INS_OUT_COLS]


def build_insider(runpercent: int = 100, only_ticker: str | None = None) -> None:
    cik2tic = cik_to_ticker()
    if only_ticker:
        cik2tic = {c: t for c, t in cik2tic.items() if t == only_ticker}
        if not cik2tic:
            sys.exit(f"{only_ticker} not mapped to a CIK in {latest_cik_map().name}")
    print(f"Insider universe: {len(tradeable_universe())} tradeable tickers, "
          f"{len(cik2tic)} issuer CIKs mapped.")

    zips = sorted(INS_RAW_DIR.glob("*_form345.zip"))
    if not zips:
        sys.exit(f"No *_form345.zip in {INS_RAW_DIR}. Run fetchers/fetch_sec_insider.py first.")

    frames = []
    for zp in tqdm(zips, desc="Quarters", unit="qtr"):
        try:
            frames.append(parse_quarter(zp, cik2tic))
        except Exception as exc:
            print(f"  {zp.name}: parse error {exc}")
    panel = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    panel = panel.sort_values(["ticker", "filed_date"]).reset_index(drop=True)
    print(f"Parsed {len(panel):,} transactions for {panel['ticker'].nunique():,} tickers "
          f"({panel['filed_date'].min().date()} -> {panel['filed_date'].max().date()})")

    INS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    INS_BY_TICKER.mkdir(parents=True, exist_ok=True)

    if not only_ticker:
        panel.to_parquet(INS_PANEL, index=False)

    tickers = sorted(panel["ticker"].unique())
    if runpercent < 100:
        tickers = tickers[: max(1, len(tickers) * runpercent // 100)]
    n = 0
    for tic, g in tqdm(panel.groupby("ticker"), total=panel["ticker"].nunique(),
                       desc="Write by_ticker", unit="tic"):
        if tic not in tickers:
            continue
        g.drop(columns=["ticker"]).reset_index(drop=True).to_parquet(
            INS_BY_TICKER / f"{tic}.parquet", index=False)
        n += 1

    print(f"\nInsider done. tall panel: {INS_PANEL if not only_ticker else '(skipped)'}")
    print(f"  by_ticker : {INS_BY_TICKER}/  ({n:,} files)")
    if only_ticker:
        g = panel
        buys = g[(g.code == "P") & (~g.is_deriv)]
        sells = g[(g.code == "S") & (~g.is_deriv)]
        print(f"  [{only_ticker}] {len(g):,} txns | {len(buys):,} open-mkt buys, "
              f"{len(sells):,} sells | last filed {g['filed_date'].max().date()}")
        print(g.tail(6).to_string(index=False))


# ###########################################################################
# ###########################################################################
# ##  PANEL 3 -- SECTOR MAP  (was build_sector_map.py)                      ##
# ##  NOTE: group_from_text / coarse_sector / COARSE_FROM_GROUP are the     ##
# ##  importable contract consumed by 7__MacroFilter.py -- keep the names.  ##
# ###########################################################################
# ###########################################################################

# Fine group -> coarse sector (for the looser sector-level cap).
COARSE_FROM_GROUP = {
    "PreciousMetals": "Materials", "MetalMiner": "Materials", "Chemicals": "Materials",
    "Agriculture": "Materials", "Mining": "Materials", "BuildingMaterials": "Materials",
    "OilGasE&P": "Energy", "OilGasServices": "Energy", "Refiner": "Energy",
    "Coal": "Energy", "Uranium": "Energy", "Midstream": "Energy",
    "Biotech": "Healthcare", "Pharma": "Healthcare", "MedicalDevice": "Healthcare",
    "HealthServices": "Healthcare", "Diagnostics": "Healthcare",
    "Semiconductor": "Tech", "Software": "Tech", "ITServices": "Tech", "Hardware": "Tech",
    "Solar": "Tech", "CommEquip": "Tech",
    "REIT": "Financials", "Bank": "Financials", "Insurance": "Financials",
    "FinanceInvest": "Financials", "AssetManager": "Financials", "Finance": "Financials",
    "Telecom": "Communications", "Media": "Communications",
    "Utility": "Utilities",
    "Restaurant": "ConsumerDisc", "Auto": "ConsumerDisc", "Retail": "ConsumerDisc",
    "Apparel": "ConsumerDisc", "Travel": "ConsumerDisc",
    "Wholesale": "ConsumerStaples", "FoodBev": "ConsumerStaples",
    "Construction": "Industrials", "Transport": "Industrials", "Aerospace": "Industrials",
    "Manufacturing": "Industrials", "Services": "Services",
}


def coarse_sector(group: str) -> str:
    return COARSE_FROM_GROUP.get(group, "Other")


def group_from_text(text: str) -> str | None:
    """Map any descriptive string to a fine IndustryGroup, or None if no confident match.
    Used for FinViz Industry, SEC sicDescription, and company names alike."""
    if not text:
        return None
    t = str(text).lower()
    # precious metals (gold/silver/PGM) pool together - the correlated selloff risk
    if any(k in t for k in ("gold", "silver", "precious metal", "platinum", "palladium")):
        return "PreciousMetals"
    if any(k in t for k in ("copper", "steel", "aluminum", "iron ore", "industrial metal",
                            "base metal", "nickel", "zinc", "lithium", "rare earth")):
        return "MetalMiner"
    if "uranium" in t:
        return "Uranium"
    if "coal" in t:
        return "Coal"
    if "biotech" in t or "biolog" in t or "biosci" in t or "in vitro" in t:
        return "Biotech"
    if "pharmac" in t or "drug manufacturer" in t or "medicinal" in t or "therapeutics" in t:
        return "Pharma"
    if "diagnostic" in t or ("research" in t and "biolog" in t):
        return "Diagnostics"
    if "medical device" in t or "medical instrument" in t or "surgical" in t or "electromedical" in t:
        return "MedicalDevice"
    if "medical care" in t or "hospital" in t or "health" in t and "service" in t:
        return "HealthServices"
    if "semiconductor" in t:
        return "Semiconductor"
    if "software" in t or "prepackaged" in t:
        return "Software"
    if "information technology service" in t or "it service" in t or "computer programming" in t \
            or "information retrieval" in t:
        return "ITServices"
    if "solar" in t:
        return "Solar"
    if "oil & gas e&p" in t or "exploration & production" in t or "crude petroleum" in t:
        return "OilGasE&P"
    if "oil & gas" in t and ("service" in t or "equipment" in t or "drilling" in t):
        return "OilGasServices"
    if "oil & gas midstream" in t or "pipeline" in t:
        return "Midstream"
    if "refining" in t or "petroleum refining" in t:
        return "Refiner"
    if "reit" in t or "real estate investment trust" in t:
        return "REIT"
    if "bank" in t:
        return "Bank"
    if "asset management" in t or "capital markets" in t:
        return "AssetManager"
    if "insurance" in t:
        return "Insurance"
    if "restaurant" in t or "eating" in t:
        return "Restaurant"
    if "auto manufacturer" in t or "motor vehicle" in t or "auto parts" in t:
        return "Auto"
    if "chemical" in t:
        return "Chemicals"
    return None


def classify_sic(sic: str, desc: str):
    by_text = group_from_text(desc)
    s = "".join(ch for ch in str(sic) if ch.isdigit())
    n = int(s) if s else 0

    if n == 1040:
        return "Materials", "PreciousMetals"
    if 1000 <= n <= 1099:
        return "Materials", "MetalMiner"
    if n in (1220, 1221):
        return "Energy", "Coal"
    if n == 1311:
        return "Energy", "OilGasE&P"
    if 1381 <= n <= 1389:
        return "Energy", "OilGasServices"
    if n == 2911:
        return "Energy", "Refiner"
    if n in (2836, 8731):
        return "Healthcare", "Biotech"
    if n in (2833, 2834, 2835):
        return "Healthcare", "Pharma"
    if n in (3826, 3841, 3842, 3845, 3851):
        return "Healthcare", "MedicalDevice"
    if 8000 <= n <= 8099:
        return "Healthcare", "HealthServices"
    if n == 3674:
        return "Tech", "Semiconductor"
    if n == 7372:
        return "Tech", "Software"
    if 7370 <= n <= 7379:
        return "Tech", "ITServices"
    if n == 6798:
        return "Financials", "REIT"
    if n in (6020, 6021, 6022, 6035, 6036):
        return "Financials", "Bank"
    if 6300 <= n <= 6411:
        return "Financials", "Insurance"
    if n in (6199, 6200, 6770, 6726):
        return "Financials", "FinanceInvest"
    if 4800 <= n <= 4899:
        return "Communications", "Telecom"
    if 4900 <= n <= 4991:
        return "Utilities", "Utility"
    if 5800 <= n <= 5813:
        return "ConsumerDisc", "Restaurant"
    if 3711 <= n <= 3716:
        return "ConsumerDisc", "Auto"
    if 2800 <= n <= 2899:
        return "Materials", "Chemicals"

    # description keyword fallback within SIC, then coarse division
    if by_text:
        return coarse_sector(by_text), by_text
    if 1500 <= n <= 1799:
        return "Industrials", "Construction"
    if 2000 <= n <= 3999:
        return "Industrials", "Manufacturing"
    if 4000 <= n <= 4999:
        return "Industrials", "TransportUtil"
    if 5200 <= n <= 5999:
        return "ConsumerDisc", "Retail"
    if 6000 <= n <= 6799:
        return "Financials", "Finance"
    if 7000 <= n <= 8999:
        return "Services", "Services"
    return "Unknown", "Unknown"


def classify(sic, desc, name, fv_sector, fv_industry):
    """Priority: FinViz Industry -> SIC -> company name. Returns (sector, group, source)."""
    g = group_from_text(fv_industry)
    if g:
        sec = fv_sector if fv_sector else coarse_sector(g)
        return sec, g, "finviz"
    sec, grp = classify_sic(sic, desc)
    if grp != "Unknown":
        return sec, grp, "sic"
    g = group_from_text(name)
    if g:
        return coarse_sector(g), g, "name"
    return "Unknown", "Unknown", "none"


def sic_for_cik(cik: int):
    fp = SUBMISSIONS_DIR / f"CIK{int(cik):010d}.json"
    if not fp.exists():
        return "", ""
    try:
        with open(fp, "r", encoding="utf-8") as f:
            d = json.load(f)
        return str(d.get("sic", "") or ""), str(d.get("sicDescription", "") or "")
    except Exception:
        return "", ""


def load_finviz_cache() -> dict:
    if not FINVIZ_CACHE.exists():
        return {}
    try:
        df = pd.read_parquet(FINVIZ_CACHE)
        out = {}
        for r in df.itertuples():
            sec, ind = str(r.FvSector), str(r.FvIndustry)
            # Skip blank/poisoned rows (see pull_finviz) so they are re-pulled instead of
            # counting as "already cached" forever. Mirrors 7__MacroFilter.load_finviz_cache.
            if not ind or ind in ("nan", "None"):
                continue
            out[str(r.Ticker).upper()] = (sec, ind)
        return out
    except Exception:
        return {}


def _save_finviz_cache(cache: dict):
    pd.DataFrame([{"Ticker": k, "FvSector": v[0], "FvIndustry": v[1]} for k, v in cache.items()]) \
        .to_parquet(FINVIZ_CACHE, index=False)


def pull_finviz(tickers, cache: dict, sleep=0.4) -> dict:
    """Best-effort FinViz Sector/Industry for tickers not already cached. Resumable.

    2026-07-28, TWO bugs fixed here:

    1. PARSER. This used finvizfinance's ticker_fundament(), which RAISES against FinViz's
       current markup (it looks for `div.quote-links`, which FinViz renamed to
       `a.quote-header_category` links). finvizfinance 1.3.0 ships the identical bug, so
       there is no upgrade to wait for - we parse via finviz_compat instead.

    2. POISON CACHE (and it made the "resumable" claim false). On failure this wrote
       `cache[t] = ("", "")`. Because load_finviz_cache() returns those blanks and `todo`
       skips anything already IN the cache, one bad run permanently marked a ticker as
       "done" with empty sector/industry - no later run would ever retry it. Combined with
       bug 1 (every call failing) a single pull would have blanked the whole universe.
       Failures are no longer cached, so they simply retry next run.

    Reports a failure count at the end; a 100% failure rate is called out loudly rather
    than silently producing an empty map.
    """
    from auxiliary.finviz_compat import sector_industry
    # Treat a cached-but-blank industry as NOT done, so previously poisoned rows self-heal.
    todo = [t for t in tickers
            if not (cache.get(t.upper()) or ("", ""))[1]]
    print(f"FinViz pull: {len(todo)} tickers (of {len(tickers)}) missing sector/industry.")
    fails = 0
    first_err = None
    for i, t in enumerate(todo):
        try:
            sec, ind = sector_industry(t)
            if ind:
                cache[t.upper()] = (sec, ind)
            else:
                fails += 1          # reachable but unclassified - do NOT cache a blank
        except Exception as e:      # noqa: BLE001
            fails += 1
            if first_err is None:
                first_err = f"{type(e).__name__}: {e}"
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(todo)} (failures so far: {fails})"); _save_finviz_cache(cache)
        time.sleep(sleep)
    _save_finviz_cache(cache)
    if todo:
        print(f"FinViz pull done: {len(todo) - fails} resolved, {fails} failed "
              f"(not cached, will retry next run).")
        if fails == len(todo):
            print(f"  !! FinViz resolved NOTHING ({fails}/{len(todo)} failed). First error: "
                  f"{first_err}. FinViz likely changed their markup again - fix the "
                  f"selectors in finviz_compat.py (check: python auxiliary/finviz_compat.py AAPL). "
                  f"7__MacroFilter's concentration cap runs on the SIC sector map until "
                  f"this is fixed.")
    return cache


def build_sector(finviz: bool = False, traded: bool = False,
                 tickers: list[str] | None = None, limit: int | None = None) -> None:
    tc = load_ticker_ciks()
    fv = load_finviz_cache()

    if finviz:
        if tickers:
            want = [t.upper() for t in tickers]
        elif traded:
            th = []
            for p in ("trade_history.parquet", "Data/TradeHistory.parquet"):
                fp = ROOT / p
                if fp.exists():
                    th.append(pd.read_parquet(fp)["Symbol"].astype(str).str.upper())
            want = sorted(set(pd.concat(th))) if th else list(tc["ticker"])
        else:
            want = list(tc["ticker"])[: limit] if limit else list(tc["ticker"])
        fv = pull_finviz(want, fv)

    rows = []
    for tkr, cik, name in zip(tc["ticker"], tc["cik"], tc["name"]):
        sic, desc = sic_for_cik(int(cik))
        fv_sec, fv_ind = fv.get(tkr, ("", ""))
        sector, group, source = classify(sic, desc, name, fv_sec, fv_ind)
        rows.append({"Ticker": tkr, "CIK": int(cik), "SIC": sic, "SICDescription": desc,
                     "Sector": sector, "IndustryGroup": group, "Source": source})
    out = pd.DataFrame(rows)
    SECTOR_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(SECTOR_OUT, index=False)

    known = out[out["IndustryGroup"] != "Unknown"]
    print(f"\nSector map done. Wrote {SECTOR_OUT}: {len(out)} tickers, {len(known)} classified "
          f"({len(known)/len(out)*100:.0f}%).  Sources: {out['Source'].value_counts().to_dict()}")
    print("Top IndustryGroups:")
    print(known["IndustryGroup"].value_counts().head(20).to_string())
    print("PreciousMetals names:", out[out["IndustryGroup"] == "PreciousMetals"]["Ticker"].tolist()[:40])


# ###########################################################################
# ##  CLI
# ###########################################################################

def _cmd_all(args) -> int:
    """Build every panel. Returns process exit code (0 = every panel OK)."""
    # 1) refresh raw sources first (each fetcher self-skips when its file is fresh)
    if args.refresh_all:
        for fn in (refresh_insider_raw, refresh_submissions_raw, refresh_fundamentals_raw):
            try:
                fn()
            except Exception as exc:
                print(f"[refresh] {fn.__name__} failed: {exc}")
    elif args.refresh_insider:
        try:
            refresh_insider_raw()
        except Exception as exc:
            print(f"[refresh] refresh_insider_raw failed: {exc}")

    # 2) build every panel; isolate failures so one bad source doesn't sink the rest
    failures: list[str] = []
    panels = [
        ("sector",       lambda: build_sector(finviz=args.finviz)),
        ("insider",      lambda: build_insider(runpercent=args.runpercent)),
        ("fundamentals", lambda: build_fundamentals(runpercent=args.runpercent, workers=args.workers)),
        # Both read the ticker<->cik map out of fundamentals_panel.parquet, so they MUST run after
        # the fundamentals panel. They feed FeatureTemplates/sfn_*, sfd_*, sfb_* -- if they stop
        # running, those columns decay silently toward zero rather than erroring, which is why
        # each writes a _BUILD_STAMP.json the feature helpers check for staleness.
        ("filing_meta",     lambda: build_filing_meta_panel(workers=args.workers)),
        ("filing_calendar", lambda: build_filing_calendar_panel(workers=args.workers)),
    ]
    for name, fn in panels:
        print(f"\n{'='*70}\n== Building panel: {name}\n{'='*70}")
        t0 = time.perf_counter()
        try:
            fn()
            print(f"  [{name}] wall: {time.perf_counter() - t0:.1f}s")
        except SystemExit as exc:                       # sub-builders sys.exit on missing inputs
            failures.append(f"{name}: {exc}")
            print(f"  [{name}] SKIPPED: {exc}")
        except Exception as exc:
            failures.append(f"{name}: {exc}")
            print(f"  [{name}] FAILED: {exc}")

    if failures:
        print(f"\nDATA PANELS: {len(panels) - len(failures)}/{len(panels)} OK. "
              f"Failures: {'; '.join(failures)}")
        return 1
    print(f"\nDATA PANELS COMPLETE: all {len(panels)} panels built.")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Unified builder for the SEC fundamentals, insider, sector-map, filing-meta and filing-calendar data panels.")
    sub = ap.add_subparsers(dest="command", required=True)

    pf = sub.add_parser("fundamentals", help="SEC companyfacts -> Data/Fundamentals/")
    pf.add_argument("--refresh", action="store_true", help="fetch the latest SEC companyfacts.zip + extract first")
    pf.add_argument("--ticker", help="build a single ticker (verbose, in-process) for debugging")
    pf.add_argument("--runpercent", type=int, default=100, help="percentage of the universe to build")
    pf.add_argument("--workers", type=int, default=min(16, (os.cpu_count() or 4)))

    pi = sub.add_parser("insider", help="SEC Form 4 data sets -> Data/Insider/")
    pi.add_argument("--refresh", action="store_true", help="fetch the latest insider quarter first")
    pi.add_argument("--ticker", help="build a single ticker (verbose) for debugging")
    pi.add_argument("--runpercent", type=int, default=100, help="percentage of the universe to write")

    ps = sub.add_parser("sector", help="SEC SIC + FinViz -> Data/SectorMap.parquet")
    ps.add_argument("--refresh", action="store_true", help="fetch the latest SEC submissions.zip (SIC) first")
    ps.add_argument("--finviz", action="store_true", help="pull FinViz sector/industry (slow, network)")
    ps.add_argument("--traded", action="store_true", help="with --finviz: pull only names that ever traded")
    ps.add_argument("--tickers", nargs="+", help="with --finviz: pull only these tickers")
    ps.add_argument("--limit", type=int, help="with --finviz: cap how many universe names to pull")

    pm = sub.add_parser("filing_meta", help="SEC companyfacts -> Data/Fundamentals/filing_meta/")
    pm.add_argument("--workers", type=int, default=min(8, (os.cpu_count() or 4)))

    pc = sub.add_parser("filing_calendar", help="SEC submissions -> Data/SEC/filing_calendar/")
    pc.add_argument("--workers", type=int, default=min(8, (os.cpu_count() or 4)))

    pa = sub.add_parser("all", help="build all five panels (the nightly pipeline entry point)")
    pa.add_argument("--refresh-all", action="store_true",
                    help="re-download ALL raw SEC sources first (companyfacts ~1-2 GB + ~10 GB extract, "
                         "submissions ~600-900 MB, insider). Fetchers self-skip files <20h old.")
    pa.add_argument("--refresh-insider", action="store_true",
                    help="re-download only the cheap insider raw (~15 MB) first")
    pa.add_argument("--finviz", action="store_true", help="also pull FinViz industry for the sector map (slow)")
    pa.add_argument("--runpercent", type=int, default=100)
    pa.add_argument("--workers", type=int, default=min(16, (os.cpu_count() or 4)))

    args = ap.parse_args()
    t0 = time.perf_counter()

    if args.command == "fundamentals":
        if args.refresh:
            refresh_fundamentals_raw()
        build_fundamentals(runpercent=args.runpercent, only_ticker=args.ticker, workers=args.workers)
    elif args.command == "insider":
        if args.refresh:
            refresh_insider_raw()
        build_insider(runpercent=args.runpercent, only_ticker=args.ticker)
    elif args.command == "filing_meta":
        build_filing_meta_panel(workers=args.workers)
    elif args.command == "filing_calendar":
        build_filing_calendar_panel(workers=args.workers)
    elif args.command == "sector":
        if args.refresh:
            refresh_submissions_raw()
        build_sector(finviz=args.finviz, traded=args.traded, tickers=args.tickers, limit=args.limit)
    elif args.command == "all":
        code = _cmd_all(args)
        print(f"  total wall: {time.perf_counter() - t0:.1f}s")
        sys.exit(code)

    print(f"  wall: {time.perf_counter() - t0:.1f}s")




if __name__ == "__main__":
    main()


