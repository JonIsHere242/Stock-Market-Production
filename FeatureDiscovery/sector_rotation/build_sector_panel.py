"""Build daily equal-weight sector return indices from PriceDataFull + SectorMap.

Outputs (all in this scratchpad dir):
  sector_returns_wide.parquet   Date x Sector equal-weight daily close-to-close returns (NaN when n<30)
  sector_counts_wide.parquet    Date x Sector member counts (post liquidity filter)
  sector_levels_wide.parquet    Date x Sector cumulative total-return index (base 1.0 at first valid date)
  market_refs.parquet           Date x [MKT_EW, SPY, QQQ, IWM, DIA, VIX_ret, VIX_Close, SPY_Close]
  stock_day_panel.parquet       Date, Ticker, Sector, ret (winsorized), ret_raw  (float32, long)
  sector_map_normalized.parquet Ticker -> Sector used here

Sector normalization:
  Technology->Tech, Basic Materials->Materials, Financial->Financials,
  Consumer Cyclical->ConsumerDisc, Real Estate->REIT, Financials/REIT group -> REIT sector,
  ConsumerStaples split from any sector via SIC-description keywords, Unknown dropped.

Liquidity filter per stock-day: Close >= 2 and 21d median dollar volume >= $500k.
Winsorize: daily cross-sectional clamp at 1%/99% quantiles across all stocks.
"""
import os, glob, sys, time
import numpy as np
import pandas as pd

ROOT = r"c:\Users\Masam\Desktop\Stock-Market"
OUT = os.path.dirname(os.path.abspath(__file__))

# ---------------- sector map ----------------
sm = pd.read_parquet(os.path.join(ROOT, "Data", "SectorMap.parquet"))
sm["Ticker"] = sm["Ticker"].str.upper()

SECTOR_NORM = {
    "Technology": "Tech", "Basic Materials": "Materials", "Financial": "Financials",
    "Consumer Cyclical": "ConsumerDisc", "Real Estate": "REIT",
}
sm["Sector"] = sm["Sector"].replace(SECTOR_NORM)
# REITs as their own sector (rate-sensitive, classic rotation bucket)
sm.loc[sm["IndustryGroup"] == "REIT", "Sector"] = "REIT"

# Consumer staples split via SIC description keywords
desc = sm["SICDescription"].fillna("").str.lower()
STAPLE_KW = ["food", "beverage", "bottled", "canned", "dairy", "meat", "bakery", "sugar",
             "confectionery", "grain", "tobacco", "cigarette", "soap", "detergent",
             "cosmetic", "perfume", "toilet", "grocery", "household appliance"]
is_staple = desc.str.contains("|".join(STAPLE_KW), regex=True)
sm.loc[is_staple, "Sector"] = "ConsumerStaples"

sm = sm[sm["Sector"] != "Unknown"][["Ticker", "Sector"]].drop_duplicates("Ticker")
sec_of = dict(zip(sm["Ticker"], sm["Sector"]))
print("sectors:", sm["Sector"].value_counts().to_string())

# ---------------- load prices ----------------
files = glob.glob(os.path.join(ROOT, "Data", "PriceDataFull", "*.parquet"))
print(f"\n{len(files)} price files")
rows, skipped = [], 0
t0 = time.time()
for i, f in enumerate(files):
    tkr = os.path.basename(f)[:-8].upper()
    sec = sec_of.get(tkr)
    if sec is None:
        skipped += 1
        continue
    try:
        df = pd.read_parquet(f, columns=["Date", "Close", "Volume"])
    except Exception:
        skipped += 1
        continue
    if len(df) < 30:
        skipped += 1
        continue
    df = df.dropna(subset=["Date", "Close"]).drop_duplicates("Date").sort_values("Date")
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    close = df["Close"].to_numpy()
    vol = df["Volume"].fillna(0).to_numpy()
    ret = np.empty(len(close)); ret[0] = np.nan
    ret[1:] = close[1:] / close[:-1] - 1.0
    dv = pd.Series(close * vol).rolling(21, min_periods=10).median().to_numpy()
    ok = (close >= 2.0) & (dv >= 5e5) & np.isfinite(ret)
    if ok.sum() < 30:
        skipped += 1
        continue
    sub = pd.DataFrame({"Date": df["Date"].to_numpy()[ok], "Ticker": tkr,
                        "Sector": sec, "ret_raw": ret[ok].astype(np.float32)})
    rows.append(sub)
    if (i + 1) % 800 == 0:
        print(f"  {i+1}/{len(files)}  {time.time()-t0:.0f}s")

panel = pd.concat(rows, ignore_index=True)
del rows
print(f"stock-day rows: {len(panel):,}  skipped files: {skipped}")

# extreme raw-return sanity clip (data errors / halts)
panel = panel[panel["ret_raw"].abs() < 1.5]

# daily cross-sectional winsorize 1%/99%
g = panel.groupby("Date")["ret_raw"]
lo = g.transform(lambda s: s.quantile(0.01))
hi = g.transform(lambda s: s.quantile(0.99))
panel["ret"] = panel["ret_raw"].clip(lo, hi).astype(np.float32)

# ---------------- aggregate ----------------
agg = panel.groupby(["Date", "Sector"])["ret"].agg(["mean", "median", "count"]).reset_index()
wide_ret = agg.pivot(index="Date", columns="Sector", values="mean").sort_index()
wide_cnt = agg.pivot(index="Date", columns="Sector", values="count").sort_index()
wide_ret = wide_ret.where(wide_cnt >= 30)
print("\ncoverage (first date each sector has n>=30):")
for c in wide_ret.columns:
    fv = wide_ret[c].first_valid_index()
    print(f"  {c:16s} {fv}  median_n={wide_cnt[c].median():.0f}")

levels = (1.0 + wide_ret.fillna(0)).cumprod().where(wide_ret.notna().cummax())

# market EW across all stocks
mkt = panel.groupby("Date")["ret"].mean().rename("MKT_EW")

# index refs
refs = {}
for name in ["SPY", "QQQ", "IWM", "DIA", "VIX"]:
    p = os.path.join(ROOT, "Data", "IndexesFull", f"{name}.parquet")
    d = pd.read_parquet(p)
    if "Date" not in d.columns:
        d = d.reset_index().rename(columns={d.index.name or "index": "Date"})
    d["Date"] = pd.to_datetime(d["Date"]).dt.normalize()
    d = d.drop_duplicates("Date").sort_values("Date").set_index("Date")
    refs[name] = d["Close"].pct_change()
    if name in ("SPY", "VIX"):
        refs[name + "_Close"] = d["Close"]
refs = pd.DataFrame(refs)
refs = refs.rename(columns={"VIX": "VIX_ret"})
mr = pd.concat([mkt, refs], axis=1).sort_index()

# ---------------- save ----------------
wide_ret.to_parquet(os.path.join(OUT, "sector_returns_wide.parquet"))
wide_cnt.to_parquet(os.path.join(OUT, "sector_counts_wide.parquet"))
levels.to_parquet(os.path.join(OUT, "sector_levels_wide.parquet"))
mr.to_parquet(os.path.join(OUT, "market_refs.parquet"))
panel[["Date", "Ticker", "Sector", "ret", "ret_raw"]].to_parquet(
    os.path.join(OUT, "stock_day_panel.parquet"), index=False)
sm.to_parquet(os.path.join(OUT, "sector_map_normalized.parquet"), index=False)
print("\nsaved to", OUT)
print("date range:", wide_ret.index.min(), "->", wide_ret.index.max(), f"({len(wide_ret)} days)")
