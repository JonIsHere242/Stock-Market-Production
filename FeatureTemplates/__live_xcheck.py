"""
__live_xcheck.py  --  cross-check a feature's signal against REALIZED trade PnL.

WHY
---
Every offline screen (__tail_screen, __diagnostics) scores features against a
SYNTHETIC label: "is this name a next-day cross-sectional top-decile winner by
log-return". That proxy can drift from what the live book actually realizes (the
intraday fill sim already showed the -1.9% stop touches 52% of trades and both
live filters cost money -- realized PnL != next-day return). This module is the
free validity gate: does a feature's ranking actually line up with the dollars the
strategy MADE on the names it really traded?

It is deliberately a SANITY GATE, not a screen. The realized-trade set is sparse
(<=4 names/day, ~1100 trades total, post-2025-06) and is already model-selected, so
N is small and the cross-section per day is tiny. Treat a result as "does this
feature contradict realized PnL?" not "does this feature have alpha?". A feature
that ranks realized winners ABOVE realized losers is corroborated; one that inverts
is a red flag worth a second look. Small-N caveats are printed loudly.

KEY ALIGNMENT  (verified against the data + project memory)
-------------
  trade EntryDate == the SIGNAL day == the feature row's Date.
  (The live broker decides on the signal day and fills next session open; the fill
   sim confirmed EntryDate is the signal day.) So we join feature panel (Ticker,Date)
   directly onto realized trades (Symbol,EntryDate). No shift.

SOURCES
-------
  sim  (default) : Data/IntradayFillSim/sim_trades.parquet  -> SimPnLPctNet
                   (realistic 10:00 entry, SPY-abort, gap-skip, bracket walk, net of
                    slippage -- the most honest realized return the house has)
  bt             : trade_history.parquet                     -> PnLPct
                   (daily-bar backtester fills; coarser but longer/denser)

API
---
  trades = load_realized(source="sim")          # Ticker, Date, ret, winner, taken
  res    = crosscheck_feature(panel, "col")     # dict of overlap stats vs realized PnL
  # panel must have Date, Ticker, and the feature column.

CLI
---
  python FeatureTemplates/__live_xcheck.py --blocks price_momentum_features --n 800
  python FeatureTemplates/__live_xcheck.py --source bt --blocks rdd_pt_loss,residual_momentum
"""
from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PRICE_DIR = ROOT / "Data" / "PriceData"

SIM_TRADES = ROOT / "Data" / "IntradayFillSim" / "sim_trades.parquet"
BT_TRADES = ROOT / "trade_history.parquet"

try:
    from scipy import stats as _ss
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ---------------------------------------------------------------------------
def load_realized(source: str = "sim", *, taken_only: bool = True) -> pd.DataFrame:
    """Return realized trades as (Ticker, Date, ret, winner, taken).

    ret    : realized return in % (SimPnLPctNet for sim, PnLPct for bt)
    winner : ret > 0
    taken  : sim only -- whether the live filters actually took the trade.
    Date   : normalized to midnight (matches the feature panel's daily Date).
    """
    source = source.lower()
    if source == "sim":
        if not SIM_TRADES.exists():
            raise FileNotFoundError(
                f"{SIM_TRADES} not found -- run 8__IntradayFillSim.py first, or use --source bt")
        df = pd.read_parquet(SIM_TRADES)
        ret_col = "SimPnLPctNet" if "SimPnLPctNet" in df.columns else "SimPnLPct"
        out = pd.DataFrame({
            "Ticker": df["Symbol"].astype(str),
            "Date": pd.to_datetime(df["EntryDate"]).dt.normalize(),
            "ret": pd.to_numeric(df[ret_col], errors="coerce"),
        })
        out["taken"] = (df.get("Status", "OK").astype(str).str.upper().eq("OK")
                        & df.get("FilterVerdict", "TAKEN").astype(str).str.upper().eq("TAKEN"))
        if taken_only:
            out = out[out["taken"]]
    elif source == "bt":
        if not BT_TRADES.exists():
            raise FileNotFoundError(f"{BT_TRADES} not found")
        df = pd.read_parquet(BT_TRADES)
        out = pd.DataFrame({
            "Ticker": df["Symbol"].astype(str),
            "Date": pd.to_datetime(df["EntryDate"]).dt.normalize(),
            "ret": pd.to_numeric(df["PnLPct"], errors="coerce"),
        })
        out["taken"] = True
    else:
        raise ValueError(f"unknown source {source!r} (use 'sim' or 'bt')")

    out = out.dropna(subset=["ret"]).reset_index(drop=True)
    out["winner"] = out["ret"] > 0.0
    return out


def _spearman(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 8:
        return np.nan, int(m.sum())
    if _HAVE_SCIPY:
        c, _ = _ss.spearmanr(a[m], b[m])
        return (float(c) if np.isfinite(c) else np.nan), int(m.sum())
    ar = pd.Series(a[m]).rank().to_numpy()
    br = pd.Series(b[m]).rank().to_numpy()
    c = np.corrcoef(ar, br)[0, 1]
    return (float(c) if np.isfinite(c) else np.nan), int(m.sum())


def crosscheck_feature(panel: pd.DataFrame, col: str, trades: pd.DataFrame,
                       *, top_q: float = 0.5) -> dict:
    """Join feature[col] onto realized trades by (Ticker, Date); summarize agreement.

    Returns a dict:
      n           overlapping (feature, realized-trade) rows
      ic          Spearman(feature value, realized ret) on the overlap  [the headline]
      hi_ret      mean realized ret among the feature's UPPER half (by value)
      lo_ret      mean realized ret among the feature's LOWER half
      spread      hi_ret - lo_ret  (>0 means feature ranks realized winners higher)
      hi_winrate  win-rate in the upper half
      base_win    overall realized win-rate on the overlap (the bar)
    """
    if col not in panel.columns:
        return {"col": col, "n": 0}
    feat = panel[["Date", "Ticker", col]].copy()
    feat["Date"] = pd.to_datetime(feat["Date"]).dt.normalize()
    feat["Ticker"] = feat["Ticker"].astype(str)
    feat = feat.dropna(subset=[col]).drop_duplicates(["Ticker", "Date"], keep="last")

    j = trades.merge(feat, on=["Ticker", "Date"], how="inner")
    n = len(j)
    out = {"col": col, "n": n}
    if n < 8:
        return out

    ic, n_ic = _spearman(j[col].to_numpy(), j["ret"].to_numpy())
    out["ic"] = ic
    out["base_win"] = float(j["winner"].mean())

    # upper/lower split by feature value (median); spread of realized return
    thr_hi = j[col].quantile(1.0 - top_q)
    thr_lo = j[col].quantile(top_q)
    hi = j[j[col] >= thr_hi]
    lo = j[j[col] <= thr_lo]
    if len(hi) >= 4 and len(lo) >= 4:
        out["hi_ret"] = float(hi["ret"].mean())
        out["lo_ret"] = float(lo["ret"].mean())
        out["spread"] = out["hi_ret"] - out["lo_ret"]
        out["hi_winrate"] = float(hi["winner"].mean())
        out["n_hi"] = int(len(hi))
        out["n_lo"] = int(len(lo))
    return out


# ---------------------------------------------------------------------------
# panel construction (reuse the tail-screen builder if available)
# ---------------------------------------------------------------------------
_cspec = importlib.util.spec_from_file_location("__common", HERE / "__common.py")
_common = importlib.util.module_from_spec(_cspec)
_cspec.loader.exec_module(_common)

_load_block = _common.load_block


def build_panel(n, seed, blocks):
    """Dependency-correct panel build -- see __common.build_panel.

    This file used to carry a byte-for-byte copy of the broken version (no
    resolve_order, no requires check, `except Exception: pass`), so any block
    with a dependency silently contributed nothing and read as 'no edge'.
    sort_panel=False preserves this tool's original row order.
    """
    return _common.build_panel(n, seed, blocks, price_dir=PRICE_DIR, sort_panel=False)


def _c(t, code):
    return f"\033[{code}m{t}\033[0m"


def main():
    ap = argparse.ArgumentParser(description="Cross-check feature signal vs realized trade PnL")
    ap.add_argument("--blocks", type=str, required=True,
                    help="comma-separated block names to score")
    ap.add_argument("--source", type=str, default="sim", choices=["sim", "bt"])
    ap.add_argument("--n", type=int, default=800, help="ticker sample for the panel")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all_trades", action="store_true",
                    help="(sim) include filter-skipped trades, not just TAKEN")
    ap.add_argument("--no_ledger", action="store_true")
    args = ap.parse_args()

    blocks = [b.strip() for b in args.blocks.split(",") if b.strip()]
    trades = load_realized(args.source, taken_only=not args.all_trades)
    print(f"\n  live cross-check  | source={args.source}  realized trades={len(trades):,}  "
          f"span {trades['Date'].min().date()}..{trades['Date'].max().date()}  "
          f"base win-rate {trades['winner'].mean():.3f}")
    if len(trades) < 50:
        print(_c("  [!] very few realized trades -- treat results as a weak sanity gate only.", "33"))

    panel, feat_cols, family = build_panel(args.n, args.seed, blocks)
    # restrict to dates overlapping the trade window (huge speedup, no effect on join)
    lo, hi = trades["Date"].min(), trades["Date"].max()
    panel = panel[(panel["Date"] >= lo) & (panel["Date"] <= hi)]
    print(f"  panel overlap window: {len(panel):,} rows over {panel['Date'].nunique()} dates")

    led = None
    if not args.no_ledger:
        try:
            from importlib import import_module
            sys.path.insert(0, str(HERE))
            tl = import_module("__trial_ledger")
            led = tl.TrialLedger(tool="live_xcheck", seed=args.seed,
                                 params={"source": args.source, "n": args.n,
                                         "blocks": blocks})
        except Exception as exc:
            print(f"  [ledger off: {exc}]")

    rows = []
    for col in feat_cols:
        r = crosscheck_feature(panel, col, trades)
        if r.get("n", 0) >= 8:
            rows.append((col, family.get(col, "?"), r))
    rows.sort(key=lambda x: (abs(x[2].get("spread", 0.0)) if np.isfinite(x[2].get("spread", np.nan)) else -1), reverse=True)

    print("\n  " + "=" * 92)
    print(f"  {'feature':30s} {'family':22s} {'N':>5} {'liveIC':>7} {'hiRet':>7} {'loRet':>7} {'spread':>7} {'hiWin':>6}")
    print("  " + "-" * 92)
    for col, fam, r in rows:
        ic = r.get("ic", np.nan)
        sp = r.get("spread", np.nan)
        hr = r.get("hi_ret", np.nan)
        lr = r.get("lo_ret", np.nan)
        hw = r.get("hi_winrate", np.nan)
        sp_s = _c(f"{sp:>7.3f}", "1;32" if (np.isfinite(sp) and sp > 0.2) else "33" if np.isfinite(sp) else "38;5;240")
        print(f"  {col[:30]:30s} {fam[:22]:22s} {r['n']:>5} "
              f"{ic:>7.3f} {hr:>7.3f} {lr:>7.3f} {sp_s} {hw:>6.2f}"
              if np.isfinite(hr) else
              f"  {col[:30]:30s} {fam[:22]:22s} {r['n']:>5} {ic:>7.3f} {'--':>7} {'--':>7} {'--':>7} {'--':>6}")
        if led is not None:
            led.add_metrics(col, {"live_ic": ic, "live_spread": sp, "live_hi_ret": hr,
                                  "live_lo_ret": lr, "live_hi_winrate": hw},
                            family=fam, n_obs=r["n"], source=args.source,
                            base_win=r.get("base_win"))
    if led is not None:
        p = led.flush()
        print(f"\n  [ledger] {len(rows)} features logged -> {p}")
    print()


if __name__ == "__main__":
    main()
