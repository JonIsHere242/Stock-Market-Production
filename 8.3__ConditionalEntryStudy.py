#!/usr/bin/env python3
"""Conditional-Entry Study (8.3__ConditionalEntryStudy.py)
=========================================================
Realistic retail entry on the minute data: you CANNOT get the 09:30 open on an
illiquid no-name ticker. So the honest baseline is "wait a few minutes, market in".
This script asks: can FIRST-BARS CONDITIONS (e.g. "skip the trade if it drops over
the first K minutes", "only enter if it's green / above VWAP") beat the naive timed
entry — i.e. use the opening minutes as a FILTER to drop failing signals?

For each trade: entry per rule on the entry day (or SKIP), exit held at the recorded
exit-day close (isolates the entry decision). Reuses 8__IntradayFillSim helpers.
Reads the NEW ensemble trades (Data/TradeHistory.parquet). Lookahead-safe: a rule
decides on bars 0..k-1 and fills at bar k's open.

  python 8.3__ConditionalEntryStudy.py
"""
import os, glob, importlib.util
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location('fillsim', '8__IntradayFillSim.py')
sim = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(sim)

TRADE_HISTORY = 'Data/TradeHistory.parquet'
INTRADAY_DIR  = os.path.join('Data', 'IntradayTradeSim')
NOTIONAL      = 10000.0            # equal-weight per trade (commission realism)
OUT           = os.path.join('analysis_output', 'conditional_entry_study.txt')


def _open0(b):  return float(b.iloc[0].Open)


# Each rule: (entry_bars_sorted) -> entry price, or None to SKIP the trade.
def r_open(b):  return _open0(b)
def r_wait(n):
    def f(b): return float(b.iloc[n].Open) if len(b) > n else None
    return f
def r_skip_red(k):
    """Drop the trade if it fell over the first k minutes; else market in at bar k."""
    def f(b):
        if len(b) <= k: return None
        return None if float(b.iloc[k-1].Close) < _open0(b) else float(b.iloc[k].Open)
    return f
def r_first_green(m):
    """Enter at the first bar (within m min) that closes above the open; else skip."""
    def f(b):
        o0 = _open0(b)
        for k in range(1, min(m, len(b) - 1) + 1):
            if float(b.iloc[k].Close) > o0:
                return float(b.iloc[k + 1].Open) if len(b) > k + 1 else float(b.iloc[k].Close)
        return None
    return f
def r_above_vwap(n):
    """At minute n, enter only if price is above the session VWAP; else skip."""
    def f(b):
        if len(b) <= n + 1: return None
        return float(b.iloc[n + 1].Open) if float(b.iloc[n].Close) > float(b.iloc[n].VWAP) else None
    return f
def r_skip_volatile(n, thr):
    """Skip if the first n-min range is > thr (illiquid open spike); else enter at bar n."""
    def f(b):
        if len(b) <= n: return None
        o0 = _open0(b)
        rng = (float(b.iloc[:n].High.max()) - float(b.iloc[:n].Low.min())) / o0
        return None if rng > thr else float(b.iloc[n].Open)
    return f

RULES = [
    ("open (ideal, can't get)", r_open),
    ("wait_1min",  r_wait(1)),
    ("wait_3min",  r_wait(3)),
    ("wait_5min  [BASELINE]", r_wait(5)),
    ("wait_10min", r_wait(10)),
    ("wait_15min", r_wait(15)),
    ("wait_30min", r_wait(30)),
    ("skip_red_3bar  @bar3",  r_skip_red(3)),
    ("skip_red_5bar  @bar5",  r_skip_red(5)),
    ("skip_red_10bar @bar10", r_skip_red(10)),
    ("first_green_15min",     r_first_green(15)),
    ("above_vwap_5min",       r_above_vwap(5)),
    ("above_vwap_10min",      r_above_vwap(10)),
    ("skip_volatile_5min>4%", r_skip_volatile(5, 0.04)),
]


def commission(shares, price):
    try: return sim.commission(int(shares), float(price))
    except Exception: return max(0.35, shares * 0.0035)


def trade_ret(entry, exit_):
    """Net return on NOTIONAL for one trade (equal-weight, with commissions)."""
    if not entry or entry <= 0 or not exit_: return None
    shares = int(NOTIONAL // entry)
    if shares <= 0: return None
    gross = (exit_ - entry) * shares
    net = gross - commission(shares, entry) - commission(shares, exit_)
    return net / NOTIONAL


def main():
    store = sim.IntradayStore(INTRADAY_DIR)
    th = pd.read_parquet(TRADE_HISTORY)
    th['EntryDate'] = pd.to_datetime(th['EntryDate']); th['ExitDate'] = pd.to_datetime(th['ExitDate'])
    if 'TradeType' in th.columns:
        th = th[th['TradeType'] == 'Long']
    th = th.sort_values('EntryDate').reset_index(drop=True)

    # Precompute per-trade entry bars + exit close (covered universe only)
    trades = []
    for r in th.itertuples(index=False):
        eb = store.day_bars(r.Symbol, r.EntryDate.date())
        xb = store.day_bars(r.Symbol, r.ExitDate.date())
        if eb is None or eb.empty or xb is None or xb.empty:
            continue
        eb = eb.sort_values('Date').reset_index(drop=True)
        trades.append((eb, sim.fill_close(xb), r.EntryDate))
    n_cov = len(trades)

    rows = []
    for name, rule in RULES:
        rets, taken = [], 0
        for eb, exit_close, _ in trades:
            try: entry = rule(eb)
            except Exception: entry = None
            if entry is None: continue
            ret = trade_ret(entry, exit_close)
            if ret is None: continue
            rets.append(ret); taken += 1
        if not rets:
            rows.append((name, 0, 0, np.nan, np.nan, np.nan, np.nan, np.nan)); continue
        rets = np.array(rets)
        wins = rets[rets > 0]; losses = rets[rets <= 0]
        pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf
        compounded = (np.prod(1 + rets) - 1) * 100
        sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else np.nan
        rows.append((name, taken, taken / n_cov * 100, rets.mean() * 100,
                     (rets > 0).mean() * 100, pf, compounded, sharpe))

    # baseline = wait_5min
    base = next(r for r in rows if 'BASELINE' in r[0])
    L = ["=" * 96,
         "CONDITIONAL-ENTRY STUDY  -  realistic retail fills on the new ensemble trades",
         "=" * 96,
         f"Covered trades: {n_cov} (exit held at recorded close; entry per rule; equal-weight ${NOTIONAL:,.0f})",
         f"BASELINE = wait_5min (market in at 09:35).  '>' marks rules that beat baseline compounded.",
         "",
         f"  {'rule':<26}{'taken':>6}{'cov%':>6}{'EV/trade%':>10}{'win%':>7}{'PF':>6}{'compound%':>11}{'Sharpe':>8}  vs base"]
    for name, taken, cov, ev, win, pf, cmp_, sh in rows:
        flag = ""
        if 'BASELINE' not in name and not np.isnan(cmp_):
            flag = " >BEATS" if cmp_ > base[6] else ""
        L.append(f"  {name:<26}{taken:>6}{cov:>6.0f}{ev:>10.3f}{win:>7.1f}{pf:>6.2f}{cmp_:>11.1f}{sh:>8.2f}{flag}")
    L += ["",
          "READ: a good conditional raises EV/trade AND compounded WITHOUT cutting coverage too",
          "hard. 'skip_red_K' is the key test of your idea (drop trades that fall after open).",
          "If a filter beats wait_5min on EV and compounded with reasonable coverage, it's a real",
          "edge improvement within realistic retail fills. Favour consistent patterns over one cell.",
          "=" * 96]
    os.makedirs('analysis_output', exist_ok=True)
    open(OUT, 'w').write("\n".join(L))
    print("\n".join(L))
    print(f"\nsaved -> {OUT}")


if __name__ == '__main__':
    main()
