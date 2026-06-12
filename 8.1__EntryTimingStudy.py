#!/usr/bin/env python3
"""
Entry/Exit Timing Study  (8.1__EntryTimingStudy.py)
===================================================
Uses the minute-accurate data in Data/IntradayTradeSim/ to ask, on the SAME set of
recorded trades, "when should I actually have entered/exited?" Everything here is an
internally-consistent comparison across one covered trade universe, so the RELATIVE
ranking of timing rules is robust even though absolute coverage is ~72%.

Sections
  A. Entry-time sweep        — fill at 09:30 open, 09:35, ... 14:00; exit at close.
  B. Gap-conditional entry   — bucket trades by overnight gap (open vs prior close);
                               does "gap-down => wait" beat entering at the open?
  C. Protective stop sweep   — entry @ 10:00, exit at close UNLESS intraday low breaches
                               a -X% stop first (full intraday path; full-coverage trades).
  D. Profit-target sweep     — entry @ 10:00, take profit if intraday high tags +Y% first.

CAVEAT: many variants are tested, so favour MONOTONE / consistent patterns over the
single best cell (which can be noise on ~800 trades). Exit is held at the close for A/B
to isolate the entry decision.

    python 8.1__EntryTimingStudy.py
"""

import os
import glob
import importlib.util
from datetime import time as dtime

import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Reuse the fill/commission helpers from the sim (filename starts with a digit -> importlib)
_spec = importlib.util.spec_from_file_location('fillsim', '8__IntradayFillSim.py')
sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sim)

INTRADAY_DIR = os.path.join('Data', 'IntradayTradeSim')
PRICE_DIR    = os.path.join('Data', 'PriceData')
OUT          = os.path.join('analysis_output', 'entry_timing_study.txt')

ENTRY_TIMES = [('open', 'open'), ('09:35', dtime(9, 35)), ('09:45', dtime(9, 45)),
               ('10:00', dtime(10, 0)), ('10:15', dtime(10, 15)), ('10:30', dtime(10, 30)),
               ('11:00', dtime(11, 0)), ('12:00', dtime(12, 0)), ('14:00', dtime(14, 0))]


# ── prior-close cache for the overnight gap ──────────────────────────────────────
_price_cache: dict[str, pd.Series] = {}

def prior_close(symbol: str, entry_date) -> float | None:
    if symbol not in _price_cache:
        fp = os.path.join(PRICE_DIR, f"{symbol}.parquet")
        if not os.path.exists(fp):
            _price_cache[symbol] = None
        else:
            df = pd.read_parquet(fp, columns=['Date', 'Close'])
            df['Date'] = pd.to_datetime(df['Date'])
            _price_cache[symbol] = df.set_index('Date')['Close'].sort_index()
    s = _price_cache[symbol]
    if s is None:
        return None
    before = s[s.index < pd.Timestamp(entry_date)]
    return float(before.iloc[-1]) if len(before) else None


def entry_fill(entry_bars, spec):
    return sim.fill_open(entry_bars) if spec == 'open' else sim.fill_at_time(entry_bars, spec)


def net_pnl(entry, exit_, notional):
    """Same-dollar sizing, IBKR commission both sides."""
    if not (entry and exit_) or entry <= 0:
        return None
    shares = int(notional // entry)
    if shares <= 0:
        return None
    gross = (exit_ - entry) * shares
    comm = sim.commission(shares, entry) + sim.commission(shares, exit_)
    return gross - comm


def hold_path(store, sym, entry_dt, exit_dt, after_time):
    """All RTH bars from entry day (>= after_time) through exit day, in order. None if a gap."""
    days = pd.bdate_range(entry_dt.normalize(), exit_dt.normalize())
    parts = []
    for d in days:
        b = store.day_bars(sym, d.date())
        if b is None or b.empty:
            return None  # incomplete path
        if d.date() == entry_dt.date():
            b = b[b['Date'].dt.time >= after_time]
        parts.append(b)
    return pd.concat(parts).sort_values('Date') if parts else None


def main():
    store = sim.IntradayStore(INTRADAY_DIR)
    th = pd.read_parquet('trade_history.parquet')
    th['EntryDate'] = pd.to_datetime(th['EntryDate'])
    th['ExitDate'] = pd.to_datetime(th['ExitDate'])
    th = th[th['TradeType'] == 'Long'] if 'TradeType' in th.columns else th

    recs = []
    for r in th.itertuples(index=False):
        e_bars = store.day_bars(r.Symbol, r.EntryDate.date())
        x_bars = store.day_bars(r.Symbol, r.ExitDate.date())
        if e_bars is None or e_bars.empty or x_bars is None or x_bars.empty:
            continue
        notional = float(r.EntryPrice) * int(r.Quantity)
        exit_close = sim.fill_close(x_bars)
        op = sim.fill_open(e_bars)
        pc = prior_close(r.Symbol, r.EntryDate)
        gap = (op / pc - 1.0) * 100 if (pc and op) else np.nan

        rec = {'Symbol': r.Symbol, 'EntryDate': r.EntryDate, 'ExitDate': r.ExitDate,
               'notional': notional, 'gap': gap, 'exit_close': exit_close}
        for label, spec in ENTRY_TIMES:
            ef = entry_fill(e_bars, spec)
            rec[f'net_{label}'] = net_pnl(ef, exit_close, notional)
            rec[f'fill_{label}'] = ef
        recs.append(rec)

    df = pd.DataFrame(recs)
    n = len(df)
    L = []
    def p(s=''): L.append(s); print(s)

    p("=" * 76)
    p("  ENTRY / EXIT TIMING STUDY  (minute-accurate, exit @ close unless noted)")
    p("=" * 76)
    p(f"  Covered trades: {n}   |   gap available on {df['gap'].notna().sum()}")
    p("  Internally-consistent: every rule is scored on the SAME trades.")

    # ── A. Entry-time sweep ──
    p("")
    p("  [A] ENTRY-TIME SWEEP  (exit @ close, same-dollar sizing, fees both sides)")
    p(f"      {'Entry':<8}{'EV/trade $':>12}{'Win %':>9}{'Total $':>12}{'Median $':>11}")
    p("      " + "-" * 50)
    a_rows = []
    for label, _ in ENTRY_TIMES:
        s = df[f'net_{label}'].dropna()
        ev, wr, tot, med = s.mean(), (s > 0).mean() * 100, s.sum(), s.median()
        a_rows.append((label, ev, wr, tot))
        p(f"      {label:<8}{ev:>12.2f}{wr:>9.2f}{tot:>12.0f}{med:>11.2f}")
    best = max(a_rows, key=lambda x: x[1])
    p(f"      -> best EV: entry @ {best[0]}  (${best[1]:.2f}/trade, total ${best[3]:,.0f})")

    # ── B. Gap-conditional entry ──
    p("")
    p("  [B] GAP-CONDITIONAL ENTRY  (overnight gap = open vs prior close)")
    p("      EV $/trade by gap bucket; does waiting beat the open?")
    g = df[df['gap'].notna()].copy()
    bins = [-np.inf, -2, -0.5, 0.5, 2, np.inf]
    names = ['gap<=-2%', '-2..-0.5%', 'flat +-0.5%', '0.5..2%', 'gap>=+2%']
    g['bucket'] = pd.cut(g['gap'], bins=bins, labels=names)
    cmp_times = ['open', '09:45', '10:00', '10:30']
    p(f"      {'Bucket':<14}{'n':>5}" + "".join(f"{t:>10}" for t in cmp_times) + f"{'best':>9}")
    p("      " + "-" * (19 + 10 * len(cmp_times) + 9))
    for nm in names:
        sub = g[g['bucket'] == nm]
        if len(sub) == 0:
            continue
        evs = {t: sub[f'net_{t}'].dropna().mean() for t in cmp_times}
        bestt = max(evs, key=evs.get)
        p(f"      {nm:<14}{len(sub):>5}" + "".join(f"{evs[t]:>10.2f}" for t in cmp_times) + f"{bestt:>9}")
    p("      (read across each row: which entry time has the highest EV for that gap)")

    # ── E. Combined 'smart gap' rule vs flat baselines ──
    p("")
    p("  [E] COMBINED GAP RULE  (skip dud cohorts, pick entry time per gap)")
    p("      Rule: gap<=-2% -> open ;  -2..-0.5% -> SKIP ;  flat -> open ;")
    p("            +0.5..2% -> open ;  gap>=+2% -> 10:30")
    chosen = {'gap<=-2%': 'open', '-2..-0.5%': None, 'flat +-0.5%': 'open',
              '0.5..2%': 'open', 'gap>=+2%': '10:30'}
    smart_nets, kept, skipped = [], 0, 0
    for _, row in g.iterrows():
        rule = chosen.get(str(row['bucket']))
        if rule is None:
            skipped += 1; continue
        v = row[f'net_{rule}']
        if pd.notna(v):
            smart_nets.append(v); kept += 1
    smart = pd.Series(smart_nets)
    all_open = g['net_open'].dropna()
    all_1000 = g['net_10:00'].dropna()
    p(f"      {'Strategy':<22}{'Trades':>8}{'EV/trade $':>12}{'Total $':>12}{'Win %':>9}")
    p("      " + "-" * 63)
    p(f"      {'All @ open':<22}{len(all_open):>8}{all_open.mean():>12.2f}{all_open.sum():>12.0f}{(all_open>0).mean()*100:>9.2f}")
    p(f"      {'All @ 10:00':<22}{len(all_1000):>8}{all_1000.mean():>12.2f}{all_1000.sum():>12.0f}{(all_1000>0).mean()*100:>9.2f}")
    p(f"      {'Smart gap rule':<22}{kept:>8}{smart.mean():>12.2f}{smart.sum():>12.0f}{(smart>0).mean()*100:>9.2f}")
    p(f"      (smart rule skipped {skipped} dud-cohort trades; deploys capital on the rest)")

    # ── Build full intraday paths once (entry @ 10:00) for C/D ──
    paths = {}
    for i in df.index:
        path = hold_path(store, df.at[i, 'Symbol'], df.at[i, 'EntryDate'], df.at[i, 'ExitDate'], dtime(10, 0))
        if path is not None and len(path):
            paths[i] = path
    base = {}
    for i, path in paths.items():
        base[i] = net_pnl(float(path.iloc[0]['Open']), df.at[i, 'exit_close'], df.at[i, 'notional'])
    base = pd.Series(base).dropna()
    n_full = len(base)

    # ── C. Protective stop sweep (entry @ 10:00, full intraday path) ──
    p("")
    p("  [C] PROTECTIVE STOP  (entry @ 10:00, exit @ close unless intraday low hits -X%)")
    p(f"      Full-path trades available: {n_full} of {n}  (multi-day-complete subset)")
    p(f"      {'Stop':<10}{'EV/trade $':>12}{'Win %':>9}{'Total $':>12}{'Stopped %':>11}")
    p("      " + "-" * 52)
    p(f"      {'none':<10}{base.mean():>12.2f}{(base>0).mean()*100:>9.2f}{base.sum():>12.0f}{0.0:>11.2f}")
    for stop in [0.5, 1.0, 2.0, 3.0, 5.0]:
        nets, stopped = {}, 0
        for i, path in paths.items():
            entry = float(path.iloc[0]['Open'])
            stop_px = entry * (1 - stop / 100)
            hit = bool((path['Low'] <= stop_px).any())
            exitpx = stop_px if hit else df.at[i, 'exit_close']
            stopped += hit
            nets[i] = net_pnl(entry, exitpx, df.at[i, 'notional'])
        s = pd.Series(nets).dropna()
        p(f"      {str(stop)+'%':<10}{s.mean():>12.2f}{(s>0).mean()*100:>9.2f}{s.sum():>12.0f}{stopped/n_full*100:>11.2f}")

    # ── D. Profit-target sweep (entry @ 10:00, full intraday path) ──
    p("")
    p("  [D] PROFIT TARGET  (entry @ 10:00, take profit if intraday high tags +Y%)")
    p(f"      {'Target':<10}{'EV/trade $':>12}{'Win %':>9}{'Total $':>12}{'Hit %':>11}")
    p("      " + "-" * 52)
    p(f"      {'none':<10}{base.mean():>12.2f}{(base>0).mean()*100:>9.2f}{base.sum():>12.0f}{0.0:>11.2f}")
    for tgt in [1.0, 2.0, 3.0, 5.0]:
        nets, hit_n = {}, 0
        for i, path in paths.items():
            entry = float(path.iloc[0]['Open'])
            tgt_px = entry * (1 + tgt / 100)
            hit = bool((path['High'] >= tgt_px).any())
            exitpx = tgt_px if hit else df.at[i, 'exit_close']
            hit_n += hit
            nets[i] = net_pnl(entry, exitpx, df.at[i, 'notional'])
        s = pd.Series(nets).dropna()
        p(f"      {str(tgt)+'%':<10}{s.mean():>12.2f}{(s>0).mean()*100:>9.2f}{s.sum():>12.0f}{hit_n/n_full*100:>11.2f}")

    p("")
    p("  NOTE: stop/target assume a fill exactly at the level (optimistic on gaps through it).")
    p("  Treat C/D as directional signal, not a promise. Favour patterns that are monotone.")
    p("=" * 76)

    os.makedirs('analysis_output', exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write("\n".join(L) + "\n")
    print(f"\n  Saved -> {OUT}")


if __name__ == '__main__':
    main()
