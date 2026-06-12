#!/usr/bin/env python3
"""Filter Stacking & Independence (8.4__FilterStacking.py)
=========================================================
Follow-up to 8.3. Two questions:
  1) Sweep the VWAP gate across minutes {2,3,5,7,10,15,30}.
  2) Are the bar-momentum filter (red) and the VWAP filter INDEPENDENT or REDUNDANT,
     and does STACKING them (take a trade only if BOTH pass) add edge or just cut coverage?

Unified convention: a gate "at minute t" observes the bar at minute t (its Close) and
fills at the NEXT bar's open (t+1) — lookahead-safe, realistic market entry.
  red_t  : take if Close[t] >= Open[0]   (hasn't dropped below open by minute t)
  vwap_t : take if Close[t] >  VWAP[t]    (trading above the session VWAP)
Stacked gates decide at max(t) and fill at max(t)+1. Exit held at recorded close.
Reads Data/TradeHistory.parquet; reuses 8__IntradayFillSim helpers.

  python 8.4__FilterStacking.py
"""
import os, importlib.util
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location('fillsim', '8__IntradayFillSim.py')
sim = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(sim)

TRADE_HISTORY = 'Data/TradeHistory.parquet'
INTRADAY_DIR  = os.path.join('Data', 'IntradayTradeSim')
NOTIONAL      = 10000.0
OUT           = os.path.join('analysis_output', 'filter_stacking_study.txt')


def commission(sh, px):
    try: return sim.commission(int(sh), float(px))
    except Exception: return max(0.35, sh * 0.0035)


def trade_ret(entry, exit_):
    if not entry or entry <= 0 or not exit_: return None
    sh = int(NOTIONAL // entry)
    if sh <= 0: return None
    return ((exit_ - entry) * sh - commission(sh, entry) - commission(sh, exit_)) / NOTIONAL


def passes(eb, conds):
    """conds = list of ('red'|'vwap', t). Returns (take_bool, entry_price_or_None).
    Decide at max t, fill at the next bar's open. None entry if any cond fails / short day."""
    tmax = max(t for _, t in conds)
    if len(eb) <= tmax + 1:
        return False, None
    o0 = float(eb.iloc[0].Open)
    for kind, t in conds:
        if kind == 'red'  and not (float(eb.iloc[t].Close) >= o0):                  return False, None
        if kind == 'vwap' and not (float(eb.iloc[t].Close) >  float(eb.iloc[t].VWAP)): return False, None
    return True, float(eb.iloc[tmax + 1].Open)


def stats(trades, conds):
    """Return dict of metrics for a gate (conds) over the covered trades."""
    rets, mask = [], []
    for eb, xc in trades:
        take, entry = passes(eb, conds)
        mask.append(take)
        if take:
            r = trade_ret(entry, xc)
            if r is not None:
                rets.append(r)
    rets = np.array(rets)
    if len(rets) == 0:
        return dict(n=0, cov=0, ev=np.nan, win=np.nan, pf=np.nan, sh=np.nan, mask=np.array(mask))
    w, l = rets[rets > 0], rets[rets <= 0]
    return dict(
        n=len(rets), cov=len(rets) / len(trades) * 100,
        ev=rets.mean() * 100, win=(rets > 0).mean() * 100,
        pf=(w.sum() / abs(l.sum())) if l.sum() != 0 else np.inf,
        sh=(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else np.nan,
        mask=np.array(mask))


def main():
    store = sim.IntradayStore(INTRADAY_DIR)
    th = pd.read_parquet(TRADE_HISTORY)
    th['EntryDate'] = pd.to_datetime(th['EntryDate']); th['ExitDate'] = pd.to_datetime(th['ExitDate'])
    if 'TradeType' in th.columns: th = th[th['TradeType'] == 'Long']
    th = th.sort_values('EntryDate').reset_index(drop=True)
    trades = []
    for r in th.itertuples(index=False):
        eb = store.day_bars(r.Symbol, r.EntryDate.date()); xb = store.day_bars(r.Symbol, r.ExitDate.date())
        if eb is None or eb.empty or xb is None or xb.empty: continue
        trades.append((eb.sort_values('Date').reset_index(drop=True), sim.fill_close(xb)))
    N = len(trades)

    TS = [2, 3, 5, 7, 10, 15, 30]
    L = ["=" * 98, "FILTER STACKING & INDEPENDENCE  -  new ensemble trades", "=" * 98,
         f"Covered trades: {N}.  Gate 'at minute t' -> decide on bar t, fill bar t+1 open.", ""]

    def row(label, s):
        return (f"  {label:<22}{s['n']:>5}{s['cov']:>6.0f}{s['ev']:>10.3f}"
                f"{s['win']:>7.1f}{s['pf']:>6.2f}{s['sh']:>8.2f}")
    hdr = f"  {'gate':<22}{'n':>5}{'cov%':>6}{'EV%':>10}{'win%':>7}{'PF':>6}{'Sharpe':>8}"

    base = stats(trades, [('red', 4)])  # ~wait_5 with a no-op-ish gate? use a pure-timing baseline:
    # pure timing baseline = take ALL, fill at minute 5 -> emulate with an always-true gate:
    base_rets = []
    for eb, xc in trades:
        if len(eb) > 6:
            r = trade_ret(float(eb.iloc[6].Open), xc)   # fill ~minute 6 (next bar after t=5)
            if r is not None: base_rets.append(r)
    br = np.array(base_rets)
    base_ev = br.mean() * 100; base_cmp = base_ev
    L += ["--- BASELINE: take ALL, market in ~minute 5 ---",
          f"  EV/trade={base_ev:.3f}%  win={(br>0).mean()*100:.1f}%  n={len(br)}", ""]

    # 1) VWAP sweep
    L += ["--- 1) VWAP gate sweep (above session VWAP at minute t) ---", hdr]
    vwap_masks = {}
    for t in TS:
        s = stats(trades, [('vwap', t)]); vwap_masks[t] = s['mask']
        L.append(row(f"vwap_{t}min", s) + ("  >EV" if s['ev'] > base_ev else ""))
    # 2) red sweep
    L += ["", "--- 2) red gate sweep (not below open at minute t) ---", hdr]
    red_masks = {}
    for t in TS:
        s = stats(trades, [('red', t)]); red_masks[t] = s['mask']
        L.append(row(f"red_{t}min", s) + ("  >EV" if s['ev'] > base_ev else ""))

    # 3) Independence — pick the strongest red & vwap, analyse overlap
    red_best = max(TS, key=lambda t: stats(trades, [('red', t)])['ev'])
    vwp_best = max(TS, key=lambda t: stats(trades, [('vwap', t)])['ev'])
    A = red_masks[red_best].astype(int); B = vwap_masks[vwp_best].astype(int)
    pA, pB = A.mean(), B.mean()
    pAB = (A & B).mean()
    phi = np.corrcoef(A, B)[0, 1] if A.std() > 0 and B.std() > 0 else np.nan
    L += ["", "=" * 98,
          f"3) INDEPENDENCE: red_{red_best}min  vs  vwap_{vwp_best}min  (do they catch the SAME trades?)",
          "=" * 98,
          f"  P(red passes)            = {pA:.3f}",
          f"  P(vwap passes)           = {pB:.3f}",
          f"  P(both pass)             = {pAB:.3f}",
          f"  If INDEPENDENT, P(both)  = {pA*pB:.3f}   (P(red)*P(vwap))",
          f"  ratio P(both)/indep      = {pAB/(pA*pB) if pA*pB>0 else float('nan'):.2f}   "
          f"(~1 = independent, >1 = redundant/catch same, <1 = complementary)",
          f"  correlation (phi)        = {phi:+.2f}"]

    # 4) Stacking — red_best AND vwap_t across t, vs each alone
    L += ["", "--- 4) STACK: red_%dmin AND vwap_t  (take only if BOTH pass) ---" % red_best, hdr]
    sa = stats(trades, [('red', red_best)])
    L.append(row(f"red_{red_best} alone", sa))
    for t in TS:
        sb = stats(trades, [('vwap', t)])
        ss = stats(trades, [('red', red_best), ('vwap', t)])
        lift = ss['ev'] - max(sa['ev'], sb['ev'])
        L.append(row(f"red{red_best}+vwap_{t}", ss) +
                 (f"  +EV{lift:+.2f} vs best-alone" if not np.isnan(lift) else ""))
    L += ["",
          "READ THE STACK: if 'red+vwap' EV > BOTH components' EV, the filters are COMPLEMENTARY",
          "(catch different bad trades) -> stacking adds real edge. If EV ~ the better component but",
          "coverage drops, they're REDUNDANT -> stacking just starves the book. The independence",
          "ratio above predicts which: ~1 independent (stack helps), >>1 redundant (stack wastes).",
          "Caveat: ~240 trades, many cells -> trust consistent patterns, confirm on a 2nd trade set.",
          "=" * 98]
    os.makedirs('analysis_output', exist_ok=True)
    open(OUT, 'w').write("\n".join(L))
    print("\n".join(L))
    print(f"\nsaved -> {OUT}")


if __name__ == '__main__':
    main()
