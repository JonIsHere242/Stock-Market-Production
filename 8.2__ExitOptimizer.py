#!/usr/bin/env python3
"""
Exit Optimizer  (8.2__ExitOptimizer.py)
=======================================
Finds hard-stop / take-profit / trailing-stop levels for the recorded trades, using the
minute-accurate intraday path of each trade (Data/IntradayTradeSim/). Entry is fixed at
the live time (default 10:00 ET); the original trade ExitDate close is the time-stop
backstop. We only add protective/profit exits INSIDE that window — same trades, same
max hold.

Intrabar model (conservative): within a 1-min bar we assume the ADVERSE level (hard stop
or trailing stop) fills before a favorable take-profit. Gap-throughs fill at the bar open
(worse for stops, better for TP) rather than the level, so we don't flatter the result.
Trailing references the high through the PRIOR bar (no intrabar lookahead).

Guardrails against curve-fitting:
  - Reports the no-exit baseline and each lever's MARGINAL effect first.
  - Then the top combos, but with an OUT-OF-SAMPLE split: the best combo is picked on the
    earlier 60% of trades (by entry date) and re-scored on the later 40%. If it doesn't
    hold up OOS, don't trust it.

    python 8.2__ExitOptimizer.py
    python 8.2__ExitOptimizer.py --entry-time open
"""

import os
import argparse
import importlib.util
from datetime import time as dtime

import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

_spec = importlib.util.spec_from_file_location('fillsim', '8__IntradayFillSim.py')
sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sim)

INTRADAY_DIR = os.path.join('Data', 'IntradayTradeSim')
OUT = os.path.join('analysis_output', 'exit_optimizer.txt')

SL_GRID = [None, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]  # hard stop % (0.5 = current live)
TP_GRID = [None, 3.0, 5.0, 8.0, 12.0, 20.0]         # take profit %
TS_GRID = [None, 3.0, 5.0, 8.0, 12.0]               # trailing stop %


def build_paths(store, entry_time):
    """Per-trade full intraday path (entry-bar -> exit-day close). Returns list of dicts."""
    th = pd.read_parquet('trade_history.parquet')
    th['EntryDate'] = pd.to_datetime(th['EntryDate'])
    th['ExitDate'] = pd.to_datetime(th['ExitDate'])
    if 'TradeType' in th.columns:
        th = th[th['TradeType'] == 'Long']

    out = []
    for r in th.itertuples(index=False):
        days = pd.bdate_range(r.EntryDate.normalize(), r.ExitDate.normalize())
        parts, ok = [], True
        for d in days:
            b = store.day_bars(r.Symbol, d.date())
            if b is None or b.empty:
                ok = False
                break
            if d.date() == r.EntryDate.date():
                if entry_time == 'open':
                    pass
                else:
                    b = b[b['Date'].dt.time >= entry_time]
                    if b.empty:
                        ok = False
                        break
            parts.append(b[['Open', 'High', 'Low', 'Close']])
        if not ok or not parts:
            continue
        path = pd.concat(parts)
        O = path['Open'].to_numpy(float)
        H = path['High'].to_numpy(float)
        L = path['Low'].to_numpy(float)
        C = path['Close'].to_numpy(float)
        e = float(O[0])
        if e <= 0 or len(O) < 1:
            continue
        notional = float(r.EntryPrice) * int(r.Quantity)
        rm_prior = np.maximum.accumulate(np.concatenate([[e], H[:-1]]))  # high through prior bar
        out.append({'sym': r.Symbol, 'entry_date': r.EntryDate, 'e': e, 'notional': notional,
                    'O': O, 'H': H, 'L': L, 'C': C, 'rm_prior': rm_prior})
    return out


def exit_net(t, sl, tp, ts, act=0.0):
    """
    Net P&L for one trade under hard stop %, take profit %, trailing %, and an optional
    trailing ARM threshold act% (trail only engages once the high is >= entry*(1+act%)).
    Returns (net, reason in {hardstop, trail, tp, close}).
    """
    e, O, H, L, C = t['e'], t['O'], t['H'], t['L'], t['C']
    stop_const = e * (1 - sl / 100) if sl else -np.inf
    tp_const = e * (1 + tp / 100) if tp else np.inf
    if ts:
        arm = e * (1 + act / 100) if act else e
        armed = t['rm_prior'] >= arm                       # trail engages after +act%
        trail = np.where(armed, t['rm_prior'] * (1 - ts / 100), -np.inf)
    else:
        trail = np.full(len(O), -np.inf)
    protect = np.maximum(stop_const, trail)

    adv = np.where(L <= protect)[0]
    fav = np.where(H >= tp_const)[0]
    i_adv = adv[0] if len(adv) else len(O)
    i_fav = fav[0] if len(fav) else len(O)

    if i_adv == len(O) and i_fav == len(O):
        fill, reason = C[-1], 'close'
    elif i_adv <= i_fav:
        lvl = protect[i_adv]
        fill = O[i_adv] if O[i_adv] < lvl else lvl   # gap-through fills worse
        reason = 'trail' if (ts and trail[i_adv] >= stop_const and trail[i_adv] > -np.inf) else 'hardstop'
    else:
        lvl = tp_const
        fill = O[i_fav] if O[i_fav] > lvl else lvl    # gap-up fills better
        reason = 'tp'

    shares = int(t['notional'] // e)
    if shares <= 0:
        return None, reason
    gross = (fill - e) * shares
    comm = sim.commission(shares, e) + sim.commission(shares, fill)
    return gross - comm, reason


def evaluate(paths, sl, tp, ts, act=0.0):
    nets, reasons = [], []
    for t in paths:
        net, reason = exit_net(t, sl, tp, ts, act)
        if net is not None:
            nets.append(net)
            reasons.append(reason)
    s = pd.Series(nets)
    rc = pd.Series(reasons).value_counts(normalize=True) * 100
    return {'ev': s.mean(), 'total': s.sum(), 'win': (s > 0).mean() * 100, 'n': len(s),
            'pct_hardstop': rc.get('hardstop', 0.0), 'pct_trail': rc.get('trail', 0.0),
            'pct_tp': rc.get('tp', 0.0), 'pct_close': rc.get('close', 0.0)}


def lab(v):
    return 'none' if v is None else f"{v:g}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--entry-time', default='10:00', help="HH:MM ET or 'open' (default 10:00)")
    args = ap.parse_args()
    if args.entry_time == 'open':
        et = 'open'
    else:
        hh, mm = (int(x) for x in args.entry_time.split(':'))
        et = dtime(hh, mm)

    store = sim.IntradayStore(INTRADAY_DIR)
    paths = build_paths(store, et)
    n = len(paths)
    L = []
    def p(s=''): L.append(s); print(s)

    p("=" * 78)
    p(f"  EXIT OPTIMIZER  (entry @ {args.entry_time}, time-stop = original ExitDate close)")
    p("=" * 78)
    p(f"  Full-path trades: {n}")
    if n == 0:
        p("  No complete intraday paths found — run the downloader / --fill-holes first.")
        return

    base = evaluate(paths, None, None, None)
    p(f"  Baseline (no stop / no TP / no trail): EV ${base['ev']:.2f}/trade, "
      f"total ${base['total']:,.0f}, win {base['win']:.1f}%")

    # ── Marginal: hard stop only ──
    p("")
    p("  [A] HARD STOP only (no TP, no trail)")
    p(f"      {'Stop':<8}{'EV $':>10}{'Win %':>9}{'Total $':>11}{'% stopped':>11}")
    p("      " + "-" * 49)
    for sl in SL_GRID:
        r = evaluate(paths, sl, None, None)
        p(f"      {lab(sl):<8}{r['ev']:>10.2f}{r['win']:>9.1f}{r['total']:>11.0f}{r['pct_hardstop']:>11.1f}")

    # ── Marginal: take profit only ──
    p("")
    p("  [B] TAKE PROFIT only (no stop, no trail)")
    p(f"      {'TP':<8}{'EV $':>10}{'Win %':>9}{'Total $':>11}{'% hit TP':>11}")
    p("      " + "-" * 49)
    for tp in TP_GRID:
        r = evaluate(paths, None, tp, None)
        p(f"      {lab(tp):<8}{r['ev']:>10.2f}{r['win']:>9.1f}{r['total']:>11.0f}{r['pct_tp']:>11.1f}")

    # ── Marginal: trailing stop only ──
    p("")
    p("  [C] TRAILING STOP only (no hard stop, no TP)")
    p(f"      {'Trail':<8}{'EV $':>10}{'Win %':>9}{'Total $':>11}{'% trailed':>11}")
    p("      " + "-" * 49)
    for ts in TS_GRID:
        r = evaluate(paths, None, None, ts)
        p(f"      {lab(ts):<8}{r['ev']:>10.2f}{r['win']:>9.1f}{r['total']:>11.0f}{r['pct_trail']:>11.1f}")

    # ── Full grid ──
    rows = []
    for sl in SL_GRID:
        for tp in TP_GRID:
            for ts in TS_GRID:
                r = evaluate(paths, sl, tp, ts)
                rows.append({'sl': sl, 'tp': tp, 'ts': ts, **r})
    grid = pd.DataFrame(rows)

    p("")
    p("  [D] TOP 12 COMBOS by EV/trade  (caveat: in-sample maxima are noisy)")
    p(f"      {'Stop':<7}{'TP':<7}{'Trail':<7}{'EV $':>9}{'Win %':>8}{'Total $':>11}{'%stop':>8}{'%tp':>7}{'%cls':>7}")
    p("      " + "-" * 71)
    for _, r in grid.sort_values('ev', ascending=False).head(12).iterrows():
        p(f"      {lab(r['sl']):<7}{lab(r['tp']):<7}{lab(r['ts']):<7}{r['ev']:>9.2f}{r['win']:>8.1f}"
          f"{r['total']:>11.0f}{(r['pct_hardstop']+r['pct_trail']):>8.1f}{r['pct_tp']:>7.1f}{r['pct_close']:>7.1f}")

    # ── Out-of-sample split ──
    p("")
    p("  [E] OUT-OF-SAMPLE CHECK  (pick best on earlier 60%, score on later 40%)")
    paths_sorted = sorted(paths, key=lambda t: t['entry_date'])
    cut = int(len(paths_sorted) * 0.6)
    train, test = paths_sorted[:cut], paths_sorted[cut:]
    p(f"      Train n={len(train)} (<= {train[-1]['entry_date'].date()}),  Test n={len(test)}")

    tr_rows = []
    for sl in SL_GRID:
        for tp in TP_GRID:
            for ts in TS_GRID:
                r = evaluate(train, sl, tp, ts)
                tr_rows.append({'sl': sl, 'tp': tp, 'ts': ts, 'ev': r['ev']})
    tr = pd.DataFrame(tr_rows).sort_values('ev', ascending=False)
    best = tr.iloc[0]
    base_tr = evaluate(train, None, None, None)
    base_te = evaluate(test, None, None, None)
    best_tr = evaluate(train, best['sl'], best['tp'], best['ts'])
    best_te = evaluate(test, best['sl'], best['tp'], best['ts'])
    p(f"      Best-on-train combo : stop {lab(best['sl'])}, TP {lab(best['tp'])}, trail {lab(best['ts'])}")
    p(f"      {'':<22}{'TRAIN EV':>12}{'TEST EV':>12}")
    p(f"      {'no exits (baseline)':<22}{base_tr['ev']:>12.2f}{base_te['ev']:>12.2f}")
    p(f"      {'best combo':<22}{best_tr['ev']:>12.2f}{best_te['ev']:>12.2f}")
    verdict = "GENERALIZES" if best_te['ev'] > base_te['ev'] else "does NOT beat baseline OOS"
    p(f"      => OOS verdict: {verdict}")

    # ── F. User's real config: hard catastrophe stop + trailing as PRIMARY exit ──
    p("")
    p("  [F] YOUR SETUP: hard catastrophe stop + trailing stop (no TP)")
    p("      Trail can be profit-ARMED: it only engages once the trade is up 'ArmAt'%, so")
    p("      the hard stop covers the downside and the trail locks gains without bailing early.")
    SL_F  = [5.0, 8.0, 10.0, 12.0]      # wide catastrophe stops
    ACT_F = [0.0, 2.0, 3.0, 5.0]        # 0 = trail from entry
    TS_F  = [3.0, 5.0, 8.0, 12.0]
    frows = []
    for sl in SL_F:
        for act in ACT_F:
            for ts in TS_F:
                r = evaluate(paths, sl, None, ts, act)
                frows.append({'sl': sl, 'act': act, 'ts': ts, **r})
    fg = pd.DataFrame(frows)
    p(f"      {'HardStop':<9}{'ArmAt':<7}{'Trail':<7}{'EV $':>9}{'Win %':>8}{'Total $':>11}{'%trail':>8}{'%hard':>7}{'%cls':>7}")
    p("      " + "-" * 71)
    for _, r in fg.sort_values('ev', ascending=False).head(10).iterrows():
        arm = 'entry' if r['act'] == 0 else lab(r['act'])
        p(f"      {lab(r['sl']):<9}{arm:<7}{lab(r['ts']):<7}{r['ev']:>9.2f}{r['win']:>8.1f}"
          f"{r['total']:>11.0f}{r['pct_trail']:>8.1f}{r['pct_hardstop']:>7.1f}{r['pct_close']:>7.1f}")
    p(f"      (ride-to-close baseline EV ${base['ev']:.2f}/trade for reference)")

    # OOS for the best [F] config
    pts = sorted(paths, key=lambda t: t['entry_date'])
    cf = int(len(pts) * 0.6)
    bf = fg.sort_values('ev', ascending=False).iloc[0]
    tr_ev = evaluate(pts[:cf], bf['sl'], None, bf['ts'], bf['act'])['ev']
    te_ev = evaluate(pts[cf:], bf['sl'], None, bf['ts'], bf['act'])['ev']
    base_te = evaluate(pts[cf:], None, None, None)['ev']
    arm = 'entry' if bf['act'] == 0 else lab(bf['act'])
    p(f"      Best [F]: hard {lab(bf['sl'])}, arm {arm}, trail {lab(bf['ts'])}  ->  "
      f"train EV {tr_ev:.2f} | test EV {te_ev:.2f} (vs no-exit test {base_te:.2f})")

    p("")
    p("  NOTE: levels assume a fill at the level (gap-throughs filled at bar open). Trailing")
    p("  references the prior-bar high. Treat [D] maxima with suspicion; trust [A]-[C] shapes,")
    p("  the [F] config search, and the [E]/[F] out-of-sample results.")
    p("=" * 78)

    os.makedirs('analysis_output', exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write("\n".join(L) + "\n")
    grid.to_csv(os.path.join('analysis_output', 'exit_optimizer_grid.csv'), index=False)
    print(f"\n  Saved -> {OUT}  (+ exit_optimizer_grid.csv)")


if __name__ == '__main__':
    main()
