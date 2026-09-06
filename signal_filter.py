#!/usr/bin/env python
"""Mechanical signal pre-filter: FilterRubric Step-1, the checks that need no web.

WHY THIS FILE EXISTS (again)
----------------------------
`Util.py` imports these four names inside a bare `try/except Exception`, and
`5__NightlyBackTester.py` calls the annotator behind `if ... is not None`. The module
itself was written and tested in an earlier session, wired into both callers, and then
never committed - it is absent from the working tree AND from git history (only
`.claude/docs/HANDOFF_signal_filter.md` survives). The result: `annotate_signals_
mechanical_filter` has been `None` in every run since, the `if` skipped with no log
line, and the pool has NEVER carried MechExclude. Verified 2026-08-27 on a freshly
generated pool.

That is the same silent-failure shape as the micro-cap gate that FinViz killed and the
52-week gate that an empty slice killed. So this module fails LOUD: an unusable input
raises or records an explicit reason, never a quiet pass.

ANNOTATE, DON'T DROP. Callers get verdict columns and decide. `9_SuperFastBroker.py`
trigger mode drops on MechExclude because it bypasses `7__MacroFilter` entirely and
would otherwise have no hard exclusions at all.

THE CHECKS
----------
  1. Price      < $2.00
  2. Market cap < $952M   (CapMillions)
  3. Weekly vol > 5%      (mean daily (High-Low)/Close over the last 5 sessions)
  4. RSI(14) in [30, 40]  -- IMPLEMENTED BUT DEFAULT OFF. See below.

WHY CHECK 4 IS OFF BY DEFAULT, AND WHY THAT IS NOT ME QUIETLY DROPPING IT
------------------------------------------------------------------------
The rubric's "death zone" exclusion is REFUTED by this project's own measurement. On
2.53M cascade rows scored through the live runner bracket, RSI 30-40 is the single most
profitable band in the book: +1.118% mean, 5.6x, 50.1% win rate, positive in BOTH halves
of the sample. Excluding it removes the best names. The band the rubric scores positively
(RSI 50-70) pays +0.133% against +0.289% below RSI 50.

Enabling it is one argument (`rsi_death_zone=True`), so the rubric's behaviour is
reproducible on demand - but it is not the default, because shipping a gate that is
measured to cut the most profitable band is a known-bad default.

Note also that this repo has an open RSI divergence: the predictor gate and the
FilterRubric measure RSI differently and disagree by 11-13 points. This module uses
Wilder smoothing, per the original handoff spec. `9p`/`5v2` use a simple-mean variant.
They are NOT interchangeable; do not compare their numbers.
"""
import argparse
import os
import shutil
from datetime import datetime

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.abspath(__file__))
PRED_DIR = os.path.join(REPO, 'Data', 'RFpredictions')
SIGNALS_FILE = os.path.join(REPO, 'Data', '0__Signals.parquet')

PRICE_FLOOR      = 2.00     # $
CAP_FLOOR_M      = 952.0    # $M
WEEKLY_VOL_CEIL  = 5.0      # %
RSI_DEAD_LO      = 30.0
RSI_DEAD_HI      = 40.0
VOL_SESSIONS     = 5


def compute_rsi14(closes, period=14):
    """Wilder-smoothed RSI. `closes` is oldest-first. None when history is short.

    Wilder, not the simple-mean variant in 9p/5v2. Two RSIs that disagree by 11-13
    points already exist in this repo; adding a third silently would be worse than
    saying which one this is.
    """
    c = np.asarray(closes, dtype=float)
    c = c[np.isfinite(c)]
    if len(c) < period + 1:
        return None
    d = np.diff(c)
    gain = np.where(d > 0, d, 0.0)
    loss = np.where(d < 0, -d, 0.0)
    ag, al = gain[:period].mean(), loss[:period].mean()
    for i in range(period, len(d)):
        ag = (ag * (period - 1) + gain[i]) / period
        al = (al * (period - 1) + loss[i]) / period
    if al == 0:
        return 100.0 if ag > 0 else 50.0
    return float(100.0 - (100.0 / (1.0 + ag / al)))


def compute_weekly_vol_pct(high, low, close, sessions=VOL_SESSIONS):
    """Mean daily (High-Low)/Close over the last `sessions`, in percent.

    None when the inputs cannot support it, so the caller records "unmeasurable"
    rather than treating a missing value as a pass.
    """
    h = np.asarray(high, dtype=float)[-sessions:]
    l = np.asarray(low, dtype=float)[-sessions:]
    c = np.asarray(close, dtype=float)[-sessions:]
    ok = np.isfinite(h) & np.isfinite(l) & np.isfinite(c) & (c > 0)
    if ok.sum() == 0:
        return None
    return float(np.mean((h[ok] - l[ok]) / c[ok]) * 100.0)


def _history(symbol):
    """(open, high, low, close) arrays for one symbol, or None when unavailable."""
    fp = os.path.join(PRED_DIR, f'{symbol}.parquet')
    if not os.path.exists(fp):
        return None
    try:
        px = pd.read_parquet(fp, columns=['Date', 'Open', 'High', 'Low', 'Close'])
    except Exception:
        return None
    if px.empty:
        return None
    px = px.sort_values('Date')
    return (px['Open'].to_numpy(float),
            px['High'].to_numpy(float),
            px['Low'].to_numpy(float),
            px['Close'].to_numpy(float))


def annotate_signals_mechanical_filter(df, rsi_death_zone=False,
                                       price_floor=PRICE_FLOOR,
                                       cap_floor_m=CAP_FLOOR_M,
                                       weekly_vol_ceil=WEEKLY_VOL_CEIL):
    """Add MechRSI14, MechWeeklyVolPct, MechExclude, MechReasons. Drops nothing.

    A check whose input is missing does NOT silently pass: it appends an explicit
    "<check> unmeasurable" reason, visible in MechReasons, without setting MechExclude.
    That keeps a data outage loud but never turns it into an automatic veto.
    """
    out = df.copy()
    rsis, vols, excl, reasons, gaps = [], [], [], [], []
    # AUDIT ONLY: stamp the trigger arm's gap-frequency count with the SAME function the
    # broker's load_pool will use to decide (auxiliary/trigger_entry.gap_count). Never an
    # exclusion here: MechExclude is the Stage-1 hard filter and must not change when the
    # trigger gate is toggled. The broker recomputes and warns on disagreement.
    from auxiliary.trigger_entry import gap_count as _gap_count

    for _, row in out.iterrows():
        sym = row.get('Symbol')
        why = []

        price = row.get('CurrentPrice', row.get('SignalPrice'))
        try:
            price = float(price)
        except (TypeError, ValueError):
            price = None
        if price is None or not np.isfinite(price):
            why.append('price unmeasurable')
        elif price < price_floor:
            why.append(f'price ${price:.2f} < ${price_floor:.2f}')

        cap = row.get('CapMillions')
        try:
            cap = float(cap)
        except (TypeError, ValueError):
            cap = None
        if cap is None or not np.isfinite(cap):
            # The FinViz outage that killed the micro-cap gate in 2026-07 produced
            # exactly this: a null cap that pd.notna() turned into a pass.
            why.append('market cap unmeasurable')
        elif cap < cap_floor_m:
            why.append(f'cap ${cap:,.0f}M < ${cap_floor_m:,.0f}M')

        hist = _history(sym)
        rsi = vol = None
        gap = None
        if hist is None:
            why.append('no price history')
        else:
            op, hi, lo, cl = hist
            gap = _gap_count(op, cl)
            rsi = compute_rsi14(cl)
            vol = compute_weekly_vol_pct(hi, lo, cl)
            if vol is None:
                why.append('weekly vol unmeasurable')
            elif vol > weekly_vol_ceil:
                why.append(f'weekly vol {vol:.2f}% > {weekly_vol_ceil:.2f}%')
            if rsi_death_zone:
                if rsi is None:
                    why.append('RSI unmeasurable')
                elif RSI_DEAD_LO <= rsi <= RSI_DEAD_HI:
                    why.append(f'RSI {rsi:.1f} in death zone '
                               f'[{RSI_DEAD_LO:.0f},{RSI_DEAD_HI:.0f}]')

        # "unmeasurable" is a recorded reason, never an exclusion.
        hard = [w for w in why if 'unmeasurable' not in w and w != 'no price history']
        rsis.append(rsi)
        vols.append(vol)
        excl.append(bool(hard))
        reasons.append('; '.join(why))
        gaps.append(gap)

    out['MechRSI14'] = rsis
    out['MechWeeklyVolPct'] = vols
    out['MechExclude'] = excl
    out['MechReasons'] = reasons
    out['GapCount20'] = gaps
    return out


def prefilter_signals_file(path=SIGNALS_FILE, write=True, drop=False, backup=True,
                           rsi_death_zone=False):
    """Annotate the signals file in place. Returns the annotated frame."""
    df = pd.read_parquet(path)
    ann = annotate_signals_mechanical_filter(df, rsi_death_zone=rsi_death_zone)
    if write:
        if backup:
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            shutil.copy2(path, f'{os.path.splitext(path)[0]}_bak_{stamp}.parquet')
        (ann[~ann['MechExclude']] if drop else ann).to_parquet(path, index=False)
    return ann


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--path', default=SIGNALS_FILE)
    p.add_argument('--no-write', action='store_true', help='report only')
    p.add_argument('--drop', action='store_true', help='physically remove excluded rows')
    p.add_argument('--rsi-death-zone', action='store_true',
                   help='also exclude RSI(14) in [30,40]. OFF by default: measured on '
                        '2.53M rows, that band is the MOST profitable in the book '
                        '(+1.118%%, 50.1%% win, positive in both halves).')
    a = p.parse_args()

    ann = prefilter_signals_file(a.path, write=not a.no_write, drop=a.drop,
                                 rsi_death_zone=a.rsi_death_zone)
    cols = ['Symbol', 'CurrentPrice', 'CapMillions', 'MechRSI14',
            'MechWeeklyVolPct', 'MechExclude', 'MechReasons']
    print(ann[[c for c in cols if c in ann.columns]].to_string(index=False))
    n = int(ann['MechExclude'].sum())
    print(f'\n{len(ann) - n} PASS / {n} EXCLUDE'
          + ('' if a.no_write else f'   -> written to {a.path}'))


if __name__ == '__main__':
    main()
