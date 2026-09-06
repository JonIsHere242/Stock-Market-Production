#!/usr/bin/env python
"""Record the morning FilterRubric's verdict onto the signal pool.

WHAT CHANGED, AND WHY THIS FILE EXISTS
--------------------------------------
Before the 2026-08-28 reach-24 changeover the rubric NARROWED: it read the pool, picked
the best 3, and wrote them to `_Buy_Signals.parquet`, which the broker then traded.
Trigger entry does not read that file at all. It arms straight off the wide pool, so a
rubric that writes a 3-name book is a rubric whose output is silently discarded.

The rubric's job is now to SHAVE. It stays in the loop as human judgment, but it marks
the worst few names rather than choosing the good ones, and the broker arms everything
it did not mark. That keeps the wide book the drawdown result was measured on while
still letting the operator throw out a name that a model cannot see is wrong: a pending
lawsuit, a halted ticker, a merger that caps the upside at the offer price.

    pool of 24  ->  rubric marks <= 4 RubricExclude  ->  broker arms ~20

THE CEILING IS ENFORCED IN THE BROKER, NOT HERE. `--max-cut` below is a courtesy check
so a mistake is caught while you are looking at it. The binding limit lives in
`9_SuperFastBroker.py` (`--trigger-rubric-max-cut`, default 4), which REFUSES a veto
deeper than the ceiling rather than trimming it, because a rubric that flagged 20 of 24
is malfunctioning and its first four flags are not trustworthy either.

USAGE

    # mark two names, with reasons
    python rubric_veto.py --exclude DC "cap below floor" --exclude BHC "halted premarket"

    # show what is currently recorded
    python rubric_veto.py --show

    # wipe the verdict (broker then arms the whole mechanically-clean pool)
    python rubric_veto.py --clear

The verdict is stamped with the pool's TargetDate. A verdict written for a previous
session is reported as STALE and the columns are reset, because a carried-over veto is
indistinguishable in the log from a deliberate one.
"""
import argparse
import os
import shutil
from datetime import datetime

import pandas as pd

# Diagnostics sidecars (diagnostics/hooks.py). No-op unless DIAG_OUT is set.
try:
    from diagnostics import hooks as _diag
except Exception:
    class _diag:
        enabled = staticmethod(lambda: False)
        dump_json = dump_parquet = append_jsonl = stamp = staticmethod(lambda *a, **k: None)

REPO = os.path.dirname(os.path.abspath(__file__))
POOL = os.path.join(REPO, 'Data', '0__Signals.parquet')

VETO_COLS = ('RubricExclude', 'RubricReasons', 'RubricStamped')


def _target_date(df):
    for c in ('TargetDate', 'SignalDate', 'CreatedDate'):
        if c in df.columns:
            d = pd.to_datetime(df[c], errors='coerce').dropna()
            if not d.empty:
                return d.max().date()
    return None


def load_pool(path=POOL):
    if not os.path.exists(path):
        raise FileNotFoundError(f'no signal pool at {path}')
    return pd.read_parquet(path)


def reset_stale(df):
    """Blank a verdict that was written for a different session. Returns (df, was_stale)."""
    if 'RubricStamped' not in df.columns:
        return df, False
    tgt = _target_date(df)
    stamped = {str(v) for v in df['RubricStamped'].dropna().unique()}
    if not stamped or stamped == {str(tgt)}:
        return df, False
    df = df.drop(columns=[c for c in VETO_COLS if c in df.columns])
    return df, True


def apply_veto(df, exclusions, max_cut=4):
    """exclusions: {SYMBOL: reason}. Returns the annotated frame."""
    out = df.copy()
    tgt = _target_date(out)
    up = {str(k).upper(): (v or 'operator veto, no reason given')
          for k, v in exclusions.items()}

    unknown = sorted(set(up) - {str(s).upper() for s in out['Symbol']})
    if unknown:
        raise SystemExit(
            f'not in the pool: {unknown}. Vetoing a name that is not a candidate is '
            f'almost always a typo, and a typo here silently vetoes nothing.')
    if len(up) > max_cut:
        raise SystemExit(
            f'{len(up)} vetoes exceeds --max-cut {max_cut}. The rubric SHAVES a wide '
            f'book now; a cut this deep is a narrowing. The broker would refuse it '
            f'whole anyway (--trigger-rubric-max-cut).')

    sym_u = out['Symbol'].astype(str).str.upper()
    out['RubricExclude'] = sym_u.isin(up)
    out['RubricReasons'] = sym_u.map(up).fillna('')
    out['RubricStamped'] = str(tgt)
    _diag.append_jsonl("M_rubric_veto", {"target": str(tgt), "exclusions": up, "n_cut": len(up),
                                         "max_cut": max_cut, "n_pool": int(len(out))})
    return out


def show(df):
    if 'RubricExclude' not in df.columns:
        print('No rubric verdict recorded. The broker will arm the whole '
              'mechanically-clean pool.')
    cols = [c for c in ('Symbol', 'UpProbability', 'CurrentPrice', 'MechExclude',
                        'MechReasons', 'RubricExclude', 'RubricReasons')
            if c in df.columns]
    print(df[cols].to_string(index=False))
    mech = int(df['MechExclude'].sum()) if 'MechExclude' in df.columns else 0
    veto = int(df['RubricExclude'].sum()) if 'RubricExclude' in df.columns else 0
    print(f'\npool {len(df)} | mechanical exclude {mech} | rubric veto {veto} '
          f'-> {len(df) - mech - veto} would arm')
    if 'MechExclude' not in df.columns:
        print('WARNING: no MechExclude column. Run signal_filter.py before trading - '
              'trigger mode bypasses 7__MacroFilter, so this is the only place the '
              'price/cap/vol hard gates are applied.')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--path', default=POOL)
    p.add_argument('--exclude', nargs=2, action='append', metavar=('SYMBOL', 'REASON'),
                   default=[], help='veto one name, with a reason. Repeatable.')
    p.add_argument('--max-cut', type=int, default=4,
                   help='refuse to write more vetoes than this (default 4)')
    p.add_argument('--clear', action='store_true', help='remove any recorded verdict')
    p.add_argument('--show', action='store_true', help='print the pool and exit')
    p.add_argument('--no-backup', action='store_true')
    a = p.parse_args()

    df = load_pool(a.path)
    df, was_stale = reset_stale(df)
    if was_stale:
        print('STALE VERDICT CLEARED: the recorded veto was stamped for a previous '
              'session and has been removed.')

    if a.show and not a.exclude and not a.clear:
        show(df)
        return

    if not a.no_backup:
        bak = f'{a.path}.{datetime.now():%Y%m%d_%H%M%S}.bak'
        shutil.copy2(a.path, bak)
        print(f'backup -> {bak}')

    if a.clear:
        out = df.drop(columns=[c for c in VETO_COLS if c in df.columns])
        print('rubric verdict cleared.')
    else:
        out = apply_veto(df, dict(a.exclude), max_cut=a.max_cut)

    out.to_parquet(a.path, index=False)
    print()
    show(out)


if __name__ == '__main__':
    main()
