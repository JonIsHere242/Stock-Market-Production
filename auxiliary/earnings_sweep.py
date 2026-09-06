"""Held-book earnings sweep. FILTER ONLY - never places, modifies, or cancels orders.

Scans the currently held book for earnings reports landing inside the hold window and
scores the consensus setup for each. Output is a per-symbol verdict for the operator:

    HOLD_OK   earnings inside the window but the consensus setup leans positive
    CAUTION   earnings inside the window and the consensus setup leans negative/unclear
    CLEAR     no earnings inside the window

Policy (operator, 2026-08-12): a positive-consensus print is acceptable to hold through;
a negative one gets FLAGGED for the operator to decide. This script must never gain the
ability to act on a position. It reads public estimate data (yfinance) and prints.

Origin: 2026-08-12, HLIT reported AMC while held (worked out +20%, but was unscreened
risk; the same night STAA was skipped for entry over the same setup). The entry rubric
screens new buys for earnings; this closes the same gap for the held book.

Usage:
    python earnings_sweep.py                    # positions from Data/position_ledger.parquet
    python earnings_sweep.py --symbols HLIT EE  # explicit list
    python earnings_sweep.py --days 7           # look-ahead window (default 7 calendar days)
"""

import argparse
import sys
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

LEDGER = 'Data/position_ledger.parquet'


def consensus_score(tkr: yf.Ticker) -> tuple[int, list[str]]:
    """Score the consensus setup. Positive score leans hold-ok, negative leans caution."""
    score, notes = 0, []

    try:
        rev = tkr.eps_revisions
        if rev is not None and '0q' in rev.index:
            up = int(rev.loc['0q'].get('upLast30days') or 0)
            dn = int(rev.loc['0q'].get('downLast30days') or 0)
            if up > dn:
                score += 1
                notes.append(f'revisions up 30d ({up} up / {dn} down)')
            elif dn > up:
                score -= 1
                notes.append(f'revisions DOWN 30d ({up} up / {dn} down)')
    except Exception:
        notes.append('revisions unavailable')

    try:
        est = tkr.earnings_estimate
        if est is not None and '0q' in est.index:
            avg = est.loc['0q'].get('avg')
            yago = est.loc['0q'].get('yearAgoEps')
            if avg is not None and yago is not None:
                if avg > yago:
                    score += 1
                    notes.append(f'est EPS {avg:.2f} vs yr-ago {yago:.2f} (growth)')
                else:
                    score -= 1
                    notes.append(f'est EPS {avg:.2f} vs yr-ago {yago:.2f} (SHRINKING)')
    except Exception:
        notes.append('estimates unavailable')

    try:
        hist = tkr.earnings_history
        if hist is not None and len(hist) > 0:
            srp = hist['surprisePercent'].dropna().tail(4)
            beats = int((srp > 0).sum())
            if beats >= 3:
                score += 1
                notes.append(f'beat {beats}/{len(srp)} recent qtrs')
            elif beats <= 1:
                score -= 1
                notes.append(f'beat only {beats}/{len(srp)} recent qtrs')
    except Exception:
        notes.append('surprise history unavailable')

    try:
        rec = (tkr.info or {}).get('recommendationMean')
        if rec is not None:
            if rec <= 2.0:
                score += 1
                notes.append(f'rec mean {rec:.2f} (bullish street)')
            elif rec >= 3.0:
                score -= 1
                notes.append(f'rec mean {rec:.2f} (bearish street)')
    except Exception:
        pass

    return score, notes


def next_earnings_date(tkr: yf.Ticker):
    try:
        cal = tkr.calendar
        dates = cal.get('Earnings Date') if isinstance(cal, dict) else None
        if dates:
            future = [d for d in dates if d >= date.today()]
            return min(future) if future else None
    except Exception:
        pass
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', nargs='*', default=None,
                    help='explicit symbols; default reads ' + LEDGER)
    ap.add_argument('--days', type=int, default=7,
                    help='calendar-day look-ahead for the hold window (default 7)')
    args = ap.parse_args()

    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        led = pd.read_parquet(LEDGER)
        symbols = sorted(led['Symbol'].unique())

    horizon = date.today() + timedelta(days=args.days)
    print(f'Earnings sweep {date.today()} | window through {horizon} | {len(symbols)} symbols')
    print('FILTER ONLY: this report flags names; the operator decides any action.\n')

    flagged = 0
    for s in symbols:
        tkr = yf.Ticker(s)
        ed = next_earnings_date(tkr)
        if ed is None:
            print(f'{s:6} CLEAR    no upcoming earnings date found (verify manually if suspicious)')
            continue
        if ed > horizon:
            print(f'{s:6} CLEAR    next earnings {ed} (outside window)')
            continue
        score, notes = consensus_score(tkr)
        verdict = 'HOLD_OK' if score >= 2 else 'CAUTION'
        if verdict == 'CAUTION':
            flagged += 1
        print(f'{s:6} {verdict:8} earnings {ed} IN WINDOW | consensus score {score:+d} | '
              + '; '.join(notes))

    print(f'\n{flagged} name(s) flagged CAUTION. For each: check the news, then the operator '
          'decides hold / trim / exit. No orders are placed by this tool.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
