"""Model-driven exits: the one exit family the live broker has never run.

WHAT THIS IS
------------
`5v2`/`5__` exit a held position on two rules that read the daily `UpProbability`
series after entry. `9_SuperFastBroker.py` never re-reads `UpProbability` at all (its
exits are the hard stop, the EOD ratchet, the scale-out and the age clock), so these
have had no production counterpart. This module is the shared implementation, imported
by the broker so live and backtest evaluate one rule, the way `trigger_entry.py` and
`bracket_config.py` already do for entries and brackets.

THE RULES, copied from 5v2 evaluate_sell_conditions (lines ~4520 and ~4560):

  momentum   UpProb[today] - UpProb[prev] <= -0.05
  prob_drop  days_held >= 3  AND  max(UpProb[prev .. prev-9]) > 0.55  AND
             UpProb[today] < 0.48

`prob_drop` deliberately excludes today's value from its max window, matching the
backtest's `range(1, min(11, len(data)))`. Both exit at market. In the backtest that
fills at the NEXT session's open; the broker evaluates at 10:00 ET on predictions
written from the prior close, which is the same bar.

WHY IT IS WORTH BUILDING, and what it is not
--------------------------------------------
Full slot sim, 5v2 trigger arm + parkinson, 4 paired shuffle seeds, momentum exit ON
vs OFF (prob_drop held on):

    AnnVol%   +6.91  t +3.63  0/4        WinRate%  -9.74  t -7.58  0/4
    AnnRet%  -21.33  t -0.35  1/4        MaxDD%    +1.21  t +0.49  1/4

Return is a NULL. This is a volatility and win-rate lever, not a return lever, and it
COSTS return-side ground: without it, mean PnLPct per trade is +0.987pp higher and
capital deployed rises from 33.9% to 39.3%. Do not sell this as a way to make money.

Removing BOTH model exits (momentum + prob_drop) gives AnnVol +6.75 / WinRate -10.81,
i.e. almost identical to removing momentum alone. The momentum rule carries the effect;
prob_drop is close to inert. Both ship because both were in the arm that was measured.

LEAK GATE, passed. The rules read the day-to-day CHANGE in a model output, so an
in-sample window would make them hindsight. The ship model's training cutoff was located
empirically (its summary.json records no train_end_date): monthly rank-IC of
UpProbability vs next-session return over 400 tickers / 158,952 name-days steps from
+0.0308 through 2025-11 to +0.0045 after, matching the documented runpercent=75 ->
2025-11-20. Splitting the books at 2025-11-25 (cutoff + 5d embargo), the win-rate effect
is +7.8pp in-sample and +7.9pp out-of-sample, and the median effect is LARGER out of
sample. Not an artifact. Caveat worth repeating when citing it: the in-sample stub is
only 176/159 trades, and a fully OOS re-run is impossible because the backtest window is
[BT_AS_OF - 400d, BT_AS_OF], which would need a BT_AS_OF in the future.

THE SENTINEL GUARD, and an honest note about it
-----------------------------------------------
`4__Predictor` hard-sets `UpProbability` to exactly 0.30 when a name's feature vector
fails the mask. 69.1% of panel rows sit at that sentinel, and a real probability falling
to it is a drop of at least -0.10, double the momentum threshold, so it would ALWAYS
fire. That would tie live exits to the pipeline's daily mask rate.

Measured, it never happens: `BT_MOMEXIT_MODE=pin` (fire ONLY on a transition into the
sentinel) fired 0 times across 4 seeds, and `BT_MOMEXIT_MODE=real` (suppress those
transitions) reproduced the shipped book BYTE-IDENTICALLY, 0 field mismatches on 198
sorted rows. So the guard below costs nothing in sample and changes no measured result.

It is on by default anyway, because live is not the backtest: a held position gets ~3.5
days of chances in the sim, and the sim's universe filter appears to exclude persistently
masked names, whereas the broker would evaluate every held name every morning against
files where the sentinel is the majority value. The guard makes the rule depend on the
signal instead of on pipeline health. Set MODEL_EXIT_SENTINEL_GUARD=0 to reproduce the
raw backtest rule exactly.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from typing import Iterable, NamedTuple, Optional, Sequence

import numpy as np
import pandas as pd

PRED_DIR = os.path.join('Data', 'RFpredictions')

MOMENTUM_DROP = -0.05      # UpProb[0] - UpProb[-1] at or below this exits
PROBDROP_MIN_DAYS = 3      # prob_drop only applies from this holding age
PROBDROP_PEAK = 0.55       # ... and only if the recent peak cleared this
PROBDROP_NOW = 0.48        # ... and today is below this
PROBDROP_LOOKBACK = 10     # prior sessions scanned for the peak, TODAY EXCLUDED

SENTINEL = 0.30            # 4__Predictor's mask-failure hard-set
SENTINEL_EPS = 1e-9

# A prediction file older than this many trading days is not a signal, it is a stale
# file. The nightly runner has died mid-pipeline before (2026-08-28, 17:00:47), which
# leaves yesterday's probabilities looking like today's.
STALE_ABORT_TDAYS = 1


def _sentinel_guard_on() -> bool:
    return os.environ.get('MODEL_EXIT_SENTINEL_GUARD', '1') not in ('0', '', 'false', 'False')


def is_sentinel(v: float) -> bool:
    """True when a value is the predictor's mask-failure hard-set, not a real estimate."""
    return v is not None and np.isfinite(v) and abs(float(v) - SENTINEL) < SENTINEL_EPS


class ExitDecision(NamedTuple):
    exit: bool
    reason: Optional[str]     # 'momentum' | 'prob_drop' | None
    detail: str               # human-readable, goes straight to the broker log


def evaluate(probs: Sequence[float], days_held: int) -> ExitDecision:
    """Apply the two model-exit rules to one name.

    `probs` is that name's UpProbability history in ASCENDING date order, ending with
    the most recent prediction. Returns a decision plus a line explaining it, including
    when the answer is "hold" -- a silent no is indistinguishable from a broken feed.
    """
    v = np.asarray([np.nan if p is None else float(p) for p in probs], dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return ExitDecision(False, None, f'insufficient history ({len(v)} usable rows)')

    cur, prev = float(v[-1]), float(v[-2])
    momentum = cur - prev

    if _sentinel_guard_on() and is_sentinel(cur) and not is_sentinel(prev):
        return ExitDecision(
            False, None,
            f'HELD: UpProb fell {prev:.3f} -> {cur:.3f} but {cur:.3f} is the mask-failure '
            f'sentinel, not an estimate. Sentinel guard suppressed this exit.')

    if momentum <= MOMENTUM_DROP:
        return ExitDecision(
            True, 'momentum',
            f'momentum {momentum:+.3f} <= {MOMENTUM_DROP} (UpProb {prev:.3f} -> {cur:.3f})')

    if days_held >= PROBDROP_MIN_DAYS:
        # Today is excluded from the peak window, matching the backtest.
        window = v[-(PROBDROP_LOOKBACK + 1):-1]
        if len(window) and float(window.max()) > PROBDROP_PEAK and cur < PROBDROP_NOW:
            return ExitDecision(
                True, 'prob_drop',
                f'peak {float(window.max()):.3f} > {PROBDROP_PEAK} over the prior '
                f'{len(window)} sessions and today {cur:.3f} < {PROBDROP_NOW} '
                f'(held {days_held}d)')

    return ExitDecision(
        False, None,
        f'hold: UpProb {prev:.3f} -> {cur:.3f} (momentum {momentum:+.3f}), held {days_held}d')


def load_probs(symbol: str, pred_dir: str = PRED_DIR):
    """Return (ascending UpProbability list, last prediction date) for one symbol.

    Raises FileNotFoundError when the name has no prediction file, which the caller must
    treat as "do not act", never as "no exit signal".
    """
    fp = os.path.join(pred_dir, f'{symbol.upper()}.parquet')
    if not os.path.exists(fp):
        raise FileNotFoundError(fp)
    df = pd.read_parquet(fp, columns=['Date', 'UpProbability'])
    df = df.dropna(subset=['UpProbability']).sort_values('Date')
    if df.empty:
        raise ValueError(f'{fp}: no usable UpProbability rows')
    last = pd.to_datetime(df['Date'].iloc[-1]).date()
    return df['UpProbability'].astype(float).tolist(), last


def staleness_tdays(last_pred: date, today: Optional[date] = None) -> int:
    """NYSE trading days between the last prediction and today. 0 means fresh."""
    today = today or datetime.now().date()
    try:
        import pandas_market_calendars as mcal
        sched = mcal.get_calendar('NYSE').schedule(
            start_date=str(last_pred), end_date=str(today))
        return max(len(sched) - 1, 0)
    except Exception:
        # No calendar available: fall back to weekday counting, which over-counts
        # holidays and therefore fails SAFE (looks staler than it is).
        return max(int(np.busday_count(last_pred, today)), 0)
