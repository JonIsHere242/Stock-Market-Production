"""SINGLE SOURCE OF TRUTH for the exit bracket.

Why this file exists
--------------------
Audit 2026-07-28 found the pipeline carrying FIVE different bracket definitions, none
of which agreed, and the backtest simulating a strategy the broker does not trade:

  1. 5__NightlyBackTester.py:2965  execute_buy_with_bracket -- HARDCODED
                                   3.0% trailing stop + 20% target   <- what the
                                   headline annualised return was actually built on
  2. 5__NightlyBackTester.py:3906  the signals-file block -- BT_STOP_PCT (default 2.0%)
                                   + TAKE_PROFIT_PERCENT 10.0, a local shadowing var
  3. 9_SuperFastBroker.py:39,104   HARD_STOP_PCT 1.9% + TAKE_PROFIT_PCT 3.5%, trail OFF
                                   <- what actually trades real money
  4. Util.STRATEGY_PARAMS          'stop_loss_percent': 5.0
  5. Util.STRATEGY_PARAMS          'stop_loss_atr_multiple': 0.75,
                                   'trailing_stop_atr_multiple': 2.0,
                                   'take_profit_percent': 20.0

Consequences that were measured, not guessed:
  * BT_STOP_PCT was INERT -- 2.0/2.5/3.0 gave byte-identical annualised return on all
    4 seeds tested, because STOP_LOSS_PERCENT only ever reached the signals file.
  * --stop_loss_atr was a declared-but-unused CLI argument.
  * self.trailing_stops was never assigned, so determine_exit_reason could not see the
    stop and 82.3% of 7795 trades fell through to the "Manual Exit" else-branch.
    "Stop Loss" was unreachable dead code and the backtest reported 0.0% stop-outs
    while the intraday fill sim measured ~47-53% on the same book.

THE LIVE BROKER IS CANONICAL. Its numbers are the ones that have earned real money, so
consolidation moves the backtest onto the broker's bracket, never the other way round.

Changing a value here changes the backtest, the signals file and the live orders
together. That is the point. There is no second place to edit.
"""
import os


def _f(env, default):
    try:
        return float(os.environ[env])
    except (KeyError, ValueError):
        return default


# ── The bracket ────────────────────────────────────────────────────────────────
# WIDENED 2026-07-29 from 1.9 / 3.5 to 3.0 / 6.0. Evidence, 8 paired seeds on the full
# slot sim with engine invariants verified per cell:
#     Calmar 2.15 -> 6.75    maxDD 19.36% -> 14.98% (better)    Sharpe 1.09 -> 2.00
#     PF 1.09 -> 1.21        ann 39.89 -> 95.47, 8/8 seeds       trades 1035 -> 829
# Stop width peaks at 3.0 in BOTH the fixed-target and the 2:1-target family, so it is an
# interior optimum rather than a boundary artifact. On the regime split it is the ONLY
# arm positive in both halves and its LARGER gain is in the weak 2026-H1 half (expR
# 0.0696 -> 0.1338), which is the opposite of what an in-sample artifact looks like.
#
# The reason this sat unfound: every prior sweep moved ONE leg. COMPREHENSIVE_VERDICT
# widened the stop while pinning the target at +3.5%, so a 4% stop was really a 0.9:1
# trade, and its stop axis and target axis appeared to contradict each other. Both legs
# have to move together to keep the trade shape.
#
# Trust Calmar and expR here, NOT ann%. This backtester compounds off days with 2-3
# trades, so the annual figure exaggerates in both directions. And 4.0/8.0 is genuinely
# too wide: negative in the weak half in both families.
#
# Rollback is one line: BRACKET_STOP_PCT=1.9 BRACKET_TP_PCT=3.5
HARD_STOP_PCT      = _f('BRACKET_STOP_PCT', 3.0)

# The risk basis the target is sized from, and the reward:risk ratio. Holding RR at 2:1
# while the stop widens is the whole point; pinning the target instead shrinks RR and
# made the strong half WORSE at every width tested.
STOP_FOR_RISK_PCT  = _f('BRACKET_RISK_PCT', 3.0)
TARGET_RR          = _f('BRACKET_TARGET_RR', 2.0)

# Take-profit, as a % above the entry fill. 3.0 * 2.0 = +6.0%.
TAKE_PROFIT_PCT    = _f('BRACKET_TP_PCT', STOP_FOR_RISK_PCT * TARGET_RR)

# Age-based exit. The broker flattens a position older than this.
MAX_HOLD_DAYS      = int(_f('BRACKET_MAX_HOLD', 5))

# Trailing stop. OFF, and this is a measured decision, not an oversight: the lab found
# every trailing setting loses (1.5% -> -0.179, 2.5% -> -0.122, 4.0% -> -0.010 pp per
# signal, all 8/8 books negative). Ratcheting the stop up cuts winners before they
# reach a 2:1 target and destroys the payoff asymmetry the strategy runs on.
# 2026-07-29 referee re-run at the missing 3.0 dose: intraday-HWM trailing is still
# 0/9 books (-0.57 pp/sig). A true exchange TRAIL order must stay off. The instrument
# that DID survive the slot sim is the EOD-repegged stop below (runner exits).
USE_TRAILING_STOP  = os.environ.get('BRACKET_TRAIL', '0') not in ('0', '', 'false', 'False')
TRAIL_PCT          = _f('BRACKET_TRAIL_PCT', 3.0)

# ── Runner exits (2026-07-29) ─────────────────────────────────────────────────
# The exit package validated on the 5.1 lab rig (memo:
# project_trail_runner_exit_package_2026_07_29): NO take-profit leg; a plain GTC
# stop re-pegged each morning to prior close * (1 - TRAIL_EOD_PCT), never lowered
# (an EOD ratchet, NOT an exchange TRAIL order); at +RUNNER_TRIG_PCT unrealized,
# sell (1 - RUNNER_KEEP_FRAC) of the position and let the runner ride the repegged
# stop up to RUNNER_MAX_HOLD_DAYS. Honest-instrument slot sim: TR 121.11 vs 78.48,
# Calmar 8.35 vs 6.70 on the discovery window; multi-seed and anchor confirms in
# the memo. FLIPPED ON 2026-07-29 after the honest-instrument confirm: 8/8 seeds
# (dTR mean +53.3pp, t~6.4) and 5 held-out anchors (3/5; both falling windows a
# wash with equal-or-better drawdown; chop window ending 2025-10-31 gives back
# 9.6pp; latest window +72.8pp). Rollback is one env var: BRACKET_RUNNER=0.
USE_RUNNER_EXITS     = os.environ.get('BRACKET_RUNNER', '1') not in ('0', '', 'false', 'False')
# SCALE-OUT CHANGED 2026-08-29: sell HALF at +15% (was 80% at +10%). Live-faithful trigger
# arm slot sim, cashadjust fixed, model exits off, paired shuffle seeds:
#   trig 15 + keep 0.5 alone vs base:  AnnRet +12.9 t 3.5 4/4, Sharpe +0.32 4/4, maxDD flat
#   keep 0.5 at +10 alone:             AnnRet +5.0 t 2.1 4/4, maxDD unchanged
#   with the gap gate (the baseline):  8 seeds AnnRet +45.5 t 9.3 8/8, maxDD -3.5 7/8,
#                                      merged mean/TRADE +0.66 t 8.2 8/8
# Widening the stop or the ratchet loses every way (the ratchet is the slot recycler);
# keeping more of a winner past +10% is the one exit change that pays. Live tickets: 12
# of 195 ledger trades since 2026-06-01 reached +15% MFE. Rollback is two env vars:
# BRACKET_RUNNER_TRIG=10 BRACKET_RUNNER_KEEP=0.2. Memo project_exit_sweeps_trigger_arm_2026_08_29.
RUNNER_TRIG_PCT      = _f('BRACKET_RUNNER_TRIG', 15.0)
RUNNER_KEEP_FRAC     = _f('BRACKET_RUNNER_KEEP', 0.50)
RUNNER_MAX_HOLD_DAYS = int(_f('BRACKET_RUNNER_MAXHOLD', 15))
TRAIL_EOD_PCT        = _f('BRACKET_TRAIL_EOD', 3.0)

# ── Rollback lever, NOT a second bracket ──────────────────────────────────────
# Set BRACKET_LEGACY=1 to make the BACKTEST ONLY reproduce its pre-consolidation
# behaviour (3% trailing stop + 20% target, no hard stop). It exists so old headline
# numbers can be regenerated for comparison. It does not affect the live broker.
LEGACY_BACKTEST    = os.environ.get('BRACKET_LEGACY', '0') not in ('0', '', 'false', 'False')
LEGACY_TRAIL_PCT   = 3.0
LEGACY_TP_PCT      = 20.0


def stop_price(entry_price):
    """Hard-stop level for a long entered at `entry_price`."""
    return entry_price * (1.0 - HARD_STOP_PCT / 100.0)


def target_price(entry_price):
    """Take-profit level for a long entered at `entry_price`."""
    return entry_price * (1.0 + TAKE_PROFIT_PCT / 100.0)


# The bracket that has actually traded real money and has a live P&L record behind it.
# 3.0/6.0 beats it on 8 seeds of IN-SAMPLE backtest and has never been traded.
LIVE_VALIDATED = (1.9, 3.5)


def is_live_validated():
    return (HARD_STOP_PCT, TAKE_PROFIT_PCT) == LIVE_VALIDATED


def live_warning():
    """Loud banner for 9_SuperFastBroker.py when the bracket is not the traded one.

    This module is shared, so widening it for research widens it for the live broker on
    its very next run. That is the correct design and it is also the obvious way to lose
    money by accident, so the broker says so every time rather than assuming it was
    intended.
    """
    if is_live_validated():
        return None
    return (f"BRACKET IS NOT THE LIVE-VALIDATED ONE. "
            f"placing -{HARD_STOP_PCT}% / +{TAKE_PROFIT_PCT}% "
            f"(live-validated is -{LIVE_VALIDATED[0]}% / +{LIVE_VALIDATED[1]}%). "
            f"3.0/6.0 is backed by 8 seeds of IN-SAMPLE backtest and has never traded. "
            f"To revert for this run: BRACKET_STOP_PCT=1.9 BRACKET_TP_PCT=3.5")


def describe():
    if LEGACY_BACKTEST:
        return (f"LEGACY backtest bracket: {LEGACY_TRAIL_PCT}% trailing stop, "
                f"+{LEGACY_TP_PCT}% target, NO hard stop (pre-2026-07-28 behaviour)")
    if USE_RUNNER_EXITS:
        return (f"RUNNER bracket: -{HARD_STOP_PCT}% stop repegged EOD to close-{TRAIL_EOD_PCT}%, "
                f"NO take-profit, scale out {(1 - RUNNER_KEEP_FRAC) * 100:.0f}% at "
                f"+{RUNNER_TRIG_PCT}%, runner max {RUNNER_MAX_HOLD_DAYS}d, "
                f"base max hold {MAX_HOLD_DAYS}d")
    return (f"bracket: -{HARD_STOP_PCT}% hard stop, +{TAKE_PROFIT_PCT}% target "
            f"({STOP_FOR_RISK_PCT}% risk x {TARGET_RR}:1), max hold {MAX_HOLD_DAYS}d, "
            f"trail {'ON @' + str(TRAIL_PCT) + '%' if USE_TRAILING_STOP else 'OFF'}")


if __name__ == '__main__':
    print(describe())
    for p in (10.0, 34.49, 218.35):
        print(f"  entry ${p:>7.2f} -> stop ${stop_price(p):>7.2f}  target ${target_price(p):>7.2f}")
