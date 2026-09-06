"""
check_book.py -- pre-trade guardrail for the execution client.

Run this every morning BEFORE launching 9_SuperFastBroker.py. It refuses a book
that is stale, empty, malformed, or already filled, which are the four ways a
customer machine ends up sending yesterday's orders at today's prices.

This exists because of a real incident: on 2026-06-26 a four-day-old
_Buy_Signals.parquet was left in place and the broker traded it as-is.

Exit codes
    0  book is good, safe to trade
    1  book is unsafe, do NOT launch the broker
    2  could not read the book at all

Usage
    client_env\\Scripts\\python.exe client\\check_book.py
    client_env\\Scripts\\python.exe client\\check_book.py --max-age-days 3
"""

import argparse
import os
import sys

import pandas as pd

# The book lives at the client root, one level up from client\.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOK = os.path.join(ROOT, "_Buy_Signals.parquet")

REQUIRED_COLUMNS = [
    "Symbol", "Status", "TargetDate", "CurrentPrice",
    "EntryPrice", "StopPrice", "TargetPrice",
]


def trading_days_between(start, end):
    """Count NYSE sessions strictly after `start` up to and including `end`.

    Falls back to a weekday count if the calendar package is unavailable, which
    still handles the Friday-to-Monday case correctly (holidays aside).
    """
    if end <= start:
        return 0
    try:
        import exchange_calendars as ec
        sessions = ec.get_calendar("XNYS").sessions_in_range(
            start.tz_localize(None) + pd.Timedelta(days=1),
            end.tz_localize(None),
        )
        return len(sessions)
    except Exception:
        return int(len(pd.bdate_range(start + pd.Timedelta(days=1), end)))


def fail(msg):
    print("  [FAIL] %s" % msg)


def ok(msg):
    print("  [OK]   %s" % msg)


def warn(msg):
    print("  [warn] %s" % msg)


def main():
    ap = argparse.ArgumentParser(description="Validate the daily book before trading.")
    ap.add_argument("--max-age-days", type=int, default=1,
                    help="How many calendar days old CreatedDate may be. Default 1.")
    ap.add_argument("--book", default=BOOK, help="Path to the book parquet.")
    args = ap.parse_args()

    print()
    print("=" * 66)
    print("  PRE-TRADE BOOK CHECK")
    print("  %s" % args.book)
    print("=" * 66)

    if not os.path.exists(args.book):
        fail("book not found. You have not received today's signals yet.")
        return 2

    try:
        df = pd.read_parquet(args.book)
    except Exception as e:
        fail("could not read the book: %s: %s" % (type(e).__name__, e))
        return 2

    problems = 0

    # --- shape -------------------------------------------------------------
    if len(df) == 0:
        fail("book is EMPTY. Nothing to trade. Do not launch the broker.")
        return 1
    ok("%d row(s) in the book" % len(df))

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        fail("missing required columns: %s" % ", ".join(missing))
        problems += 1
    else:
        ok("schema looks right (%d columns)" % len(df.columns))

    # --- freshness ---------------------------------------------------------
    today = pd.Timestamp.now().normalize()

    if "CreatedDate" in df.columns:
        created = pd.to_datetime(df["CreatedDate"]).dt.normalize()
        # Age is measured in TRADING sessions, not calendar days. A book built
        # Friday evening for Monday's open is 3 calendar days old on Monday but
        # zero sessions stale, and must not trip the alarm.
        age_days = trading_days_between(created.max(), today)
        if age_days > args.max_age_days:
            fail("book is %d trading session(s) old (max allowed %d). STALE, do not trade."
                 % (age_days, args.max_age_days))
            problems += 1
        else:
            ok("book is %d trading session(s) old" % age_days)
    else:
        warn("no CreatedDate column, cannot check freshness")

    # TargetDate is the session the book is FOR. It should be today or the next
    # trading day, never in the past.
    if "TargetDate" in df.columns:
        target = pd.to_datetime(df["TargetDate"]).dt.normalize()
        if (target < today).any():
            past = sorted(set(target[target < today].astype(str)))
            fail("TargetDate is in the PAST (%s). This book is for a session that already happened."
                 % ", ".join(past))
            problems += 1
        else:
            ok("TargetDate %s is today or later" % target.max().date())

    # --- tradeable rows ----------------------------------------------------
    if "Status" in df.columns:
        pending = df[df["Status"].astype(str).str.lower() == "pending"]
        if len(pending) == 0:
            fail("no rows with Status 'Pending'. Everything here is already filled or closed.")
            problems += 1
        else:
            ok("%d pending order(s)" % len(pending))
    else:
        pending = df

    # --- sanity on the numbers --------------------------------------------
    if not missing:
        bad_price = df[pd.to_numeric(df["CurrentPrice"], errors="coerce").fillna(0) <= 0]
        if len(bad_price):
            fail("%d row(s) have a non-positive CurrentPrice" % len(bad_price))
            problems += 1

        dupes = df["Symbol"].duplicated().sum()
        if dupes:
            fail("%d duplicate symbol(s) in the book" % dupes)
            problems += 1

    # --- what will actually be traded --------------------------------------
    print()
    print("  Book contents:")
    show = ["Symbol", "Status", "CurrentPrice", "EntryPrice", "StopPrice", "TargetPrice"]
    show = [c for c in show if c in df.columns]
    for _, r in df[show].iterrows():
        parts = []
        for c in show:
            v = r[c]
            parts.append("%s" % v if not isinstance(v, float) else "%.2f" % v)
        print("    " + "  ".join(p.ljust(10) for p in parts))

    print()
    print("=" * 66)
    if problems:
        print("  RESULT: %d PROBLEM(S). DO NOT LAUNCH THE BROKER." % problems)
        print("  Ask for a fresh book before trading.")
        print("=" * 66)
        print()
        return 1

    print("  RESULT: book is good. Safe to launch the broker.")
    print("=" * 66)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
