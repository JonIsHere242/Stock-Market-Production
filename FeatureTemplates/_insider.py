"""
_insider.py  --  Shared POINT-IN-TIME insider-transaction (SEC Form 4) loader for feature blocks.

Auto-skipped by the framework (leading _ keeps it out of block discovery). It is the insider
analogue of _fundamentals.py / _indexes.py: a HELPER that insider-dependent blocks import to reach
data that is NOT in the per-ticker OHLCV frame.

Unlike fundamentals (a point-in-time SNAPSHOT carried forward), insider signal is rolling FLOW:
"how much net buying, by whom, how recently, over the trailing N days". So instead of a single
as_of() snapshot this helper exposes windowed_primitives(): a set of trailing-window aggregates of
informative transactions, computed strictly backward.

THE ONE RULE: POINT-IN-TIME ON `filed_date` (FILING_DATE), NOT `trans_date`
--------------------------------------------------------------------------
A Form 4 may be filed up to two business days AFTER the transaction. We aggregate everything on
`filed_date` (the day it became public). Every rolling sum at calendar day d covers (d-W, d]; every
days-since / last-price lookup is a BACKWARD merge_asof on filed_date. A trading day therefore only
ever sees filings already public by that day -- truncating future price bars can never change a past
value, so insider features pass FeatureDiscovery/validate_feature.py's causality test. Aggregating on
trans_date would be (mild) future leakage. Same discipline as _fundamentals.py keying on `filed`.

INFORMATIVE vs ROUTINE codes
----------------------------
Open-market P (purchase) and S (sale) carry the signal; A (grant), M (exercise), F (tax) are routine.
windowed_primitives() splits them out so a block can build conviction features from P/S and treat
grants/exercises separately.

DATA SOURCE
-----------
Data/Insider/by_ticker/{TICKER}.parquet (one row per transaction, sorted by filed_date), built by
build_insider_panel.py from the SEC Insider Transactions Data Sets.

PUBLIC API
----------
    available()                       -> list[str]    # tickers with a by_ticker parquet
    load_insider(ticker)              -> pd.DataFrame  # raw event frame (fresh copy), cached
    windowed_primitives(df, ...)      -> pd.DataFrame  # df + `_ins_*` trailing-window primitives
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INSIDER_DIR = Path(__file__).resolve().parent.parent / "Data" / "Insider" / "by_ticker"

DEFAULT_WINDOWS = (30, 90, 180, 365)
BREADTH_WINDOW = 90        # window for distinct-owner breadth counts
DECAY_HALFLIFE = 30        # days, for the exponentially-decayed net-dollar-flow primitive

_FRAME_CACHE: dict[str, pd.DataFrame | None] = {}

# Primitive (scratch) columns this helper writes. The block composes named features from these and
# then drops everything beginning with `_ins_`.
_ADDITIVE = ["buy_n", "sell_n", "buy_sh", "sell_sh", "buy_val", "sell_val",
             "grant_sh", "optexer_n", "ceo_buy", "cfo_buy", "officer_buy", "dir_buy",
             "rank_w", "rank_lw"]

# Per-transaction role-seniority rank, used by rank_w/rank_lw sums and the maxrank rolling-max.
# CFO(6) > CEO(5) > other officer(3) > director(2) > 10%-owner(1) > other(0). CFO-over-CEO per
# Wang-Shin-Francis (CFOs' trades more informative). First matching flag wins.
def _role_rank(f: pd.DataFrame) -> np.ndarray:
    return np.select(
        [f["is_cfo"], f["is_ceo"], f["is_officer"], f["is_director"], f["is_tenpct"]],
        [6.0, 5.0, 3.0, 2.0, 1.0], default=0.0)


def _path(ticker: str) -> Path:
    return INSIDER_DIR / f"{ticker}.parquet"


def available() -> list[str]:
    if not INSIDER_DIR.is_dir():
        return []
    return sorted(p.stem for p in INSIDER_DIR.glob("*.parquet"))


def _load_cached(ticker: str) -> pd.DataFrame | None:
    key = ticker.upper()
    if key in _FRAME_CACHE:
        return _FRAME_CACHE[key]
    path = _path(ticker)
    if not path.exists():
        _FRAME_CACHE[key] = None
        return None
    f = pd.read_parquet(path)
    f["filed_date"] = pd.to_datetime(f["filed_date"], errors="coerce").dt.normalize()
    for c in ("shares", "price", "value", "shares_after"):
        if c in f.columns:
            f[c] = pd.to_numeric(f[c], errors="coerce")
    for c in ("is_officer", "is_director", "is_tenpct", "is_ceo", "is_cfo", "is_deriv"):
        if c in f.columns:
            f[c] = f[c].fillna(False).astype(bool)
    f["rank"] = _role_rank(f)
    f = f.dropna(subset=["filed_date"]).sort_values("filed_date").reset_index(drop=True)
    _FRAME_CACHE[key] = f
    return f


def load_insider(ticker: str) -> pd.DataFrame:
    f = _load_cached(ticker)
    return f.copy() if f is not None else pd.DataFrame()


def scratch_columns(windows=DEFAULT_WINDOWS) -> list[str]:
    cols = [f"_ins_{p}_{w}" for w in windows for p in _ADDITIVE]
    cols += [f"_ins_maxrank_{w}" for w in windows]
    cols += [f"_ins_unique_buyers_{BREADTH_WINDOW}", f"_ins_unique_sellers_{BREADTH_WINDOW}"]
    cols += ["_ins_days_since_buy", "_ins_days_since_sell", "_ins_days_since_any",
             "_ins_last_buy_price", "_ins_ewm_net_val"]
    return cols


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #

def _daily_aggregate(ev: pd.DataFrame) -> pd.DataFrame:
    """One row per filed_date with the additive primitives summed over that day's filings."""
    nd = ev[~ev["is_deriv"]]
    buys = nd[(nd["code"] == "P") & (nd["ad"] == "A")]
    sells = nd[(nd["code"] == "S") & (nd["ad"] == "D")]
    grants = nd[nd["code"] == "A"]
    optexer = nd[nd["code"] == "M"]

    # log-dollar weights for the value-weighted role-rank (rank_w / rank_lw -> mean rank = rank_w/rank_lw)
    buy_lw = np.log1p(buys["value"].clip(lower=0))

    parts = {
        "buy_n": buys.groupby("filed_date").size(),
        "sell_n": sells.groupby("filed_date").size(),
        "buy_sh": buys.groupby("filed_date")["shares"].sum(min_count=1),
        "sell_sh": sells.groupby("filed_date")["shares"].sum(min_count=1),
        "buy_val": buys.groupby("filed_date")["value"].sum(min_count=1),
        "sell_val": sells.groupby("filed_date")["value"].sum(min_count=1),
        "grant_sh": grants.groupby("filed_date")["shares"].sum(min_count=1),
        "optexer_n": optexer.groupby("filed_date").size(),
        "ceo_buy": buys[buys["is_ceo"]].groupby("filed_date").size(),
        "cfo_buy": buys[buys["is_cfo"]].groupby("filed_date").size(),
        "officer_buy": buys[buys["is_officer"]].groupby("filed_date").size(),
        "dir_buy": buys[buys["is_director"]].groupby("filed_date").size(),
        "rank_w": buys.assign(_x=buys["rank"] * buy_lw).groupby("filed_date")["_x"].sum(min_count=1),
        "rank_lw": buys.assign(_x=buy_lw).groupby("filed_date")["_x"].sum(min_count=1),
    }
    daily = pd.DataFrame(parts).fillna(0.0)
    daily.index.name = "filed_date"
    return daily


def _window_distinct_owners(ev_sub: pd.DataFrame, window_days: int) -> pd.Series:
    """Step series (indexed by date) of the distinct-owner count over a trailing `window_days`.

    The count changes when an event ENTERS the window (its filed_date) and when one EXITS it
    (filed_date + window). We evaluate the distinct count at exactly those breakpoints, so the step
    series is decay-aware (it drops as old events age out) and bounded by O(events) -- not by the
    number of trading days. A backward merge_asof onto trading dates then reads it PIT-safely.
    """
    if ev_sub.empty:
        return pd.Series(dtype="float64")
    fd = ev_sub["filed_date"].to_numpy("datetime64[ns]")
    owner = ev_sub["owner_cik"].to_numpy()
    order = np.argsort(fd)
    fd, owner = fd[order], owner[order]

    win = np.timedelta64(window_days, "D")
    breakpoints = np.unique(np.concatenate([fd, fd + win]))
    out_dates, out_counts = [], []
    for t in breakpoints:
        lo = np.searchsorted(fd, t - win, side="right")  # filed_date > t-win
        hi = np.searchsorted(fd, t, side="right")         # filed_date <= t
        out_dates.append(t)
        out_counts.append(np.unique(owner[lo:hi]).size if hi > lo else 0)
    return pd.Series(out_counts, index=pd.DatetimeIndex(out_dates), dtype="float64")


def _days_since(dates: pd.DatetimeIndex, sorted_event_dates: np.ndarray) -> np.ndarray:
    """For each date, calendar days since the most recent event_date <= it (NaN if none yet)."""
    if sorted_event_dates.size == 0:
        return np.full(len(dates), np.nan)
    idx = np.searchsorted(sorted_event_dates, dates.to_numpy("datetime64[ns]"), side="right") - 1
    out = np.full(len(dates), np.nan)
    ok = idx >= 0
    diff = dates.to_numpy("datetime64[ns]")[ok] - sorted_event_dates[idx[ok]]
    out[ok] = diff / np.timedelta64(1, "D")
    return out


def windowed_primitives(df: pd.DataFrame, windows=DEFAULT_WINDOWS,
                        breadth_window: int = BREADTH_WINDOW,
                        decay_halflife: int = DECAY_HALFLIFE) -> pd.DataFrame:
    """
    Attach trailing-window insider primitives (`_ins_*`) to a per-ticker OHLCV frame, PIT-safe.

    Additive primitives (counts/shares/dollars, optionally by role) are summed on a complete daily
    calendar grid and rolled over each window so the value at trading day T is exactly the sum over
    (T-W, T]. days-since / last-buy-price / breadth use backward merge_asof. Rows before the ticker's
    first filing (or tickers with no insider coverage) come back as 0 for counts, NaN for prices/days.
    """
    out = df
    scols = scratch_columns(windows)
    ticker = str(df["Ticker"].iloc[0]) if "Ticker" in df.columns and len(df) else None
    ev = _load_cached(ticker) if ticker else None

    dates = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()

    def _fill_no_data():
        for c in scols:
            if c.startswith(("_ins_days_since", "_ins_last_buy_price", "_ins_ewm_net_val")):
                out[c] = np.nan
            else:
                out[c] = 0.0
        return out

    if ev is None or ev.empty:
        return _fill_no_data()

    daily = _daily_aggregate(ev)
    # ev may have rows but NONE in the informative P/S/A/M categories (e.g. only derivative or
    # 'F'-tax rows) -> `daily` comes back empty and daily.index.min() is NaT, which makes the
    # pd.date_range() below raise "Neither `start` nor `end` can be NaT". Treat that (and an
    # all-NaT trading-date frame) exactly like no-coverage: zero/NaN defaults, same as ev empty.
    if daily.empty or pd.isna(daily.index.min()) or pd.isna(dates.max()):
        return _fill_no_data()

    daily["net_val"] = daily["buy_val"].fillna(0.0) - daily["sell_val"].fillna(0.0)

    # Complete daily calendar grid spanning events through the last trading date we must serve.
    start = daily.index.min()
    end = max(daily.index.max(), dates.max())
    grid = pd.date_range(start=start, end=end, freq="D")
    daily = daily.reindex(grid, fill_value=0.0)

    target = dates.to_numpy("datetime64[ns]")

    # --- additive rolling-window sums -> read off at each trading date ---
    for w in windows:
        rolled = daily[_ADDITIVE].rolling(f"{w}D").sum()
        vals = rolled.reindex(grid).reindex(pd.DatetimeIndex(target))
        for p in _ADDITIVE:
            col = vals[p].to_numpy()
            out[f"_ins_{p}_{w}"] = np.where(np.isnan(col), 0.0, col)

    # --- rolling MAX of buyer seniority rank over each window (not additive -> separate max) ---
    _b = ev[(~ev["is_deriv"]) & (ev["code"] == "P") & (ev["ad"] == "A")]
    mr_daily = (_b.groupby("filed_date")["rank"].max().reindex(grid, fill_value=0.0)
                if len(_b) else pd.Series(0.0, index=grid))
    for w in windows:
        mrw = mr_daily.rolling(f"{w}D").max().reindex(pd.DatetimeIndex(target)).to_numpy()
        out[f"_ins_maxrank_{w}"] = np.where(np.isnan(mrw), 0.0, mrw)

    # --- exponentially-decayed net-dollar flow ---
    ewm = daily["net_val"].ewm(halflife=decay_halflife, adjust=False).mean()
    out["_ins_ewm_net_val"] = ewm.reindex(pd.DatetimeIndex(target)).to_numpy()

    # --- days-since (backward) ---
    nd = ev[~ev["is_deriv"]]
    buy_dates = np.sort(nd.loc[(nd["code"] == "P") & (nd["ad"] == "A"), "filed_date"]
                        .to_numpy("datetime64[ns]"))
    sell_dates = np.sort(nd.loc[(nd["code"] == "S") & (nd["ad"] == "D"), "filed_date"]
                         .to_numpy("datetime64[ns]"))
    any_dates = np.sort(np.concatenate([buy_dates, sell_dates])) if (buy_dates.size or sell_dates.size) \
        else np.array([], dtype="datetime64[ns]")
    didx = pd.DatetimeIndex(target)
    out["_ins_days_since_buy"] = _days_since(didx, buy_dates)
    out["_ins_days_since_sell"] = _days_since(didx, sell_dates)
    out["_ins_days_since_any"] = _days_since(didx, any_dates)

    # --- last open-market buy price (backward merge_asof) ---
    buys = nd[(nd["code"] == "P") & (nd["ad"] == "A")][["filed_date", "price"]].dropna()
    if not buys.empty:
        bp = (buys.sort_values("filed_date").groupby("filed_date")["price"].last()
              .rename("p").reset_index())
        left = pd.DataFrame({"d": didx, "__row": np.arange(len(didx))}).sort_values("d")
        merged = pd.merge_asof(left, bp, left_on="d", right_on="filed_date", direction="backward")
        out["_ins_last_buy_price"] = merged.sort_values("__row")["p"].to_numpy()
    else:
        out["_ins_last_buy_price"] = np.nan

    # --- breadth: distinct buyers / sellers over the breadth window (backward step series) ---
    for label, sub in (("buyers", nd[(nd["code"] == "P") & (nd["ad"] == "A")]),
                       ("sellers", nd[(nd["code"] == "S") & (nd["ad"] == "D")])):
        col = f"_ins_unique_{label}_{breadth_window}"
        step = _window_distinct_owners(sub[["filed_date", "owner_cik"]], breadth_window)
        if step.empty:
            out[col] = 0.0
            continue
        step = step.sort_index().rename("c").reset_index().rename(columns={"index": "d"})
        left = pd.DataFrame({"d": didx, "__row": np.arange(len(didx))}).sort_values("d")
        merged = pd.merge_asof(left, step, on="d", direction="backward")
        out[col] = np.where(np.isnan(merged.sort_values("__row")["c"].to_numpy()),
                            0.0, merged.sort_values("__row")["c"].to_numpy())

    return out
