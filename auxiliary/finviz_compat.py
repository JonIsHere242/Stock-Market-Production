#!/usr/bin/env python
"""
finviz_compat.py - a layout-resilient replacement for finvizfinance's ticker_fundament().

WHY THIS FILE EXISTS (root cause, diagnosed 2026-07-28)
───────────────────────────────────────────────────────
`finvizfinance(t).ticker_fundament()` raises

    AttributeError: 'NoneType' object has no attribute 'find_all'
    .../finvizfinance/quote.py:144  ->  links = quote_links.find_all("a")

because FinViz RENAMED the Sector/Industry/Country markup. The library looks for

    soup.find("div", class_="quote-links")            # GONE -> None

FinViz now emits the three classification links as

    <a class="quote-header_category" href="screener?v=111&f=sec_technology">Technology</a>
    <a class="quote-header_category" href="screener?v=111&f=ind_consumerelectronics">...</a>
    <a class="quote-header_category" href="screener?v=111&f=geo_usa">USA</a>

Three things matter about this:

  1. It is NOT bot protection and NOT a dead page. The fetch succeeds (243 KB of HTML),
     and the line immediately BEFORE the crash - `h2.quote-header_ticker-wrapper_company`
     - still parses ("Apple Inc"). Only the classification div moved.

  2. `table.snapshot-table2` - which is where **Market Cap** lives - is STILL PRESENT AND
     STILL PARSEABLE. The market cap was never actually unavailable. The library just
     crashed three lines before reaching it. That is the whole reason `CapMillions` came
     back null for every row and the micro-cap gate went silently inert.

  3. UPGRADING DOES NOT FIX IT. finvizfinance 1.3.0 (latest on PyPI, vs 1.2.0 installed)
     ships the byte-identical broken `find("div", class_="quote-links")`. There is no
     upstream fix to wait for, so the parse is repaired here instead. It is repaired HERE
     rather than by editing site-packages so it survives `pip install -r requirements`.

Also note FinViz reworked the snapshot table itself: `Shs Outstand` is GONE, and
`Enterprise Value` / `Dividend Est.` are new. Any code keying off removed fields is
silently degraded - see `EXPECTED_SNAPSHOT_FIELDS` and `selftest()` below.

DESIGN RULES (this bug was caused by silence, so):
  * Per-field degradation, never wholesale collapse: a future rename of the *category*
    links must not take Market Cap down with it, and vice versa.
  * A missing snapshot table is a HARD error (`FinvizLayoutError`), because that means
    real breakage - a redesign, a block page, or a delisted ticker - and the caller must
    be able to tell "FinViz is broken" apart from "this field is legitimately '-'".
  * NEVER return a plausible-looking empty dict. That is the failure mode that cost
    weeks of a dead gate.

LOOKAHEAD WARNING - READ BEFORE USING THIS FOR RESEARCH
────────────────────────────────────────────────────────
`ticker_fundament()` returns FinViz's **CURRENT** market cap, i.e. today's cap regardless
of the date being simulated. Using it inside a backtest to gate a 2023 session leaks
2026 information (and worse, survivorship: a name that has since 10x'd looks large-cap
back when it was a micro). For anything historical use the point-in-time panel
(`Data/MarketCaps/historical_market_caps.parquet`) instead. FinViz caps are appropriate
for LIVE decisions only, where "current" and "as of the session" coincide.
"""

from __future__ import annotations

import re

__all__ = [
    "FinvizLayoutError",
    "ticker_fundament",
    "market_cap_millions",
    "sector_industry",
    "selftest",
]


class FinvizLayoutError(RuntimeError):
    """FinViz's page structure no longer matches what we parse (or we were blocked).

    Raised ONLY when the core snapshot table is unreachable - i.e. the scrape is
    genuinely broken, not merely missing one optional field. Callers should treat this as
    'this data source is down', log it loudly, and fall back to the PIT panel.
    """


# Fields we rely on downstream. `selftest()` checks these against a live mega-cap so a
# future FinViz rename shows up as a failing canary instead of a silent null column.
EXPECTED_SNAPSHOT_FIELDS = ("Market Cap",)
EXPECTED_HEADER_FIELDS = ("Company", "Sector", "Industry")

# Category links, in resolve order. Each entry is (find_all args, kwargs).
_CATEGORY_SELECTORS = (
    ("a", {"class_": "quote-header_category"}),      # current (2026-07)
    ("a", {"class_": "tab-link"}),                   # older FinViz
)
_COMPANY_SELECTORS = (
    ("h2", {"class_": "quote-header_ticker-wrapper_company"}),
    ("h1", {"class_": "quote-header_ticker-wrapper_company"}),
)
_TABLE_SELECTORS = (
    ("table", {"class_": "snapshot-table2"}),
    ("table", {"class_": "snapshot-table"}),
)

# Which screener filter prefix means what, so we key off the STABLE href contract
# (f=sec_/ind_/geo_) rather than fragile link ORDER. The old library did `links[0]`,
# `links[1]`, `links[2]` positionally - that breaks the moment FinViz adds or reorders a
# chip, and it silently mislabels rather than failing.
_HREF_ROLE = (("f=sec_", "Sector"), ("f=ind_", "Industry"), ("f=geo_", "Country"))


def _first(soup, selectors, find_all=False):
    """Return the first hit across a fallback chain of selectors ([] / None if none match)."""
    for name, kw in selectors:
        hit = soup.find_all(name, **kw) if find_all else soup.find(name, **kw)
        if hit:
            return hit
    return [] if find_all else None


def _fetch_soup(ticker: str):
    """Fetch the quote page, reusing finvizfinance's own session/headers/timeout/proxy
    config so `set_proxy`/`set_timeout` still apply and we stay a good citizen."""
    from finvizfinance.util import web_scrap
    return web_scrap("https://finviz.com/quote.ashx?t={}".format(ticker))


def ticker_fundament(ticker: str, soup=None) -> dict:
    """Drop-in replacement for `finvizfinance(t).ticker_fundament()`.

    Returns the same shape the rest of the repo already expects: a flat dict of raw
    strings including 'Company', 'Sector', 'Industry', 'Country' and every
    snapshot-table row ('Market Cap', 'P/E', 'Volatility', ...).

    Raises FinvizLayoutError if the snapshot table cannot be found - that is the
    unmistakable 'FinViz scrape is broken' signal. Individual missing fields are simply
    absent from the dict (callers use .get()), so one rename cannot nuke the rest.
    """
    if soup is None:
        soup = _fetch_soup(ticker)

    info: dict[str, str] = {}

    # ── Company name (optional) ────────────────────────────────────────────────────
    node = _first(soup, _COMPANY_SELECTORS)
    if node is not None:
        info["Company"] = node.text.strip()

    # ── Sector / Industry / Country, resolved BY HREF not by position ──────────────
    for a in _first(soup, _CATEGORY_SELECTORS, find_all=True):
        href = a.get("href") or ""
        for token, role in _HREF_ROLE:
            if token in href and role not in info:
                info[role] = a.text.strip()
                break

    # ── Snapshot table: the actual fundamentals, incl. Market Cap (REQUIRED) ───────
    table = _first(soup, _TABLE_SELECTORS)
    if table is None:
        raise FinvizLayoutError(
            f"FinViz snapshot table not found for {ticker!r} "
            f"(tried {[kw['class_'] for _, kw in _TABLE_SELECTORS]}; page was "
            f"{len(str(soup))} bytes). FinViz redesigned the page, blocked us, or the "
            f"ticker is delisted. Market cap is UNAVAILABLE from FinViz - fall back to "
            f"Data/MarketCaps/historical_market_caps.parquet and fix this parser."
        )

    cells = []
    for row in table.find_all("tr"):
        cells += [td.text.strip() for td in row.find_all("td")]
    # The table is a flat sequence of label/value pairs.
    for label, value in zip(cells[0::2], cells[1::2]):
        if label and label not in info:
            info[label] = value

    return info


_CAP_MULT = {"T": 1_000_000.0, "B": 1000.0, "M": 1.0, "K": 0.001}


def market_cap_millions(fundament: dict | str) -> float | None:
    """FinViz 'Market Cap' string -> cap in $ MILLIONS. None if legitimately absent ('-').

    Accepts either the fundament dict or the raw string. Handles T/B/M/K suffixes and
    thousands separators. Note 'T' is included defensively: FinViz currently renders even
    a $5T company as '4994.88B', but that is a presentation choice they could change.
    """
    raw = fundament.get("Market Cap") if isinstance(fundament, dict) else fundament
    if raw is None:
        return None
    s = str(raw).upper().strip().replace(",", "")
    if not s or s in ("-", "NAN", "NONE"):
        return None
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*([TBMK]?)", s)
    if not m:
        return None
    val = float(m.group(1)) * _CAP_MULT.get(m.group(2), 1.0)
    return val if val > 0 else None


def sector_industry(ticker: str, soup=None) -> tuple[str, str]:
    """(sector, industry) as raw FinViz strings; ('', '') if the chips are missing.

    Feeds 7__MacroFilter's Stage-4 concentration cap and build_data_panels' sector map.
    FinViz Industry is the cleanest source we have - it tags foreign gold miners that SEC
    SIC codes miss - which is why it is worth scraping at all.
    """
    f = ticker_fundament(ticker, soup=soup)
    return str(f.get("Sector", "") or ""), str(f.get("Industry", "") or "")


def selftest(ticker: str = "AAPL", verbose: bool = True,
             min_cap_m: float | None = None) -> bool:
    """Canary: prove the parser still works against a live quote page.

    Run this (or wire it into the nightly) so the NEXT FinViz redesign surfaces as a
    failing check in one line, instead of as a silently null column for weeks.

    `min_cap_m` is a floor on the parsed cap that catches a broken cap PARSER (as opposed
    to a broken fetch) - e.g. a mis-handled suffix turning $5T into $5M. It defaults to
    $100B for the built-in AAPL canary and to merely >0 for any other ticker, since we
    cannot know a priori how big an arbitrary name should be.
    Returns True on success; prints detail on failure.
    """
    if min_cap_m is None:
        min_cap_m = 100_000.0 if ticker.upper() == "AAPL" else 0.0
    ok = True
    try:
        f = ticker_fundament(ticker)
    except Exception as e:                                    # noqa: BLE001
        if verbose:
            print(f"FINVIZ SELFTEST FAILED at fetch/parse: {type(e).__name__}: {e}")
        return False

    missing = [k for k in EXPECTED_HEADER_FIELDS + EXPECTED_SNAPSHOT_FIELDS
               if not f.get(k)]
    if missing:
        ok = False
        if verbose:
            print(f"FINVIZ SELFTEST: fields missing/empty for {ticker}: {missing}")

    cap = market_cap_millions(f)
    if cap is None or cap <= min_cap_m:
        ok = False
        if verbose:
            print(f"FINVIZ SELFTEST: implausible market cap for {ticker}: {cap} ($M) "
                  f"(floor ${min_cap_m:,.0f}M) from raw {f.get('Market Cap')!r}")

    if verbose and ok:
        print(f"FINVIZ SELFTEST OK  {ticker}: {f.get('Company')} | "
              f"{f.get('Sector')} / {f.get('Industry')} | cap ${cap:,.0f}M "
              f"| {len(f)} fields")
    return ok


if __name__ == "__main__":
    import sys
    tickers = sys.argv[1:] or ["AAPL", "WEN", "NKTR"]
    all_ok = True
    for t in tickers:
        all_ok &= selftest(t)
    sys.exit(0 if all_ok else 1)
