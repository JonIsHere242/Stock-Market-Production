"""
sources/nber.py  --  NBER working papers (keyless JSON listing).

NBER working papers are finance/econ by construction, so this source BROWSES the newest
papers (sorted by publication date) rather than running keyword queries -- the downstream
relevance triage filters out the macro/micro/labour papers. Author names arrive as HTML
anchors and the date as a human string ("June 2026"), both normalised here.
"""

from __future__ import annotations

import re

from . import _base as B

NAME = "nber"
_API = ("https://www.nber.org/api/v1/working_page_listing/contentType/"
        "working_paper/_/_/search")

_YEAR_RE = re.compile(r"(19|20)\d{2}")


def _to_record(r: dict) -> dict:
    url = r.get("url") or ""
    abs_url = f"https://www.nber.org{url}" if url.startswith("/") else url
    m = _YEAR_RE.search(str(r.get("displaydate") or ""))
    year = m.group(0) if m else ""
    return B.make_record(
        source=NAME, native_id=r.get("nid") or url.rsplit("/", 1)[-1],
        title=r.get("title", ""),
        abstract=B.strip_html(r.get("abstract", "")),
        authors=B.strip_html(r.get("authors", "")),
        published=year,
        year=year,
        venue="NBER Working Paper",
        primary_category="q-fin.GN",                  # curated econ/finance venue
        abs_url=abs_url,
    )


def fetch(session, *, queries, since_year, per_source, pause=0.5, **_) -> list[dict]:
    out: dict[str, dict] = {}
    per_page = 100
    page = 1
    while len(out) < per_source and page <= 10:
        data = B.get_json(session, _API,
                          params={"page": page, "perPage": per_page,
                                  "sortBy": "public_date"}, pause=pause)
        results = data.get("results") or []
        if not results:
            break
        for r in results:
            rec = _to_record(r)
            if since_year and rec["year"] and rec["year"] < str(since_year):
                continue
            out.setdefault(rec["paper_id"], rec)
        page += 1
    return list(out.values())[:per_source]
