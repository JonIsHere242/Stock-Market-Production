"""
sources/econbiz.py  --  EconBiz (ZBW) economics & business literature search (keyless).

Strong econ/finance coverage (indexes RePEc, EconStor, journals). NOTE: the search
endpoint returns metadata WITHOUT abstracts -- titles only. That still scores in the
relevance triage (title hits count), but expect lower recall than abstract-bearing
sources. Pulling abstracts would need a per-record detail call (skipped to stay cheap).
"""

from __future__ import annotations

import re

from . import _base as B

NAME = "econbiz"
_API = "https://api.econbiz.de/v1/search"
_YEAR_RE = re.compile(r"(19|20)\d{2}")


def _year(date_field) -> str:
    """EconBiz `date` is a list like ['2025'] or ['[2023]'] (sometimes a bare string)."""
    if isinstance(date_field, (list, tuple)):
        date_field = " ".join(str(d) for d in date_field)
    m = _YEAR_RE.search(str(date_field or ""))
    return m.group(0) if m else ""


def _to_record(hit: dict) -> dict:
    rid = hit.get("id", "")
    yr = _year(hit.get("date"))
    return B.make_record(
        source=NAME, native_id=rid or hit.get("title", ""),
        title=hit.get("title", ""),
        abstract="",                                  # not in the search response
        authors=hit.get("creator") or [],
        published=yr,
        year=yr,
        venue=hit.get("source") or "",
        categories=hit.get("type") or [],
        abs_url=f"https://www.econbiz.de/Record/{rid}" if rid else "",
    )


def fetch(session, *, queries, since_year, per_source, pause=0.4, **_) -> list[dict]:
    size = min(25, per_source)

    def page(q):
        data = B.get_json(session, _API, params={"q": q, "size": size}, pause=pause)
        hits = (data.get("hits") or {}).get("hits") or []
        return [_to_record(h) for h in hits]

    recs = B.collect(queries, per_source, page)
    if since_year:
        recs = [r for r in recs if not r["year"] or r["year"] >= str(since_year)]
    return recs
