"""
sources/core.py  --  CORE (core.ac.uk) open-access aggregator search.

Aggregates ~300M OA papers with full text. REQUIRES a free API key in CORE_API_KEY
(register at https://core.ac.uk/services/api). Without it the source cleanly skips.
"""

from __future__ import annotations

import os

from . import _base as B

NAME = "core"
_API = "https://api.core.ac.uk/v3/search/works"


def _to_record(w: dict) -> dict:
    authors = [a.get("name", "") for a in (w.get("authors") or [])]
    doi = w.get("doi") or ""
    journals = w.get("journals") or []
    venue = (journals[0].get("title") if journals and isinstance(journals[0], dict)
             else "") or w.get("publisher", "")
    return B.make_record(
        source=NAME, native_id=str(w.get("id", "")) or doi,
        title=w.get("title", ""),
        abstract=w.get("abstract", "") or "",
        authors=authors,
        published=str(w.get("publishedDate") or w.get("yearPublished") or ""),
        doi=doi,
        venue=venue,
        abs_url=(w.get("links") or [{}])[0].get("url", "") if w.get("links") else "",
        pdf_url=w.get("downloadUrl", "") or "",
    )


def fetch(session, *, queries, since_year, per_source, pause=0.5, **_) -> list[dict]:
    key = os.environ.get("CORE_API_KEY")
    if not key:
        raise B.SourceUnavailable("set CORE_API_KEY to enable (free at core.ac.uk)")
    headers = {"Authorization": f"Bearer {key}"}
    limit = min(25, per_source)

    def page(q):
        q2 = f"{q} AND yearPublished>={since_year}" if since_year else q
        data = B.get_json(session, _API, params={"q": q2, "limit": limit},
                          pause=pause, headers=headers)
        return [_to_record(w) for w in data.get("results", [])]

    return B.collect(queries, per_source, page)
