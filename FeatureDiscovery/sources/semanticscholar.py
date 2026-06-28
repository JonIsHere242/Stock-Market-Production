"""
sources/semanticscholar.py  --  Semantic Scholar Graph API paper search.

Huge corpus with abstracts + citation counts. The shared (unauthenticated) pool is heavily
rate-limited (HTTP 429); set S2_API_KEY (free from https://www.semanticscholar.org/product/api)
for reliable access. Without a key we try politely and, on a hard 429, raise SourceUnavailable
so the orchestrator records a clean skip instead of failing the whole run.
"""

from __future__ import annotations

import os

from . import _base as B

NAME = "semanticscholar"
_API = "https://api.semanticscholar.org/graph/v1/paper/search"
_FIELDS = "title,abstract,year,venue,publicationDate,citationCount,externalIds,openAccessPdf,authors"


def _to_record(p: dict) -> dict:
    ext = p.get("externalIds") or {}
    oa = p.get("openAccessPdf") or {}
    return B.make_record(
        source=NAME, native_id=p.get("paperId", ""),
        title=p.get("title", ""),
        abstract=p.get("abstract", "") or "",
        authors=[a.get("name", "") for a in (p.get("authors") or [])],
        published=p.get("publicationDate", "") or str(p.get("year", "") or ""),
        year=p.get("year"),
        doi=ext.get("DOI", ""),
        venue=p.get("venue", ""),
        abs_url=f"https://www.semanticscholar.org/paper/{p.get('paperId','')}",
        pdf_url=oa.get("url", "") or "",
    )


def fetch(session, *, queries, since_year, per_source, pause=1.0, **_) -> list[dict]:
    key = os.environ.get("S2_API_KEY")
    headers = {"x-api-key": key} if key else None
    if not key:
        pause = max(pause, 3.0)                        # be extra gentle on the shared pool
    limit = min(50, per_source)

    def page(q):
        params = {"query": q, "limit": limit, "fields": _FIELDS}
        if since_year:
            params["year"] = f"{since_year}-"
        data = B.get_json(session, _API, params=params, pause=pause,
                          headers=headers, skip_on_429=True, tries=4)
        return [_to_record(p) for p in data.get("data", [])]

    return B.collect(queries, per_source, page)
