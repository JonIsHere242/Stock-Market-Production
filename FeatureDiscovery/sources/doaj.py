"""
sources/doaj.py  --  Directory of Open Access Journals article search (keyless).

Full open-access journal articles with abstracts. The API can be slow, so we use a longer
timeout. The search term goes in the URL path (URL-encoded).
"""

from __future__ import annotations

from urllib.parse import quote

from . import _base as B

NAME = "doaj"
_API = "https://doaj.org/api/v2/search/articles/{q}"


def _to_record(hit: dict) -> dict:
    bj = hit.get("bibjson") or {}
    doi = ""
    for ident in bj.get("identifier") or []:
        if (ident.get("type") or "").lower() == "doi":
            doi = ident.get("id", "")
    abs_url = pdf_url = ""
    for ln in bj.get("link") or []:
        if (ln.get("type") or "") == "fulltext":
            abs_url = ln.get("url", "")
            if (ln.get("content_type") or "").lower() == "application/pdf":
                pdf_url = ln.get("url", "")
    journal = (bj.get("journal") or {}).get("title", "")
    return B.make_record(
        source=NAME, native_id=hit.get("id") or (doi or bj.get("title", "")),
        title=bj.get("title", ""),
        abstract=bj.get("abstract", ""),
        authors=[a.get("name", "") for a in (bj.get("author") or [])],
        published=str(bj.get("year", "")),
        doi=doi,
        venue=journal,
        categories=[s.get("term", "") for s in (bj.get("subject") or [])],
        abs_url=abs_url,
        pdf_url=pdf_url,
    )


def fetch(session, *, queries, since_year, per_source, pause=0.5, **_) -> list[dict]:
    page_size = min(25, per_source)

    def page(q):
        url = _API.format(q=quote(q))
        data = B.get_json(session, url, params={"pageSize": page_size},
                          pause=pause, timeout=50)
        return [_to_record(h) for h in data.get("results", [])]

    recs = B.collect(queries, per_source, page)
    if since_year:
        recs = [r for r in recs if not r["year"] or r["year"] >= str(since_year)]
    return recs
