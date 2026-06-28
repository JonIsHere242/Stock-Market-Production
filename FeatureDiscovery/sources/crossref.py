"""
sources/crossref.py  --  Crossref free-text works search (keyless, polite pool via mailto).

Indexes published-journal DOIs. Abstracts (when present) are JATS XML, so we strip tags.
A companion source, crossref_journals.py, reuses _to_record to pull the latest issues of
a curated set of top finance journals by ISSN.
"""

from __future__ import annotations

from . import _base as B

NAME = "crossref"
_API = "https://api.crossref.org/works"


def _date(obj: dict) -> str:
    """Crossref date-parts -> 'YYYY-MM-DD' (or 'YYYY' / 'YYYY-MM' if partial)."""
    for key in ("published", "published-online", "published-print", "issued", "created"):
        dp = (obj.get(key) or {}).get("date-parts") or []
        if dp and dp[0]:
            parts = [f"{int(p):02d}" if i else f"{int(p):04d}"
                     for i, p in enumerate(dp[0]) if p is not None]
            return "-".join(parts)
    return ""


def _to_record(it: dict, source=NAME) -> dict:
    title = it.get("title") or []
    venue = it.get("container-title") or []
    authors = [B.clean(f"{a.get('given','')} {a.get('family','')}")
               for a in (it.get("author") or [])]
    pdf = ""
    for ln in it.get("link") or []:
        if ln.get("content-type") == "application/pdf":
            pdf = ln.get("URL", "")
            break
    return B.make_record(
        source=source, native_id=it.get("DOI") or (title[0] if title else ""),
        title=title[0] if title else "",
        abstract=B.strip_html(it.get("abstract", "")),
        authors=authors,
        published=_date(it),
        doi=it.get("DOI") or "",
        venue=venue[0] if venue else "",
        categories=it.get("subject") or [],
        abs_url=it.get("URL") or "",
        pdf_url=pdf,
    )


def fetch(session, *, queries, since_year, per_source, pause=0.4, **_) -> list[dict]:
    rows = min(30, per_source)

    def page(q):
        params = {"query": q, "rows": rows, "mailto": B.MAILTO,
                  "select": "DOI,title,abstract,container-title,author,issued,"
                            "published,created,URL,link,subject"}
        if since_year:
            params["filter"] = f"from-pub-date:{since_year}-01-01"
        data = B.get_json(session, _API, params=params, pause=pause)
        return [_to_record(it) for it in data.get("message", {}).get("items", [])]

    return B.collect(queries, per_source, page)
