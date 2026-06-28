"""
sources/hal.py  --  HAL (French national open archive) search (keyless).

Solr-backed API with good economics/finance coverage (mirrors many q-fin preprints and
working papers). We request only the fields we need and sort newest-first.
"""

from __future__ import annotations

from . import _base as B

NAME = "hal"
_API = "https://api.archives-ouvertes.fr/search/"
_FL = ("docid,title_s,abstract_s,authFullName_s,producedDate_s,"
       "doiId_s,uri_s,journalTitle_s")


def _first(v):
    if isinstance(v, (list, tuple)):
        return v[0] if v else ""
    return v or ""


def _to_record(d: dict) -> dict:
    return B.make_record(
        source=NAME, native_id=str(d.get("docid", "")),
        title=_first(d.get("title_s")),
        abstract=_first(d.get("abstract_s")),
        authors=d.get("authFullName_s") or [],
        published=str(d.get("producedDate_s", "")),
        doi=d.get("doiId_s") or "",
        venue=_first(d.get("journalTitle_s")),
        abs_url=d.get("uri_s") or "",
    )


def fetch(session, *, queries, since_year, per_source, pause=0.4, **_) -> list[dict]:
    rows = min(25, per_source)

    def page(q):
        params = {"q": q, "rows": rows, "wt": "json", "fl": _FL,
                  "sort": "producedDate_tdate desc"}
        if since_year:
            params["fq"] = f"producedDate_tdate:[{since_year}-01-01T00:00:00Z TO *]"
        data = B.get_json(session, _API, params=params, pause=pause)
        return [_to_record(d) for d in (data.get("response") or {}).get("docs", [])]

    return B.collect(queries, per_source, page)
