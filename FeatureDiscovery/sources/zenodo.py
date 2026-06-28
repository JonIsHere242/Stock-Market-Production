"""
sources/zenodo.py  --  Zenodo open repository search (keyless).

Very broad (datasets, software, preprints, published papers). High recall but noisy, so we
restrict to type=publication and lean on the finance queries + downstream relevance gate to
keep the signal. An optional ZENODO_TOKEN raises rate limits but is not required.
"""

from __future__ import annotations

import os

from . import _base as B

NAME = "zenodo"
_API = "https://zenodo.org/api/records"


def _to_record(h: dict) -> dict:
    md = h.get("metadata") or {}
    rid = str(h.get("id", ""))
    links = h.get("links") or {}
    journal = (md.get("journal") or {}).get("title", "")
    return B.make_record(
        source=NAME, native_id=rid or h.get("doi", ""),
        title=md.get("title", ""),
        abstract=B.strip_html(md.get("description", "")),
        authors=[c.get("name", "") for c in (md.get("creators") or [])],
        published=md.get("publication_date", ""),
        doi=h.get("doi") or md.get("doi") or "",
        venue=journal,
        categories=md.get("keywords") or [],
        abs_url=links.get("self_html") or (f"https://zenodo.org/records/{rid}" if rid else ""),
    )


def fetch(session, *, queries, since_year, per_source, pause=0.4, **_) -> list[dict]:
    size = min(25, per_source)
    token = os.environ.get("ZENODO_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else None

    def page(q):
        data = B.get_json(session, _API,
                          params={"q": q, "size": size, "type": "publication",
                                  "sort": "bestmatch"},
                          pause=pause, headers=headers)
        return [_to_record(h) for h in (data.get("hits") or {}).get("hits", [])]

    recs = B.collect(queries, per_source, page)
    if since_year:
        recs = [r for r in recs if not r["year"] or r["year"] >= str(since_year)]
    return recs
