"""
sources/osf.py  --  OSF preprints, SocArXiv provider (keyless).

SocArXiv carries social-science preprints including economics/finance. The OSF API has no
full-text query (filter[q] is rejected), so we filter by single finance KEYWORDS in the
title (filter[title] is a 'contains' match) and loop a small vocabulary. Author names need
a separate contributors call, so they are left blank (titles+abstracts still score).
"""

from __future__ import annotations

from . import _base as B

NAME = "osf"
_API = "https://api.osf.io/v2/preprints/"

_KEYWORDS = ["stock", "stocks", "returns", "volatility", "trading", "momentum",
             "market", "price", "forecast", "portfolio", "equity"]


def _to_record(d: dict) -> dict:
    a = d.get("attributes") or {}
    pid = d.get("id", "")
    links = d.get("links") or {}
    return B.make_record(
        source=NAME, native_id=pid,
        title=a.get("title", ""),
        abstract=a.get("description", ""),
        published=a.get("date_published", "") or a.get("date_created", ""),
        doi=a.get("doi") or "",
        venue="SocArXiv",
        abs_url=links.get("html") or (f"https://osf.io/{pid}" if pid else ""),
    )


def fetch(session, *, queries, since_year, per_source, pause=0.4, **_) -> list[dict]:
    out: dict[str, dict] = {}
    for kw in _KEYWORDS:
        params = {"filter[provider]": "socarxiv", "filter[title]": kw,
                  "page[size]": 20, "sort": "-date_published"}
        if since_year:
            params["filter[date_published][gte]"] = f"{since_year}-01-01"
        try:
            data = B.get_json(session, _API, params=params, pause=pause)
        except Exception as exc:                     # noqa: BLE001
            print(f"      [warn] osf keyword {kw!r} failed: {exc}")
            continue
        for d in data.get("data", []):
            rec = _to_record(d)
            out.setdefault(rec["paper_id"], rec)
        if len(out) >= per_source:
            break
    return list(out.values())[:per_source]
