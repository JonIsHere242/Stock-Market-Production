"""
sources/openalex.py  --  OpenAlex works search (keyless, ~250M works).

Full-text relevance search over title+abstract, filtered to recent years. OpenAlex
returns the abstract as an inverted index (word -> [positions]); we reconstruct it.
Polite-pool access via the mailto param (no key required).
"""

from __future__ import annotations

from . import _base as B

NAME = "openalex"
_API = "https://api.openalex.org/works"

# OpenAlex concept ids for Economics (C162324750) and Finance (C10138342). Default to
# restricting the search to these, because a bare keyword search leaks hard non-finance
# noise (plant biology, spectroscopy, lung-function tests) -- exactly the "not related"
# problem. Pass openalex_broad=True to disable and search the whole corpus.
_ECON_FIN_CONCEPTS = "C162324750|C10138342"


def _reconstruct_abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inv.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def _to_record(w: dict) -> dict:
    native = (w.get("id") or "").rsplit("/", 1)[-1]          # W123...
    loc = (w.get("primary_location") or {}).get("source") or {}
    pdf = ((w.get("best_oa_location") or {}) or {}).get("pdf_url") or \
          ((w.get("primary_location") or {}) or {}).get("pdf_url") or ""
    concepts = [c.get("display_name", "") for c in (w.get("concepts") or [])[:6]]
    return B.make_record(
        source=NAME, native_id=native,
        title=w.get("display_name") or w.get("title") or "",
        abstract=_reconstruct_abstract(w.get("abstract_inverted_index")),
        authors=[(a.get("author") or {}).get("display_name", "")
                 for a in (w.get("authorships") or [])],
        published=w.get("publication_date") or "",
        doi=w.get("doi") or "",
        venue=loc.get("display_name") or "",
        categories=concepts,
        abs_url=w.get("id") or "",
        pdf_url=pdf or "",
    )


def fetch(session, *, queries, since_year, per_source, pause=0.4,
          openalex_broad=False, **_) -> list[dict]:
    filters = []
    if since_year:
        filters.append(f"from_publication_date:{since_year}-01-01")
    if not openalex_broad:
        filters.append(f"concepts.id:{_ECON_FIN_CONCEPTS}")
    flt = ",".join(filters)
    per_page = min(50, per_source)

    def page(q):
        params = {"search": q, "per_page": per_page, "mailto": B.MAILTO,
                  "select": "id,display_name,title,abstract_inverted_index,doi,"
                            "publication_date,primary_location,best_oa_location,"
                            "authorships,concepts"}
        if flt:
            params["filter"] = flt
        data = B.get_json(session, _API, params=params, pause=pause)
        return [_to_record(w) for w in data.get("results", [])]

    return B.collect(queries, per_source, page)
