"""
sources/crossref_journals.py  --  Latest articles from a curated list of top quant/finance
journals, pulled directly by ISSN. This is the highest-relevance net: instead of guessing
from a keyword search, it walks the actual tables of contents of JF / JFE / RFS / Quant
Finance / J. Financial Data Science / etc. (see FINANCE_JOURNAL_ISSNS in _base.py).

Reuses crossref._to_record so the schema matches the free-text Crossref source exactly.
"""

from __future__ import annotations

from . import _base as B
from .crossref import _to_record

NAME = "crossref_journals"
_API = "https://api.crossref.org/journals/{issn}/works"


def fetch(session, *, queries, since_year, per_source, pause=0.4, **_) -> list[dict]:
    issns = list(B.FINANCE_JOURNAL_ISSNS.values())
    per_journal = max(10, per_source // max(1, len(issns)))
    out: dict[str, dict] = {}
    for name, issn in B.FINANCE_JOURNAL_ISSNS.items():
        params = {"rows": per_journal, "sort": "published", "order": "desc",
                  "mailto": B.MAILTO,
                  "select": "DOI,title,abstract,container-title,author,issued,"
                            "published,created,URL,link,subject"}
        if since_year:
            params["filter"] = f"from-pub-date:{since_year}-01-01"
        try:
            data = B.get_json(session, _API.format(issn=issn), params=params, pause=pause)
        except Exception as exc:                     # noqa: BLE001
            print(f"      [warn] journal {name} ({issn}) failed: {exc}")
            continue
        for it in data.get("message", {}).get("items", []):
            rec = _to_record(it, source=NAME)
            # Mark curated finance venues so the relevance gate credits them (these ARE markets).
            rec["primary_category"] = "q-fin.GN"
            out.setdefault(rec["paper_id"], rec)
        if len(out) >= per_source:
            break
    return list(out.values())[:per_source]
