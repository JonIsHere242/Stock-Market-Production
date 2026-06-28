"""
sources/_base.py  --  Shared plumbing for every paper source adapter.

Every adapter (openalex.py, crossref.py, ...) returns a list of records in ONE
common schema so they can all be merged into the same Data/PaperFeed/papers.parquet
store that arxiv_fetch.py already writes. The canonical builder is make_record().

SCHEMA (superset of the original arxiv_fetch schema -- old columns kept verbatim so
relevance_score.py and generate_feature.py keep working unchanged):
    paper_id          "<source>:<native_id>"   universal stable key
    source            "openalex" | "arxiv" | ...
    arxiv_id          native arXiv id for arXiv rows, else == paper_id
                      (generate_feature.py keys on this column; an opaque string is fine)
    doi               normalised DOI ("" if none) -- used for cross-source dedup
    version, title, abstract, authors, primary_category, categories,
    published, updated, year, venue, abs_url, pdf_url, fetched_at

DESIGN NOTES
------------
- One module per source. Each exposes  NAME  and
      fetch(session, *, queries, since_year, per_source, pause) -> list[dict]
- Sources that support free-text search iterate `queries` (finance-specific, so the
  wide net stays on-topic). Sources that only browse (NBER) ignore `queries`.
- Anything that fails for one source must NOT kill the run -- the orchestrator wraps
  each fetch in try/except. Raise SourceUnavailable for an expected, explainable skip
  (missing API key, hard rate-limit) so the orchestrator can report it cleanly.
"""

from __future__ import annotations

import html
import re
import time
from datetime import datetime, timezone

import requests

USER_AGENT = "feature-discovery/0.2 (quant research; mailto:masamunex9000@gmail.com)"
MAILTO     = "masamunex9000@gmail.com"   # polite-pool identifier for OpenAlex / Crossref


class SourceUnavailable(RuntimeError):
    """Raised for an expected, explainable skip (no key, hard rate-limit)."""


# ---------------------------------------------------------------------------
# Finance-specific query net.  These drive the searchable sources so the wide
# net pulls equity/markets work, not generic ML. Edit freely.
# ---------------------------------------------------------------------------
DEFAULT_QUERIES = [
    "cross-sectional stock returns",
    "stock return predictability",
    "equity return forecasting",
    "technical indicators trading strategy",
    "price momentum factor equities",
    "time series momentum",
    "mean reversion equities",
    "realized volatility forecasting",
    "volatility prediction stock market",
    "machine learning stock return prediction",
    "deep learning financial time series",
    "trend following trading signals",
    "empirical asset pricing machine learning",
    "stock ranking cross-section",
    "anomaly detection financial time series",
    "price pattern prediction stock",
    "factor investing equity returns",
    "intraday price movement prediction",
]

# Curated top quant/finance journals -> ISSN, for the journal-targeted Crossref source.
FINANCE_JOURNAL_ISSNS = {
    "Journal of Finance":                     "0022-1082",
    "Journal of Financial Economics":         "0304-405X",
    "Review of Financial Studies":            "0893-9454",
    "Journal of Financial & Quant. Analysis": "0022-1090",
    "Journal of Banking & Finance":           "0378-4266",
    "Quantitative Finance":                   "1469-7688",
    "Journal of Empirical Finance":           "0927-5398",
    "Journal of Financial Markets":           "1386-4181",
    "Journal of Portfolio Management":        "0095-4918",
    "Financial Analysts Journal":             "0015-198X",
    "Review of Asset Pricing Studies":        "2045-9920",
    "Journal of Financial Data Science":      "2640-3943",
}


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")


def clean(text) -> str:
    """Collapse whitespace, unescape HTML entities; tolerate None/list."""
    if text is None:
        return ""
    if isinstance(text, (list, tuple)):
        text = " ".join(str(t) for t in text if t)
    return " ".join(html.unescape(str(text)).split()).strip()


def strip_html(text) -> str:
    """Drop HTML/JATS tags (Crossref abstracts, NBER author anchors) then clean.

    Crossref abstracts often start with a literal 'Abstract' label and carry &amp;-encoded
    entities; clean() runs html.unescape so &amp;amp;nbsp; etc. don't survive into the store.
    """
    if not text:
        return ""
    out = clean(_TAG_RE.sub(" ", str(text)))
    return re.sub(r"^\s*abstract[:\s]+", "", out, flags=re.IGNORECASE)


def norm_doi(doi) -> str:
    """Normalise a DOI to bare '10.xxxx/...' lowercase form ("" if none)."""
    if not doi:
        return ""
    d = str(doi).strip().lower()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d)
    d = d.replace("doi:", "").strip()
    return d if d.startswith("10.") else ""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def year_of(published: str, fallback=None) -> str:
    if published and len(published) >= 4 and published[:4].isdigit():
        return published[:4]
    return str(fallback) if fallback else ""


# ---------------------------------------------------------------------------
# Canonical record builder -- the ONLY way adapters create rows.
# ---------------------------------------------------------------------------

def make_record(*, source: str, native_id: str, title: str,
                abstract="", authors="", published="", updated="",
                doi="", venue="", categories="", primary_category="",
                abs_url="", pdf_url="", year=None, version="") -> dict:
    native_id = str(native_id)
    paper_id  = f"{source}:{native_id}"
    if isinstance(authors, (list, tuple)):
        authors = "; ".join(clean(a) for a in authors if a)
    if isinstance(categories, (list, tuple)):
        categories = "; ".join(clean(c) for c in categories if c)
    pub = clean(published)
    return {
        "paper_id":         paper_id,
        "source":           source,
        "arxiv_id":         native_id if source == "arxiv" else paper_id,
        "doi":              norm_doi(doi),
        "version":          version,
        "title":            clean(title),
        "abstract":         clean(abstract),
        "authors":          clean(authors),
        "primary_category": clean(primary_category) or source,
        "categories":       clean(categories),
        "published":        pub,
        "updated":          clean(updated) or pub,
        "year":             year_of(pub, year),
        "venue":            clean(venue),
        "abs_url":          abs_url or "",
        "pdf_url":          pdf_url or "",
        "fetched_at":       now_iso(),
    }


# ---------------------------------------------------------------------------
# HTTP helper with retry + polite pause. Returns parsed JSON (or raises).
# ---------------------------------------------------------------------------

def get_json(session: requests.Session, url: str, *, params=None, pause=0.4,
             tries=3, timeout=40, headers=None, skip_on_429=False):
    """
    GET url, return .json(). Retries transient errors. If skip_on_429 and the server
    keeps returning 429 (unauthenticated rate-limit), raise SourceUnavailable so the
    orchestrator records a clean skip instead of a crash.
    """
    h = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if headers:
        h.update(headers)
    last = None
    for attempt in range(1, tries + 1):
        try:
            r = session.get(url, params=params, headers=h, timeout=timeout)
            if r.status_code == 429:
                if skip_on_429 and attempt == tries:
                    raise SourceUnavailable("hard rate-limit (429)")
                last = RuntimeError("429")
                time.sleep(pause * 3 * attempt)
                continue
            r.raise_for_status()
            time.sleep(pause)
            return r.json()
        except SourceUnavailable:
            raise
        except Exception as exc:                     # noqa: BLE001 -- retry any transport error
            last = exc
            time.sleep(pause * attempt)
    raise RuntimeError(f"request failed after {tries} tries: {last}")


def collect(queries, per_source, page_fn, *, dedupe_on="paper_id"):
    """
    Run page_fn(query) -> list[record] across queries, dedupe by `dedupe_on`,
    stop once per_source is reached. Used by the searchable adapters to spread the
    net across many finance queries rather than going deep on one.
    """
    out: dict[str, dict] = {}
    for q in queries:
        try:
            recs = page_fn(q)
        except SourceUnavailable:
            raise
        except Exception as exc:                     # noqa: BLE001
            print(f"      [warn] query {q!r} failed: {type(exc).__name__}: {exc}")
            continue
        for r in recs:
            out.setdefault(r[dedupe_on], r)
        if len(out) >= per_source:
            break
    return list(out.values())[:per_source]
