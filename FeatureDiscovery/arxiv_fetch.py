"""
arxiv_fetch.py  --  Incremental arXiv paper puller for the feature-discovery pipeline.

WHAT IT DOES
------------
Polls the free arXiv API for recent papers in the quant/ML categories, parses the
Atom feed (stdlib xml.etree -- no feedparser dependency), and appends new papers to a
deduplicated parquet store at Data/PaperFeed/papers.parquet. Designed to be run daily:
it only keeps papers newer than --days and dedupes by arXiv id, so re-runs are cheap and
idempotent.

This is step 1 of the pipeline (fetch). Step 2 is relevance_score.py (cheap triage).
Codegen + validation are deliberately downstream and not wired in yet.

USAGE
-----
  # Pull the last 2 days across the default categories, cap 300 papers
  python FeatureDiscovery/arxiv_fetch.py --days 2 --max-results 300

  # First-ever backfill: last 30 days, larger cap
  python FeatureDiscovery/arxiv_fetch.py --days 30 --max-results 2000

  # Custom categories / output
  python FeatureDiscovery/arxiv_fetch.py --categories q-fin.ST q-fin.CP --out Data/PaperFeed/papers.parquet

NOTES
-----
- arXiv asks for a descriptive User-Agent and <= ~1 request / 3s; both are honored.
- Sorted by submittedDate descending; paging stops once a whole page predates the cutoff.
- The store schema is stable so relevance_score.py can add its columns in place.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from xml.etree import ElementTree as ET

import pandas as pd
import requests

ROOT      = Path(__file__).resolve().parent.parent
OUT_PATH  = ROOT / "Data" / "PaperFeed" / "papers.parquet"
API_URL   = "http://export.arxiv.org/api/query"
USER_AGENT = "feature-discovery/0.1 (research; contact: local user)"

DEFAULT_CATEGORIES = ["q-fin.ST", "q-fin.CP", "cs.LG", "stat.ML"]

# Atom / arXiv XML namespaces
_NS = {
    "atom":   "http://www.w3.org/2005/Atom",
    "arxiv":  "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

# Politeness: arXiv recommends >= 3s between requests.
_PAGE_PAUSE_S = 3.0
_PAGE_SIZE    = 100      # results per request (arXiv allows up to ~2000; 100 is polite)
_MAX_RETRIES  = 3


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _clean(text: str | None) -> str:
    """Collapse whitespace/newlines that arXiv embeds in titles and abstracts."""
    if not text:
        return ""
    return " ".join(text.split()).strip()


def _entry_to_dict(entry: ET.Element) -> dict:
    """Convert one Atom <entry> element into a flat record."""
    raw_id = _clean(entry.findtext("atom:id", default="", namespaces=_NS))
    # raw_id looks like http://arxiv.org/abs/2406.12345v2
    tail = raw_id.rsplit("/abs/", 1)[-1]
    if "v" in tail and tail.rsplit("v", 1)[-1].isdigit():
        arxiv_id, version = tail.rsplit("v", 1)
    else:
        arxiv_id, version = tail, ""

    authors = [
        _clean(a.findtext("atom:name", default="", namespaces=_NS))
        for a in entry.findall("atom:author", _NS)
    ]

    primary = entry.find("arxiv:primary_category", _NS)
    primary_cat = primary.get("term", "") if primary is not None else ""
    categories = [c.get("term", "") for c in entry.findall("atom:category", _NS)]

    abs_url = ""
    pdf_url = ""
    for link in entry.findall("atom:link", _NS):
        rel  = link.get("rel", "")
        typ  = link.get("type", "")
        href = link.get("href", "")
        if link.get("title") == "pdf" or typ == "application/pdf":
            pdf_url = href
        elif rel == "alternate":
            abs_url = href

    return {
        "arxiv_id":         arxiv_id,
        "version":          version,
        "title":            _clean(entry.findtext("atom:title", default="", namespaces=_NS)),
        "abstract":         _clean(entry.findtext("atom:summary", default="", namespaces=_NS)),
        "authors":          "; ".join(a for a in authors if a),
        "primary_category": primary_cat,
        "categories":       "; ".join(c for c in categories if c),
        "published":        _clean(entry.findtext("atom:published", default="", namespaces=_NS)),
        "updated":          _clean(entry.findtext("atom:updated", default="", namespaces=_NS)),
        "abs_url":          abs_url,
        "pdf_url":          pdf_url,
        "fetched_at":       datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _parse_feed(xml_bytes: bytes) -> list[dict]:
    root = ET.fromstring(xml_bytes)
    return [_entry_to_dict(e) for e in root.findall("atom:entry", _NS)]


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def _request_page(session: requests.Session, query: str, start: int, page_size: int) -> bytes:
    params = {
        "search_query": query,
        "start":        start,
        "max_results":  page_size,
        "sortBy":       "submittedDate",
        "sortOrder":    "descending",
    }
    url = f"{API_URL}?{urlencode(params)}"
    last_err: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            resp.raise_for_status()
            return resp.content
        except Exception as exc:           # noqa: BLE001 -- retry on any transport error
            last_err = exc
            print(f"    [retry {attempt}/{_MAX_RETRIES}] {type(exc).__name__}: {exc}")
            time.sleep(_PAGE_PAUSE_S * attempt)
    raise RuntimeError(f"arXiv request failed after {_MAX_RETRIES} tries: {last_err}")


def _published_dt(rec: dict) -> datetime | None:
    try:
        return datetime.fromisoformat(rec["published"].replace("Z", "+00:00"))
    except (ValueError, KeyError):
        return None


def _fetch_one_category(
    session: requests.Session,
    category: str,
    cutoff: datetime,
    per_category: int,
    page_size: int,
) -> list[dict]:
    """Page one category newest-first until cutoff / quota / feed end."""
    query = f"cat:{category}"
    out: list[dict] = []
    start = 0
    while len(out) < per_category:
        xml = _request_page(session, query, start, page_size)
        page = _parse_feed(xml)
        if not page:
            break

        page_all_old = True
        for rec in page:
            dt = _published_dt(rec)
            if dt is not None and dt >= cutoff:
                page_all_old = False
                out.append(rec)
                if len(out) >= per_category:
                    break

        if page_all_old:
            break
        start += page_size
        time.sleep(_PAGE_PAUSE_S)

    return out[:per_category]


def fetch_arxiv(
    categories: list[str],
    cutoff:     datetime,
    per_category: int,
    page_size:  int = _PAGE_SIZE,
) -> list[dict]:
    """
    Fetch each category SEPARATELY with its own quota, then merge.

    Querying one combined `cat:A OR cat:B ...` sorted by date lets high-volume
    categories (cs.LG, stat.ML) crowd out low-volume ones (q-fin.*). Per-category
    quotas guarantee the finance categories -- the ones we actually want -- get pulled.
    """
    session = requests.Session()
    out: list[dict] = []
    for cat in categories:
        print(f"  category {cat} (cutoff {cutoff.date()}, quota {per_category}) ...")
        recs = _fetch_one_category(session, cat, cutoff, per_category, page_size)
        print(f"    kept {len(recs)}")
        out.extend(recs)
        time.sleep(_PAGE_PAUSE_S)

    # de-dupe within this run (a paper can be cross-listed in several categories)
    seen: set[str] = set()
    deduped = []
    for r in out:
        if r["arxiv_id"] not in seen:
            seen.add(r["arxiv_id"])
            deduped.append(r)
    return deduped


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def _load_store(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception as exc:           # noqa: BLE001
            print(f"  [WARN] could not read existing store ({exc}); starting fresh.")
    return pd.DataFrame()


def merge_and_save(new_records: list[dict], path: Path) -> tuple[int, int]:
    """Append new records, dedupe by arxiv_id (keep newest fetch), save. Returns (new, total)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(new_records)
    old_df = _load_store(path)

    if old_df.empty:
        merged = new_df
        n_new  = len(new_df)
    elif new_df.empty:
        merged = old_df
        n_new  = 0
    else:
        existing_ids = set(old_df["arxiv_id"])
        n_new  = int((~new_df["arxiv_id"].isin(existing_ids)).sum())
        # new rows first so drop_duplicates(keep='first') prefers the fresh fetch metadata
        merged = pd.concat([new_df, old_df], ignore_index=True)
        merged = merged.drop_duplicates(subset="arxiv_id", keep="first")

    if not merged.empty and "published" in merged.columns:
        merged = merged.sort_values("published", ascending=False).reset_index(drop=True)

    merged.to_parquet(path, index=False)
    return n_new, len(merged)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental arXiv paper puller")
    parser.add_argument("--days", type=int, default=2,
                        help="Keep papers submitted within the last N days (default 2)")
    parser.add_argument("--per-category", type=int, default=100,
                        help="Max papers to pull per category (default 100). Total store "
                             "grows by up to per-category x n-categories each run.")
    parser.add_argument("--categories", nargs="*", default=DEFAULT_CATEGORIES,
                        help=f"arXiv categories (default: {' '.join(DEFAULT_CATEGORIES)})")
    parser.add_argument("--page-size", type=int, default=_PAGE_SIZE,
                        help=f"Results per API request (default {_PAGE_SIZE})")
    parser.add_argument("--out", default=str(OUT_PATH),
                        help="Output parquet store path")
    args = parser.parse_args()

    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
    out_path = Path(args.out)

    print(f"arXiv fetch  --  categories: {', '.join(args.categories)}")
    print(f"  cutoff: {cutoff.isoformat(timespec='seconds')}  |  per-category: {args.per_category}")
    print()

    t0 = time.time()
    records = fetch_arxiv(args.categories, cutoff, args.per_category, args.page_size)
    n_new, n_total = merge_and_save(records, out_path)

    print()
    print(f"Done in {time.time() - t0:.1f}s")
    print(f"  fetched this run : {len(records)}")
    print(f"  new to store     : {n_new}")
    print(f"  store total      : {n_total}  ->  {out_path}")


if __name__ == "__main__":
    main()
