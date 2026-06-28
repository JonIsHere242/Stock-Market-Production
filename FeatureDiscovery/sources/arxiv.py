"""
sources/arxiv.py  --  Adapter wrapping the existing arxiv_fetch.py so arXiv joins the
unified multi-source pull. arXiv browses by CATEGORY (newest-first within a day window),
not by keyword, so this adapter ignores `queries`/`since_year` and instead uses its own
category list + day window (passed through opts: arxiv_days, arxiv_categories).

Records from arxiv_fetch already carry the legacy schema; we re-wrap them through
make_record so they gain source/paper_id/doi/venue/year like every other source.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import _base as B

NAME = "arxiv"

# Make the sibling arxiv_fetch.py importable (FeatureDiscovery/ is not a package).
_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))
import arxiv_fetch  # noqa: E402

# Wider than arxiv_fetch's default: include the full q-fin family + econ + the ML cats.
DEFAULT_CATEGORIES = [
    "q-fin.ST", "q-fin.CP", "q-fin.TR", "q-fin.PM", "q-fin.RM",
    "q-fin.MF", "q-fin.GN", "q-fin.PR", "econ.EM", "cs.LG", "stat.ML",
]


def _rewrap(rec: dict) -> dict:
    return B.make_record(
        source=NAME, native_id=rec.get("arxiv_id", ""),
        title=rec.get("title", ""),
        abstract=rec.get("abstract", ""),
        authors=rec.get("authors", ""),
        published=rec.get("published", ""),
        updated=rec.get("updated", ""),
        categories=rec.get("categories", ""),
        primary_category=rec.get("primary_category", ""),
        abs_url=rec.get("abs_url", ""),
        pdf_url=rec.get("pdf_url", ""),
        version=rec.get("version", ""),
    )


def fetch(session, *, queries, since_year, per_source, pause=3.0,
          arxiv_days=4, arxiv_categories=None, **_) -> list[dict]:
    categories = arxiv_categories or DEFAULT_CATEGORIES
    cutoff = datetime.now(timezone.utc) - timedelta(days=arxiv_days)
    per_category = min(per_source, 100)
    raw = arxiv_fetch.fetch_arxiv(categories, cutoff, per_category)
    return [_rewrap(r) for r in raw]
