"""
sources package  --  registry of paper-source adapters for the feature-discovery pipeline.

Each adapter module exposes:
    NAME : str
    fetch(session, *, queries, since_year, per_source, pause, **opts) -> list[dict]
and builds rows via _base.make_record so every source shares one schema.

The orchestrator (fetch_papers.py) reads REGISTRY / DEFAULT_SOURCES / SOURCE_PRIORITY here.
"""

from __future__ import annotations

from . import (arxiv, core, crossref, crossref_journals, doaj, econbiz, hal,
               nber, openalex, osf, semanticscholar, zenodo)

# name -> module
REGISTRY = {m.NAME: m for m in (
    arxiv, openalex, crossref, crossref_journals, doaj, econbiz,
    nber, osf, zenodo, hal, semanticscholar, core,
)}

# Fully keyless -- run by default.
KEYLESS = ["arxiv", "openalex", "crossref", "crossref_journals", "doaj",
           "econbiz", "nber", "osf", "zenodo", "hal"]

# Work better (or only) with a free API key; included by default but skip cleanly if unset.
KEY_OPTIONAL = ["semanticscholar", "core"]

DEFAULT_SOURCES = KEYLESS + KEY_OPTIONAL

# Quality-tilted subset (from the source-audit): the sources that actually surface
# OHLCV-implementable, low-adaptation equity signals. OpenAlex (canonical tail anomalies),
# curated finance journals, and EconBiz (topical precision) lead; DOAJ/HAL are kept but
# yield only on precise queries; Zenodo/OSF are dropped (junk / wrong field).
HIGH_YIELD = ["arxiv", "openalex", "crossref_journals", "econbiz", "doaj", "hal",
              "semanticscholar", "core"]

# Cross-source dedup tie-break: when the same paper appears in several sources, keep the
# row from the source highest in this list (earlier = preferred). arXiv first because we
# already process those ids; curated finance journals next; broad/noisy repos last.
SOURCE_PRIORITY = [
    "arxiv", "crossref_journals", "nber", "openalex", "semanticscholar",
    "crossref", "doaj", "hal", "core", "econbiz", "osf", "zenodo",
]
