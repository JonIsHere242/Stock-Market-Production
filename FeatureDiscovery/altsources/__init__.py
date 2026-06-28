"""
altsources package  --  speculative ("alt-paper") ingestion sources.

Each adapter exposes:
    NAME : str
    fetch(session, *, limit, **opts) -> list[{"native_id","title","url","text","source"}]
returning RAW text items. The orchestrator (fetch_alt.py) runs each item through
extract.py (claude -p) to mint a mock paper, then merges into Data/PaperFeed/papers.parquet.

The `web` source is the universal catch-all (any URL / file / inline text).
"""

from __future__ import annotations

from . import github, hackernews, reddit, rss, stackexchange, web, youtube

REGISTRY = {m.NAME: m for m in (github, youtube, rss, hackernews, reddit, stackexchange, web)}

# Run by default when --sources is omitted. `web` is opt-in (needs explicit urls/files).
DEFAULT_SOURCES = ["github", "youtube", "rss", "hackernews", "reddit", "stackexchange"]
