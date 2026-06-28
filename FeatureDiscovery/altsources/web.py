"""
altsources/web.py  --  The universal catch-all: ingest LITERALLY ANYTHING.

This is the "alt alt alt" door. Point it at any URL (a blog, a forum thread, a transcript page,
a Rust-strategy wiki), any local text file (a pasted YouTube transcript, an essay, a PDF dumped
to .txt), or inline text, and it becomes a candidate -- the extractor tries to find a computable
OHLCV feature analogy in it. Breadth is unbounded by design; the downstream gates are the filter.

Driven entirely by opts (no defaults): web_urls / web_files / web_texts. Nothing to fetch if
none are given (the orchestrator just reports 0).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from . import _base as B

NAME = "web"


def _hid(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "replace")).hexdigest()[:12]


def fetch(session, *, limit, web_urls=None, web_files=None, web_texts=None, **_) -> list[dict]:
    out: list[dict] = []

    for url in (web_urls or []):
        try:
            text = B.html_to_text(B.get(session, url, timeout=40).text)
        except Exception as exc:                   # noqa: BLE001
            print(f"      [warn] web url {url} failed: {exc}")
            continue
        if len(text) >= 60:
            out.append({"native_id": _hid(url), "title": url, "url": url,
                        "text": text, "source": NAME})

    for fp in (web_files or []):
        try:
            text = Path(fp).read_text(encoding="utf-8", errors="replace")
        except Exception as exc:                   # noqa: BLE001
            print(f"      [warn] web file {fp} failed: {exc}")
            continue
        if len(text) >= 60:
            out.append({"native_id": _hid(fp), "title": Path(fp).name, "url": str(fp),
                        "text": text, "source": NAME})

    for i, text in enumerate(web_texts or []):
        if text and len(text) >= 60:
            out.append({"native_id": _hid(text), "title": f"inline-{i}", "url": "",
                        "text": text, "source": NAME})

    return out[:limit] if limit else out
