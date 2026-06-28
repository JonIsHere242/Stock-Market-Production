"""
altsources/_base.py  --  Shared plumbing for the SPECULATIVE ("alt-paper") ingestion sources.

Difference from the academic `sources/` package: those return clean title+abstract you can
score directly. Alt sources return RAW, messy text (a repo README, a YouTube transcript, a
blog post, a Reddit thread, literally any URL or pasted essay). That raw text is fed to the
LLM extraction spine (extract.py), which decides whether it can be mapped to an OHLCV-computable
feature and, if so, emits a compact "mock paper" (title+abstract) in the SAME papers.parquet
schema -- flagged is_mock=True -- so it rides the existing relevance_score -> generate_feature
-> validate_feature rails like any other paper.

So an alt-source adapter's job is ONLY: produce a list of raw items
    {"native_id","title","url","text","source"}
The orchestrator (fetch_alt.py) runs each through extract.py and make_mock_record.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import requests

# Reuse the academic schema builder so mock papers share one store layout.
_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))
from sources import _base as academic          # noqa: E402

USER_AGENT = academic.USER_AGENT
SourceUnavailable = academic.SourceUnavailable
clean = academic.clean


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def get(session: requests.Session, url: str, *, headers=None, timeout=30, tries=3):
    h = {"User-Agent": USER_AGENT}
    if headers:
        h.update(headers)
    last = None
    for attempt in range(1, tries + 1):
        try:
            r = session.get(url, headers=h, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as exc:               # noqa: BLE001
            last = exc
    raise RuntimeError(f"GET failed after {tries}: {last}")


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_TAG = re.compile(r"(?is)<(script|style)[^>]*>.*?</\1>")
_ANY = re.compile(r"(?s)<[^>]+>")


def html_to_text(html: str) -> str:
    """Crude but robust HTML -> text: drop script/style, strip tags, unescape, collapse."""
    if not html:
        return ""
    html = _TAG.sub(" ", html)
    return clean(_ANY.sub(" ", html))


def truncate(text: str, max_chars: int) -> str:
    text = text or ""
    return text if len(text) <= max_chars else text[:max_chars] + " ...[truncated]"


# ---------------------------------------------------------------------------
# RSS / Atom (stdlib only) -- powers the Quantocracy + generic-blog source
# ---------------------------------------------------------------------------

_CONTENT_NS = "{http://purl.org/rss/1.0/modules/content/}encoded"
_ATOM = "{http://www.w3.org/2005/Atom}"


def parse_feed(xml_text: str) -> list[dict]:
    """Return [{title, link, summary}] from an RSS 2.0 or Atom feed."""
    out: list[dict] = []
    try:
        root = ET.fromstring(xml_text.encode("utf-8", "replace"))
    except ET.ParseError:
        return out
    # RSS 2.0
    for item in root.iter("item"):
        title = item.findtext("title", "")
        link = item.findtext("link", "")
        summary = item.findtext("description", "") or ""
        enc = item.find(_CONTENT_NS)
        if enc is not None and (enc.text or ""):
            summary = enc.text                      # full HTML body if present
        out.append({"title": clean(title), "link": clean(link),
                    "summary": html_to_text(summary)})
    if out:
        return out
    # Atom
    for entry in root.iter(f"{_ATOM}entry"):
        title = entry.findtext(f"{_ATOM}title", "")
        link_el = entry.find(f"{_ATOM}link")
        link = link_el.get("href", "") if link_el is not None else ""
        summary = (entry.findtext(f"{_ATOM}summary", "")
                   or entry.findtext(f"{_ATOM}content", "") or "")
        out.append({"title": clean(title), "link": clean(link),
                    "summary": html_to_text(summary)})
    return out


# ---------------------------------------------------------------------------
# Mock-paper record
# ---------------------------------------------------------------------------

def make_mock_record(*, source: str, native_id: str, title: str, abstract: str,
                     url: str = "", method_family: str = "", mapping_note: str = "",
                     extractor_model: str = "", raw_excerpt: str = "") -> dict:
    """
    Build a papers.parquet row for an LLM-extracted 'mock paper'. Same schema as the
    academic sources (via make_record) plus alt-only provenance columns so a human can
    audit where a speculative feature idea actually came from.
    """
    rec = academic.make_record(
        source=source, native_id=native_id,
        title=title, abstract=abstract,
        primary_category=f"alt.{method_family}" if method_family else "alt",
        categories=method_family,
        abs_url=url,
    )
    rec["is_mock"] = True
    rec["raw_url"] = url
    rec["mapping_note"] = mapping_note
    rec["extractor_model"] = extractor_model
    rec["raw_excerpt"] = truncate(raw_excerpt, 600)
    return rec
