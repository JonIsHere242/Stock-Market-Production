"""
altsources/hackernews.py  --  Hacker News stories about quant/algo trading (keyless Algolia API).

HN surfaces practitioner write-ups, Show HN tools, and "how I built my strategy" threads. The
Algolia search API is free and returns title + story_text; --hn-fulltext follows the story URL.
"""

from __future__ import annotations

from urllib.parse import quote

from . import _base as B

NAME = "hackernews"
_API = "https://hn.algolia.com/api/v1/search?query={q}&tags=story&hitsPerPage=20"

DEFAULT_QUERIES = ["quant trading strategy", "algorithmic trading signal",
                   "stock prediction features", "technical indicator backtest",
                   "market microstructure", "time series forecasting trading"]


def fetch(session, *, limit, hn_queries=None, hn_fulltext=False, **_) -> list[dict]:
    queries = hn_queries or DEFAULT_QUERIES
    out: dict[str, dict] = {}
    for q in queries:
        try:
            data = B.get(session, _API.format(q=quote(q)), timeout=30).json()
        except Exception as exc:                   # noqa: BLE001
            print(f"      [warn] hn query {q!r} failed: {exc}")
            continue
        for h in data.get("hits", []):
            oid = h.get("objectID", "")
            title = h.get("title") or h.get("story_title") or ""
            url = h.get("url") or f"https://news.ycombinator.com/item?id={oid}"
            text = h.get("story_text") or ""
            if hn_fulltext and h.get("url"):
                try:
                    text = B.html_to_text(B.get(session, h["url"], timeout=25).text)
                except Exception:                  # noqa: BLE001
                    pass
            text = f"{title}. {text}".strip()
            if title and len(text) >= 40:
                out.setdefault(oid, {"native_id": oid, "title": B.clean(title),
                                     "url": url, "text": text, "source": NAME})
        if len(out) >= limit:
            break
    return list(out.values())[:limit]
