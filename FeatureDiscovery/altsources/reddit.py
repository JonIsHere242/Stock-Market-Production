"""
altsources/reddit.py  --  Quant/algo subreddits as feature ideas (best-effort).

Reddit increasingly 403s unauthenticated JSON. We try the public .json endpoint with a
browser-ish UA; on a hard block the source skips cleanly (set a real UA via REDDIT_UA, or wire
OAuth, to make it reliable). Signal-to-noise here is low -- the downstream gate is the filter.
"""

from __future__ import annotations

import os

from . import _base as B

NAME = "reddit"
_API = "https://www.reddit.com/r/{sub}/top.json?t=year&limit=50"

DEFAULT_SUBS = ["algotrading", "quant", "quantfinance"]


def fetch(session, *, limit, reddit_subs=None, **_) -> list[dict]:
    subs = reddit_subs or DEFAULT_SUBS
    ua = os.environ.get("REDDIT_UA",
                        "Mozilla/5.0 (compatible; feature-discovery/0.3; research)")
    blocked = 0
    out: list[dict] = []
    for sub in subs:
        try:
            data = B.get(session, _API.format(sub=sub),
                         headers={"User-Agent": ua}, timeout=30).json()
        except Exception as exc:                   # noqa: BLE001
            blocked += 1
            print(f"      [warn] reddit r/{sub} failed: {exc}")
            continue
        for child in data.get("data", {}).get("children", []):
            d = child.get("data", {})
            title = d.get("title", "")
            text = f"{title}. {d.get('selftext','')}".strip()
            if len(text) < 80:                     # link-only posts carry no method text
                continue
            out.append({"native_id": d.get("id", title), "title": B.clean(title),
                        "url": "https://www.reddit.com" + d.get("permalink", ""),
                        "text": text, "source": NAME})
            if len(out) >= limit:
                return out
    if blocked == len(subs) and not out:
        raise B.SourceUnavailable("reddit blocked all requests (403); set REDDIT_UA or wire OAuth")
    return out
