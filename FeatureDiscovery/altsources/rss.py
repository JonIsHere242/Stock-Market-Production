"""
altsources/rss.py  --  Quant blogs & aggregators via RSS (keyless).

The flagship is Quantocracy -- a human-curated daily mashup of quant-blog links, so it's a
high-density, pre-filtered firehose. Also ships a few direct quant-blog feeds, and accepts
arbitrary feeds via --rss-feeds. Item summaries usually carry enough method gist for the
extractor; --rss-fulltext fetches each article page for the full body.
"""

from __future__ import annotations

from . import _base as B

NAME = "rss"

DEFAULT_FEEDS = {
    # --- aggregators / quant-blogs (curated link firehoses) ---
    "quantocracy":   "https://quantocracy.com/feed/",
    "alphaarchitect": "https://alphaarchitect.com/feed/",
    "robotwealth":   "https://robotwealth.com/feed/",
    "quantstart":    "https://www.quantstart.com/rss/",
    "quantinsti":    "https://blog.quantinsti.com/rss/",
    # --- practitioner / institutional research (denser method content, less picked-over) ---
    # These shops publish replicable factor/signal research with explicit constructions -- a
    # high-yield vein the academic sources miss. Feeds that 404 just warn-and-skip.
    "alphascientist": "https://alphascientist.com/feed.xml",
    "quantitativo":   "https://www.quantitativo.com/feed",
    "newfound":       "https://blog.thinknewfound.com/feed/",
    "factorresearch": "https://www.factorresearch.com/feed",
    "mlfin":          "https://www.mlfinlab.com/feed",
    "quantpedia":     "https://quantpedia.com/feed/",
    "twosigma":       "https://www.twosigma.com/feed/",
    "raposa":         "https://raposa.trade/blog/rss/",
    "concretum":      "https://www.concretumgroup.com/blog/feed/",
}


def fetch(session, *, limit, rss_feeds=None, rss_fulltext=False,
          per_feed=15, **_) -> list[dict]:
    feeds = rss_feeds if rss_feeds else DEFAULT_FEEDS
    if isinstance(feeds, (list, tuple)):           # bare URLs from CLI
        feeds = {f"feed{i}": u for i, u in enumerate(feeds)}

    out: list[dict] = []
    for name, url in feeds.items():
        try:
            items = B.parse_feed(B.get(session, url, timeout=40).text)
        except Exception as exc:                   # noqa: BLE001
            print(f"      [warn] rss {name} failed: {exc}")
            continue
        for it in items[:per_feed]:
            text = it["summary"]
            if rss_fulltext and it["link"]:
                try:
                    text = B.html_to_text(B.get(session, it["link"], timeout=30).text)
                except Exception:                  # noqa: BLE001
                    pass
            text = f"{it['title']}. {text}".strip()
            if len(text) < 60:
                continue
            out.append({"native_id": it["link"] or it["title"], "title": it["title"],
                        "url": it["link"], "text": text, "source": NAME})
            if len(out) >= limit:
                return out
    return out
