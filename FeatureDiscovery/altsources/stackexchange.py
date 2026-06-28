"""
altsources/stackexchange.py  --  Quantitative Finance Stack Exchange (keyless).

WHY THIS SOURCE (beyond the obvious slop)
-----------------------------------------
quant.stackexchange.com is a high-density, PEER-VOTED corpus of *method* discussion: how
practitioners actually compute a signal, the gotchas, and the closed-form. Unlike a paper
abstract (which sells a result) a top-voted Q&A usually contains the exact transform. Sorting
by votes is a free quality filter -- the crowd already ranked implementability for us.

The Stack Exchange API is keyless for modest volume (gzipped JSON, generous anon quota), so this
fits the alt-source contract cleanly: pull top-voted questions (with body) for a set of
method-bearing tags, hand the raw text to extract.py, which mints a mock paper if an
OHLCV-computable feature can be distilled.

USAGE (via fetch_alt.py)
  python FeatureDiscovery/fetch_alt.py --sources stackexchange
  python FeatureDiscovery/fetch_alt.py --sources stackexchange --se-tags volatility momentum entropy
"""

from __future__ import annotations

from . import _base as B

NAME = "stackexchange"

API = "https://api.stackexchange.com/2.3/search/advanced"

# Tags that carry computable, OHLCV-mappable methods. Microstructure/options-heavy tags are
# deliberately omitted -- the downstream relevance gate would cap them as needs-external-data.
DEFAULT_TAGS = [
    "time-series", "volatility", "returns", "technical-analysis", "factor-models",
    "signal-processing", "momentum", "mean-reversion", "forecasting", "correlation",
    "trend", "statistics", "feature-engineering", "machine-learning", "risk",
]


def _strip(html_text: str) -> str:
    return B.html_to_text(html_text or "")


def fetch(session, *, limit, se_tags=None, per_tag=8, se_min_score=5, **_) -> list[dict]:
    tags = se_tags if se_tags else DEFAULT_TAGS
    out: list[dict] = []
    seen: set[str] = set()
    for tag in tags:
        params = {
            "order": "desc", "sort": "votes", "tagged": tag, "site": "quant",
            "pagesize": per_tag, "filter": "withbody",  # withbody = include question body
        }
        try:
            data = B.get(session, _url(API, params), timeout=40).json()
        except Exception as exc:                       # noqa: BLE001 -- one tag can't kill the source
            print(f"      [warn] stackexchange tag {tag!r} failed: {exc}")
            continue
        for it in data.get("items", []):
            qid = str(it.get("question_id") or "")
            if not qid or qid in seen:
                continue
            if int(it.get("score", 0)) < se_min_score:
                continue
            seen.add(qid)
            title = B.clean(it.get("title", ""))
            body = _strip(it.get("body", ""))
            tagline = ", ".join(it.get("tags", []))
            text = f"{title}. [tags: {tagline}] {body}".strip()
            if len(text) < 80:
                continue
            out.append({
                "native_id": qid,
                "title": title,
                "url": it.get("link", f"https://quant.stackexchange.com/q/{qid}"),
                "text": B.truncate(text, 6000),
                "source": NAME,
            })
            if len(out) >= limit:
                return out
    return out


def _url(base: str, params: dict) -> str:
    from urllib.parse import urlencode
    return f"{base}?{urlencode(params)}"
