"""
altsources/youtube.py  --  Quant-YouTube transcripts as feature ideas.

Keyless: a channel's latest uploads come from its public RSS feed
(youtube.com/feeds/videos.xml?channel_id=UC...); transcripts come from the free
`youtube-transcript-api` (no key, auto-captions OK). Channels can be given as raw UC ids,
@handles, or full URLs -- handles/URLs are resolved by scraping the channelId from the page.

Requires `pip install youtube-transcript-api`; without it the source skips cleanly.
"""

from __future__ import annotations

import re

from . import _base as B

NAME = "youtube"
_RSS = "https://www.youtube.com/feeds/videos.xml?channel_id={cid}"

# Curated quant channels (name -> channel_id). Extend via --youtube-channels (id/@handle/url).
DEFAULT_CHANNELS = {
    "neurotrader": "UCSh87zxGNu8q8iOInRK6E9w",
}

_VID = re.compile(r"<yt:videoId>(.*?)</yt:videoId>")
_TITLE = re.compile(r"<media:title>(.*?)</media:title>")
_CID = re.compile(r'"(?:channelId|externalId|browseId)":"(UC[\w-]{22})"')


def _resolve_channel_id(session, ref: str) -> str | None:
    if re.fullmatch(r"UC[\w-]{22}", ref):
        return ref
    url = ref if ref.startswith("http") else f"https://www.youtube.com/{ref.lstrip('/')}"
    try:
        m = _CID.search(B.get(session, url).text)
        return m.group(1) if m else None
    except Exception:                              # noqa: BLE001
        return None


def _transcript(video_id: str) -> str:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError as exc:
        raise B.SourceUnavailable("pip install youtube-transcript-api") from exc
    try:
        # API shape differs across versions; support both.
        if hasattr(YouTubeTranscriptApi, "get_transcript"):
            chunks = YouTubeTranscriptApi.get_transcript(video_id)
        else:
            chunks = YouTubeTranscriptApi().fetch(video_id).to_raw_data()
        return " ".join(c["text"] for c in chunks if c.get("text"))
    except Exception:                              # noqa: BLE001 -- no captions / blocked
        return ""


def _channel_videos(session, cid: str, per_channel: int) -> list[tuple[str, str]]:
    xml = B.get(session, _RSS.format(cid=cid)).text
    ids = _VID.findall(xml)
    titles = _TITLE.findall(xml)
    return list(zip(ids, titles))[:per_channel]


def fetch(session, *, limit, youtube_channels=None, per_channel=8, **_) -> list[dict]:
    if youtube_channels:
        channels = {}
        for ref in youtube_channels:
            cid = _resolve_channel_id(session, ref)
            if cid:
                channels[ref] = cid
    else:
        channels = dict(DEFAULT_CHANNELS)

    out: list[dict] = []
    for name, cid in channels.items():
        try:
            vids = _channel_videos(session, cid, per_channel)
        except Exception as exc:                   # noqa: BLE001
            print(f"      [warn] youtube channel {name} failed: {exc}")
            continue
        for vid, title in vids:
            text = _transcript(vid)                # raises SourceUnavailable if lib missing
            if len(text) < 200:                    # no usable captions
                continue
            out.append({"native_id": vid, "title": B.clean(title),
                        "url": f"https://www.youtube.com/watch?v={vid}",
                        "text": f"{title}. {text}", "source": NAME})
            if len(out) >= limit:
                return out
    return out
