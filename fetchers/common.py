"""Shared utilities for all data fetchers.

Centralizes HTTP session config, rate limiting, paths, and atomic writes
so per-source fetchers stay focused on the URL contract of their source.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# On Windows cp1252 default stdout fails on Unicode (arrows, etc.) used in our
# log messages. Reconfigure to UTF-8 once, here, so every fetcher inherits it.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, Exception):
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "Data"

SEC_USER_AGENT = "Jonathan Bellmont masamunex9000@gmail.com"
SEC_HOST_HEADER = "www.sec.gov"
SEC_RATE_LIMIT_HZ = 8.0  # SEC permits 10/s; we stay well under


class RateLimiter:
    """Token-bucket-ish: at most `hz` calls per second across all threads."""

    def __init__(self, hz: float):
        self.min_interval = 1.0 / hz
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            if now < self._next_allowed:
                time.sleep(self._next_allowed - now)
                now = time.monotonic()
            self._next_allowed = now + self.min_interval


_sec_limiter = RateLimiter(SEC_RATE_LIMIT_HZ)


def make_session(host: str | None = None, sec: bool = False) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    if sec:
        s.headers.update({
            "User-Agent": SEC_USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
        })
        if host:
            s.headers["Host"] = host
    else:
        s.headers["User-Agent"] = "Stock-Market-Research/1.0 (masamunex9000@gmail.com)"
    return s


def sec_get(session: requests.Session, url: str, **kwargs) -> requests.Response:
    """SEC-rate-limited GET. Use for all *.sec.gov requests."""
    _sec_limiter.wait()
    return session.get(url, timeout=60, **kwargs)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(data)
    tmp.replace(path)


@contextmanager
def atomic_stream(path: Path):
    """Stream large downloads to a .tmp file then atomically rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    f = tmp.open("wb")
    try:
        yield f
        f.close()
        tmp.replace(path)
    except Exception:
        f.close()
        if tmp.exists():
            tmp.unlink()
        raise


def stream_download(session: requests.Session, url: str, dest: Path,
                    rate_limited_get=None, chunk: int = 1 << 20,
                    headers: dict | None = None) -> int:
    """Download `url` to `dest` with streaming. Returns bytes written.

    `rate_limited_get` is an optional callable(session, url, **kw) so SEC
    downloads can share the global rate limiter; defaults to session.get.
    """
    getter = rate_limited_get or (lambda s, u, **kw: s.get(u, timeout=120, **kw))
    with getter(session, url, stream=True, headers=headers or {}) as r:
        r.raise_for_status()
        written = 0
        with atomic_stream(dest) as f:
            for blk in r.iter_content(chunk_size=chunk):
                if blk:
                    f.write(blk)
                    written += len(blk)
        return written


def file_age_hours(path: Path) -> float:
    if not path.exists():
        return float("inf")
    return (time.time() - path.stat().st_mtime) / 3600.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)
