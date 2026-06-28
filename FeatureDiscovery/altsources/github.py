"""
altsources/github.py  --  Mine quant GitHub repos (README + description) as feature ideas.

For code-bearing creators (e.g. neurotrader888) the repo IS the method -- sharper than a
video transcript. Keyless path: list a curated set of quant USERS' repos and pull each README
(both endpoints work unauthenticated at low volume). With GITHUB_TOKEN set, also run repo
SEARCH across topics so the net widens to the whole quant GitHub.
"""

from __future__ import annotations

import os

from . import _base as B

NAME = "github"
_API = "https://api.github.com"

# Curated quant users whose repos map onto OHLCV feature construction. Extend via --github-users.
DEFAULT_USERS = ["neurotrader888", "je-suis-tm"]

# Used only when GITHUB_TOKEN is present (search needs auth).
SEARCH_QUERIES = ["quant trading strategy", "alpha factor signal stocks",
                  "technical indicator python", "time series features trading"]


def _headers():
    h = {"Accept": "application/vnd.github+json"}
    tok = os.environ.get("GITHUB_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def _readme(session, full_name: str) -> str:
    try:
        r = B.get(session, f"{_API}/repos/{full_name}/readme",
                  headers={**_headers(), "Accept": "application/vnd.github.raw+json"})
        return r.text
    except Exception:                              # noqa: BLE001 -- no README is fine
        return ""


def _user_repos(session, user: str, limit: int) -> list[dict]:
    r = B.get(session, f"{_API}/users/{user}/repos?per_page=100&sort=updated",
              headers=_headers())
    out = []
    for repo in r.json()[:limit]:
        full = repo.get("full_name", "")
        desc = repo.get("description") or ""
        readme = _readme(session, full)
        text = f"{desc}\n\n{readme}".strip()
        if len(text) < 40:
            continue
        out.append({"native_id": full, "title": full, "url": repo.get("html_url", ""),
                    "text": text, "source": NAME})
    return out


def _search(session, limit: int) -> list[dict]:
    out = []
    for q in SEARCH_QUERIES:
        try:
            r = B.get(session, f"{_API}/search/repositories?q={q}&sort=stars&per_page=10",
                      headers=_headers())
        except Exception:                          # noqa: BLE001
            continue
        for repo in r.json().get("items", []):
            full = repo.get("full_name", "")
            text = f"{repo.get('description') or ''}\n\n{_readme(session, full)}".strip()
            if len(text) >= 40:
                out.append({"native_id": full, "title": full, "url": repo.get("html_url", ""),
                            "text": text, "source": NAME})
        if len(out) >= limit:
            break
    return out[:limit]


def fetch(session, *, limit, github_users=None, **_) -> list[dict]:
    users = github_users or DEFAULT_USERS
    out: dict[str, dict] = {}
    for u in users:
        try:
            for item in _user_repos(session, u, limit):
                out.setdefault(item["native_id"], item)
        except Exception as exc:                   # noqa: BLE001
            print(f"      [warn] github user {u} failed: {exc}")
        if len(out) >= limit:
            break
    if os.environ.get("GITHUB_TOKEN") and len(out) < limit:
        for item in _search(session, limit - len(out)):
            out.setdefault(item["native_id"], item)
    return list(out.values())[:limit]
