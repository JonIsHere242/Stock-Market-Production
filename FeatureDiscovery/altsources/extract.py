"""
altsources/extract.py  --  The LLM extraction spine for speculative ingestion.

Takes ANY raw text (repo README, video transcript, blog post, Reddit thread, a pasted essay
about a Brazilian Rust player wiping a base) and asks the locally-authenticated `claude` CLI
(`claude -p`, headless, runs on your Max subscription -- NO API key / NO credits) whether it
can be mapped to a method computable from daily OHLCV (+index/VIX/PIT-fundamentals) as a
per-stock or cross-sectional feature. If yes, it emits a compact "mock paper" (title+abstract)
in quant-method vocabulary so the existing relevance_score gate can triage it and
generate_feature can codegen it.

The prompt deliberately ALLOWS structural-analogy mapping (cascades, networks, regimes,
exhaustion, clustering) so the net is absurdly wide -- but it HARD-FORBIDS methods that need
data we don't have, and the downstream leakage / marginal-top-decile / multi-seed gates remain
the real filter. Garbage in is cheap: the gate throws it out, the only cost is one $0 call.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent     # repo root (cwd for claude)

_PROMPT = """You mine ARBITRARY text for QUANT TRADING FEATURE IDEAS.

Decide whether the CONTENT below describes -- or can be CREATIVELY but HONESTLY mapped by
structural analogy to -- a method computable as a NUMERIC FEATURE from DAILY stock data:
open, high, low, close, volume (+ optionally a market index / VIX, + point-in-time
fundamentals). The feature is either a per-stock time-series signal or a cross-sectional
signal (a value comparable across stocks each day).

You MAY map non-finance content by its STRUCTURE, e.g.:
  - a game about cascading base collapse -> a cascade / self-excitation / avalanche feature
    on returns or volume (Hawkes-like clustering).
  - a network of alliances / raids -> a rolling correlation-network centrality feature.
  - troop morale / exhaustion -> a trend-exhaustion / mean-reversion pressure feature.
Be imaginative about the structure, but the RESULTING method MUST be genuinely computable
from OHLCV alone, causally (no future data).

HARD RULE -> answer computable=false if the only viable method needs data we do NOT have:
limit order book, tick/intraday microstructure, order flow, bid/ask, news/sentiment/text,
the options surface or greeks, satellite / alt-data, on-chain/blockchain.

Respond with ONLY one JSON object, no prose:
{"computable": true or false,
 "title": "concise feature-method title",
 "abstract": "150-300 words: the INPUTS (which OHLCV/index series), the exact TRANSFORM
   (windows, normalisation, ranking), and the SIGNAL it yields (time-series or cross-sectional).
   Use precise quant/method vocabulary (rolling, z-score, entropy, visibility graph, Hawkes,
   regime, quantile, etc.) so it reads like a feature spec.",
 "method_family": "one of: momentum, reversal, volatility, volume, entropy, network, regime,
   fractal, spectral, microstructure-proxy, seasonality, cross-sectional, other",
 "mapping_note": "one line: how you mapped the source to this method (note if analogical)"}
If nothing computable applies, respond exactly {"computable": false}.

SOURCE: %(source)s -- %(title)s
%(url)s
CONTENT:
%(content)s
"""


def _claude_exe() -> str:
    for name in ("claude", "claude.cmd"):
        found = shutil.which(name)
        if found:
            return found
    fallback = Path(os.environ.get("APPDATA", "")) / "npm" / "claude.cmd"
    return str(fallback) if fallback.exists() else "claude"


def _run_claude(prompt: str, model: str | None, timeout: int) -> str:
    cmd = [_claude_exe(), "-p", "--output-format", "text"]
    if model:
        cmd += ["--model", model]
    proc = subprocess.run(
        cmd, input=prompt, capture_output=True, text=True,
        encoding="utf-8", errors="replace",      # claude emits UTF-8; Windows locale mangles it
        timeout=timeout, cwd=str(ROOT),
    )
    return proc.stdout or ""


def _parse_json(raw: str) -> dict | None:
    """Pull the first balanced {...} object out of claude's text output."""
    if not raw:
        return None
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start:i + 1])
                except json.JSONDecodeError:
                    # tolerate trailing commas / smart quotes
                    blob = re.sub(r",\s*}", "}", raw[start:i + 1]).replace("“", '"').replace("”", '"')
                    try:
                        return json.loads(blob)
                    except json.JSONDecodeError:
                        return None
    return None


def extract_method(raw_text: str, *, source: str, title: str = "", url: str = "",
                   model: str | None = "sonnet", timeout: int = 180,
                   max_chars: int = 6000) -> dict | None:
    """
    Run the extraction. Returns the parsed dict (computable/title/abstract/method_family/
    mapping_note) when computable, else None. Never raises on a bad LLM response -- a parse
    failure or computable=false just yields None (skip).
    """
    content = (raw_text or "").strip()
    if len(content) < 40:
        return None
    if len(content) > max_chars:
        content = content[:max_chars] + " ...[truncated]"
    prompt = _PROMPT % {"source": source, "title": title or "(untitled)",
                        "url": url or "", "content": content}
    try:
        out = _run_claude(prompt, None if model in (None, "default") else model, timeout)
    except subprocess.TimeoutExpired:
        return None
    parsed = _parse_json(out)
    if not parsed or not parsed.get("computable"):
        return None
    if not parsed.get("title") or not parsed.get("abstract"):
        return None
    parsed["_model"] = model or "default"
    return parsed
