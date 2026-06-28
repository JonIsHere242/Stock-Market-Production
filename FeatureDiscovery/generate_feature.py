"""
generate_feature.py  --  Automatic paper -> feature-block codegen (stage 3 of the pipeline).

WHAT IT DOES
------------
End-to-end, no API key / no API credits required:
  1. DETECT  -- load the scored paper store (arxiv_fetch.py + relevance_score.py).
  2. FILTER  -- pick the top-ranked, non-junk, OHLCV-implementable paper not yet done.
  3. GENERATE-- ask the locally-authenticated `claude` CLI (your Max subscription, headless
                `claude -p`) to write ONE feature block matching the FeatureTemplates contract.
  4. WRITE   -- save it to FeatureTemplates/<name>.py.
  5. VALIDATE-- import it, run compute() on a real ticker, check the columns are populated,
                non-constant, and report a quick Spearman IC vs next-day return.

This is a PROTOTYPE of the codegen stage. The generated block is a CANDIDATE only -- it has
NOT passed the leakage / top-decile / multi-seed gates that decide whether a feature is real.
Treat the output as "worth a human look", not "ship it".

USAGE
-----
  python FeatureDiscovery/generate_feature.py                 # top candidate
  python FeatureDiscovery/generate_feature.py --paper_id 2506.01234
  python FeatureDiscovery/generate_feature.py --model opus --validate_ticker AAPL

AUTH
----
Uses whatever `claude` is logged into (Claude Code subscription). No ANTHROPIC_API_KEY needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT       = Path(__file__).resolve().parent.parent
TEMPLATES  = ROOT / "FeatureTemplates"
STORE      = ROOT / "Data" / "PaperFeed" / "papers.parquet"
GEN_LOG    = ROOT / "Data" / "PaperFeed" / "codegen_log.txt"
GEN_DONE   = ROOT / "Data" / "PaperFeed" / "generated.json"
PRICE_DIR  = ROOT / "Data" / "PriceData"

ALLOWED_IMPORTS = ["pandas", "numpy", "scipy", "networkx", "math", "typing",
                   # blessed for the helper-import idiom (loading _indexes / _fundamentals by path)
                   "importlib", "pathlib"]

# The contract handed to the model. Mirrors FeatureTemplates/__example_template.py plus the
# sandbox constraints, distilled so the model has everything it needs in one prompt.
CONTRACT = f"""\
You are generating exactly ONE Python feature-block file for a modular feature pipeline.

THE FILE MUST DEFINE EXACTLY TWO MODULE-LEVEL OBJECTS:
  METADATA = {{"name","description","requires","produces","tags","version","author"}}
  def compute(df: pd.DataFrame) -> pd.DataFrame

HOW compute() IS CALLED:
  - df is ONE stock at a time, ascending by Date, with columns:
      Date, Ticker, Open, High, Low, Close, Volume
  - You may ONLY ADD the columns listed in METADATA["produces"].
  - NEVER modify, drop, rename, sort, or reindex existing columns. Always `return df`.

HARD RULES (the sandbox enforces these -- code that violates them is rejected):
  - Imports allowed ONLY from: {", ".join(ALLOWED_IMPORTS)}. No other imports. No file/network/IO.
  - No lookahead/future leakage: use only current-and-past data. Rolling windows are fine.
    NEVER use df[...].shift(-k) (negative shift) or full-series stats that peek ahead.
  - Stateless: no globals mutated, no prints, no logging, no training, no randomness without a
    fixed seed. Deterministic.
  - Must run fast on ~700 rows (well under 100 ms). Prefer vectorised pandas/numpy; avoid
    O(n^2) python loops over every row where a rolling op will do.
  - Column names: lowercase snake_case, no leading digit, no '%', no spaces/parens.
    Prefix the family clearly so names are unique (e.g. tda_, topo_, <concept>_).
  - Leading NaNs from rolling windows are expected and fine -- do NOT fill the whole frame.

OPTIONAL: SEC FUNDAMENTALS (point-in-time)
  Besides OHLCV you MAY use point-in-time SEC fundamentals via the shared helper
  FeatureTemplates/_fundamentals.py (import it BY FILE PATH, exactly like vix_features.py imports
  _indexes.py -- do NOT `from FeatureTemplates import ...`):

      import importlib.util as _ilu
      from pathlib import Path as _Path
      _spec = _ilu.spec_from_file_location("_fundamentals",
                  _Path(__file__).resolve().parent / "_fundamentals.py")
      _fundamentals = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_fundamentals)

  Then inside compute():
      df = _fundamentals.as_of(df, fields=["net_margin", "roe", "eps_diluted_ttm"])
  as_of() does a BACKWARD merge_asof keyed on filed_date (the date the number became public), so
  each row only sees filings already public by that day -- this is lookahead-safe and is what lets
  the feature pass the causality gate. It returns the requested fields as columns named
  `fund_<field>` (e.g. fund_net_margin). NEVER merge fundamentals on the period-end date -- that is
  future leakage and the gate will FAIL you. Available fields include (call canonical_fields() to
  see all): revenue, net_income, gross_profit, operating_income, assets, equity, cash,
  long_term_debt, shares_outstanding, eps_basic, eps_diluted, operating_cash_flow, capex; TTM
  rollups (revenue_ttm, net_income_ttm, eps_diluted_ttm, fcf_ttm, ...); ratios (gross_margin,
  operating_margin, net_margin, roe, roa, current_ratio, debt_to_equity, asset_turnover,
  book_value_per_share, sales_per_share). Price-relative multiples (P/E, P/B, P/S) you compute
  yourself: divide df['Close'] by the matching fund_* per-share field. The `fund_*` columns are
  SCRATCH -- compute your produced feature(s) from them, then drop any `fund_*` you are not listing
  in METADATA["produces"]. Coverage is partial (~84% of names; ETFs/foreign have none) -> leading
  and missing-coverage rows are NaN; that is expected, do not fill the whole frame.

FEASIBILITY (important):
  If the paper's method is fundamentally cross-sectional (ranks across many stocks at once),
  needs model training, or needs data beyond OHLCV (and beyond the SEC fundamentals helper above),
  you CANNOT implement it faithfully here. In that case, implement the closest PER-TICKER proxy
  (OHLCV +/- fundamentals) that captures the paper's core signal, and say so honestly in
  METADATA["description"] and METADATA["author"].

OUTPUT FORMAT:
  Output ONLY the file content inside a single ```python code fence. No prose before or after.
"""


def _log(msg: str) -> None:
    GEN_LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    with open(GEN_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _load_done() -> set[str]:
    if GEN_DONE.exists():
        try:
            return set(json.loads(GEN_DONE.read_text()))
        except Exception:
            return set()
    return set()


def _mark_done(arxiv_id: str) -> None:
    done = _load_done()
    done.add(arxiv_id)
    GEN_DONE.write_text(json.dumps(sorted(done), indent=2))


# ---------------------------------------------------------------------------
# 1-2. detect + filter
# ---------------------------------------------------------------------------

def select_paper(df: pd.DataFrame, paper_id: str | None) -> pd.Series | None:
    if "rel_score" not in df.columns:
        sys.exit("Papers not scored yet -- run relevance_score.py first.")
    if paper_id:
        hit = df[df["arxiv_id"] == paper_id]
        return hit.iloc[0] if len(hit) else None

    done = _load_done()
    cand = df.dropna(subset=["rel_score"])
    cand = cand[(~cand["needs_external_data"].astype(bool)) & (~cand["arxiv_id"].isin(done))]
    cand = cand.sort_values("rel_score", ascending=False)
    return cand.iloc[0] if len(cand) else None


def ensure_scored(df: pd.DataFrame) -> pd.DataFrame:
    """Re-score the store in-process so a fresh fetch's unscored rows can't be skipped.

    A bare arxiv_fetch appends rows with no rel_* columns; selecting before scoring
    silently drops the (often highest-value) new papers. Scoring is cheap + deterministic,
    so just (re)run it here to keep fetch -> generate self-contained.
    """
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location("relevance_score",
                                        Path(__file__).resolve().parent / "relevance_score.py")
    rs = _ilu.module_from_spec(spec)
    spec.loader.exec_module(rs)
    df = rs.score_store(df)
    df["rel_keep"] = df["rel_score"] >= 5.0
    df.to_parquet(STORE, index=False)
    return df


# ---------------------------------------------------------------------------
# 3. generate via the `claude` CLI (subscription auth, headless)
# ---------------------------------------------------------------------------

def _claude_exe() -> str:
    # On Windows the npm global ships claude.cmd; shutil.which resolves it via PATHEXT.
    for name in ("claude", "claude.cmd"):
        found = shutil.which(name)
        if found:
            return found
    fallback = Path(os.environ.get("APPDATA", "")) / "npm" / "claude.cmd"
    return str(fallback) if fallback.exists() else "claude"


def paper_block(paper: pd.Series) -> str:
    return ("PAPER TO IMPLEMENT:\n"
            f"Title: {paper.get('title','')}\n"
            f"Categories: {paper.get('categories','')}\n"
            f"Abstract: {paper.get('abstract','')}\n")


def idea_block(idea: str) -> str:
    return ("FEATURE IDEA TO IMPLEMENT (the user's own hypothesis -- treat it as the spec; the "
            "same OHLCV-only / no-lookahead / contract rules below still apply):\n"
            + idea.strip() + "\n")


def build_prompt(source_block: str) -> str:
    return CONTRACT + "\n\n" + source_block


def run_codegen(prompt: str, model: str | None, timeout: int) -> str:
    cmd = [_claude_exe(), "-p", "--output-format", "text"]
    if model:
        cmd += ["--model", model]
    _log(f"codegen: invoking {cmd[0]} (model={model or 'default'}, timeout={timeout}s)")
    proc = subprocess.run(
        cmd, input=prompt, capture_output=True, text=True,
        encoding="utf-8", errors="replace",   # claude emits UTF-8; Windows locale would mangle it
        timeout=timeout, cwd=str(ROOT),
    )
    if proc.returncode != 0:
        _log(f"codegen: claude exited {proc.returncode}: {proc.stderr[:400]}")
    return proc.stdout


def extract_code(raw: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", raw, re.S)
    code = (m.group(1) if m else raw).strip()
    return code


def derive_name(code: str) -> str | None:
    m = re.search(r'["\']name["\']\s*:\s*["\']([a-zA-Z_][a-zA-Z0-9_]*)["\']', code)
    return m.group(1) if m else None


def check_imports(code: str) -> list[str]:
    """Return any imported top-level modules outside the allowlist."""
    bad = []
    for line in code.splitlines():
        m = re.match(r"\s*(?:import|from)\s+([a-zA-Z0-9_\.]+)", line)
        if m:
            top = m.group(1).split(".")[0]
            if top not in ALLOWED_IMPORTS:
                bad.append(top)
    return sorted(set(bad))


# ---------------------------------------------------------------------------
# 5. validate  --  delegated to validate_feature.py (leakage + strategy-aware signal gate),
#                  invoked from main() after the block is written.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _load_validator():
    import importlib.util as _ilu
    vspec = _ilu.spec_from_file_location(
        "validate_feature", Path(__file__).resolve().parent / "validate_feature.py")
    vf = _ilu.module_from_spec(vspec)
    vspec.loader.exec_module(vf)
    return vf


def generate_one(source_id: str, source_block: str, display: str, args, vf) -> str:
    """codegen -> write -> gate -> quarantine-on-FAIL. Returns the verdict."""
    _log("=" * 70)
    _log(f"SOURCE  {display}")

    raw = run_codegen(build_prompt(source_block),
                      None if args.model == "default" else args.model, args.timeout)
    if not raw.strip():
        _log("codegen returned empty output -- skipping.")
        _mark_done(source_id)
        return "EMPTY"

    code = extract_code(raw)
    bad = check_imports(code)
    if bad:
        _log(f"  [WARN] disallowed imports present: {bad} (block may fail to load)")

    stem = args.out_name or derive_name(code) or f"feat_{source_id}"
    stem = re.sub(r"[^A-Za-z0-9_]", "_", stem)
    # Single-underscore prefix == UNPROVEN candidate: the framework skips it (not auto-discovered),
    # so a fresh generation can never leak into the model before it earns promotion.
    path = TEMPLATES / f"_{stem}.py"
    if path.exists():
        path = TEMPLATES / f"_{re.sub(r'[^A-Za-z0-9_]', '_', f'{stem}_{source_id}')}.py"
    path.write_text(code, encoding="utf-8")
    _log(f"WROTE candidate  {path.relative_to(ROOT)}  ({len(code.splitlines())} lines)  "
         f"[ _ = unproven, hidden from discovery ]")

    report = vf.validate_block(path, n=args.val_n, seed=13, verbose=True)
    verdict = report["verdict"]
    _log(f"VALIDATE  {verdict}")
    for r in report.get("reasons", []):
        _log(f"    {r}")

    if verdict == "FAIL":
        rej = ROOT / "FeatureDiscovery" / "_rejected"
        rej.mkdir(parents=True, exist_ok=True)
        dest = rej / path.name
        path.replace(dest)
        _log(f"QUARANTINED {path.name} -> {dest.relative_to(ROOT)} (leak/broken/redundant)")
    elif verdict == "PASS":
        live = path.name[1:]   # drop the leading underscore
        _log(f"PASS (quick gate). After the heavy backtest gate, PROMOTE to live with: "
             f"rename {path.name} -> {live}")

    _mark_done(source_id)
    return verdict


def main() -> None:
    ap = argparse.ArgumentParser(description="Automatic paper/idea -> feature-block codegen + gate")
    ap.add_argument("--paper_id", default=None, help="Force a specific arXiv id (paper mode)")
    ap.add_argument("--idea", default=None,
                    help="Generate from YOUR OWN idea text instead of a paper")
    ap.add_argument("--idea_file", default=None, help="Read the idea text from a file")
    ap.add_argument("--count", type=int, default=1,
                    help="How many to generate (papers: distinct papers; idea: variations)")
    ap.add_argument("--model", default="sonnet", help="claude model alias (sonnet|opus|default)")
    ap.add_argument("--timeout", type=int, default=300, help="codegen timeout seconds")
    ap.add_argument("--val_n", type=int, default=40, help="ticker sample for the validation gate")
    ap.add_argument("--out_name", default=None, help="override the block filename stem")
    args = ap.parse_args()

    vf = _load_validator()

    idea_text = None
    if args.idea_file:
        idea_text = Path(args.idea_file).read_text(encoding="utf-8")
    elif args.idea:
        idea_text = args.idea

    # ---- idea mode: codegen from the user's own hypothesis (no paper store needed) ----
    if idea_text is not None:
        base = "idea_" + hashlib.md5(idea_text.encode("utf-8")).hexdigest()[:8]
        for it in range(args.count):
            sid = base + (f"_{it + 1}" if args.count > 1 else "")
            disp = f"IDEA  {idea_text.strip()[:80]}  (variation {it + 1}/{args.count})"
            generate_one(sid, idea_block(idea_text), disp, args, vf)
        _log("DONE")
        return

    # ---- paper mode --------------------------------------------------------------------
    if not STORE.exists():
        sys.exit("No paper store -- run arxiv_fetch.py then relevance_score.py first.")
    df = pd.read_parquet(STORE)
    df = ensure_scored(df)   # keep fetch -> generate self-contained (re-score any new rows)

    made = 0
    for _ in range(args.count):
        paper = select_paper(df, args.paper_id)
        if paper is None:
            _log("No eligible paper found (all done or all flagged external-data).")
            break
        disp = (f"[{paper['rel_score']:.2f}] {paper['arxiv_id']}  {paper['title'][:70]}  | "
                f"{paper.get('rel_reasons', '')[:90]}")
        generate_one(paper["arxiv_id"], paper_block(paper), disp, args, vf)
        made += 1
        if args.paper_id:
            break
    _log(f"DONE ({made} generated)")


if __name__ == "__main__":
    main()
