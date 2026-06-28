"""
fetch_papers.py  --  Unified multi-source paper puller for the feature-discovery pipeline.

Supersedes running arxiv_fetch.py alone: it pulls from many academic sources (arXiv,
OpenAlex, Crossref + curated finance journals, DOAJ, EconBiz, NBER, OSF/SocArXiv, Zenodo,
HAL, and -- with a free key -- Semantic Scholar and CORE), normalises every result into the
SAME Data/PaperFeed/papers.parquet schema, dedupes ACROSS sources (by DOI, else title),
and re-runs the cheap relevance triage so the store is immediately consistent.

Why a wide net: arXiv alone is a category dump that drags in generic ML. The other sources
support finance-specific SEARCH, so the net stays on-topic (see DEFAULT_QUERIES in
sources/_base.py). One paper that lands in several sources is kept once, preferring the
richest copy (abstract present) from the highest-priority source.

USAGE
-----
  # Everything (keyless sources always run; key-optional skip cleanly if no key set)
  python FeatureDiscovery/fetch_papers.py

  # Only some sources, deeper pull, papers since 2019
  python FeatureDiscovery/fetch_papers.py --sources openalex crossref_journals nber \
      --per-source 400 --since-year 2019

  # List what's available, or pull without re-scoring
  python FeatureDiscovery/fetch_papers.py --list-sources
  python FeatureDiscovery/fetch_papers.py --no-score

Set S2_API_KEY (Semantic Scholar) and/or CORE_API_KEY to enable the key-optional sources.
After a run, generate_feature.py / relevance_score.py consume the store exactly as before.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# Run-as-script import setup: this file lives in FeatureDiscovery/.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import sources                     # noqa: E402
from sources import _base as B     # noqa: E402
import relevance_score             # noqa: E402

ROOT      = _HERE.parent
OUT_PATH  = ROOT / "Data" / "PaperFeed" / "papers.parquet"

# Columns make_record produces (legacy store rows are backfilled to match before merge).
_NEW_COLS = ["paper_id", "source", "doi", "year", "venue"]
_NORM_RE  = re.compile(r"[^a-z0-9]+")


# ---------------------------------------------------------------------------
# Source resolution
# ---------------------------------------------------------------------------

def resolve_sources(names: list[str]) -> list[str]:
    if not names or names == ["all"] or names == ["default"]:
        return list(sources.DEFAULT_SOURCES)
    if names == ["keyless"]:
        return list(sources.KEYLESS)
    if names == ["high-yield"] or names == ["high_yield"]:
        return list(sources.HIGH_YIELD)
    out = []
    for n in names:
        if n not in sources.REGISTRY:
            raise SystemExit(f"Unknown source {n!r}. Available: "
                             f"{', '.join(sources.REGISTRY)}")
        out.append(n)
    return out


def load_queries(path: str | None) -> list[str]:
    if not path:
        return list(B.DEFAULT_QUERIES)
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    qs = [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]
    return qs or list(B.DEFAULT_QUERIES)


# ---------------------------------------------------------------------------
# Fetch driver
# ---------------------------------------------------------------------------

def run_sources(source_names, queries, since_year, per_source, arxiv_days, opts=None):
    opts = opts or {}
    session = requests.Session()
    all_recs: list[dict] = []
    report: list[tuple[str, str, int, str]] = []   # (name, status, n, note)

    for name in source_names:
        mod = sources.REGISTRY[name]
        print(f"  {name:<18} ...", flush=True)
        t0 = time.time()
        try:
            recs = mod.fetch(session, queries=queries, since_year=since_year,
                             per_source=per_source, arxiv_days=arxiv_days, **opts)
            recs = [r for r in recs if r.get("title")]
            all_recs.extend(recs)
            report.append((name, "ok", len(recs), f"{time.time()-t0:.0f}s"))
            print(f"  {name:<18} kept {len(recs)}  ({time.time()-t0:.0f}s)")
        except B.SourceUnavailable as exc:
            report.append((name, "skipped", 0, str(exc)))
            print(f"  {name:<18} SKIPPED  ({exc})")
        except Exception as exc:                   # noqa: BLE001 -- one bad source can't kill the run
            report.append((name, "error", 0, f"{type(exc).__name__}: {exc}"))
            print(f"  {name:<18} ERROR    ({type(exc).__name__}: {exc})")
    return all_recs, report


# ---------------------------------------------------------------------------
# Merge + cross-source dedup
# ---------------------------------------------------------------------------

def _norm_title(t: str) -> str:
    return _NORM_RE.sub("", str(t or "").lower())


def _dedupe_key(row) -> str:
    doi = str(row.get("doi") or "")
    if doi:
        return "doi:" + doi
    t = _norm_title(row.get("title", ""))
    if len(t) >= 12:                # very short titles are unreliable keys
        return "title:" + t
    return "pid:" + str(row.get("paper_id") or row.get("arxiv_id") or id(row))


def _backfill_legacy(df: pd.DataFrame) -> pd.DataFrame:
    """Give a pre-existing (arXiv-only) store the new columns so it merges cleanly."""
    if df.empty:
        return df
    if "source" not in df:
        df["source"] = "arxiv"
    if "paper_id" not in df:
        df["paper_id"] = "arxiv:" + df.get("arxiv_id", "").astype(str)
    if "doi" not in df:
        df["doi"] = ""
    if "venue" not in df:
        df["venue"] = ""
    if "year" not in df:
        df["year"] = df.get("published", "").astype(str).str.slice(0, 4)
    for c in _NEW_COLS:
        df[c] = df[c].fillna("")
    return df


def merge_store(new_records, out_path: Path):
    new_df = pd.DataFrame(new_records)
    old_df = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
    old_df = _backfill_legacy(old_df)

    if new_df.empty and old_df.empty:
        raise SystemExit("Nothing fetched and no existing store -- aborting.")

    combined = pd.concat([old_df, new_df], ignore_index=True, sort=False)
    for c in _NEW_COLS + ["title", "abstract", "doi"]:
        if c in combined:
            combined[c] = combined[c].fillna("")

    n_before = len(combined)
    prio = {s: i for i, s in enumerate(sources.SOURCE_PRIORITY)}
    combined["_key"]   = combined.apply(_dedupe_key, axis=1)
    combined["_hasab"] = (combined["abstract"].astype(str).str.len() > 0).astype(int)
    combined["_prio"]  = combined["source"].map(lambda s: prio.get(s, 999))
    # Prefer: has-abstract, then highest-priority source. keep='first' after this sort.
    combined = combined.sort_values(["_hasab", "_prio"], ascending=[False, True])
    combined = combined.drop_duplicates(subset="_key", keep="first")
    combined = combined.drop(columns=["_key", "_hasab", "_prio"])
    n_dupes = n_before - len(combined)

    return combined.reset_index(drop=True), len(new_df), n_dupes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Unified multi-source paper puller")
    ap.add_argument("--sources", nargs="*", default=["all"],
                    help="Source names, or 'all' / 'keyless' / 'default'. "
                         f"Available: {', '.join(sources.REGISTRY)}")
    ap.add_argument("--per-source", type=int, default=250,
                    help="Max papers kept per source (default 250)")
    ap.add_argument("--since-year", type=int, default=2017,
                    help="Drop papers published before this year (0 = no filter, default 2017)")
    ap.add_argument("--arxiv-days", type=int, default=4,
                    help="arXiv look-back window in days (arXiv browses by date, default 4)")
    ap.add_argument("--queries-file", default=None,
                    help="Optional file of finance search queries, one per line (# = comment)")
    ap.add_argument("--out", default=str(OUT_PATH), help="Output parquet store")
    ap.add_argument("--no-score", action="store_true",
                    help="Skip the relevance_score re-run (store left unscored for new rows)")
    ap.add_argument("--threshold", type=float, default=5.0,
                    help="rel_keep threshold passed to the relevance scorer (default 5.0)")
    ap.add_argument("--openalex-broad", action="store_true",
                    help="Disable OpenAlex's Economics/Finance concept filter (searches the "
                         "whole corpus; noisier but wider)")
    ap.add_argument("--list-sources", action="store_true",
                    help="List available sources and exit")
    args = ap.parse_args()

    if args.list_sources:
        print("Available sources:")
        for n in sources.REGISTRY:
            tag = "keyless" if n in sources.KEYLESS else "needs/uses API key"
            print(f"  {n:<20} {tag}")
        print(f"\nDefault run: {', '.join(sources.DEFAULT_SOURCES)}")
        return

    source_names = resolve_sources(args.sources)
    queries = load_queries(args.queries_file)
    since_year = args.since_year or 0
    out_path = Path(args.out)

    print(f"Multi-source paper fetch  --  {len(source_names)} sources, "
          f"{len(queries)} queries, per-source {args.per_source}, "
          f"since {since_year or 'any'}")
    print()

    t0 = time.time()
    records, report = run_sources(source_names, queries, since_year,
                                  args.per_source, args.arxiv_days,
                                  opts={"openalex_broad": args.openalex_broad})
    print(f"\n  fetched {len(records)} rows across {len(source_names)} sources "
          f"in {time.time()-t0:.0f}s")

    merged, n_new_raw, n_dupes = merge_store(records, out_path)

    if not args.no_score:
        merged = relevance_score.score_store(merged)
        merged["rel_keep"] = merged["rel_score"] >= args.threshold

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if "published" in merged.columns:
        merged = merged.sort_values("published", ascending=False).reset_index(drop=True)
    merged.to_parquet(out_path, index=False)

    # ---- summary ----
    print("\n" + "=" * 64)
    print(f"{'source':<20}{'status':<10}{'kept':>6}   note")
    print("-" * 64)
    for name, status, n, note in report:
        print(f"{name:<20}{status:<10}{n:>6}   {note}")
    print("-" * 64)
    print(f"fetched this run (raw, pre-dedup) : {n_new_raw}")
    print(f"cross-source duplicates removed   : {n_dupes}")
    print(f"store total                       : {len(merged)}  ->  {out_path}")
    if not args.no_score:
        n_keep = int(merged["rel_keep"].sum())
        n_ext  = int(merged["needs_external_data"].sum())
        by_src = merged["source"].value_counts()
        print(f"relevance: keep(>= {args.threshold}) {n_keep}  |  external-capped {n_ext}")
        print("store by source: " + ", ".join(f"{s}={c}" for s, c in by_src.items()))
    print("\nNext: python FeatureDiscovery/relevance_score.py --top 40   "
          "(or go straight to generate_feature.py)")


if __name__ == "__main__":
    main()
