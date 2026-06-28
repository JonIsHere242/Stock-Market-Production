r"""
fetch_alt.py  --  Speculative "alt-paper" ingestion: turn ANY text into candidate features.

The maximally-wide front-end to the feature-discovery pipeline. It fetches raw text from
quant GitHub repos, YouTube transcripts, quant-blog RSS (Quantocracy & friends), Hacker News,
Reddit, and a universal URL/file/text catch-all, then runs each item through the `claude -p`
extraction spine (altsources/extract.py, $0 on your Max subscription) which tries -- creatively
but honestly -- to distill an OHLCV-computable feature method out of it. Survivors are written
to Data/PaperFeed/papers.parquet as `is_mock=True` rows so they ride the SAME relevance_score
-> generate_feature -> validate_feature gates as real papers. The gates remain the real filter,
so casting absurdly wide is cheap: junk just gets thrown out downstream.

USAGE
-----
  # See what the fetchers pull WITHOUT spending any LLM calls (cheap, fast):
  python FeatureDiscovery/fetch_alt.py --dry-run

  # Real run: default sources, cap 30 extractions, sonnet extractor
  python FeatureDiscovery/fetch_alt.py --max-extract 30

  # Mine specific creators / feeds / channels
  python FeatureDiscovery/fetch_alt.py --sources github youtube \
      --github-users neurotrader888 je-suis-tm \
      --youtube-channels UCSh87zxGNu8q8iOInRK6E9w @QuantPy

  # The universal door: ingest literally anything
  python FeatureDiscovery/fetch_alt.py --sources web \
      --urls https://some-blog/post --files C:\path\rust_base_wipe_transcript.txt

Idempotent: items already in the store (by source:native_id) are skipped BEFORE the LLM call.
Needs the `claude` CLI logged in (same as generate_feature.py). YouTube needs
`pip install youtube-transcript-api`. GITHUB_TOKEN / REDDIT_UA widen those sources.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import altsources                          # noqa: E402
from altsources import _base as AB         # noqa: E402
from altsources import extract as EX       # noqa: E402
import fetch_papers                        # noqa: E402  (reuse merge_store)
import relevance_score                     # noqa: E402

ROOT     = _HERE.parent
OUT_PATH = ROOT / "Data" / "PaperFeed" / "papers.parquet"
_ALT_COLS = ["is_mock", "raw_url", "mapping_note", "extractor_model", "raw_excerpt"]


def resolve_sources(names):
    if not names or names == ["all"] or names == ["default"]:
        return list(altsources.DEFAULT_SOURCES)
    for n in names:
        if n not in altsources.REGISTRY:
            raise SystemExit(f"Unknown alt-source {n!r}. Available: "
                             f"{', '.join(altsources.REGISTRY)}")
    return list(names)


def existing_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    df = pd.read_parquet(out_path, columns=None)
    ids = set()
    if "paper_id" in df:
        ids |= set(df["paper_id"].dropna().astype(str))
    if "arxiv_id" in df:
        ids |= set(df["arxiv_id"].dropna().astype(str))
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description="Speculative alt-paper ingestion -> mock papers")
    ap.add_argument("--sources", nargs="*", default=["all"],
                    help=f"Alt-sources or 'all'. Available: {', '.join(altsources.REGISTRY)}")
    ap.add_argument("--limit", type=int, default=15, help="Max RAW items pulled per source")
    ap.add_argument("--max-extract", type=int, default=30,
                    help="Max LLM extractions this run (cost cap; default 30)")
    ap.add_argument("--model", default="sonnet", help="Extractor model (sonnet|opus|default)")
    ap.add_argument("--timeout", type=int, default=180, help="Per-extraction timeout seconds")
    ap.add_argument("--dry-run", action="store_true",
                    help="Fetch raw items and report; NO LLM calls (free)")
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--no-score", action="store_true", help="Skip relevance re-scoring")
    ap.add_argument("--threshold", type=float, default=5.0)
    ap.add_argument("--list-sources", action="store_true")
    # per-source config
    ap.add_argument("--github-users", nargs="*", default=None)
    ap.add_argument("--youtube-channels", nargs="*", default=None)
    ap.add_argument("--rss-feeds", nargs="*", default=None)
    ap.add_argument("--rss-fulltext", action="store_true")
    ap.add_argument("--reddit-subs", nargs="*", default=None)
    ap.add_argument("--hn-queries", nargs="*", default=None)
    ap.add_argument("--hn-fulltext", action="store_true")
    ap.add_argument("--urls", nargs="*", default=None, help="web source: URLs to ingest")
    ap.add_argument("--files", nargs="*", default=None, help="web source: local text files")
    args = ap.parse_args()

    if args.list_sources:
        for n in altsources.REGISTRY:
            print(f"  {n}")
        print(f"\nDefault: {', '.join(altsources.DEFAULT_SOURCES)}")
        return

    source_names = resolve_sources(args.sources)
    out_path = Path(args.out)
    opts = {
        "github_users": args.github_users,
        "youtube_channels": args.youtube_channels,
        "rss_feeds": args.rss_feeds, "rss_fulltext": args.rss_fulltext,
        "reddit_subs": args.reddit_subs,
        "hn_queries": args.hn_queries, "hn_fulltext": args.hn_fulltext,
        "web_urls": args.urls, "web_files": args.files, "web_texts": None,
    }

    print(f"Alt-paper ingestion  --  sources: {', '.join(source_names)}  "
          f"| limit/src {args.limit} | max-extract {args.max_extract}"
          f"{' | DRY RUN' if args.dry_run else ''}\n")

    # ---- 1. fetch raw items per source ----
    session = requests.Session()
    raw_items: list[dict] = []
    fetch_report = []
    for name in source_names:
        mod = altsources.REGISTRY[name]
        t0 = time.time()
        try:
            items = mod.fetch(session, limit=args.limit, **opts)
            raw_items.extend(items)
            fetch_report.append((name, "ok", len(items), f"{time.time()-t0:.0f}s"))
            print(f"  {name:<12} raw items: {len(items)}  ({time.time()-t0:.0f}s)")
            for it in items[:3]:
                print(f"       - {it['title'][:78]}")
        except AB.SourceUnavailable as exc:
            fetch_report.append((name, "skipped", 0, str(exc)))
            print(f"  {name:<12} SKIPPED ({exc})")
        except Exception as exc:                   # noqa: BLE001
            fetch_report.append((name, "error", 0, f"{type(exc).__name__}: {exc}"))
            print(f"  {name:<12} ERROR ({type(exc).__name__}: {exc})")

    # idempotency: drop items already in the store before spending LLM calls
    seen = existing_ids(out_path)
    fresh = [it for it in raw_items if f"{it['source']}:{it['native_id']}" not in seen]
    print(f"\n  raw items: {len(raw_items)}  | new (not in store): {len(fresh)}  "
          f"| already-seen skipped: {len(raw_items)-len(fresh)}")

    if args.dry_run:
        print("\n[DRY RUN] stopping before extraction. Re-run without --dry-run to extract.")
        return

    # ---- 2. LLM extraction -> mock papers (budget-capped) ----
    mock_records = []
    n_extract = n_computable = 0
    for it in fresh:
        if n_extract >= args.max_extract:
            print(f"  [budget] hit --max-extract {args.max_extract}; stopping extraction")
            break
        n_extract += 1
        res = EX.extract_method(it["text"], source=it["source"], title=it["title"],
                                url=it["url"], model=args.model, timeout=args.timeout)
        tag = "SKIP"
        if res:
            n_computable += 1
            tag = f"OK [{res.get('method_family','?')}]"
            mock_records.append(AB.make_mock_record(
                source=it["source"], native_id=it["native_id"],
                title=res["title"], abstract=res["abstract"], url=it["url"],
                method_family=res.get("method_family", ""),
                mapping_note=res.get("mapping_note", ""),
                extractor_model=res.get("_model", args.model),
                raw_excerpt=it["text"]))
        print(f"    [{n_extract:>3}] {tag:<22} {it['source']}: {it['title'][:54]}")

    print(f"\n  extracted {n_extract} | computable mock papers: {n_computable}")
    if not mock_records:
        print("  nothing computable this run -- store unchanged.")
        return

    # ---- 3. merge into the shared store (reuse fetch_papers dedup) + re-score ----
    merged, n_new_raw, n_dupes = fetch_papers.merge_store(mock_records, out_path)
    merged["is_mock"] = merged.get("is_mock", False)
    merged["is_mock"] = merged["is_mock"].fillna(False)
    for c in _ALT_COLS[1:]:
        if c in merged:
            merged[c] = merged[c].fillna("")

    if not args.no_score:
        merged = relevance_score.score_store(merged)
        merged["rel_keep"] = merged["rel_score"] >= args.threshold

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if "published" in merged.columns:
        merged = merged.sort_values("published", ascending=False).reset_index(drop=True)
    merged.to_parquet(out_path, index=False)

    # ---- summary ----
    print("\n" + "=" * 64)
    for name, status, n, note in fetch_report:
        print(f"  {name:<12}{status:<9}{n:>4}   {note}")
    print("-" * 64)
    n_mock_total = int(merged["is_mock"].sum()) if "is_mock" in merged else 0
    print(f"new mock papers added : {len(mock_records)} (dupes removed {n_dupes})")
    print(f"store total           : {len(merged)}  | mock rows total: {n_mock_total}")
    if not args.no_score:
        keep = int((merged["rel_keep"] & merged.get("is_mock", False)).sum())
        print(f"mock rows passing rel>= {args.threshold}: {keep}")
    print(f"-> {out_path}")
    print("\nNext: review mock rows, then  "
          "python FeatureDiscovery/generate_feature.py  (or --idea from a mock abstract)")


if __name__ == "__main__":
    main()
