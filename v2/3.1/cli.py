"""Orchestrator CLI.

  python 3.1/cli.py list     [--family-dir D] [--legacy-dir D]
  python 3.1/cli.py catalog  [--out DIR] ...
  python 3.1/cli.py demo
"""
from __future__ import annotations

import argparse
from pathlib import Path

from registry import build_catalog, discover, write_catalog

_DEFAULT_FAM = Path(__file__).resolve().parent / "families"


def _discover(args):
    return discover(
        family_dir=Path(args.family_dir) if args.family_dir else None,
        legacy_dir=Path(args.legacy_dir) if args.legacy_dir else None,
        include_candidates=args.include_candidates,
    )


def cmd_list(args):
    blocks, families = _discover(args)
    print(f"{len(families)} families -> {len(blocks)} blocks "
          f"({sum(len(b.produces) for b in blocks)} feature columns)")
    for f in families:
        print(f"  [family] {f.name:<20} x{f.cardinality():<4} {','.join(f.tags)}")
    for b in blocks:
        if b.family is None:
            print(f"  [block]  {b.name:<28} -> {b.produces}")


def cmd_catalog(args):
    blocks, families = _discover(args)
    cat = build_catalog(blocks, families)
    jp, mp = write_catalog(cat, Path(args.out))
    print(f"wrote {jp} and {mp}")
    print(f"{cat['n_blocks']} blocks / {cat['n_feature_columns']} cols / {cat['n_families']} families")


def cmd_demo(_args):
    from demo import main
    main()


def cmd_demo_beyond(_args):
    from demo_beyond import main
    main()


def cmd_demo_alpha(_args):
    from demo_alpha import main
    main()


def cmd_viz(args):
    from graph import FeatureGraph
    from viz import write_viz
    blocks, _ = _discover(args)
    out = write_viz(FeatureGraph(blocks), Path(args.out))
    print(f"wrote {out['dot']}, {out['mermaid']}, {out['families']} "
          f"({out['n_nodes']} nodes / {out['n_edges']} edges)")


def cmd_search(args):
    from catalog_search import build_index
    blocks, families = _discover(args)
    idx = build_index(build_catalog(blocks, families))
    for name, score in idx.search(args.query, k=args.k):
        print(f"  {score:5.3f}  {name}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="3.1__FeatureFramework.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    funcs = {"list": cmd_list, "catalog": cmd_catalog, "viz": cmd_viz, "search": cmd_search}
    for name in ("list", "catalog", "viz", "search"):
        sp = sub.add_parser(name)
        sp.add_argument("--family-dir", default=str(_DEFAULT_FAM))
        sp.add_argument("--legacy-dir", default=None, help="e.g. ../FeatureTemplates")
        sp.add_argument("--include-candidates", action="store_true")
        if name == "catalog":
            sp.add_argument("--out", default="_catalog")
        if name == "viz":
            sp.add_argument("--out", default="_viz")
        if name == "search":
            sp.add_argument("query")
            sp.add_argument("-k", type=int, default=10)
        sp.set_defaults(func=funcs[name])

    sub.add_parser("demo").set_defaults(func=cmd_demo)
    sub.add_parser("demo-beyond").set_defaults(func=cmd_demo_beyond)
    sub.add_parser("demo-alpha").set_defaults(func=cmd_demo_alpha)
    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
