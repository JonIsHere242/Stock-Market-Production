"""Discovery + the machine-readable catalog.

Two scaling moves live here:
  * FAMILY-FIRST DISCOVERY — we import family *modules* and expand them, instead of
    importing one file per feature. 10k features cost the import of a handful of
    generator modules, dodging the importlib-over-N-files wall.
  * THE CATALOG — a JSON + Markdown manifest auto-built from metadata so an LLM/agent
    can answer "do we already have X / what's adjacent / what's unexplored" WITHOUT
    reading 75k+ LOC of source. This is the navigability artifact.

Backward compatible: `discover(legacy_dir=...)` ingests your existing
FeatureTemplates/*.py blocks unchanged.
"""
from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

from block import Block, FeatureFamily, load_legacy_block
from graph import FeatureGraph


def load_families_from_file(path: Path) -> list[FeatureFamily]:
    path = Path(path)
    spec = importlib.util.spec_from_file_location(f"_fam_{path.stem}", path)
    if spec is None or spec.loader is None:
        return []
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fams = list(getattr(mod, "FAMILIES", []))
    for fam in fams:
        if isinstance(fam, FeatureFamily) and fam._defining_file is None:
            fam._defining_file = path
    return [f for f in fams if isinstance(f, FeatureFamily)]


def discover(
    *,
    family_dir: Path | None = None,
    legacy_dir: Path | None = None,
    include_candidates: bool = False,
) -> tuple[list[Block], list[FeatureFamily]]:
    families: list[FeatureFamily] = []
    blocks: list[Block] = []

    if family_dir is not None:
        for f in sorted(Path(family_dir).glob("*.py")):
            if f.name.startswith("__"):
                continue
            families.extend(load_families_from_file(f))
    for fam in families:
        blocks.extend(fam.expand())

    if legacy_dir is not None:
        for f in sorted(Path(legacy_dir).glob("*.py")):
            if f.name.startswith("__"):
                continue
            if f.name.startswith("_") and not include_candidates:
                continue
            b = load_legacy_block(f)
            if b is not None:
                blocks.append(b)

    return blocks, families


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #
def build_catalog(
    blocks: list[Block],
    families: list[FeatureFamily],
    *,
    cost_ms: dict[str, float] | None = None,
    scores: dict[str, dict[str, float]] | None = None,
) -> dict:
    graph = FeatureGraph(blocks)
    cost_ms = cost_ms or {}
    scores = scores or {}

    tag_counts = Counter()
    feature_cols = 0
    block_entries = []
    for b in blocks:
        tag_counts.update(b.tags)
        feature_cols += len(b.produces)
        block_entries.append(
            {
                "name": b.name,
                "family": b.family,
                "params": b.params,
                "produces": b.produces,
                "requires": b.requires,
                "tags": b.tags,
                "kind": b.kind,
                "description": b.description,
                "cost_ms": round(cost_ms.get(b.name, 0.0), 4),
                "scores": scores.get(b.name, {}),
                "upstream": sorted(graph.rev[b.name]),
                "downstream": sorted(graph.fwd[b.name]),
            }
        )

    fam_entries = {
        f.name: {
            "cardinality": f.cardinality(),
            "param_grid": {k: list(v) for k, v in f.param_grid.items()},
            "tags": f.tags,
            "requires": f.requires,
            "kind": f.kind,
            "description": f.description,
        }
        for f in families
    }

    return {
        "n_blocks": len(blocks),
        "n_feature_columns": feature_cols,
        "n_families": len(families),
        "families": fam_entries,
        "tags": dict(tag_counts.most_common()),
        "blocks": block_entries,
    }


def write_catalog(catalog: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "catalog.json"
    md_path = out_dir / "CATALOG.md"
    json_path.write_text(json.dumps(catalog, indent=1, sort_keys=True), "utf-8")

    lines = [
        "# Feature Catalog (auto-generated)",
        "",
        f"- **{catalog['n_blocks']:,}** feature blocks producing "
        f"**{catalog['n_feature_columns']:,}** columns from "
        f"**{catalog['n_families']}** families.",
        "",
        "## Families",
        "",
        "| family | cardinality | tags | grid |",
        "|---|---|---|---|",
    ]
    for name, f in sorted(catalog["families"].items()):
        grid = ", ".join(f"{k}={v}" for k, v in f["param_grid"].items())
        lines.append(f"| `{name}` | {f['cardinality']} | {', '.join(f['tags'])} | {grid} |")

    lines += ["", "## Tag taxonomy", "", "| tag | # blocks |", "|---|---|"]
    for tag, n in catalog["tags"].items():
        lines.append(f"| {tag} | {n} |")

    md_path.write_text("\n".join(lines) + "\n", "utf-8")
    return json_path, md_path
