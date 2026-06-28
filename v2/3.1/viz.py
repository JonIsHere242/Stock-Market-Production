"""Render the feature dependency DAG to Graphviz DOT and Mermaid text.

Hamilton can draw its DAG straight from code; this gives the v2 feature
framework the same affordance with stdlib only (no graphviz binary, no
network). A human -- or an LLM reading the .dot/.mmd -- can then SEE the
block graph: who feeds whom, grouped by family or kind, plus a zoomed-out
family rollup for libraries too large to read block-by-block. Output is
fully deterministic so it can feed content hashing elsewhere.
"""
from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

# Fixed, stable palette. Colors cycle if a graph has more distinct values.
_PALETTE: list[str] = [
    "#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1",
    "#76b7b2", "#edc948", "#ff9da7", "#9c755f", "#bab0ac",
]


def _color_key(block: Any, color_by: str) -> str:
    """Return the grouping value for a block ('' for missing/None)."""
    val = getattr(block, color_by, None)
    return "" if val is None else str(val)


def _color_map(graph: Any, color_by: str) -> dict[str, str]:
    """Stable value -> hex-color map, sorted so output is deterministic."""
    values = sorted({_color_key(b, color_by) for b in graph.blocks.values()})
    return {v: _PALETTE[i % len(_PALETTE)] for i, v in enumerate(values)}


def _dot_escape(text: str) -> str:
    """Escape a string for a DOT double-quoted literal."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


def to_dot(graph: Any, color_by: str = "family") -> str:
    """Render the block DAG as a Graphviz DOT string.

    One filled node per block (label = block name). Nodes are colored by the
    `color_by` attribute ("family" or "kind"); each distinct value gets a
    stable palette color. A directed edge A -> B is drawn for every consumer B
    in graph.fwd[A]. A legend cluster maps value -> color. Iteration is sorted
    so identical graphs yield byte-identical DOT.
    """
    if color_by not in ("family", "kind"):
        raise ValueError("color_by must be 'family' or 'kind'")
    cmap = _color_map(graph, color_by)

    lines: list[str] = ["digraph features {", "  rankdir=LR;",
                        '  node [style=filled, shape=box, fontname="Helvetica"];']

    # Nodes (sorted by name for determinism).
    for name in sorted(graph.blocks):
        block = graph.blocks[name]
        color = cmap[_color_key(block, color_by)]
        lines.append(
            f'  "{_dot_escape(name)}" '
            f'[label="{_dot_escape(name)}", fillcolor="{color}"];'
        )

    # Edges: producer -> consumer, sorted on both ends.
    for src in sorted(graph.fwd):
        for dst in sorted(graph.fwd[src]):
            lines.append(f'  "{_dot_escape(src)}" -> "{_dot_escape(dst)}";')

    # Legend cluster (value -> color), sorted.
    lines.append("  subgraph cluster_legend {")
    lines.append(f'    label="legend: {color_by}";')
    lines.append("    style=dashed;")
    for value in sorted(cmap):
        shown = value if value else "(none)"
        node_id = f"legend_{_dot_escape(shown)}"
        lines.append(
            f'    "{node_id}" '
            f'[label="{_dot_escape(shown)}", fillcolor="{cmap[value]}"];'
        )
    lines.append("  }")

    lines.append("}")
    return "\n".join(lines) + "\n"


def _mermaid_ids(names: list[str]) -> dict[str, str]:
    """Map block names -> unique Mermaid-safe node ids.

    Non-alphanumeric chars become '_'. Sanitized names can collide, so a
    short stable index suffix guarantees uniqueness. Input is sorted first
    so the index assignment is deterministic.
    """
    ids: dict[str, str] = {}
    for i, name in enumerate(sorted(names)):
        safe = "".join(c if c.isalnum() else "_" for c in name)
        ids[name] = f"n{i}_{safe}"
    return ids


def to_mermaid(graph: Any, direction: str = "TD") -> str:
    """Render the block DAG as a Mermaid flowchart string.

    Node ids are sanitized + index-suffixed (collision-safe); node text shows
    the original block name. Edges follow graph.fwd. Deterministic ordering.
    """
    ids = _mermaid_ids(list(graph.blocks))
    lines: list[str] = [f"flowchart {direction}"]

    for name in sorted(graph.blocks):
        text = escape(name).replace('"', "&quot;")
        lines.append(f'    {ids[name]}["{text}"]')

    for src in sorted(graph.fwd):
        for dst in sorted(graph.fwd[src]):
            lines.append(f"    {ids[src]} --> {ids[dst]}")

    return "\n".join(lines) + "\n"


def _family_of(block: Any) -> str:
    """Family label for rollup; family-less blocks get their own node."""
    fam = getattr(block, "family", None)
    return str(fam) if fam else f"block:{block.name}"


def family_rollup(graph: Any) -> str:
    """Render a Mermaid graph COLLAPSED to the family level.

    One node per family (plus one per family-less block). An edge family X ->
    family Y exists if any block in X feeds any block in Y. This is the
    zoomed-out view for libraries where the per-block graph is unreadable.
    Deterministic.
    """
    fam_of = {name: _family_of(b) for name, b in graph.blocks.items()}
    fam_edges: set[tuple[str, str]] = set()
    for src in graph.fwd:
        for dst in graph.fwd[src]:
            fx, fy = fam_of.get(src), fam_of.get(dst)
            if fx is not None and fy is not None and fx != fy:
                fam_edges.add((fx, fy))

    families = sorted(set(fam_of.values()))
    ids = _mermaid_ids(families)

    lines: list[str] = ["flowchart TD"]
    for fam in families:
        text = escape(fam).replace('"', "&quot;")
        lines.append(f'    {ids[fam]}["{text}"]')
    for fx, fy in sorted(fam_edges):
        lines.append(f"    {ids[fx]} --> {ids[fy]}")
    return "\n".join(lines) + "\n"


def write_viz(graph: Any, out_dir: Any) -> dict:
    """Write graph.dot, graph.mmd and families.mmd into out_dir.

    Returns a dict of the written paths plus node/edge counts.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dot_path = out / "graph.dot"
    mmd_path = out / "graph.mmd"
    fam_path = out / "families.mmd"

    dot_path.write_text(to_dot(graph), encoding="utf-8")
    mmd_path.write_text(to_mermaid(graph), encoding="utf-8")
    fam_path.write_text(family_rollup(graph), encoding="utf-8")

    n_edges = sum(len(v) for v in graph.fwd.values())
    return {
        "dot": str(dot_path),
        "mermaid": str(mmd_path),
        "families": str(fam_path),
        "n_nodes": len(graph.blocks),
        "n_edges": n_edges,
    }


if __name__ == "__main__":
    import tempfile
    from types import SimpleNamespace

    def _b(name: str, produces: list[str], requires: list[str],
           family: str, kind: str = "per_ticker") -> SimpleNamespace:
        return SimpleNamespace(
            name=name, produces=produces, requires=requires,
            tags=[], family=family, kind=kind,
        )

    # Tiny graph: 5 blocks, 2 families, a couple of edges.
    #   raw -> returns -> momentum     (family "price")
    #   raw -> volume_z                (family "volume")
    #   returns -> vol_ratio           (family "volume")
    blocks = [
        _b("raw", ["close", "volume"], [], "price", kind="panel"),
        _b("returns", ["ret_1d"], ["close"], "price"),
        _b("momentum::w=10", ["mom_10"], ["ret_1d"], "price"),
        _b("volume_z", ["vol_z"], ["volume"], "volume"),
        _b("vol_ratio", ["vol_ratio"], ["ret_1d"], "volume"),
    ]
    bmap = {b.name: b for b in blocks}
    produced_by = {c: b.name for b in blocks for c in b.produces}
    fwd: dict[str, set[str]] = {b.name: set() for b in blocks}
    rev: dict[str, set[str]] = {b.name: set() for b in blocks}
    for b in blocks:
        for col in b.requires:
            src = produced_by.get(col)
            if src and src != b.name:
                fwd[src].add(b.name)
                rev[b.name].add(src)
    g = SimpleNamespace(blocks=bmap, fwd=fwd, rev=rev)

    dot = to_dot(g, color_by="family")
    mmd = to_mermaid(g)
    fam = family_rollup(g)
    print("===== DOT =====")
    print(dot)
    print("===== MERMAID =====")
    print(mmd)
    print("===== FAMILY ROLLUP =====")
    print(fam)

    with tempfile.TemporaryDirectory(prefix="ffv2_viz_") as td:
        info = write_viz(g, td)
        print("===== write_viz =====")
        print(info)

        # Assertions / smoke checks.
        assert '"returns" -> "momentum::w=10";' in dot, "missing producer->consumer edge"
        assert "raw" in dot and "vol_ratio" in dot, "missing node names"
        assert "subgraph cluster_legend" in dot, "missing legend cluster"
        assert dot == to_dot(g, color_by="family"), "to_dot not deterministic"
        assert to_dot(g) == Path(info["dot"]).read_text(encoding="utf-8"), "written DOT mismatch"

        assert mmd.startswith("flowchart TD"), "mermaid missing header"
        assert "-->" in mmd, "mermaid missing edges"
        # momentum::w=10 must be sanitized in mermaid ids.
        assert "momentum__w_10" in mmd, "mermaid id not sanitized"

        assert fam.startswith("flowchart TD"), "rollup missing header"
        assert "price" in fam and "volume" in fam, "rollup missing family nodes"
        # price feeds volume (returns -> vol_ratio).
        n_price = _mermaid_ids(sorted({_family_of(b) for b in blocks}))["price"]
        n_volume = _mermaid_ids(sorted({_family_of(b) for b in blocks}))["volume"]
        assert f"{n_price} --> {n_volume}" in fam, "rollup missing price->volume edge"

        assert info["n_nodes"] == 5, f"expected 5 nodes, got {info['n_nodes']}"
        assert info["n_edges"] == 4, f"expected 4 edges, got {info['n_edges']}"

    print("PASS: to_dot produced nodes, edges, legend")
    print("PASS: to_dot deterministic (two calls identical)")
    print("PASS: to_mermaid header + sanitized ids + edges")
    print("PASS: family_rollup collapsed graph with cross-family edge")
    print("PASS: write_viz wrote 3 files, counts correct")
