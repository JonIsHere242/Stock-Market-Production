"""The unit of computation.

Two authoring styles, ONE runtime contract:

  1. Block         — a single feature unit. Either hand-written (your existing
                     FeatureTemplates contract: METADATA + compute(df)->df) or
                     emitted by a FeatureFamily.
  2. FeatureFamily — a generator. Declares a parameter grid and a builder; .expand()
                     yields many Blocks. This is the "10k features from hundreds of
                     LOC" lever — discovery loads the FAMILY, not 10k files, so it
                     dodges the importlib-over-N-files wall.

Every Block honours the cherished invariant: compute(df)->df ADDS columns, never
drops/reshapes/reorders rows.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Callable

import pandas as pd

from hashing import code_version


@dataclass
class Block:
    name: str
    produces: list[str]
    requires: list[str]
    compute: Callable[[pd.DataFrame], pd.DataFrame]
    tags: list[str] = field(default_factory=list)
    version: str = "1"
    kind: str = "per_ticker"            # "per_ticker" | "panel"
    family: str | None = None
    params: dict | None = None
    description: str = ""
    # provenance for content hashing
    _defining_file: Path | None = None
    _version_fn: Callable | None = None  # the fn whose source defines behaviour

    def code_version(self, project_root: Path) -> str:
        fn = self._version_fn or self.compute
        extra = repr(sorted((self.params or {}).items())) + f"|v={self.version}"
        return code_version(
            fn=fn,
            defining_file=self._defining_file,
            project_root=Path(project_root),
            extra=extra,
        )

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        before = set(df.columns)
        out = self.compute(df)
        if out is None:
            raise ValueError(f"block {self.name}: compute returned None")
        if len(out) != len(df):
            raise ValueError(
                f"block {self.name}: row count changed {len(df)}->{len(out)} "
                "(violates add-columns-never-reshape contract)"
            )
        missing = set(self.produces) - set(out.columns)
        if missing:
            raise ValueError(f"block {self.name}: declared but did not produce {missing}")
        dropped = before - set(out.columns)
        if dropped:
            raise ValueError(f"block {self.name}: dropped input columns {dropped}")
        return out


@dataclass
class FeatureFamily:
    """A parametrized feature generator.

    builder(params) -> (produces: list[str], compute: callable)   where compute
    adds exactly `produces` to the frame. The family folds its params into each
    emitted Block's content hash, so changing the grid or the builder invalidates
    precisely the affected features.
    """
    name: str
    param_grid: dict[str, list]
    builder: Callable[[dict], tuple[list[str], Callable[[pd.DataFrame], pd.DataFrame]]]
    requires: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    kind: str = "per_ticker"
    description: str = ""
    _defining_file: Path | None = None

    def _combos(self) -> list[dict]:
        if not self.param_grid:
            return [{}]
        keys = list(self.param_grid)
        return [dict(zip(keys, vals)) for vals in product(*(self.param_grid[k] for k in keys))]

    def expand(self) -> list[Block]:
        blocks: list[Block] = []
        for params in self._combos():
            produces, compute = self.builder(params)
            blocks.append(
                Block(
                    name=f"{self.name}::" + ",".join(f"{k}={v}" for k, v in sorted(params.items())),
                    produces=list(produces),
                    requires=list(self.requires),
                    compute=compute,
                    tags=list(self.tags),
                    kind=self.kind,
                    family=self.name,
                    params=params,
                    description=self.description,
                    _defining_file=self._defining_file,
                    _version_fn=self.builder,   # hash the builder, not the closure
                )
            )
        return blocks

    def cardinality(self) -> int:
        n = 1
        for v in self.param_grid.values():
            n *= len(v)
        return n


# --------------------------------------------------------------------------- #
# Legacy adapter: ingest your existing FeatureTemplates/*.py blocks UNCHANGED
# --------------------------------------------------------------------------- #
def load_legacy_block(path: Path) -> Block | None:
    """Wrap a file following the FeatureTemplates contract (METADATA + compute)."""
    path = Path(path)
    spec = importlib.util.spec_from_file_location(f"_legacy_{path.stem}", path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return None
    meta = getattr(mod, "METADATA", None)
    compute = getattr(mod, "compute", None)
    if not isinstance(meta, dict) or not callable(compute):
        return None
    return Block(
        name=meta.get("name", path.stem),
        produces=list(meta.get("produces", [])),
        requires=list(meta.get("requires", [])),
        compute=compute,
        tags=list(meta.get("tags", [])),
        version=str(meta.get("version", "1")),
        description=meta.get("description", ""),
        _defining_file=path,
        _version_fn=compute,
    )
