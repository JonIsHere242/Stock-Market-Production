"""Content-addressed incremental cache — the headline feature.

Two stores (Bazel's proven shape):
  * MANIFEST  — maps a block's cache_key -> {column: content_hash}.
  * CAS       — a content-addressable store of column blobs, named by content hash.
                Two blocks (or tickers, or runs) that produce a bit-identical
                column store ONE physical blob. Free dedup at 10TB.

cache_key(block) = sha( block.code_version          # source + helper-closure + env
                        + sorted(data_version(c) for c in block.requires) )

A block is a cache HIT iff its code, its transitive helper closure, its library
env, AND every input column it consumes are unchanged. Then we never run it — we
just stitch the stored columns back in. That is what turns a full 28GB recompute
into a delta rebuild.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from block import Block
from hashing import data_version, hash_text


class Cache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.cas_dir = self.root / "cas"
        self.manifest_path = self.root / "manifest.json"
        self.snapshot_path = self.root / "code_versions.json"
        self.cas_dir.mkdir(parents=True, exist_ok=True)
        self._manifest: dict[str, dict] = self._load(self.manifest_path)
        self._snapshot: dict[str, str] = self._load(self.snapshot_path)
        self.stats = {"hit": 0, "miss": 0, "blobs_written": 0, "blobs_deduped": 0}

    # -- persistence ----------------------------------------------------------
    @staticmethod
    def _load(p: Path) -> dict:
        if p.is_file():
            try:
                return json.loads(p.read_text("utf-8"))
            except (OSError, json.JSONDecodeError):
                return {}
        return {}

    @staticmethod
    def _atomic_write(p: Path, obj: dict) -> None:
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, sort_keys=True), "utf-8")
        tmp.replace(p)

    def flush(self) -> None:
        self._atomic_write(self.manifest_path, self._manifest)
        self._atomic_write(self.snapshot_path, self._snapshot)

    # -- keys -----------------------------------------------------------------
    @staticmethod
    def cache_key(block_code_version: str, upstream_versions: dict[str, str], requires: list[str]) -> str:
        ups = [f"{c}={upstream_versions.get(c, 'MISSING')}" for c in sorted(requires)]
        return hash_text(block_code_version, *ups)

    # -- CAS ------------------------------------------------------------------
    def _blob_path(self, content_hash: str) -> Path:
        return self.cas_dir / f"{content_hash}.npy"

    def store_column(self, arr: np.ndarray) -> str:
        ch = data_version(arr)
        p = self._blob_path(ch)
        if p.exists():
            self.stats["blobs_deduped"] += 1
        else:
            np.save(p, np.ascontiguousarray(arr), allow_pickle=False)
            self.stats["blobs_written"] += 1
        return ch

    def load_column(self, content_hash: str) -> np.ndarray:
        return np.load(self._blob_path(content_hash), allow_pickle=False)

    # -- block-level get/put --------------------------------------------------
    def get(self, key: str) -> dict[str, str] | None:
        """Return {column: content_hash} if cached AND all blobs still present."""
        entry = self._manifest.get(key)
        if entry is None:
            return None
        cols = entry["produced"]
        if all(self._blob_path(ch).exists() for ch in cols.values()):
            self.stats["hit"] += 1
            return cols
        return None  # blob evicted -> treat as miss

    def put(self, key: str, produced: dict[str, np.ndarray]) -> dict[str, str]:
        cols = {name: self.store_column(arr) for name, arr in produced.items()}
        self._manifest[key] = {"produced": cols}
        self.stats["miss"] += 1
        return cols

    # -- changed-block detection (for `--select modified+`) -------------------
    def detect_changed(self, blocks: list[Block], project_root: Path) -> set[str]:
        changed: set[str] = set()
        for b in blocks:
            cv = b.code_version(project_root)
            if self._snapshot.get(b.name) != cv:
                changed.add(b.name)
        return changed

    def update_snapshot(self, blocks: list[Block], project_root: Path) -> None:
        for b in blocks:
            self._snapshot[b.name] = b.code_version(project_root)
