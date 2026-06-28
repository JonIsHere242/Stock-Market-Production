"""Content-addressed hashing — the spine of incremental materialization.

This is the piece that makes FeatureFrameworkV2 "Hamilton + 10": Hamilton hashes a node's
OWN source but NOT the helper functions it calls or the library versions it runs
under, so editing a shared helper (Util.py, _marketcap.py) silently serves STALE
cache. We close that hole by hashing the *transitive local-module closure* plus a
pinned environment fingerprint.

A block's `code_version` = sha(
    AST-normalized source of the compute (or family builder + params)
    + sources of every project-local module it transitively imports
    + environment fingerprint (python + numpy/pandas/polars/xgboost versions)
)

A column's `data_version` = sha(its bytes). Identical content → identical version
→ free cross-ticker/cross-run dedup in the CAS.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import sys
import textwrap
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

_HASHLEN = 16  # 16 hex chars (64 bits) is plenty for collision-free local use


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:_HASHLEN]


def hash_text(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:_HASHLEN]


# --------------------------------------------------------------------------- #
# Source hashing (formatting/comment-insensitive via AST normalization)
# --------------------------------------------------------------------------- #
def ast_normalized_source(fn: Callable) -> str:
    """AST dump of a function's source — ignores comments, docstrings and
    whitespace so cosmetic edits don't bust the cache (Hamilton does the same)."""
    try:
        src = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        # closures / C functions / REPL-defined: fall back to a stable repr
        return f"<no-source:{getattr(fn, '__qualname__', repr(fn))}>"
    try:
        tree = ast.parse(src)
        _strip_docstrings(tree)
        return ast.dump(tree, annotate_fields=False)
    except SyntaxError:
        return src


def _strip_docstrings(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(
                getattr(body[0], "value", None), ast.Constant
            ) and isinstance(body[0].value.value, str):
                node.body = body[1:]


# --------------------------------------------------------------------------- #
# Transitive project-local import closure  (the Hamilton blind-spot fix)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=4096)
def _module_source(path_str: str) -> str:
    try:
        return Path(path_str).read_text("utf-8", "replace")
    except OSError:
        return ""


def _resolve_local_import(name: str, project_root: Path) -> Path | None:
    """Best-effort map an import name like 'Util' or 'pkg.mod' to a file under
    project_root. Returns None for stdlib / third-party (deliberately not hashed —
    those are covered by the environment fingerprint)."""
    rel = name.replace(".", "/")
    for cand in (project_root / f"{rel}.py", project_root / rel / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def local_closure_files(entry_file: Path, project_root: Path) -> list[Path]:
    """Transitive set of project-local .py files reachable from entry_file via
    import statements. Stops at the project boundary (stdlib/3rd-party excluded)."""
    project_root = project_root.resolve()
    seen: set[Path] = set()
    stack = [entry_file.resolve()]
    while stack:
        f = stack.pop()
        if f in seen or not f.is_file():
            continue
        seen.add(f)
        try:
            tree = ast.parse(_module_source(str(f)))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            names: Iterable[str] = ()
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            for n in names:
                tgt = _resolve_local_import(n, project_root)
                if tgt and tgt.resolve() not in seen:
                    stack.append(tgt.resolve())
    return sorted(seen)


@lru_cache(maxsize=None)
def env_fingerprint() -> str:
    """Pinned-ish fingerprint of the libraries that can change feature output."""
    parts = [f"py{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"]
    for lib in ("numpy", "pandas", "polars", "xgboost", "scipy"):
        try:
            parts.append(f"{lib}={importlib.import_module(lib).__version__}")
        except Exception:
            parts.append(f"{lib}=NA")
    return hash_text(*parts)


def code_version(
    *,
    fn: Callable,
    defining_file: Path | None,
    project_root: Path,
    extra: str = "",
) -> str:
    """Full content version for a unit of compute.

    `extra` lets a family fold its parameter dict into the version so two features
    emitted from the same builder with different params get distinct versions.
    """
    pieces = [ast_normalized_source(fn), env_fingerprint(), extra]
    if defining_file is not None:
        for f in local_closure_files(Path(defining_file), Path(project_root)):
            pieces.append(_sha(_module_source(str(f)).encode("utf-8", "replace")))
    return hash_text(*pieces)


# --------------------------------------------------------------------------- #
# Data versioning
# --------------------------------------------------------------------------- #
def data_version(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    if a.dtype.kind in "OUSV":  # object/unicode/bytes/void: tobytes() is not stable
        h.update(repr(a.tolist()).encode("utf-8", "replace"))
    else:
        h.update(a.tobytes())
    return h.hexdigest()[:_HASHLEN]
