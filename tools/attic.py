#!/usr/bin/env python
"""
attic.py: move dead weight out of the working tree, reversibly.

Nothing is ever deleted. Every entry named in tools/attic_manifest.yaml is MOVED
into _ATTIC/<bucket>/ and recorded as one row in _ATTIC/MANIFEST.csv, so any
single entry can be put back with `restore`.

    python tools/attic.py plan                 what would move, with sizes
    python tools/attic.py plan --step 2a       just one step
    python tools/attic.py apply --step 2a      do it, append to the manifest
    python tools/attic.py restore <path>       put one entry back
    python tools/attic.py verify               is anything still referencing the attic?

The verify pass is the safety net and it is deliberately picky about what counts
as a reference. A backtester fork named in a docstring for provenance is NOT a
reference; the same name in an `import` or in a subprocess argument list IS. The
distinction is drawn with `ast`, not with grep, because six of the seven
backtester forks in this repo are cited in prose by live code and none of those
citations are real edges.
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime as _dt
import os
import re
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
ATTIC = ROOT / "_ATTIC"
MANIFEST_CSV = ATTIC / "MANIFEST.csv"
SPEC = ROOT / "tools" / "attic_manifest.yaml"

CSV_FIELDS = ["src", "dst", "bucket", "step", "reason", "bytes", "mtime_newest", "moved_at"]

# Directories the reference scan never reads. stock_env is the venv (27k files),
# _ATTIC is by definition the thing we are scanning FOR.
SCAN_SKIP_DIRS = {"stock_env", "_ATTIC", ".git", "__pycache__", "node_modules"}
SCAN_SUFFIXES = {".py", ".ps1", ".bat", ".cmd"}


# --------------------------------------------------------------------------- #
#  size / time helpers
# --------------------------------------------------------------------------- #

def tree_stats(path: Path, *, walk: bool = True) -> tuple[int, float]:
    """
    (total bytes, newest mtime) for a file or a directory tree.

    Walking is opt-in for directories because this repo's labs run to 12 GB
    apiece and a full stat pass over experimental/ takes many minutes, while the
    moves themselves are same-drive renames that are instant. Pass walk=False to
    get (-1, dir mtime) and keep the sweep responsive.
    """
    if path.is_file():
        st = path.stat()
        return st.st_size, st.st_mtime
    if not walk:
        return -1, path.stat().st_mtime
    total, newest = 0, 0.0
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            try:
                st = (Path(dirpath) / fn).stat()
            except OSError:
                continue
            total += st.st_size
            newest = max(newest, st.st_mtime)
    return total, newest


def human(n: int) -> str:
    if n < 0:
        return "dir"
    for unit in ("B", "K", "M", "G", "T"):
        if n < 1024 or unit == "T":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}T"


def stamp(mtime: float) -> str:
    if not mtime:
        return "-"
    return _dt.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")


# --------------------------------------------------------------------------- #
#  the manifest spec
# --------------------------------------------------------------------------- #

def load_spec() -> list[dict]:
    """Flatten the yaml into one dict per entry, in step order."""
    if not SPEC.exists():
        sys.exit(f"[FATAL] no manifest at {SPEC}")
    raw = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    out = []
    for step in raw["steps"]:
        for entry in step["entries"]:
            # an entry is either a bare path string or {path:, reason:}
            if isinstance(entry, str):
                path, reason = entry, step.get("reason", "")
            else:
                path, reason = entry["path"], entry.get("reason", step.get("reason", ""))
            out.append({
                "step": step["id"],
                "bucket": step["bucket"],
                "title": step["title"],
                "path": path,
                "reason": reason,
            })
    return out


def resolve(entry: dict) -> list[Path]:
    """Expand an entry's path, which may be a glob, into real existing paths."""
    pat = entry["path"]
    if any(ch in pat for ch in "*?["):
        return sorted(p for p in ROOT.glob(pat) if p.exists())
    p = ROOT / pat
    return [p] if p.exists() else []


# --------------------------------------------------------------------------- #
#  plan / apply
# --------------------------------------------------------------------------- #

def cmd_plan(args) -> None:
    entries = [e for e in load_spec() if not args.step or e["step"] == args.step]
    grand_bytes = 0
    missing = []
    current_step = None

    for entry in entries:
        if entry["step"] != current_step:
            current_step = entry["step"]
            print(f"\n=== {entry['step']}  {entry['title']}   -> _ATTIC/{entry['bucket']}/")
        paths = resolve(entry)
        if not paths:
            missing.append(entry["path"])
            continue
        for p in paths:
            size, newest = tree_stats(p, walk=args.size)
            grand_bytes += max(size, 0)
            rel = p.relative_to(ROOT).as_posix()
            print(f"  {human(size):>8}  {stamp(newest)}  {rel}")

    total = human(grand_bytes) if args.size else "(pass --size to measure)"
    print(f"\nTOTAL: {total} across {len(entries)} entries")
    if missing:
        print(f"\n[note] {len(missing)} manifest entries not on disk (already moved, or never existed):")
        for m in missing:
            print(f"       {m}")


def cmd_apply(args) -> None:
    entries = [e for e in load_spec() if not args.step or e["step"] == args.step]
    if not entries:
        sys.exit(f"[FATAL] no manifest entries for step {args.step!r}")

    ATTIC.mkdir(exist_ok=True)
    new_csv = not MANIFEST_CSV.exists()
    moved_bytes = 0
    rows = []

    for entry in entries:
        for src in resolve(entry):
            rel = src.relative_to(ROOT)
            dst = ATTIC / entry["bucket"] / rel
            if dst.exists():
                print(f"  [skip] destination exists: {dst.relative_to(ROOT)}")
                continue
            size, newest = tree_stats(src, walk=args.size)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            moved_bytes += max(size, 0)
            rows.append({
                "src": rel.as_posix(),
                "dst": dst.relative_to(ROOT).as_posix(),
                "bucket": entry["bucket"],
                "step": entry["step"],
                "reason": entry["reason"],
                "bytes": size,
                "mtime_newest": stamp(newest),
                "moved_at": _dt.datetime.now().isoformat(timespec="seconds"),
            })
            print(f"  {human(size):>8}  {rel.as_posix()}  ->  {entry['bucket']}/")

    if rows:
        with MANIFEST_CSV.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            if new_csv:
                w.writeheader()
            w.writerows(rows)

    print(f"\nmoved {len(rows)} entries, {human(moved_bytes)} -> _ATTIC/")
    print("run `python tools/attic.py verify` next")


def cmd_restore(args) -> None:
    if not MANIFEST_CSV.exists():
        sys.exit("[FATAL] no _ATTIC/MANIFEST.csv, nothing has been moved")
    rows = list(csv.DictReader(MANIFEST_CSV.open(encoding="utf-8")))
    want = args.path.replace("\\", "/").rstrip("/")
    hits = [r for r in rows if r["src"] == want or r["src"].startswith(want + "/")]
    if not hits:
        sys.exit(f"[FATAL] {want!r} is not in the manifest. `grep {want} _ATTIC/MANIFEST.csv`")

    for r in hits:
        src, dst = ROOT / r["dst"], ROOT / r["src"]
        if not src.exists():
            print(f"  [skip] gone from attic: {r['dst']}")
            continue
        if dst.exists():
            print(f"  [skip] already back in tree: {r['src']}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        print(f"  restored {r['src']}  ({r['reason']})")


# --------------------------------------------------------------------------- #
#  verify, the part that matters
# --------------------------------------------------------------------------- #

def _py_hard_refs(path: Path) -> set[str]:
    """
    Names a .py file genuinely depends on: imported modules, plus every string
    literal that is a real value rather than a docstring.

    Comments never reach the AST at all, and a bare string Expr (a docstring) is
    dropped explicitly, so provenance prose cannot produce a false positive.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, ValueError):
        return set()

    docstring_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                docstring_nodes.add(id(body[0].value))

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add(a.name.split(".")[0])
                names.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".")[0])
                names.add(node.module)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstring_nodes:
                names.add(node.value)
    return names


def _script_hard_refs(path: Path) -> set[str]:
    """Non-comment text of a .ps1/.bat/.cmd, as one blob."""
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if s.startswith("#") or s.lower().startswith("rem ") or s.startswith("::"):
            continue
        out.append(line)
    return {"\n".join(out)}


def surviving_files() -> list[Path]:
    files = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SCAN_SKIP_DIRS]
        for fn in filenames:
            if Path(fn).suffix.lower() in SCAN_SUFFIXES:
                files.append(Path(dirpath) / fn)
    return files


def cmd_verify(args) -> None:
    if not MANIFEST_CSV.exists():
        print("nothing moved yet, no _ATTIC/MANIFEST.csv")
        return

    rows = list(csv.DictReader(MANIFEST_CSV.open(encoding="utf-8")))

    # Map every .py now in the attic back to the repo-relative path it came FROM.
    # Matching on bare names instead would drown the report: decay_lab/_prun/w*/
    # held eight copies of the root pipeline, and xgb_znver4/ is a vendored
    # xgboost full of files called core.py, config.py, compat.py and sklearn.py.
    # `import sklearn` must not resolve to a vendored namesake.
    attic_origin: dict[str, str] = {}          # original repo-relative path -> reason
    for r in rows:
        dst = ROOT / r["dst"]
        base = Path(*Path(r["dst"]).parts[2:])   # strip "_ATTIC/<bucket>/"
        if dst.is_file():
            if dst.suffix == ".py":
                attic_origin[base.as_posix()] = r["src"]
        elif dst.exists():
            for q in dst.rglob("*.py"):
                attic_origin[(base / q.relative_to(dst)).as_posix()] = r["src"]

    print(f"scanning for references to {len(attic_origin)} attic'd modules ...")

    def looks_like_path(s: str) -> bool:
        """
        Reject prose that merely ends in '.py'. FeatureTemplates blocks carry long
        descriptive docstrings, and one of them ends a sentence with '..py', which
        is enough to make a naive endswith() try to stat a 400-character filename.
        """
        return (
            s.endswith(".py")
            and len(s) < 200
            and not any(c.isspace() for c in s)
        )

    def gone(relpath: str) -> str | None:
        """Reason this path is unavailable, or None if it still resolves."""
        try:
            if (ROOT / relpath).exists():
                return None
        except OSError:
            return None
        return attic_origin.get(relpath)

    findings = []
    for f in surviving_files():
        here = f.parent.relative_to(ROOT).as_posix()
        here = "" if here == "." else here + "/"

        if f.suffix == ".py":
            refs = _py_hard_refs(f)
            for ref in refs:
                # An `import X` resolves to a sibling module or one at the repo
                # root, the two places this project actually puts its modules.
                for cand in (f"{here}{ref}.py", f"{ref}.py"):
                    reason = gone(cand)
                    if reason:
                        findings.append((f, f"import {ref}", reason))
                        break
                # A string literal naming a .py is a subprocess target or a
                # spec_from_file_location path; resolve it the same two ways.
                if looks_like_path(ref):
                    norm = ref.replace("\\", "/").lstrip("./")
                    for cand in (norm, f"{here}{Path(norm).name}"):
                        reason = gone(cand)
                        if reason:
                            findings.append((f, f"path {ref}", reason))
                            break
        else:
            blob = next(iter(_script_hard_refs(f)))
            for hit in set(re.findall(r"[\w./\\-]+\.py", blob)):
                norm = hit.replace("\\", "/").lstrip("./")
                for cand in (norm, f"{here}{Path(norm).name}"):
                    reason = gone(cand)
                    if reason:
                        findings.append((f, f"script {hit}", reason))
                        break

    findings = sorted(set(findings), key=lambda t: str(t[0]))

    if not findings:
        print("OK. No surviving file imports or invokes anything in the attic.")
        return

    print(f"\n[FAIL] {len(findings)} live reference(s) into the attic:\n")
    for f, what, origin in findings:
        print(f"  {f.relative_to(ROOT).as_posix()}")
        print(f"      needs {what}   (attic'd as part of {origin})")
    print("\nrestore the entries above, or fix the references, before going further.")
    sys.exit(1)


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="show what would move")
    p.add_argument("--step", help="limit to one manifest step, e.g. 2a")
    p.add_argument("--size", action="store_true",
                   help="measure directory sizes (slow: minutes over experimental/)")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("apply", help="perform the moves")
    p.add_argument("--step", help="limit to one manifest step, e.g. 2a")
    p.add_argument("--size", action="store_true",
                   help="record exact byte counts in the manifest (slow)")
    p.set_defaults(func=cmd_apply)

    p = sub.add_parser("restore", help="move one entry back into the tree")
    p.add_argument("path", help="original path, as recorded in the src column")
    p.set_defaults(func=cmd_restore)

    p = sub.add_parser("verify", help="check nothing live still points into the attic")
    p.set_defaults(func=cmd_verify)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
