"""
map_pipeline.py -- generate a live "map of what is going on" for this project.

It documents every pipeline stage (script, role, inputs, outputs) and then
scans the filesystem for the ACTUAL artifacts each stage produces -- file
counts, total size, and newest modification time -- so the map always reflects
the real state of the repo, not a stale description.

Usage:
    python map_pipeline.py                 # print map + write PIPELINE_MAP.md
    python map_pipeline.py --no-write       # print only, don't write the file
    python map_pipeline.py --json           # also emit pipeline_map.json

Called automatically by  setup.ps1 -ColdStart  after a full pipeline run, but
safe to run any time on its own.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
#  Stage definitions  (order = execution order)
#  outputs: filesystem paths (relative to project root) this stage produces.
# ---------------------------------------------------------------------------
STAGES = [
    {
        "id": "1", "script": "1__TickerDownloader.py", "args": "--ImmediateDownload",
        "role": "Pull the current SEC ticker/CIK universe of tradable securities.",
        "inputs": ["SEC EDGAR (network)"],
        "outputs": ["Data/TickerCikData"],
    },
    {
        "id": "2", "script": "2__PriceDownloader.py", "args": "--RefreshMode",
        "role": "Download / refresh daily OHLCV price history per ticker (yfinance).",
        "inputs": ["Data/TickerCikData", "Yahoo Finance (network)"],
        "outputs": ["Data/PriceData"],
    },
    {
        "id": "3", "script": "3__AlphaSensitivity.py", "args": "--runpercent 100",
        "role": "Feature engineering -- technical / synthetic / matrix-power indicators.",
        "inputs": ["Data/PriceData"],
        "outputs": ["Data/ProcessedData"],
    },
    {
        "id": "4", "script": "4__Predictor.py", "args": "--predict_only  (or --runpercent 75 to retrain)",
        "role": "Train + calibrate the XGBoost classifier and score every ticker.",
        "inputs": ["Data/ProcessedData"],
        "outputs": ["Data/SimpleModel", "Data/RFpredictions"],
    },
    {
        "id": "5", "script": "5__NightlyBackTester.py", "args": "--force",
        "role": "Backtest the strategy and emit the day's buy signals + trade history.",
        "inputs": ["Data/RFpredictions", "Data/PriceData"],
        "outputs": ["_Buy_Signals.parquet", "trade_history.parquet", "Data/0__signals.parquet"],
    },
    {
        "id": "7", "script": "7__MacroFilter.py", "args": "(optional macro overlay)",
        "role": "LLM/macro overlay that can veto signals (needs Claud-API-KEY.txt).",
        "inputs": ["Data/0__signals.parquet", "Data/FRED", "Anthropic API"],
        "outputs": ["Data/0__signals.parquet"],
    },
    {
        "id": "9", "script": "9_SuperFastBroker.py", "args": "(live)",
        "role": "Execute the filtered signals live via Interactive Brokers (IBKR).",
        "inputs": ["_Buy_Signals.parquet", "IBKR TWS/Gateway"],
        "outputs": ["_Live_trades.parquet"],
    },
]

# Optional, independent data layer (free macro / fundamentals / positioning).
DATA_LAYER = {
    "script": "fetch_all_data.py  (+ fetchers/)",
    "role": "Idempotent pull of SEC / FRED / FINRA / CFTC / Treasury / KenFrench / Shiller data.",
    "outputs": ["Data/SEC", "Data/FRED", "Data/FINRA", "Data/CFTC_COT",
                "Data/Treasury", "Data/KenFrench", "Data/Shiller",
                "Data/Wikipedia", "Data/UnifiedPanel"],
}

# ---------------------------------------------------------------------------
#  Filesystem inventory
# ---------------------------------------------------------------------------
def _human(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024 or unit == "TB":
            return f"{nbytes:.0f}{unit}" if unit == "B" else f"{nbytes:.1f}{unit}"
        nbytes /= 1024
    return f"{nbytes:.1f}TB"


def inventory(rel_path: str) -> dict:
    """Return {exists, n_files, size, newest} for a file or directory."""
    p = ROOT / rel_path
    if not p.exists():
        return {"exists": False, "n_files": 0, "size": 0, "newest": None}
    if p.is_file():
        st = p.stat()
        return {"exists": True, "n_files": 1, "size": st.st_size,
                "newest": dt.datetime.fromtimestamp(st.st_mtime)}
    files = [f for f in p.rglob("*") if f.is_file()]
    if not files:
        return {"exists": True, "n_files": 0, "size": 0, "newest": None}
    size = sum(f.stat().st_size for f in files)
    newest = max(dt.datetime.fromtimestamp(f.stat().st_mtime) for f in files)
    return {"exists": True, "n_files": len(files), "size": size, "newest": newest}


def _fmt_inv(inv: dict) -> str:
    if not inv["exists"]:
        return "MISSING (not produced yet)"
    if inv["n_files"] == 0:
        return "empty"
    age = ""
    if inv["newest"]:
        hrs = (dt.datetime.now() - inv["newest"]).total_seconds() / 3600
        age = f", newest {inv['newest']:%Y-%m-%d %H:%M} ({hrs:.0f}h ago)"
    return f"{inv['n_files']} file(s), {_human(inv['size'])}{age}"


# ---------------------------------------------------------------------------
#  Rendering
# ---------------------------------------------------------------------------
def build_markdown() -> str:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L = []
    L.append("# Pipeline Map\n")
    L.append(f"_Auto-generated by `map_pipeline.py` on {now}. Re-run any time to refresh._\n")

    # Flow diagram
    L.append("## Flow\n")
    L.append("```mermaid")
    L.append("flowchart TD")
    L.append("    D[fetch_all_data.py<br/>SEC/FRED/FINRA/CFTC...] -.optional overlay.-> S7")
    for i, s in enumerate(STAGES):
        nid = f"S{s['id']}"
        label = f"{s['id']} {s['script']}".replace("_", "&#95;")
        L.append(f"    {nid}[\"{label}\"]")
    chain = [s for s in STAGES if s["id"] in {"1", "2", "3", "4", "5"}]
    L.append("    " + " --> ".join(f"S{s['id']}" for s in chain))
    L.append("    S5 --> S7 --> S9")
    L.append("```\n")

    # Per-stage detail
    L.append("## Stages\n")
    for s in STAGES:
        L.append(f"### Stage {s['id']} — `{s['script']}`")
        L.append(f"- **Run:** `python {s['script']} {s['args']}`")
        L.append(f"- **Role:** {s['role']}")
        L.append(f"- **Inputs:** {', '.join('`'+i+'`' if '/' in i else i for i in s['inputs'])}")
        L.append("- **Outputs:**")
        for o in s["outputs"]:
            L.append(f"    - `{o}` — {_fmt_inv(inventory(o))}")
        L.append("")

    # Data layer
    L.append("## Independent data layer\n")
    L.append(f"- **Run:** `python {DATA_LAYER['script'].split()[0]}`")
    L.append(f"- **Role:** {DATA_LAYER['role']}")
    L.append("- **Outputs:**")
    for o in DATA_LAYER["outputs"]:
        L.append(f"    - `{o}` — {_fmt_inv(inventory(o))}")
    L.append("")

    # Layout note
    L.append("## Repo layout\n")
    L.append("- `auxiliary/` — `0__*` pre-pipeline EDA / analysis scripts (not in the nightly run).")
    L.append("- `showcase/` — portfolio HTML/PDF artifacts.")
    L.append("- `.claude/docs/` — session notes, handoffs, replication reports.")
    L.append("- `fetchers/` — per-source data fetchers used by `fetch_all_data.py`.")
    L.append("- `setup.ps1` / `requirements.txt` — environment bootstrap.")
    L.append("- `run_pipeline.py` — freshness-aware runner for stages 2–5.")
    L.append("")
    return "\n".join(L)


def build_json() -> dict:
    out = {"generated": dt.datetime.now().isoformat(), "stages": []}
    for s in STAGES:
        out["stages"].append({
            **{k: s[k] for k in ("id", "script", "args", "role", "inputs")},
            "outputs": {o: _fmt_inv(inventory(o)) for o in s["outputs"]},
        })
    out["data_layer"] = {o: _fmt_inv(inventory(o)) for o in DATA_LAYER["outputs"]}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-write", action="store_true", help="print only, don't write PIPELINE_MAP.md")
    ap.add_argument("--json", action="store_true", help="also write pipeline_map.json")
    a = ap.parse_args()

    md = build_markdown()
    print(md)

    if not a.no_write:
        (ROOT / "PIPELINE_MAP.md").write_text(md, encoding="utf-8")
        print(f"\n[written] {ROOT / 'PIPELINE_MAP.md'}")
    if a.json:
        (ROOT / "pipeline_map.json").write_text(json.dumps(build_json(), indent=2), encoding="utf-8")
        print(f"[written] {ROOT / 'pipeline_map.json'}")


if __name__ == "__main__":
    main()
