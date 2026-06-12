"""
param_stress.py  --  is book ~8 OVERFIT to the other strategy params?

The seed-robustness run showed book 8 wins across model seeds. This asks the other
half: does the book sweet-spot survive when we PERTURB the other strategy levers
(probability gate, hold-time, take-profit, position sizing)? If the best book jumps
around as those change, 8 is entangled/overfit. If the argmax book stays in ~6-12
across every param regime, it's a genuine structural sweet spot.

Grid = {param regimes} x {book sizes}, each a full-universe fast backtest, run in
parallel across all cores (each its own crash-isolated subprocess). Resumable,
incremental, writes a regime x book Sharpe matrix + a per-regime best-book column.

NEVER touches the live broker; read-only on the prediction dir.

Examples:
  python param_stress.py                                  # ensemble preds, 9 regimes x 6 books
  python param_stress.py --data_dir Data/RFpredictions    # on the live single-model preds
  python param_stress.py --seeds_glob "Data/_seedrobust/RFpred_s10*"  # fold in a few seeds
"""
import argparse
import concurrent.futures
import csv
import glob
import os
import re
import statistics
import subprocess
import sys
import threading
import time

PY = sys.executable
ANSI = re.compile(r"\x1b\[[0-9;]*m")
_print_lock = threading.Lock()
_csv_lock = threading.Lock()

# Each regime = baseline EXCEPT the listed override(s). Only levers the strategy
# ACTUALLY responds to (a smoke test proved up_prob/take_profit/risk_per_trade are
# non-binding here): entry selectivity (BT_PLOW/BT_PHIGH percentile gate in can_buy)
# and hold-time (position_timeout). Baseline = live defaults (PLOW 65, PHIGH 98,
# timeout 5). Each regime is {flags, env}.
REGIMES = {
    "baseline":      {"flags": [],                            "env": {}},
    "timeout_short": {"flags": ["--position_timeout", "3"],   "env": {}},
    "timeout_long":  {"flags": ["--position_timeout", "8"],   "env": {}},
    "phigh_strict":  {"flags": [],                            "env": {"BT_PHIGH": "95"}},
    "phigh_loose":   {"flags": [],                            "env": {"BT_PHIGH": "99.5"}},
    # NOTE: BT_PLOW (entry floor) and up_prob/take_profit/risk_per_trade were proven
    # non-binding at full universe — dropped so the hour isn't spent on no-op cells.
}

METRIC_PATTERNS = [
    ("ann",     r"Annualized Return %:\s*([-\d.]+)"),
    ("sharpe",  r"Sharpe Ratio:\s*([-\d.]+)"),
    ("dd",      r"Max Drawdown %:\s*([-\d.]+)"),
    ("trades",  r"Total Trades:\s*([-\d.]+)"),
]
FIELDS = ["label", "pred", "regime", "book", "ok", "minutes"] + [k for k, _ in METRIC_PATTERNS]


def log(m):
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def parse_metrics(txt):
    txt = ANSI.sub("", txt)
    out = {}
    for k, p in METRIC_PATTERNS:
        m = re.search(p, txt)
        if m:
            try:
                out[k] = float(m.group(1))
            except ValueError:
                pass
    return out


def build_jobs(preds, books):
    jobs = []
    for pdir in preds:
        pname = os.path.basename(pdir.rstrip("/\\"))
        for rname, spec in REGIMES.items():
            for b in books:
                jobs.append({"pred": pdir, "pname": pname, "regime": rname,
                             "flags": spec["flags"], "env": spec["env"], "book": b,
                             "label": f"{pname}|{rname}|k{b}"})
    return jobs


def run_one(job, args):
    label = job["label"]
    safe = label.replace("|", "_")
    cmd = [PY, "-u", args.backtester, "--data_dir", job["pred"], "--sample", str(args.sample),
           "--max_positions", str(job["book"])] + job["flags"]
    env = dict(os.environ)
    env["BT_INNER_WORKERS"] = str(args.inner_workers)
    env.update(job.get("env", {}))          # regime env overrides (BT_PLOW/BT_PHIGH)
    if args.sample < 100:
        env["BT_SAMPLE_SEED"] = str(args.sample_seed)
    row = {"label": label, "pred": job["pname"], "regime": job["regime"],
           "book": job["book"], "ok": False}
    t0 = time.time()
    log(f"START {label}")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout,
                           env=env, encoding="utf-8", errors="replace")
        row["minutes"] = round((time.time() - t0) / 60, 2)
        with open(os.path.join(args.out, f"bt_{safe}.log"), "w", encoding="utf-8") as lf:
            lf.write(r.stdout or "")
        res = parse_metrics(r.stdout or "")
        row.update(res)
        row["ok"] = res.get("sharpe") is not None
        log(f"DONE  {label}: {row['minutes']:.1f}m sharpe={res.get('sharpe')} ann={res.get('ann')}")
    except subprocess.TimeoutExpired:
        row["minutes"] = round((time.time() - t0) / 60, 2)
        log(f"TIMEOUT {label}")
    except Exception as e:
        row["minutes"] = round((time.time() - t0) / 60, 2)
        log(f"FAIL  {label}: {repr(e)[:140]}")
    return row


def load_done(csv_path):
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if str(r.get("ok")).lower() in ("true", "1"):
                    done.add(r["label"])
    return done


def aggregate(csv_path, out, books):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if str(r.get("ok")).lower() not in ("true", "1"):
                continue
            try:
                r["book"] = int(r["book"])
                r["sharpe"] = float(r["sharpe"]) if r.get("sharpe") not in (None, "", "None") else None
                r["ann"] = float(r["ann"]) if r.get("ann") not in (None, "", "None") else None
            except (ValueError, TypeError):
                continue
            rows.append(r)
    if not rows:
        log("no rows to aggregate")
        return

    preds = sorted({r["pred"] for r in rows})
    regimes = [rn for rn in REGIMES if any(r["regime"] == rn for r in rows)]
    bks = sorted({r["book"] for r in rows})

    def cell(pred, regime, book, metric="sharpe"):
        for r in rows:
            if r["pred"] == pred and r["regime"] == regime and r["book"] == book:
                return r.get(metric)
        return None

    lines = ["=" * 100,
             "PARAM STRESS — is book ~8 overfit to the other strategy params?", "=" * 100,
             f"prediction set(s): {', '.join(preds)}",
             f"books: {bks}   regimes: {len(regimes)}   generated {time.strftime('%Y-%m-%d %H:%M')}", ""]

    best_book_counts = {}
    eight_ranks = []
    for pred in preds:
        lines += [f"### preds = {pred}  —  SHARPE by regime (rows) x book (cols):", ""]
        hdr = "  " + f"{'regime':<15}" + "".join(f"{('k'+str(b)):>8}" for b in bks) + f"{'best':>8}{'k8 rank':>9}"
        lines.append(hdr)
        lines.append("  " + "-" * (15 + 8 * len(bks) + 17))
        for rn in regimes:
            vals = {b: cell(pred, rn, b) for b in bks}
            present = {b: v for b, v in vals.items() if v is not None}
            if present:
                best_b = max(present, key=lambda b: present[b])
                best_book_counts[best_b] = best_book_counts.get(best_b, 0) + 1
                order = sorted(present, key=lambda b: present[b], reverse=True)
                k8rank = (order.index(8) + 1) if 8 in order else None
                if k8rank:
                    eight_ranks.append(k8rank)
            else:
                best_b, k8rank = None, None
            row_s = "  " + f"{rn:<15}" + "".join(
                (f"{vals[b]:>8.2f}" if vals[b] is not None else f"{'-':>8}") for b in bks)
            row_s += f"{('k'+str(best_b)) if best_b else '-':>8}"
            row_s += f"{(str(k8rank)) if k8rank else '-':>9}"
            lines.append(row_s)
        lines.append("")

    # MARGINAL: book size across ALL (pred x regime) cells — the headline robustness view
    cells = {}
    for r in rows:
        cells.setdefault((r["pred"], r["regime"]), {})[r["book"]] = r.get("sharpe")
    perbook = {b: [] for b in bks}
    mwins = {b: 0 for b in bks}
    mtop2 = {b: 0 for b in bks}
    mrank = {b: [] for b in bks}
    for _key, bd in cells.items():
        present = {b: v for b, v in bd.items() if v is not None}
        if not present:
            continue
        order = sorted(present, key=lambda b: present[b], reverse=True)
        mwins[order[0]] += 1
        for b in order[:2]:
            mtop2[b] += 1
        for i, b in enumerate(order):
            mrank[b].append(i + 1)
        for b, v in present.items():
            perbook[b].append(v)
    lines += ["-" * 100,
              f"MARGINAL — book size across all {len(cells)} (pred x regime) cells:", "",
              "  " + f"{'book':>5}{'Sharpe mn':>11}{'+-sd':>8}{'min':>8}{'max':>8}"
              f"{'wins':>7}{'top2':>7}{'rank_mn':>9}{'n':>4}"]
    for b in bks:
        vals = perbook[b]
        if vals:
            mean = statistics.mean(vals)
            sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            rk = statistics.mean(mrank[b]) if mrank[b] else 0.0
            lines.append("  " + f"{b:>5}{mean:>11.2f}{sd:>8.2f}{min(vals):>8.2f}{max(vals):>8.2f}"
                         f"{mwins[b]:>7}{mtop2[b]:>7}{rk:>9.2f}{len(vals):>4}")
    lines.append("")

    # verdict
    lines += ["-" * 100, "VERDICT:", ""]
    lines.append(f"  best-book frequency across all regimes x preds: "
                 + ", ".join(f"k{b}:{best_book_counts[b]}" for b in sorted(best_book_counts)))
    if eight_ranks:
        lines.append(f"  book=8 rank across regimes: mean {statistics.mean(eight_ranks):.2f}  "
                     f"median {statistics.median(eight_ranks):.0f}  "
                     f"best {min(eight_ranks)}  worst {max(eight_ranks)}  "
                     f"(of {len(bks)} books)")
        top2 = sum(1 for r in eight_ranks if r <= 2)
        lines.append(f"  book=8 in top-2 for {top2}/{len(eight_ranks)} regimes "
                     f"({100*top2/len(eight_ranks):.0f}%)")
    winners = sorted(best_book_counts)
    lines.append("")
    lines.append(f"  argmax-book ranges k{winners[0]}..k{winners[-1]} across regimes — "
                 + ("CLUSTERED near the plateau => NOT overfit." if winners and winners[0] >= 6 and winners[-1] <= 16
                    else "spread is wide — inspect."))
    lines += ["", "=" * 100, f"raw: {csv_path}", "=" * 100]
    report = "\n".join(lines)
    with open(os.path.join(out, "PARAM_STRESS_REPORT.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report, flush=True)


def main():
    ap = argparse.ArgumentParser(description="Param-robustness stress test around book size.")
    ap.add_argument("--data_dir", default="Data/RFpredictions_ens")
    ap.add_argument("--preds", default=None, help="Explicit comma list of prediction dirs (overrides --data_dir/--seeds_glob).")
    ap.add_argument("--seeds_glob", default=None, help="Run across multiple pred dirs (folds in seed noise).")
    ap.add_argument("--books", default="4,6,8,10,12,16")
    ap.add_argument("--sample", type=float, default=100)
    ap.add_argument("--sample_seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--inner_workers", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--backtester", default="5__NightlyBackTester_fast.py")
    ap.add_argument("--out", default="Data/_paramstress")
    ap.add_argument("--agg_only", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "results.csv")
    books = [int(x) for x in args.books.split(",")]

    if args.agg_only:
        aggregate(csv_path, args.out, books)
        return

    if args.preds:
        preds = [d for d in args.preds.split(",") if os.path.isdir(d)]
    elif args.seeds_glob:
        preds = sorted([d for d in glob.glob(args.seeds_glob) if os.path.isdir(d)])
    else:
        preds = [args.data_dir]
    jobs = build_jobs(preds, books)
    done = load_done(csv_path)
    pending = [j for j in jobs if j["label"] not in done]
    log(f"{len(jobs)} cells ({len(done)} done, {len(pending)} pending) | {len(REGIMES)} regimes x "
        f"{len(books)} books x {len(preds)} preds | workers={args.workers}")

    write_header = not os.path.exists(csv_path)
    f_csv = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_csv, fieldnames=FIELDS)
    if write_header:
        writer.writeheader(); f_csv.flush()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_one, j, args): j for j in pending}
            for fut in concurrent.futures.as_completed(futs):
                row = fut.result()
                with _csv_lock:
                    writer.writerow({k: row.get(k) for k in FIELDS}); f_csv.flush()
    finally:
        f_csv.close()
    aggregate(csv_path, args.out, books)


if __name__ == "__main__":
    main()
