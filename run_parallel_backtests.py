"""
run_parallel_backtests.py  --  run many backtests CONCURRENTLY and rank them.

The backtest is the bottleneck of the iteration loop, and a single cerebro.run is
single-threaded Python. This harness fans N backtests across cores AT ONCE -- each
in its own subprocess (crash-isolated) -- parses each one's metrics, and writes a
ranked report. Built for two studies:

  * BOOK-SIZE SWEEP  (the breadth/variance lever): same predictions, vary the book
    size top-4 -> top-50 and see how risk-adjusted return + the noise band move.
  * MULTI-SEED / MULTI-MODEL TOURNAMENT: one prediction dir per seed/model, ranked
    head to head (optionally x book sizes).

Design (follows the overnight-solver method): resumable (skips jobs already done),
incremental (writes results.csv as each job finishes), crash-tolerant (try/except
+ per-job timeout, one bad job never kills the run), isolated (writes only under
--out), non-interactive.

Each job runs 5__NightlyBackTester_fast.py WITHOUT --force (finviz export = ~10x
slower) and read-only on the prediction dir. NEVER touches the live broker.

Examples
--------
  # Book-size sweep on the live predictions:
  python run_parallel_backtests.py --book_sweep 4,8,12,16,24,32,40,50

  # Sweep a candidate model's predictions, quick smoke on a 25% sample:
  python run_parallel_backtests.py --data_dir Data/RFpredictions_ens --book_sweep 4,12,50 --sample 25

  # Multi-seed tournament: one dir per seed, each swept over a few book sizes:
  python run_parallel_backtests.py --seeds_glob "Data/RFpred_seed*" --book_sweep 4,20,50

Memory note: each full-universe job holds ~2-3 GB. On 67 GB / 16 cores, ~12 parallel
jobs is safe. Raise --workers if you have headroom; lower it if you sweep big books.
"""
import argparse
import concurrent.futures
import csv
import glob
import os
import re
import subprocess
import sys
import threading
import time

PY = sys.executable
ANSI = re.compile(r"\x1b\[[0-9;]*m")
_print_lock = threading.Lock()
_csv_lock = threading.Lock()

# Columns harvested from the backtester's console output. The core five
# (ann/sharpe/dd/trades/total_ret) are the same regexes the speedup validator
# proved out; the rest are best-effort (skipped silently if not present).
METRIC_PATTERNS = [
    ("total_ret",     r"Total Return %:\s*([-\d.]+)"),
    ("ann",           r"Annualized Return %:\s*([-\d.]+)"),
    ("sharpe",        r"Sharpe Ratio:\s*([-\d.]+)"),
    ("sortino",       r"Sortino Ratio:\s*([-\d.]+)"),
    ("calmar",        r"Calmar Ratio:\s*([-\d.]+)"),
    ("dd",            r"Max Drawdown %:\s*([-\d.]+)"),
    ("trades",        r"Total Trades:\s*([-\d.]+)"),
    ("win",           r"Win Rate \(after fees\) %:\s*([-\d.]+)"),
    ("profit_factor", r"Profit Factor:\s*([-\d.]+)"),
]
FIELDNAMES = ["label", "data_dir", "book", "minutes", "ok"] + [k for k, _ in METRIC_PATTERNS]


def log(msg):
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_metrics(txt):
    txt = ANSI.sub("", txt)
    out = {}
    for key, pat in METRIC_PATTERNS:
        m = re.search(pat, txt)
        if m:
            try:
                out[key] = float(m.group(1))
            except ValueError:
                pass
    return out


def build_jobs(args):
    """Cartesian product of {prediction dirs} x {book sizes}."""
    dirs = []
    if args.seeds_glob:
        dirs = [d for d in sorted(glob.glob(args.seeds_glob)) if os.path.isdir(d)]
        if not dirs:
            log(f"WARNING: --seeds_glob '{args.seeds_glob}' matched no directories; "
                f"falling back to --data_dir {args.data_dir}")
    if not dirs:
        dirs = [args.data_dir]

    books = [int(x) for x in args.book_sweep.split(",")] if args.book_sweep else [None]

    jobs = []
    for d in dirs:
        base = os.path.basename(d.rstrip("/\\")) or "preds"
        for b in books:
            label = base + (f"_k{b}" if b is not None else "")
            jobs.append({"data_dir": d, "book": b, "label": label})
    return jobs


def run_one(job, args):
    """Run a single backtest subprocess, parse + return its result row."""
    label = job["label"]
    log_path = os.path.join(args.out, f"bt_{label}.log")
    cmd = [PY, "-u", args.backtester, "--data_dir", job["data_dir"], "--sample", str(args.sample)]
    if job["book"] is not None:
        cmd += ["--max_positions", str(job["book"])]

    env = dict(os.environ)
    env["BT_INNER_WORKERS"] = str(args.inner_workers)  # cap inner loader pool per job
    if args.sample < 100:
        env["BT_SAMPLE_SEED"] = str(args.sample_seed)  # same universe across jobs

    row = {"label": label, "data_dir": job["data_dir"], "book": job["book"], "ok": False}
    t0 = time.time()
    log(f"START {label}: book={job['book']} dir={job['data_dir']}")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=args.timeout, env=env, encoding="utf-8", errors="replace")
        row["minutes"] = round((time.time() - t0) / 60, 2)
        with open(log_path, "w", encoding="utf-8") as lf:
            lf.write(r.stdout or "")
            lf.write("\n=====STDERR=====\n")
            lf.write(r.stderr or "")
        res = parse_metrics(r.stdout or "")
        row.update(res)
        row["ok"] = res.get("ann") is not None
        if row["ok"]:
            log(f"DONE  {label}: {row['minutes']:.1f}m  ann={res.get('ann')}  "
                f"sharpe={res.get('sharpe')}  dd={res.get('dd')}  trades={res.get('trades')}")
        else:
            log(f"DONE  {label}: {row['minutes']:.1f}m  but NO metrics parsed "
                f"(rc={r.returncode}) -> see {log_path}")
    except subprocess.TimeoutExpired:
        row["minutes"] = round((time.time() - t0) / 60, 2)
        log(f"TIMEOUT {label} after {args.timeout}s")
    except Exception as e:  # noqa: keep the run alive on any single-job failure
        row["minutes"] = round((time.time() - t0) / 60, 2)
        log(f"FAIL  {label}: {repr(e)[:160]}")
    return row


def load_done_labels(csv_path):
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if str(r.get("ok")).lower() in ("true", "1"):
                    done.add(r["label"])
    return done


def read_all_rows(csv_path):
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                for k, _ in METRIC_PATTERNS:
                    if r.get(k) not in (None, "", "None"):
                        try:
                            r[k] = float(r[k])
                        except ValueError:
                            r[k] = None
                    else:
                        r[k] = None
                rows.append(r)
    # de-dup by label, keep the last (most recent) row
    by_label = {}
    for r in rows:
        by_label[r["label"]] = r
    return list(by_label.values())


def write_report(csv_path, out_dir):
    rows = [r for r in read_all_rows(csv_path) if str(r.get("ok")).lower() in ("true", "1")]
    lines = ["=" * 92, "PARALLEL BACKTEST REPORT", "=" * 92,
             f"{len(rows)} successful jobs   ({time.strftime('%Y-%m-%d %H:%M')})", ""]

    def fnum(v, w, p):
        try:
            return f"{float(v):>{w}.{p}f}"
        except (TypeError, ValueError):
            return f"{'-':>{w}}"

    header = (f"  {'label':<28}{'book':>5}{'ann%':>9}{'Sharpe':>8}{'Sortino':>9}"
              f"{'maxDD%':>8}{'trades':>8}{'win%':>7}{'totRet%':>10}{'min':>7}")

    def table(sorted_rows, title):
        out = [title, header, "  " + "-" * 90]
        for r in sorted_rows:
            out.append(
                f"  {r['label'][:28]:<28}"
                f"{(str(r.get('book')) if r.get('book') not in (None,'','None') else '-'):>5}"
                f"{fnum(r.get('ann'),9,1)}{fnum(r.get('sharpe'),8,2)}{fnum(r.get('sortino'),9,2)}"
                f"{fnum(r.get('dd'),8,1)}{fnum(r.get('trades'),8,0)}{fnum(r.get('win'),7,1)}"
                f"{fnum(r.get('total_ret'),10,1)}{fnum(r.get('minutes'),7,1)}")
        return out

    def keyf(metric):
        def k(r):
            v = r.get(metric)
            try:
                return float(v)
            except (TypeError, ValueError):
                return float("-inf")
        return k

    lines += table(sorted(rows, key=keyf("sharpe"), reverse=True), "RANKED BY SHARPE (risk-adjusted):")
    lines += [""]
    lines += table(sorted(rows, key=keyf("ann"), reverse=True), "RANKED BY ANNUALIZED RETURN:")

    # If a single dir was swept over book sizes, show the breadth curve explicitly.
    dirs = {r["data_dir"] for r in rows}
    if len(dirs) == 1 and any(r.get("book") not in (None, "", "None") for r in rows):
        curve = sorted(rows, key=lambda r: float(r.get("book") or 0))
        lines += ["", "BOOK-SIZE CURVE (breadth lever, ordered by book size):"]
        lines += table(curve, "")

    lines += ["", "=" * 92,
              f"raw rows: {csv_path}   per-job logs: {out_dir}/bt_<label>.log", "=" * 92]
    report = "\n".join(lines)
    rep_path = os.path.join(out_dir, "PARALLEL_REPORT.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report, flush=True)
    log(f"report -> {rep_path}")


def main():
    ap = argparse.ArgumentParser(description="Run many backtests in parallel and rank them.")
    ap.add_argument("--data_dir", default="Data/RFpredictions",
                    help="Prediction dir to backtest (read-only). Default = live signals.")
    ap.add_argument("--seeds_glob", default=None,
                    help="Glob of multiple prediction dirs for a multi-seed/model tournament, "
                         "e.g. \"Data/RFpred_seed*\". Each dir is one row (x book sizes).")
    ap.add_argument("--book_sweep", default=None,
                    help="Comma list of book sizes (max_positions), e.g. 4,8,12,16,24,32,50. "
                         "Omit to use the strategy default.")
    ap.add_argument("--sample", type=float, default=100,
                    help="Percent of tickers per job (pass-through). Use small for a smoke test.")
    ap.add_argument("--sample_seed", type=int, default=42,
                    help="Seed for ticker sampling (BT_SAMPLE_SEED). With <100%% sample this "
                         "makes every job draw the SAME universe so book sizes compare fairly. "
                         "Ignored at --sample 100 (full universe is already deterministic).")
    ap.add_argument("--workers", type=int, default=min(12, (os.cpu_count() or 8)),
                    help="Concurrent backtests. ~2-3 GB each; keep workers*3GB under free RAM.")
    ap.add_argument("--inner_workers", type=int, default=3,
                    help="Inner data-load pool width per job (sets BT_INNER_WORKERS). "
                         "workers*inner_workers ~= core count is a good target.")
    ap.add_argument("--timeout", type=int, default=3600, help="Per-job timeout in seconds.")
    ap.add_argument("--backtester", default="5__NightlyBackTester_fast.py",
                    help="Backtester script to invoke per job.")
    ap.add_argument("--out", default="Data/_parallel", help="Output dir (csv, logs, report).")
    ap.add_argument("--fresh", action="store_true",
                    help="Ignore existing results.csv and rerun everything.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "results.csv")

    if args.fresh and os.path.exists(csv_path):
        os.replace(csv_path, csv_path + f".bak_{time.strftime('%Y%m%d_%H%M%S')}")

    jobs = build_jobs(args)
    done = set() if args.fresh else load_done_labels(csv_path)
    pending = [j for j in jobs if j["label"] not in done]

    log(f"{len(jobs)} jobs ({len(done)} already done, {len(pending)} pending) | "
        f"workers={args.workers} inner={args.inner_workers} sample={args.sample} "
        f"backtester={args.backtester}")
    if not pending:
        log("nothing to run; writing report from existing results.")
        write_report(csv_path, args.out)
        return

    write_header = not os.path.exists(csv_path)
    f_csv = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_csv, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()
        f_csv.flush()

    t_start = time.time()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_one, j, args): j for j in pending}
            for fut in concurrent.futures.as_completed(futs):
                row = fut.result()
                with _csv_lock:
                    writer.writerow({k: row.get(k) for k in FIELDNAMES})
                    f_csv.flush()
    finally:
        f_csv.close()

    log(f"all jobs finished in {(time.time()-t_start)/60:.1f} min")
    write_report(csv_path, args.out)


if __name__ == "__main__":
    main()
