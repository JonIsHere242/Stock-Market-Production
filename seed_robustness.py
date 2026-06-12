"""
seed_robustness.py  --  how robust is the book-size sweet spot across model seeds?

The book-size curve on ONE prediction set is deterministic, but the predictions
themselves are one draw from a noise-dominated model (the +-15pp/month seed lottery).
This harness answers: across K INDEPENDENTLY-seeded single-seed models, does book ~8
stay near-optimal, and how wide is the band?

Pipeline (all isolated under Data/_seedrobust/ — never touches Data/RFpredictions,
Data/SimpleModel, or the live signal files):
  PHASE 1  generate K single-seed models (4.1__Predictor.py --no_tune --n_seeds 1
           --reuse, shared read-only prep cache), each -> its own RFpred_s<seed> dir.
           Sequential (each fit already uses all cores); resumable (skips a seed whose
           prediction dir is already populated).
  PHASE 2  parallel book-size sweep across ALL seed dirs (run_parallel_backtests.py
           --seeds_glob), full universe, deterministic.
  PHASE 3  aggregate: per book size, mean +- std of Sharpe/ann across seeds, plus how
           often each book is the per-seed winner / top-2. Writes SEED_ROBUSTNESS.txt.

Resumable end to end: rerun with the same args to continue. Use --phase to run a
single phase. NEVER runs the live broker.

Examples:
  python seed_robustness.py --n_seeds 12 --books 4,6,8,12,16,24
  python seed_robustness.py --phase agg            # rebuild report from existing sweep
"""
import argparse
import csv
import glob
import os
import statistics
import subprocess
import sys
import time

PY = sys.executable
ROOT = "Data/_seedrobust"
MODEL_DIR = f"{ROOT}/model"                 # shared (sequential) — holds the prep cache
SWEEP_DIR = f"{ROOT}/_sweep"
PRED_GLOB = f"{ROOT}/RFpred_s*"
REPORT = f"{ROOT}/SEED_ROBUSTNESS.txt"
PREDICTOR = "4.1__Predictor.py"
N_EXPECTED_TICKERS = 4000                   # a seed dir with >= this many parquets is "done"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def seed_dir(seed):
    return f"{ROOT}/RFpred_s{seed}"


def seed_done(seed):
    d = seed_dir(seed)
    return os.path.isdir(d) and len(glob.glob(f"{d}/*.parquet")) >= N_EXPECTED_TICKERS


# ── PHASE 1: generate single-seed prediction sets ────────────────────────────────
def gen_seeds(seeds, timeout):
    if not (os.path.exists(f"{MODEL_DIR}/PreparedData/train.parquet")
            and os.path.exists(f"{MODEL_DIR}/PreparedData/calib.parquet")):
        log(f"ERROR: prep cache missing under {MODEL_DIR}/PreparedData/. "
            f"Copy Data/XGBPipeline/PreparedData/{{train,calib}}.parquet there first.")
        return False
    os.makedirs(ROOT, exist_ok=True)
    for s in seeds:
        if seed_done(s):
            log(f"seed {s}: already done ({len(glob.glob(seed_dir(s)+'/*.parquet'))} preds) — skip")
            continue
        odir = seed_dir(s)
        cmd = [PY, "-u", PREDICTOR, "--no_tune", "--n_seeds", "1", "--seed", str(s),
               "--reuse", "--model_dir", MODEL_DIR, "--output_dir", odir]
        log(f"seed {s}: training -> {odir}")
        t0 = time.time()
        try:
            with open(f"{ROOT}/gen_s{s}.log", "w", encoding="utf-8") as lf:
                r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                   timeout=timeout, encoding="utf-8", errors="replace")
            n = len(glob.glob(f"{odir}/*.parquet"))
            ok = (r.returncode == 0 and n >= N_EXPECTED_TICKERS)
            log(f"seed {s}: {'OK' if ok else 'INCOMPLETE'} {n} preds in {(time.time()-t0)/60:.1f}m "
                f"(rc={r.returncode})")
        except subprocess.TimeoutExpired:
            log(f"seed {s}: TIMEOUT after {timeout}s — continuing")
        except Exception as e:
            log(f"seed {s}: FAIL {repr(e)[:160]} — continuing")
    done = [s for s in seeds if seed_done(s)]
    log(f"PHASE 1 complete: {len(done)}/{len(seeds)} seeds usable")
    return len(done) > 0


# ── PHASE 2: parallel book sweep across all seed dirs ────────────────────────────
def run_sweep(books, sample, workers, inner_workers, timeout):
    dirs = sorted(glob.glob(PRED_GLOB))
    if not dirs:
        log("ERROR: no seed prediction dirs found; run PHASE 1 first.")
        return False
    log(f"PHASE 2: sweeping books {books} across {len(dirs)} seed dirs "
        f"({len(books.split(','))*len(dirs)} backtests)")
    cmd = [PY, "-u", "run_parallel_backtests.py",
           "--seeds_glob", PRED_GLOB, "--book_sweep", books,
           "--sample", str(sample), "--workers", str(workers),
           "--inner_workers", str(inner_workers), "--timeout", str(timeout),
           "--out", SWEEP_DIR]
    try:
        subprocess.run(cmd, timeout=timeout * 200)  # generous outer cap
    except Exception as e:
        log(f"sweep launcher error: {repr(e)[:160]}")
    ok = os.path.exists(f"{SWEEP_DIR}/results.csv")
    log(f"PHASE 2 {'complete' if ok else 'FAILED'}")
    return ok


# ── PHASE 3: aggregate across seeds ──────────────────────────────────────────────
def aggregate():
    csv_path = f"{SWEEP_DIR}/results.csv"
    if not os.path.exists(csv_path):
        log(f"no sweep results at {csv_path}")
        return
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if str(r.get("ok")).lower() not in ("true", "1"):
                continue
            # label = RFpred_s<seed>_k<book>
            lab = r["label"]
            try:
                seed = lab.split("_k")[0].replace("RFpred_s", "")
                book = int(lab.split("_k")[1])
            except (IndexError, ValueError):
                continue
            def fl(k):
                try:
                    return float(r[k])
                except (TypeError, ValueError, KeyError):
                    return None
            rows.append({"seed": seed, "book": book, "ann": fl("ann"),
                         "sharpe": fl("sharpe"), "dd": fl("dd"), "trades": fl("trades")})

    if not rows:
        log("no usable rows to aggregate")
        return

    seeds = sorted({r["seed"] for r in rows})
    books = sorted({r["book"] for r in rows})

    # per-seed winner (by sharpe) and top-2 membership
    win_by_sharpe = {b: 0 for b in books}
    top2_by_sharpe = {b: 0 for b in books}
    win_by_ann = {b: 0 for b in books}
    for s in seeds:
        srows = [r for r in rows if r["seed"] == s and r["sharpe"] is not None]
        if not srows:
            continue
        ssorted = sorted(srows, key=lambda r: r["sharpe"], reverse=True)
        win_by_sharpe[ssorted[0]["book"]] += 1
        for r in ssorted[:2]:
            top2_by_sharpe[r["book"]] += 1
        asorted = sorted([r for r in srows if r["ann"] is not None],
                         key=lambda r: r["ann"], reverse=True)
        if asorted:
            win_by_ann[asorted[0]["book"]] += 1

    def stats(book, metric):
        vals = [r[metric] for r in rows if r["book"] == book and r[metric] is not None]
        if not vals:
            return (None, None, None, None, 0)
        mean = statistics.mean(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        return (mean, sd, min(vals), max(vals), len(vals))

    lines = ["=" * 96, "SEED ROBUSTNESS — book-size sweep across independent model seeds", "=" * 96,
             f"seeds: {len(seeds)}  ({', '.join(seeds)})", f"books: {books}",
             f"generated {time.strftime('%Y-%m-%d %H:%M')}", "",
             "Per book size, across seeds:", ""]
    hdr = (f"  {'book':>5}{'Sharpe mean':>13}{'+-sd':>8}{'min':>8}{'max':>8}"
           f"{'ann mean':>11}{'+-sd':>9}{'win%S':>8}{'top2%S':>8}{'win%R':>8}{'n':>4}")
    lines.append(hdr)
    lines.append("  " + "-" * 92)
    ns = len(seeds)
    for b in books:
        sm, ssd, smin, smax, n = stats(b, "sharpe")
        am, asd, *_ = stats(b, "ann")
        def f(v, w, p):
            return f"{v:>{w}.{p}f}" if v is not None else f"{'-':>{w}}"
        lines.append(
            f"  {b:>5}{f(sm,13,2)}{f(ssd,8,2)}{f(smin,8,2)}{f(smax,8,2)}"
            f"{f(am,11,1)}{f(asd,9,1)}"
            f"{100*win_by_sharpe[b]/ns:>7.0f}%{100*top2_by_sharpe[b]/ns:>7.0f}%"
            f"{100*win_by_ann[b]/ns:>7.0f}%{n:>4}")

    # verdict on book=8
    best_mean_sharpe = max(books, key=lambda b: (stats(b, "sharpe")[0] or -9))
    lines += ["", "  win%S  = how often this book is the per-seed Sharpe winner",
              "  top2%S = how often it lands in the per-seed Sharpe top-2",
              "  win%R  = how often it is the per-seed annualized-return winner", ""]
    sm8 = stats(8, "sharpe")
    lines += [f"  Best mean-Sharpe book: {best_mean_sharpe}",
              f"  Book=8: mean Sharpe {sm8[0]:.2f} +- {sm8[1]:.2f} (min {sm8[2]:.2f}, max {sm8[3]:.2f}), "
              f"per-seed Sharpe-winner {100*win_by_sharpe.get(8,0)/ns:.0f}%, "
              f"top-2 {100*top2_by_sharpe.get(8,0)/ns:.0f}%" if sm8[0] is not None else
              "  Book=8 not in sweep set."]
    lines += ["", "=" * 96, f"raw: {csv_path}", "=" * 96]
    report = "\n".join(lines)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report, flush=True)
    log(f"report -> {REPORT}")


def parse_seeds(spec, n):
    if spec:
        return [int(x) for x in spec.split(",")]
    base = 101
    return [base + i for i in range(n)]


def main():
    ap = argparse.ArgumentParser(description="Cross-seed book-size robustness.")
    ap.add_argument("--n_seeds", type=int, default=12, help="How many seeds (101,102,...).")
    ap.add_argument("--seeds", default=None, help="Explicit comma list of seeds (overrides --n_seeds).")
    ap.add_argument("--books", default="4,6,8,12,16,24", help="Book sizes to sweep.")
    ap.add_argument("--sample", type=float, default=100, help="Universe %% per backtest (100 = full, deterministic).")
    ap.add_argument("--workers", type=int, default=10, help="Concurrent backtests in the sweep.")
    ap.add_argument("--inner_workers", type=int, default=3)
    ap.add_argument("--gen_timeout", type=int, default=2400, help="Per-seed train+predict timeout (s).")
    ap.add_argument("--bt_timeout", type=int, default=3600, help="Per-backtest timeout (s).")
    ap.add_argument("--phase", choices=["all", "gen", "sweep", "agg"], default="all")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds, args.n_seeds)
    os.makedirs(ROOT, exist_ok=True)
    log(f"seeds={seeds} books={args.books} phase={args.phase}")

    if args.phase in ("all", "gen"):
        if not gen_seeds(seeds, args.gen_timeout) and args.phase == "all":
            log("no seeds generated; stopping.")
            return
    if args.phase in ("all", "sweep"):
        run_sweep(args.books, args.sample, args.workers, args.inner_workers, args.bt_timeout)
    if args.phase in ("all", "sweep", "agg"):
        aggregate()


if __name__ == "__main__":
    main()
