"""Autonomous experiment orchestrator TEMPLATE (overnight-solver skill).

Adapt the 3 marked spots: UNITS, run_produce(), run_evaluate()/parse().
Everything else is the robust scaffold that survived a real overnight run:
  - resumable (skip-if-done)         - try/except + continue (one failure != dead run)
  - incremental result writes        - per-unit timeouts
  - parallel eval pool + retry       - touches only its own scratch dir

Run unattended:  python -u <thisfile>.py   (launch with run_in_background, NO trailing &)
Remember to `mkdir -p` the scratch dir BEFORE redirecting a log into it.
"""
import glob, os, re, subprocess, sys, time
import pandas as pd

# ---- config (ADAPT) --------------------------------------------------------
OUT = "Data/_myexperiment"          # scratch dir — nothing live
RESULTS = f"{OUT}/results.csv"
PARALLEL = 3                         # eval concurrency; tune to avoid resource limits
EVAL_TIMEOUT = 3600                  # seconds per eval unit
PRODUCE_TIMEOUT = 1800              # seconds per produce unit
os.makedirs(OUT, exist_ok=True)

# Each unit = one piece of work. ADAPT to your problem.
UNITS = [{"tag": f"u{i}", "args": ["--example", str(i)]} for i in range(8)]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---- produce phase (e.g. train/build) — sequential, skip done -------------
def produced(tag):
    """ADAPT: True if this unit's artifact already exists."""
    return os.path.exists(f"{OUT}/{tag}/done.marker")


def run_produce(unit):
    """ADAPT: build the unit's artifact via subprocess (a script you wrote)."""
    tag, args = unit["tag"], unit["args"]
    with open(f"{OUT}/{tag}_produce.log", "w") as lf:
        subprocess.run([sys.executable, "your_worker.py", *args, "--out", f"{OUT}/{tag}"],
                       stdout=lf, stderr=subprocess.STDOUT, check=True, timeout=PRODUCE_TIMEOUT)


# ---- evaluate phase (e.g. backtest/measure) — parallel + retry ------------
def evaluate_cmd(unit):
    """ADAPT: return the subprocess argv that evaluates the unit and prints metrics."""
    return [sys.executable, "your_evaluator.py", "--in", f"{OUT}/{unit['tag']}"]


def parse(text):
    """ADAPT: extract metrics from an eval unit's stdout. Return {} on failure."""
    out = {}
    for key, pat in [("metric_a", r"Metric A:\s*([-\d.]+)"),
                     ("metric_b", r"Metric B:\s*([-\d.]+)")]:
        m = re.search(pat, text)
        if m:
            out[key] = float(m.group(1))
    return out


# ---------------------------------------------------------------------------
def append_results(rows):
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS, mode="a", header=not os.path.exists(RESULTS), index=False)


t0 = time.time()
log(f"START {len(UNITS)} units -> {OUT}")

# Phase A: produce (sequential; each uses all cores)
for u in UNITS:
    if produced(u["tag"]):
        log(f"produce {u['tag']}: exists, skip")
        continue
    log(f"produce {u['tag']} ...")
    try:
        run_produce(u)
    except Exception as e:
        log(f"produce {u['tag']} FAILED: {repr(e)[:140]}")
log(f"produce phase {(time.time()-t0)/60:.1f}m")

# Phase B: evaluate (parallel pool + retry on empty/crash; incremental)
done_tags = set()
if os.path.exists(RESULTS):
    try:
        done_tags = set(pd.read_csv(RESULTS)["tag"].astype(str))
    except Exception:
        pass

pending = [u for u in UNITS if u["tag"] not in done_tags and produced(u["tag"])]
running, results = {}, {}


def launch(u):
    lf = open(f"{OUT}/{u['tag']}_eval.log", "w")
    return subprocess.Popen(evaluate_cmd(u), stdout=lf, stderr=subprocess.STDOUT), lf


while pending or running:
    while pending and len(running) < PARALLEL:
        u = pending.pop(0)
        log(f"eval {u['tag']}: launch")
        p, lf = launch(u)
        running[p] = (u, lf, time.time())
    time.sleep(5)
    for p in [q for q in list(running) if q.poll() is not None]:
        u, lf, st = running.pop(p)
        lf.close()
        m = parse(open(f"{OUT}/{u['tag']}_eval.log").read())
        results[u["tag"]] = m
        if m:
            append_results([{"tag": u["tag"], **m}])
            log(f"eval {u['tag']}: {m}")
        else:
            log(f"eval {u['tag']}: EMPTY (will retry solo)")

# retry empties once, sequentially (avoids the parallel-resource crash)
for u in [u for u in UNITS if not results.get(u["tag"]) and produced(u["tag"])]:
    log(f"retry {u['tag']} solo")
    p, lf = launch(u)
    p.wait(); lf.close()
    m = parse(open(f"{OUT}/{u['tag']}_eval.log").read())
    if m:
        append_results([{"tag": u["tag"], **m}])
    results[u["tag"]] = m

# Phase C: aggregate -> report (ADAPT the analysis)
log("aggregating")
try:
    res = pd.read_csv(RESULTS)
    report = ["=" * 70, "EXPERIMENT REPORT", "=" * 70, res.describe().to_string(),
              "", res.to_string(index=False), "=" * 70]
except Exception as e:
    report = [f"aggregate failed: {e}"]
open(f"{OUT}/REPORT.txt", "w").write("\n".join(report))
log(f"DONE {(time.time()-t0)/60:.1f}m -> {OUT}/REPORT.txt")
