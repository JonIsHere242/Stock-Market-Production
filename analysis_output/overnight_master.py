"""OVERNIGHT MASTER — chains PART 2 + PART 3 after PART 1, fully autonomous.

PART 1 (prod-params seed-ensemble) is launched separately and running. This master
waits for its report, then runs PART 2 (regularized-params ensemble) and PART 3
(16-seed prod ensemble), then assembles MORNING_REPORT.txt. Guarantees the cores
stay busy through the night without depending on notification timing. Light poll
(sleeps) until PART 1 done — no CPU contention. Does NOT touch the live broker or
anything live; only runs the predictor (-> temp dirs) and the backtester COPY.
"""
import os, subprocess, sys, time

PY = sys.executable


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] [MASTER] {m}", flush=True)


def wait_report(path, max_hr=4):
    log(f"waiting for {path} ...")
    t0 = time.time()
    while not os.path.exists(path):
        if time.time() - t0 > max_hr * 3600:
            log(f"TIMEOUT waiting for {path}; proceeding")
            return False
        time.sleep(120)
    log(f"found {path} after {(time.time()-t0)/60:.1f}m")
    return True


def run_part(name, script, logpath):
    log(f"START {name}: {script}")
    os.makedirs(os.path.dirname(logpath), exist_ok=True)
    try:
        with open(logpath, "w") as lf:
            subprocess.run([PY, script], stdout=lf, stderr=subprocess.STDOUT,
                           check=False, timeout=4 * 3600)
        log(f"DONE {name}")
    except Exception as e:
        log(f"{name} ERROR: {repr(e)[:160]}")


t0 = time.time()
# 1) wait for PART 1 (already running standalone)
wait_report("Data/_ensemble/ENSEMBLE_REPORT.txt", max_hr=4)

# 2) PART 2 — regularized-params ensemble
run_part("PART2-reg", "analysis_output/overnight_ensemble_reg.py", "Data/_ensemble_reg/run.log")

# 3) PART 3 — 16-seed prod ensemble (extend variance curve)
run_part("PART3-16seed", "analysis_output/overnight_part3.py", "Data/_ensemble/part3.log")

# 4) assemble MORNING_REPORT
log("assembling MORNING_REPORT")
sections = [
    ("PROD-PARAMS SEED-ENSEMBLE (PART 1: single-seed dist vs ens 2/4/8)", "Data/_ensemble/ENSEMBLE_REPORT.txt"),
    ("PROD-PARAMS 16-SEED EXTENSION (PART 3: ens 12/16)", "Data/_ensemble/ENSEMBLE_REPORT_part3.txt"),
    ("REGULARIZED-PARAMS SEED-ENSEMBLE (PART 2: ens 2/4/8)", "Data/_ensemble_reg/ENSEMBLE_REPORT.txt"),
]
out = ["#" * 92,
       "MORNING REPORT — seed-ensemble investigation (overnight, autonomous)",
       f"generated {time.strftime('%Y-%m-%d %H:%M')}   total {(time.time()-t0)/60:.0f}m",
       "#" * 92, "",
       "THESIS: the model is ~+/-15pp/month seed-sensitive (one draw from a high-variance",
       "distribution). Averaging predict_proba across N seeds reduces that variance BY",
       "CONSTRUCTION -> expect better Sharpe / lower drawdown / steadier months than any",
       "single seed. Also compares prod (depth-8 Optuna) vs regularized (depth-5) params,",
       "both ensembled, to finally resolve the params question with the noise removed.", "",
       "HOW TO READ: in each section, compare the ensemble rows to the SINGLE-SEED",
       "DISTRIBUTION (mean/std/range). Ship a seed-ensemble if ensemble Sharpe > single",
       "mean Sharpe AND ensemble maxDD < single mean maxDD AND OOS_vol drops with seeds.", ""]
for title, path in sections:
    out += ["=" * 92, title, "=" * 92]
    out.append(open(path).read() if os.path.exists(path) else f"(missing: {path})")
    out.append("")
out += ["#" * 92,
        "Agent will add the ship/no-ship synthesis on top of this when it wakes.",
        "#" * 92]
with open("Data/_ensemble/MORNING_REPORT.txt", "w") as f:
    f.write("\n".join(out))
log(f"MASTER COMPLETE in {(time.time()-t0)/60:.0f}m -> Data/_ensemble/MORNING_REPORT.txt")
