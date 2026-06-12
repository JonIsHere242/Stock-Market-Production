"""OVERNIGHT MASTER 2 — runs PART 4 (window-2 robustness) after the main chain.

Waits (light poll) for the main master's MORNING_REPORT.txt, then runs PART 4
(seed-ensemble on an independent OOS window, Jul-Oct 2025) to confirm the ensemble
benefit isn't specific to Jan-Apr 2026. No CPU contention while waiting; touches
nothing live.
"""
import os, subprocess, sys, time
PY = sys.executable


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] [MASTER2] {m}", flush=True)


def wait(path, max_hr=6):
    log(f"waiting for {path}")
    t0 = time.time()
    while not os.path.exists(path):
        if time.time() - t0 > max_hr * 3600:
            log("timeout; proceeding")
            return False
        time.sleep(120)
    log(f"found after {(time.time()-t0)/60:.0f}m")
    return True


wait("Data/_ensemble/MORNING_REPORT.txt", max_hr=6)
log("main chain done -> launching PART 4 (window-2 ensemble)")
os.makedirs("Data/_ensemble_w2", exist_ok=True)
try:
    with open("Data/_ensemble_w2/run.log", "w") as lf:
        subprocess.run([PY, "analysis_output/overnight_part4_window2.py"],
                       stdout=lf, stderr=subprocess.STDOUT, timeout=4 * 3600)
    log("PART 4 done")
except Exception as e:
    log(f"PART 4 error: {repr(e)[:160]}")
log("MASTER2 complete -> Data/_ensemble_w2/ENSEMBLE_REPORT.txt")
