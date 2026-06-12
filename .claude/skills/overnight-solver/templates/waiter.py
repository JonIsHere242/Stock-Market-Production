"""Waiter / chain TEMPLATE (overnight-solver skill).

Keeps compute continuous WITHOUT the agent needing to wake up: poll (cheap sleep)
for the previous part's report file, then launch the next part. Chain several of
these (or list multiple steps below) to run PART 2 -> PART 3 -> ... unattended.

Launch with run_in_background (NO trailing &). It uses ~0 CPU while waiting.
The previous part (PART 1) is typically launched separately and already running.
"""
import os, subprocess, sys, time

PY = sys.executable

# (label, report-file-to-wait-for, script-to-run-next, log-path-for-that-script)
# The next script runs only AFTER its wait-file appears. Each step's own report
# becomes the wait-file for the following step, so they daisy-chain.
STEPS = [
    ("PART2", "Data/_part1/REPORT.txt",  "analysis_output/part2.py", "Data/_part2/run.log"),
    ("PART3", "Data/_part2/REPORT.txt",  "analysis_output/part3.py", "Data/_part3/run.log"),
]
MAX_WAIT_HR = 6


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] [WAITER] {m}", flush=True)


def wait_for(path, max_hr):
    log(f"waiting for {path}")
    t0 = time.time()
    while not os.path.exists(path):
        if time.time() - t0 > max_hr * 3600:
            log(f"TIMEOUT waiting {path}; proceeding anyway")
            return False
        time.sleep(120)                      # cheap poll; no CPU
    log(f"found {path} after {(time.time()-t0)/60:.0f}m")
    return True


def run(script, logpath):
    os.makedirs(os.path.dirname(logpath), exist_ok=True)   # mkdir BEFORE redirect
    try:
        with open(logpath, "w") as lf:
            subprocess.run([PY, script], stdout=lf, stderr=subprocess.STDOUT,
                           check=False, timeout=4 * 3600)
    except Exception as e:
        log(f"{script} ERROR: {repr(e)[:160]}")


for label, wait_file, script, logpath in STEPS:
    wait_for(wait_file, MAX_WAIT_HR)
    log(f"START {label}: {script}")
    run(script, logpath)
    log(f"DONE {label}")

log("WAITER complete")
