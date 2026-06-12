---
name: overnight-solver
description: Autonomous overnight research/compute workflow — point intelligence at a hard problem and let it run unattended for hours while the user sleeps. Use when the user says things like "work on X overnight", "run this while I sleep", "set up a nightly run", "go ham while I'm gone", "point this at my problem for the night", or otherwise hands off a heavy, long-running, multi-experiment job to run unattended. Decomposes the problem into a self-chaining pipeline of robust, resumable experiments that keep compute continuously busy, verifies a clean start, monitors at milestones, and leaves a morning synthesis report.
---

# Overnight Solver

A repeatable method for handing a hard, compute-heavy problem to the agent to work
on autonomously for hours. The goal: **keep the machine busy continuously** (not
"20 min of work then idle"), **survive crashes**, **never need user input mid-run**,
and leave a **clear morning verdict**. Distilled from a real overnight session.

## Core principle

The agent is bursty (build → launch → wait → analyze → repeat) but COMPUTE must be
continuous. Bridge the gap with **self-chaining orchestrators**: each experiment
writes a report file on completion; lightweight "waiter" scripts poll for that file
and auto-launch the next experiment. The cores stay pegged for hours WITHOUT the
agent needing to be awake — the agent just wakes at milestones to sanity-check and
adapt. One overnight session ran ~7.5 hrs of continuous compute off ~4 agent bursts.

## Workflow

### 0. Scope fast (the user is going to sleep — minimize questions)
- Get the GOAL and the SUCCESS METRIC. One or two questions MAX. If the user is
  already gone / said "just do it", infer from context and recent work, make
  defensible choices, and **document them in the morning report** instead of asking.
- Confirm what's OFF-LIMITS. Default hard rules: touch nothing live/production
  (copy-and-edit instead), back up anything you might overwrite, no irreversible or
  outward-facing actions (no live trading/broker, no sends/posts, no deploys).
- Once scoped, DO NOT ask again. Use `AskUserQuestion` zero times after the user is
  asleep — make the call and log it.

### 1. Decompose into a chain of experiments
- Break the problem into N self-contained experiments (PART 1..N). Each: takes
  inputs, runs compute, writes ONE report file (its completion signal).
- Order so each part is valuable independently — partial completion (a crash at
  hour 5) still yields a usable answer.
- **Prefer many small parts over one monolith.** Each part boundary is a free
  checkpoint, resume point, and "is it broken?" gate. It also lets you ADAPT:
  results of PART 1 can reshape PART 2.
- Sequence parts so each can use ALL cores (don't run two heavy parts at once).

### 2. Build each experiment as a robust autonomous script
Every script MUST be (see `templates/orchestrator.py`):
- **Resumable** — skip any unit whose output already exists (`skip-if-exists`).
- **Crash-tolerant** — `try/except` around every subprocess; log the failure and
  CONTINUE. One bad unit must not kill the run.
- **Incremental** — append results to disk after EACH unit, so a mid-run crash
  still leaves a partial report.
- **Bounded** — per-unit `subprocess.run(..., timeout=...)`.
- **Non-interactive** — no prompts, no `input()`, no flags that block on a TTY.
- **Isolated** — write only to a dedicated scratch dir (e.g. `Data/_<exp>/`); never
  to live paths. Back up anything shared you might clobber FIRST.

### 3. Chain the parts with waiter orchestrators
- A "waiter" polls (sleeps ~120s) for the previous part's report file, then launches
  the next part (see `templates/waiter.py`). No CPU while waiting.
- Launch the chain so the WHOLE night runs without depending on the agent waking up
  on time. The agent's re-invocations become a bonus (analyze/adapt), not a
  dependency.
- Keep waiters dumb and robust: poll with a max-timeout, `try/except` the launch.

### 4. Launch + VERIFY CLEAN START (do not skip)
- Launch with `run_in_background: true` and **NO trailing `&`** (the tool backgrounds
  it; adding `&` orphans the real process and it gets torn down).
- **`python -u`** for unbuffered logs you can actually read live.
- Within ~1–2 min, verify it cleared the first heavy step (data load + first unit)
  WITHOUT crashing. Catching a typo now saves an 8-hour blind failure. Check: the
  log advanced past loading, a process is alive, no traceback.

### 5. Monitor at milestones (the "midpoint check")
- You're auto-re-invoked at each background task's completion. At each: read the
  report, sanity-check the numbers are plausible (not all-NaN, not a parse fail),
  confirm the next part launched, and ADAPT the remaining plan if results warrant.
- If a part is ONE long job with no natural mid-checkpoints, schedule a midpoint
  wake-up (ScheduleWakeup / a one-shot cron) to verify it's not silently hung —
  rather than discovering breakage only at the end.
- Do NOT poll a healthy running job in a tight loop — that wastes cycles. The
  completion notification IS the signal.

### 6. Morning synthesis
- Concatenate the raw part-reports into one `MORNING_REPORT.txt`, then write YOUR
  verdict on top: the answer, the evidence, the recommendation, and the caveats.
- Distinguish SIGNAL from NOISE explicitly (see gotcha below). State confidence.
- Save durable findings to memory. List leftover scratch dirs and offer cleanup.

## Reusable templates
- `templates/orchestrator.py` — a generic experiment runner: a list of UNITS, a
  produce-phase (sequential, skip-if-done) and an evaluate-phase (parallel pool with
  retry + incremental results), then aggregate → report. Adapt `UNITS`, `run_unit`,
  `parse`.
- `templates/waiter.py` — polls for a report file, then launches the next script.
  Chain several to run PART 2 → PART 3 → ... unattended.

## Hard-won gotchas (READ — each cost real time)
1. **`mkdir` the output dir BEFORE any shell `>` redirect into it.** `python x.py >
   Data/_new/run.log` fails instantly if `Data/_new/` doesn't exist yet (the shell
   opens the log before the script's `os.makedirs` runs). Exit 1, empty log.
2. **No trailing `&` with `run_in_background: true`.** It double-backgrounds; the
   tool's wrapper exits 0 immediately and the orphaned job gets killed mid-run.
3. **Buffered logs lie.** Redirected stdout is block-buffered — a "stuck" log may
   just be unflushed. Use `python -u`. A process can be alive and working while its
   log looks frozen.
4. **Parallelism has a resource ceiling.** Many parallel heavy subprocesses can hit
   OS limits (Windows: `OSError 22 / WinError 1450 insufficient resources`). Tune the
   pool (3-wide was safe where 4 crashed) and RETRY failed units once, solo.
5. **Isolate or accept clobber on shared output files.** Parallel jobs writing the
   same output path race. Either give each its own scratch dir/cwd, or only parse
   stdout and ignore the shared file (back it up first, restore after).
6. **Beat the noise floor.** If comparing models/configs/runs, a SINGLE run can be
   noise-dominated (one project had ±15pp/month swing from the RNG seed alone). Use
   MULTIPLE seeds and average; treat differences smaller than the seed-std as noise.
   A single-path comparison can flip sign on a re-run — don't trust it.
7. **Validate at the REAL objective, not a cheap proxy.** A proxy metric (band-level
   IC, a sub-window) can look great while the true objective (full strategy backtest)
   says the opposite. Always confirm at the level the user actually cares about
   before declaring a win.
8. **Resume must skip COMPLETED work, not just trained artifacts.** If a later phase
   deletes intermediate files (to save disk), the resume check for the earlier phase
   must also treat "already in final results" as done — or it re-does finished work.
9. **Tune timeouts to the SLOWEST unit, and never write the report only at the end.**
   A part-level `subprocess.run(timeout=...)` that fires before the end-of-run report
   is written loses EVERYTHING, even completed units (real loss: a window's backtests
   ran ~33 min each, not the ~21 assumed, so a 4-hr part-timeout killed it mid-run
   after finishing most units — but the script saved results only at the end, so all
   of it was lost). Fixes: (a) measure one unit's real time before sizing the timeout
   and add generous margin; (b) write results INCREMENTALLY (as the template does) so
   a timeout/crash near the end still leaves a partial report worth reading.

## Safety (non-negotiable)
- Touch nothing live/production. Copy-and-edit; revert the original to pristine.
- Back up anything you might overwrite (timestamped, to a `_backups/` dir).
- No irreversible or outward-facing actions while unattended: no live trading/broker
  connections, no network sends/posts, no deploys, no `git push`.
- If a step would do any of the above, SKIP it and note it for the morning instead.
