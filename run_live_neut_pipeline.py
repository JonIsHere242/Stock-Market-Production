"""LIVE pipeline re-run with the validated dose-0.3 pred-space neutralization (2026-07-02).

Mirrors rerun_for_live.ps1 (pull -> features -> predict -> backtest -> signal) with ONE
insert between predict and backtest:
    neutralize Data/RFpredictions (dose 0.3, factors from the FRESH Data/ProcessedData_v2)
    and swap the neutralized preds into Data/RFpredictions (raw copy kept for rollback).
Evidence for the overlay: live-window backtest 77.64%/1.81/DD20.1 -> 89.24%/2.00/DD18.5,
robust across BT_SAMPLE_SEEDs + the pinned window (see
.claude/docs/SESSION_NOTES_leak_fix_and_neutralization_2026_07_01.md).

The broker step is intentionally NOT here. Run 7__MacroFilter.py after this completes.
Logs -> Data/_live_rerun_logs_<stamp>/. ASCII only.
"""
import os, shutil, subprocess, sys, time

ROOT = r"c:\Users\Masam\Desktop\Stock-Market"
os.chdir(ROOT)
PY = sys.executable
STAMP = time.strftime("%Y%m%d_%H%M")
LOGDIR = os.path.join("Data", "_live_rerun_logs_%s" % STAMP)
os.makedirs(LOGDIR, exist_ok=True)
os.environ["PYTHONPATH"] = ROOT


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(os.path.join(LOGDIR, "pipeline.log"), "a") as fh:
        fh.write(line + "\n")


def step(name, cmd, timeout=14400, env_extra=None):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    log("STEP %s: START  (%s)" % (name, " ".join(cmd)))
    t0 = time.time()
    with open(os.path.join(LOGDIR, "%s.log" % name), "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                            env=env, timeout=timeout).returncode
    log("STEP %s: rc=%s (%.0fs)" % (name, rc, time.time() - t0))
    if rc != 0:
        log("STEP %s FAILED -- ABORTING PIPELINE (see %s/%s.log)" % (name, LOGDIR, name))
        raise SystemExit(1)


def main():
    # 0) backups (rollback-safe). NOTE: COPY, not rename — 2__PriceDownloader --RefreshMode
    # reads its ticker list FROM Data/RFpredictions, so the live dir must stay in place
    # (the predict step overwrites the per-ticker files, same as the production flow).
    log("=== 0) backups (stamp %s) ===" % STAMP)
    if os.path.isdir("Data/RFpredictions") and not os.path.isdir("Data/RFpredictions_bak_%s" % STAMP):
        shutil.copytree("Data/RFpredictions", "Data/RFpredictions_bak_%s" % STAMP)
        log("Data/RFpredictions COPIED -> Data/RFpredictions_bak_%s" % STAMP)
    if os.path.exists("Data/0__signals.parquet"):
        shutil.copy2("Data/0__signals.parquet", "Data/0__signals_bak_%s.parquet" % STAMP)
        log("signal backed up -> Data/0__signals_bak_%s.parquet" % STAMP)

    # 1) fresh prices
    step("1_prices", [PY, "2__PriceDownloader.py", "--RefreshMode"])

    # 2) features (exact production exclude list)
    step("2_features", [PY, "3__FeatureFramework.py", "--all", "--exclude",
                        "vvg", "vaq_vg", "rvg_wl", "volume_spectral_splatter",
                        "--workers", "32"])

    # 3) predict_only with the ship model
    step("3_predict", [PY, "4__Predictor.py", "--input_dir", "Data/ProcessedData_v2",
                       "--predict_only", "--model_dir", "Data/_ship_v2/model",
                       "--output_dir", "Data/RFpredictions"])

    # 3.5) NEUTRALIZE (dose 0.3, factors from the FRESH v2 panel) and swap in
    step("35_neutralize", [PY, "experimental/residreg3/neutralize_preds.py",
                           "--src", "Data/RFpredictions",
                           "--out_base", "Data/_neut_live_%s" % STAMP,
                           "--panel", "Data/ProcessedData_v2",
                           "--doses", "0.3"])
    neut_dir = os.path.join("Data", "_neut_live_%s" % STAMP, "ship_neut30")
    n_files = len([f for f in os.listdir(neut_dir) if f.endswith(".parquet")])
    if n_files < 1000:
        log("neutralized dir too small (%d files) -- ABORT, RFpredictions left RAW" % n_files)
        raise SystemExit(1)
    os.rename("Data/RFpredictions", "Data/RFpredictions_raw_%s" % STAMP)
    os.rename(neut_dir, "Data/RFpredictions")
    log("swapped: Data/RFpredictions = NEUTRALIZED preds (raw kept at "
        "Data/RFpredictions_raw_%s)" % STAMP)

    # 3.6) verify the SIGNAL DAY actually got neutralized (factor coverage on last date)
    import pandas as pd
    import numpy as np
    changed = same = 0
    for t in ("AAPL", "NVDA", "XOM", "JPM", "KO", "WMT", "TSLA", "AMD"):
        try:
            a = pd.read_parquet("Data/RFpredictions_raw_%s/%s.parquet" % (STAMP, t),
                                columns=["Date", "UpProbability"]).iloc[-1]
            b = pd.read_parquet("Data/RFpredictions/%s.parquet" % t,
                                columns=["Date", "UpProbability"]).iloc[-1]
            if a["Date"] == b["Date"]:
                if abs(float(a["UpProbability"]) - float(b["UpProbability"])) > 1e-9:
                    changed += 1
                else:
                    same += 1
        except Exception:
            pass
    log("signal-day neutralization check: %d/%d sample tickers changed on the last date"
        % (changed, changed + same))
    if changed == 0:
        log("WARNING: last-date scores identical to raw -- factor panel may be stale for "
            "the signal day; the book would be the plain ship signal.")

    # 4) production backtester --force (backtest gate + 12-pool live-signal export)
    step("4_backtest_signal", [PY, "5__NightlyBackTester.py", "--force"],
         env_extra={"BT_SAMPLE_SEED": "42"})

    # 5) verify signal
    try:
        d = pd.read_parquet("Data/0__signals.parquet")
        log("SIGNAL: rows=%d  TargetDate=%s  Created=%s  symbols=%s"
            % (len(d), d["TargetDate"].max(), d.get("CreatedDate", pd.Series([None])).max(),
               ",".join(map(str, d.get("Symbol", pd.Series([])).tolist()[:12]))))
    except Exception as ex:
        log("signal verify failed: %r" % ex)

    log("PIPELINE COMPLETE -- next: 7__MacroFilter.py (self-waits to the post-open window)")


if __name__ == "__main__":
    main()
