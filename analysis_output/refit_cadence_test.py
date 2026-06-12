"""Strategy-level refit-cadence test (the one surviving OOS lever).

Isolates FRESHNESS: trains models with IDENTICAL fixed production params, varying
only the training cutoff, then backtests each and compares:
  STATIC track = the Dec-cutoff model's Jan/Feb/Mar/Apr returns (= live model's
                 situation: one model, aging 1->4 months stale)
  FRESH track  = each month scored by a model trained through the PRIOR month
                 (always ~1 month fresh): Jan<-Dec, Feb<-Jan, Mar<-Feb, Apr<-Mar

If FRESH clearly beats STATIC on the strategy-level monthly returns, a refit cadence
is worth establishing. If they're a wash, staleness isn't costing us and we leave
the live model alone. Params are held fixed throughout — only freshness varies.
"""
import glob, os, re, subprocess, sys

# Fixed production params (from Data/XGBPipeline/summary.json) — held constant so
# the ONLY thing that varies across models is the training cutoff.
PROD = ["--n_estimators", "304", "--max_depth", "8", "--learning_rate", "0.0509",
        "--min_child_weight", "15", "--subsample", "0.886",
        "--colsample_bytree", "0.444", "--reg_alpha", "0.0048",
        "--reg_lambda", "0.0223"]

# (tag, train_end_date, first-OOS-month this model is "fresh" for)
CUTOFFS = [("c1231", "2025-12-31", "2026-01"),
           ("c0131", "2026-01-31", "2026-02"),
           ("c0228", "2026-02-28", "2026-03"),
           ("c0331", "2026-03-31", "2026-04")]

ANSI = re.compile(r"\x1b\[[0-9;]*m")
os.makedirs("Data/_refit", exist_ok=True)


def parse_monthly(txt):
    """Extract {YYYY-MM: return%} from the 'Strategy Monthly Performance' block."""
    txt = ANSI.sub("", txt)
    out, in_sec = {}, False
    for line in txt.splitlines():
        if "Strategy Monthly Performance" in line:
            in_sec = True
            continue
        if in_sec:
            if "Monthly Excess" in line or "Strategy Yearly" in line:
                break
            m = re.match(r"\s*(\d{4}-\d{2}):\s+(-?\d+\.?\d*)", line)
            if m:
                out[m.group(1)] = float(m.group(2))
    return out


# Phase 1: train all cutoff models (fixed prod params, no Optuna). Skip any
# already built. Sequential — each fit uses all cores and is fast (~4 min).
for tag, cut, _ in CUTOFFS:
    mdir = f"Data/_refit/{tag}"
    if (os.path.exists(f"{mdir}/xgb.joblib")
            and len(glob.glob(f"{mdir}/rf/*.parquet")) > 100):
        print(f"===== {tag}: model exists, skip train =====", flush=True)
        continue
    print(f"===== {tag}: train through {cut} =====", flush=True)
    with open(f"Data/_refit/{tag}_train.log", "w") as lf:
        subprocess.run([sys.executable, "4__Predictor.py", "--no_tune",
                        "--train_end_date", cut, *PROD,
                        "--model_dir", mdir, "--output_dir", f"{mdir}/rf"],
                       stdout=lf, stderr=subprocess.STDOUT, check=True)

# Phase 2: backtest all 4 IN PARALLEL (independent; ~21 min each, ~21 min wall).
# NO --force: that triggers the slow finviz live-export; backtest metrics unchanged.
print("\n===== launching 4 backtests in parallel (no --force) =====", flush=True)
procs = {}
for tag, _, _ in CUTOFFS:
    lf = open(f"Data/_refit/{tag}_backtest.log", "w")
    procs[tag] = (subprocess.Popen([sys.executable, "5__NightlyBackTester.py",
                                    "--data_dir", f"Data/_refit/{tag}/rf"],
                                   stdout=lf, stderr=subprocess.STDOUT), lf)
results = {}
for tag, (p, lf) in procs.items():
    p.wait()
    lf.close()
    monthly = parse_monthly(open(f"Data/_refit/{tag}_backtest.log").read())
    results[tag] = monthly
    print(f"  {tag} done -> monthly: {monthly}", flush=True)

# Build the comparison.
static = {m: results["c1231"].get(m) for m in ["2026-01", "2026-02", "2026-03", "2026-04"]}
fresh = {"2026-01": results["c1231"].get("2026-01"),   # Jan: both use Dec model
         "2026-02": results["c0131"].get("2026-02"),
         "2026-03": results["c0228"].get("2026-03"),
         "2026-04": results["c0331"].get("2026-04")}


def compound(d):
    p = 1.0
    for v in d.values():
        if v is not None:
            p *= (1 + v / 100.0)
    return (p - 1) * 100.0


print("\n" + "=" * 64)
print("REFIT-CADENCE — strategy-level monthly return % (params fixed)")
print("=" * 64)
print(f"  {'month':<9} {'STATIC (Dec model)':>20} {'FRESH (1mo refit)':>20}")
for m in ["2026-01", "2026-02", "2026-03", "2026-04"]:
    s, f = static.get(m), fresh.get(m)
    flag = ""
    if s is not None and f is not None:
        flag = "  <- FRESH wins" if f > s + 0.5 else ("  <- STATIC wins" if s > f + 0.5 else "  ~tie")
    print(f"  {m:<9} {('' if s is None else f'{s:+.2f}'):>20} "
          f"{('' if f is None else f'{f:+.2f}'):>20}{flag}")
print("-" * 64)
print(f"  {'compounded':<9} {compound(static):>19.2f}% {compound(fresh):>19.2f}%")
print("=" * 64)
print("Jan is identical (both use the Dec model). Feb/Mar/Apr is the real test:")
print("does a 1-month-fresh model beat the aging Dec model at the STRATEGY level?")
