"""Interleaved old/new A/B, in-process (no pool scheduling noise)."""
import contextlib, importlib.util, io, statistics, time, warnings
import pandas as pd
warnings.filterwarnings("ignore")

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stderr(io.StringIO()):
        spec.loader.exec_module(m)
    return m

import glob, random
paths = sorted(glob.glob("Data/PriceData/*.parquet"))
TICKERS = [p.split("\\")[-1].replace(".parquet","") for p in random.Random(11).sample(paths, 15)]
frames = {t: pd.read_parquet(f"Data/PriceData/{t}.parquet") for t in TICKERS}

def run(mod):
    t0 = time.perf_counter()
    for t in TICKERS:
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            mod.run_pipeline_timed(frames[t].copy())
    return time.perf_counter() - t0

old = load("old", "_orig_ff.py")
new = load("new", "3__FeatureFramework.py")

run(old); run(new)          # discard warm-up rep for both

o, n = [], []
for rep in range(4):
    a = run(old); o.append(a)
    b = run(new); n.append(b)
    print(f"  rep{rep}: old={a:6.2f}s   new={b:6.2f}s", flush=True)

print(f"\n{len(TICKERS)} tickers/rep, 4 interleaved reps (warm-up discarded)")
print(f"  OLD  median {statistics.median(o):6.2f}s   min {min(o):6.2f}s")
print(f"  NEW  median {statistics.median(n):6.2f}s   min {min(n):6.2f}s")
print(f"  median delta: {(statistics.median(n)-statistics.median(o))/statistics.median(o)*100:+.1f}%")
print(f"  min    delta: {(min(n)-min(o))/min(o)*100:+.1f}%")
