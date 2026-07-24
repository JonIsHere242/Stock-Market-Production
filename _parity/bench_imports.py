"""Machine-independent evidence: count block-module executions per N tickers."""
import contextlib, importlib.util, io, warnings
import pandas as pd
warnings.filterwarnings("ignore")

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stderr(io.StringIO()):
        spec.loader.exec_module(m)
    return m

TICKERS = ["AAPL","MSFT","KO","TEX","BIP"]
frames = {t: pd.read_parquet(f"Data/PriceData/{t}.parquet") for t in TICKERS}

for label, path in [("OLD", "_orig_ff.py"), ("NEW", "3__FeatureFramework.py")]:
    mod = load(label, path)
    calls = {"n": 0}
    real_exec = importlib.util.module_from_spec
    # count exec_module calls made from inside discover_blocks
    orig_spec = importlib.util.spec_from_file_location
    def counting_spec(name, location, *a, **kw):
        if "FeatureTemplates" in str(location):
            calls["n"] += 1
        return orig_spec(name, location, *a, **kw)
    importlib.util.spec_from_file_location = counting_spec
    try:
        for t in TICKERS:
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                mod.run_pipeline_timed(frames[t].copy())
    finally:
        importlib.util.spec_from_file_location = orig_spec
    print(f"{label}: {calls['n']:,} block-module loads for {len(TICKERS)} tickers "
          f"({calls['n']//len(TICKERS)} per ticker)")
