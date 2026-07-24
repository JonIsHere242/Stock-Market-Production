"""Bit-parity gate: current pipeline output vs the pre-refactor baseline.

  stock_env\\Scripts\\python.exe _parity\\check.py

Exit 0 only if every baseline ticker reproduces exactly (same columns, same order,
same values incl. NaN placement). Any drift is printed column-by-column.
"""
import contextlib
import importlib.util
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "_parity" / "baseline"

spec = importlib.util.spec_from_file_location("ff", ROOT / "3__FeatureFramework.py")
ff = importlib.util.module_from_spec(spec)
with contextlib.redirect_stderr(io.StringIO()):
    spec.loader.exec_module(ff)

fails = []
for bp in sorted(BASE.glob("*.parquet")):
    ticker = bp.stem
    want = pd.read_parquet(bp)
    df = pd.read_parquet(ROOT / "Data" / "PriceData" / f"{ticker}.parquet")
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        got, _ = ff.run_pipeline_timed(df)

    if list(got.columns) != list(want.columns):
        missing = sorted(set(want.columns) - set(got.columns))
        extra = sorted(set(got.columns) - set(want.columns))
        order = set(got.columns) == set(want.columns)
        fails.append(f"{ticker}: COLUMNS differ "
                     f"(missing={missing[:6]} extra={extra[:6]} same_set_wrong_order={order})")
        continue
    if len(got) != len(want):
        fails.append(f"{ticker}: ROWS {len(got)} != {len(want)}")
        continue

    bad = []
    for c in want.columns:
        a, b = want[c].to_numpy(), got[c].to_numpy()
        if a.dtype.kind in "fc" and b.dtype.kind in "fc":
            if not np.array_equal(a, b, equal_nan=True):
                d = np.nanmax(np.abs(a - b)) if np.isfinite(a - b).any() else float("nan")
                bad.append(f"{c}(maxdiff={d:.3g})")
        elif not a.tolist() == b.tolist():
            bad.append(c)
    if bad:
        fails.append(f"{ticker}: {len(bad)} cols differ -> {bad[:8]}")

if fails:
    print("PARITY FAIL")
    for f in fails:
        print("  ", f)
    sys.exit(1)
print(f"PARITY OK  --  {len(list(BASE.glob('*.parquet')))} tickers bit-identical")
