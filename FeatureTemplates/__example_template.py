"""
================================================================================
FEATURE BLOCK TEMPLATE  —  READ THIS BEFORE WRITING A NEW FEATURE
================================================================================

PURPOSE
-------
This file is a fully documented example of how to write a feature block for the
FeatureFramework pipeline (3__FeatureFramework.py).

An AI agent writing a new feature should only need to read THIS file. It does
not need to see the orchestrator, other feature blocks, or any pipeline code.

HOW THE PIPELINE WORKS (brief)
-------------------------------
1. 3__FeatureFramework.py scans the FeatureTemplates/ folder at runtime.
2. It imports every .py file whose name does NOT start with an underscore (_).
   This file starts with __ so it is skipped — it is documentation only.
3. From each imported file it reads two things:
     - METADATA  (dict)  — describes the block and its column dependencies
     - compute() (func)  — the actual feature computation
4. It builds a dependency graph from METADATA["requires"] / METADATA["produces"]
   and runs blocks in the correct order so no block ever sees a missing column.
5. After all blocks run the columns are reordered:
     Date, Ticker, Open, High, Low, Close, Volume  →  then everything else A–Z

WHAT YOU NEED TO PROVIDE
-------------------------
Every feature file must expose exactly two things at module level:

    METADATA : dict   — see the template below for every required key
    compute  : func   — signature: (df: pd.DataFrame) -> pd.DataFrame

Nothing else is required. Keep files focused: one feature or one tightly related
family of features per file. If a file grows past ~150 lines, split it.

NAMING CONVENTIONS
------------------
Column names:
  - Use lowercase_snake_case                   e.g.  rsi_14
  - Include the parameter in the name          e.g.  rsi_14  not just  rsi
  - Prefix families with a short tag           e.g.  vol_atr_14,  mom_roc_5d
  - Never overwrite Date, Open, High, Low, Close, Volume  (these are sacred)
  - Never create a column called "label", "target", or "ret_5d" (reserved)

File names:
  - Match the primary feature concept          e.g.  rsi.py, volume_metrics.py
  - Use lowercase_snake_case
  - Start with _ or __ to be excluded from auto-discovery (templates, helpers)

METADATA KEYS
-------------
  name        (str, required)
      Unique identifier for this block. Used in log output and --exclude flags.
      Must match the filename for clarity.  e.g. "rsi"

  description (str, required)
      One-sentence human-readable description of what this block computes.

  requires    (list[str], required)
      Column names that MUST exist in df before compute() is called.
      The orchestrator will run any block that PRODUCES these columns first.
      Base OHLCV columns are always present: Open, High, Low, Close, Volume
      (and Date, Ticker if available). You do not need to list those in requires
      unless you want to be explicit for documentation purposes.

  produces    (list[str], required)
      Column names that compute() will ADD to df.
      The orchestrator uses this to resolve dependencies between blocks.
      Be exact: list every new column name here so dependents can find them.

  tags        (list[str], required)
      Arbitrary labels for grouping/filtering.
      Suggested categories: "momentum", "volume", "volatility", "trend",
      "mean_reversion", "market_regime", "gp", "experimental"

  version     (str, optional)
      Semantic version string. Default "1.0". Increment when compute() changes
      in a way that would change the output values.

  author      (str, optional)
      Free text. Useful for tracking which paper / idea a feature came from.

compute() CONTRACT
------------------
  Input:
    df          pd.DataFrame with at least the columns listed in requires.
                The DataFrame is per-ticker (one stock at a time), sorted
                ascending by date with no gaps in index.

  Output:
    The same df with new columns added. Returning df is mandatory even if you
    mutate it in place (some callers depend on the return value).

  Rules:
    - ONLY add the columns listed in METADATA["produces"].
    - Do NOT drop, rename, or modify any existing column.
    - Do NOT sort or reindex df (the orchestrator handles that).
    - Do NOT import from 3__AlphaSensitivity.py or other pipeline files.
    - Handle NaN gracefully: rolling windows produce NaN at the start — that
      is expected and fine. Do not fill or drop NaN rows.
    - Keep it stateless: no globals, no side effects, no file I/O.

DEPENDENCY EXAMPLE
------------------
If your block needs a column produced by another block, just list it in
requires. The orchestrator will ensure the producer runs first:

    # In momentum_score.py:
    METADATA = {
        "requires": ["return_1d", "return_5d"],   # produced by returns.py
        "produces": ["momentum_score"],
        ...
    }

    # returns.py will automatically run before momentum_score.py

If a required column is missing AND no registered block produces it, the
orchestrator will print a warning and skip your block rather than crash.

================================================================================
EXAMPLE: RSI (Relative Strength Index, 14-period)
================================================================================
"""

import pandas as pd

# ---------------------------------------------------------------------------
# METADATA  —  fill this in for every new feature file
# ---------------------------------------------------------------------------
METADATA = {
    "name":        "example_rsi",
    "description": "14-period Relative Strength Index (RSI) using Wilder smoothing",
    "requires":    ["Close"],          # columns that must exist before compute() runs
    "produces":    ["rsi_14"],         # columns that compute() will add to df
    "tags":        ["momentum", "technical"],
    "version":     "1.0",
    "author":      "example template",
}

# ---------------------------------------------------------------------------
# compute()  —  the feature logic goes here
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 14-period RSI using Wilder's smoothed moving average.

    RSI = 100 - 100 / (1 + RS)   where RS = avg_gain / avg_loss over 14 bars.
    Values near 100 = strongly overbought.
    Values near 0   = strongly oversold.
    First 13 rows will be NaN (insufficient history) — this is expected.
    """

    # ---- Step 1: compute daily price changes --------------------------------
    delta = df["Close"].diff()

    # ---- Step 2: separate gains and losses (losses are positive numbers) ----
    gain = delta.clip(lower=0)          # positive where price went up, else 0
    loss = (-delta).clip(lower=0)       # positive where price went down, else 0

    # ---- Step 3: Wilder smoothed average (equivalent to EWM alpha=1/period) -
    period = 14
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    # ---- Step 4: RSI --------------------------------------------------------
    rs = avg_gain / avg_loss.replace(0, float("nan"))   # avoid div-by-zero
    df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    # ---- Always return df ---------------------------------------------------
    return df


# ---------------------------------------------------------------------------
# NOTE: This file is skipped by auto-discovery because its name starts with _.
# Copy the METADATA + compute() pattern into a new file (without the leading _)
# and it will be picked up automatically on the next pipeline run.
# ---------------------------------------------------------------------------
