"""Generate per-ticker RFpredictions parquets from a saved simple_xgb model.

Reads:  Data/SimpleModel/xgb.joblib            (trained model)
        Data/ProcessedData/*.parquet           (per-ticker features)
Writes: Data/RFpredictions/<TICKER>.parquet    (backtester-compatible)

Output columns match the format 5__NightlyBackTester.py expects:
    Date, Open, High, Low, Close, Volume,
    UpProbability, DownProbability,
    PositiveThreshold, NegativeThreshold,
    UpPrediction (0/1/-1),
    VIX_Close   + optional pass-through columns

UpProbability rescaling (the 'shifting around' the user mentioned):
    The backtester gates raw UpProbability to be in [0.45, 0.70].  Our model's
    raw score lives in roughly [0.05, 0.65] with most rows near 0.20 (because
    the topq label has base rate 0.20).  Without rescaling, virtually no row
    would pass the backtester gate.

    We map per-day percentile rank of model score:
        top top_frac of each day -> UpProb linearly in [0.45, 0.70]
        rest of universe         -> UpProb linearly in [0.30, 0.44]
    Non-universe rows                -> UpProb = 0.30, UpPrediction = -1

    So at inference time the backtester sees the top ~1% per day (about
    10-15 names per day with the default top_frac=0.01) as "gate-passing"
    candidates, and the rest are below the gate.

The set of features the model expects is read directly from
    model.feature_names_in_
so whatever combination of drop_vol_features / add_xs_features / topq_frac the
model was trained with is honoured automatically.

Run:
    python predict_to_rf.py
    python predict_to_rf.py --top_frac_per_day 0.01
    python predict_to_rf.py --model_path Data/SimpleModel/xgb.joblib --output_dir Data/RFpredictions
"""

import os
import sys
import time
import argparse
import logging

import numpy as np
import pandas as pd
from joblib import load
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="Data/SimpleModel/xgb.joblib")
parser.add_argument("--input_dir", default="Data/ProcessedData")
parser.add_argument("--output_dir", default="Data/RFpredictions")
parser.add_argument("--date_col", default="Date")
parser.add_argument("--ticker_col", default="Ticker")
parser.add_argument("--top_frac_per_day", type=float, default=0.01,
                    help="Per-day fraction that gets mapped to UpProb in "
                         "[0.45, 0.70].  Default 0.01 = top 1% per day "
                         "(roughly 10-15 names with universe ~1400/day).")
parser.add_argument("--max_files", type=int, default=None,
                    help="Cap on tickers loaded (for quick tests).")
args = parser.parse_args()


# -------------------------------------------------------------------------- #
class Phase:
    def __init__(self, name): self.name = name
    def __enter__(self):
        self.t0 = time.time()
        logging.info(f">>> {self.name} ...")
        return self
    def __exit__(self, *exc):
        logging.info(f"<<< {self.name} done in {time.time()-self.t0:.1f}s.")


# -------------------------------------------------------------------------- #
def apply_quality_filter(df,
                         min_close=5.0,
                         min_dollar_volume=5_000_000.0,
                         max_atr_pct=0.05,
                         rsi_exclude_lo=30.0,
                         rsi_exclude_hi=40.0):
    """FilterRubric Step 1.  Returns a boolean mask aligned to df.index."""
    mask = pd.Series(True, index=df.index)
    if "Close" in df.columns:
        mask &= df["Close"] >= min_close
    if "dollar_volume_ma_10" in df.columns:
        mask &= df["dollar_volume_ma_10"] >= min_dollar_volume
    if "atr_percentage" in df.columns:
        mask &= df["atr_percentage"] <= max_atr_pct
    if "RSI" in df.columns:
        mask &= ~((df["RSI"] >= rsi_exclude_lo) & (df["RSI"] < rsi_exclude_hi))
    return mask


def downcast(df):
    floats = df.select_dtypes(include=["float64"]).columns
    if len(floats):
        df[floats] = df[floats].astype("float32")
    return df


def map_pct_rank_to_upprob(pct_rank, top_frac):
    """Map per-day percentile rank to UpProbability.

    top top_frac of each day  ->  [0.45, 0.70]   (passes backtester gate)
    rest                       ->  [0.30, 0.44]   (below gate, won't fire)
    """
    threshold = 1.0 - top_frac
    above = pct_rank >= threshold
    up = np.empty_like(pct_rank, dtype=np.float64)
    # Top band: linear [0.45, 0.70] across the top_frac slice
    up[above] = 0.45 + 0.25 * (pct_rank[above] - threshold) / max(top_frac, 1e-9)
    # Bottom band: linear [0.30, 0.44] across the bottom 1-top_frac slice
    up[~above] = 0.30 + 0.14 * pct_rank[~above] / max(threshold, 1e-9)
    return np.clip(up, 0.30, 0.70)


# -------------------------------------------------------------------------- #
def main():
    if not os.path.exists(args.model_path):
        logging.error(f"Model not found at {args.model_path}.")
        return
    os.makedirs(args.output_dir, exist_ok=True)

    with Phase("Load saved model"):
        model = load(args.model_path)
        if not hasattr(model, "feature_names_in_"):
            logging.error("Model has no feature_names_in_; can't determine which "
                          "features to provide at inference.")
            return
        feat_names = list(model.feature_names_in_)
        xs_features = [c for c in feat_names if c.endswith("_xs")]
        raw_features = [c for c in feat_names if not c.endswith("_xs")]
        xs_source_features = [c[:-3] for c in xs_features]
        logging.info(f"  model expects {len(feat_names)} features: "
                     f"{len(raw_features)} raw + {len(xs_features)} xs")

    with Phase("Load all ticker files"):
        files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".parquet"))
        if args.max_files:
            files = files[:args.max_files]
            logging.info(f"  --max_files {args.max_files}: using {len(files)} of "
                         f"{len(os.listdir(args.input_dir))} tickers")
        parts = []
        for fn in tqdm(files, desc="Loading"):
            try:
                df = pd.read_parquet(os.path.join(args.input_dir, fn))
                if args.date_col not in df.columns or df.empty:
                    continue
                if args.ticker_col not in df.columns:
                    df[args.ticker_col] = os.path.splitext(fn)[0]
                df[args.date_col] = pd.to_datetime(df[args.date_col])
                df = df.sort_values(args.date_col).reset_index(drop=True)
                df = downcast(df)
                parts.append(df)
            except Exception as e:
                logging.warning(f"  skipping {fn}: {e}")
        if not parts:
            logging.error("No data loaded.")
            return
        combined = pd.concat(parts, ignore_index=True)
        combined = combined.sort_values(args.date_col).reset_index(drop=True)
        del parts
        logging.info(f"  combined shape: {combined.shape}, "
                     f"{combined[args.ticker_col].nunique()} tickers, "
                     f"{combined[args.date_col].nunique()} dates")

    with Phase("Apply universe filter (mark, don't drop)"):
        quality_mask = apply_quality_filter(combined)
        n_pass = int(quality_mask.sum())
        logging.info(f"  universe-passing: {n_pass:,} / {len(combined):,} "
                     f"({100*n_pass/len(combined):.1f}%)")

    with Phase("Fill any missing model features with neutral values"):
        # Some raw features the model expects may not be in this inference
        # snapshot (renamed, removed upstream, etc.) -- fill with 0.
        missing_raw = [c for c in raw_features if c not in combined.columns]
        for c in missing_raw:
            combined[c] = 0.0
        if missing_raw:
            logging.info(f"  filled {len(missing_raw)} missing raw features "
                         f"with 0  (examples: {missing_raw[:5]})")

        missing_xs_source = [c for c in xs_source_features if c not in combined.columns]
        for c in missing_xs_source:
            combined[c] = 0.0
        if missing_xs_source:
            logging.info(f"  filled {len(missing_xs_source)} missing xs-source "
                         f"features with 0  (examples: {missing_xs_source[:5]})")

    with Phase("Compute cross-sectional ranks within universe-passing rows"):
        if xs_features:
            filtered = combined.loc[quality_mask, xs_source_features]
            grouped = filtered.groupby(combined.loc[quality_mask, args.date_col],
                                       sort=False)
            ranks = grouped.rank(pct=True, method="average", na_option="keep")
            ranks.columns = [c + "_xs" for c in xs_source_features]
            # Initialise xs cols to 0.5 (neutral); fill passing rows with ranks
            for col in xs_features:
                combined[col] = 0.5
            combined.loc[quality_mask, xs_features] = ranks[xs_features].values.astype(np.float32)
            logging.info(f"  computed xs ranks for {len(xs_features)} features")
        else:
            logging.info("  no xs features expected by model; skipping")

    with Phase("Predict"):
        X = combined[feat_names].astype(np.float32)
        # XGB handles NaN natively in trees, but if any inf slipped through,
        # replace with NaN so XGB knows it's missing.
        X = X.replace([np.inf, -np.inf], np.nan)
        scores = model.predict_proba(X)[:, 1].astype(np.float32)
        logging.info(f"  scores: min={scores.min():.4f} p50={np.median(scores):.4f} "
                     f"p95={np.percentile(scores, 95):.4f} max={scores.max():.4f}")

    with Phase("Rescale scores to backtester UpProbability gate"):
        # Per-day percentile rank ONLY within universe-passing rows.  Non-passing
        # rows get pct_rank = 0 (below threshold) and are set to UpProb 0.30 below.
        pct_rank = np.zeros(len(combined), dtype=np.float32)
        filt_idx = np.where(quality_mask.values)[0]
        if len(filt_idx) > 0:
            df_filt = pd.DataFrame({
                "s": scores[filt_idx],
                "d": combined[args.date_col].values[filt_idx],
            })
            pr = df_filt.groupby("d", sort=False)["s"].rank(pct=True, method="average").values
            pct_rank[filt_idx] = pr.astype(np.float32)

        up_prob = map_pct_rank_to_upprob(pct_rank, args.top_frac_per_day)
        up_prob[~quality_mask.values] = 0.30  # non-universe rows: below gate

        # UpPrediction = 1 only for universe-passing AND top fraction
        is_top = pct_rank >= (1.0 - args.top_frac_per_day)
        up_pred = np.where(quality_mask.values & is_top, 1,
                           np.where(quality_mask.values, 0, -1)).astype(np.int8)

        n_fire = int((up_pred == 1).sum())
        n_days = combined[args.date_col].nunique()
        logging.info(f"  fires: {n_fire:,} of {n_pass:,} universe-passing "
                     f"({100*n_fire/max(n_pass,1):.2f}%) "
                     f"-> avg {n_fire/max(n_days,1):.1f} fires/day across "
                     f"{n_days} dates")
        logging.info(f"  UpProb distribution (universe-passing): "
                     f"min={up_prob[quality_mask.values].min():.3f} "
                     f"p50={np.median(up_prob[quality_mask.values]):.3f} "
                     f"p99={np.percentile(up_prob[quality_mask.values], 99):.3f} "
                     f"max={up_prob[quality_mask.values].max():.3f}")
        # How many universe-passing rows actually land in [0.45, 0.70] gate?
        in_gate = (up_prob >= 0.45) & (up_prob <= 0.70) & quality_mask.values
        logging.info(f"  rows in backtester gate [0.45, 0.70]: "
                     f"{int(in_gate.sum()):,} ({100*in_gate.sum()/max(n_pass,1):.2f}% of universe)")

    with Phase("Assemble output + write per-ticker parquets"):
        combined["UpProbability"] = np.clip(up_prob, 0.01, 0.99).astype(np.float32)
        combined["DownProbability"] = (1.0 - combined["UpProbability"]).astype(np.float32)
        combined["PositiveThreshold"] = np.float32(1.0 - args.top_frac_per_day)
        combined["NegativeThreshold"] = np.float32(np.nan)
        combined["UpPrediction"] = up_pred
        combined["raw_score"] = scores.astype(np.float32)  # for debugging only

        out_cols = [args.date_col, "Open", "High", "Low", "Close", "Volume",
                    "UpProbability", "DownProbability", "PositiveThreshold",
                    "NegativeThreshold", "UpPrediction"]
        if "VIX_Close" in combined.columns:
            out_cols.append("VIX_Close")
        for opt in ["Distance to Resistance (%)", "Distance to Support (%)", "volatility"]:
            if opt in combined.columns:
                out_cols.append(opt)
        out_cols = [c for c in out_cols if c in combined.columns]

        # Add raw_score for debugging (not used by backtester)
        if "raw_score" in combined.columns:
            out_cols.append("raw_score")

        n_tickers = combined[args.ticker_col].nunique()
        pbar = tqdm(total=n_tickers, desc="Writing predictions")
        n_written = 0
        for tk, grp in combined.groupby(args.ticker_col, sort=False):
            try:
                grp[out_cols].to_parquet(
                    os.path.join(args.output_dir, f"{tk}.parquet"), index=False)
                n_written += 1
            except Exception as e:
                logging.warning(f"  failed to write {tk}: {e}")
            pbar.update(1)
        pbar.close()
        logging.info(f"  wrote {n_written:,} ticker parquets to {args.output_dir}")

    logging.info(f"\nDone.  Run the backtester next:")
    logging.info(f"    python 5__NightlyBackTester.py")


if __name__ == "__main__":
    main()
