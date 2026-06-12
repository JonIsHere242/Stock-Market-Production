"""Adversarial validation — find features whose distribution shifts across the
~Sep-2025 regime boundary.

A classifier is trained to tell "pre-collapse" rows (date < SPLIT) from
"post-collapse" rows (date >= SPLIT) using the model's features. If it succeeds
(AUC >> 0.5), the eras are distinguishable, and the features it leans on hardest
are the ones that DRIFTED. Features that are both (a) strong era-discriminators
here AND (b) important to the return model are the dangerous regime-fingerprints:
they carry signal that is specific to the old regime and won't transfer OOS.

Run AFTER the walk-forward finishes (avoids CPU contention). Cross-reference the
output against the decontam pattern list in 4__Predictor.py:is_regime_feature.

  python analysis_output/adversarial_validation.py
  python analysis_output/adversarial_validation.py --split 2025-09-01 --max_files 1500
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# Reuse the production model's feature view + filters.
sys.argv_backup = sys.argv
sys.argv = [sys.argv[0]]  # keep 4__Predictor argparse from grabbing our flags
import importlib.util
spec = importlib.util.spec_from_file_location("predictor", "4__Predictor.py")
pred = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pred)
sys.argv = sys.argv_backup

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="2025-09-01",
                help="Date boundary: rows before = era0 (pre-collapse), after = era1.")
ap.add_argument("--input_dir", default="Data/ProcessedData")
ap.add_argument("--max_files", type=int, default=1500,
                help="Ticker cap (adversarial val is robust to sampling).")
ap.add_argument("--model_dir", default="Data/XGBPipeline")
A = ap.parse_args()

split = pd.Timestamp(A.split)

# ---- load (raw base features only — distribution shift lives in the raw cols) ----
files = sorted(f for f in os.listdir(A.input_dir) if f.endswith(".parquet"))[:A.max_files]
def _load(fn):
    t = pq.read_table(os.path.join(A.input_dir, fn))
    if "Date" not in t.schema.names or len(t) == 0:
        return None
    return t.to_pandas()
parts = []
with ThreadPoolExecutor(max_workers=16) as ex:
    futs = {ex.submit(_load, fn): fn for fn in files}
    for f in as_completed(futs):
        r = f.result()
        if r is not None:
            parts.append(r)
df = pd.concat(parts, ignore_index=True)
df["Date"] = pd.to_datetime(df["Date"])
print(f"loaded {df.shape[0]:,} rows from {len(parts)} tickers")

# Same universe gate as training, so we measure shift on the tradeable universe.
df, _ = pred.apply_quality_filter(df)

base_cols = pred.select_base_features(df)
era = (df["Date"] >= split).astype(int).values
print(f"era split @ {split.date()}: pre={int((era==0).sum()):,}  post={int((era==1).sum()):,}")

X = df[base_cols].astype(np.float32).replace([np.inf, -np.inf], np.nan)

# Temporal balance: subsample the larger era so the classifier can't win on base rate.
rng = np.random.default_rng(42)
idx0 = np.where(era == 0)[0]; idx1 = np.where(era == 1)[0]
n = min(len(idx0), len(idx1))
keep = np.concatenate([rng.choice(idx0, n, replace=False), rng.choice(idx1, n, replace=False)])
rng.shuffle(keep)
Xb, yb = X.iloc[keep], era[keep]
cut = int(len(keep) * 0.7)

clf = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.6, tree_method="hist",
                    eval_metric="auc", n_jobs=-1, random_state=42)
clf.fit(Xb.iloc[:cut], yb[:cut])
auc = roc_auc_score(yb[cut:], clf.predict_proba(Xb.iloc[cut:])[:, 1])
print(f"\nADVERSARIAL AUC = {auc:.4f}  (0.5 = eras indistinguishable; >>0.5 = strong drift)")

imp = (pd.DataFrame({"feature": base_cols, "adv_importance": clf.feature_importances_})
       .sort_values("adv_importance", ascending=False).reset_index(drop=True))

# Tag which of these the model also relies on, and which decontam already drops.
summ_path = os.path.join(A.model_dir, "summary.json")
model_top = set()
if os.path.exists(summ_path):
    import json
    with open(summ_path) as f:
        model_top = {r["feature"].replace("_xs", "")
                     for r in json.load(f).get("top_features", [])}
imp["in_model_top20"] = imp["feature"].isin(model_top)
imp["decontam_drops"] = imp["feature"].apply(pred.is_regime_feature)

print("\n=== Top 25 era-discriminating features (highest distribution drift) ===")
print(imp.head(25).to_string(index=False))

danger = imp[(imp["in_model_top20"]) & (imp["adv_importance"] > 0)].head(15)
print("\n=== DANGER: high drift AND in the model's top-20 predictive features ===")
print(danger.to_string(index=False) if len(danger) else "  (none of the model's top features are strong drift drivers)")

out = os.path.join(A.model_dir, "adversarial_validation.parquet")
imp.to_parquet(out, index=False)
print(f"\nsaved -> {out}")
