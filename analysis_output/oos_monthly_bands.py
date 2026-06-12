"""Per-month robustness check on the OOS band finding.

Reads Data/XGBPipeline/oos_scores.parquet (written by `4__Predictor.py --oos_only`)
and asks: is "shoulder beats top-1%" consistent across months, or one-month luck?

For each month it reports per-day mean return (the strategy-relevant metric) for
the top-1% band, the 0.90-0.93 / 0.90-0.95 shoulder, and the bottom-50% baseline,
plus the BETA-ADJUSTED spread (band minus baseline) which strips market drift.
"""
import pandas as pd
import numpy as np

df = pd.read_parquet("Data/XGBPipeline/oos_scores.parquet")
df["Date"] = pd.to_datetime(df["Date"])
df["month"] = df["Date"].dt.to_period("M").astype(str)


def perday_ret(sub):
    """Mean across days of each day's mean return, in %."""
    if len(sub) == 0:
        return np.nan
    return sub.groupby("Date")["_ret"].mean().mean() * 100.0


def band(g, lo, hi):
    return g[(g["_rank"] >= lo) & (g["_rank"] < hi)]


rows = []
for (mon, split), g in df.groupby(["month", "split"]):
    if split == "embargo":
        continue
    base = perday_ret(band(g, 0.0, 0.50))          # bottom-50% = market-beta proxy
    top1 = perday_ret(band(g, 0.99, 1.0001))
    sh93 = perday_ret(band(g, 0.90, 0.93))
    sh95 = perday_ret(band(g, 0.90, 0.95))
    rows.append({
        "month": mon, "split": split, "days": g["Date"].nunique(),
        "base%": base, "top1%": top1, "sh.90-.93%": sh93, "sh.90-.95%": sh95,
        "top1-base": top1 - base, "sh.90-.93-base": sh93 - base,
        "sh.90-.95-base": sh95 - base,
    })

out = pd.DataFrame(rows).sort_values(["split", "month"])
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda x: f"{x:7.4f}")

print("=" * 110)
print("PER-MONTH per-day mean return % by band  (base = bottom-50% = market drift proxy)")
print("=" * 110)
for split in ["train", "calib", "oos"]:
    s = out[out["split"] == split]
    if len(s) == 0:
        continue
    print(f"\n### split = {split} ###")
    print(s.drop(columns="split").to_string(index=False))

print("\n" + "=" * 110)
print("ROBUSTNESS: months where shoulder(.90-.95) net-of-base BEATS top1 net-of-base")
print("=" * 110)
oos = out[out["split"] == "oos"]
if len(oos):
    wins = (oos["sh.90-.95-base"] > oos["top1-base"]).sum()
    print(f"OOS months: shoulder>top1 (net of base) in {wins}/{len(oos)} months")
    print(f"OOS mean net-edge:  top1={oos['top1-base'].mean():.4f}  "
          f"shoulder.90-.95={oos['sh.90-.95-base'].mean():.4f}")
    print(f"OOS median net-edge: top1={oos['top1-base'].median():.4f}  "
          f"shoulder.90-.95={oos['sh.90-.95-base'].median():.4f}")
