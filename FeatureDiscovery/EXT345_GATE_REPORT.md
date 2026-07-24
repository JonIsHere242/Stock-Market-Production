# ext3 / ext4 / ext5 candidate gate -- n=120 (2026-06-28)

Gated the previously-ungated `_cand_ext3/ext4/ext5_*.py` blocks (84 total) with
`validate_feature.py --batch "_cand_ext[345]_*.py" --n 120 --keep_fail`.
Results were only in `Data/PaperFeed/battery_results.csv` (overwritten by the next gate),
so they are preserved here.

**Verdicts:** 13 PASS, 65 WEAK, 5 FAIL, 1 ERROR.

## PASS (promote to the heavy multi-seed backtest gate)

| block | IC | OOS_IC | maxcorr | note |
|---|---:|---:|---:|---|
| cand_ext3_beta_instability | 0.0445 | 0.0607 | 0.86 | ties overall-best OOS |
| cand_ext5_covar | 0.0439 | 0.0596 | 0.95 | redundant-ish (high corr) |
| cand_ext4_beta_regime_instability | 0.0416 | 0.0588 | 0.85 |  |
| cand_ext4_idio_vol_multifactor | 0.0468 | 0.0585 | 0.94 |  |
| cand_ext4_hybrid_semibeta_idiovol | 0.0446 | 0.0578 | 0.95 |  |
| cand_ext4_vol_of_vol | 0.0372 | 0.0469 | 0.88 |  |
| cand_ext4_growth_beta | 0.0347 | 0.0363 | 0.87 |  |
| cand_ext5_path_winding | 0.0405 | 0.0310 | 0.67 | NEW axis (path geometry), lower corr |
| cand_ext4_size_beta | 0.0280 | 0.0298 | 0.92 |  |
| cand_ext3_semibeta_trend | 0.0242 | 0.0296 | 0.87 |  |
| cand_ext5_quantile_beta | 0.0284 | 0.0275 | 0.81 |  |
| cand_ext3_semibeta_vs_vix | 0.0260 | 0.0266 | 0.87 |  |
| cand_ext5_vpt_divergence | 0.0164 | 0.0256 | 0.80 |  |

## Most orthogonal (low corr, marginal-value candidates -- WEAK standalone)

| block | verdict | OOS_IC | maxcorr |
|---|---|---:|---:|
| cand_ext3_automutual_info | WEAK | 0.0207 | 0.43 |
| cand_ext3_entropy_rate | WEAK | 0.0200 | 0.45 |

## FAIL / ERROR (discard or fix)

- cand_ext3_halloween_sas -- GARBAGE (Series.day attr); **calendar/seasonality dead vein**
- cand_ext3_holiday_proximity -- GARBAGE (Timestamp.astype); calendar vein
- cand_ext3_quarter_end_dressing -- **LEAK** (causality failed); calendar vein
- cand_ext5_body_wick -- REDUNDANT (duplicates existing candle features)
- cand_ext5_close_accumulation -- GARBAGE (boolean index mismatch)
- cand_ext5_ma_alignment -- ERROR (walrus rebind of comprehension var `k_idx`)

The 3 calendar/seasonality FAILs confirm the skill's "dead vein" note (per-ticker
calendar leaks/sparse). The 3 GARBAGE/ERROR blocks are codegen bugs, not idea failures.
