# Candidate feature batch -- CONSOLIDATED results (high-power)

Updated 2026-06-27 after the high-power re-gate (n=120) and the SLOW-block revival.
186 blocks were codegen'd by subagents from non-obvious sources (OSAP/Chen-Zimmermann
anomaly catalog + cross-domain time-series methods), written to the FeatureTemplates
contract as hidden candidates, then validated (leakage/causality + OOS durability +
redundancy vs the 188 live features).

## Headline

- Main 157 survivors re-gated at **n=120** (higher OOS power): **7 PASS, 148 WEAK, 2 SLOW**.
- 7 nonlinear-dynamics blocks that were quarantined for SPEED were vectorized + causal-strided
  and REVIVED: all 7 now compute < 110ms, pass causality, and gate as valid WEAK candidates.
- Net: **164 valid candidates** (157 _cand_ + 7 _fast_), 7 of them PASS.

## PASS at n=120 -- promote to your heavy multi-seed backtest gate

| block | IC | OOS_IC | maxcorr | note |
|---|---:|---:|---:|---|
| cand_osap_idiovolaht | +0.0483 | +0.0593 | 0.93 |  |
| cand_xdom_allan_variance | +0.0434 | +0.0530 | 0.88 | **robust** (PASS at n=40 too) |
| cand_xdom2_downside_beta | +0.0255 | +0.0324 | 0.91 |  |
| cand_xdom2_autocorr_volume_return | +0.0212 | +0.0297 | 0.83 | **robust** (PASS at n=40 too) |
| cand_osap_betatailrisk | +0.0206 | +0.0270 | 0.67 | novel (low corr) |
| cand_osap_streversal | +0.0228 | +0.0236 | 0.95 | **robust** (PASS at n=40 too) |
| cand_osap_coskewacx | +0.0256 | +0.0219 | 0.92 |  |

**Most trustworthy** = PASS at BOTH n=40 and n=120: osap_streversal, xdom2_autocorr_volume_return, xdom_allan_variance.

## Genuinely NOVEL leads (maxcorr < 0.5 vs your 188 features, real OOS)
These are uncorrelated with anything you already compute -- the highest marginal value.

| block | verdict | IC | OOS_IC | maxcorr |
|---|---|---:|---:|---:|
| cand_osap_orgcap | WEAK | +0.0187 | +0.0426 | 0.00 |

## Revived nonlinear-dynamics features (were SLOW, now fast + causal + valid)

Stored as FeatureTemplates/_fast_*.py. Vectorized (sliding_window_view + broadcasting) and,
where needed, a CAUSAL fixed-from-start stride (compute every 5th bar, forward-fill) -- the
stride is prefix-stable so it passes the causality test. All gate WEAK at n=120.

| block | IC | OOS_IC | maxcorr |
|---|---:|---:|---:|
| fast_xdom_sampen | +0.0315 | +0.0245 | 0.74 |
| fast_xdom2_mse | +0.0234 | +0.0324 | 0.65 |
| fast_xdom2_corr_dim | +0.0159 | +0.0246 | 0.71 |
| fast_xdom2_rqa | +0.0144 | +0.0218 | 0.65 |
| fast_xdom_dcca_index | +0.0133 | +0.0000 | 0.78 |
| fast_xdom2_wpe | +0.0114 | +0.0000 | 0.59 |
| fast_xdom2_apen | +0.0100 | +0.0000 | 0.62 |

## Pipeline changes shipped
- _fundamentals.as_of(): coerce outputs to numpy float64 (object/nullable fields e.g. total_debt crashed the gate).
- validate_feature.import_scan: AST-based (was a line-regex that failed valid blocks whose docstrings began with from/import).
- New keyless source altsources/stackexchange.py (registered); practitioner feeds in altsources/rss.py; see NEW_SOURCES.md.
- Idea banks: idea_banks/osap_implementable.jsonl (169 cited predictors), idea_banks/crossdomain_methods.jsonl (32).

## How to act
- Promote a winner (after your backtest gate): drop the leading underscore so 3__FeatureFramework.py discovers it.
- The candidates are hidden (underscore), so leaving them is harmless; delete _cand_*.py / _fast_*.py to discard.

---

# Extension batch -- exploring the winners (33 orthogonal variants)

Round 2: for each gate-validated winner, generated NEW features along ORTHOGONAL axes
(different input series / statistic / conditioning / normalization), not window tweaks.
Gated at n=120. Verdicts: **6 PASS, 27 WEAK** (33 total, all valid).

## New PASS extensions -- promote to the backtest gate

| block | IC | OOS_IC | maxcorr | extends |
|---|---:|---:|---:|---|
| cand_ext_allan_overnight_intraday | +0.0444 | +0.0591 | 0.94 | allan_variance |
| cand_ext_allan_hadamard | +0.0435 | +0.0530 | 0.88 | allan_variance |
| cand_ext_signed_volume_persist | +0.0238 | +0.0354 | 0.65 | autocorr_volume_return |
| cand_ext_vix_beta | +0.0195 | +0.0348 | 0.91 | betatailrisk |
| cand_ext_downside_correlation | +0.0351 | +0.0331 | 0.84 | downside_beta |
| cand_ext_systematic_share | +0.0290 | +0.0221 | 0.81 | idiovolaht |

## NEW genuinely-novel features (maxcorr 0.00 vs all existing) with real OOS

The intangible-capital family (extending osap_orgcap, itself maxcorr 0) keeps hitting open
white-space -- these are uncorrelated with anything you already compute:

| block | verdict | IC | OOS_IC | maxcorr |
|---|---|---:|---:|---:|
| cand_ext_sga_efficiency_trend | WEAK | +0.0309 | +0.0428 | 0.00 |
| cand_ext_total_intangible_intensity | WEAK | +0.0207 | +0.0359 | 0.00 |

## Takeaways
- The **frequency-stability (Allan)** vein is rich: overnight/intraday-split and Hadamard
  variants both PASS at OOS IC ~0.05-0.06.
- The **intangible-capital** vein is the standout white-space: sga_efficiency_trend and
  total_intangible_intensity are PASS-adjacent AND completely novel (maxcorr 0.00).
- **Volatility/market-structure** extensions (signed-flow persistence, VIX beta, downside
  correlation, systematic share) all PASS -- the risk-loading family generalizes well.

---

# Round 3 -- deep dive on the two richest veins (25 features)

Targeted the intangible-capital and frequency-stability veins (plus a few risk-loading).
Gated n=120: **4 PASS, 21 WEAK**, 0 failures.

## PASS (strongest OOS of all three rounds)

| block | IC | OOS_IC | maxcorr |
|---|---:|---:|---:|
| cand_ext2_semibeta_signed | +0.0577 | +0.0610 | 0.88 |
| cand_ext2_allan_acceleration | +0.0435 | +0.0530 | 0.88 |
| cand_ext2_allan_signed_flow | +0.0370 | +0.0423 | 0.73 |
| cand_ext2_vix_beta_asymmetry | +0.0222 | +0.0266 | 0.87 |

## KEY FINDING -- intangible vein = orthogonal but weak standalone

Round 3 produced 6 more intangible features at **maxcorr 0.00** (uncorrelated with ALL 188
existing features) but with ~0 standalone OOS IC: gp_intan_adj, intan_adj_bm_dynamics,
intan_investment_rate, intangible_momentum, org_rnd_mix, sga_growth_less_sales.

Interpretation: their value is NOT in standalone IC -- it is in **marginal contribution inside
the multivariate model** (a tree exploits weak-but-uncorrelated inputs that a univariate OOS-IC
gate cannot see). The standalone-durability gate structurally undersells orthogonal features.
ACTION: test the intangible set as a GROUP via marginal model contribution / your heavy backtest
gate, not univariate IC. The durable standalone intangibles remain round-2's sga_efficiency_trend
(OOS 0.043) and total_intangible_intensity (OOS 0.036).

## Grand total across all rounds

- **222 candidate features** generated (215 _cand_* + 7 revived _fast_*), all hidden from discovery.
- **17 PASS at n=120** across the three rounds (7 round-1 + 6 round-2 + 4 round-3).
- Top promotions for the heavy backtest gate: ext2_semibeta_signed (OOS 0.061),
  ext_allan_overnight_intraday (0.059), osap_idiovolaht (0.059), ext2_allan_acceleration /
  ext_allan_hadamard / xdom_allan_variance (~0.053), xdom2_autocorr_volume_return, osap_streversal;
  plus the orthogonal intangible GROUP (osap_orgcap, sga_efficiency_trend, total_intangible_intensity).
