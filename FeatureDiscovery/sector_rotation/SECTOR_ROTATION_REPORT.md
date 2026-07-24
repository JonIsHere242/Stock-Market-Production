# Sector Rotation, Quantified (2026-07-07)

**Question:** is "sector rotation" a real, exploitable mechanic or a macro-guy myth?
**Data:** `Data/PriceDataFull` (4,055 tickers, 1996-2026) x `Data/SectorMap.parquet`, normalized to
12 equal-weight sector indices (REITs split from Financials, ConsumerStaples carved from SIC
descriptions), 11.2M liquid stock-days (close >= $2, 21d median $vol >= $500k), winsorized 1%/99%.
"Relative" = sector return minus equal-weight all-stock mean. Panel builder: `build_sector_panel.py`
(in this folder). Full interactive report + charts: https://claude.ai/code/artifact/a55d02a1-31b5-455d-865e-5bbee10d96f3

**Method:** 6 analysis lenses, pre-registered significance bar (|t|>=3 NW or block-bootstrap p<0.005
PLUS sign consistency in >=3/4 subperiods: pre-2008 / 2008-15 / 2016-20 / 2021-26). Every headline
claim independently re-derived from scratch by an adversarial verifier agent (13 CONFIRMED,
3 WEAKENED, 0 REFUTED). LEAK-FREE = trailing-info-only, eligible as a feature; RETRO = hindsight.

## Verdicts

| Claim | Verdict | Tag |
|---|---|---|
| Weekly/monthly cross-sector lead-lag | **MYTH** - 1/132 (5d) and 0/132 (21d) pairs significant = exact chance rate | LEAK-FREE |
| Daily cross-sector lead-lag | Real but tiny: VAR kills all but ~1-4 pairs (mostly Utilities-sourced), 2-4 bps/day gross, decaying since 2008, below cost floor; partly stale-close microstructure | LEAK-FREE |
| "Tech dumped -> buy Y" (792 trigger/response rules) | **NOISE** - survivor set is construction-fragile, matches multiplicity expectation. Tech-down predicts nothing; Industrials-down->Healthcare sign flips across constructions | LEAK-FREE |
| Post-crash snapback | Only near-usable conditional: Tech +72bps/21d rel. after SPY 5d-crash (4/4 subperiods) but overlap-honest t=1.9, fam-wise p=.04; it is BETA reversion, not rotation. Sizing tilt at most | LEAK-FREE |
| Sector momentum (rotation trade) | THIN: all (k,h) cells positive (never reversal!) but only k=5d/h=1d passes (t=3.3, ~9%/yr gross); dot-com-concentrated, ~0 since 2008, dies with skip-a-day execution, breakeven cost ~3.3bps | LEAK-FREE |
| Sector mean-reversion / contrarian rotation | ABSENT at every horizon - never fade sector leaders | LEAK-FREE |
| Defensive-cyclical spread -> fwd market returns | NO (|t|<=1.1, sign-unstable; timing rule loses to buy-hold 24/24 constructions; exit-state days have ABOVE-avg next-day returns) | LEAK-FREE |
| Defensive spread -> fwd realized vol beyond VIX | Small but real (t=3.2, 4/4 signs, R2 .52->.54; top quintile ~+2 vol pts). Risk thermometer, not return signal | LEAK-FREE |
| Business-cycle rotation clock | Predicts next 6-mo leader at chance (25.5% vs 24.4%, p=.35); clock-successor portfolio loses money. Verifier residue: same-STAGE persistence 37% vs 28% (p=.01-.09) - untested lead | RETRO |
| Leadership rotates across episodes | YES (descriptive): 7 distinct #1 sectors across 9 episodes, mean cross-episode rank corr ~0; structure = ONE risk-on/off axis + episode idiosyncrasy. Corr-PCA PC1 IS cyclical-vs-defensive; Energy is the top VARIANCE source only | RETRO |
| Stable co-movement blocks | Staples-Utilities +0.64 (4/4), Energy-Materials +0.38 (98% of windows), Tech-Utilities see-saw -0.43 (92%); ConsDisc-ConsStaples "risk pair" = null | RETRO |

## Feature triage (stock-level, next-day, vs production model shape)

- **f1/f2/f3** sector trailing 21d/63d relative momentum (+rank): **DEAD**. IC ~0.002, sign flips
  post-2008, no 5d-horizon rescue. f3 identical to f1 for trees.
- **f4** stock 5d return minus sector 5d return: **WEAK**. Strong standalone (t=-11) but 0.93
  rank-corr with plain rev5; marginal value post-2008 = -0.08 bps (t=-0.1). Repackaged reversal.
- **f5** trailing 21d cross-sector dispersion (std of sector 21d relative returns): **PROMOTE** as
  regime/interaction feature. High dispersion -> stronger next-day reversal efficacy: post-2008
  rev5 decile spread -27 bps/day in top-f5 quintile vs +3 in bottom (~30 bps swing in the model's
  main edge), 4/4 subperiod signs, survives log-VIX control pooled (t=3.0-3.8; ~half is VIX-shared).
  Verifier matched construction to 1e-16. Route through the normal feature-factory multi-seed gate.

## Honesty box

Survivor-biased price lake (relative results partly insulated; pre-2008 magnitudes most suspect).
Equal-weight indices overweight small caps (stale closes can manufacture daily lead-lag - the
surviving daily pairs are treated as microstructure, not alpha). Sector tags are today's
classification applied to history. All strategy numbers gross of costs. MarketCaps history only
2023-09+ so no cap-weighted variant.

## Bottom line

Sector rotation is **real as a description, dead as a prediction**. Sectors take turns leading, but
the next leader is not forecastable from sector returns (pairwise, weekly, monthly, or clock).
What survives: 1-day continuation (decayed), crash beta-snapback (weak), and **f5 sector
dispersion as a reversal-regime conditioner** - the one thing worth putting through the gate.
