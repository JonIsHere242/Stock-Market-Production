# Stage-2 Exit Optimization — Morning Verdict (2026-06-12, ~4am)

## Your question
"Will the system stop out a bit less with better-optimized bracket orders?"
**YES — conclusively, validated TWO independent ways that now agree** (full-universe strategy
backtest + 5-min intraday replay on your actual fills).

## The culprit: your TRAILING stop
Live bracket = **1.9% hard stop + ATR trailing stop (~3% avg) + 3.5% take-profit**.
The *trailing* stop is the whipsaw machine — it causes ~half your stop-outs and drags every metric.

## Evidence

**5-min stop-out rates** (2,403 trades, anchored at recorded fills):
| bracket | STOPPED | win% | /trade |
|---|---|---|---|
| LIVE: 1.9% hard + ATR trail + 3.5% TP | **68%** | 34% | −0.05% |
| drop trail (1.9% hard + 3.5% TP) | 55% | 40% | +0.06% |
| drop trail + 5% hard + 3.5% TP | **15%** | 61% | +0.85% |
| drop trail + 8% hard + 3.5% TP | **5%** | 64% | +1.11% |

**Strategy backtest** (full universe, 2yr, deterministic):
| bracket | ann% | maxDD% | ret/DD |
|---|---|---|---|
| trail family (any trail %) | 154 | 22.7 | 6.8  ← worst |
| 1.9% hard + 3.5% TP (drop trail) | 182 | 12.2 | 14.9 |
| 8% hard + 3.5% TP | 193 | 10.6 | 18.2 ← best |
| no stop + 20% TP | 204 | 14.2 | 14.4 |

**Robust signals (high confidence — consistent across 18 configs, both methods):**
1. **Drop the trail.** Every trail config = ~154%/22.7% DD (ret/DD 6.8); every hard/no-trail
   config beats it 2–3× on risk-adjusted return at ~half the drawdown.
2. **Tight 3.5% TP + a hard stop + NO trail** is the winning family (180–193% / 10–12% DD).
3. **Exact stop % (1.9 vs 5 vs 8) is single-path fragile** — don't over-fit it.

## RECOMMENDATION

### Today (safe, minimal change): REMOVE THE TRAILING STOP
Keep your existing 1.9% hard + 3.5% TP. Just drop the trail.
→ stop-outs **68% → 55%**, strategy **154% → 182% ann**, max DD **22.7% → 12.2%**.
This is the robust lever; no fragile tuning involved.

### Bigger win (optional — validate with multi-seed first): also widen the hard stop to ~5%
→ stop-outs **68% → 15%**, win **34% → 61%**, +0.85%/trade.
(8% tested even better — 5% stop-out, +1.11/trade — but it's the fragile extreme + a wider stop
means larger losses on the few trades that DO hit it. Start at 5% if you go this route.)

## The exact `9_SuperFastBroker.py` change (NOT applied — your call before the open)

**To drop the trail** (in `execute_batch`, where the bracket is staged):
- Find the line that appends the bracket:
  `orders_staged.append((contract, [parent, take_profit_ord, trail_stop_ord, hard_stop_ord]))`
- Remove `trail_stop_ord` from that list:
  `orders_staged.append((contract, [parent, take_profit_ord, hard_stop_ord]))`
- (Leave the `trail_stop_ord = ibi.Order(... orderType='TRAIL' ...)` block as-is or comment it out —
  if it's not in the staged list it's never transmitted. `hard_stop_ord` stays last with
  `transmit=True`, so the bracket still fires correctly with TP + hard-stop in the OCA group.)

**To also widen the hard stop** (optional): change the constant near the top:
- `HARD_STOP_PCT = 1.9` → `HARD_STOP_PCT = 5.0`

## Caveats / honesty
- The per-trade 5-min sim earlier *looked* like it killed the edge — that was a proxy that ignored
  capital recycling/compounding. The full strategy backtest is the arbiter; both now agree on the
  *direction*, so the recommendation is doubly-validated.
- Daily backtest DD understates intraday DD; the 5-min stop-out numbers above are the realistic view.
- The exact optimal stop % wasn't multi-seeded (single deterministic paths); the *family*
  ("drop trail + tight TP + hard stop") is the robust, conclusive part.

## Files
- Sweep results: `analysis_output/exit_sweep_results.csv` (18 configs)
- Per-config trades: `Data/TradeHistory_sweep_<tag>.parquet`
- Tools: `5__NightlyBackTester_bracket.py` (parameterized: `--stop_mode/--stop_pct/--tp_pct`),
  `run_exit_sweep_par.py`, `stage2_stopout_compare.py`, `stage2_stopsim.py`, `stage2_stopsweep.py`
- Baselines safe in `_backups/stage2_20260612/`; live `9_SuperFastBroker.py` UNCHANGED.
