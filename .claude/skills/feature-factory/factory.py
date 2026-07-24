#!/usr/bin/env python
"""
factory.py  --  the engine behind the `feature-factory` skill.

Spammable feature generation + gating for the Stock-Market FeatureTemplates pipeline.
Encodes the lessons from the 2026-06-27/28 feature-factory campaign:
  - bias generation toward PRODUCTIVE + ORTHOGONAL veins (risk-loading, multi-index
    factors, frequency-stability, intangible capital, info-theory, momentum-quality, ...);
  - never re-emit a CONCEPT already generated (concepts_done.txt ledger);
  - gate at HIGH power (n=120) -- low-n IC is noise (a 0.14 OOS "winner" at n=40 evaporated);
  - heavy/strided metrics must use a CAUSAL fixed-from-start grid (i % stride == 0), never
    anchor to the last bar (that re-anchors under truncation and leaks).

SUBCOMMANDS
-----------
  factory.py specs  --vein rotate --n 30 --batch 0701a   # write N fresh specs, print ids JSON
  factory.py gate   --batch 0701a [--n 120]              # gate that batch, append to ledger+REPORT
  factory.py status                                      # cumulative scoreboard
  factory.py veins                                       # list veins
"""
from __future__ import annotations
import argparse, csv, glob, itertools, json, os, random, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SPECS_DIR = HERE / "specs"
LEDGER = HERE / "ledger.csv"
REPORT = HERE / "REPORT.md"
CONCEPTS = HERE / "concepts_done.txt"
TEMPLATES = ROOT / "FeatureTemplates"
VALIDATOR = ROOT / "FeatureDiscovery" / "validate_feature.py"
BATTERY = ROOT / "Data" / "PaperFeed" / "battery_results.csv"

# ---- producer/consumer QUEUE (parallel mode) ------------------------------------------
# The seam that lets an IDEATION agent (producer) and a FORGE/code-monkey agent (consumer)
# work concurrently and safely. Single-writer-per-file + atomic os.replace handoff:
#   producer  -> writes queue/pending/<vein>__<slug>.spec  (topup / enqueue)
#   consumer  -> atomically claims pending -> claimed, codegens+gates, releases -> done
# Concept == filename stem == the dedup key (also recorded in concepts_done.txt).
QUEUE      = HERE / "queue"
Q_PENDING  = QUEUE / "pending"     # authored, waiting for the code monkey
Q_CLAIMED  = QUEUE / "claimed"     # in flight (claimed by a forge run)
Q_DONE     = QUEUE / "done"        # codegenned + gated
Q_MANIFEST = QUEUE / "manifests"   # <forgebatch>.json -> {batch, ids, concepts}

IDXNOTE = ("Use the _indexes helper (import by path): _indexes.index_close('SPY'|'QQQ'|'IWM'|'DIA') "
           "-> Close Series; _indexes.vix_daily_close() -> [Date,vix_close]; align via merge_asof backward; "
           "degrade to NaN if missing. ")
FUNDNOTE = ("Use _fundamentals.as_of(df, fields=[...]) (import by path) -> fund_<field>, backward merge on "
            "filed_date (lookahead-safe); NEVER period_end/period_start; drop scratch fund_ cols. ")
STRIDENOTE = ("If the metric is expensive (>100ms), compute it on a CAUSAL fixed-from-start grid "
              "(only where i % 5 == 0, measured from the series start) and forward-fill -- NEVER anchor the "
              "grid to the last bar (it re-anchors under truncation and leaks). ")

# ---------------------------------------------------------------------------
# VEIN CATALOG  --  each entry: (subtype_template, definition_template, {param:[values]})
# Bias: veins proven productive in the campaign get more angles. The combinatorial space
# is in the thousands -> effectively unlimited spam fuel. Add angles/params freely.
# ---------------------------------------------------------------------------
CATALOG = {
 "beta_risk": [
  ("{stat}beta_{ref}_{win}",
   IDXNOTE+"Rolling {win}d {stat}beta of stock returns vs {ref} (downside=only {ref}-down days, upside=only "
   "{ref}-up days, full=all days); produce the beta level, its short-vs-long ratio, and the down-minus-up asymmetry.",
   {"stat":["downside ","upside ","full "], "ref":["SPY","QQQ","IWM"], "win":[120,250]}),
  ("beta_instability_{ref}_{win}",
   IDXNOTE+"Compute rolling 40d beta vs {ref} repeatedly, then produce the {win}d std of that beta series "
   "(beta instability), its range, and the down-regime-minus-up-regime instability.",
   {"ref":["SPY","QQQ","IWM"], "win":[120,180]}),
  ("semibeta_signed_{win}",
   IDXNOTE+"Bollerslev-Li-Patton signed semibetas vs SPY over {win}d: decompose cov by sign quadrant into "
   "concordant-down (beta_N), concordant-up (beta_P), discordant-mixed (beta_M); produce all three.",
   {"win":[120,250]}),
  ("tail_risk_composite_{win}",
   IDXNOTE+"Composite tail-risk vs SPY over {win}d: z-score (over 252d) the downside beta, negative coskewness, "
   "and cokurtosis, equal-weight average; produce the composite and its 60d change.", {"win":[252]}),
  ("conditional_beta_vix_{win}",
   IDXNOTE+"Beta to SPY over {win}d split by VIX regime (median VIX over trailing 250d): high-VIX beta, low-VIX "
   "beta, and the stress spread.", {"win":[120,250]}),
  ("downside_idio_beta_{win}",
   IDXNOTE+"From a {win}d market-model (stock on SPY) take residuals; produce downside/upside residual "
   "semideviation ratio and the residual covariance with the market on its worst-quartile days.", {"win":[120]}),
 ],
 "factor": [
  ("{fac}_beta_{win}",
   IDXNOTE+"Build factor = {facdesc}. Rolling {win}d 2-factor OLS of stock return on (SPY, factor); produce the "
   "factor beta, the market beta, and R^2.", {"fac":["size","growth","style"], "win":[120,250]}),
  ("idio_vol_multifactor_{win}",
   IDXNOTE+"Rolling {win}d 3-factor OLS (SPY,QQQ,IWM); produce residual std (clean idiosyncratic vol), R^2 "
   "(systematic share), and a 20d z-score of the idio vol.", {"win":[120,250]}),
  ("dual_beta_spread_{win}",
   IDXNOTE+"Rolling {win}d beta to QQQ minus beta to IWM (large-growth vs small) and its 60d change.", {"win":[120,250]}),
  ("breadth_sensitivity_{win}",
   IDXNOTE+"Rolling {win}d beta of stock return on d log(IWM/SPY) (market-breadth proxy) and its 20d change.", {"win":[120]}),
  ("relative_strength_multi_{win}",
   IDXNOTE+"Stock {win}d cumulative return minus QQQ's, minus IWM's, minus SPY's, and the worst across them.", {"win":[63,126]}),
 ],
 "freq_stability": [
  ("allan_{ser}_{win}",
   "Allan deviation at tau=5 over a {win}d rolling window of {serdesc}: non-overlapping tau-blocks, block means, "
   "sigma_A=sqrt(0.5*mean(diff(block_means)^2)); produce the level, its ratio to rolling std, and a 20d z-score.",
   {"ser":["ret","absret","vol","range","signedflow","overnight","intraday"], "win":[80]}),
  ("hadamard_{win}",
   "3-sample Hadamard deviation (2nd diff of tau=5 block means, drift-INSENSITIVE) over {win}d on log-returns; "
   "produce Hadamard dev, Allan dev, and their ratio (separates drift from flicker/white).", {"win":[80]}),
  ("allan_acceleration_{win}",
   "Rolling Allan deviation (tau=5,{win}d) of log-returns; produce its 10d change and the change-of-change "
   "(acceleration) -- speed of noise-regime shift.", {"win":[80]}),
  ("fano_burstiness_{win}",
   "Counting-statistics burstiness over {win}d (16 blocks of 5d): per-block count of large-move days "
   "(|ret|>window 80th pct); Fano = var/mean of counts (>1 bursty); produce return-Fano and volume-spike-Fano.", {"win":[80]}),
  ("vol_of_vol_{win}",
   "Rolling 20d realized vol series, then its {win}d std/mean (vol-of-vol), the level, and a 60d trend.", {"win":[60]}),
 ],
 "intangible": [
  ("{kind}_to_{denom}",
   FUNDNOTE+"OC = capitalized SG&A (=gross_profit_ttm-operating_income_ttm-rnd_expense_ttm, >=0)/4 via perpetual "
   "inventory at 30%/yr (quarterly delta=1-0.7^0.25); RDC = capitalized rnd_expense_ttm/4 at 20%/yr. IC={kinddesc}. "
   "Produce IC/{denom}, its YoY (252d) growth, and 252d trend ({denom}=market means Close*shares_outstanding).",
   {"kind":["oc","rdc","total"], "denom":["assets","revenue_ttm","market"]}),
  ("intan_adj_{metric}",
   FUNDNOTE+"IC = OC(30%/yr)+RDC(20%/yr) perpetual inventory. Produce the intangible-adjusted {metricdesc}, the "
   "unadjusted version, and the wedge.", {"metric":["roa","gp","bm"]}),
  ("intangible_momentum",
   FUNDNOTE+"Intangible intensity = (SG&A + rnd_expense_ttm)/revenue_ttm; produce its 126d change (ramp), 252d "
   "change, and acceleration.", {}),
  ("sga_efficiency_trend",
   FUNDNOTE+"Organizational efficiency = revenue_ttm/SG&A (=gross_profit_ttm-operating_income_ttm); produce level, "
   "252d trend, and gross_profit_ttm/SG&A.", {}),
  ("rnd_to_capex",
   FUNDNOTE+"Produce rnd_expense_ttm/capex_ttm, rnd/(rnd+capex), and its 252d change.", {}),
  ("org_rnd_mix",
   FUNDNOTE+"OC and RDC via perpetual inventory; produce OC/(OC+RDC) (process vs knowledge firm type) and its 252d change.", {}),
 ],
 "momentum_quality": [
  ("info_discreteness",
   "Da-Gurun-Warachka information discreteness: ID = sign(PRET)*(%neg_days-%pos_days) over trailing 252d skipping "
   "last 21d; produce ID, |PRET|, and the continuous-info signal PRET*(-ID).", {}),
  ("momentum_smoothness",
   "Over 12m (skip last month): fraction of positive ~21d sub-periods (consistency), net/sum-abs path efficiency, "
   "and the signed product with momentum.", {}),
  ("momentum_gap",
   "Novy-Marx intermediate (t-12m..t-7m) minus recent (t-6m..t-1m) momentum, and both components.", {}),
  ("high52_dynamics",
   "George-Hwang: Close/trailing-252d-high (proximity), 20d change (approach speed), bars since the 252d high.", {}),
  ("residual_momentum_{win}",
   IDXNOTE+"Blitz-Huij-Martens residual momentum: {win}d market-model residuals (stock on SPY); 11-month "
   "(skip last) cumulative residual / residual vol, and the 6-month version.", {"win":[120]}),
 ],
 "microstructure": [
  ("corwin_schultz",
   "Corwin-Schultz (2012) high-low spread from consecutive-day H/L; produce rolling 21d mean spread (clip>=0) and 60d trend.", {}),
  ("roll_spread",
   "Roll (1984) implied spread = 2*sqrt(-cov21(ret_t,ret_{t-1})) when autocov<0 else 0; level and 60d change.", {}),
  ("abdi_ranaldo",
   "Abdi-Ranaldo (2017) spread, CAUSAL (eta_{t-1}=(lnH+lnL)/2 lagged): S=sqrt(max(4(c-eta_{t-1})(c_{t-1}-eta_{t-1}),0)); 21d mean and 60d trend.", {}),
  ("amihud_dispersion_{win}",
   "Amihud illiquidity |ret|/(Close*Volume); rolling {win}d mean, coefficient of variation, and 20d change.", {"win":[60]}),
  ("vpin_proxy_{win}",
   "VPIN-lite: buy/sell-classify daily volume via CLV ((Close-Low)-(High-Close))/(High-Low); order imbalance "
   "|buy-sell|/total; rolling {win}d mean OI and 20d change.", {"win":[50]}),
  ("pastor_stambaugh_{win}",
   "Pastor-Stambaugh gamma: rolling {win}d OLS of next-day return on sign(ret)*Close*Volume (causal pairs); gamma and 20d change.", {"win":[60]}),
  ("kyle_obizhaeva_{win}",
   "Kyle-Obizhaeva invariance: rolling {win}d impact = Parkinson-vol / dollar_volume^(1/3), activity = 1/(dollar_volume*vol)^(1/3); produce impact and its 60d trend.", {"win":[21]}),
 ],
 "info_theory": [
  ("{measure}_{ser}_{win}",
   STRIDENOTE+"Rolling {win}d {measuredesc} of {serdesc}; produce the level, a 20d change, and a 60d z-score.",
   {"measure":["sampen","apen","permutation_entropy","weighted_perm_entropy","lempelziv","tsallis2","renyi2"],
    "ser":["ret","absret","vol"], "win":[100,120]}),
  ("transfer_entropy_vix_{win}",
   IDXNOTE+STRIDENOTE+"Approximate transfer entropy from VIX changes to next-day |return| over {win}d via binned "
   "conditional-MI (3 bins each); produce TE and its 20d change.", {"win":[120]}),
  ("automutual_info_{win}",
   STRIDENOTE+"Rolling {win}d auto-mutual-information of returns at lag 1 and 2 (terciles via rolling quantiles); "
   "lag1 AMI, lag2 AMI, and their ratio.", {"win":[120]}),
  ("complexity_entropy_{win}",
   STRIDENOTE+"Bandt-Pompe permutation entropy H (d=4) and Jensen-Shannon statistical complexity C over {win}d; "
   "produce H, C, and the distance from the (H=1,C=0) random corner.", {"win":[120]}),
 ],
 "distribution": [
  ("{moment}_dynamics_{win}",
   "Rolling {win}d {momentdesc} of returns; produce level, 20d slope (direction the distribution evolves), and 60d z-score.",
   {"moment":["skew","kurtosis","downside_semivar_share"], "win":[60]}),
  ("dist_drift_{metric}",
   "Compare last 20d return distribution vs prior 60d via {metric2desc}; produce the drift and its 20d change.",
   {"metric":["wasserstein","ks"]}),
  ("semivol_ratio_{win}",
   "Rolling {win}d downside semideviation / upside semideviation, its 60d trend, and the downside level.", {"win":[40,60]}),
  ("crest_impulse_{win}",
   "Vibration-analysis shape factors on returns over {win}d: crest=max|r|/rms(r), impulse=max|r|/mean|r|; produce both and the 20d change of crest.", {"win":[40]}),
 ],
 "trend": [
  ("ma_spectrum",
   "Close/SMA(L)-1 for L in {{5,10,20,50,100,200}}; produce mean across scales (strength), fraction above (breadth), "
   "and Spearman monotonicity (short>long = clean trend).", {}),
  ("ma_alignment",
   "Fraction of SMA pairs {{5,10,20,50,100,200}} in bullish order (shorter>longer) and its 20d change.", {}),
  ("trend_intensity_{win}",
   "Rolling {win}d R^2 of log-price on time and the slope/residual-std ratio.", {"win":[60,120]}),
  ("efficiency_ratio_{win}",
   "Kaufman efficiency = |Close_t-Close_{t-{win}}| / sum(|dClose|) over {win}d; produce it, its 20d change, and 1-ER (choppiness).", {"win":[20,40]}),
 ],
 "drawdown": [
  ("pain_mar_{win}",
   "Rolling {win}d Pain Index (mean drawdown depth from running peak), MAR ratio (return/maxDD), Martin ratio (return/rms-drawdown).", {"win":[120]}),
  ("underwater_{win}",
   "Fraction of last {win}d underwater, current underwater run length, recovery factor (gain from trough/maxDD).", {"win":[120]}),
  ("drawdown_beta_{win}",
   IDXNOTE+"Rolling {win}d slope of the stock's drawdown series on SPY's drawdown series, and their correlation.", {"win":[120]}),
 ],
 "volume_price": [
  ("obv_divergence_{win}",
   "OBV=cumsum(sign(ret)*Volume); rolling {win}d correlation between OBV trend and price trend (negative=distribution), "
   "its 20d change, and an OBV z-score.", {"win":[40]}),
  ("vw_momentum_gap_{win}",
   "Volume-weighted {win}d cumulative return (sum(ret*vol)/sum(vol)) minus the simple {win}d cumulative return, and its 20d change.", {"win":[60]}),
  ("updown_volume_ratio_{win}",
   "Rolling {win}d up-day volume / down-day volume and its 20d change.", {"win":[40,60]}),
  ("vpt_divergence_{win}",
   "VPT=cumsum(ret*Volume); {win}d slope of normalized VPT minus slope of normalized price, and a 60d z-score.", {"win":[20]}),
 ],
}

_FAC = {"size":"IWM return - SPY return (small-minus-big)",
        "growth":"QQQ return - SPY return (growth/tech-minus-market)",
        "style":"QQQ return - IWM return (growth-minus-small)"}
_SER = {"ret":"daily log-returns","absret":"absolute daily log-returns","vol":"first-differences of log(Volume)",
        "range":"the Parkinson range ln(High/Low)","signedflow":"signed order-flow sign(ret)*detrended-log-volume",
        "overnight":"overnight returns ln(Open/prevClose)","intraday":"intraday returns ln(Close/Open)"}
_KIND = {"oc":"OC","rdc":"RDC","total":"OC+RDC"}
_METRIC = {"roa":"return on capital operating_income_ttm/(assets+IC)","gp":"gross profitability gross_profit_ttm/(assets+IC)",
           "bm":"book-to-market (equity+IC)/(Close*shares_outstanding)"}
_MEASURE = {"sampen":"sample entropy SampEn(m=2,r=0.2*std)","apen":"approximate entropy ApEn(m=2,r=0.2*std)",
            "permutation_entropy":"permutation entropy (order d=3)","weighted_perm_entropy":"weighted permutation entropy (order d=3)",
            "lempelziv":"Lempel-Ziv complexity of the above-rolling-median binarization","tsallis2":"Tsallis q=2 entropy (10-bin rolling-quantile histogram)",
            "renyi2":"Renyi order-2 entropy (10-bin rolling-quantile histogram)"}
_MOMENT = {"skew":"skewness","kurtosis":"excess kurtosis","downside_semivar_share":"downside semivariance share RS-/(RS-+RS+)"}
_METRIC2 = {"wasserstein":"1-Wasserstein (earth-mover) distance between sorted quantiles","ks":"Kolmogorov-Smirnov statistic max|F1-F2|"}


def slug(s): return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


GOAL = ("a NEW, ORTHOGONAL feature (distinct signal axis). The redundancy gate rejects near-"
        "duplicates of the ~190 live features and existing candidates -- prefer genuinely different "
        "structure. Per-ticker; causal/no-lookahead; vectorize <100ms; guard divisions; prefix every "
        "produced column with the spec id.")


def spec_txt(fileid, vein, batch, ddef):
    """The contract spec file workflow.js reads (one per id)."""
    return (f"SPEC ID: {fileid}\nVEIN: {vein}\nBATCH: {batch}\nGOAL: {GOAL}\n\n"
            f"METHOD / DEFINITION:\n{ddef}\n")


# --------------------------------------------------------------------------------------
# QUEUE primitives (shared by producer `topup`/`enqueue` and consumer `claim`/gate-release)
# --------------------------------------------------------------------------------------
def ensure_queue():
    for d in (Q_PENDING, Q_CLAIMED, Q_DONE, Q_MANIFEST):
        d.mkdir(parents=True, exist_ok=True)


def queue_concepts():
    """Every concept currently anywhere in the queue (pending/claimed/done)."""
    ensure_queue()
    return {p.stem for d in (Q_PENDING, Q_CLAIMED, Q_DONE) for p in d.glob("*.spec")}


def _pending_count():
    ensure_queue()
    return len(list(Q_PENDING.glob("*.spec")))


def catalog_fresh():
    """(vein, concept, definition) tuples from the CATALOG not yet done or queued."""
    seen = load_done_concepts() | queue_concepts()
    return [(v, cpt, ddef) for v in CATALOG for (v, cpt, ddef) in expand_vein(v) if cpt not in seen]


def publish_spec(vein, slugpart, definition):
    """Atomically add one spec to pending/ (dedup-checked). Returns True if added."""
    ensure_queue()
    concept = f"{vein}__{slugpart}"
    if concept in load_done_concepts() or concept in queue_concepts():
        return False
    body = f"VEIN: {vein}\nSLUG: {slugpart}\n\nDEFINITION:\n{str(definition).strip()}\n"
    tmp = Q_PENDING / f".{concept}.tmp"
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, Q_PENDING / f"{concept}.spec")   # atomic publish (consumer never sees partial)
    with open(CONCEPTS, "a", encoding="utf-8") as f:
        f.write(concept + "\n")
    return True


def expand_vein(vein):
    """Yield (vein, concept, definition) for every param combination."""
    out = []
    for subtype, deftpl, grid in CATALOG[vein]:
        keys = list(grid)
        combos = [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])] if keys else [{}]
        for c in combos:
            d = dict(c)
            d["facdesc"] = _FAC.get(c.get("fac", ""), "")
            d["serdesc"] = _SER.get(c.get("ser", ""), "")
            d["kinddesc"] = _KIND.get(c.get("kind", ""), "")
            d["metricdesc"] = _METRIC.get(c.get("metric", ""), "")
            d["measuredesc"] = _MEASURE.get(c.get("measure", ""), "")
            d["momentdesc"] = _MOMENT.get(c.get("moment", ""), "")
            d["metric2desc"] = _METRIC2.get(c.get("metric", ""), "")
            try:
                concept = vein + "__" + slug(subtype.format(**c))
                ddef = deftpl.format(**d)
            except Exception:
                continue
            out.append((vein, concept, ddef))
    return out


def load_done_concepts():
    if CONCEPTS.exists():
        return set(l.strip() for l in CONCEPTS.read_text(encoding="utf-8").splitlines() if l.strip())
    return set()


def vein_coverage():
    cov = {v: 0 for v in CATALOG}
    for cpt in load_done_concepts():
        v = cpt.split("__", 1)[0]
        if v in cov:
            cov[v] += 1
    return cov


def cmd_specs(args):
    SPECS_DIR.mkdir(parents=True, exist_ok=True)
    done = load_done_concepts()
    if args.vein not in CATALOG and args.vein not in ("all", "rotate"):
        sys.exit(f"unknown vein {args.vein!r}; choose: {', '.join(CATALOG)} | all | rotate")
    veins = list(CATALOG) if args.vein in ("all", "rotate") else [args.vein]

    universe = [(v, cpt, ddef) for v in veins for (v, cpt, ddef) in expand_vein(v) if cpt not in done]
    if not universe:
        sys.exit("No fresh concepts left in the requested vein(s) -- widen CATALOG params or pick another vein.")

    rng = random.Random(args.seed if args.seed is not None else len(done) * 7919 + 17)
    if args.vein == "rotate":
        cov = vein_coverage()
        universe.sort(key=lambda t: (cov.get(t[0], 0), rng.random()))
    else:
        rng.shuffle(universe)

    pick = universe[:args.n]
    batch = slug(args.batch) or "x"
    ids, new_concepts = [], []
    for vein, cpt, ddef in pick:
        slug_part = cpt.split("__", 1)[1]
        fileid = f"ff{batch}_{vein}_{slug_part}"
        ids.append(fileid)
        new_concepts.append(cpt)
        (SPECS_DIR / f"{fileid}.txt").write_text(spec_txt(fileid, vein, batch, ddef), encoding="utf-8")

    with open(CONCEPTS, "a", encoding="utf-8") as f:
        for cpt in new_concepts:
            f.write(cpt + "\n")

    print(f"[factory] wrote {len(ids)} fresh specs to {SPECS_DIR} (batch {batch}); "
          f"{len(universe)} fresh concepts available, {len(done)} done.", file=sys.stderr)
    print(json.dumps(ids))   # stdout = ids JSON for the workflow `args`


def _vein_of(block):
    for v in CATALOG:
        if re.search(rf"ff[a-z0-9]*_{re.escape(v)}_", block):
            return v
    m = re.search(r"ff[a-z0-9]+_([a-z]+)_", block)   # best-effort label for research veins
    return m.group(1) if m else ""


def cmd_gate(args):
    batch = slug(args.batch)
    if not batch:
        sys.exit("gate requires --batch <tag> (the same tag used for specs)")
    target = f"_cand_ff{batch}_*.py"
    print(f"[factory] gating {target} at n={args.n} ...", file=sys.stderr)
    subprocess.run([sys.executable, str(VALIDATOR), "--batch", target, "--n", str(args.n),
                    "--keep_fail", "--max_ms", "150"], cwd=str(ROOT))
    if not BATTERY.exists():
        sys.exit("no battery_results.csv produced (did any files match the glob?)")
    import pandas as pd
    df = pd.read_csv(BATTERY)
    for c in ("ic", "oos_ic", "maxcorr"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce").fillna(0.0)
    df["id"] = df["block"]
    df["vein"] = df["block"].map(_vein_of)
    df["batch"] = batch
    cols = ["id", "vein", "batch", "verdict", "ic", "oos_ic", "maxcorr"]
    new = df[cols]
    if LEDGER.exists():
        old = pd.read_csv(LEDGER)
        comb = pd.concat([old, new], ignore_index=True).drop_duplicates("id", keep="last")
    else:
        comb = new
    comb.to_csv(LEDGER, index=False)

    from collections import Counter
    pas = df[df.verdict == "PASS"].sort_values("oos_ic", key=lambda s: s.abs(), ascending=False)
    nov = df[df.maxcorr < 0.4].sort_values("oos_ic", key=lambda s: s.abs(), ascending=False)
    print("\n==== GATE RESULTS (batch %s) ====" % batch, file=sys.stderr)
    print("verdicts:", dict(Counter(df.verdict)), file=sys.stderr)
    print(f"PASS ({len(pas)}):", file=sys.stderr)
    for _, r in pas.head(20).iterrows():
        print(f"  {r.block:<48} OOS_IC={r.oos_ic:+.4f} maxcorr={r.maxcorr:.2f}", file=sys.stderr)
    print(f"NOVEL maxcorr<0.4 ({len(nov)}):", file=sys.stderr)
    for _, r in nov.head(20).iterrows():
        print(f"  {r.verdict:<5} {r.block:<48} OOS_IC={r.oos_ic:+.4f} maxcorr={r.maxcorr:.2f}", file=sys.stderr)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(f"\n## batch {batch} -- {dict(Counter(df.verdict))}\n\n| block | verdict | OOS_IC | maxcorr |\n|---|---|---:|---:|\n")
        for _, r in df.sort_values("oos_ic", key=lambda s: s.abs(), ascending=False).head(25).iterrows():
            f.write(f"| {r.block} | {r.verdict} | {r.oos_ic:+.4f} | {r.maxcorr:.2f} |\n")
    print(f"[factory] ledger -> {LEDGER} | report -> {REPORT}", file=sys.stderr)

    # If this batch was claimed from the queue, release the in-flight specs to done/.
    man = Q_MANIFEST / f"{batch}.json"
    if man.exists():
        concepts = json.loads(man.read_text(encoding="utf-8")).get("concepts", [])
        moved = 0
        for c in concepts:
            src = Q_CLAIMED / f"{c}.spec"
            if src.exists():
                os.replace(src, Q_DONE / f"{c}.spec"); moved += 1
        print(f"[factory] released {moved} claimed specs -> {Q_DONE}", file=sys.stderr)


def cmd_status(args):
    if not LEDGER.exists():
        print("no ledger yet -- run some batches first.")
        return
    import pandas as pd
    from collections import Counter
    df = pd.read_csv(LEDGER)
    for c in ("ic", "oos_ic", "maxcorr"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce").fillna(0.0)
    print(f"FEATURE-FACTORY SCOREBOARD  ({len(df)} candidates gated)")
    print("verdicts:", dict(Counter(df.verdict)))
    print("\nby vein (count | #PASS):")
    for v in CATALOG:
        sub = df[df.vein == v]
        print(f"  {v:<18} {len(sub):>4} | PASS {int((sub.verdict=='PASS').sum())}")
    print("\nTOP 25 by |OOS_IC|:")
    for _, r in df[df.verdict.isin(["PASS","WEAK"])].sort_values("oos_ic", key=lambda s: s.abs(), ascending=False).head(25).iterrows():
        print(f"  {r.verdict:<5} {r.id:<50} OOS_IC={r.oos_ic:+.4f} maxcorr={r.maxcorr:.2f}")


# --------------------------------------------------------------------------------------
# PRODUCER commands (ideation agent): queue-status / topup / enqueue
# --------------------------------------------------------------------------------------
def cmd_queue_status(args):
    ensure_queue()
    info = {"pending": _pending_count(),
            "claimed": len(list(Q_CLAIMED.glob("*.spec"))),
            "done": len(list(Q_DONE.glob("*.spec"))),
            "catalog_fresh": len(catalog_fresh())}
    info["needs_topup"] = info["pending"] < args.min
    if args.json:
        print(json.dumps(info))
    else:
        print(f"QUEUE  pending={info['pending']}  in-flight={info['claimed']}  done={info['done']}  "
              f"| catalog_fresh={info['catalog_fresh']}  needs_topup(<{args.min})={info['needs_topup']}")


def cmd_topup(args):
    """Fill pending/ from the residual CATALOG (free) up to --target, rotating by coverage."""
    ensure_queue()
    fresh = catalog_fresh()
    cov = vein_coverage()
    rng = random.Random(len(load_done_concepts()) * 7919 + 17)
    fresh.sort(key=lambda t: (cov.get(t[0], 0), rng.random()))
    added = 0
    for vein, cpt, ddef in fresh:
        if _pending_count() >= args.target:
            break
        if publish_spec(vein, cpt.split("__", 1)[1], ddef):
            added += 1
    remaining = len(catalog_fresh())
    out = {"added": added, "pending": _pending_count(), "catalog_fresh": remaining,
           "catalog_exhausted": remaining == 0, "still_below_target": _pending_count() < args.target}
    print(json.dumps(out))


def cmd_enqueue(args):
    """Publish RESEARCH-mode specs: JSON array of {vein, slug, definition}. Dedup is automatic."""
    raw = Path(args.json).read_text(encoding="utf-8") if args.json else sys.stdin.read()
    data = json.loads(raw)
    if isinstance(data, dict):
        data = [data]
    added = skipped = 0
    for item in data:
        ok = publish_spec(slug(item["vein"]), slug(item["slug"]), item["definition"])
        added += ok
        skipped += (not ok)
    print(json.dumps({"added": added, "skipped_dups": skipped, "pending": _pending_count()}))


# --------------------------------------------------------------------------------------
# CONSUMER command (forge / code monkey): claim
# --------------------------------------------------------------------------------------
def cmd_claim(args):
    """Atomically claim up to --n pending specs (FIFO), materialise specs/<id>.txt for the
    codegen workflow, write a manifest, and print the claimed ids JSON (pass to workflow.js)."""
    ensure_queue()
    SPECS_DIR.mkdir(parents=True, exist_ok=True)
    batch = slug(args.batch) or "x"
    pend = sorted(Q_PENDING.glob("*.spec"), key=lambda p: p.stat().st_mtime)   # FIFO
    ids, concepts = [], []
    for p in pend:
        if len(ids) >= args.n:
            break
        concept = p.stem
        vein, sep, sl = concept.partition("__")
        if not sep:
            vein, sl = "", concept
        dest = Q_CLAIMED / p.name
        try:
            os.replace(p, dest)            # atomic claim -> lock (skip if another forge took it)
        except OSError:
            continue
        body = dest.read_text(encoding="utf-8")
        ddef = body.split("DEFINITION:", 1)[-1].strip()
        fileid = f"ff{batch}_{vein}_{sl}"
        (SPECS_DIR / f"{fileid}.txt").write_text(spec_txt(fileid, vein, batch, ddef), encoding="utf-8")
        ids.append(fileid); concepts.append(concept)
    (Q_MANIFEST / f"{batch}.json").write_text(
        json.dumps({"batch": batch, "ids": ids, "concepts": concepts}), encoding="utf-8")
    print(f"[factory] claimed {len(ids)} specs (batch {batch}); {_pending_count()} still pending.",
          file=sys.stderr)
    print(json.dumps(ids))   # stdout = ids JSON for the codegen workflow `args`


def main():
    ap = argparse.ArgumentParser(description="feature-factory engine")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("specs"); s.add_argument("--vein", default="rotate"); s.add_argument("--n", type=int, default=30)
    s.add_argument("--batch", default="x"); s.add_argument("--seed", type=int, default=None); s.set_defaults(fn=cmd_specs)
    g = sub.add_parser("gate"); g.add_argument("--batch", default=""); g.add_argument("--n", type=int, default=120); g.set_defaults(fn=cmd_gate)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("veins").set_defaults(fn=lambda a: print("\n".join(CATALOG)))
    # producer (ideation agent)
    qs = sub.add_parser("queue-status"); qs.add_argument("--min", type=int, default=50); qs.add_argument("--json", action="store_true"); qs.set_defaults(fn=cmd_queue_status)
    tu = sub.add_parser("topup"); tu.add_argument("--target", type=int, default=100); tu.set_defaults(fn=cmd_topup)
    eq = sub.add_parser("enqueue"); eq.add_argument("--json", default="", help="path to JSON array of {vein,slug,definition}; omit to read stdin"); eq.set_defaults(fn=cmd_enqueue)
    # consumer (forge / code monkey)
    cl = sub.add_parser("claim"); cl.add_argument("--n", type=int, default=40); cl.add_argument("--batch", default="x"); cl.set_defaults(fn=cmd_claim)
    args = ap.parse_args(); args.fn(args)


if __name__ == "__main__":
    main()
