"""
__relatedness_map.py  --  STEP 2 of FEATURE_EDA_METHODOLOGY.md: the relatedness map.

WHY
---
The house ships ~166 feature blocks / ~1200+ live columns but has never had a
block-level redundancy map that is (a) per-day cross-sectional (the structure the
MODEL actually experiences), (b) multi-seed stable, and (c) action-oriented (it
NOMINATES a de-collinearized incumbent BASIS = one keeper per codependence cluster).
That basis is the load-bearing INPUT to Step 3 (the conditional tail-screen): you can
only ask "does this candidate add anything the book doesn't already have" once you
have ~35-60 clean, non-rank-deficient incumbent columns to condition on.

This implements the 7-step "relatedness map" pipeline from the methodology doc:
  1. PANEL   -- one topo-correct float32 panel (run_pipeline_timed over a ticker sample,
                ALL blocks). Cached to Data/_eda_review/panel.parquet (Step 3 reuses it).
  2. DEPEND  -- per-day cross-sectionally RANKED |Spearman| dependence, AVERAGED over
                days INCREMENTALLY (never hold all daily matrices in RAM).
  3. CLUSTER -- scipy hierarchical (distance = 1 - |avg dep|, average linkage); k chosen
                by silhouette over a small grid (~20-60).
  4. STABLE  -- >=4 ticker SUBSAMPLES derived from the ONE panel; recluster each; report
                co-association + adjusted-Rand + a single Nogueira 0-1 stability index;
                final clustering = consensus (average-linkage on the co-association matrix).
  5. KEEPER  -- per cluster pick the keeper = highest vol/trend-neutralized tail-lift edge
                (reuse __tail_screen helpers); tiebreak by lowest compute cost. Export the
                cluster keepers = INCUMBENT BASIS -> Data/_eda_review/basis.json.
  6. BREADTH -- (a) MP-denoised effective rank / participation ratio of the avg dep matrix
                (DIRECTIONAL HEADLINE ONLY, per the doc caveat), and (b) the bootstrapped
                stable-cluster-count (the robust effective-breadth answer).
  7. OUTPUT  -- Data/_eda_review/clusters.csv + block_map.csv (family x family seriated
                dependence) + prints the eRank / stable-cluster headline + trial-ledger.

RAM / CPU COURTESY (hard house constraint -- other model/backtest jobs run concurrently):
  thread caps set BEFORE numpy import; the panel is float32; ONE shared panel is built and
  ticker subsamples are DERIVED from it (no fresh full panel per seed); the daily dependence
  matrices are averaged incrementally.

USAGE
-----
  stock_env\\Scripts\\python.exe FeatureTemplates\\__relatedness_map.py            # full: --n 350 seeds 4
  ...\\python.exe FeatureTemplates\\__relatedness_map.py --smoke                    # n~120, 15 blocks, 2 seeds
  ...\\python.exe FeatureTemplates\\__relatedness_map.py --n 350 --seeds 4
  ...\\python.exe FeatureTemplates\\__relatedness_map.py --rebuild_panel            # ignore the cache

KNOBS:  --n tickers(350)  --seeds(4)  --kmin/--kmax silhouette grid(20/60)  --min_names(25)
        --max_cols cap feature count for RAM(0=all)  --rebuild_panel  --no_ledger  --smoke
"""
from __future__ import annotations

import os
# single-core + courteous to any running training job (set BEFORE numpy import) -- copied
# verbatim from __tail_screen.py per the house constraint.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import importlib.util
import json
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PRICE_DIR = ROOT / "Data" / "PriceData"
OUT_DIR = ROOT / "Data" / "_eda_review"
PANEL_CACHE = OUT_DIR / "panel.parquet"
SMOKE_CACHE = OUT_DIR / "panel_smoke.parquet"

OHLCV = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]

# A ~15-block smoke subset. INCLUDES a cross-block-dependent chain (returns ->
# momentum_score, price_differential_ratio -> price_differential_signal_pack) so the
# smoke proves the panel is built topo-correct and dependent columns populate.
SMOKE_BLOCKS = [
    "returns", "momentum_score",                       # cross-block: momentum needs returns
    "price_differental_ratio", "price_differential_signal_pack",  # cross-block chain
    "volatility_indicators", "rsi", "amihud_size_illiquidity",
    "price_structure_metrics", "asym_updown_moves", "anchoring_52w_gh",
    "residual_momentum", "beta_dynamics", "variance_ratio",
    "capital_gains_overhang", "salience_theory_value",
]


# tiny ANSI (mirrors __tail_screen.py)
def _c(t, code):
    return f"\033[{code}m{t}\033[0m"


# ---------------------------------------------------------------------------
# Framework bootstrap (topo-correct panel) + __tail_screen helper reuse
# ---------------------------------------------------------------------------
def _load_framework():
    spec = importlib.util.spec_from_file_location("framework", ROOT / "3__FeatureFramework.py")
    fw = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec.loader.exec_module(fw)
    return fw


def _load_tail_screen():
    sys.path.insert(0, str(HERE))
    from importlib import import_module
    return import_module("__tail_screen")


# ---------------------------------------------------------------------------
# Step 1 -- PANEL (topo-correct, all blocks, float32, cached)
# ---------------------------------------------------------------------------
def build_panel(n, seed, fw, smoke_blocks=None, max_cols=0):
    """Build ONE float32 panel by running run_pipeline_timed per ticker (topo-correct).

    smoke_blocks: if given, EXCLUDE every other block (smoke = a ~15-block subset; we
    pass exclude=all-others so the framework still resolves dependencies among the kept set).
    """
    paths = sorted(PRICE_DIR.glob("*.parquet"))
    rng = random.Random(seed)
    sample = rng.sample(paths, min(n, len(paths)))

    exclude = None
    if smoke_blocks is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            all_blocks = fw.discover_blocks()
        keep = set(smoke_blocks)
        exclude = [b for b in all_blocks if b not in keep]

    frames = []
    feat_cols_seen = None
    t0 = time.perf_counter()
    ok = 0
    for p in sample:
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        if "Date" not in df.columns and df.index.name == "Date":
            df = df.reset_index()
        if "Ticker" not in df.columns:
            df["Ticker"] = p.stem
        df = df.sort_values("Date").reset_index(drop=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out, _timing = fw.run_pipeline_timed(df, exclude=exclude, verbose=False)
        except Exception:
            continue
        # downcast feature cols to float32 (keep OHLCV float64 for ranking precision parity)
        feat = [c for c in out.columns if c not in OHLCV]
        for c in feat:
            if out[c].dtype.kind == "f":
                out[c] = out[c].astype(np.float32)
        frames.append(out)
        feat_cols_seen = feat
        ok += 1

    panel = pd.concat(frames, ignore_index=True)
    panel["Date"] = pd.to_datetime(panel["Date"])
    panel = panel.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    feat_cols = [c for c in panel.columns if c not in OHLCV]
    # drop all-NaN / constant columns up front (they can't cluster and pollute eRank)
    keep_feat = []
    for c in feat_cols:
        col = panel[c].to_numpy()
        fin = np.isfinite(col)
        if fin.sum() < 500:
            continue
        if np.nanstd(col[fin].astype(np.float64)) < 1e-12:
            continue
        keep_feat.append(c)
    if max_cols and len(keep_feat) > max_cols:
        keep_feat = keep_feat[:max_cols]
    panel = panel[OHLCV + keep_feat]
    print(f"  built panel: {ok} tickers, {len(panel):,} rows, {len(keep_feat)} usable features, "
          f"{time.perf_counter() - t0:.1f}s")
    return panel, keep_feat


def family_of(fw, feat_cols):
    """col -> owning block name (last writer wins, matches resolve_order behaviour)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        blocks = fw.discover_blocks()
    fam = {}
    for bname, info in blocks.items():
        for col in info["meta"].get("produces", []):
            fam[col] = bname
    return {c: fam.get(c, "?") for c in feat_cols}


# ---------------------------------------------------------------------------
# Step 2 -- per-day cross-sectional |Spearman| dependence, AVERAGED incrementally
# ---------------------------------------------------------------------------
def avg_dependence(panel, feat_cols, min_names):
    """Average over days of the per-day |Spearman| matrix, built INCREMENTALLY.

    Per day: cross-sectionally rank each feature (pct), demean, and accumulate the
    Pearson-on-ranks numerator/denominators. We never materialize a day's full matrix
    beyond a single (k x k) outer product, and never hold all days' matrices in RAM.
    Returns (abs_corr_matrix kxk, n_days_used).
    """
    F = panel[feat_cols].to_numpy(dtype=np.float32)
    dates = panel["Date"].to_numpy()
    k = len(feat_cols)
    codes, _ = pd.factorize(dates)
    order = np.argsort(codes, kind="stable")
    cs = codes[order]
    bounds = np.flatnonzero(np.diff(cs)) + 1

    acc = np.zeros((k, k), dtype=np.float64)   # sum of per-day |corr|
    cnt = np.zeros((k, k), dtype=np.float64)   # days each pair was jointly valid
    n_days = 0
    for idx in np.split(order, bounds):
        if len(idx) < min_names:
            continue
        block = F[idx]                          # (names x k)
        finite = np.isfinite(block)
        # need >= min_names valid for a feature to participate this day
        col_ok = finite.sum(axis=0) >= min_names
        if col_ok.sum() < 2:
            continue
        # cross-sectional rank per feature (NaNs -> excluded by masking to mean later)
        ranks = np.full(block.shape, np.nan, dtype=np.float64)
        for j in np.flatnonzero(col_ok):
            colj = block[:, j]
            m = np.isfinite(colj)
            r = pd.Series(colj[m]).rank().to_numpy()
            tmp = np.full(len(colj), np.nan)
            tmp[m] = (r - r.mean())
            # normalize to unit norm so the outer product is a correlation directly
            nrm = np.sqrt(np.nansum(tmp * tmp))
            if nrm > 0:
                tmp = tmp / nrm
            ranks[:, j] = tmp
        # pairwise correlation = sum over names of (ranks_i * ranks_j), with NaN->0,
        # but only counting pairs where BOTH features are present for >= min_names names.
        R = np.nan_to_num(ranks, nan=0.0)
        present = np.isfinite(ranks)
        corr = R.T @ R                          # (k x k) per-day correlation on co-present names
        # validity per pair: both columns had enough names
        pair_ok = np.outer(col_ok, col_ok)
        # also require some shared support
        shared = (present.astype(np.float64).T @ present.astype(np.float64)) >= min_names
        good = pair_ok & shared
        acc[good] += np.abs(corr[good])
        cnt[good] += 1.0
        n_days += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        avg = np.where(cnt > 0, acc / cnt, 0.0)
    # symmetrize + unit diagonal
    avg = 0.5 * (avg + avg.T)
    np.fill_diagonal(avg, 1.0)
    avg = np.clip(avg, 0.0, 1.0)
    return avg, n_days


# ---------------------------------------------------------------------------
# Step 3 -- hierarchical clustering, k by silhouette
# ---------------------------------------------------------------------------
def cluster_from_dep(abs_dep, kmin, kmax):
    """Average-linkage hierarchical clustering on distance = 1 - |dep|; k by silhouette."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    from sklearn.metrics import silhouette_score

    k = abs_dep.shape[0]
    dist = 1.0 - abs_dep
    np.fill_diagonal(dist, 0.0)
    dist = 0.5 * (dist + dist.T)
    dist = np.clip(dist, 0.0, 1.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")

    kmax = min(kmax, k - 1)
    kmin = max(2, min(kmin, kmax))
    grid = sorted(set(int(x) for x in np.linspace(kmin, kmax, min(9, kmax - kmin + 1))))
    best_k, best_s, best_labels = grid[0], -2.0, None
    for kk in grid:
        labels = fcluster(Z, t=kk, criterion="maxclust")
        if len(set(labels)) < 2:
            continue
        try:
            s = silhouette_score(dist, labels, metric="precomputed")
        except Exception:
            continue
        if s > best_s:
            best_k, best_s, best_labels = kk, s, labels
    if best_labels is None:
        best_labels = fcluster(Z, t=min(kmin, k - 1), criterion="maxclust")
        best_k = len(set(best_labels))
    return best_labels, best_k, best_s, Z


# ---------------------------------------------------------------------------
# Step 4 -- stability across ticker subsamples (co-association + adj-Rand + Nogueira)
# ---------------------------------------------------------------------------
def subsample_clusterings(panel, feat_cols, seeds, kmin, kmax, min_names):
    """Recluster on >=N ticker subsamples DERIVED from the one panel (no rebuild).

    Returns list of label-arrays (len k) aligned to feat_cols.
    """
    tickers = panel["Ticker"].unique().tolist()
    labelings = []
    for s in seeds:
        rng = random.Random(s)
        # ~70% ticker subsample (bootstrap-style, without replacement)
        m = max(10, int(0.7 * len(tickers)))
        sub = set(rng.sample(tickers, min(m, len(tickers))))
        sp = panel[panel["Ticker"].isin(sub)]
        dep, nd = avg_dependence(sp, feat_cols, min_names)
        labels, kk, sil, _ = cluster_from_dep(dep, kmin, kmax)
        labelings.append(labels)
        print(f"    subsample seed={s}: {len(sub)} tics, {nd} days -> k={kk} silhouette={sil:.3f}")
    return labelings


def coassociation(labelings):
    """Fraction of subsamples in which each pair lands in the same cluster (k x k)."""
    n = len(labelings[0])
    co = np.zeros((n, n), dtype=np.float64)
    for lab in labelings:
        same = (lab[:, None] == lab[None, :]).astype(np.float64)
        co += same
    co /= len(labelings)
    return co


def adjusted_rand_pairs(labelings):
    from sklearn.metrics import adjusted_rand_score
    vals = []
    for i in range(len(labelings)):
        for j in range(i + 1, len(labelings)):
            vals.append(adjusted_rand_score(labelings[i], labelings[j]))
    return vals


def nogueira_stability(labelings):
    """Nogueira 0-1 stability index on the per-feature co-membership indicator.

    We adapt the Nogueira (JMLR 2018) stability statistic to clustering by treating, for
    each feature, the binary vector "is this feature in the same cluster as feature f0?"
    Concretely we compute it on the co-association: stability = 1 - mean(per-pair variance
    of the same-cluster indicator) / (mean p-bar*(1-p-bar)). 1 = identical partitions.
    """
    M = len(labelings)
    if M < 2:
        return np.nan
    co = coassociation(labelings)                     # p-hat per pair
    n = co.shape[0]
    iu = np.triu_indices(n, k=1)
    p = co[iu]                                        # mean same-cluster freq per pair
    # unbiased per-pair variance of a Bernoulli over M draws: M/(M-1) * p(1-p)
    var = (M / (M - 1.0)) * p * (1.0 - p)
    pbar = p.mean()
    denom = pbar * (1.0 - pbar)
    if denom <= 0:
        return 1.0
    return float(1.0 - var.mean() / denom)


def consensus_clustering(co, kmin, kmax):
    """Final clustering = average-linkage on the co-association DISTANCE (1 - co)."""
    return cluster_from_dep(co, kmin, kmax)


# ---------------------------------------------------------------------------
# Step 5 -- per-cluster keeper by neutralized tail-lift edge (reuse __tail_screen)
# ---------------------------------------------------------------------------
def compute_neut_edges(panel, feat_cols, ts, min_names, winner_pct=0.10, q=0.10, folds=5):
    """vol/trend-neutralized tail-lift edge per feature (the __tail_screen 'neut' number)."""
    panel = ts.add_context(panel)
    dates = panel["Date"].to_numpy()
    is_winner = (panel["_fwd_rank"].to_numpy() >= 1.0 - winner_pct)
    valid = np.isfinite(panel["_fwd_rank"].to_numpy()) & (panel["_day_n"].to_numpy() >= min_names)
    uniq = np.array(sorted(pd.unique(dates)))
    edges = np.linspace(0, len(uniq), folds + 1).astype(int)
    date_fold = {}
    for kf in range(folds):
        for d in uniq[edges[kf]:edges[kf + 1]]:
            date_fold[d] = kf
    fold_code = np.array([date_fold.get(d, -1) for d in dates])

    out = {}
    for col in feat_cols:
        if col not in panel.columns:
            out[col] = np.nan
            continue
        r2 = ts._xs_neutralize_multi(panel, col, ts.NEUT_FACTORS)
        nft, nfb, _ = ts.tail_lift_folds(dates, r2, is_winner, valid, fold_code, folds, q, winner_pct)
        et = abs((nft if np.isfinite(nft) else 1.0) - 1.0)
        eb = abs((nfb if np.isfinite(nfb) else 1.0) - 1.0)
        out[col] = max(et, eb)
    return out


def load_costs():
    """Per-feature compute cost (us/fval) from the trial ledger if diagnostics logged it.

    Cheap best-effort: returns {} if unavailable. Used only as a keeper TIEBREAK.
    """
    return {}   # diagnostics logs IC not us/fval into the ledger; tiebreak falls back to name.


# ---------------------------------------------------------------------------
# Step 6 -- effective breadth (MP eRank headline + stable-cluster count)
# ---------------------------------------------------------------------------
def effective_rank_mp(abs_dep, panel_rows, k):
    """MP-denoised effective rank + participation ratio of the avg dependence matrix.

    DIRECTIONAL HEADLINE ONLY (the doc is explicit: MP assumes near-iid entries which
    engineered features violate; T<N geometry makes the count partly ill-posed).
    """
    # treat abs_dep as a correlation-like matrix; eigenspectrum
    try:
        w = np.linalg.eigvalsh(abs_dep)
    except Exception:
        return dict(erank=np.nan, participation=np.nan, mp_lambda_plus=np.nan, n_above_mp=np.nan)
    w = np.clip(w, 0.0, None)
    # Marchenko-Pastur upper edge for q = N_features / T_obs
    T = max(panel_rows, k + 1)
    qratio = k / T
    lam_plus = (1.0 + np.sqrt(qratio)) ** 2
    n_above = int((w > lam_plus).sum())
    # effective rank (Roy-Vetterli): exp(entropy of normalized eigenvalues)
    ws = w[w > 1e-12]
    p = ws / ws.sum()
    erank = float(np.exp(-(p * np.log(p)).sum()))
    # participation ratio
    participation = float((ws.sum() ** 2) / (ws ** 2).sum())
    return dict(erank=erank, participation=participation,
                mp_lambda_plus=float(lam_plus), n_above_mp=n_above)


def bootstrap_stable_clustercount(labelings):
    """Robust effective-breadth: mean +/- std of the cluster count across subsamples."""
    counts = [len(set(l)) for l in labelings]
    return float(np.mean(counts)), float(np.std(counts)), counts


# ---------------------------------------------------------------------------
# Step 7 -- block-level (family x family) seriated dependence map
# ---------------------------------------------------------------------------
def block_map(abs_dep, feat_cols, fam):
    fams = sorted(set(fam[c] for c in feat_cols))
    fidx = {f: i for i, f in enumerate(fams)}
    nf = len(fams)
    acc = np.zeros((nf, nf)); cnt = np.zeros((nf, nf))
    for i, ci in enumerate(feat_cols):
        for j, cj in enumerate(feat_cols):
            a, b = fidx[fam[ci]], fidx[fam[cj]]
            acc[a, b] += abs_dep[i, j]; cnt[a, b] += 1
    with np.errstate(invalid="ignore"):
        bm = np.where(cnt > 0, acc / cnt, 0.0)
    # seriate via hierarchical leaf order on (1 - bm)
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
        d = 1.0 - bm; np.fill_diagonal(d, 0.0); d = 0.5 * (d + d.T); d = np.clip(d, 0, 1)
        order = leaves_list(linkage(squareform(d, checks=False), method="average"))
    except Exception:
        order = np.arange(nf)
    fams_ord = [fams[i] for i in order]
    bm_ord = bm[np.ix_(order, order)]
    return pd.DataFrame(bm_ord, index=fams_ord, columns=fams_ord)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=350)
    ap.add_argument("--seeds", type=int, default=4, help="number of ticker subsamples for stability")
    ap.add_argument("--seed", type=int, default=42, help="base seed for the shared panel")
    ap.add_argument("--kmin", type=int, default=20)
    ap.add_argument("--kmax", type=int, default=60)
    ap.add_argument("--min_names", type=int, default=25)
    ap.add_argument("--max_cols", type=int, default=0, help="cap feature count (RAM); 0=all")
    ap.add_argument("--rebuild_panel", action="store_true")
    ap.add_argument("--no_ledger", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.n = min(args.n, 120)
        args.seeds = min(args.seeds, 2)
        args.kmin, args.kmax = 4, 12

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fw = _load_framework()
    ts = _load_tail_screen()

    print(f"\n  RELATEDNESS MAP (Step 2)  | n={args.n} seeds={args.seeds} "
          f"k=[{args.kmin},{args.kmax}] smoke={args.smoke}")

    # ---- Step 1: panel (cached unless smoke / rebuild) --------------------
    smoke_blocks = SMOKE_BLOCKS if args.smoke else None
    cache_path = SMOKE_CACHE if args.smoke else PANEL_CACHE
    use_cache = cache_path.exists() and not args.rebuild_panel
    if use_cache:
        panel = pd.read_parquet(cache_path)
        panel["Date"] = pd.to_datetime(panel["Date"])
        feat_cols = [c for c in panel.columns if c not in OHLCV]
        print(f"  reusing cached panel: {len(panel):,} rows, {len(feat_cols)} features "
              f"({cache_path})")
    else:
        panel, feat_cols = build_panel(args.n, args.seed, fw, smoke_blocks=smoke_blocks,
                                       max_cols=args.max_cols)
        try:
            panel.to_parquet(cache_path, index=False)
            print(f"  cached panel -> {cache_path}")
        except Exception as exc:
            print(f"  [panel cache off: {exc}]")

    fam = family_of(fw, feat_cols)
    # smoke topo-correctness assertion: a cross-block dependent col must be present+populated
    if args.smoke:
        dep_col = next((c for c in ("momentum_score_63d",
                                     "price_differential_ratio_logscale_zscore_20d")
                        if c in panel.columns), None)
        assert dep_col is not None, "smoke: no cross-block-dependent column present (topo broken)"
        nn = int(np.isfinite(pd.to_numeric(panel[dep_col], errors="coerce")).sum())
        assert nn > 100, f"smoke: dependent col {dep_col} not populated ({nn})"
        print(f"  [smoke] topo-correct OK: dependent col '{dep_col}' populated ({nn} finite)")

    # ---- Step 2: avg per-day dependence -----------------------------------
    t0 = time.perf_counter()
    abs_dep, n_days = avg_dependence(panel, feat_cols, args.min_names)
    print(f"  avg dependence matrix: {abs_dep.shape[0]}x{abs_dep.shape[1]} over {n_days} days "
          f"({time.perf_counter() - t0:.1f}s)")

    # ---- Step 3: cluster k by silhouette ----------------------------------
    labels, k_sel, sil, _Z = cluster_from_dep(abs_dep, args.kmin, args.kmax)
    print(f"  primary clustering: k={k_sel} silhouette={sil:.3f}")

    # ---- Step 4: stability across subsamples ------------------------------
    print(f"  stability over {args.seeds} ticker subsamples:")
    sub_seeds = [args.seed + 100 * (i + 1) for i in range(args.seeds)]
    labelings = subsample_clusterings(panel, feat_cols, sub_seeds, args.kmin, args.kmax, args.min_names)
    co = coassociation(labelings)
    ari = adjusted_rand_pairs(labelings)
    nog = nogueira_stability(labelings)
    cons_labels, cons_k, cons_sil, _ = consensus_clustering(co, args.kmin, args.kmax)
    print(f"    adjusted-Rand pairs: mean={np.mean(ari):.3f} min={np.min(ari):.3f}  "
          f"Nogueira stability={nog:.3f}")
    print(f"    CONSENSUS clustering (co-association): k={cons_k} silhouette={cons_sil:.3f}")

    # use the consensus labels as the FINAL clustering
    final_labels = cons_labels
    per_feat_stability = co.mean(axis=1)   # mean co-membership = per-feature stability proxy

    # ---- Step 5: keepers + basis -----------------------------------------
    edges = compute_neut_edges(panel.copy(), feat_cols, ts, args.min_names)
    costs = load_costs()
    cl_members = {}
    for i, c in enumerate(feat_cols):
        cl_members.setdefault(int(final_labels[i]), []).append(c)
    keepers = {}
    for cid, members in cl_members.items():
        # keeper = highest neut edge; tiebreak lowest cost; then name
        best = max(members, key=lambda c: (edges.get(c, -1) if np.isfinite(edges.get(c, np.nan)) else -1,
                                           -costs.get(c, 0.0), c))
        keepers[cid] = best
    basis = sorted(keepers.values())
    with (OUT_DIR / "basis.json").open("w") as fh:
        json.dump(basis, fh, indent=1)
    print(f"  incumbent BASIS = {len(basis)} cluster keepers -> {OUT_DIR / 'basis.json'}")

    # max dependence of each feature to the basis (excluding itself)
    bidx = [feat_cols.index(c) for c in basis if c in feat_cols]
    max_dep_to_basis = {}
    for i, c in enumerate(feat_cols):
        deps = [abs_dep[i, j] for j in bidx if j != i]
        max_dep_to_basis[c] = float(max(deps)) if deps else 0.0

    # ---- Step 6: effective breadth ---------------------------------------
    mp = effective_rank_mp(abs_dep, len(panel), len(feat_cols))
    sc_mean, sc_std, sc_counts = bootstrap_stable_clustercount(labelings)

    # ---- Step 7: outputs --------------------------------------------------
    rows = []
    for i, c in enumerate(feat_cols):
        cid = int(final_labels[i])
        rows.append(dict(col=c, family=fam.get(c, "?"), cluster_id=cid,
                         cluster_size=len(cl_members[cid]),
                         is_keeper=bool(keepers.get(cid) == c),
                         neut_edge=float(edges.get(c, np.nan)),
                         max_dep_to_basis=max_dep_to_basis.get(c, np.nan),
                         stability=float(per_feat_stability[i])))
    cdf = pd.DataFrame(rows).sort_values(["cluster_id", "is_keeper", "neut_edge"],
                                         ascending=[True, False, False])
    cdf.to_csv(OUT_DIR / "clusters.csv", index=False)

    bm = block_map(abs_dep, feat_cols, fam)
    bm.to_csv(OUT_DIR / "block_map.csv")

    print("\n  " + "=" * 70)
    print(f"  EFFECTIVE BREADTH")
    print("  " + "-" * 70)
    print(f"    {_c('eRank (MP, DIRECTIONAL HEADLINE only)', '1;33')}: "
          f"erank={mp['erank']:.1f}  participation={mp['participation']:.1f}  "
          f"eig>MP_edge={mp['n_above_mp']} (lambda+={mp['mp_lambda_plus']:.2f})")
    print(f"    {_c('stable-cluster count (ROBUST answer)', '1;32')}: "
          f"{sc_mean:.1f} +/- {sc_std:.1f}  (per-subsample {sc_counts}); consensus k={cons_k}")
    print(f"    Nogueira stability index: {nog:.3f}   adj-Rand mean: {np.mean(ari):.3f}")
    print("  " + "=" * 70)
    print(f"  outputs: clusters.csv ({len(cdf)} rows)  basis.json ({len(basis)})  "
          f"block_map.csv ({bm.shape[0]}x{bm.shape[1]})  -> {OUT_DIR}")

    # ---- trial ledger -----------------------------------------------------
    if not args.no_ledger:
        try:
            from importlib import import_module
            sys.path.insert(0, str(HERE))
            tl = import_module("__trial_ledger")
            led = tl.TrialLedger(tool="relatedness_map", seed=args.seed,
                                 params={"n": args.n, "seeds": args.seeds,
                                         "kmin": args.kmin, "kmax": args.kmax,
                                         "min_names": args.min_names, "smoke": args.smoke,
                                         "n_features": len(feat_cols)})
            led.add_metrics("_BREADTH", {"erank_mp": mp["erank"], "participation": mp["participation"],
                                         "stable_cluster_mean": sc_mean, "stable_cluster_std": sc_std,
                                         "consensus_k": cons_k, "nogueira_stability": nog,
                                         "adj_rand_mean": float(np.mean(ari))},
                            family="CONTROL", n_obs=n_days)
            for i, c in enumerate(feat_cols):
                led.add_metrics(c, {"neut_edge": edges.get(c, np.nan),
                                    "max_dep_to_basis": max_dep_to_basis.get(c, np.nan),
                                    "cluster_stability": float(per_feat_stability[i])},
                                family=fam.get(c, "?"), n_obs=n_days,
                                cluster_id=int(final_labels[i]),
                                is_keeper=bool(keepers.get(int(final_labels[i])) == c))
            p = led.flush()
            print(f"  [ledger] {len(feat_cols)} features + breadth logged -> {p}")
        except Exception as exc:
            print(f"  [ledger off: {exc}]")
    print()


if __name__ == "__main__":
    main()
