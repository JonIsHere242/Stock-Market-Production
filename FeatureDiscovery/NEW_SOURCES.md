# New information sources for the feature-discovery pipeline

Beyond the "obvious slop" (the arXiv q-fin / generic-ML category dump). Ranked by expected
yield of *OHLCV- or PIT-fundamentals-computable* feature ideas that are **not already mined**.
Implemented items are wired into the fetchers; the rest are documented with access notes so
they can be added when worth a key.

## Tier 1 — turnkey, cited, implementation-grade (WIRED as idea banks)

1. **Open-Source Asset Pricing (Chen & Zimmermann)** — `idea_banks/osap_implementable.jsonl`
   `SignalDoc.csv` documents **331 academic cross-sectional predictors** with exact
   definitions, signs, citations, and a `Cat.Data` tag. **169 are implementable from our data**
   (Accounting 99, Price 45, Trading 13, Other 12). This is the densest, least-overfit vein:
   peer-reviewed, replication-grade, formula-complete. The accounting predictors map directly
   onto the new `_fundamentals.as_of` PIT helper — the project's documented white-space.
   Keyless: `raw.githubusercontent.com/OpenSourceAP/CrossSection/master/SignalDoc.csv`.
2. **JKP Global Factor Data (Jensen, Kelly, Pedersen 2023)** — ~150 cluster-organised factors
   with open SAS/Python construction code (github.com/bkelly-lab/ReplicationCrisis). Overlaps
   OSAP but adds the *cluster* taxonomy (quality, value, low-risk, momentum, ...) useful for
   orthogonality. Not yet pulled; the construction code is the spec. **Add next.**

## Tier 2 — peer-voted / practitioner method corpora (WIRED as fetchers)

3. **Quantitative Finance Stack Exchange** — `altsources/stackexchange.py` (NEW, keyless).
   Top-*voted* Q&A = crowd-ranked implementability; bodies usually contain the exact transform.
   Swept by method tag (time-series, volatility, factor-models, signal-processing, ...).
4. **Institutional / practitioner research RSS** — extended `altsources/rss.py` DEFAULT_FEEDS.
   Added AlphaScientist, Quantitativo, Newfound/ReSolve, FactorResearch, Quantpedia, Two Sigma,
   Concretum, Raposa, MlFinLab. These publish *replicable* signal constructions the academic
   sources miss; denser and less picked-over than arXiv.

## Tier 3 — high value, needs a key or heavier ingestion (DOCUMENTED, not yet wired)

5. **Kaggle finance competitions + notebooks** (Jane Street, Optiver, Two Sigma, G-Research,
   JPX) — winning notebooks are feature goldmines. Needs Kaggle API token; ingest via a new
   `altsources/kaggle.py` (kernels list → notebook source text → extract.py).
6. **SSRN / FEN (Financial Economics Network)** — the largest finance working-paper repository.
   No clean API and hostile to scraping, BUT its abstracts are indexed in OpenAlex / Semantic
   Scholar / Crossref (already wired). Best reached *indirectly* via those, filtered by SSRN DOI.
7. **Patents (trading & signal-processing methods)** — USPTO PatentsView (key now required) or
   Google Patents (no official API). A wholly unmined structural vein (TA-indicator and DSP
   patents). Add `altsources/patents.py` when a key is provisioned; until then, patent-derived
   methods are seeded into the cross-domain bank.
8. **Central-bank / institutional research** — BIS, Fed (FEDS Notes / FRBSF Letter), ECB, IMF.
   Aggregated by **RePEc / IDEAS** (OAI-PMH) and **EconStor (ZBW)** (`econstor.eu/oai`, keyless,
   confirmed reachable). EconBiz/NBER already cover part; a `sources/econstor.py` OAI adapter
   would widen the macro/quality-factor net. **Add next.**

## Tier 4 — cross-domain method transfer (WIRED as idea bank)

9. **`idea_banks/crossdomain_methods.jsonl`** — 32 curated time-series methods from *other
   fields*, deliberately avoiding families already mined (wavelet, Hurst, permutation-entropy,
   visibility-graph, matrix-power):
   - **Biomedical / HRV:** Hjorth parameters, Poincaré SD1/SD2, sample/approximate entropy.
   - **Nonlinear dynamics / econophysics:** Higuchi & Katz & Petrosian fractal dimension,
     Lempel-Ziv complexity, DFA crossover, DCCA vs index, Rényi entropy, ordinal-transition entropy.
   - **DSP / audio / vibration:** spectral entropy/centroid/bandwidth, Teager-Kaiser energy,
     crest/impulse factors, Hilbert instantaneous frequency, Allan deviation.
   - **Information theory / change detection:** transfer entropy (VIX→stock), volume↔vol mutual
     information, CUSUM change-points, Wasserstein & KS distributional drift, SSA leading-energy.

## Dead ends (intentionally excluded)
News/sentiment/Twitter/Reddit-text, limit-order-book/tick microstructure, full options surface
& greeks, satellite/credit-card alt-data, on-chain/crypto — all need data a per-ticker OHLCV
`compute(df)` (+ index/VIX + PIT fundamentals) cannot see, so the relevance gate hard-caps them.

---
*Generated 2026-06-27. The fetchers feed `Data/PaperFeed/papers.parquet`; the idea banks feed
codegen directly. See `generate_idea_blocks` workflow for the parallel paper→block codegen.*
