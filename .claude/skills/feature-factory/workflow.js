export const meta = {
  name: 'feature_factory_codegen',
  description: 'Fan out subagents to codegen quant feature blocks from feature-factory spec files, matching the FeatureTemplates contract',
  phases: [{ title: 'Codegen', detail: 'one subagent per spec writes a _cand_<id>.py candidate block' }],
}

const ROOT = 'c:/Users/Masam/Desktop/Stock-Market'
const SPECS = ROOT + '/.claude/skills/feature-factory/specs'

const FIELDS = 'asset_turnover, assets, assets_current, book_value_per_share, capex, capex_ttm, cash, cost_of_revenue, cost_of_revenue_ttm, current_ratio, debt_to_equity, dividends_paid, dividends_paid_ttm, eps_basic, eps_basic_ttm, eps_diluted, eps_diluted_ttm, equity, fcf_ttm, goodwill, gross_margin, gross_profit, gross_profit_ttm, interest_expense, interest_expense_ttm, inventory, liabilities, liabilities_current, long_term_debt, net_income, net_income_ttm, net_margin, operating_cash_flow, operating_cash_flow_ttm, operating_income, operating_income_ttm, operating_margin, ppe_net, receivables, revenue, revenue_ttm, rnd_expense, rnd_expense_ttm, roa, roe, sales_per_share, shares_outstanding, total_debt'

const CONTRACT = [
'You are generating EXACTLY ONE Python feature-block file for a modular quant feature pipeline.',
'The file MUST define exactly two module-level objects:',
'  METADATA = {"name","description","requires","produces","tags","version","author"}',
'  def compute(df: pd.DataFrame) -> pd.DataFrame',
'',
'HOW compute() IS CALLED:',
'  - df is ONE stock at a time, ascending by Date, with columns: Date, Ticker, Open, High, Low, Close, Volume.',
'  - You may ONLY ADD the columns listed in METADATA["produces"]. Never modify/drop/rename/sort/reindex existing columns. Always `return df`.',
'',
'HARD RULES (a sandbox gate enforces these; violations are auto-rejected):',
'  - Imports allowed ONLY from: pandas, numpy, scipy, networkx, math, typing, importlib, pathlib, warnings, collections, functools, itertools, dataclasses, enum, __future__. NO sklearn/statsmodels/ta/pandas_ta/requests. No file or network IO (except the two blessed by-path helper imports below).',
'  - NO lookahead/leakage: use only current-and-past data. Rolling windows are fine. NEVER use a negative shift df[col].shift(-k), iloc[i+1], negative diff, or any full-series statistic that peeks ahead. The gate truncates the series at several cut points and requires past value[t] to be IDENTICAL with or without future bars.',
'  - CAUSAL STRIDING: if a metric is too slow to compute every bar, compute it only on a FIXED-FROM-START grid (bars where i % stride == 0 measured from the series start) and forward-fill -- NEVER anchor the grid to the last bar (it re-anchors under truncation and FAILS the causality test).',
'  - Stateless & deterministic: no globals mutated, no prints, no logging, no training, no randomness. Fast: < 100ms on ~700 rows. Vectorise with pandas/numpy; avoid O(n^2) python loops over every row (a rolling op or sliding_window_view is fine for small windows).',
'  - Column names: lowercase snake_case, no leading digit, no %, no spaces/parens. Guard ALL divisions against divide-by-zero (replace 0 denominators with np.nan); never emit inf/-inf. Leading NaNs from rolling windows are EXPECTED -- do not fill the whole frame.',
'  - METADATA["requires"] MUST be a subset of {Open, High, Low, Close, Volume} (or empty). Do NOT depend on columns produced by other blocks -- compute everything you need internally from OHLCV. This is required so the block can be validated standalone.',
'  - UNIQUENESS: prefix EVERY produced column with the spec id (given below) so it cannot collide with the ~190 existing features, e.g. produces = ["<SPECID>_main", "<SPECID>_slope"]. Keep names readable.',
'  - STRUCTURAL: every column you list in METADATA["produces"] MUST be created on EVERY code path (incl. empty/no-coverage/early-return branches -- initialise them to np.nan up front). A produces/compute mismatch hard-fails the gate.',
'',
'OPTIONAL HELPER 1 -- market index / VIX (import BY FILE PATH, never `from FeatureTemplates import`):',
'  import importlib.util as _ilu; from pathlib import Path as _P',
'  _s=_ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent/"_indexes.py"); _indexes=_ilu.module_from_spec(_s); _s.loader.exec_module(_indexes)',
'  API: _indexes.index_close("SPY") -> pd.Series Close indexed by DatetimeIndex Date (symbols SPY,QQQ,IWM,DIA,VIX).',
'       _indexes.vix_daily_close() -> DataFrame[["Date","vix_close"]] ready for pd.merge_asof(df.sort_values("Date"), vix_daily_close(), on="Date", direction="backward").',
'  When you join index/VIX, merge with merge_asof direction="backward" on Date (lookahead-safe); degrade to NaN if the symbol is missing.',
'',
'OPTIONAL HELPER 2 -- point-in-time SEC fundamentals (import BY FILE PATH):',
'  _s2=_ilu.spec_from_file_location("_fundamentals", _P(__file__).resolve().parent/"_fundamentals.py"); _fundamentals=_ilu.module_from_spec(_s2); _s2.loader.exec_module(_fundamentals)',
'  Inside compute(): df = _fundamentals.as_of(df, fields=["net_margin","assets", ...])  # returns columns named fund_<field> as numpy float64',
'  as_of() does a BACKWARD merge_asof on filed_date (the date the number became public) -> lookahead-safe. NEVER reference period_end/period_start and NEVER forward-merge or read the panel yourself -- the gate hard-fails that.',
'  Available fields ('+FIELDS+'). TTM rollups end in _ttm. Price multiples (P/E,P/B,P/S) you compute yourself: Close / matching fund per-share field. fund_* columns are SCRATCH -- compute your produced feature(s), then drop any fund_* you are NOT listing in produces. Coverage ~84% (ETFs/foreign have none) -> leading & missing rows NaN; that is expected, do not fill.',
'',
'FEASIBILITY: If the method is inherently cross-sectional (ranks across many stocks), needs training, or needs data beyond OHLCV+index+PIT-fundamentals, implement the closest faithful PER-TICKER proxy that captures the same economic signal, and say so honestly in METADATA["description"] and ["author"].',
].join('\n')

phase('Codegen')
const ids = typeof args === 'string' ? JSON.parse(args) : args
log('Codegen: ' + ids.length + ' specs -> _cand_<id>.py candidate blocks')

const SCHEMA = {
  type: 'object',
  properties: {
    id: { type: 'string' },
    written: { type: 'boolean' },
    produces: { type: 'array', items: { type: 'string' } },
    uses_fundamentals: { type: 'boolean' },
    uses_index: { type: 'boolean' },
    note: { type: 'string' },
  },
  required: ['id', 'written', 'produces', 'note'],
  additionalProperties: false,
}

function buildPrompt(id) {
  return [
    CONTRACT, '',
    'YOUR TASK:',
    '1. Read the spec file: ' + SPECS + '/' + id + '.txt',
    '2. Implement the closest faithful, leakage-free, vectorised feature block for that method.',
    '   - The spec id is: ' + id + '  (use it as the column prefix and METADATA["name"] = "' + id + '").',
    '   - Produce 1-3 related columns (a level + a dynamic/slope/asymmetry variant is often good).',
    '3. Write the file with the Write tool to EXACTLY: ' + ROOT + '/FeatureTemplates/_cand_' + id + '.py',
    '4. Do NOT run anything or validate. Just write ONE correct file. ALWAYS write a file (best honest proxy if hard).',
    'Then return the manifest object.',
  ].join('\n')
}

const results = await parallel(ids.map(id => () =>
  agent(buildPrompt(id), { label: id, phase: 'Codegen', schema: SCHEMA,
    model: 'sonnet', effort: 'medium', agentType: 'general-purpose' })))

const written = results.filter(Boolean).filter(r => r.written)
const failed = ids.filter((id, i) => !results[i] || !results[i].written)
log('Codegen done: ' + written.length + '/' + ids.length + ' written ('
  + written.filter(r => r.uses_fundamentals).length + ' fundamentals, '
  + written.filter(r => r.uses_index).length + ' index/VIX)')
return { requested: ids.length, written: written.length, failed_ids: failed,
  notes: written.map(r => ({ id: r.id, produces: r.produces, note: r.note })) }
