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
const ids = ["ff0703g_ohlc_range_meilijson_mvu", "ff0703g_ohlc_range_gk_meilijson_blend", "ff0703g_ohlc_range_kunitomo_bridge", "ff0703g_ohlc_range_bali_weinbaum_ev", "ff0703g_ohlc_range_corwin_schultz_spread", "ff0703g_ohlc_range_abdi_ranaldi_spread", "ff0703g_ohlc_range_rs_updown_asymmetry", "ff0703g_ohlc_range_excursion_energy_ratio", "ff0703g_ohlc_range_meilijson_c2c_efficiency", "ff0703g_ohlc_range_clv_dispersion", "ff0703g_ohlc_range_range_subadditivity", "ff0703g_ohlc_range_estimator_disagreement", "ff0703g_ohlc_range_carr_ar1_phi", "ff0703g_ohlc_range_range_mrev_halflife", "ff0703g_ohlc_range_range_acf_1to5", "ff0703g_ohlc_range_range_surprise_ewma_z", "ff0703g_ohlc_range_vol_of_range_norm", "ff0703g_ohlc_range_range_surprise_autocorr", "ff0703g_ohlc_range_range_expansion_streak_asym", "ff0703g_ohlc_range_range_quantile_position", "ff0703g_ohlc_range_carr_leverage_asym", "ff0703g_ohlc_range_range_of_range_accel", "ff0703g_ohlc_range_truerange_carr_surprise", "ff0703g_ohlc_range_log_range_momentum_ratio", "ff0703g_ohlc_range_range_realized_quarticity", "ff0703g_ohlc_range_logrange_excess_kurtosis_60", "ff0703g_ohlc_range_intvar_to_ccvar_ratio", "ff0703g_ohlc_range_range_hill_tail_index", "ff0703g_ohlc_range_quarticity_noise_ratio", "ff0703g_ohlc_range_range_hyperflatness_6th", "ff0703g_ohlc_range_quarticity_term_structure", "ff0703g_ohlc_range_jump_variation_share", "ff0703g_ohlc_range_quarticity_up_down_asym", "ff0703g_ohlc_range_raw_range_dist_kurtosis", "ff0703g_ohlc_range_quarticity_aggregation_scaling", "ff0703g_ohlc_range_quartic_range_shock", "ff0703g_intraday_moment_gk_realized_skew", "ff0703g_intraday_moment_close_mid_skew_moment", "ff0703g_intraday_moment_leg_power_asymmetry", "ff0703g_intraday_moment_overnight_intraday_coskew", "ff0703g_intraday_moment_bigrange_jump_direction", "ff0703g_intraday_moment_path_triplet_skew", "ff0703g_intraday_moment_neg_skew_day_fraction", "ff0703h_intraday_moment_range_pos_third_moment", "ff0703h_intraday_moment_gap_conditioned_intraday_skew", "ff0703h_intraday_moment_amaya_intraday_realized_skew", "ff0703h_intraday_moment_downside_excursion_asymmetry", "ff0703h_intraday_moment_close_skew_trend_accel", "ff0703h_intraday_moment_jump_var_share", "ff0703h_intraday_moment_signed_jump_var", "ff0703h_intraday_moment_jump_intensity", "ff0703h_intraday_moment_bipower_jump_frac", "ff0703h_intraday_moment_jump_diffusion_leverage", "ff0703h_intraday_moment_jump_diffusion_corr", "ff0703h_intraday_moment_overnight_qv_dominance", "ff0703h_intraday_moment_jump_skew_ratio", "ff0703h_intraday_moment_jump_clustering_ratio", "ff0703h_intraday_moment_truncated_diffusion_ratio", "ff0703h_intraday_moment_jump_energy_concentration", "ff0703h_intraday_moment_jvshare_instability", "ff0703h_intraday_moment_rs_semivar_ratio", "ff0703h_intraday_moment_rs_down_share", "ff0703h_intraday_moment_signed_close_semivar_ratio", "ff0703h_intraday_moment_semivar_overnight_intraday_split", "ff0703h_intraday_moment_rs_updown_flip_rate", "ff0703h_intraday_moment_rs_semivar_skew", "ff0703h_intraday_moment_downside_semivar_concentration", "ff0703h_intraday_moment_semivar_vol_weighted_ratio", "ff0703h_intraday_moment_gap_conditional_intraday_semivar", "ff0703h_intraday_moment_rs_updown_persistence", "ff0703h_intraday_moment_semivar_ratio_term_structure", "ff0703h_intraday_moment_signed_overnight_semivar", "ff0703h_tugofwar_spread_tugofwar", "ff0703h_tugofwar_cum_overnight_mom", "ff0703h_tugofwar_cum_intraday_mom", "ff0703h_tugofwar_onid_corr", "ff0703h_tugofwar_id_on_reversal_beta", "ff0703h_tugofwar_overnight_move_share", "ff0703h_tugofwar_spread_dominance_freq", "ff0703h_tugofwar_spread_z", "ff0703h_tugofwar_tug_streak", "ff0703h_tugofwar_gap_fade_conditional", "ff0703h_tugofwar_component_sharpe_spread", "ff0703h_tugofwar_spread_acceleration", "ff0703h_tugofwar_disagree_freq", "ff0703h_tugofwar_disagree_magnitude", "ff0703i_tugofwar_net_tug", "ff0703i_tugofwar_tension_energy", "ff0703i_tugofwar_cancellation_ratio", "ff0703i_tugofwar_session_corr", "ff0703i_tugofwar_offset_beta", "ff0703i_tugofwar_win_asymmetry", "ff0703i_tugofwar_gap_fight_asymmetry", "ff0703i_tugofwar_disagree_persistence", "ff0703i_tugofwar_net_tug_asymmetry", "ff0703i_tugofwar_acute_tension", "ff0703i_tugofwar_gap_fill_rate", "ff0703i_tugofwar_gap_fill_depth", "ff0703i_tugofwar_on_id_reversal_beta", "ff0703i_tugofwar_gap_continuation_corr", "ff0703i_tugofwar_cross_session_handoff_beta", "ff0703i_tugofwar_gap_overshoot_rate", "ff0703i_tugofwar_reversal_day_frequency", "ff0703i_tugofwar_gap_sign_runlength", "ff0703i_tugofwar_reversal_magnitude_ratio", "ff0703i_tugofwar_unfilled_gap_drift", "ff0703i_tugofwar_gap_close_location", "ff0703i_tugofwar_twoday_gap_meanrev_corr", "ff0703i_tugofwar_fill_conditional_reversal_beta", "ff0703i_ohlc_pos_strength_persistence", "ff0703i_ohlc_pos_strong_close_run", "ff0703i_ohlc_pos_weak_strong_run_asymmetry", "ff0703i_ohlc_pos_conditional_strength_by_direction", "ff0703i_ohlc_pos_strength_dispersion", "ff0703i_ohlc_pos_strength_range_weighting", "ff0703i_ohlc_pos_strength_sign_flips", "ff0703i_ohlc_pos_strength_gap_followthrough", "ff0703i_ohlc_pos_rejection_at_breakouts", "ff0703i_ohlc_pos_support_at_breakdowns", "ff0703i_ohlc_pos_strength_return_divergence", "ff0703i_ohlc_pos_strength_ewma_acceleration", "ff0703i_ohlc_pos_pivot_position_level", "ff0703i_ohlc_pos_above_pivot_freq_40", "ff0703i_ohlc_pos_close_tp_range_asym", "ff0703i_ohlc_pos_weighted_close_bias", "ff0703i_ohlc_pos_pivot_band_position", "ff0703i_ohlc_pos_pivot_breakout_asym", "ff0703i_ohlc_pos_open_pivot_gap", "ff0703i_ohlc_pos_pivot_straddle_freq", "ff0703j_ohlc_pos_wc_tp_open_pull", "ff0703j_ohlc_pos_pivot_side_streak", "ff0703j_ohlc_pos_second_pivot_reach_asym", "ff0703j_ohlc_pos_pivot_pin_distance", "ff0703j_ohlc_pos_pos_velocity_accel", "ff0703j_ohlc_pos_extreme_dwell_asym", "ff0703j_ohlc_pos_pos_ar1_reversion", "ff0703j_ohlc_pos_pos_detrend_z", "ff0703j_ohlc_pos_dwell_run_length", "ff0703j_ohlc_pos_midline_cross_rate", "ff0703j_ohlc_pos_nested_channel_gap", "ff0703j_ohlc_pos_pos_drift_asym", "ff0703j_ohlc_pos_stoch_kd_gap", "ff0703j_ohlc_pos_failed_breakout_count", "ff0703j_ohlc_pos_time_since_extreme", "ff0703j_ohlc_pos_pos_volume_thrust", "ff0703j_ohlc_shape_inside_bar_compression_freq", "ff0703j_ohlc_shape_outside_bar_expansion_asym", "ff0703j_ohlc_shape_nr7_narrowest_range_freq", "ff0703j_ohlc_shape_wr7_widest_range_freq", "ff0703j_ohlc_shape_trend_bar_conviction_freq", "ff0703j_ohlc_shape_trend_bar_directional_bias", "ff0703j_ohlc_shape_doji_indecision_freq", "ff0703j_ohlc_shape_true_range_gap_share", "ff0703j_ohlc_shape_gap_share_direction_asym", "ff0703j_ohlc_shape_gap_dominant_bar_freq", "ff0703j_ohlc_shape_compression_expansion_balance", "ff0703j_ohlc_shape_compression_persistence", "ff0703j_ohlc_shape_range_pctile_squeeze", "ff0703j_ohlc_shape_contract_run_length", "ff0703j_ohlc_shape_squeeze_min20_state", "ff0703j_ohlc_shape_expansion_direction_bias", "ff0703j_ohlc_shape_coil_compression_ratio", "ff0703j_ohlc_shape_range_bandwidth_tightness", "ff0703j_ohlc_shape_contraction_magnitude", "ff0703j_ohlc_shape_squeeze_release_gap_dir", "ff0703j_ohlc_shape_directional_release_runlen", "ff0703j_ohlc_shape_asymmetric_expansion_range", "ff0703j_ohlc_shape_coil_then_expand_flag", "ff0703j_ohlc_shape_range_compression_zscore", "ff0703j_ohlc_shape_expansion_energy_skew"]
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
