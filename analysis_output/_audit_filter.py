import pandas as pd
import numpy as np
import re
import glob as globmod

# ============================================================
# STEP 1 - Load finviz and parse numeric columns
# ============================================================
fv = pd.read_parquet('Data/FundamentalData/finviz_cache_20251214.parquet')

def parse_finviz_num(val, is_pct=False):
    """Parse finviz string values to float."""
    if pd.isna(val) or val in ['-', '', 'N/A']:
        return np.nan
    s = str(val).strip().replace(',', '')
    neg = s.startswith('(') and s.endswith(')')
    if neg:
        s = '-' + s[1:-1]
    if s.endswith('%'):
        v = float(s[:-1])
        return v / 100 if is_pct else v
    for suffix, mult in [('B', 1e9), ('M', 1e6), ('K', 1e3)]:
        if s.endswith(suffix):
            return float(s[:-1]) * mult
    try:
        return float(s)
    except:
        return np.nan

fv['mktcap']        = fv['Market Cap'].apply(parse_finviz_num)
fv['avg_vol']       = fv['Avg Volume'].apply(parse_finviz_num)
fv['price']         = fv['Price'].apply(parse_finviz_num)
fv['inst_own']      = fv['Inst Own'].apply(lambda x: parse_finviz_num(x, is_pct=True))
fv['float_shares']  = fv['Shs Float'].apply(parse_finviz_num)
fv['insider_own']   = fv['Insider Own'].apply(lambda x: parse_finviz_num(x, is_pct=True))
fv['short_float']   = fv['Short Float'].apply(lambda x: parse_finviz_num(x, is_pct=True))
fv['beta']          = fv['Beta'].apply(parse_finviz_num)
fv['employees']     = fv['Employees'].apply(parse_finviz_num)
fv['profit_margin'] = fv['Profit Margin'].apply(lambda x: parse_finviz_num(x, is_pct=True))
fv['debt_eq']       = fv['Debt/Eq'].apply(parse_finviz_num)
fv['roe']           = fv['ROE'].apply(lambda x: parse_finviz_num(x, is_pct=True))
fv['country']       = fv['Country']
fv['sector']        = fv['Sector']
fv['industry']      = fv['Industry']
fv['ticker']        = fv['Symbol']

print("Finviz loaded:", fv.shape)

# ============================================================
# STEP 2 - Load ticker filter (most recent non-raw parquet)
# ============================================================
files = sorted(globmod.glob('Data/TickerCikData/*.parquet'))
filtered_files = [f for f in files if '_raw' not in f]
print(f"\nAll parquet files found: {files}")
print(f"Non-raw files: {filtered_files}")

latest = filtered_files[-1]
print(f"Loading: {latest}")
tickers_df = pd.read_parquet(latest)
print("Columns:", tickers_df.columns.tolist())
print(tickers_df.head(5).to_string())
print(f"Total filtered tickers: {len(tickers_df)}")

# ============================================================
# STEP 3 - What passes name filter but is fundamentally garbage?
# ============================================================
merged = tickers_df.merge(
    fv[['ticker','mktcap','avg_vol','price','inst_own','float_shares',
        'employees','country','sector','industry','short_float','beta','profit_margin']],
    on='ticker', how='left'
)

print(f"\n=== Stocks passing name filter: {len(tickers_df)} ===")
print(f"Matched in finviz: {merged['mktcap'].notna().sum()}")

# Nano-caps passing filter (market cap < $50M)
nanocap = merged[merged['mktcap'] < 50e6].sort_values('mktcap')
print(f"\nNano-caps (<$50M mktcap) passing filter: {len(nanocap)}")
print(nanocap[['ticker','mktcap','avg_vol','price','sector']].head(20).to_string())

# Very illiquid (avg vol < 50K)
illiquid = merged[merged['avg_vol'] < 50_000].sort_values('avg_vol')
print(f"\nIlliquid (<50K avg daily vol) passing filter: {len(illiquid)}")
print(illiquid[['ticker','avg_vol','mktcap','price','sector']].head(20).to_string())

# Near-zero institutional ownership
low_inst = merged[(merged['inst_own'] < 0.01) & merged['inst_own'].notna()].sort_values('inst_own')
print(f"\nLow institutional ownership (<1%) passing filter: {len(low_inst)}")
print(low_inst[['ticker','inst_own','mktcap','avg_vol','sector']].head(20).to_string())

# Penny stocks
penny = merged[merged['price'] < 1].sort_values('price')
print(f"\nPenny stocks (<$1) passing filter: {len(penny)}")
print(penny[['ticker','price','mktcap','avg_vol','sector']].head(20).to_string())

# Non-US country
foreign = merged[merged['country'].notna() & (merged['country'] != 'USA')].sort_values('country')
print(f"\nNon-US stocks passing filter: {len(foreign)}")
print(foreign[['ticker','country','mktcap','avg_vol','sector']].head(30).to_string())

# Zero employees (shell companies)
zero_emp = merged[(merged['employees'] == 0) | (merged['employees'] < 5)]
print(f"\nZero/tiny employee count passing filter: {len(zero_emp)}")
print(zero_emp[['ticker','employees','mktcap','sector','industry']].head(20).to_string())

# High short interest (>30%) - potential battleground/distressed stocks
high_short = merged[merged['short_float'] > 0.30].sort_values('short_float', ascending=False)
print(f"\nHigh short interest (>30% float) passing filter: {len(high_short)}")
print(high_short[['ticker','short_float','mktcap','avg_vol','sector']].head(20).to_string())

# ============================================================
# STEP 4 - What does the name filter incorrectly REJECT from finviz?
# ============================================================
def is_problematic_ticker(company_name):
    if not isinstance(company_name, str):
        return False
    name = company_name.lower().strip()
    instant_reject_words = [
        'acquisition','spac','etf','fund','reit','warrant','preferred',
        'holdings','vehicle','ventures','income','merger','capital',
        'ai','blockchain','metaverse','crypto','bitcoin','quantum','web3',
        'tokenized','digital asset','grant','subsidy','incentive','rebate',
        'tax credit','stimulus','municipal','public-private','federal funding',
        'sustainability','esg','climate','carbon','renewable credit',
        'green energy','impact investing','net zero','stakeholder','social impact',
        'equity focused','resilience','global initiative','multilateral','foundation',
        'charitable','compliance','regulatory solutions','certification',
        'verification','governance solutions','monitoring platform',
        'recycling solutions','circular economy','waste-to-energy',
        'sustainable materials','plastic alternatives','solutions','platform',
        'ecosystem','network','alliance','machine learning solutions',
        'data-driven','next generation','disruptive technology','innovation platform',
    ]
    for word in instant_reject_words:
        if word in name:
            return True
    problematic_patterns = [
        r'special purpose',r'blank check',r'shell company',r'merger corp',
        r'exchange traded',r'real estate investment',r'mutual fund',r'index fund',
        r'investment trust',r'bond fund',r'growth fund',r'dividend fund',
        r'closed.*end',r'open.*end',r'vanguard',r'ishares',r'spdr',r'invesco',
        r'direxion',r'proshares',r'wisdomtree',r'blackrock',r'fidelity',
        r'capital.*corp\.?\s+(i{2,}|iv|v|vi{1,3}|ix|x|\d)',
        r'(corp|inc|ltd|llc|company|co)\.?\s+(i{2,}|iv|v|vi{1,3}|ix|x)',
        r'(corp|inc|ltd|llc|company|co)\.?\s+[2-9]',
        r'\d+x\s',r'leveraged',r'inverse',r'bear.*etf',r'bull.*etf',
        r'ultra.*short',r'ultra.*long',r'volatility',r'\bvix\b',
        r'\bright[s]?\b',r'\bunit[s]?\b',r'when issued',r'\bstub[s]?\b',
        r'series [a-z]',r'class [a-z]',r'bankruptcy',r'liquidat',r'defunct',
        r'dissolved',r'delisted',r'chapter 11',r'development authority',
        r'economic development',r'infrastructure fund',r'energy transition',
        r'environmental solutions',r'distributed ledger',r'intelligent platform',
        r'advanced analytics',r'unknown',r'placeholder',r'temporary',
        r'test.*corp',r'^tbd\b',r'^tba\b',
    ]
    for pattern in problematic_patterns:
        if re.search(pattern, name, re.IGNORECASE):
            return True
    return False

# Apply to finviz companies (filter out ETF rows)
fv_stocks = fv[fv['sector'].notna() & (fv['sector'] != '-')].copy()
fv_stocks['rejected_by_name'] = fv_stocks['Company'].apply(is_problematic_ticker)

rejected = fv_stocks[fv_stocks['rejected_by_name']].sort_values('mktcap', ascending=False)
print(f"\n=== Finviz stocks REJECTED by name filter: {len(rejected)} ===")
print("Top rejected by market cap (large = false positive):")
print(rejected[['ticker','Company','mktcap','avg_vol','sector','industry']].head(40).to_string())

def which_word_triggered(company_name):
    if not isinstance(company_name, str):
        return 'non-string'
    name = company_name.lower().strip()
    instant_reject_words = [
        'acquisition','spac','etf','fund','reit','warrant','preferred',
        'holdings','vehicle','ventures','income','merger','capital',
        'ai','blockchain','metaverse','crypto','bitcoin','quantum','web3',
        'tokenized','digital asset','solutions','platform','ecosystem',
        'network','alliance',
    ]
    for word in instant_reject_words:
        if word in name:
            return f"keyword: '{word}'"
    return 'pattern_match'

rejected['trigger'] = rejected['Company'].apply(which_word_triggered)
print("\nRejection triggers for large-cap false positives:")
print(rejected[rejected['mktcap'] > 1e9][['ticker','Company','mktcap','trigger']].to_string())

# ============================================================
# STEP 5 - Sector / Industry / Country distribution
# ============================================================
print("\n=== Sector distribution of passing tickers (finviz-matched) ===")
print(merged['sector'].value_counts().head(20))

print("\n=== Industry breakdown of passing tickers ===")
print(merged['industry'].value_counts().head(30))

print("\n=== Country breakdown of passing tickers ===")
print(merged['country'].value_counts().head(20))

# ============================================================
# STEP 6 - Threshold stats
# ============================================================
print("\n=== Market cap distribution (passing tickers with finviz data) ===")
print(merged['mktcap'].describe())
print(f"Below $100M: {(merged['mktcap'] < 100e6).sum()}")
print(f"Below $50M:  {(merged['mktcap'] < 50e6).sum()}")

print("\n=== Avg daily volume distribution ===")
print(merged['avg_vol'].describe())
print(f"Below 100K: {(merged['avg_vol'] < 100_000).sum()}")
print(f"Below 50K:  {(merged['avg_vol'] < 50_000).sum()}")

print("\n=== Price distribution ===")
print(merged['price'].describe())
print(f"Below $1:  {(merged['price'] < 1).sum()}")
print(f"Below $5:  {(merged['price'] < 5).sum()}")

print("\n=== Beta distribution ===")
print(merged['beta'].describe())
print(f"Beta > 5: {(merged['beta'] > 5).sum()}")
print(f"Beta < 0: {(merged['beta'] < 0).sum()}")



# ============================================================
# BONUS - Combined garbage score (multiple red flags at once)
# ============================================================



print("\n=== COMBINED GARBAGE: stocks failing 3+ screens ===")
def count_red_flags(row):
    flags = 0
    if pd.notna(row['mktcap']) and row['mktcap'] < 50e6:       flags += 1
    if pd.notna(row['avg_vol']) and row['avg_vol'] < 50_000:    flags += 1
    if pd.notna(row['price']) and row['price'] < 1:             flags += 1
    if pd.notna(row['inst_own']) and row['inst_own'] < 0.05:    flags += 1
    if pd.notna(row['short_float']) and row['short_float'] > 0.30: flags += 1
    if pd.notna(row['beta']) and row['beta'] > 5:               flags += 1
    if pd.notna(row['employees']) and row['employees'] < 10:    flags += 1
    return flags

merged['red_flags'] = merged.apply(count_red_flags, axis=1)
garbage = merged[merged['red_flags'] >= 3].sort_values('red_flags', ascending=False)
print(f"Stocks with 3+ red flags: {len(garbage)}")
print(garbage[['ticker','red_flags','mktcap','avg_vol','price','inst_own','short_float','sector']].head(30).to_string())

print("\n=== DONE ===")


