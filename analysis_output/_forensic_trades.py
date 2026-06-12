import pandas as pd
import numpy as np
import sys

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 40)

p = r"C:\Users\Masam\Desktop\Stock-Market\trade_history.parquet"
df = pd.read_parquet(p)

print("=" * 80)
print("1. SCHEMA")
print("=" * 80)
print(f"Rows: {len(df)}")
print(f"Cols: {list(df.columns)}")
print("Dtypes:")
print(df.dtypes)
print("\nFirst 3 rows:")
print(df.head(3))

# Identify key cols
date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
print(f"\nDate-like cols: {date_cols}")

# Pick primary date col
dcol = None
for cand in ['exit_date', 'sell_date', 'close_date', 'date', 'entry_date', 'buy_date', 'open_date']:
    if cand in df.columns:
        dcol = cand
        break
if dcol is None and date_cols:
    dcol = date_cols[0]
print(f"Using date col: {dcol}")
if dcol:
    df[dcol] = pd.to_datetime(df[dcol])
    print(f"Date range: {df[dcol].min()} -> {df[dcol].max()}")

# Identify pnl col
pnl_col = None
for cand in ['PnLPct', 'pnl_pct', 'return', 'returns', 'ret', 'pct_return', 'profit_pct', 'PnL', 'pnl', 'profit']:
    if cand in df.columns:
        pnl_col = cand
        break
print(f"P&L col: {pnl_col}")

ticker_col = None
for cand in ['Symbol', 'ticker', 'symbol', 'Ticker']:
    if cand in df.columns:
        ticker_col = cand
        break
print(f"Ticker col: {ticker_col}")

print("\n" + "=" * 80)
print("2. TRADES BY MONTH")
print("=" * 80)
if dcol:
    by_mo = df.groupby(df[dcol].dt.to_period('M')).size()
    print(by_mo)

print("\n" + "=" * 80)
print("3. TOP 20 TICKERS BY TRADE COUNT")
print("=" * 80)
if ticker_col:
    print(df[ticker_col].value_counts().head(20))
    print(f"\nUnique tickers: {df[ticker_col].nunique()}")

print("\n" + "=" * 80)
print("4. P&L DISTRIBUTION")
print("=" * 80)
if pnl_col:
    s = pd.to_numeric(df[pnl_col], errors='coerce')
    print(f"count={s.count()} mean={s.mean():.6f} median={s.median():.6f} std={s.std():.6f}")
    print(f"min={s.min():.6f} max={s.max():.6f} skew={s.skew():.4f} kurt={s.kurt():.4f}")
    print(f"win_rate(>0): {(s > 0).mean():.4f}")
    print("Quantiles:")
    print(s.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))
    print("\nHistogram bins:")
    bins = np.linspace(s.min(), s.max(), 21)
    counts, edges = np.histogram(s.dropna(), bins=bins)
    for i, c in enumerate(counts):
        print(f"  [{edges[i]:+.4f}, {edges[i+1]:+.4f}]: {c}")

print("\n" + "=" * 80)
print("5. WIN RATE BY MONTH")
print("=" * 80)
if dcol and pnl_col:
    g = df.groupby(df[dcol].dt.to_period('M'))[pnl_col]
    monthly = pd.DataFrame({
        'n_trades': g.size(),
        'win_rate': g.apply(lambda x: (x > 0).mean()),
        'mean_ret': g.mean(),
        'sum_ret': g.sum(),
    })
    print(monthly)

print("\n" + "=" * 80)
print("6. HOLD TIME")
print("=" * 80)
hold_candidates = [c for c in df.columns if 'hold' in c.lower() or 'duration' in c.lower() or 'days' in c.lower()]
print(f"Hold cols: {hold_candidates}")
for c in hold_candidates:
    print(f"  {c}: min={df[c].min()} max={df[c].max()} mean={df[c].mean()}")
if 'entry_date' in df.columns and 'exit_date' in df.columns:
    ed = pd.to_datetime(df['entry_date']); xd = pd.to_datetime(df['exit_date'])
    hd = (xd - ed).dt.days
    print(f"exit-entry days: min={hd.min()} max={hd.max()} mean={hd.mean():.4f}")
    print(hd.value_counts().head(10))

print("\n" + "=" * 80)
print("7. ENTRY/EXIT PRICE LOOK-AHEAD CHECK")
print("=" * 80)
price_cols = [c for c in df.columns if any(k in c.lower() for k in ['entry_price','exit_price','buy_price','sell_price','open','high','low','close'])]
print(f"Price-related cols: {price_cols}")
for c in ['entry_price', 'exit_price', 'buy_price', 'sell_price']:
    if c in df.columns:
        print(f"  {c}: min={df[c].min()} max={df[c].max()} mean={df[c].mean():.4f}")
if 'entry_price' in df.columns and 'exit_price' in df.columns:
    spread = (df['exit_price'] - df['entry_price']) / df['entry_price']
    print(f"(exit-entry)/entry: mean={spread.mean():.6f} median={spread.median():.6f}")

print("\n" + "=" * 80)
print("8. TRADE SIZE")
print("=" * 80)
size_cols = [c for c in df.columns if any(k in c.lower() for k in ['size','shares','qty','quantity','amount','dollar','value','capital','position'])]
print(f"Size-related cols: {size_cols}")
for c in size_cols:
    try:
        s = pd.to_numeric(df[c], errors='coerce')
        print(f"  {c}: min={s.min()} max={s.max()} mean={s.mean():.2f} std={s.std():.2f}")
    except Exception as e:
        print(f"  {c}: err {e}")

print("\n" + "=" * 80)
print("9. OUTLIER TRADES")
print("=" * 80)
if pnl_col:
    print("TOP 5 WINS:")
    print(df.nlargest(5, pnl_col).to_string())
    print("\nTOP 5 LOSSES:")
    print(df.nsmallest(5, pnl_col).to_string())

print("\n" + "=" * 80)
print("10. NULLS / WEIRD VALUES")
print("=" * 80)
print("Null counts:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
if pnl_col:
    s = pd.to_numeric(df[pnl_col], errors='coerce')
    print(f"P&L infs: {np.isinf(s).sum()}, nans: {s.isna().sum()}")
    print(f"P&L exactly zero: {(s == 0).sum()}")

print("\n" + "=" * 80)
print("FOCUS: 2026-02 (best month +24.10%)")
print("=" * 80)
if dcol and pnl_col:
    feb = df[(df[dcol] >= '2026-02-01') & (df[dcol] < '2026-03-01')].copy()
    print(f"Feb 2026 trade count: {len(feb)}")
    if len(feb):
        print(f"Feb mean ret: {feb[pnl_col].mean():.6f}, sum: {feb[pnl_col].sum():.6f}, win rate: {(feb[pnl_col] > 0).mean():.4f}")
        print("\nTop 10 Feb 2026 trades by P&L:")
        print(feb.nlargest(10, pnl_col).to_string())
        if ticker_col:
            print("\nFeb 2026 ticker contribution (sum of P&L):")
            tc = feb.groupby(ticker_col)[pnl_col].agg(['sum','count','mean']).sort_values('sum', ascending=False)
            print(tc.head(15))
            top_tkr_sum = tc['sum'].iloc[0] if len(tc) else 0
            total_sum = feb[pnl_col].sum()
            if total_sum:
                print(f"\nTop ticker share of Feb total P&L: {top_tkr_sum/total_sum:.2%}")
                print(f"Top 3 tickers share: {tc['sum'].head(3).sum()/total_sum:.2%}")
                print(f"Top 5 tickers share: {tc['sum'].head(5).sum()/total_sum:.2%}")
