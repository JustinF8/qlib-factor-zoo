# -*- coding: utf-8 -*-
"""Backtest with 100 stocks TQ data"""
import sys, os, pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

base = os.path.dirname(os.path.abspath(__file__))
df = pd.read_parquet(os.path.join(base, 'tdx_data_100.parquet'))
print(f'Data: {len(df)} rows, {df.code.nunique()} stocks, {df.date.min().date()} ~ {df.date.max().date()}')

# Factors
sys.path.insert(0, base)
from factor_compute import _get_factor_names_from_handlers, select_30_factors, compute_factors_from_library
print('\n[Factors] Loading from library...')
factor_map = _get_factor_names_from_handlers()
FACTOR_30 = select_30_factors(factor_map)
print(f'Selected {len(FACTOR_30)} factors')
df = compute_factors_from_library(df, FACTOR_30)

# Labels
print('\n[Labels] Computing...')
df = df.sort_values(['code','date']).reset_index(drop=True)
df['label'] = (df.groupby('code')['close'].shift(-5) - df['close']) / (df['close'] + 1e-12)
df['label_cls'] = (df['label'] > 0).astype(int)

# Features
print('\n[Features] Preparing...')
factor_names = list(FACTOR_30.keys())
available = [f for f in factor_names if f in df.columns]
missing = set(factor_names) - set(available)
if missing: print(f'Missing: {missing}')

fd = df[['date','code','close'] + available + ['label','label_cls']].copy()
fd = fd.dropna().replace([np.inf, -np.inf], np.nan).dropna()
for f in available:
    col = fd[f]; m, s = col.mean(), col.std()
    if s > 0: fd[f] = col.clip(m - 10*s, m + 10*s)
print(f'Features: {len(fd)} rows, {len(available)} factors')

# Split
train_end = '2025-03-31'
train_mask = fd['date'] <= train_end
test_mask = fd['date'] > train_end
X_train = fd.loc[train_mask, available].values
y_train = fd.loc[train_mask, 'label'].values
X_test = fd.loc[test_mask, available].values
y_test = fd.loc[test_mask, 'label'].values
y_cls_test = fd.loc[test_mask, 'label_cls'].values
print(f'Train: {len(X_train)}, Test: {len(X_test)}')

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Train
print('\n[Train] LightGBM...')
split_idx = int(len(X_train_s) * 0.8)
params = {
    'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
    'num_leaves': 63, 'learning_rate': 0.03, 'feature_fraction': 0.7,
    'bagging_fraction': 0.7, 'bagging_freq': 5, 'verbose': -1,
    'seed': 42, 'n_jobs': -1, 'min_data_in_leaf': 50,
}
td = lgb.Dataset(X_train_s[:split_idx], label=y_train[:split_idx])
vd = lgb.Dataset(X_train_s[split_idx:], label=y_train[split_idx:], reference=td)
model = lgb.train(params, td, valid_sets=[td, vd], num_boost_round=800,
                   callbacks=[lgb.early_stopping(80), lgb.log_evaluation(100)])

# Evaluate
print('\n[Evaluate]...')
y_pred = model.predict(X_test_s)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
ic = np.corrcoef(y_pred, y_test)[0, 1]
acc = np.mean((y_pred > 0).astype(int) == y_cls_test)

# IC by month
fd_test = fd[test_mask].copy()
fd_test['pred'] = y_pred
fd_test['month'] = fd_test['date'].dt.to_period('M')
monthly_ic = fd_test.groupby('month').apply(lambda g: np.corrcoef(g['pred'], g['label'])[0,1] if len(g)>10 else np.nan).dropna()
ic_mean = monthly_ic.mean()
ic_ir = ic_mean / (monthly_ic.std() + 1e-12)

imp = pd.DataFrame({
    'factor': available,
    'importance': model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

print(f'RMSE: {rmse:.6f}  IC: {ic:.4f}  Acc: {acc:.2%}')
print(f'Monthly IC mean: {ic_mean:.4f}  IC IR: {ic_ir:.2f}')
print('Top 10 factors:')
for _, row in imp.head(10).iterrows():
    src = FACTOR_30.get(row['factor'], ('?','?',''))
    print(f'  {row["factor"]:25s} [{src[0]:10s}] [{src[1]:8s}] imp={row["importance"]:.0f}')

# Backtest
print('\n[Backtest]...')
df_test = fd_test.copy()
df_test = df_test.sort_values(['code','date'])
df_test['daily_ret'] = df_test.groupby('code')['close'].pct_change().shift(-1)

port = []
for date, group in df_test.groupby('date'):
    valid = group.dropna(subset=['daily_ret'])
    if len(valid) < 5: continue
    n = min(30, len(valid))
    top = valid.nlargest(n, 'pred')
    port.append({'date': date, 'ret': top['daily_ret'].mean(),
                  'bm': valid['daily_ret'].mean(), 'n': n})

port_df = pd.DataFrame(port).sort_values('date')
port_df['cum'] = (1 + port_df['ret']).cumprod()
port_df['bm_cum'] = (1 + port_df['bm']).cumprod()

total_ret = port_df['cum'].iloc[-1] - 1
bm_ret = port_df['bm_cum'].iloc[-1] - 1
ann_ret = (1 + total_ret) ** (252 / len(port_df)) - 1
ann_vol = port_df['ret'].std() * np.sqrt(252)
sharpe = ann_ret / (ann_vol + 1e-12)
max_dd = (port_df['cum'] / port_df['cum'].cummax() - 1).min()
wr = (port_df['ret'] > 0).mean()
excess = total_ret - bm_ret
xs = port_df['ret'] - port_df['bm']
ir = xs.mean() / (xs.std() + 1e-12) * np.sqrt(252)

print('\n' + '='*65)
print(f'  Backtest: {port_df["date"].iloc[0].date()} ~ {port_df["date"].iloc[-1].date()}')
print(f'  Data: 100 stocks (TQ real data), Top 30 equal-weight daily')
print(f'  {"":20s} {"Strategy":>12s} {"Benchmark":>12s}')
print(f'  {"Cum Return":20s} {total_ret:>11.2%}  {bm_ret:>11.2%}')
print(f'  {"Excess":20s} {excess:>11.2%}')
print(f'  {"Ann Return":20s} {ann_ret:>11.2%}')
print(f'  {"Ann Vol":20s} {ann_vol:>11.2%}')
print(f'  {"Sharpe":20s} {sharpe:>11.2f}')
print(f'  {"Info Ratio":20s} {ir:>11.2f}')
print(f'  {"Max Drawdown":20s} {max_dd:>11.2%}')
print(f'  {"Win Rate":20s} {wr:>11.2%}')
print(f'  {"Trading Days":20s} {len(port_df):>11d}')
print('='*65)

# Monthly IC summary
print('\n[Monthly IC]')
for m, ic_v in monthly_ic.items():
    bar = '+' * max(0, int(ic_v*50)) + '-' * max(0, int(-ic_v*50))
    print(f'  {m}: {ic_v:+.4f} {bar}')
print(f'  Mean IC: {ic_mean:+.4f}, IC IR: {ic_ir:.2f}')
print('Done!')
