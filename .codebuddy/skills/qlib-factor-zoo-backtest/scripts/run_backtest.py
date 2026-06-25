# -*- coding: utf-8 -*-
"""Qlib Factor Zoo 回测运行脚本

此脚本封装了完整的多因子策略回测流程：
  1. 从通达信TQ加载数据（或使用缓存）
  2. 从六大因子库动态加载因子定义
  3. pandas向量化计算因子值
  4. LightGBM训练
  5. 评估 + 回测

用法:
  python run_backtest.py                          # 默认: 从tdx_data_100.parquet缓存加载100只股票
  python run_backtest.py --load                   # 先加载数据再回测
  python run_backtest.py --synthetic              # 使用模拟数据
  python run_backtest.py --stocks 50              # 使用50只股票
  python run_backtest.py --top 20                 # 每日选Top20
"""

import sys
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

# Fix Windows GBK encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# Ensure examples directory is on path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.join(BASE_DIR, '..', '..', 'examples')
if os.path.exists(EXAMPLES_DIR):
    sys.path.insert(0, EXAMPLES_DIR)

from factor_compute import (
    _get_factor_names_from_handlers,
    select_30_factors,
    compute_factors_from_library,
)


def load_data_from_tdx(codes, start_date, end_date, cache_file=None):
    """从通达信TQ加载数据"""
    import time
    import threading

    # 优先使用缓存
    if cache_file and os.path.exists(cache_file):
        df = pd.read_parquet(cache_file)
        print(f"从缓存加载: {len(df)} 条, {df['code'].nunique()} 只")
        return df

    tdx_path = r'd:/zd_zyb(x64 26011715)GA/PYPlugins/user'
    sys.path.insert(0, tdx_path)
    from tqcenter import tq
    tq.initialize(tdx_path)

    def to_tdx_code(c):
        if c.startswith(('6', '9')): return f'{c}.SH'
        elif c.startswith(('0', '3', '2')): return f'{c}.SZ'
        elif c.startswith('8'): return f'{c}.SZ' if c.startswith('83') else f'{c}.SH'
        return f'{c}.SH'

    all_data = []
    for i, code in enumerate(codes):
        ctdx = to_tdx_code(code)
        print(f'{i+1}/{len(codes)} {code}', end=' ', flush=True)
        result = {}
        def fetch():
            try:
                result['data'] = tq.get_market_data(
                    field_list=['Open','High','Low','Close','Volume'],
                    stock_list=[ctdx], period='1d',
                    start_time=start_date.replace('-',''),
                    end_time=end_date.replace('-',''))
            except Exception as e:
                result['error'] = str(e)
        t = threading.Thread(target=fetch, daemon=True)
        t.start()
        t.join(timeout=8)
        if t.is_alive():
            print('TIMEOUT')
            continue
        if 'error' in result:
            print('ERR')
            continue
        data = result.get('data')
        if data is None:
            print('NONE')
            continue
        df = pd.DataFrame()
        for tf, of in [('Open','open'),('High','high'),('Low','low'),('Close','close'),('Volume','volume')]:
            if tf in data:
                fd = data[tf]
                if isinstance(fd, pd.DataFrame):
                    df[of] = fd[ctdx] if ctdx in fd.columns else fd.iloc[:,0]
        if not df.empty:
            df.index = pd.to_datetime(df.index)
            df = df.sort_index().reset_index().rename(columns={'index':'date'})
            df['code'] = code
            all_data.append(df[['date','code','open','close','high','low','volume']])
            print(f'OK {len(df)}r')
        else:
            print('EMPTY')

    tq.close()
    full = pd.concat(all_data, ignore_index=True)
    if cache_file:
        full.to_parquet(cache_file, index=False)
    print(f'数据: {len(full)} 条, {full.code.nunique()} 只')
    return full


def generate_synthetic_data(codes, start_date, end_date):
    """生成模拟数据"""
    np.random.seed(42)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    all_data = []
    for code in codes:
        n = len(dates)
        returns = np.random.normal(0.0003, 0.018, n)
        close = 20 * np.exp(np.cumsum(returns))
        close = np.maximum(close, 1)
        df = pd.DataFrame({
            'date': dates, 'code': code,
            'open': close * (1 + np.random.normal(0, 0.005, n)),
            'close': close,
            'high': close * (1 + np.abs(np.random.normal(0, 0.015, n))),
            'low': close * (1 - np.abs(np.random.normal(0, 0.015, n))),
            'volume': np.random.lognormal(15, 0.8, n),
        })
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)


def run_backtest(df, top_k=30, train_end='2025-03-31'):
    """运行完整回测流程"""
    # 因子加载
    print('\n' + '='*60)
    print('[Step 1] 加载因子库...')
    factor_map = _get_factor_names_from_handlers()
    for lib, names in factor_map.items():
        print(f'  {lib:12s}: {len(names)} 个因子')
    FACTOR_30 = select_30_factors(factor_map)
    print(f'  选取 {len(FACTOR_30)} 个因子')

    # 因子计算
    print('\n[Step 2] 计算因子...')
    df = compute_factors_from_library(df, FACTOR_30)

    # 标签
    print('\n[Step 3] 计算标签...')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df['label'] = (df.groupby('code')['close'].shift(-5) - df['close']) / (df['close'] + 1e-12)
    df['label_cls'] = (df['label'] > 0).astype(int)

    # 特征
    print('\n[Step 4] 准备特征...')
    factor_names = list(FACTOR_30.keys())
    available = [f for f in factor_names if f in df.columns]
    missing = set(factor_names) - set(available)
    if missing:
        print(f'  缺失因子: {missing}')

    fd = df[['date','code','close'] + available + ['label','label_cls']].copy()
    fd = fd.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    for f in available:
        col = fd[f]; m, s = col.mean(), col.std()
        if s > 0: fd[f] = col.clip(m - 10*s, m + 10*s)
    print(f'  特征矩阵: {len(fd)} 条, {len(available)} 个因子')

    # 切分
    train_mask = fd['date'] <= train_end
    test_mask = fd['date'] > train_end
    X_train = fd.loc[train_mask, available].values
    y_train = fd.loc[train_mask, 'label'].values
    X_test = fd.loc[test_mask, available].values
    y_test = fd.loc[test_mask, 'label'].values
    y_cls_test = fd.loc[test_mask, 'label_cls'].values
    print(f'  训练: {len(X_train)} | 测试: {len(X_test)}')

    # 标准化
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 训练
    print('\n[Step 5] 训练 LightGBM...')
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

    # 评估
    print('\n[Step 6] 评估...')
    y_pred = model.predict(X_test_s)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    ic = np.corrcoef(y_pred, y_test)[0, 1]
    acc = np.mean((y_pred > 0).astype(int) == y_cls_test)

    fd_test = fd[test_mask].copy()
    fd_test['pred'] = y_pred
    fd_test['month'] = fd_test['date'].dt.to_period('M')
    monthly_ic = fd_test.groupby('month').apply(
        lambda g: np.corrcoef(g['pred'], g['label'])[0,1] if len(g)>10 else np.nan
    ).dropna()
    ic_mean = monthly_ic.mean()
    ic_ir = ic_mean / (monthly_ic.std() + 1e-12)

    imp = pd.DataFrame({
        'factor': available,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

    print(f'  RMSE: {rmse:.6f}  |  IC: {ic:.4f}  |  Acc: {acc:.2%}')
    print(f'  Monthly IC mean: {ic_mean:.4f}  |  IC IR: {ic_ir:.2f}')
    print(f'\n  Top 10 因子:')
    for _, row in imp.head(10).iterrows():
        src = FACTOR_30.get(row['factor'], ('?','?',''))
        print(f'    {row["factor"]:25s} [{src[0]:10s}] [{src[1]:8s}] imp={row["importance"]:.0f}')

    # 回测
    print('\n[Step 7] 回测...')
    fd_test = fd_test.sort_values(['code','date'])
    fd_test['daily_ret'] = fd_test.groupby('code')['close'].pct_change().shift(-1)

    port = []
    for date, group in fd_test.groupby('date'):
        valid = group.dropna(subset=['daily_ret'])
        if len(valid) < 5:
            continue
        n = min(top_k, len(valid))
        top = valid.nlargest(n, 'pred')
        port.append({
            'date': date,
            'ret': top['daily_ret'].mean(),
            'bm': valid['daily_ret'].mean(),
            'n': n,
        })

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
    print(f'  回测期: {port_df["date"].iloc[0].date()} ~ {port_df["date"].iloc[-1].date()}')
    print(f'  股票池: {fd_test["code"].nunique()} 只 | 每日 Top{top_k} 等权')
    print(f'  {"":20s} {"策略":>12s} {"基准":>12s}')
    print(f'  {"累计收益":20s} {total_ret:>11.2%}  {bm_ret:>11.2%}')
    print(f'  {"超额收益":20s} {excess:>11.2%}')
    print(f'  {"年化收益":20s} {ann_ret:>11.2%}')
    print(f'  {"年化波动":20s} {ann_vol:>11.2%}')
    print(f'  {"夏普比率":20s} {sharpe:>11.2f}')
    print(f'  {"信息比率":20s} {ir:>11.2f}')
    print(f'  {"最大回撤":20s} {max_dd:>11.2%}')
    print(f'  {"胜率":20s} {wr:>11.2%}')
    print(f'  {"交易天数":20s} {len(port_df):>11d}')
    print('='*65)

    # 月度IC
    print('\n[月度 IC]')
    for m, ic_v in monthly_ic.items():
        bar = '+' * max(0, int(ic_v*50)) + '-' * max(0, int(-ic_v*50))
        print(f'  {m}: {ic_v:+.4f} {bar}')
    print(f'  Mean IC: {ic_mean:+.4f}, IC IR: {ic_ir:.2f}')

    # 因子来源分布
    print('\n[因子来源分布]')
    src_count = {}
    for f in available:
        src = FACTOR_30.get(f, ('?','?',''))[0]
        src_count[src] = src_count.get(src, 0) + 1
    for src, cnt in sorted(src_count.items()):
        print(f'  {src:12s}: {cnt} 个')

    return model, scaler, imp, port_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Qlib Factor Zoo 多因子策略回测')
    parser.add_argument('--load', action='store_true', help='从TQ加载数据(否则用缓存)')
    parser.add_argument('--synthetic', action='store_true', help='使用模拟数据')
    parser.add_argument('--stocks', type=int, default=100, help='股票数量')
    parser.add_argument('--top', type=int, default=30, help='每日选股数')
    parser.add_argument('--train-end', default='2025-03-31', help='训练截止日期')
    parser.add_argument('--start', default='2023-01-01', help='数据开始日期')
    parser.add_argument('--end', default='2026-05-31', help='数据结束日期')
    args = parser.parse_args()

    print('='*60)
    print('  Qlib Factor Zoo 多因子策略回测')
    print('  六大因子库 × 5 = 30因子 + LightGBM')
    print('='*60)

    # 股票池
    codes = ['000001','000002','000063','000066','000069','000100','000157','000166','000301','000333',
             '000338','000408','000425','000538','000568','000596','000617','000625','000630','000651',
             '000661','000708','000725','000768','000776','000786','000792','000800','000807','000858',
             '000876','000895','000938','000963','000977','000983','000999','001289','001965','001979',
             '002001','002027','002049','002050','002074','002129','002142','002179','002180','002230',
             '002236','002241','002252','002271','002304','002311','002352','002371','002410','002415',
             '002459','002460','002463','002466','002475','002493','002594','002601','002648','002709',
             '002714','002736','002812','002841','002916','002920','002938','003816','300014','300015',
             '300033','300059','300124','300274','300308','300316','300347','300408','300413','300433',
             '300442','300450','300498','300502','300529','300628','300661','300750','300751','300759']
    codes = codes[:args.stocks]

    if args.synthetic:
        df = generate_synthetic_data(codes, args.start, args.end)
    elif args.load:
        cache = os.path.join(EXAMPLES_DIR, f'tdx_data_{args.stocks}.parquet')
        df = load_data_from_tdx(codes, args.start, args.end, cache_file=cache)
    else:
        # 使用缓存
        cache = os.path.join(EXAMPLES_DIR, f'tdx_data_{args.stocks}.parquet')
        if not os.path.exists(cache):
            # 尝试 100 只的缓存
            cache = os.path.join(EXAMPLES_DIR, 'tdx_data_100.parquet')
        if os.path.exists(cache):
            df = pd.read_parquet(cache)
            if args.stocks < 100:
                df = df[df['code'].isin(codes)]
            print(f'从缓存加载: {len(df)} 条, {df["code"].nunique()} 只')
        else:
            df = load_data_from_tdx(codes, args.start, args.end, cache_file=cache)

    model, scaler, imp, port = run_backtest(df, top_k=args.top, train_end=args.train_end)
    print('\n[Done] 回测完成!')
