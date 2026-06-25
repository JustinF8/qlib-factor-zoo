# -*- coding: utf-8 -*-
"""
30因子机器学习多因子策略
从六大因子库中各选取5个最不相关的因子，组成30因子组合。
因子定义从项目自身的因子库组件动态加载，不硬编码因子公式。
数据源：通达信TQ / AKShare / 模拟数据
模型：LightGBM

六大因子库来源：
  Alpha360  - 原始价格/成交量时间序列（最近60天）
  Alpha158  - K线形态 + 滚动统计因子
  JQ110     - 聚宽110+因子（动量/情绪/技术/风险/风格）
  Alpha101  - WorldQuant 101 Alpha因子
  GTJA191   - 国泰君安191因子
  TDXGS     - 通达信/同花顺技术指标因子

依赖安装：
  pip install akshare pandas numpy scikit-learn lightgbm joblib
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import io
# Fix Windows GBK encoding issue for Unicode characters
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import akshare as ak
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# ============================================================
# 0. 从因子库组件动态加载因子定义（不硬编码）
# ============================================================
# 因子定义来自项目 qlib/contrib/data/ 下的六大因子库 Handler
# factor_compute 模块提供桥接：从 Handler 读取因子名 → pandas 向量化计算

from factor_compute import (
    _get_factor_names_from_handlers,
    select_30_factors,
    compute_factors_from_library,
)

# 运行时动态获取 FACTOR_30
FACTOR_30 = None  # 将在 main() 中初始化

# ============================================================
# 1. 数据获取模块（AKShare）
# ============================================================

def fetch_stock_pool():
    """获取沪深300成分股作为股票池"""
    try:
        df = ak.index_stock_cons_csindex(symbol="000300")
        codes = df["成分券代码"].tolist()
        print(f"获取沪深300成分股: {len(codes)} 只")
        return codes
    except Exception as e:
        print(f"获取成分股失败: {e}，使用预设股票池")
        # 预设一批代表性股票
        return ["000001", "000002", "000333", "000651", "000858",
                "600000", "600036", "600276", "600519", "600900",
                "601318", "601398", "601857", "603259", "688981"]


def fetch_stock_daily(code, start_date, end_date, cache_dir="./stock_cache"):
    """获取单只股票日线数据（带超时、重试和本地缓存）"""
    import time

    # 检查缓存
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{code}_{start_date}_{end_date}.parquet")
    if os.path.exists(cache_file):
        try:
            df = pd.read_parquet(cache_file)
            if len(df) > 50:
                return df
        except:
            pass

    # 从AKShare获取（最多重试3次，每次超时15秒）
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 使用 signal alarm 做超时控制（仅 Unix），Windows 用简单方式
            start = time.time()
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust="qfq"
            )
            elapsed = time.time() - start
            if elapsed > 20:
                print(f"    {code} 请求耗时 {elapsed:.0f}s，跳过")

            if df is None or len(df) == 0:
                return None

            # AKShare 返回的列名是中文
            col_map = {
                "日期": "date", "开盘": "open", "收盘": "close",
                "最高": "high", "最低": "low", "成交量": "volume",
                "成交额": "amount", "换手率": "turnover",
            }
            rename_dict = {k: v for k, v in col_map.items() if k in df.columns}
            df = df.rename(columns=rename_dict)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

            required_cols = ["date", "open", "close", "high", "low", "volume"]
            for col in required_cols:
                if col not in df.columns:
                    return None

            df["code"] = code
            result = df[["date", "code", "open", "close", "high", "low", "volume"]]

            # 保存缓存
            try:
                result.to_parquet(cache_file, index=False)
            except:
                pass

            return result
        except Exception as e:
            err_msg = str(e)[:80]
            if attempt < max_retries - 1:
                wait = 1 + attempt  # 1秒、2秒 逐次递增
                print(f"    {code} 第{attempt+1}次失败({err_msg})，{wait}s后重试...")
                time.sleep(wait)
            else:
                print(f"    {code} 获取失败，跳过 ({err_msg})")
    return None


def generate_synthetic_data(codes, start_date, end_date):
    """当AKShare不可用时，生成模拟数据用于策略演示"""
    import numpy as np
    np.random.seed(42)

    dates = pd.date_range(start=start_date, end=end_date, freq="B")
    all_data = []

    for code in codes:
        n = len(dates)
        # 用随机游走生成价格序列
        returns = np.random.normal(0.0003, 0.018, n)  # 日收益均值约0.03%，波动1.8%
        close = 20 * np.exp(np.cumsum(returns))
        close = np.maximum(close, 1)

        # 生成OHLC
        high = close * (1 + np.abs(np.random.normal(0, 0.015, n)))
        low = close * (1 - np.abs(np.random.normal(0, 0.015, n)))
        open_price = close * (1 + np.random.normal(0, 0.005, n))

        # 成交量
        volume = np.random.lognormal(15, 0.8, n)

        df = pd.DataFrame({
            "date": dates,
            "code": code,
            "open": open_price,
            "close": close,
            "high": high,
            "low": low,
            "volume": volume,
        })
        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)


def build_universe_data(codes, start_date, end_date):
    """构建全股票池日线数据"""
    import time
    all_data = []
    akshare_available = False
    total = len(codes)

    for i, code in enumerate(codes):
        # 每只股票都显示进度，避免用户以为卡死
        print(f"  进度: {i+1}/{total} ({code})", end=" ", flush=True)
        df = fetch_stock_daily(code, start_date, end_date)
        if df is not None and len(df) > 100:
            all_data.append(df)
            akshare_available = True
            print(f"[OK {len(df)}条]")
        else:
            print("[失败]")

        # AKShare 请求间隔，避免被限流
        if i < total - 1:
            time.sleep(0.3)

    if not all_data:
        print("  AKShare 数据获取失败，使用模拟数据进行演示...")
        full_df = generate_synthetic_data(codes, start_date, end_date)
        print(f"模拟数据: {len(full_df)} 条记录, {full_df['code'].nunique()} 只股票")
        return full_df

    full_df = pd.concat(all_data, ignore_index=True)
    print(f"数据构建完成: {len(full_df)} 条记录, {full_df['code'].nunique()} 只股票")
    return full_df


# ============================================================
# 2. 因子计算模块（从因子库组件动态加载）
# ============================================================

def compute_factors(df, factor_selections=None):
    """计算因子（使用 factor_compute 桥接模块，不硬编码公式）

    因子定义从 qlib/contrib/data/ 下六大因子库的 get_feature_config() 动态获取，
    因子计算使用 pandas 向量化实现，不依赖 Qlib 初始化。

    Parameters
    ----------
    df : pd.DataFrame
        包含 date, code, open, close, high, low, volume 列
    factor_selections : dict, optional
        {factor_name: (library, category, description)}

    Returns
    -------
    df : pd.DataFrame
    """
    if factor_selections is None:
        global FACTOR_30
        factor_selections = FACTOR_30

    return compute_factors_from_library(df, factor_selections)


# ============================================================
# 3. 标签与特征工程
# ============================================================

def compute_labels(df, forward_period=5):
    """计算未来N日收益率作为标签"""
    df = df.sort_values(["code", "date"]).reset_index(drop=True)

    future_close = df.groupby("code")["close"].shift(-forward_period)
    df["label"] = (future_close - df["close"]) / (df["close"] + 1e-12)

    # 分类标签：未来上涨>0为1，否则为0
    df["label_cls"] = (df["label"] > 0).astype(int)

    return df


def prepare_features(df, factor_selections=None):
    """准备特征矩阵，处理 inf/NaN"""
    if factor_selections is None:
        global FACTOR_30
        factor_selections = FACTOR_30
    factor_names = list(factor_selections.keys())

    # 检查因子是否都存在
    available = [f for f in factor_names if f in df.columns]
    missing = set(factor_names) - set(available)
    if missing:
        print(f"警告：以下因子缺失: {missing}")

    # 删除含NaN的行（保留close列用于回测时计算真实日收益）
    feature_df = df[["date", "code", "close"] + available + ["label", "label_cls"]].copy()
    feature_df = feature_df.dropna()

    # 处理 inf 值：将 inf 替换为 NaN 再删除
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.dropna()

    # 截尾处理：将超过 10 倍标准差的极端值裁剪
    for f in available:
        if f in feature_df.columns:
            col = feature_df[f]
            mean, std = col.mean(), col.std()
            if std > 0:
                upper = mean + 10 * std
                lower = mean - 10 * std
                feature_df[f] = col.clip(lower, upper)

    print(f"特征矩阵: {len(feature_df)} 条有效记录, {len(available)} 个因子")
    return feature_df, available


# ============================================================
# 4. 模型训练与评估
# ============================================================

def train_lightgbm_regression(X_train, y_train, X_val, y_val):
    """LightGBM 回归模型（预测收益率）"""
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
        "n_jobs": -1,
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        num_boost_round=500,
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(50)
        ],
    )

    return model


def train_lightgbm_classifier(X_train, y_train, X_val, y_val):
    """LightGBM 分类模型（预测涨跌方向）"""
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
        "n_jobs": -1,
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        num_boost_round=500,
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(50)
        ],
    )

    return model


def evaluate_model(model, X_test, y_test, y_cls_test, scaler, factor_names, factor_selections=None):
    """评估模型并输出结果"""
    if factor_selections is None:
        global FACTOR_30
        factor_selections = FACTOR_30

    # 预测
    y_pred_reg = model.predict(X_test)

    # 回归指标
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_reg))
    ic = np.corrcoef(y_pred_reg, y_test)[0, 1]

    # 分类指标
    y_pred_cls = (y_pred_reg > 0).astype(int)
    accuracy = np.mean(y_pred_cls == y_cls_test)

    # 因子重要性
    importance = pd.DataFrame({
        "factor": factor_names,
        "importance": model.feature_importance(importance_type="gain")
    }).sort_values("importance", ascending=False)

    print("\n" + "=" * 70)
    print("  模型评估结果")
    print("=" * 70)
    print(f"  RMSE:     {rmse:.6f}")
    print(f"  IC (Rank): {ic:.4f}")
    print(f"  Accuracy:  {accuracy:.2%}")
    print(f"\n  Top 10 重要因子:")
    for _, row in importance.head(10).iterrows():
        src = factor_selections.get(row["factor"], ("?", "?", ""))
        print(f"    {row['factor']:25s}  [{src[0]:10s}] [{src[1]:8s}] importance={row['importance']:.0f}")

    return importance


def backtest_simple(model, df_test, factor_names, scaler, top_k=10, forward_period=5):
    """简单回测：每日选预测收益最高的top_k只股票等权持有

    关键修复：
    1. 使用真实的每日收益率(close.pct_change)而非label(未来N日收益率)
       label是未来forward_period日的总收益，直接当成日收益会导致前视偏差和收益高估
    2. 将5日label拆分为约等于日收益: daily_ret ≈ label / forward_period
       但更准确的方式是直接用单日pct_change
    """
    X = scaler.transform(df_test[factor_names].values)
    df_test = df_test.copy()
    df_test["pred"] = model.predict(X)

    # 计算真实的单日收益率：当日收盘买入，次日收盘卖出
    # pct_change() 得到的是当日相对前一日的收益率
    # shift(-1) 将其转换为"下一日"的收益率（即持仓1天的真实收益）
    df_test = df_test.sort_values(["code", "date"])
    df_test["daily_real_ret"] = df_test.groupby("code")["close"].pct_change().shift(-1)
    # 例如: date=1月3日, pct_change得到(1月3日close/1月2日close-1)
    #       shift(-1)后变成(1月4日close/1月3日close-1)
    #       这就是1月3日收盘买入、1月4日收盘卖出的真实日收益

    # 按日期分组，选top_k
    portfolio = []
    for date, group in df_test.groupby("date"):
        # 筛选有真实日收益的股票
        valid = group.dropna(subset=["daily_real_ret"])
        if len(valid) < 3:
            continue
        n_pick = min(top_k, len(valid))
        top = valid.nlargest(n_pick, "pred")
        daily_ret = top["daily_real_ret"].mean()  # 等权，使用真实日收益
        benchmark_ret = valid["daily_real_ret"].mean()  # 基准：等权持有全部
        portfolio.append({
            "date": date,
            "daily_return": daily_ret,
            "benchmark_return": benchmark_ret,
            "n_stocks": n_pick,
        })

    port_df = pd.DataFrame(portfolio).sort_values("date")

    # 累计收益
    port_df["cum_return"] = (1 + port_df["daily_return"]).cumprod()
    port_df["benchmark_cum"] = (1 + port_df["benchmark_return"]).cumprod()

    # 回测指标
    total_ret = port_df["cum_return"].iloc[-1] - 1
    bm_total_ret = port_df["benchmark_cum"].iloc[-1] - 1
    ann_ret = (1 + total_ret) ** (252 / len(port_df)) - 1
    ann_vol = port_df["daily_return"].std() * np.sqrt(252)
    sharpe = ann_ret / (ann_vol + 1e-12) if ann_vol > 0 else 0
    max_dd = (port_df["cum_return"] / port_df["cum_return"].cummax() - 1).min()
    win_rate = (port_df["daily_return"] > 0).mean()

    # 超额收益
    excess_ret = total_ret - bm_total_ret
    excess_series = port_df["daily_return"] - port_df["benchmark_return"]
    ir = excess_series.mean() / (excess_series.std() + 1e-12) * np.sqrt(252)

    print("\n" + "=" * 70)
    print(f"  回测结果（测试期 {port_df['date'].iloc[0].date()} ~ {port_df['date'].iloc[-1].date()}）")
    print(f"  每日选 Top{top_k} 等权 | 股票池 {df_test['code'].nunique()} 只")
    print(f"  注意: 收益基于真实单日收益率(close.pct_change)，非label(未来{forward_period}日)")
    print("=" * 70)
    print(f"  {'':20s} {'策略':>12s} {'基准(等权)':>12s}")
    print(f"  {'累计收益:':20s} {total_ret:>11.2%}  {bm_total_ret:>11.2%}")
    print(f"  {'超额收益:':20s} {excess_ret:>11.2%}")
    print(f"  {'年化收益:':20s} {ann_ret:>11.2%}")
    print(f"  {'年化波动:':20s} {ann_vol:>11.2%}")
    print(f"  {'夏普比率:':20s} {sharpe:>11.2f}")
    print(f"  {'信息比率(IR):':20s} {ir:>11.2f}")
    print(f"  {'最大回撤:':20s} {max_dd:>11.2%}")
    print(f"  {'胜率:':20s} {win_rate:>11.2%}")
    print(f"  {'交易天数:':20s} {len(port_df):>11d}")

    return port_df


# ============================================================
# 4.5 本地通达信 TQ 数据源加载
# ============================================================

def build_universe_data_from_tdx(codes, start_date, end_date):
    """从本地通达信 TQ 接口加载数据（带缓存和超时保护）"""
    import time
    import threading

    tdx_path = 'd:/zd_zyb(x64 26011715)GA/PYPlugins/user'
    if tdx_path not in sys.path:
        sys.path.append(tdx_path)

    # 优先使用已保存的整体缓存文件（100只 > 15只）
    for bulk_cache in ["./tdx_data_100.parquet", "./tdx_data_15.parquet"]:
        if os.path.exists(bulk_cache):
            df = pd.read_parquet(bulk_cache)
            print(f"  从缓存加载: {len(df)} 条记录, {df['code'].nunique()} 只股票")
            return df

    # 缓存目录
    cache_dir = "./tdx_cache"
    os.makedirs(cache_dir, exist_ok=True)

    _tq = None
    _tq_lock = threading.Lock()

    def _get_tq():
        nonlocal _tq
        if _tq is not None:
            return _tq
        with _tq_lock:
            if _tq is not None:
                return _tq
            from tqcenter import tq
            tq.initialize(tdx_path)
            _tq = tq
            print("  通达信 TQ 接口初始化成功")
            return _tq

    def _to_tdx_code(c):
        c = c.strip()
        if '.' in c:
            return c
        if c.startswith(('6', '9')):
            return f"{c}.SH"
        elif c.startswith(('0', '3', '2')):
            return f"{c}.SZ"
        elif c.startswith('8'):
            return f"{c}.SZ" if c.startswith('83') else f"{c}.SH"
        return f"{c}.SH"

    tq = _get_tq()
    all_data = []
    total = len(codes)
    tdx_field_list = ['Open', 'High', 'Low', 'Close', 'Volume']
    start_fmt = start_date.replace("-", "")
    end_fmt = end_date.replace("-", "")
    timeout_sec = 10  # 单次请求超时秒数
    skip_count = 0

    def _fetch_one(code, code_tdx):
        """单只股票数据获取（在子线程中执行，可被超时中断）"""
        cache_file = os.path.join(cache_dir, f"{code}_{start_fmt}_{end_fmt}.parquet")
        if os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                if len(df) > 50:
                    return df
            except:
                pass

        data = tq.get_market_data(
            field_list=tdx_field_list,
            stock_list=[code_tdx],
            period='1d',
            start_time=start_fmt,
            end_time=end_fmt,
        )

        if data is None:
            return None

        df = pd.DataFrame()
        if isinstance(data, dict):
            for tdx_f, out_f in [('Open','open'),('High','high'),('Low','low'),('Close','close'),('Volume','volume')]:
                for fname in [tdx_f, tdx_f.lower(), tdx_f.upper()]:
                    if fname in data:
                        fd = data[fname]
                        if isinstance(fd, pd.DataFrame):
                            df[out_f] = fd[code_tdx] if code_tdx in fd.columns else fd.iloc[:, 0]
                        elif isinstance(fd, pd.Series):
                            df[out_f] = fd
                        elif isinstance(fd, list):
                            df[out_f] = pd.Series(fd)
                        break
        elif isinstance(data, pd.DataFrame):
            for tdx_f, out_f in [('Open','open'),('High','high'),('Low','low'),('Close','close'),('Volume','volume')]:
                for fname in [tdx_f, tdx_f.lower(), tdx_f.upper(), tdx_f.capitalize()]:
                    if fname in data.columns:
                        df[out_f] = data[fname]
                        break

        if df.empty:
            return None

        df.index = pd.to_datetime(df.index)
        df = df.sort_index().reset_index()
        df.rename(columns={'index': 'date'}, inplace=True)
        df['code'] = code

        cols = ['date', 'code', 'open', 'close', 'high', 'low', 'volume']
        df = df[[c for c in cols if c in df.columns]]

        if len(df) > 50:
            try:
                df.to_parquet(cache_file, index=False)
            except:
                pass
            return df
        return None

    for i, code in enumerate(codes):
        print(f"  进度: {i+1}/{total} ({code})", end=" ", flush=True)
        code_tdx = _to_tdx_code(code)

        try:
            # 用 threading.Thread + join(timeout) 做超时控制
            result_container = {}
            exception_container = {}

            def _fetch_wrapper():
                try:
                    result_container['data'] = _fetch_one(code, code_tdx)
                except Exception as e:
                    exception_container['error'] = e

            t = threading.Thread(target=_fetch_wrapper, daemon=True)
            t.start()
            t.join(timeout=timeout_sec)

            if t.is_alive():
                # 线程超时未结束（TQ底层C扩展卡住），跳过这只股票
                # daemon线程不会阻塞进程退出
                print(f"[超时{timeout_sec}s,跳过]")
                skip_count += 1
            elif 'error' in exception_container:
                print(f"[错误: {str(exception_container['error'])[:60]}]")
                skip_count += 1
            elif 'data' in result_container:
                result = result_container['data']
                if result is not None:
                    all_data.append(result)
                    print(f"[OK {len(result)}条]")
                else:
                    print("[跳过]")
                    skip_count += 1
        except Exception as e:
            print(f"[错误: {str(e)[:60]}]")
            skip_count += 1

        if i < total - 1:
            time.sleep(0.2)

    if skip_count > 0:
        print(f"  跳过 {skip_count} 只股票（超时/无数据）")

    if not all_data:
        print("  通达信 TQ 数据获取失败，回退到模拟数据...")
        return generate_synthetic_data(codes, start_date, end_date)

    full_df = pd.concat(all_data, ignore_index=True)
    print(f"  通达信数据构建完成: {len(full_df)} 条记录, {full_df['code'].nunique()} 只股票")
    return full_df


# ============================================================
# 5. 主流程
# ============================================================

def main(use_synthetic=False, use_tdx=False):
    global FACTOR_30

    data_source = "通达信TQ本地" if use_tdx else ("模拟数据" if use_synthetic else "AKShare")
    print("=" * 70)
    print("  30因子机器学习多因子策略")
    print("  六大因子库 × 5因子 = 30因子组合")
    print(f"  模型: LightGBM | 数据: {data_source}")
    print("  因子来源: 项目因子库组件 (qlib/contrib/data/)")
    print("=" * 70)

    # ---------- Step 0: 从因子库动态加载因子定义 ----------
    print("\n[Step 0] 从因子库组件加载因子定义...")
    try:
        factor_map = _get_factor_names_from_handlers()
        for lib, names in factor_map.items():
            print(f"  {lib:12s}: {len(names)} 个因子可用")
        FACTOR_30 = select_30_factors(factor_map)
        if not FACTOR_30 or len(FACTOR_30) < 10:
            print("  警告: 因子库加载不足，使用内置备选因子集")
            FACTOR_30 = _get_fallback_factors()
    except Exception as e:
        print(f"  警告: 因子库加载失败 ({e})，使用内置备选因子集")
        FACTOR_30 = _get_fallback_factors()

    # ---------- 数据参数 ----------
    start_date = "2023-01-01"
    end_date = "2026-05-31"
    train_end = "2025-03-31"  # 训练集截止（2023-01 ~ 2025-03）

    # ---------- Step 1: 获取数据 ----------
    print("\n[Step 1] 获取股票数据...")
    codes = fetch_stock_pool()
    codes = codes[:100]
    print(f"使用前{len(codes)}只股票")

    if use_synthetic:
        print("  使用模拟数据模式...")
        df = generate_synthetic_data(codes, start_date, end_date)
    elif use_tdx:
        print("  使用本地通达信 TQ 数据源...")
        df = build_universe_data_from_tdx(codes, start_date, end_date)
    else:
        df = build_universe_data(codes, start_date, end_date)
    print(f"数据范围: {df['date'].min().date()} ~ {df['date'].max().date()}")

    # ---------- Step 2: 计算因子 ----------
    print(f"\n[Step 2] 计算 {len(FACTOR_30)} 个因子（使用因子库定义）...")
    df = compute_factors(df, FACTOR_30)

    # ---------- Step 3: 计算标签 ----------
    print("\n[Step 3] 计算标签（未来5日收益率）...")
    df = compute_labels(df, forward_period=5)

    # ---------- Step 4: 准备特征 ----------
    print("\n[Step 4] 准备特征矩阵...")
    feature_df, factor_names = prepare_features(df, FACTOR_30)

    # 划分训练/测试集
    train_mask = feature_df["date"] <= train_end
    test_mask = feature_df["date"] > train_end

    X_train = feature_df.loc[train_mask, factor_names].values
    y_train = feature_df.loc[train_mask, "label"].values

    X_test = feature_df.loc[test_mask, factor_names].values
    y_test = feature_df.loc[test_mask, "label"].values
    y_cls_test = feature_df.loc[test_mask, "label_cls"].values

    print(f"训练集: {len(X_train)} 条, 测试集: {len(X_test)} 条")

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------- Step 5: 训练模型 ----------
    print("\n[Step 5] 训练 LightGBM 模型...")
    split_idx = int(len(X_train_scaled) * 0.8)
    X_tr, X_val = X_train_scaled[:split_idx], X_train_scaled[split_idx:]
    y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

    model = train_lightgbm_regression(X_tr, y_tr, X_val, y_val)

    # ---------- Step 6: 评估 ----------
    print("\n[Step 6] 评估模型...")
    importance = evaluate_model(model, X_test_scaled, y_test, y_cls_test, scaler, factor_names, FACTOR_30)

    # ---------- Step 7: 回测 ----------
    print("\n[Step 7] 简单回测...")
    port_df = backtest_simple(model, feature_df[test_mask], factor_names, scaler, top_k=10)

    # ---------- 因子来源分布 ----------
    print("\n" + "=" * 70)
    print("  因子来源分布")
    print("=" * 70)
    source_count = {}
    for f in factor_names:
        src = FACTOR_30.get(f, ("?", "?", ""))[0]
        source_count[src] = source_count.get(src, 0) + 1
    for src, cnt in sorted(source_count.items()):
        print(f"  {src:12s}: {cnt} 个因子")

    print("\n[完成] 策略运行结束！")
    return model, scaler, importance, port_df


def _get_fallback_factors():
    """内置备选因子集（当因子库无法加载时使用）"""
    return {
        "A360_CLOSE5":   ("Alpha360", "动量", "5日收盘价回溯"),
        "A360_OPEN20":   ("Alpha360", "跳空", "20日开盘价回溯"),
        "A360_HIGH10":   ("Alpha360", "阻力", "10日最高价回溯"),
        "A360_LOW30":    ("Alpha360", "支撑", "30日最低价回溯"),
        "A360_VOLUME15": ("Alpha360", "量能", "15日成交量回溯"),
        "A158_KLEN":     ("Alpha158", "波动", "K线实体长度"),
        "A158_BETA20":   ("Alpha158", "趋势", "20日价格斜率"),
        "A158_CORR20":   ("Alpha158", "量价", "20日价量相关系数"),
        "A158_RSV10":    ("Alpha158", "位置", "10日RSV"),
        "A158_VSTD10":   ("Alpha158", "量波", "10日成交量波动率"),
        "JQ110_ROC_020": ("JQ110", "动量", "20日变动率"),
        "JQ110_VR_026":  ("JQ110", "情绪", "26日容量比率"),
        "JQ110_MACD_DIF":("JQ110", "技术", "MACD快慢线差值"),
        "JQ110_VAR_020": ("JQ110", "风险", "20日收益方差"),
        "JQ110_BETA_060":("JQ110", "风格", "60日Beta系数"),
        "ALPHA001":      ("Alpha101", "反转", "Alpha101 #001"),
        "ALPHA012":      ("Alpha101", "量价背离", "Alpha101 #012"),
        "ALPHA028":      ("Alpha101", "规模", "Alpha101 #028"),
        "ALPHA046":      ("Alpha101", "日内", "Alpha101 #046"),
        "ALPHA083":      ("Alpha101", "时序", "Alpha101 #083"),
        "GTJA001":       ("GTJA191", "反转", "GTJA #001"),
        "GTJA032":       ("GTJA191", "波动", "GTJA #032"),
        "GTJA052":       ("GTJA191", "量价", "GTJA #052"),
        "GTJA101":       ("GTJA191", "形态", "GTJA #101"),
        "GTJA155":       ("GTJA191", "统计", "GTJA #155"),
        "TDXGS_RSI_14":  ("TDXGS", "RSI", "14日RSI"),
        "TDXGS_CCI_20":  ("TDXGS", "CCI", "20日CCI"),
        "TDXGS_ATR_14":  ("TDXGS", "ATR", "14日ATR"),
        "TDXGS_BOLL_UP_20": ("TDXGS", "BOLL", "20日布林上轨"),
        "TDXGS_EMA_20":  ("TDXGS", "EMA", "20日EMA"),
    }


if __name__ == "__main__":
    import sys
    use_synthetic = "--synthetic" in sys.argv or "--sim" in sys.argv
    use_tdx = "--tdx" in sys.argv or "--tongdaxin" in sys.argv
    model, scaler, importance, port_df = main(use_synthetic=use_synthetic, use_tdx=use_tdx)
