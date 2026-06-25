---
name: qlib-factor-zoo-backtest
description: >
  This skill provides a complete workflow for multi-factor strategy backtesting using the Qlib Factor Zoo.
  It covers loading stock data (from AKShare or TDX TQ local interface), computing factors from six major
  factor libraries (Alpha360, Alpha158, Alpha101, GTJA191, TDXGS, JQ110), training LightGBM models,
  and running backtests. This skill should be used when the user wants to run, modify, or create
  quantitative trading strategy backtests using the qlib-factor-zoo project's factor libraries.
  Trigger phrases include: "回测", "因子策略", "多因子选股", "运行策略", "修改策略回测",
  "factor backtest", "multi-factor strategy", "LightGBM选股".
---

# Qlib Factor Zoo 多因子策略回测

基于 qlib-factor-zoo 项目的六大因子库（Alpha360 / Alpha158 / Alpha101 / GTJA191 / TDXGS / JQ110）进行多因子选股策略回测的完整工作流。

## 项目概述

qlib-factor-zoo 是一个基于 Microsoft Qlib 的因子库扩展项目，包含 **约 1013 个因子**，覆盖六大类：

| 因子库    | 数量 | 来源                 | 特点                     |
|-----------|------|----------------------|--------------------------|
| Alpha360  | 360  | Qlib 原生            | 60日原始量价回溯          |
| Alpha158  | 158  | Qlib 原生            | K线形态 + 滚动统计        |
| Alpha101  | 101  | WorldQuant 2015      | 101 公式化 Alpha          |
| GTJA191   | 191  | 国泰君安 2014 研报   | A股经典多因子             |
| TDXGS     | ~90  | 通达信/同花顺指标    | A股技术分析指标           |
| JQ110     | ~113 | 聚宽策略因子         | 动量/情绪/技术/风险/风格  |

## 核心文件

项目关键文件（相对于项目根目录 `qlib-factor-zoo/`）：

| 文件                              | 作用                                             |
|-----------------------------------|--------------------------------------------------|
| `examples/multi_factor_strategy_30.py` | 主策略文件：30因子 + LightGBM 完整流程 (809行)  |
| `examples/factor_compute.py`      | 因子计算桥接模块：从因子库读定义 → pandas向量化计算 (593行) |
| `examples/run_bt100.py`           | 100只股票真实数据回测脚本                        |
| `examples/load_tdx100.py`         | 从通达信TQ加载100只股票数据                      |
| `examples/load_tdx15.py`          | 从通达信TQ加载15只股票数据                       |
| `qlib/contrib/data/handler.py`    | 六大因子库 Handler 类定义 (1027行)               |
| `qlib/contrib/data/loader.py`     | Alpha360DL / Alpha158DL 表达式生成               |
| `qlib/contrib/data/loader_alpha101.py` | Alpha101DL 101个表达式 (797行)              |
| `qlib/contrib/data/loader_gtja191.py`  | GTJA191DL 191个表达式 (1266行)              |
| `docs/六大因子库使用说明.md`      | 六大因子库完整文档 (615行)                       |

## 工作流程

### 1. 数据获取

支持三种数据源：

**A. 通达信 TQ 本地接口** (推荐，用于实盘场景)
```python
from multi_factor_strategy_30 import build_universe_data_from_tdx
df = build_universe_data_from_tdx(codes, start_date, end_date)
```
- 优先检查 `tdx_data_100.parquet` 缓存
- 使用 threading + timeout 保护，防止单只股票卡住
- 缓存到 `tdx_cache/` 目录

**B. AKShare 在线接口**
```python
from multi_factor_strategy_30 import build_universe_data
df = build_universe_data(codes, start_date, end_date)
```
- 带缓存到 `stock_cache/` 目录
- 最多重试3次，逐次递增等待

**C. 模拟数据** (快速测试)
```python
from multi_factor_strategy_30 import generate_synthetic_data
df = generate_synthetic_data(codes, start_date, end_date)
```

数据格式要求：包含 `date, code, open, close, high, low, volume` 列。

### 2. 因子计算

因子定义从六大因子库动态加载，使用 pandas 向量化计算（不依赖 Qlib 初始化）：

```python
from factor_compute import (
    _get_factor_names_from_handlers,  # 获取各因子库的所有因子名
    select_30_factors,                # 从各库选5个代表性因子
    compute_factors_from_library,     # pandas向量化计算因子值
)

# 加载因子定义
factor_map = _get_factor_names_from_handlers()

# 选取30个因子 (每库5个)
FACTOR_30 = select_30_factors(factor_map)

# 计算因子值
df = compute_factors_from_library(df, FACTOR_30)
```

**因子计算逻辑** (在 `factor_compute.py` 中)：
- Alpha360: `shift(field, window) / (close + 1e-12)` — 回溯价格比
- Alpha158: KLEN(振幅)/BETA(趋势)/CORR(量价相关)/RSV(位置)/VSTD(量波)
- JQ110: ROC(变动率)/VR(容量比率)/MACD_DIF/VAR(方差)/BETA
- Alpha101: 向量化实现 ALPHA001(corr)/ALPHA012(sign*delta)/ALPHA028/ALPHA046/ALPHA083
- GTJA191: GTJA001/032/052/101/155 简化向量化
- TDXGS: RSI(中国式SMA)/CCI/ATR/BOLL_UP/EMA

### 3. 模型训练

使用 LightGBM 回归模型，预测未来5日收益率：

```python
import lightgbm as lgb

params = {
    'objective': 'regression', 'metric': 'rmse',
    'boosting_type': 'gbdt', 'num_leaves': 63,
    'learning_rate': 0.03, 'feature_fraction': 0.7,
    'bagging_fraction': 0.7, 'bagging_freq': 5,
    'seed': 42, 'n_jobs': -1, 'min_data_in_leaf': 50,
}
```

- 训练/验证/测试按时间切分（训练: ~2023-01 ~ 2025-03，测试: ~2025-04 ~ 2026-05）
- 使用 `early_stopping(80)` 防止过拟合
- 标签: `(close_shift(-5) - close) / close` （未来5日收益率）

### 4. 回测

每日选预测收益最高的 Top K 只股票等权持有：

```python
# 真实日收益（非label，避免前视偏差）
df['daily_ret'] = df.groupby('code')['close'].pct_change().shift(-1)

# 每日选Top N
for date, group in df.groupby('date'):
    top = valid.nlargest(n, 'pred')
    daily_ret = top['daily_ret'].mean()
```

**关键修复**: 回测使用真实的单日 `pct_change().shift(-1)` 而非 label(未来5日收益)，避免前视偏差。

### 5. 评估指标

- **RMSE**: 回归均方根误差
- **IC**: 预测值与真实值的相关系数
- **Accuracy**: 方向预测准确率
- **月度 IC**: 按月计算的 IC，评估因子稳定性
- **IC IR**: IC 均值 / IC 标准差
- **累计收益 / 超额收益**
- **夏普比率 / 信息比率**
- **最大回撤 / 胜率**

## 运行方式

### 主策略（3种模式）

```bash
# 模拟数据快速测试
cd examples
python multi_factor_strategy_30.py --synthetic

# 通达信 TQ 本地数据
python multi_factor_strategy_30.py --tdx

# AKShare 在线数据 (默认)
python multi_factor_strategy_30.py
```

### 100只股票完整回测

```bash
cd examples
# Step 1: 加载数据
python load_tdx100.py

# Step 2: 运行回测
python run_bt100.py
```

### 15只股票快速回测

```bash
cd examples
python load_tdx15.py
python run_tdx_bt.py
```

## 因子选择原则

从六大因子库各选5个分属不同类别的因子：

- Alpha360: 5个不同字段(close/open/high/low/volume) + 不同窗口
- Alpha158: KLEN(波动) / BETA20(趋势) / CORR20(量价) / RSV10(位置) / VSTD10(量波)
- JQ110: ROC(动量) / VR(情绪) / MACD_DIF(技术) / VAR(风险) / BETA(风格)
- Alpha101: ALPHA001(反转) / ALPHA012(量价背离) / ALPHA028(规模) / ALPHA046(日内) / ALPHA083(时序)
- GTJA191: GTJA001(反转) / GTJA032(波动) / GTJA052(量价) / GTJA101(形态) / GTJA155(统计)
- TDXGS: RSI(RSI) / CCI(CCI) / ATR(ATR) / BOLL_UP(BOLL) / EMA(EMA)

## 常见问题

### 因子库加载失败

`factor_compute.py` 内置备选因子集 `_get_fallback_factors()`，当因子库无法加载时自动使用硬编码的30因子。

### 股票数量建议

- 最少 15 只（效果较差，IC 可能为负）
- 推荐 100+ 只（IC 转为正，夏普可达 2.0+）
- 更多股票可进一步提升稳定性

### 数据加载超时

TQ 接口对某些股票可能卡住，已使用 `threading.Thread + join(timeout=10s)` 保护。超时股票会自动跳过。

### 前视偏差

已修复：回测使用 `close.pct_change().shift(-1)` 真实日收益，非 label(未来5日收益)。

## 依赖

```
pip install akshare pandas numpy scikit-learn lightgbm joblib pyarrow
```

通达信 TQ 需要本地已安装的通达信客户端和 TQ 插件。
