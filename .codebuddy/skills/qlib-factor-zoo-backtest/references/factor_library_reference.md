# Qlib Factor Zoo 因子库参考文档

## 六大因子库完整说明

### 1. Alpha360 — 原始量价 360

- **文件位置**: `qlib/contrib/data/handler.py` (类定义), `qlib/contrib/data/loader.py` (表达式)
- **因子数量**: 360 个 (6字段 × 60日回溯)
- **命名格式**: `CLOSE0`~`CLOSE59`, `OPEN0`~`OPEN59`, `HIGH0`~`HIGH59`, `LOW0`~`LOW59`, `VWAP0`~`VWAP59`, `VOLUME0`~`VOLUME59`
- **公式**: `Ref($field, d) / ($close + 1e-12)` (价格) / `Ref($volume, d) / ($volume + 1e-12)` (成交量)
- **特点**: 不做任何统计加工，保留最原始的量价信息，适合深度学习端到端
- **标签**: `Ref($close, -2) / Ref($close, -1) - 1` (未来1日收益率)

### 2. Alpha158 — 经典量价 158

- **文件位置**: `qlib/contrib/data/handler.py` (类定义), `qlib/contrib/data/loader.py` (表达式)
- **因子数量**: 158 个
- **因子分组**: kbar(9) / price(25) / volume(5) / ROC(5) / MA(5) / STD(5) / BETA(5) / RSQR(5) / RESI(5) / MAX(5) / MIN(5) / QTLU(5) / QTLD(5) / RANK(5) / RSV(5) / IMAX(5) / IMIN(5) / IMXD(5) / CORR(5) / CORD(5) / CNTP/CNTN/CNTD(各5) / SUMP/SUMN/SUMD(各5) / VMA(5) / VSTD(5) / WVMA(5) / VSUMP/VSUMN/VSUMD(各5)
- **特点**: Qlib 最经典的因子集，适合 LightGBM/XGBoost
- **标签**: `Ref($close, -2) / Ref($close, -1) - 1`

### 3. Alpha101 — WorldQuant 101 公式化 Alpha

- **文件位置**: `qlib/contrib/data/handler.py` (类定义), `qlib/contrib/data/loader_alpha101.py` (101个表达式)
- **因子数量**: 101 个
- **命名格式**: `ALPHA001` ~ `ALPHA101`
- **来源**: Kakushadze (2015), "101 Formulaic Alphas", arXiv:1601.00991
- **特点**: 表达式高度复杂，大量使用 Rank/CsRank/Corr/Cov/Delta 等截面算子
- **使用的自定义算子**: `TsArgmax`, `TsArgmin`

### 4. GTJA191 — 国泰君安 191 Alpha

- **文件位置**: `qlib/contrib/data/handler.py` (类定义), `qlib/contrib/data/loader_gtja191.py` (191个表达式)
- **因子数量**: 191 个
- **命名格式**: `GTJA001` ~ `GTJA191`
- **来源**: 国泰君安证券研究所 2014 年 Alpha 因子研报
- **特点**: A股市场经典因子集，大量使用 `SMA(X, N, M)` 算子（中国式递归移动平均）
- **使用的自定义算子**: `SMA`, `TsArgmax`, `TsArgmin`, `Amount`

### 5. TDXGS — 通达信/同花顺 技术指标 ~90

- **文件位置**: `qlib/contrib/data/handler.py` (类定义，表达式内联)
- **因子数量**: ~90 个
- **命名格式**: `TDXGS_<指标名>_<参数>`，如 `TDXGS_RSI_14`, `TDXGS_CCI_20`
- **因子分组**: EMA(7) / MA(4) / ATR(4) / RSI(4) / BIAS(3) / BBI(1) / WR(2) / CCI(2) / DMI(8) / BOLL(6) / PSY(4) / ROC(2) / MTM(2) / TRIX(4) / VR(2) / CR(2) / ARBR(2) / OBV(1) / MFI(2) / DPO(2) / TAQ(6) / KTN(3) / EMV(2) / MASS(2) / DFMA(2) / STD(2) / 加工因子(7)
- **特点**: 完整覆盖通达信A股软件常用技术指标，所有公式按通达信口径实现
- **标签**: `Ref($close, -2) / Ref($close, -1) - 1`

### 6. JQ110 — 聚宽 110+ 因子

- **文件位置**: `qlib/contrib/data/handler.py` (JQ110DataHandler + JQ110DL)
- **因子数量**: ~113 个
- **命名格式**: `JQ110_<指标名>_<参数>`
- **因子分组**:
  - MOMENTUM(38): ROC/BIAS/Aroon/BBI/CCI/CR/MASS/TRIX/Price1M~1Y/Rank/BullBear/VPT/Volume1M/PLRC
  - EMOTION(36): VOL/DAVOL/turnover_vol/TVMA/VEMA/VSTD/ARBR/ATR/PSY/VMACD/VOSC/VR/VROC/WVAD/money_flow
  - TECHNICAL(17): EMA/EMAC/MAC/MACDC/BOLL/MFI
  - RISK(12): Variance/SharpeRatio/Skewness/Kurtosis
  - STYLE(10): daily_std/hist_sigma/residual_vol/cumulative_range/momentum/liquidity/share_turnover/beta
- **特点**: 包含 RISK 和 STYLE 分组，其他因子库不含
- **标签**: `Ref($close, -5) / $close - 1` (未来5日收益率)

## 因子选取对照表

| 因子库    | 选取的5个因子 | 类别映射 |
|-----------|-------------|---------|
| Alpha360  | CLOSE5, OPEN20, HIGH10, LOW30, VOLUME15 | 动量/跳空/阻力/支撑/量能 |
| Alpha158  | KLEN, BETA20, CORR20, RSV10, VSTD10 | 波动/趋势/量价/位置/量波 |
| JQ110     | JQ110_ROC_020, JQ110_VR, JQ110_MACDC, JQ110_Variance_020, JQ110_beta | 动量/情绪/技术/风险/风格 |
| Alpha101  | ALPHA001, ALPHA012, ALPHA028, ALPHA046, ALPHA083 | 反转/量价背离/规模/日内/时序 |
| GTJA191   | GTJA001, GTJA032, GTJA052, GTJA101, GTJA155 | 反转/波动/量价/形态/统计 |
| TDXGS     | TDXGS_RSI_14, TDXGS_CCI_20, TDXGS_ATR_14, TDXGS_BOLL_UP_20, TDXGS_EMA_20 | RSI/CCI/ATR/BOLL/EMA |

## 自定义算子总表

### 基础算子 (Alpha101/GTJA191 依赖)
| 算子 | 签名 | 说明 |
|------|------|------|
| `TsArgmax` | `TsArgmax(X, N)` | 滚动窗口最大值位置取值 |
| `TsArgmin` | `TsArgmin(X, N)` | 滚动窗口最小值位置取值 |
| `SMA` | `SMA(X, N, M)` | 中国式递归移动平均 |
| `Amount` | `Amount()` | 成交额 (vwap*volume 代理) |

### 通达信技术指标 (TDXGS 依赖)
ATR, RSV, RSI, BIAS, BBI, WR, CCI, PDI, MDI, ADX, ADXR, BOLL_UP/DN/MID, PSY, PSYMA, ROC, MAROC, MTM, MTMMA, TRIX, TRMA, VR, CR, AR, BR, OBV, MFI, DPO, MADPO, TAQ_UP/DN/MID, KTN_UP/DN/MID, EMV, MAEMV, MASS, MA_MASS, DFMA_DIF, DFMA_DIFMA, MA, STD_TDX

### 聚宽新增算子 (JQ110 依赖)
AroonUp, AroonDown, BullPower, BearPower, VPT, WVAD, VOSC, Variance, Skewness, Kurtosis, SharpeRatio, PriceRank, CumulativeRange, MoneyFlow

## 适用场景推荐

| 场景 | 推荐因子库 |
|------|-----------|
| 深度学习端到端 (LSTM/Transformer) | Alpha360 |
| 经典机器学习 (LightGBM/XGBoost) | Alpha158 |
| 国际量化因子研究 | Alpha101 |
| A股传统多因子选股 | GTJA191 |
| A股技术分析策略 | TDXGS |
| 多维度策略因子 | JQ110 |
| 全维度因子覆盖 | Alpha158 + TDXGS + JQ110 |
