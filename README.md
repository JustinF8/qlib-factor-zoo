# Qlib Factor Zoo

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

基于 **Microsoft Qlib** 扩展的**六大因子库**，专为中国 A 股市场优化的量化因子工程。

> 本项目 Fork 自 [Microsoft Qlib](https://github.com/microsoft/qlib)，在其基础上新增了聚宽(JQ110)、通达信(TDXGS)等 A 股特色因子库，并提供通达信数据源接入工具。

---

## 六大因子库

| 因子库 | 因子数 | 来源 | 说明 |
|--------|--------|------|------|
| **Alpha360** | 360 | Qlib 原生 | 原始量价回溯60日，适合深度学习端到端 |
| **Alpha158** | 158 | Qlib 原生 | 经典量价统计特征（K线形态+滚动统计） |
| **Alpha101** | 101 | WorldQuant | 101 公式化 Alpha 因子 |
| **GTJA191** | 191 | 国泰君安 | 191 短周期交易 Alpha 因子 |
| **TDXGS** | ~90 | 通达信/同花顺 | 经典技术指标（MACD/KDJ/RSI/BOLL 等） |
| **JQ110** | ~113 | 聚宽 | 动量/情绪/技术/风险/风格多维度因子 |

**总计约 1013 个因子**，覆盖价格、成交量、技术指标、统计特征、截面排名等维度。

---

## 项目结构

```
qlib-factor-zoo/
├── qlib/                    # Qlib 核心库（含六大因子库实现）
│   ├── contrib/data/        # ★ 因子库核心代码
│   │   ├── handler.py       # 六大 DataHandler 类
│   │   ├── custom_ops.py    # 50+ 自定义算子
│   │   ├── loader.py        # Alpha158/Alpha360 表达式生成
│   │   ├── loader_alpha101.py  # Alpha101 表达式
│   │   └── loader_gtja191.py   # GTJA191 表达式
│   ├── data/                # 数据层
│   ├── model/               # 模型层
│   ├── backtest/            # 回测引擎
│   └── ...
├── tools/                   # 实用工具
│   ├── tdx_data_to_qlib.py  # 通达信数据 → Qlib 格式转换
│   └── check_six_handlers.py # 六大因子库健康检查
├── examples/                # 策略示例
│   ├── multi_factor_strategy_30.py       # 30因子 LightGBM 策略
│   └── multi_factor_strategy_30_fast.py  # 向量化加速版
├── docs/                    # 文档
│   ├── 六大因子库使用说明.md
│   ├── 五因子库调用流程详解.md
│   └── 聚宽380因子复现分析报告.md
├── LICENSE                  # MIT License (Microsoft)
└── README.md                # 本文件
```

---

## 快速开始

### 安装

```bash
# 克隆项目
git clone https://github.com/YOUR_USERNAME/qlib-factor-zoo.git
cd qlib-factor-zoo

# 安装依赖
pip install -e .
# 或
pip install pyyaml numpy pandas mlflow redis dill fire lightgbm gym cvxpy pyarrow
```

### 使用六大因子库

```python
import qlib
from qlib.config import C

# 初始化 Qlib
qlib.init(provider_uri="your_data_path")

# 使用 Alpha158 因子库
from qlib.contrib.data.handler import Alpha158
handler = Alpha158(instruments="csi300", start_time="2023-01-01", end_time="2024-01-01")
features = handler.fetch()

# 使用 TDXGS 通达信因子库
from qlib.contrib.data.handler import TDXGS
handler = TDXGS(instruments="csi300", start_time="2023-01-01", end_time="2024-01-01")
features = handler.fetch()
```

### 通达信数据转换

```bash
# 一键导出+转换
python tools/tdx_data_to_qlib.py all --pool csi300 --qlib_dir ./tdx_qlib_data

# 分步操作
python tools/tdx_data_to_qlib.py export --pool csi300 --start 20230101 --end 20240628
python tools/tdx_data_to_qlib.py dump --csv_dir ./tdx_csv_data --qlib_dir ./tdx_qlib_data
```

### 运行示例策略

```bash
# 30因子 LightGBM 多因子策略
python examples/multi_factor_strategy_30.py

# 向量化加速版
python examples/multi_factor_strategy_30_fast.py
```

### 健康检查

```bash
python tools/check_six_handlers.py
```

---

## 自定义算子

`qlib/contrib/data/custom_ops.py` 提供了 50+ 个自定义算子，包括：

- **趋势类**: SMA, MTM, ROC, BIAS
- **波动类**: ATR, STD_TDX
- **超买超卖**: RSI, CCI, WR
- **量价关系**: OBV, VR, EMV, MASS
- **人气指标**: AR, BR, CR
- **通道类**: KTN_UP/DN/MID, TAQ_UP/DN/MID
- **其他**: TsArgmax, TsArgmin, DFMA_DIF, Amount 等

---

## 致谢

- 本项目基于 **[Microsoft Qlib](https://github.com/microsoft/qlib)** (MIT License)
- Alpha101 因子公式来自 Kakushadze "101 Formulaic Alphas"
- GTJA191 因子来自国泰君安 "191 Short-period Trading Alpha Factors"
- 通达信技术指标参考通达信/同花顺公式体系
- 聚宽因子参考 JoinQuant 因子文档

---

## 许可证

本项目基于 MIT 许可证开源，原始 Qlib 代码版权归 **Microsoft Corporation** 所有。

详见 [LICENSE](LICENSE) 文件。
