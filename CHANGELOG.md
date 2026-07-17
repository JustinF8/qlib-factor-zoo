# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Fixed

- **Alpha101**: 修复 `ALPHA101` 因子公式错误，原公式混入 GTJA191 公式，现已还原 WorldQuant 原版区间公式：
  - 原公式：`($close-$open)/($high-$low+1e-12)*$volume`
  - 修复后：`Rank(($close-Min($low,12))/(Max($high,12)-Min($low,12)), 1)`

- **GTJA191**: 修复 `GTJA001` 因子公式，添加外层 Rank 并修正 ts_rank 窗口参数，对齐聚宽官方实现：
  - 原公式：`-1*Corr(Rank(Delta(Log($volume+1),1),1), Rank(($close-$open)/$open,1), 6)`
  - 修复后：`Rank(-1*Corr(Rank(Delta(Log($volume+1),1),6), Rank(($close-$open)/$open,6), 6), 1)`

- **GTJA191**: 修复 `GTJA191` 因子公式，添加外层 Rank，对齐聚宽官方实现：
  - 原公式：`Corr(Mean($volume,20),$low,5)+(($high+$low)/2)-$close`
  - 修复后：`Rank(Corr(Mean($volume,20),$low,5)+(($high+$low)/2)-$close, 1)`

### Added

- **行业中性化 `IndustryNeutralize` processor**（对齐 WorldQuant / 聚宽 `IndNeutralize`）：
  - 新增 `qlib/contrib/data/industry_map.py`：用通达信(tqcenter)数据源获取**申万一级('16')**行业分类，构建 个股(任意格式代码)→申万行业名 映射，落盘缓存（默认 30 天 TTL）。级别映射：`industry`/`sector`→申万一级('16')，`subindustry`→申万二级('17')。
  - 新增 `qlib/data/dataset/processor.py::IndustryNeutralize`：对 `fields_group` 中每个因子列，在每个交易日截面内，减去该股票所属申万行业的组内均值（等价于对行业哑变量回归取残差）。数据源不可用或某股票无行业标签时保持原值（不崩溃、不产生 NaN）。
  - `GTJA191` / `Alpha101` handler 新增 `industry_neutralize=False` 开关：置 `True` 时自动在因子计算后、截面标准化前注入 `IndustryNeutralize(fields_group="feature")`。
  - 也可在 handler 配置中直接写 `{"class": "IndustryNeutralize", "kwargs": {"fields_group": "feature"}}`。
  - 背景：qlib-factor-zoo 原 GTJA191 与 Alpha101 均未做行业中性化（Alpha101 甚至主动丢弃了原版 `indneutralize`）。行业中性化可消除行业聚集/行业 beta 混淆，使多空组合在各行业间更均衡、alpha 更纯净。
  - 验证：`_test_indneutralize_proc.py` —— 真实 instrument（煤炭 601088/600188、食品饮料 600519/000858）经通达信解析出真实申万一级标签，`IndustryNeutralize` 按 [日期,申万一级] 去均值后组内均值=5.55e-17；虚构代码保持原值、无 NaN。

### Reference

- 修复依据：根目录的四个因子复现参考文件
  - `Alpha101因子 - JoinQuant.mhtml`
  - `alpha191因子 - JoinQuant.mhtml`
  - `WorldQuant Alpha101 对齐聚宽.txt`
  - `对齐聚宽 jqlib.alpha191 官方原版.txt`
