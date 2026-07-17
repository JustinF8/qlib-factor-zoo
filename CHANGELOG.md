# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- **文档**: 添加 `docs/聚宽380因子复现分析报告.md`，记录聚宽380标准因子的复现分析结果，包括脚本评估、问题分析、可复现因子清单及推荐复现路线图

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

### Reference

- 修复依据：根目录的四个因子复现参考文件
  - `Alpha101因子 - JoinQuant.mhtml`
  - `alpha191因子 - JoinQuant.mhtml`
  - `WorldQuant Alpha101 对齐聚宽.txt`
  - `对齐聚宽 jqlib.alpha191 官方原版.txt`
