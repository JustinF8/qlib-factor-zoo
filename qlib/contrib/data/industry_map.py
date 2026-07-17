"""
申万行业分类映射 —— 用于 qlib-factor-zoo 因子行业中性化（IndustryNeutralize）。

聚宽 / WorldQuant IndNeutralize 默认按**申万一级行业**分组；本模块用通达信
(tqcenter) 数据源获取申万 一级('16')/二级('17') 行业分类，构建
个股(任意格式代码) -> 申万行业名 的映射。

映射关系（对齐 WorldQuant IndClass 三级，按聚宽默认口径）：
    industry     -> 申万一级 ('16' 研究行业一级)   聚宽 IndNeutralize 默认口径
    sector       -> 申万一级 ('16')               申万无更粗层级，同用一级
    subindustry  -> 申万二级 ('17')               更细，作 subindustry 近似

数据来源：tqcenter（通达信量化客户端，在线，无本地回退）。
若数据源不可用或返回空，则映射为空 -> 中性化 processor 对该股票保持原值（不崩溃）。

构建策略（按板块名称反查）：
  对每个申万级别，先 get_stock_list(market) 取板块列表(名称)，再
  get_stock_list_in_sector(板块【名称】) 取成份股，反查为 个股->行业名。
  （注意：按板块代码 881xxx.SH 查询返回空，必须按名称查询。）

代码格式：tqcenter 返回 600519.SH / 000858.SZ；qlib instrument 多为 6 位数字
（如 600519）。本模块统一归一化为 tdx 格式做映射查找。
"""
from __future__ import annotations

import os
import sys
import pickle
from typing import Dict, List, Optional

import pandas as pd

# tqcenter 插件路径（与 tools/tdx_data_to_qlib.py 保持一致）
TDX_PLUGIN_PATH = "d:/zd_zyb(x64 26011715)GA/PYPlugins/user"

# 申万行业分类(market) -> IndClass 级别
_LEVEL_MARKETS = {
    "industry": "16",       # 申万一级（聚宽 IndNeutralize 默认口径，最常用）
    "sector": "16",         # 申万无更粗层级，同用一级
    "subindustry": "17",    # 申万二级（更细，作为 subindustry 近似）
}

_CACHE_FILE = os.path.join(os.path.dirname(__file__), ".cache", "industry_map.pkl")
_TTL_DAYS = 30

# 内存缓存: level -> {tdx_code: industry_name}
_MEM: Dict[str, Dict[str, str]] = {}
_MEM_LOADED = False
_TQ = None


def _to_tdx(code: str) -> str:
    """将任意格式股票代码归一化为通达信格式 600519.SH / 000858.SZ。"""
    s = str(code).strip().upper()
    if s.endswith((".SH", ".SZ", ".BJ", ".SS")):
        return s[:-3] + "." + s[-2:] if s.endswith((".SH", ".SZ", ".BJ")) else s[:-3] + ".SH"
    for p in ("SH", "SZ", "BJ"):
        if s.startswith(p):
            return s[2:] + "." + p
    if s.startswith(("6", "9")):
        return f"{s}.SH"
    if s.startswith(("0", "3", "2")):
        return f"{s}.SZ"
    if s.startswith("8"):
        return f"{s}.SH"  # 科创板/北交所粗略归沪市
    return f"{s}.SH"


def _provider():
    global _TQ
    if _TQ is not None:
        return _TQ
    if not os.path.exists(TDX_PLUGIN_PATH):
        return None
    if TDX_PLUGIN_PATH not in sys.path:
        sys.path.append(TDX_PLUGIN_PATH)
    try:
        from tqcenter import tq

        tq.initialize(TDX_PLUGIN_PATH)
        _TQ = tq
        return _TQ
    except Exception:
        return None


def _load_cache():
    global _MEM, _MEM_LOADED
    if _MEM_LOADED:
        return
    _MEM_LOADED = True
    try:
        if os.path.exists(_CACHE_FILE) and (
            pd.Timestamp.now().timestamp() - os.path.getmtime(_CACHE_FILE)
        ) < _TTL_DAYS * 86400:
            with open(_CACHE_FILE, "rb") as f:
                _MEM = pickle.load(f)
    except Exception:
        _MEM = {}


def _save_cache():
    try:
        os.makedirs(os.path.dirname(_CACHE_FILE), exist_ok=True)
        with open(_CACHE_FILE, "wb") as f:
            pickle.dump(_MEM, f)
    except Exception:
        pass


def _build_level(level: str) -> Dict[str, str]:
    """批量构建单个级别的 个股(tdx代码)->行业名 映射。

    注意：tqcenter 的 get_stock_list_in_sector 对申万板块**按名称**查询才返回
    成份股（按板块代码 881xxx.SH 查询返回空），故此处用板块 Name 反查。
    """
    p = _provider()
    if p is None:
        return {}
    market = _LEVEL_MARKETS[level]
    sectors = p.get_stock_list(market=market, list_type=1)
    if not sectors:
        return {}
    mapping: Dict[str, str] = {}
    for item in sectors:
        name = item.get("Name", "") if isinstance(item, dict) else item
        if not name:
            continue
        try:
            constituents = p.get_stock_list_in_sector(block_code=name, list_type=0)
        except Exception:
            constituents = []
        for c in constituents:
            cc = c.get("Code", c) if isinstance(c, dict) else c
            if cc:
                mapping[cc] = name
    return mapping


def _ensure_level(level: str) -> Dict[str, str]:
    _load_cache()
    if level not in _MEM:
        m = _build_level(level)
        if m:
            _MEM[level] = m
            _save_cache()
    return _MEM.get(level, {})


def get_industry_labels(instruments, level: str = "industry") -> Dict[str, Optional[str]]:
    """批量：instruments(任意格式) -> {instrument: 申万行业名 or None}。

    返回值以输入 instrument 为 key（不转换），便于直接 map 回 qlib 索引。
    """
    if level not in _LEVEL_MARKETS:
        return {inst: None for inst in instruments}
    mapping = _ensure_level(level)
    out = {}
    for inst in instruments:
        out[inst] = mapping.get(_to_tdx(inst))
    return out
