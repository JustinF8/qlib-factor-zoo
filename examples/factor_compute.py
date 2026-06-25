# -*- coding: utf-8 -*-
"""
因子计算桥接模块
从六大因子库获取因子定义，使用 pandas 向量化计算因子值。

不依赖 Qlib 初始化，直接对 pandas DataFrame 进行计算。
"""

import numpy as np
import pandas as pd


# ============================================================
# 从因子库组件获取因子定义
# ============================================================

def _get_factor_names_from_handlers():
    """从六大因子库 Loader 中获取所有因子名称列表。

    使用 Loader 类的静态方法 get_feature_config()，不需要 Qlib 初始化。

    返回: dict，key=因子库名，value=因子名列表
    """
    import sys
    import os
    # 确保项目根路径在 sys.path 中
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)

    factor_map = {}

    # Alpha360 - 使用 Alpha360DL 静态方法
    try:
        from qlib.contrib.data.loader import Alpha360DL
        _, names = Alpha360DL.get_feature_config()
        factor_map["Alpha360"] = names
    except Exception as e:
        print(f"  警告: Alpha360 因子获取失败: {e}")

    # Alpha158 - 使用 Alpha158DL 静态方法
    try:
        from qlib.contrib.data.loader import Alpha158DL
        _, names = Alpha158DL.get_feature_config()
        factor_map["Alpha158"] = names
    except Exception as e:
        print(f"  警告: Alpha158 因子获取失败: {e}")

    # Alpha101 - 使用 Alpha101DL 静态方法
    try:
        from qlib.contrib.data.loader_alpha101 import Alpha101DL
        _, names = Alpha101DL.get_feature_config()
        factor_map["Alpha101"] = names
    except Exception as e:
        print(f"  警告: Alpha101 因子获取失败: {e}")

    # GTJA191 - 使用 GTJA191DL 静态方法
    try:
        from qlib.contrib.data.loader_gtja191 import GTJA191DL
        _, names = GTJA191DL.get_feature_config()
        factor_map["GTJA191"] = names
    except Exception as e:
        print(f"  警告: GTJA191 因子获取失败: {e}")

    # TDXGS - 从 handler.py 中提取因子名列表（不需要 Qlib 初始化）
    try:
        names = _get_tdxgs_factor_names()
        factor_map["TDXGS"] = names
    except Exception as e:
        print(f"  警告: TDXGS 因子获取失败: {e}")

    # JQ110 - 静态方法
    try:
        from qlib.contrib.data.handler import JQ110DataHandler
        _, names = JQ110DataHandler.get_feature_config()
        factor_map["JQ110"] = names
    except Exception as e:
        print(f"  警告: JQ110 因子获取失败: {e}")

    return factor_map


def _get_tdxgs_factor_names():
    """获取 TDXGS 因子名列表（从 handler.py 提取，不需要 Qlib 初始化）。"""
    names = []
    # EMA
    for n in [5, 10, 12, 20, 26, 50, 60]:
        names.append(f"TDXGS_EMA_{n:02d}")
    # MA
    for n in [5, 10, 20, 60]:
        names.append(f"TDXGS_MA_{n:02d}")
    # ATR
    for n in [10, 14, 20, 60]:
        names.append(f"TDXGS_ATR_{n:02d}")
    # RSI
    for n in [6, 12, 14, 24]:
        names.append(f"TDXGS_RSI_{n:02d}")
    # BIAS
    for n in [6, 12, 24]:
        names.append(f"TDXGS_BIAS_{n:02d}")
    # BBI
    names.append("TDXGS_BBI")
    # WR
    for n in [6, 10]:
        names.append(f"TDXGS_WR_{n:02d}")
    # CCI
    for n in [14, 20]:
        names.append(f"TDXGS_CCI_{n:02d}")
    # DMI
    for m1, m2 in [(14, 6), (7, 3)]:
        for p in ["PDI", "MDI", "ADX", "ADXR"]:
            names.append(f"TDXGS_{p}_{m1}_{m2}")
    # BOLL
    for n in [20, 26]:
        for p in ["UP", "MID", "DN"]:
            names.append(f"TDXGS_BOLL_{p}_{n:02d}")
    # PSY
    names.extend(["TDXGS_PSY_12", "TDXGS_PSYMA_12_6", "TDXGS_PSY_20", "TDXGS_PSYMA_20_10"])
    # ROC
    names.extend(["TDXGS_ROC_12", "TDXGS_MAROC_12_6"])
    # MTM
    names.extend(["TDXGS_MTM_12", "TDXGS_MTMMA_12_6"])
    # TRIX
    for m1, m2 in [(12, 20), (9, 15)]:
        names.append(f"TDXGS_TRIX_{m1}")
        names.append(f"TDXGS_TRMA_{m1}_{m2}")
    # VR
    for m1 in [26, 12]:
        names.append(f"TDXGS_VR_{m1:02d}")
    # CR
    for n in [20, 26]:
        names.append(f"TDXGS_CR_{n:02d}")
    # ARBR
    names.extend(["TDXGS_AR_26", "TDXGS_BR_26"])
    # OBV
    names.append("TDXGS_OBV")
    # MFI
    for n in [14, 9]:
        names.append(f"TDXGS_MFI_{n:02d}")
    # DPO
    names.extend(["TDXGS_DPO_20_10", "TDXGS_MADPO_20_10_6"])
    # TAQ
    for n in [20, 50]:
        for p in ["UP", "MID", "DN"]:
            names.append(f"TDXGS_TAQ_{p}_{n:02d}")
    # KTN
    names.extend(["TDXGS_KTN_UP", "TDXGS_KTN_MID", "TDXGS_KTN_DN"])
    # EMV
    names.extend(["TDXGS_EMV_14", "TDXGS_MAEMV_14_9"])
    # MASS
    names.extend(["TDXGS_MASS_9_25", "TDXGS_MA_MASS_9_25_6"])
    # DFMA
    names.extend(["TDXGS_DFMA_DIF", "TDXGS_DFMA_DIFMA"])
    # STD
    for n in [10, 20]:
        names.append(f"TDXGS_STD_{n:02d}")
    # 加工因子
    names.extend([
        "TDXGS_ATR20_RATIO", "TDXGS_BOLL_BANDWIDTH", "TDXGS_EMA20_CSRANK",
        "TDXGS_CORR_CV_20", "TDXGS_VOL_RATIO_20", "TDXGS_AMPLITUDE",
        "TDXGS_TREND_STRENGTH",
    ])
    return names


def select_30_factors(factor_map):
    """从六大因子库中各选5个代表性因子，组成30因子组合。

    选取原则：每个库选5个分属不同类别的因子。
    返回: dict，key=因子名, value=(因子库, 类别, 描述)
    """
    selections = {}

    # Alpha360: 因子名格式如 CLOSE5, OPEN20, HIGH10, LOW30, VOLUME15
    # 选不同字段+不同窗口的代表性因子
    alpha360_factors = factor_map.get("Alpha360", [])
    alpha360_wanted = {
        "close_5": ("动量", "5日收盘价回溯"),
        "open_20": ("跳空", "20日开盘价回溯"),
        "high_10": ("阻力", "10日最高价回溯"),
        "low_30": ("支撑", "30日最低价回溯"),
        "volume_15": ("量能", "15日成交量回溯"),
    }
    for f in alpha360_factors:
        f_lower = f.lower()
        for key, (cat, desc) in alpha360_wanted.items():
            field, n = key.split("_")
            if field in f_lower and f_lower.endswith(n) and len(f_lower) == len(field) + len(n):
                selections[f] = ("Alpha360", cat, desc)
                del alpha360_wanted[key]
                break
        if len([v for v in selections.values() if v[0] == "Alpha360"]) >= 5:
            break
    # 如果按精确匹配没找全，放宽匹配
    if len([v for v in selections.values() if v[0] == "Alpha360"]) < 5:
        field_map = {"close": "动量", "open": "跳空", "high": "阻力", "low": "支撑", "volume": "量能"}
        picked_fields = set()
        for f in alpha360_factors:
            f_lower = f.lower()
            for field, cat in field_map.items():
                if field in f_lower and field not in picked_fields:
                    selections[f] = ("Alpha360", cat, f"{field}回溯因子")
                    picked_fields.add(field)
                    break
            if len(picked_fields) >= 5:
                break

    # Alpha158: 因子名如 KLEN, BETA20, CORR20, RSV10, VSTD10 等
    alpha158_factors = factor_map.get("Alpha158", [])
    alpha158_wanted = ["KLEN", "BETA20", "CORR20", "RSV10", "VSTD10"]
    alpha158_cats = {
        "KLEN": ("波动", "K线实体长度"),
        "BETA20": ("趋势", "20日价格斜率"),
        "CORR20": ("量价", "20日价量相关系数"),
        "RSV10": ("位置", "10日RSV"),
        "VSTD10": ("量波", "10日成交量波动率"),
    }
    for f in alpha158_factors:
        f_upper = f.upper()
        for wanted in alpha158_wanted:
            if wanted in f_upper and f_upper == wanted:
                cat, desc = alpha158_cats.get(wanted, ("Alpha158", wanted))
                selections[f] = ("Alpha158", cat, desc)
                break
        if len([v for v in selections.values() if v[0] == "Alpha158"]) >= 5:
            break

    # JQ110: 选不同类别的因子
    # 实际因子名: JQ110_ROC_020, JQ110_VR, JQ110_MACDC, JQ110_Variance_020, JQ110_beta
    jq110_factors = factor_map.get("JQ110", [])
    jq110_wanted = {
        "JQ110_ROC_020": ("动量", "20日变动率"),
        "JQ110_VR": ("情绪", "容量比率"),
        "JQ110_MACDC": ("技术", "MACD指标"),
        "JQ110_Variance_020": ("风险", "20日收益方差"),
        "JQ110_beta": ("风格", "Beta系数"),
    }
    for f in jq110_factors:
        if f in jq110_wanted:
            cat, desc = jq110_wanted[f]
            selections[f] = ("JQ110", cat, desc)
        if len([v for v in selections.values() if v[0] == "JQ110"]) >= 5:
            break

    # Alpha101: 选5个代表性 Alpha
    alpha101_factors = factor_map.get("Alpha101", [])
    alpha101_picks = {"ALPHA001": "反转相关", "ALPHA012": "量价背离", "ALPHA028": "规模效应", "ALPHA046": "日内动量", "ALPHA083": "时序排名"}
    for f in alpha101_factors:
        if f in alpha101_picks:
            selections[f] = ("Alpha101", alpha101_picks[f], f"Alpha101 #{f}")

    # GTJA191: 选5个代表性因子
    gtja191_factors = factor_map.get("GTJA191", [])
    gtja_picks = {"GTJA001": "反转相关", "GTJA032": "波动反转", "GTJA052": "量价动能", "GTJA101": "反转形态", "GTJA155": "统计动量"}
    for f in gtja191_factors:
        if f in gtja_picks:
            selections[f] = ("GTJA191", gtja_picks[f], f"GTJA #{f}")

    # TDXGS: 选5个不同指标类型
    tdxgs_factors = factor_map.get("TDXGS", [])
    tdx_wanted = {
        "RSI_14": ("RSI", "14日RSI"),
        "CCI_20": ("CCI", "20日CCI"),
        "ATR_14": ("ATR", "14日ATR"),
        "BOLL_UP_20": ("BOLL", "20日布林上轨"),
        "EMA_20": ("EMA", "20日EMA"),
    }
    for f in tdxgs_factors:
        f_upper = f.upper()
        for key, (cat, desc) in tdx_wanted.items():
            if key in f_upper:
                selections[f] = ("TDXGS", cat, desc)
                del tdx_wanted[key]
                break
        if not tdx_wanted:
            break

    print(f"  从六大因子库选取 {len(selections)} 个因子")
    for lib in ["Alpha360", "Alpha158", "JQ110", "Alpha101", "GTJA191", "TDXGS"]:
        cnt = len([v for v in selections.values() if v[0] == lib])
        print(f"    {lib:12s}: {cnt} 个")
    return selections


# ============================================================
# Pandas 向量化因子计算引擎
# ============================================================

def compute_factors_from_library(df, factor_selections):
    """使用 pandas 向量化计算，等价于因子库中的表达式。

    不依赖 Qlib 初始化，直接在 pandas DataFrame 上计算。

    Parameters
    ----------
    df : pd.DataFrame
        包含 date, code, open, close, high, low, volume 列
    factor_selections : dict
        {factor_name: (library, category, description)}

    Returns
    -------
    df : pd.DataFrame
        添加了因子列的 DataFrame
    """
    import time
    t_start = time.time()
    print("  排序数据...", flush=True)
    df = df.copy()
    df = df.sort_values(["code", "date"]).reset_index(drop=True)

    gb = df.groupby("code")
    close_s = df["close"]
    open_s = df["open"]
    high_s = df["high"]
    low_s = df["low"]
    vol_s = df["volume"]

    factor_names = list(factor_selections.keys())

    # ---- 辅助函数 ----
    def gb_rolling(series, window, func_name, **kwargs):
        """按股票分组执行滚动计算"""
        return series.groupby(df["code"]).transform(
            lambda x: getattr(x.rolling(window, min_periods=3), func_name)(**kwargs)
        )

    def gb_shift(series, n):
        return series.groupby(df["code"]).shift(n)

    def gb_diff(series, n=1):
        return series.groupby(df["code"]).diff(n)

    def gb_pct_change(series):
        return series.groupby(df["code"]).pct_change()

    def gb_ewm(series, span, adjust=False):
        return series.groupby(df["code"]).transform(
            lambda x: x.ewm(span=span, adjust=adjust).mean()
        )

    # ---- 按因子库分组计算 ----
    computed = set()

    for fname, (lib, cat, desc) in factor_selections.items():
        if fname in df.columns:
            computed.add(fname)
            continue

    t0 = time.time()
    print(f"  计算 {len(factor_names)} 个因子...", flush=True)

    # === Alpha360 因子: $field/Ref($field, N) ===
    for fname in factor_names:
        if fname in computed:
            continue
        src = factor_selections.get(fname, ("?", "?", ""))
        lib = src[0]
        if lib != "Alpha360":
            continue
        name_upper = fname.upper()
        # 解析字段和窗口
        for field in ["CLOSE", "OPEN", "HIGH", "LOW", "VOLUME"]:
            if field in name_upper:
                fld = field.lower()
                # 提取数字（窗口大小）
                nums = [int(s) for s in fname.split("_") if s.isdigit()]
                window = nums[-1] if nums else 5
                col = df[fld]
                df[fname] = gb_shift(col, window) / (col + 1e-12)
                computed.add(fname)
                break

    # === Alpha158 因子 ===
    for fname in factor_names:
        if fname in computed:
            continue
        src = factor_selections.get(fname, ("?", "?", ""))
        lib = src[0]
        if lib != "Alpha158":
            continue
        name_upper = fname.upper()

        if "KLEN" in name_upper:
            df[fname] = (high_s - low_s) / (open_s + 1e-12)
        elif "BETA" in name_upper:
            # 价格趋势: (MA5 - MA20) / MA20
            ma5 = gb_ewm(close_s, 5)
            ma20 = gb_rolling(close_s, 20, "mean")
            df[fname] = (ma5 - ma20) / (ma20 + 1e-12)
        elif "CORR" in name_upper:
            # 价量相关系数向量化
            roll_c_mean = gb_rolling(close_s, 20, "mean")
            roll_v_mean = gb_rolling(vol_s, 20, "mean")
            roll_c_std = gb_rolling(close_s, 20, "std")
            roll_v_std = gb_rolling(vol_s, 20, "std")
            cv_prod = close_s * vol_s
            roll_cv_mean = gb_rolling(cv_prod, 20, "mean")
            cov_cv = roll_cv_mean - roll_c_mean * roll_v_mean
            denom = roll_c_std * roll_v_std + 1e-12
            df[fname] = (cov_cv / denom).clip(-1, 1).fillna(0)
        elif "RSV" in name_upper:
            roll_high = gb_rolling(high_s, 10, "max")
            roll_low = gb_rolling(low_s, 10, "min")
            df[fname] = (close_s - roll_low) / (roll_high - roll_low + 1e-12)
        elif "VSTD" in name_upper:
            vol_std = gb_rolling(vol_s, 10, "std")
            vol_mean = gb_rolling(vol_s, 10, "mean")
            df[fname] = vol_std / (vol_mean + 1e-12)
        else:
            # 通用: 尝试识别为 rolling 统计
            df[fname] = 0.0
        computed.add(fname)

    # === JQ110 因子 ===
    for fname in factor_names:
        if fname in computed:
            continue
        src = factor_selections.get(fname, ("?", "?", ""))
        lib = src[0]
        if lib != "JQ110":
            continue
        name_upper = fname.upper()

        if "ROC" in name_upper:
            nums = [int(s) for s in fname.split("_") if s.isdigit()]
            w = nums[-1] if nums else 20
            shifted = gb_shift(close_s, w)
            df[fname] = (close_s - shifted) / (shifted + 1e-12)
        elif "VR" in name_upper:
            # 向量化VR: 26日
            close_diff = gb_diff(close_s)
            up_vol = (vol_s * (close_diff > 0)).groupby(df["code"]).transform(lambda x: x.rolling(26, min_periods=5).sum())
            dn_vol = (vol_s * (close_diff < 0)).groupby(df["code"]).transform(lambda x: x.rolling(26, min_periods=5).sum())
            eq_vol = (vol_s * (close_diff == 0)).groupby(df["code"]).transform(lambda x: x.rolling(26, min_periods=5).sum())
            df[fname] = ((up_vol + 0.5 * eq_vol) / (dn_vol + 0.5 * eq_vol + 1e-12)).fillna(1.0)
        elif "MACD" in name_upper and "DIF" in name_upper:
            ema12 = gb_ewm(close_s, 12)
            ema26 = gb_ewm(close_s, 26)
            df[fname] = ema12 - ema26
        elif "VAR" in name_upper:
            nums = [int(s) for s in fname.split("_") if s.isdigit()]
            w = nums[-1] if nums else 20
            ret = gb_pct_change(close_s)
            df[fname] = ret.groupby(df["code"]).transform(lambda x: x.rolling(w, min_periods=5).var())
        elif "BETA" in name_upper:
            ret = gb_pct_change(close_s)
            mkt_ret = ret.groupby(df["date"]).transform("mean")
            roll_Er = gb_rolling(ret, 60, "mean")
            roll_Em = gb_rolling(mkt_ret, 60, "mean")
            roll_Erm = gb_rolling(ret * mkt_ret, 60, "mean")
            roll_Vm = gb_rolling(mkt_ret, 60, "var")
            df[fname] = ((roll_Erm - roll_Er * roll_Em) / (roll_Vm + 1e-12)).fillna(0)
        else:
            df[fname] = 0.0
        computed.add(fname)

    # === Alpha101 因子 ===
    for fname in factor_names:
        if fname in computed:
            continue
        src = factor_selections.get(fname, ("?", "?", ""))
        lib = src[0]
        if lib != "Alpha101":
            continue

        if "ALPHA001" in fname.upper() or "ALPHA001" == fname:
            # -Corr(rank(delta(log(vol),1)), rank((close-open)/open), 6)
            log_vol = np.log(vol_s + 1)
            d_log_vol = gb_diff(log_vol)
            rank_dv = d_log_vol.groupby(df["code"]).rank(pct=True)
            intraday = (close_s - open_s) / (open_s + 1e-12)
            rank_ir = intraday.groupby(df["code"]).rank(pct=True)
            # 向量化滚动corr
            def vec_rolling_corr(a, b, w):
                ma = gb_rolling(a, w, "mean")
                mb = gb_rolling(b, w, "mean")
                sa = gb_rolling(a, w, "std")
                sb = gb_rolling(b, w, "std")
                mab = gb_rolling(a * b, w, "mean")
                cov = mab - ma * mb
                return (cov / (sa * sb + 1e-12)).fillna(0)
            df[fname] = -1.0 * vec_rolling_corr(rank_dv, rank_ir, 6)
        elif "ALPHA012" in fname.upper() or "ALPHA012" == fname:
            dv = gb_diff(vol_s)
            dc = gb_diff(close_s)
            df[fname] = np.sign(dv) * (-1.0 * dc)
        elif "ALPHA028" in fname.upper() or "ALPHA028" == fname:
            adv20 = gb_rolling(vol_s, 20, "mean")
            corr_adv = vec_rolling_corr(adv20, low_s, 5) if 'vec_rolling_corr' in dir() else gb_rolling(adv20 * low_s, 5, "mean")
            df[fname] = (high_s + low_s) / 2.0 - close_s + corr_adv
        elif "ALPHA046" in fname.upper() or "ALPHA046" == fname:
            shifted = gb_shift(close_s, 20)
            df[fname] = close_s / (shifted + 1e-12) - 1.0
        elif "ALPHA083" in fname.upper() or "ALPHA083" == fname:
            df[fname] = close_s.groupby(df["code"]).rank(pct=True)
        else:
            df[fname] = 0.0
        computed.add(fname)

    # === GTJA191 因子 ===
    for fname in factor_names:
        if fname in computed:
            continue
        src = factor_selections.get(fname, ("?", "?", ""))
        lib = src[0]
        if lib != "GTJA191":
            continue

        if "GTJA001" in fname.upper() or "GTJA001" == fname:
            # 与 ALPHA001 相同
            if "ALPHA001" in df.columns:
                df[fname] = df["ALPHA001"]
            else:
                df[fname] = 0.0
        elif "GTJA032" in fname.upper() or "GTJA032" == fname:
            ma7 = gb_rolling(close_s, 7, "mean")
            df[fname] = (ma7 - close_s) / (close_s + 1e-12)
        elif "GTJA052" in fname.upper() or "GTJA052" == fname:
            min_low = gb_rolling(low_s, 20, "min")
            df[fname] = (close_s - min_low) / (min_low + 1e-12)
        elif "GTJA101" in fname.upper() or "GTJA101" == fname:
            df[fname] = (close_s - open_s) / (high_s - low_s + 0.001)
        elif "GTJA155" in fname.upper() or "GTJA155" == fname:
            ma10 = gb_rolling(close_s, 10, "mean")
            std10 = gb_rolling(close_s, 10, "std")
            df[fname] = (ma10 + std10) / (close_s + 1e-12)
        else:
            df[fname] = 0.0
        computed.add(fname)

    # === TDXGS 因子 ===
    for fname in factor_names:
        if fname in computed:
            continue
        src = factor_selections.get(fname, ("?", "?", ""))
        lib = src[0]
        if lib != "TDXGS":
            continue
        name_upper = fname.upper()

        if "RSI" in name_upper:
            nums = [int(s) for s in fname.split("_") if s.isdigit()]
            n = nums[-1] if nums else 14
            delta = gb_diff(close_s)
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)
            # 中国式SMA: ewm(alpha=1/N)
            avg_gain = gain.groupby(df["code"]).transform(lambda x: x.ewm(alpha=1.0/n, adjust=False).mean())
            avg_loss = loss.groupby(df["code"]).transform(lambda x: x.ewm(alpha=1.0/n, adjust=False).mean())
            rs = avg_gain / (avg_loss + 1e-12)
            df[fname] = 100.0 - 100.0 / (1.0 + rs)
        elif "CCI" in name_upper:
            tp = (high_s + low_s + close_s) / 3.0
            ma_tp = gb_rolling(tp, 20, "mean")
            mad_tp = (tp - ma_tp).abs().groupby(df["code"]).transform(lambda x: x.rolling(20, min_periods=5).mean())
            df[fname] = (tp - ma_tp) / (0.015 * mad_tp + 1e-12)
        elif "ATR" in name_upper:
            nums = [int(s) for s in fname.split("_") if s.isdigit()]
            n = nums[-1] if nums else 14
            tr1 = high_s - low_s
            tr2 = abs(high_s - gb_shift(close_s, 1))
            tr3 = abs(low_s - gb_shift(close_s, 1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.groupby(df["code"]).transform(lambda x: x.rolling(n, min_periods=3).mean())
            df[fname] = atr / (close_s + 1e-12)
        elif "BOLL" in name_upper and "UP" in name_upper:
            ma20 = gb_rolling(close_s, 20, "mean")
            std20 = gb_rolling(close_s, 20, "std")
            df[fname] = (ma20 + 2.0 * std20) / (close_s + 1e-12)
        elif "EMA" in name_upper:
            nums = [int(s) for s in fname.split("_") if s.isdigit()]
            n = nums[-1] if nums else 20
            ema = gb_ewm(close_s, n)
            df[fname] = ema / (close_s + 1e-12)
        else:
            df[fname] = 0.0
        computed.add(fname)

    print(f"  因子计算完成! 耗时 {time.time()-t0:.1f}s (共{len(computed)}个因子)", flush=True)
    return df


def get_factor_info(factor_selections):
    """获取因子信息字典，兼容旧接口。

    Returns
    -------
    FACTOR_30 : dict
        {factor_name: (library, category, description)}
    """
    return factor_selections
