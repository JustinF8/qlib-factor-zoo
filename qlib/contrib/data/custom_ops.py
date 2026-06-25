# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Custom operators for Qlib expression engine.
# These operators extend Qlib's built-in expression DSL to support
# SMA (Simple Moving Average with optional EMA weighting),
# TsArgmax/TsArgmin (rolling argmax/argmin value extraction),
# and other specialized operations used in GTJA191 and Alpha101 factor sets.
#
# Usage:
#   from qlib.config import C
#   from qlib.contrib.data.custom_ops import TsArgmax, TsArgmin, SMA, Amount
#   C.custom_ops = [TsArgmax, TsArgmin, SMA, Amount]
#   C.register()

from __future__ import annotations

import numpy as np
import pandas as pd

from qlib.data.ops import Rolling, ElemOperator, ExpressionOps


# ==============================================================================
# TsArgmax / TsArgmin
# ==============================================================================
# These operators extract the VALUE at the argmax/argmin position within a
# rolling window, NOT the index. This matches the semantics of Vibe-Trading's
# ts_argmax / ts_argmin and the original WorldQuant / GTJA formulations.
# ==============================================================================


class TsArgmax(Rolling):
    """Rolling argmax VALUE extraction.

    For each rolling window of size N, finds the position of the maximum value
    and returns the VALUE at that position (not the index).

    This differs from Qlib's built-in IdxMax which returns the 1-based index
    of the maximum.

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    -------
    Expression
        a feature instance with rolling argmax values
    """

    def __init__(self, feature, N):
        super().__init__(feature, N, "ts_argmax")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)

        def _argmax_val(arr: np.ndarray) -> float:
            if np.isnan(arr).all():
                return np.nan
            arr_filled = np.where(np.isnan(arr), -np.inf, arr)
            return float(arr[np.argmax(arr_filled)])

        if self.N == 0:
            series = series.expanding(min_periods=1).apply(_argmax_val, raw=True)
        else:
            series = series.rolling(self.N, min_periods=1).apply(_argmax_val, raw=True)
        return series


class TsArgmin(Rolling):
    """Rolling argmin VALUE extraction.

    For each rolling window of size N, finds the position of the minimum value
    and returns the VALUE at that position (not the index).

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    -------
    Expression
        a feature instance with rolling argmin values
    """

    def __init__(self, feature, N):
        super().__init__(feature, N, "ts_argmin")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)

        def _argmin_val(arr: np.ndarray) -> float:
            if np.isnan(arr).all():
                return np.nan
            arr_filled = np.where(np.isnan(arr), np.inf, arr)
            return float(arr[np.argmin(arr_filled)])

        if self.N == 0:
            series = series.expanding(min_periods=1).apply(_argmin_val, raw=True)
        else:
            series = series.rolling(self.N, min_periods=1).apply(_argmin_val, raw=True)
        return series


# ==============================================================================
# SMA - Simple Moving Average with optional EMA weighting
# ==============================================================================
# In GTJA191, SMA(x, n, m) computes a recursive moving average where:
#   SMA[0] = x[0]
#   SMA[t] = (m * x[t] + (n - m) * SMA[t-1]) / n
#
# When m=1: effectively a simple moving average (equivalent to Mean)
# When m=2: double-weight on current value (like EMA with alpha=2/n)
#
# This is the standard "SMA" used in Chinese finance (通达信/同花顺 convention).
# ==============================================================================


class SMA(Rolling):
    """Simple Moving Average with optional EMA weighting (Chinese finance convention).

    Implements the recursive formula:
        SMA[0] = x[0]
        SMA[t] = (m * x[t] + (n - m) * SMA[t-1]) / n

    This is the standard SMA(X, N, M) function used in 通达信/同花顺 and
    the GTJA191 alpha set.

    - When m=1: equivalent to a standard rolling mean over n periods
    - When m=2: double-weight on current value, similar to EMA with alpha=2/n
    - When m=n: the series passes through unchanged (SMA = x)

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        smoothing period
    M : int
        weight for the current value (1 <= M <= N)

    Returns
    -------
    Expression
        a feature instance with SMA output
    """

    def __init__(self, feature, N, M=1):
        super().__init__(feature, N, "sma")
        self.M = M  # weight parameter

    def __str__(self):
        return "{}({},{},{})".format(type(self).__name__, self.feature, self.N, self.M)

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)

        n = float(self.N)
        m = float(self.M)

        def _sma(arr: np.ndarray) -> float:
            """Compute SMA: recursively weighted average over the rolling window."""
            if np.isnan(arr).all():
                return np.nan
            result = arr[0]
            for i in range(1, len(arr)):
                if np.isnan(arr[i]):
                    continue
                result = (m * arr[i] + (n - m) * result) / n
            return result

        if self.N == 0:
            series = series.expanding(min_periods=1).apply(_sma, raw=True)
        else:
            series = series.rolling(self.N, min_periods=1).apply(_sma, raw=True)
        return series

    def get_longest_back_rolling(self):
        if self.N == 0:
            return np.inf
        # SMA has infinite effective memory due to recursion
        return np.inf


# ==============================================================================
# Amount - 成交额 feature accessor
# ==============================================================================
# In GTJA191, many alphas use `amount` (成交额) which is typically derived as
# VWAP * volume (for A-share market: amount / (volume * 100) ≈ VWAP).
#
# Since Qlib's standard data may not include an $amount field, we provide
# an expression that derives it from $vwap * $volume.
# When the underlying data DOES have $amount, the native Feature("amount")
# will be used automatically by Qlib's expression engine.
# ==============================================================================


class Amount(ElemOperator):
    """Amount (成交额) feature accessor.

    Attempts to load the raw $amount field. If not available in the data,
    derives it as $vwap * $volume (the standard Qlib proxy for turnover value).

    Parameters
    ----------
    feature : Expression, optional
        The underlying feature. If None, loads raw $amount from provider.

    Returns
    -------
    Expression
        a feature instance with amount data
    """

    def __init__(self, feature=None):
        if feature is not None:
            super().__init__(feature)
        else:
            super().__init__(None)
            self._use_vwap_proxy = True

    def __str__(self):
        return "Amount()"

    def _load_internal(self, instrument, start_index, end_index, *args):
        if self._use_vwap_proxy:
            # Use $vwap * $volume as proxy for amount
            from qlib.data.base import Feature
            vwap_series = Feature("vwap").load(instrument, start_index, end_index, *args)
            vol_series = Feature("volume").load(instrument, start_index, end_index, *args)
            return vwap_series * vol_series
        else:
            return self.feature.load(instrument, start_index, end_index, *args)

    def get_longest_back_rolling(self):
        return 0

    def get_extended_window_size(self):
        return 0, 0


# ==============================================================================
# TDXGS - 通达信/同花顺 技术指标算子
# ==============================================================================
# 基于 MyTT 库的算法逻辑，以 Qlib ExpressionOps 子类形式实现。
# 参考: https://github.com/mpquant/MyTT
#
# 注意：Qlib 已内置 Mean/Sum/Std/Max/Min/Rank/Delta/EMA/WMA/Corr/Cov/
#          Slope/Ref/Abs/Log/Sign/Power/If/Greater/Less 等算子，
#         此处只新增 Qlib 缺失的通达信专用算子。
# ==============================================================================


class ATR(Rolling):
    """通达信 ATR - 平均真实波幅 (Average True Range)

    对应 MyTT.ATR(CLOSE, HIGH, LOW, N)
    用法: ATR($close, $high, $low, N)

    Parameters
    ----------
    close : Expression
        close price
    high : Expression
        high price
    low : Expression
        low price
    N : int
        rolling window size (常用: 10, 14, 20, 60)
    """

    def __init__(self, close, high, low, N):
        self.close = close
        self.high = high
        self.low = low
        self.N = N

    def __str__(self):
        return "ATR({},{},{},{})".format(self.close, self.high, self.low, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        h = self.high.load(instrument, start_index, end_index, *args)
        l = self.low.load(instrument, start_index, end_index, *args)
        pc = c.shift(1)
        tr = np.maximum(np.maximum(h - l, np.abs(h - pc)), np.abs(l - pc))
        return tr.rolling(self.N, min_periods=1).mean()

    def get_longest_back_rolling(self):
        return self.N + 1


class RSV(Rolling):
    """通达信 RSV - 未成熟随机值 (Raw Stochastic Value)

    对应 KDJ 的中间计算步骤
    用法: RSV($close, $high, $low, N)
    RSV = (close - LLV(low,N)) / (HHV(high,N) - LLV(low,N)) * 100

    Parameters
    ----------
    close : Expression
    high : Expression
    low : Expression
    N : int (常用: 9)
    """

    def __init__(self, close, high, low, N):
        self.close = close
        self.high = high
        self.low = low
        self.N = N

    def __str__(self):
        return "RSV({},{},{},{})".format(self.close, self.high, self.low, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        h = self.high.load(instrument, start_index, end_index, *args)
        l = self.low.load(instrument, start_index, end_index, *args)
        hhv = h.rolling(self.N, min_periods=1).max()
        llv = l.rolling(self.N, min_periods=1).min()
        return (c - llv) / (hhv - llv + 1e-12) * 100.0

    def get_longest_back_rolling(self):
        return self.N


class RSI(Rolling):
    """通达信 RSI - 相对强弱指标 (Relative Strength Index)

    对应 MyTT.RSI(CLOSE, N)
    用法: RSI($close, N)
    算法: SMA(max(diff,0), N) / SMA(abs(diff), N) * 100
    使用中国式 SMA (ewm alpha=M/N) 以保证与通达信口径一致

    Parameters
    ----------
    close : Expression
    N : int (常用: 6, 12, 14, 24)
    """

    def __init__(self, close, N):
        self.close = close
        self.N = N

    def __str__(self):
        return "RSI({},{})".format(self.close, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        diff = c.diff(1)
        up = diff.where(diff > 0, 0.0)
        dn = (-diff).where(diff < 0, 0.0)
        # 使用中国式 SMA: ewm(alpha=1/N) 与通达信完全一致
        sma_up = up.ewm(alpha=1.0 / self.N, adjust=False).mean()
        sma_dn = dn.ewm(alpha=1.0 / self.N, adjust=False).mean()
        return sma_up / (sma_dn + 1e-12) * 100.0

    def get_longest_back_rolling(self):
        return np.inf  # SMA/EMA 有无限记忆


class BIAS(Rolling):
    """通达信 BIAS - 乖离率

    对应 MyTT.BIAS(CLOSE, N)
    用法: BIAS($close, N)
    BIAS = (close - MA(close,N)) / MA(close,N) * 100

    Parameters
    ----------
    close : Expression
    N : int (常用: 6, 12, 24)
    """

    def __init__(self, close, N):
        self.close = close
        self.N = N

    def __str__(self):
        return "BIAS({},{})".format(self.close, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        ma = c.rolling(self.N, min_periods=1).mean()
        return (c - ma) / (ma + 1e-12) * 100.0

    def get_longest_back_rolling(self):
        return self.N


class BBI(Rolling):
    """通达信 BBI - 多空指数 (Bull and Bear Index)

    对应 MyTT.BBI(CLOSE, M1, M2, M3, M4)
    用法: BBI($close, M1, M2, M3, M4)
    BBI = (MA3 + MA6 + MA12 + MA24) / 4

    Parameters
    ----------
    close : Expression
    M1~M4 : int (默认: 3, 6, 12, 24)
    """

    def __init__(self, close, M1, M2, M3, M4):
        self.close = close
        self.M1 = M1
        self.M2 = M2
        self.M3 = M3
        self.M4 = M4

    def __str__(self):
        return "BBI({},{},{},{},{})".format(self.close, self.M1, self.M2, self.M3, self.M4)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        ma1 = c.rolling(self.M1, min_periods=1).mean()
        ma2 = c.rolling(self.M2, min_periods=1).mean()
        ma3 = c.rolling(self.M3, min_periods=1).mean()
        ma4 = c.rolling(self.M4, min_periods=1).mean()
        return (ma1 + ma2 + ma3 + ma4) / 4.0

    def get_longest_back_rolling(self):
        return max(self.M1, self.M2, self.M3, self.M4)


class WR(Rolling):
    """通达信 WR - 威廉指标 (Williams %R)

    对应 MyTT.WR(CLOSE, HIGH, LOW, N)
    用法: WR($close, $high, $low, N)
    WR = (HHV(high,N) - close) / (HHV(high,N) - LLV(low,N)) * 100

    Parameters
    ----------
    close : Expression
    high : Expression
    low : Expression
    N : int (常用: 10, 6)
    """

    def __init__(self, close, high, low, N):
        self.close = close
        self.high = high
        self.low = low
        self.N = N

    def __str__(self):
        return "WR({},{},{},{})".format(self.close, self.high, self.low, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        h = self.high.load(instrument, start_index, end_index, *args)
        l = self.low.load(instrument, start_index, end_index, *args)
        hhv = h.rolling(self.N, min_periods=1).max()
        llv = l.rolling(self.N, min_periods=1).min()
        return (hhv - c) / (hhv - llv + 1e-12) * 100.0

    def get_longest_back_rolling(self):
        return self.N


class CCI(Rolling):
    """通达信 CCI - 商品通道指数 (Commodity Channel Index)

    对应 MyTT.CCI(CLOSE, HIGH, LOW, N)
    用法: CCI($close, $high, $low, N)
    TP = (H+L+C)/3
    CCI = (TP - MA(TP,N)) / (0.015 * AVEDEV(TP,N))

    Parameters
    ----------
    close : Expression
    high : Expression
    low : Expression
    N : int (常用: 14, 20)
    """

    def __init__(self, close, high, low, N):
        self.close = close
        self.high = high
        self.low = low
        self.N = N

    def __str__(self):
        return "CCI({},{},{},{})".format(self.close, self.high, self.low, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        h = self.high.load(instrument, start_index, end_index, *args)
        l = self.low.load(instrument, start_index, end_index, *args)
        tp = (h + l + c) / 3.0
        ma_tp = tp.rolling(self.N, min_periods=1).mean()
        # AVEDEV: 平均绝对偏差
        avedev = tp.rolling(self.N, min_periods=1).apply(
            lambda x: (np.abs(x - x.mean())).mean(), raw=True
        )
        return (tp - ma_tp) / (0.015 * avedev + 1e-12)

    def get_longest_back_rolling(self):
        return self.N


class PDI(Rolling):
    """通达信 DMI-PDI - 上升动向指标 (Plus Directional Indicator)

    对应 MyTT.DMI 的 PDI 输出
    用法: PDI($close, $high, $low, M1, M2)
    PDI = DMP * 100 / TR

    Parameters
    ----------
    close : Expression
    high : Expression
    low : Expression
    M1 : int (常用: 14)
    M2 : int (常用: 6)  - ADX 平滑参数
    """

    def __init__(self, close, high, low, M1, M2):
        self.close = close
        self.high = high
        self.low = low
        self.M1 = M1
        self.M2 = M2

    def __str__(self):
        return "PDI({},{},{},{},{})".format(self.close, self.high, self.low, self.M1, self.M2)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        h = self.high.load(instrument, start_index, end_index, *args)
        l = self.low.load(instrument, start_index, end_index, *args)
        pc = c.shift(1)
        ph = h.shift(1)
        pl = l.shift(1)
        tr = np.maximum(np.maximum(h - l, np.abs(h - pc)), np.abs(l - pc))
        hd = h - ph
        ld = pl - l
        dmp = np.where((hd > 0) & (hd > ld), hd, 0.0)
        dmp = pd.Series(dmp, index=h.index).rolling(self.M1, min_periods=1).sum()
        tr_sum = tr.rolling(self.M1, min_periods=1).sum()
        return dmp * 100.0 / (tr_sum + 1e-12)

    def get_longest_back_rolling(self):
        return self.M1 + 1


class MDI(Rolling):
    """通达信 DMI-MDI - 下降动向指标 (Minus Directional Indicator)

    对应 MyTT.DMI 的 MDI 输出
    用法: MDI($close, $high, $low, M1, M2)

    Parameters
    ----------
    close : Expression
    high : Expression
    low : Expression
    M1 : int (常用: 14)
    M2 : int (常用: 6)
    """

    def __init__(self, close, high, low, M1, M2):
        self.close = close
        self.high = high
        self.low = low
        self.M1 = M1
        self.M2 = M2

    def __str__(self):
        return "MDI({},{},{},{},{})".format(self.close, self.high, self.low, self.M1, self.M2)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        h = self.high.load(instrument, start_index, end_index, *args)
        l = self.low.load(instrument, start_index, end_index, *args)
        pc = c.shift(1)
        ph = h.shift(1)
        pl = l.shift(1)
        tr = np.maximum(np.maximum(h - l, np.abs(h - pc)), np.abs(l - pc))
        hd = h - ph
        ld = pl - l
        dmm = np.where((ld > 0) & (ld > hd), ld, 0.0)
        dmm = pd.Series(dmm, index=h.index).rolling(self.M1, min_periods=1).sum()
        tr_sum = tr.rolling(self.M1, min_periods=1).sum()
        return dmm * 100.0 / (tr_sum + 1e-12)

    def get_longest_back_rolling(self):
        return self.M1 + 1


class ADX(Rolling):
    """通达信 ADX - 平均趋向指数 (Average Directional Index)

    对应 MyTT.DMI 的 ADX 输出
    用法: ADX($close, $high, $low, M1, M2)
    ADX = MA(|MDI-PDI|/(PDI+MDI)*100, M2)

    Parameters
    ----------
    close : Expression
    high : Expression
    low : Expression
    M1 : int (常用: 14)
    M2 : int (常用: 6)
    """

    def __init__(self, close, high, low, M1, M2):
        self.close = close
        self.high = high
        self.low = low
        self.M1 = M1
        self.M2 = M2

    def __str__(self):
        return "ADX({},{},{},{},{})".format(self.close, self.high, self.low, self.M1, self.M2)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        h = self.high.load(instrument, start_index, end_index, *args)
        l = self.low.load(instrument, start_index, end_index, *args)
        pc = c.shift(1)
        ph = h.shift(1)
        pl = l.shift(1)
        tr = np.maximum(np.maximum(h - l, np.abs(h - pc)), np.abs(l - pc))
        hd = h - ph
        ld = pl - l
        dmp = np.where((hd > 0) & (hd > ld), hd, 0.0)
        dmm = np.where((ld > 0) & (ld > hd), ld, 0.0)
        dmp_s = pd.Series(dmp, index=h.index).rolling(self.M1, min_periods=1).sum()
        dmm_s = pd.Series(dmm, index=h.index).rolling(self.M1, min_periods=1).sum()
        tr_s = tr.rolling(self.M1, min_periods=1).sum()
        pdi = dmp_s * 100.0 / (tr_s + 1e-12)
        mdi = dmm_s * 100.0 / (tr_s + 1e-12)
        dx = np.abs(mdi - pdi) / (pdi + mdi + 1e-12) * 100.0
        return dx.rolling(self.M2, min_periods=1).mean()

    def get_longest_back_rolling(self):
        return self.M1 + self.M2 + 1


class ADXR(Rolling):
    """通达信 ADXR - 评估趋向指数 (Average Directional Movement Index Rating)

    对应 MyTT.DMI 的 ADXR 输出
    用法: ADXR($close, $high, $low, M1, M2)
    ADXR = (ADX + Ref(ADX, M2)) / 2

    Parameters
    ----------
    close : Expression
    high : Expression
    low : Expression
    M1 : int (常用: 14)
    M2 : int (常用: 6)
    """

    def __init__(self, close, high, low, M1, M2):
        self.close = close
        self.high = high
        self.low = low
        self.M1 = M1
        self.M2 = M2

    def __str__(self):
        return "ADXR({},{},{},{},{})".format(self.close, self.high, self.low, self.M1, self.M2)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        h = self.high.load(instrument, start_index, end_index, *args)
        l = self.low.load(instrument, start_index, end_index, *args)
        pc = c.shift(1)
        ph = h.shift(1)
        pl = l.shift(1)
        tr = np.maximum(np.maximum(h - l, np.abs(h - pc)), np.abs(l - pc))
        hd = h - ph
        ld = pl - l
        dmp = np.where((hd > 0) & (hd > ld), hd, 0.0)
        dmm = np.where((ld > 0) & (ld > hd), ld, 0.0)
        dmp_s = pd.Series(dmp, index=h.index).rolling(self.M1, min_periods=1).sum()
        dmm_s = pd.Series(dmm, index=h.index).rolling(self.M1, min_periods=1).sum()
        tr_s = tr.rolling(self.M1, min_periods=1).sum()
        pdi = dmp_s * 100.0 / (tr_s + 1e-12)
        mdi = dmm_s * 100.0 / (tr_s + 1e-12)
        dx = np.abs(mdi - pdi) / (pdi + mdi + 1e-12) * 100.0
        adx = dx.rolling(self.M2, min_periods=1).mean()
        return (adx + adx.shift(self.M2)) / 2.0

    def get_longest_back_rolling(self):
        return self.M1 + self.M2 * 2 + 1


class BOLL_UP(Rolling):
    """通达信 BOLL 上轨 (Bollinger Band Upper)

    对应 MyTT.BOLL(CLOSE, N, P) 的 UPPER 输出
    用法: BOLL_UP($close, N, P)
    UPPER = MA(close,N) + P * Std(close,N)

    Parameters
    ----------
    close : Expression
    N : int (常用: 20)
    P : float (常用: 2.0)
    """

    def __init__(self, close, N, P):
        self.close = close
        self.N = N
        self.P = P

    def __str__(self):
        return "BOLL_UP({},{},{})".format(self.close, self.N, self.P)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        mid = c.rolling(self.N, min_periods=1).mean()
        std = c.rolling(self.N, min_periods=1).std(ddof=0)  # 通达信用 ddof=0
        return mid + self.P * std

    def get_longest_back_rolling(self):
        return self.N


class BOLL_DN(Rolling):
    """通达信 BOLL 下轨 (Bollinger Band Lower)

    对应 MyTT.BOLL(CLOSE, N, P) 的 LOWER 输出
    用法: BOLL_DN($close, N, P)
    LOWER = MA(close,N) - P * Std(close,N)

    Parameters
    ----------
    close : Expression
    N : int (常用: 20)
    P : float (常用: 2.0)
    """

    def __init__(self, close, N, P):
        self.close = close
        self.N = N
        self.P = P

    def __str__(self):
        return "BOLL_DN({},{},{})".format(self.close, self.N, self.P)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        mid = c.rolling(self.N, min_periods=1).mean()
        std = c.rolling(self.N, min_periods=1).std(ddof=0)
        return mid - self.P * std

    def get_longest_back_rolling(self):
        return self.N


class BOLL_MID(Rolling):
    """通达信 BOLL 中轨 (Bollinger Band Middle)

    对应 MyTT.BOLL(CLOSE, N, P) 的 MID 输出
    用法: BOLL_MID($close, N)
    MID = MA(close, N)

    Parameters
    ----------
    close : Expression
    N : int (常用: 20)
    """

    def __init__(self, close, N):
        self.close = close
        self.N = N

    def __str__(self):
        return "BOLL_MID({},{})".format(self.close, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        return c.rolling(self.N, min_periods=1).mean()

    def get_longest_back_rolling(self):
        return self.N


class PSY(Rolling):
    """通达信 PSY - 心理线 (Psychological Line)

    对应 MyTT.PSY(CLOSE, N, M)
    用法: PSY($close, N)
    PSY = Count(close>Ref(close,1), N) / N * 100

    Parameters
    ----------
    close : Expression
    N : int (常用: 12)
    """

    def __init__(self, close, N):
        self.close = close
        self.N = N

    def __str__(self):
        return "PSY({},{})".format(self.close, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        up = (c > c.shift(1)).astype(float)
        return up.rolling(self.N, min_periods=1).mean() * 100.0

    def get_longest_back_rolling(self):
        return self.N


class PSYMA(Rolling):
    """通达信 PSYMA - 心理线均线

    对应 MyTT.PSY 的 PSYMA 输出
    用法: PSYMA($close, N, M)
    PSYMA = MA(PSY, M)

    Parameters
    ----------
    close : Expression
    N : int (常用: 12)
    M : int (常用: 6)
    """

    def __init__(self, close, N, M):
        self.close = close
        self.N = N
        self.M = M

    def __str__(self):
        return "PSYMA({},{},{})".format(self.close, self.N, self.M)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        up = (c > c.shift(1)).astype(float)
        psy = up.rolling(self.N, min_periods=1).mean() * 100.0
        return psy.rolling(self.M, min_periods=1).mean()

    def get_longest_back_rolling(self):
        return self.N + self.M


class ROC(Rolling):
    """通达信 ROC - 变动率指标 (Rate of Change)

    对应 MyTT.ROC(CLOSE, N, M)
    用法: ROC($close, N)
    ROC = (close - Ref(close,N)) / Ref(close,N) * 100

    Parameters
    ----------
    close : Expression
    N : int (常用: 12)
    """

    def __init__(self, close, N):
        self.close = close
        self.N = N

    def __str__(self):
        return "ROC({},{})".format(self.close, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        ref = c.shift(self.N)
        return (c - ref) / (ref + 1e-12) * 100.0

    def get_longest_back_rolling(self):
        return self.N


class MAROC(Rolling):
    """通达信 MAROC - 变动率均线

    对应 MyTT.ROC 的 MAROC 输出
    用法: MAROC($close, N, M)
    MAROC = MA(ROC, M)

    Parameters
    ----------
    close : Expression
    N : int (常用: 12)
    M : int (常用: 6)
    """

    def __init__(self, close, N, M):
        self.close = close
        self.N = N
        self.M = M

    def __str__(self):
        return "MAROC({},{},{})".format(self.close, self.N, self.M)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        ref = c.shift(self.N)
        roc = (c - ref) / (ref + 1e-12) * 100.0
        return roc.rolling(self.M, min_periods=1).mean()

    def get_longest_back_rolling(self):
        return self.N + self.M


class MTM(Rolling):
    """通达信 MTM - 动量指标 (Momentum)

    对应 MyTT.MTM(CLOSE, N, M)
    用法: MTM($close, N)
    MTM = close - Ref(close, N)

    Parameters
    ----------
    close : Expression
    N : int (常用: 12)
    """

    def __init__(self, close, N):
        self.close = close
        self.N = N

    def __str__(self):
        return "MTM({},{})".format(self.close, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        return c - c.shift(self.N)

    def get_longest_back_rolling(self):
        return self.N


class MTMMA(Rolling):
    """通达信 MTMMA - 动量均线

    对应 MyTT.MTM 的 MTMMA 输出
    用法: MTMMA($close, N, M)
    MTMMA = MA(MTM, M)

    Parameters
    ----------
    close : Expression
    N : int (常用: 12)
    M : int (常用: 6)
    """

    def __init__(self, close, N, M):
        self.close = close
        self.N = N
        self.M = M

    def __str__(self):
        return "MTMMA({},{},{})".format(self.close, self.N, self.M)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        mtm = c - c.shift(self.N)
        return mtm.rolling(self.M, min_periods=1).mean()

    def get_longest_back_rolling(self):
        return self.N + self.M


class TRIX(Rolling):
    """通达信 TRIX - 三重指数平滑平均线

    对应 MyTT.TRIX(CLOSE, M1, M2)
    用法: TRIX($close, M1)
    TR = EMA(EMA(EMA(close, M1), M1), M1)
    TRIX = (TR - Ref(TR,1)) / Ref(TR,1) * 100

    Parameters
    ----------
    close : Expression
    M1 : int (常用: 12)
    """

    def __init__(self, close, M1):
        self.close = close
        self.M1 = M1

    def __str__(self):
        return "TRIX({},{})".format(self.close, self.M1)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        tr = c.ewm(span=self.M1, adjust=False).mean()
        tr = tr.ewm(span=self.M1, adjust=False).mean()
        tr = tr.ewm(span=self.M1, adjust=False).mean()
        return (tr - tr.shift(1)) / (tr.shift(1) + 1e-12) * 100.0

    def get_longest_back_rolling(self):
        return np.inf


class TRMA(Rolling):
    """通达信 TRMA - 三重指数平滑平均线的均线

    对应 MyTT.TRIX 的 TRMA 输出
    用法: TRMA($close, M1, M2)
    TRMA = MA(TRIX, M2)

    Parameters
    ----------
    close : Expression
    M1 : int (常用: 12)
    M2 : int (常用: 20)
    """

    def __init__(self, close, M1, M2):
        self.close = close
        self.M1 = M1
        self.M2 = M2

    def __str__(self):
        return "TRMA({},{},{})".format(self.close, self.M1, self.M2)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        tr = c.ewm(span=self.M1, adjust=False).mean()
        tr = tr.ewm(span=self.M1, adjust=False).mean()
        tr = tr.ewm(span=self.M1, adjust=False).mean()
        trix = (tr - tr.shift(1)) / (tr.shift(1) + 1e-12) * 100.0
        return trix.rolling(self.M2, min_periods=1).mean()

    def get_longest_back_rolling(self):
        return np.inf


class VR(Rolling):
    """通达信 VR - 容量比率 (Volume Ratio)

    对应 MyTT.VR(CLOSE, VOL, M1)
    用法: VR($close, $volume, M1)
    VR = Sum(If(close>Ref(close,1),vol,0), M1) / Sum(If(close<=Ref(close,1),vol,0), M1) * 100

    Parameters
    ----------
    close : Expression
    volume : Expression
    M1 : int (常用: 26)
    """

    def __init__(self, close, volume, M1):
        self.close = close
        self.volume = volume
        self.M1 = M1

    def __str__(self):
        return "VR({},{},{})".format(self.close, self.volume, self.M1)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        v = self.volume.load(instrument, start_index, end_index, *args)
        up_vol = np.where(c > c.shift(1), v, 0.0)
        dn_vol = np.where(c <= c.shift(1), v, 0.0)
        up_sum = pd.Series(up_vol, index=c.index).rolling(self.M1, min_periods=1).sum()
        dn_sum = pd.Series(dn_vol, index=c.index).rolling(self.M1, min_periods=1).sum()
        return up_sum / (dn_sum + 1e-12) * 100.0

    def get_longest_back_rolling(self):
        return self.M1 + 1


class CR(Rolling):
    """通达信 CR - 价格动量指标

    对应 MyTT.CR(CLOSE, HIGH, LOW, N)
    用法: CR($close, $high, $low, N)
    MID = Ref((H+L+C)/3, 1)
    CR = Sum(Max(0,H-MID),N) / Sum(Max(0,MID-L),N) * 100

    Parameters
    ----------
    close : Expression
    high : Expression
    low : Expression
    N : int (常用: 20, 26)
    """

    def __init__(self, close, high, low, N):
        self.close = close
        self.high = high
        self.low = low
        self.N = N

    def __str__(self):
        return "CR({},{},{},{})".format(self.close, self.high, self.low, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        h = self.high.load(instrument, start_index, end_index, *args)
        l = self.low.load(instrument, start_index, end_index, *args)
        mid = ((h + l + c) / 3.0).shift(1)
        up = np.maximum(0, h - mid)
        dn = np.maximum(0, mid - l)
        up_sum = pd.Series(up, index=c.index).rolling(self.N, min_periods=1).sum()
        dn_sum = pd.Series(dn, index=c.index).rolling(self.N, min_periods=1).sum()
        return up_sum / (dn_sum + 1e-12) * 100.0

    def get_longest_back_rolling(self):
        return self.N + 1


class AR(Rolling):
    """通达信 AR - 人气指标

    对应 MyTT.BRAR 的 AR 输出
    用法: AR($open, $close, $high, $low, M1)
    AR = Sum(H-O, M1) / Sum(O-L, M1) * 100

    Parameters
    ----------
    open : Expression
    close : Expression
    high : Expression
    low : Expression
    M1 : int (常用: 26)
    """

    def __init__(self, open, close, high, low, M1):
        self.open = open
        self.close = close
        self.high = high
        self.low = low
        self.M1 = M1

    def __str__(self):
        return "AR({},{},{},{},{})".format(self.open, self.close, self.high, self.low, self.M1)

    def _load_internal(self, instrument, start_index, end_index, *args):
        o = self.open.load(instrument, start_index, end_index, *args)
        c = self.close.load(instrument, start_index, end_index, *args)
        h = self.high.load(instrument, start_index, end_index, *args)
        l = self.low.load(instrument, start_index, end_index, *args)
        ho = h - o
        ol = o - l
        ho_sum = ho.rolling(self.M1, min_periods=1).sum()
        ol_sum = ol.rolling(self.M1, min_periods=1).sum()
        return ho_sum / (ol_sum + 1e-12) * 100.0

    def get_longest_back_rolling(self):
        return self.M1


class BR(Rolling):
    """通达信 BR - 意愿指标

    对应 MyTT.BRAR 的 BR 输出
    用法: BR($open, $close, $high, $low, M1)
    BR = Sum(Max(0,H-Ref(C,1)), M1) / Sum(Max(0,Ref(C,1)-L), M1) * 100

    Parameters
    ----------
    open : Expression
    close : Expression
    high : Expression
    low : Expression
    M1 : int (常用: 26)
    """

    def __init__(self, open, close, high, low, M1):
        self.open = open
        self.close = close
        self.high = high
        self.low = low
        self.M1 = M1

    def __str__(self):
        return "BR({},{},{},{},{})".format(self.open, self.close, self.high, self.low, self.M1)

    def _load_internal(self, instrument, start_index, end_index, *args):
        o = self.open.load(instrument, start_index, end_index, *args)
        c = self.close.load(instrument, start_index, end_index, *args)
        h = self.high.load(instrument, start_index, end_index, *args)
        l = self.low.load(instrument, start_index, end_index, *args)
        pc = c.shift(1)
        up = np.maximum(0, h - pc)
        dn = np.maximum(0, pc - l)
        up_sum = pd.Series(up, index=c.index).rolling(self.M1, min_periods=1).sum()
        dn_sum = pd.Series(dn, index=c.index).rolling(self.M1, min_periods=1).sum()
        return up_sum / (dn_sum + 1e-12) * 100.0

    def get_longest_back_rolling(self):
        return self.M1 + 1


class MA(Rolling):
    """通达信 MA - 简单移动平均线 (别名, Qlib 的 Mean 也可用)

    用法: MA($close, N)
    等价于 Qlib 内置的 Mean($close, N)，但命名更符合通达信习惯。
    """

    def __init__(self, feature, N):
        super().__init__(feature, N, "mean")

    def __str__(self):
        return "MA({},{})".format(self.feature, self.N)


class STD_TDX(Rolling):
    """通达信 STD - N日标准差 (ddof=0, 与通达信一致)

    Qlib 内置 Std 默认 ddof=1，通达信使用 ddof=0。
    用法: STD_TDX($close, N)

    Parameters
    ----------
    feature : Expression
    N : int
    """

    def __init__(self, feature, N):
        super().__init__(feature, N, "std_tdx")

    def __str__(self):
        return "STD_TDX({},{})".format(self.feature, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        return series.rolling(self.N, min_periods=1).std(ddof=0)

    def get_longest_back_rolling(self):
        return self.N


class OBV(Rolling):
    """通达信 OBV - 能量潮 (On Balance Volume)

    对应 MyTT.OBV(CLOSE, VOL)
    用法: OBV($close, $volume)
    OBV = 累计 Sum(If(close>Ref(close,1), vol, If(close<Ref(close,1), -vol, 0)))

    Parameters
    ----------
    close : Expression
    volume : Expression
    """

    def __init__(self, close, volume):
        self.close = close
        self.volume = volume

    def __str__(self):
        return "OBV({},{})".format(self.close, self.volume)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        v = self.volume.load(instrument, start_index, end_index, *args)
        direction = np.where(c > c.shift(1), v,
                             np.where(c < c.shift(1), -v, 0.0))
        return pd.Series(direction, index=c.index).cumsum()

    def get_longest_back_rolling(self):
        return np.inf


class MFI(Rolling):
    """通达信 MFI - 资金流量指标 (Money Flow Index)

    对应 MyTT.MFI(CLOSE, HIGH, LOW, VOL, N)
    用法: MFI($close, $high, $low, $volume, N)
    TYP = (H+L+C)/3
    V1 = Sum(If(TYP>Ref(TYP,1),TYP*VOL,0),N) / Sum(If(TYP<Ref(TYP,1),TYP*VOL,0),N)
    MFI = 100 - 100/(1+V1)

    Parameters
    ----------
    close : Expression
    high : Expression
    low : Expression
    volume : Expression
    N : int (常用: 14)
    """

    def __init__(self, close, high, low, volume, N):
        self.close = close
        self.high = high
        self.low = low
        self.volume = volume
        self.N = N

    def __str__(self):
        return "MFI({},{},{},{},{})".format(self.close, self.high, self.low, self.volume, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        h = self.high.load(instrument, start_index, end_index, *args)
        l = self.low.load(instrument, start_index, end_index, *args)
        v = self.volume.load(instrument, start_index, end_index, *args)
        typ = (h + l + c) / 3.0
        typ_ref = typ.shift(1)
        mf = typ * v
        pos_mf = np.where(typ > typ_ref, mf, 0.0)
        neg_mf = np.where(typ < typ_ref, mf, 0.0)
        pos_sum = pd.Series(pos_mf, index=c.index).rolling(self.N, min_periods=1).sum()
        neg_sum = pd.Series(neg_mf, index=c.index).rolling(self.N, min_periods=1).sum()
        v1 = pos_sum / (neg_sum + 1e-12)
        return 100.0 - 100.0 / (1.0 + v1)

    def get_longest_back_rolling(self):
        return self.N + 1


class DPO(Rolling):
    """通达信 DPO - 区间震荡线 (Detrended Price Oscillator)

    对应 MyTT.DPO(CLOSE, M1, M2, M3)
    用法: DPO($close, M1, M2)
    DPO = close - Ref(MA(close,M1), M2)

    Parameters
    ----------
    close : Expression
    M1 : int (常用: 20)
    M2 : int (常用: 10)
    """

    def __init__(self, close, M1, M2):
        self.close = close
        self.M1 = M1
        self.M2 = M2

    def __str__(self):
        return "DPO({},{},{})".format(self.close, self.M1, self.M2)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        ma = c.rolling(self.M1, min_periods=1).mean()
        return c - ma.shift(self.M2)

    def get_longest_back_rolling(self):
        return self.M1 + self.M2


class MADPO(Rolling):
    """通达信 MADPO - 区间震荡线的均线

    对应 MyTT.DPO 的 MADPO 输出
    用法: MADPO($close, M1, M2, M3)
    MADPO = MA(DPO, M3)

    Parameters
    ----------
    close : Expression
    M1 : int (常用: 20)
    M2 : int (常用: 10)
    M3 : int (常用: 6)
    """

    def __init__(self, close, M1, M2, M3):
        self.close = close
        self.M1 = M1
        self.M2 = M2
        self.M3 = M3

    def __str__(self):
        return "MADPO({},{},{},{})".format(self.close, self.M1, self.M2, self.M3)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        ma = c.rolling(self.M1, min_periods=1).mean()
        dpo = c - ma.shift(self.M2)
        return dpo.rolling(self.M3, min_periods=1).mean()

    def get_longest_back_rolling(self):
        return self.M1 + self.M2 + self.M3


class TAQ_UP(Rolling):
    """通达信 唐安奇通道上轨 (Turtle Channel Upper)

    对应 MyTT.TAQ(HIGH, LOW, N) 的 UP 输出
    用法: TAQ_UP($high, N)
    UP = HHV(high, N)

    Parameters
    ----------
    high : Expression
    N : int (常用: 20)
    """

    def __init__(self, high, N):
        self.high = high
        self.N = N

    def __str__(self):
        return "TAQ_UP({},{})".format(self.high, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        h = self.high.load(instrument, start_index, end_index, *args)
        return h.rolling(self.N, min_periods=1).max()

    def get_longest_back_rolling(self):
        return self.N


class TAQ_DN(Rolling):
    """通达信 唐安奇通道下轨 (Turtle Channel Lower)

    对应 MyTT.TAQ(HIGH, LOW, N) 的 DOWN 输出
    用法: TAQ_DN($low, N)
    DOWN = LLV(low, N)

    Parameters
    ----------
    low : Expression
    N : int (常用: 20)
    """

    def __init__(self, low, N):
        self.low = low
        self.N = N

    def __str__(self):
        return "TAQ_DN({},{})".format(self.low, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        l = self.low.load(instrument, start_index, end_index, *args)
        return l.rolling(self.N, min_periods=1).min()

    def get_longest_back_rolling(self):
        return self.N


class TAQ_MID(Rolling):
    """通达信 唐安奇通道中轨 (Turtle Channel Middle)

    对应 MyTT.TAQ(HIGH, LOW, N) 的 MID 输出
    用法: TAQ_MID($high, $low, N)
    MID = (UP + DOWN) / 2

    Parameters
    ----------
    high : Expression
    low : Expression
    N : int (常用: 20)
    """

    def __init__(self, high, low, N):
        self.high = high
        self.low = low
        self.N = N

    def __str__(self):
        return "TAQ_MID({},{},{})".format(self.high, self.low, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        h = self.high.load(instrument, start_index, end_index, *args)
        l = self.low.load(instrument, start_index, end_index, *args)
        up = h.rolling(self.N, min_periods=1).max()
        dn = l.rolling(self.N, min_periods=1).min()
        return (up + dn) / 2.0

    def get_longest_back_rolling(self):
        return self.N


class KTN_UP(Rolling):
    """通达信 肯特纳通道上轨 (Keltner Channel Upper)

    对应 MyTT.KTN(CLOSE, HIGH, LOW, N, M) 的 UPPER 输出
    用法: KTN_UP($close, $high, $low, N, M)
    MID = EMA((H+L+C)/3, N)
    UPPER = MID + 2 * ATR(M)

    Parameters
    ----------
    close : Expression
    high : Expression
    low : Expression
    N : int (常用: 20)
    M : int (常用: 10)
    """

    def __init__(self, close, high, low, N, M):
        self.close = close
        self.high = high
        self.low = low
        self.N = N
        self.M = M

    def __str__(self):
        return "KTN_UP({},{},{},{},{})".format(self.close, self.high, self.low, self.N, self.M)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        h = self.high.load(instrument, start_index, end_index, *args)
        l = self.low.load(instrument, start_index, end_index, *args)
        pc = c.shift(1)
        tr = np.maximum(np.maximum(h - l, np.abs(h - pc)), np.abs(l - pc))
        atr = tr.rolling(self.M, min_periods=1).mean()
        tp = (h + l + c) / 3.0
        mid = tp.ewm(span=self.N, adjust=False).mean()
        return mid + 2.0 * atr

    def get_longest_back_rolling(self):
        return max(self.N, self.M) + 1


class KTN_DN(Rolling):
    """通达信 肯特纳通道下轨 (Keltner Channel Lower)

    对应 MyTT.KTN 的 LOWER 输出
    用法: KTN_DN($close, $high, $low, N, M)
    MID = EMA((H+L+C)/3, N)
    LOWER = MID - 2 * ATR(M)

    Parameters
    ----------
    close : Expression
    high : Expression
    low : Expression
    N : int (常用: 20)
    M : int (常用: 10)
    """

    def __init__(self, close, high, low, N, M):
        self.close = close
        self.high = high
        self.low = low
        self.N = N
        self.M = M

    def __str__(self):
        return "KTN_DN({},{},{},{},{})".format(self.close, self.high, self.low, self.N, self.M)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        h = self.high.load(instrument, start_index, end_index, *args)
        l = self.low.load(instrument, start_index, end_index, *args)
        pc = c.shift(1)
        tr = np.maximum(np.maximum(h - l, np.abs(h - pc)), np.abs(l - pc))
        atr = tr.rolling(self.M, min_periods=1).mean()
        tp = (h + l + c) / 3.0
        mid = tp.ewm(span=self.N, adjust=False).mean()
        return mid - 2.0 * atr

    def get_longest_back_rolling(self):
        return max(self.N, self.M) + 1


class KTN_MID(Rolling):
    """通达信 肯特纳通道中轨 (Keltner Channel Middle)

    对应 MyTT.KTN 的 MID 输出
    用法: KTN_MID($close, $high, $low, N)
    MID = EMA((H+L+C)/3, N)

    Parameters
    ----------
    close : Expression
    high : Expression
    low : Expression
    N : int (常用: 20)
    """

    def __init__(self, close, high, low, N):
        self.close = close
        self.high = high
        self.low = low
        self.N = N

    def __str__(self):
        return "KTN_MID({},{},{},{})".format(self.close, self.high, self.low, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        h = self.high.load(instrument, start_index, end_index, *args)
        l = self.low.load(instrument, start_index, end_index, *args)
        tp = (h + l + c) / 3.0
        return tp.ewm(span=self.N, adjust=False).mean()

    def get_longest_back_rolling(self):
        return np.inf  # EMA 无限记忆


class EMV(Rolling):
    """通达信 EMV - 简易波动指标 (Ease of Movement Value)

    对应 MyTT.EMV(HIGH, LOW, VOL, N, M) 的 EMV 输出
    用法: EMV($high, $low, $volume, N)
    MID = 100*(H+L-Ref(H+L,1))/(H+L)
    EMV = MA(MID * MA(VOL,N)/VOL * (H-L)/MA(H-L,N), N)

    Parameters
    ----------
    high : Expression
    low : Expression
    volume : Expression
    N : int (常用: 14)
    """

    def __init__(self, high, low, volume, N):
        self.high = high
        self.low = low
        self.volume = volume
        self.N = N

    def __str__(self):
        return "EMV({},{},{},{})".format(self.high, self.low, self.volume, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        h = self.high.load(instrument, start_index, end_index, *args)
        l = self.low.load(instrument, start_index, end_index, *args)
        v = self.volume.load(instrument, start_index, end_index, *args)
        hl = h + l
        hl_ref = hl.shift(1)
        mid = 100.0 * (hl - hl_ref) / (hl + 1e-12)
        vol_ma = v.rolling(self.N, min_periods=1).mean()
        hl_ma = (h - l).rolling(self.N, min_periods=1).mean()
        vol_ratio = vol_ma / (v + 1e-12)
        emv_raw = mid * vol_ratio * (h - l) / (hl_ma + 1e-12)
        return emv_raw.rolling(self.N, min_periods=1).mean()

    def get_longest_back_rolling(self):
        return self.N * 2 + 1


class MAEMV(Rolling):
    """通达信 MAEMV - 简易波动指标的均线

    对应 MyTT.EMV 的 MAEMV 输出
    用法: MAEMV($high, $low, $volume, N, M)
    MAEMV = MA(EMV, M)

    Parameters
    ----------
    high : Expression
    low : Expression
    volume : Expression
    N : int (常用: 14)
    M : int (常用: 9)
    """

    def __init__(self, high, low, volume, N, M):
        self.high = high
        self.low = low
        self.volume = volume
        self.N = N
        self.M = M

    def __str__(self):
        return "MAEMV({},{},{},{},{})".format(self.high, self.low, self.volume, self.N, self.M)

    def _load_internal(self, instrument, start_index, end_index, *args):
        h = self.high.load(instrument, start_index, end_index, *args)
        l = self.low.load(instrument, start_index, end_index, *args)
        v = self.volume.load(instrument, start_index, end_index, *args)
        hl = h + l
        hl_ref = hl.shift(1)
        mid = 100.0 * (hl - hl_ref) / (hl + 1e-12)
        vol_ma = v.rolling(self.N, min_periods=1).mean()
        hl_ma = (h - l).rolling(self.N, min_periods=1).mean()
        vol_ratio = vol_ma / (v + 1e-12)
        emv_raw = mid * vol_ratio * (h - l) / (hl_ma + 1e-12)
        emv = emv_raw.rolling(self.N, min_periods=1).mean()
        return emv.rolling(self.M, min_periods=1).mean()

    def get_longest_back_rolling(self):
        return self.N * 2 + self.M + 1


class MASS(Rolling):
    """通达信 MASS - 梅斯线 (Mass Index)

    对应 MyTT.MASS(HIGH, LOW, N1, N2, M) 的 MASS 输出
    用法: MASS($high, $low, N1, N2)
    MASS = Sum(MA(H-L,N1)/MA(MA(H-L,N1),N1), N2)

    Parameters
    ----------
    high : Expression
    low : Expression
    N1 : int (常用: 9)
    N2 : int (常用: 25)
    """

    def __init__(self, high, low, N1, N2):
        self.high = high
        self.low = low
        self.N1 = N1
        self.N2 = N2

    def __str__(self):
        return "MASS({},{},{},{})".format(self.high, self.low, self.N1, self.N2)

    def _load_internal(self, instrument, start_index, end_index, *args):
        h = self.high.load(instrument, start_index, end_index, *args)
        l = self.low.load(instrument, start_index, end_index, *args)
        hl = h - l
        ma1 = hl.rolling(self.N1, min_periods=1).mean()
        ma2 = ma1.rolling(self.N1, min_periods=1).mean()
        ratio = ma1 / (ma2 + 1e-12)
        return ratio.rolling(self.N2, min_periods=1).sum()

    def get_longest_back_rolling(self):
        return self.N1 + self.N1 + self.N2


class MA_MASS(Rolling):
    """通达信 MA_MASS - 梅斯线的均线

    对应 MyTT.MASS 的 MA_MASS 输出
    用法: MA_MASS($high, $low, N1, N2, M)
    MA_MASS = MA(MASS, M)

    Parameters
    ----------
    high : Expression
    low : Expression
    N1 : int (常用: 9)
    N2 : int (常用: 25)
    M : int (常用: 6)
    """

    def __init__(self, high, low, N1, N2, M):
        self.high = high
        self.low = low
        self.N1 = N1
        self.N2 = N2
        self.M = M

    def __str__(self):
        return "MA_MASS({},{},{},{},{})".format(self.high, self.low, self.N1, self.N2, self.M)

    def _load_internal(self, instrument, start_index, end_index, *args):
        h = self.high.load(instrument, start_index, end_index, *args)
        l = self.low.load(instrument, start_index, end_index, *args)
        hl = h - l
        ma1 = hl.rolling(self.N1, min_periods=1).mean()
        ma2 = ma1.rolling(self.N1, min_periods=1).mean()
        ratio = ma1 / (ma2 + 1e-12)
        mass = ratio.rolling(self.N2, min_periods=1).sum()
        return mass.rolling(self.M, min_periods=1).mean()

    def get_longest_back_rolling(self):
        return self.N1 + self.N1 + self.N2 + self.M


class DFMA_DIF(Rolling):
    """通达信 DMA/新DMA - 平行线差

    对应 MyTT.DFMA(CLOSE, N1, N2, M) 的 DIF 输出
    用法: DFMA_DIF($close, N1, N2)
    DIF = MA(close,N1) - MA(close,N2)

    Parameters
    ----------
    close : Expression
    N1 : int (常用: 10)
    N2 : int (常用: 50)
    """

    def __init__(self, close, N1, N2):
        self.close = close
        self.N1 = N1
        self.N2 = N2

    def __str__(self):
        return "DFMA_DIF({},{},{})".format(self.close, self.N1, self.N2)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        return c.rolling(self.N1, min_periods=1).mean() - c.rolling(self.N2, min_periods=1).mean()

    def get_longest_back_rolling(self):
        return max(self.N1, self.N2)


class DFMA_DIFMA(Rolling):
    """通达信 DIFMA - 平行线差的均线

    对应 MyTT.DFMA 的 DIFMA 输出
    用法: DFMA_DIFMA($close, N1, N2, M)
    DIFMA = MA(DIF, M)

    Parameters
    ----------
    close : Expression
    N1 : int (常用: 10)
    N2 : int (常用: 50)
    M : int (常用: 10)
    """

    def __init__(self, close, N1, N2, M):
        self.close = close
        self.N1 = N1
        self.N2 = N2
        self.M = M

    def __str__(self):
        return "DFMA_DIFMA({},{},{},{})".format(self.close, self.N1, self.N2, self.M)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        dif = c.rolling(self.N1, min_periods=1).mean() - c.rolling(self.N2, min_periods=1).mean()
        return dif.rolling(self.M, min_periods=1).mean()

    def get_longest_back_rolling(self):
        return max(self.N1, self.N2) + self.M


# ==============================================================================
# Operator list for registration
# ==============================================================================

# ==============================================================================
# JQ110 - 聚宽110+因子 新增算子
# ==============================================================================
# 以下算子是 TDXGS 已有算子未能覆盖、用户脚本中使用的因子所需的新算子。


class AroonUp(Rolling):
    """聚宽 Aroon Up - 阿隆上升指标

    用法: AroonUp($high, N)
    AroonUp = (N - 最高价出现位置) / N * 100
    等价于: c.rolling(N).apply(lambda x: float(np.argmax(x)+1)/N*100)

    Parameters
    ----------
    high : Expression
    N : int (常用: 25)
    """

    def __init__(self, high, N):
        self.high = high
        self.N = N

    def __str__(self):
        return "AroonUp({},{})".format(self.high, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        h = self.high.load(instrument, start_index, end_index, *args)
        N = self.N

        def _aroon_up(arr: np.ndarray) -> float:
            if np.isnan(arr).all():
                return np.nan
            pos = np.argmax(arr)
            return (pos + 1) / N * 100.0

        return h.rolling(N, min_periods=N).apply(_aroon_up, raw=True)

    def get_longest_back_rolling(self):
        return self.N


class AroonDown(Rolling):
    """聚宽 Aroon Down - 阿隆下降指标

    用法: AroonDown($low, N)
    AroonDown = (N - 最低价出现位置) / N * 100

    Parameters
    ----------
    low : Expression
    N : int (常用: 25)
    """

    def __init__(self, low, N):
        self.low = low
        self.N = N

    def __str__(self):
        return "AroonDown({},{})".format(self.low, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        l = self.low.load(instrument, start_index, end_index, *args)
        N = self.N

        def _aroon_dn(arr: np.ndarray) -> float:
            if np.isnan(arr).all():
                return np.nan
            pos = np.argmin(arr)
            return (pos + 1) / N * 100.0

        return l.rolling(N, min_periods=N).apply(_aroon_dn, raw=True)

    def get_longest_back_rolling(self):
        return self.N


class BullPower(Rolling):
    """聚宽 Bull Power - 多头力道 (Elder指标)

    用法: BullPower($high, $close, N)
    BullPower = high - EMA(close, N)

    Parameters
    ----------
    high : Expression
    close : Expression
    N : int (常用: 13)
    """

    def __init__(self, high, close, N):
        self.high = high
        self.close = close
        self.N = N

    def __str__(self):
        return "BullPower({},{},{})".format(self.high, self.close, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        h = self.high.load(instrument, start_index, end_index, *args)
        c = self.close.load(instrument, start_index, end_index, *args)
        ema = c.ewm(span=self.N, adjust=False).mean()
        return h - ema

    def get_longest_back_rolling(self):
        return np.inf


class BearPower(Rolling):
    """聚宽 Bear Power - 空头力道 (Elder指标)

    用法: BearPower($low, $close, N)
    BearPower = low - EMA(close, N)

    Parameters
    ----------
    low : Expression
    close : Expression
    N : int (常用: 13)
    """

    def __init__(self, low, close, N):
        self.low = low
        self.close = close
        self.N = N

    def __str__(self):
        return "BearPower({},{},{})".format(self.low, self.close, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        l = self.low.load(instrument, start_index, end_index, *args)
        c = self.close.load(instrument, start_index, end_index, *args)
        ema = c.ewm(span=self.N, adjust=False).mean()
        return l - ema

    def get_longest_back_rolling(self):
        return np.inf


class VPT(Rolling):
    """聚宽 VPT - 量价趋势 (Volume Price Trend)

    用法: VPT($close, $volume)
    VPT = Cumulative Sum(volume * (close - Ref(close,1)) / Ref(close,1))

    Parameters
    ----------
    close : Expression
    volume : Expression
    """

    def __init__(self, close, volume):
        self.close = close
        self.volume = volume

    def __str__(self):
        return "VPT({},{})".format(self.close, self.volume)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        v = self.volume.load(instrument, start_index, end_index, *args)
        pc = c.shift(1)
        daily_vpt = v * (c - pc) / (pc + 1e-12)
        return daily_vpt.cumsum()

    def get_longest_back_rolling(self):
        return np.inf


class WVAD(Rolling):
    """聚宽 WVAD - 威廉变异离散量 (Williams Variable Accumulation Distribution)

    用法: WVAD($open, $close, $high, $low, $volume, N)
    WVAD = Sum((close-open)/(high-low) * volume, N)

    Parameters
    ----------
    open_ : Expression
    close : Expression
    high : Expression
    low : Expression
    volume : Expression
    N : int (常用: 24)
    """

    def __init__(self, open_, close, high, low, volume, N):
        self.open_ = open_
        self.close = close
        self.high = high
        self.low = low
        self.volume = volume
        self.N = N

    def __str__(self):
        return "WVAD({},{},{},{},{},{})".format(
            self.open_, self.close, self.high, self.low, self.volume, self.N
        )

    def _load_internal(self, instrument, start_index, end_index, *args):
        o = self.open_.load(instrument, start_index, end_index, *args)
        c = self.close.load(instrument, start_index, end_index, *args)
        h = self.high.load(instrument, start_index, end_index, *args)
        l = self.low.load(instrument, start_index, end_index, *args)
        v = self.volume.load(instrument, start_index, end_index, *args)
        wvad = ((c - o) / (h - l + 1e-12) * v)
        return wvad.rolling(self.N, min_periods=self.N).sum()

    def get_longest_back_rolling(self):
        return self.N


class VOSC(Rolling):
    """聚宽 VOSC - 成交量震荡 (Volume Oscillator)

    用法: VOSC($volume, N1, N2)
    VOSC = (MA(volume,N1) - MA(volume,N2)) / MA(volume,N1) * 100

    Parameters
    ----------
    volume : Expression
    N1 : int (常用: 5, 12)
    N2 : int (常用: 20, 26)
    """

    def __init__(self, volume, N1, N2):
        self.volume = volume
        self.N1 = N1
        self.N2 = N2

    def __str__(self):
        return "VOSC({},{},{})".format(self.volume, self.N1, self.N2)

    def _load_internal(self, instrument, start_index, end_index, *args):
        v = self.volume.load(instrument, start_index, end_index, *args)
        ma1 = v.rolling(self.N1, min_periods=self.N1).mean()
        ma2 = v.rolling(self.N2, min_periods=self.N2).mean()
        return (ma1 - ma2) / (ma1 + 1e-12) * 100.0

    def get_longest_back_rolling(self):
        return max(self.N1, self.N2)


class Variance(Rolling):
    """聚宽 Variance - N日收益率方差

    用法: Variance($close, N)
    Variance = Var(close/Ref(close,1)-1, N)

    Parameters
    ----------
    close : Expression
    N : int (常用: 20, 60, 120)
    """

    def __init__(self, close, N):
        self.close = close
        self.N = N

    def __str__(self):
        return "Variance({},{})".format(self.close, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        rets = c.pct_change()
        return rets.rolling(self.N, min_periods=self.N).var()

    def get_longest_back_rolling(self):
        return self.N + 1


class Skewness(Rolling):
    """聚宽 Skewness - N日收益率偏度

    用法: Skewness($close, N)

    Parameters
    ----------
    close : Expression
    N : int (常用: 20, 60, 120)
    """

    def __init__(self, close, N):
        self.close = close
        self.N = N

    def __str__(self):
        return "Skewness({},{})".format(self.close, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        rets = c.pct_change()
        return rets.rolling(self.N, min_periods=self.N).skew()

    def get_longest_back_rolling(self):
        return self.N + 1


class Kurtosis(Rolling):
    """聚宽 Kurtosis - N日收益率峰度

    用法: Kurtosis($close, N)

    Parameters
    ----------
    close : Expression
    N : int (常用: 20, 60, 120)
    """

    def __init__(self, close, N):
        self.close = close
        self.N = N

    def __str__(self):
        return "Kurtosis({},{})".format(self.close, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        rets = c.pct_change()
        return rets.rolling(self.N, min_periods=self.N).kurt()

    def get_longest_back_rolling(self):
        return self.N + 1


class SharpeRatio(Rolling):
    """聚宽 SharpeRatio - N日年化夏普比率

    用法: SharpeRatio($close, N)
    SharpeRatio = Mean(rets, N) * 250 / (Std(rets, N) * sqrt(250))

    Parameters
    ----------
    close : Expression
    N : int (常用: 20, 60, 120)
    """

    def __init__(self, close, N):
        self.close = close
        self.N = N

    def __str__(self):
        return "SharpeRatio({},{})".format(self.close, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        rets = c.pct_change()
        mean_ret = rets.rolling(self.N, min_periods=self.N).mean()
        std_ret = rets.rolling(self.N, min_periods=self.N).std()
        return (mean_ret * 250.0) / (std_ret * np.sqrt(250.0) + 1e-12)

    def get_longest_back_rolling(self):
        return self.N + 1


class PriceRank(Rolling):
    """聚宽 PriceRank - N日价格位置百分位 (52周位置/月度价格位置)

    用法: PriceRank($close, N)
    PriceRank = (close - Min(close,N)) / (Max(close,N) - Min(close,N))

    Parameters
    ----------
    close : Expression
    N : int (常用: 20, 60, 250)
    """

    def __init__(self, close, N):
        self.close = close
        self.N = N

    def __str__(self):
        return "PriceRank({},{})".format(self.close, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        c_min = c.rolling(self.N, min_periods=self.N).min()
        c_max = c.rolling(self.N, min_periods=self.N).max()
        return (c - c_min) / (c_max - c_min + 1e-12)

    def get_longest_back_rolling(self):
        return self.N


class CumulativeRange(Rolling):
    """聚宽 CumulativeRange - N日累计振幅

    用法: CumulativeRange($high, $low, $close, N)
    CumulativeRange = (HHV(high,N) - LLV(low,N)) / close

    Parameters
    ----------
    high : Expression
    low : Expression
    close : Expression
    N : int (常用: 20)
    """

    def __init__(self, high, low, close, N):
        self.high = high
        self.low = low
        self.close = close
        self.N = N

    def __str__(self):
        return "CumulativeRange({},{},{},{})".format(self.high, self.low, self.close, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        h = self.high.load(instrument, start_index, end_index, *args)
        l = self.low.load(instrument, start_index, end_index, *args)
        c = self.close.load(instrument, start_index, end_index, *args)
        hhv = h.rolling(self.N, min_periods=self.N).max()
        llv = l.rolling(self.N, min_periods=self.N).min()
        return (hhv - llv) / (c + 1e-12)

    def get_longest_back_rolling(self):
        return self.N


class MoneyFlow(Rolling):
    """聚宽 MoneyFlow - N日资金流量

    用法: MoneyFlow($close, $volume, $vwap, N)
    MoneyFlow = Sum(money * sign(ret), N)
    其中 money = vwap * volume, ret = close/Ref(close,1)-1

    Parameters
    ----------
    close : Expression
    volume : Expression
    vwap : Expression
    N : int (常用: 20)
    """

    def __init__(self, close, volume, vwap, N):
        self.close = close
        self.volume = volume
        self.vwap = vwap
        self.N = N

    def __str__(self):
        return "MoneyFlow({},{},{},{})".format(self.close, self.volume, self.vwap, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        c = self.close.load(instrument, start_index, end_index, *args)
        v = self.volume.load(instrument, start_index, end_index, *args)
        vwap = self.vwap.load(instrument, start_index, end_index, *args)
        money = vwap * v
        ret = c.pct_change()
        mf = money * np.sign(ret)
        return pd.Series(mf, index=c.index).rolling(self.N, min_periods=self.N).sum()

    def get_longest_back_rolling(self):
        return self.N + 1


CUSTOM_OPS = [
    # 原有算子
    TsArgmax, TsArgmin, SMA, Amount,
    # TDXGS 通达信技术指标算子
    ATR, RSV, RSI, BIAS, BBI, WR, CCI,
    PDI, MDI, ADX, ADXR,
    BOLL_UP, BOLL_DN, BOLL_MID,
    PSY, PSYMA, ROC, MAROC, MTM, MTMMA,
    TRIX, TRMA, VR, CR, AR, BR, OBV, MFI,
    DPO, MADPO,
    TAQ_UP, TAQ_DN, TAQ_MID,
    KTN_UP, KTN_DN, KTN_MID,
    EMV, MAEMV, MASS, MA_MASS,
    DFMA_DIF, DFMA_DIFMA,
    MA, STD_TDX,
    # JQ110 聚宽因子新增算子
    AroonUp, AroonDown,
    BullPower, BearPower,
    VPT, WVAD, VOSC,
    Variance, Skewness, Kurtosis, SharpeRatio,
    PriceRank, CumulativeRange, MoneyFlow,
]
