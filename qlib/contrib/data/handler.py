# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from qlib.contrib.data.loader import Alpha158DL, Alpha360DL
from qlib.contrib.data.loader_alpha101 import Alpha101DL
from qlib.contrib.data.loader_gtja191 import GTJA191DL
from qlib.data.dataset.loader import QlibDataLoader
from ...data.dataset.handler import DataHandlerLP
from ...data.dataset.processor import Processor
from ...utils import get_callable_kwargs
from ...data.dataset import processor as processor_module
from inspect import getfullargspec

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy registration of custom operators (TsArgmax, TsArgmin, SMA, Amount)
# These are needed by Alpha101 and GTJA191 expression definitions.
# Registration happens once, on first import of this module.
# ---------------------------------------------------------------------------
_custom_ops_registered = False


def _register_custom_ops_once():
    """Register custom operators with Qlib's expression engine (idempotent)."""
    global _custom_ops_registered
    if _custom_ops_registered:
        return
    try:
        from qlib.config import C
        from qlib.contrib.data.custom_ops import (
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
        )
        if "custom_ops" not in C:
            C["custom_ops"] = []
        existing_names = {op.__name__ for op in C["custom_ops"] if not isinstance(op, dict)}
        for op in [
            TsArgmax, TsArgmin, SMA, Amount,
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
        ]:
            if op.__name__ not in existing_names:
                C["custom_ops"].append(op)
                logger.debug("Registered custom operator: %s", op.__name__)
        _custom_ops_registered = True
    except ImportError:
        logger.warning("Could not register custom operators; Alpha101/GTJA191/TDXGS may use fallback expressions")
        _custom_ops_registered = True  # don't retry


_register_custom_ops_once()


def check_transform_proc(proc_l, fit_start_time, fit_end_time):
    new_l = []
    for p in proc_l:
        if not isinstance(p, Processor):
            klass, pkwargs = get_callable_kwargs(p, processor_module)
            args = getfullargspec(klass).args
            if "fit_start_time" in args and "fit_end_time" in args:
                assert (
                    fit_start_time is not None and fit_end_time is not None
                ), "Make sure `fit_start_time` and `fit_end_time` are not None."
                pkwargs.update(
                    {
                        "fit_start_time": fit_start_time,
                        "fit_end_time": fit_end_time,
                    }
                )
            proc_config = {"class": klass.__name__, "kwargs": pkwargs}
            if isinstance(p, dict) and "module_path" in p:
                proc_config["module_path"] = p["module_path"]
            new_l.append(proc_config)
        else:
            new_l.append(p)
    return new_l


_DEFAULT_LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
]
_DEFAULT_INFER_PROCESSORS = [
    {"class": "ProcessInf", "kwargs": {}},
    {"class": "ZScoreNorm", "kwargs": {}},
    {"class": "Fillna", "kwargs": {}},
]


class Alpha360(DataHandlerLP):
    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=_DEFAULT_INFER_PROCESSORS,
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        filter_pipe=None,
        inst_processors=None,
        **kwargs,
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": Alpha360DL.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            learn_processors=learn_processors,
            infer_processors=infer_processors,
            **kwargs,
        )

    def get_label_config(self):
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]



class JQ110DataHandler(DataHandlerLP):
    """聚宽110+因子 数据处理器。

    基于用户脚本的 calculate_all_factors() 函数，生成以下五大类因子：
    1. MOMENTUM   - 动量因子 (ROC/BIAS/Aroon/CCI/CR/MASS/TRIX/BBI等)
    2. EMOTION    - 情绪量能因子 (VOL/DAVOL/ARBR/ATR/PSY/VOSC/VR/WVAD等)
    3. TECHNICAL  - 技术指标 (EMA/MA/MACD/BOLL/MFI等)
    4. RISK       - 风险统计 (Variance/Skewness/Kurtosis/SharpeRatio)
    5. STYLE      - 风格因子 (Beta/RelativeStrength/CumulativeRange等)

    总计约 110 个因子。
    """

    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
        freq="day",
        inst_processor=None,
        **kwargs,
    ):
        # 继承 DataHandlerLP 的完整流程
        super().__init__(
            instruments,
            start_time=start_time,
            end_time=end_time,
            freq=freq,
            data_loader=self.get_data_loader(),
            inst_processor=inst_processor,
            **kwargs,
        )

    def get_data_loader(self):
        """获取 JQ110 因子数据加载器"""
        return JQ110DL()

    @staticmethod
    def get_feature_config():
        """生成所有 JQ110 因子表达式列表。

        内置算子 (Qlib): Mean/Sum/Std/Max/Min/Rank/Delta/EMA/WMA/Corr/Cov/
                          Slope/Ref/Abs/Log/Sign/Power/If/Greater/Less/CsRank
        自定义算子 (custom_ops.py): 见 CUSTOM_OPS 完整列表
        """
        fields = []
        names = []

        # =============================================================
        # 第1组：MOMENTUM 动量因子 (38个)
        # =============================================================

        # 1.1 ROC 变动率系列
        for w in [6, 12, 20, 60, 120]:
            fields.append(f"($close-Ref($close,{w}))/(Ref($close,{w})+1e-12)")
            names.append(f"JQ110_ROC_{w:03d}")

        # 1.2 BIAS 乖离率系列
        for w in [5, 10, 20, 60]:
            fields.append(f"$close/Mean($close,{w})-1")
            names.append(f"JQ110_BIAS_{w:02d}")

        # 1.3 Aroon 阿隆指标
        fields.append("AroonUp($high, 25)")
        names.append("JQ110_aroon_up_25")
        fields.append("AroonDown($low, 25)")
        names.append("JQ110_aroon_down_25")

        # 1.4 BBI / BBIC
        fields.append("BBI($close, 3, 6, 12, 24)")
        names.append("JQ110_BBI")
        fields.append("$close/BBI($close, 3, 6, 12, 24)-1")
        names.append("JQ110_BBIC")

        # 1.5 CCI 系列
        for w in [10, 15, 20, 88]:
            fields.append(f"CCI($close, $high, $low, {w})")
            names.append(f"JQ110_CCI_{w:03d}")

        # 1.6 CR20
        fields.append("CR($close, $high, $low, 20)")
        names.append("JQ110_CR20")

        # 1.7 MASS 梅斯线
        fields.append("MASS($high, $low, 9, 25)")
        names.append("JQ110_MASS")

        # 1.8 TRIX 系列
        for w in [5, 10]:
            fields.append(f"TRIX($close, {w})")
            names.append(f"JQ110_TRIX_{w:02d}")

        # 1.9 Price1M / Price3M / Price1Y (价格月度/季度/年度变动)
        fields.append("$close/Ref($close, 20)-1")
        names.append("JQ110_Price1M")
        fields.append("$close/Ref($close, 60)-1")
        names.append("JQ110_Price3M")
        fields.append("$close/Ref($close, 250)-1")
        names.append("JQ110_Price1Y")

        # 1.10 Rank1M (滚动排名百分位)
        fields.append("PriceRank($close, 20)")
        names.append("JQ110_Rank1M")

        # 1.11 52周价格位置
        fields.append("PriceRank($close, 250)")
        names.append("JQ110_52week_rank")

        # 1.12 Bull/Bear Power (Elder多空力道)
        fields.append("BullPower($high, $close, 13)")
        names.append("JQ110_bull_power")
        fields.append("BearPower($low, $close, 13)")
        names.append("JQ110_bear_power")

        # 1.13 VPT 系列 (量价趋势)
        fields.append("VPT($close, $volume)")
        names.append("JQ110_VPT")
        fields.append("Delta(VPT($close, $volume), 1)")
        names.append("JQ110_single_day_VPT")
        for w in [6, 12]:
            fields.append(f"Mean(Delta(VPT($close,$volume),1),{w})")
            names.append(f"JQ110_single_day_VPT_{w:02d}")

        # 1.14 Volume1M (量价综合)
        fields.append("($volume/(Mean($volume,20)+1e-12))*($close/Ref($close,20)-1)")
        names.append("JQ110_Volume1M")

        # 1.15 PLRC 系列 (线性回归斜率 → 用 Slope 替代)
        for w in [6, 12, 24]:
            fields.append(f"Slope($close, {w})/($close+1e-12)")
            names.append(f"JQ110_PLRC_{w:02d}")

        # =============================================================
        # 第2组：EMOTION 情绪量能因子 (36个)
        # =============================================================

        # 2.1 VOL 成交量比系列
        for w in [5, 10, 20, 60, 120, 240]:
            fields.append(f"Mean($volume,{w})/(Mean($volume,250)+1e-12)")
            names.append(f"JQ110_VOL_{w:03d}")

        # 2.2 DAVOL 短期成交量比系列
        for w in [5, 10, 20]:
            fields.append(f"Mean($volume,{w})/(Mean($volume,120)+1e-12)")
            names.append(f"JQ110_DAVOL_{w:02d}")

        # 2.3 turnover_volatility (量波动率)
        fields.append("Std($volume,20)/(Mean($volume,20)+1e-12)")
        names.append("JQ110_turnover_volatility")

        # 2.4 TVMA / TVSTD (成交额MA/STD, 用 $vwap*$volume 替代 $money)
        for w in [6, 20]:
            fields.append(f"Mean($vwap*$volume,{w})")
            names.append(f"JQ110_TVMA_{w:02d}")
            fields.append(f"Std($vwap*$volume,{w})")
            names.append(f"JQ110_TVSTD_{w:02d}")

        # 2.5 VEMA / VSTD (成交量EMA/STD)
        for w in [5, 10, 12, 26]:
            fields.append(f"EMA($volume,{w})")
            names.append(f"JQ110_VEMA_{w:02d}")
        for w in [10, 20]:
            fields.append(f"Std($volume,{w})")
            names.append(f"JQ110_VSTD_{w:02d}")

        # 2.6 AR / BR / ARBR
        fields.append("AR($open, $close, $high, $low, 20)")
        names.append("JQ110_AR")
        fields.append("BR($open, $close, $high, $low, 20)")
        names.append("JQ110_BR")
        fields.append("AR($open,$close,$high,$low,20)/(BR($open,$close,$high,$low,20)+1e-12)")
        names.append("JQ110_ARBR")

        # 2.7 ATR 系列
        for w in [6, 14]:
            fields.append(f"ATR($close, $high, $low, {w})")
            names.append(f"JQ110_ATR_{w:02d}")

        # 2.8 PSY 心理线
        fields.append("PSY($close, 12)")
        names.append("JQ110_PSY")

        # 2.9 VMACD 系列 (量MACD)
        fields.append("EMA($volume,12)-EMA($volume,26)")
        names.append("JQ110_VDIFF")
        fields.append("EMA(EMA($volume,12)-EMA($volume,26),9)")
        names.append("JQ110_VDEA")
        fields.append("2*(EMA($volume,12)-EMA($volume,26)-EMA(EMA($volume,12)-EMA($volume,26),9))")
        names.append("JQ110_VMACD")

        # 2.10 VOSC 成交量震荡
        fields.append("VOSC($volume, 5, 20)")
        names.append("JQ110_VOSC")

        # 2.11 VR 容量比率
        fields.append("VR($close, $volume, 20)")
        names.append("JQ110_VR")

        # 2.12 VROC 系列 (量变动率)
        for w in [6, 12]:
            fields.append(f"($volume-Ref($volume,{w}))/(Ref($volume,{w})+1e-12)")
            names.append(f"JQ110_VROC_{w:02d}")

        # 2.13 WVAD / MAWVAD
        fields.append("WVAD($open, $close, $high, $low, $volume, 24)")
        names.append("JQ110_WVAD")
        fields.append("Mean(WVAD($open,$close,$high,$low,$volume,24),6)")
        names.append("JQ110_MAWVAD")

        # 2.14 money_flow_20
        fields.append("MoneyFlow($close, $volume, $vwap, 20)")
        names.append("JQ110_money_flow_20")

        # =============================================================
        # 第3组：TECHNICAL 技术指标因子 (17个)
        # =============================================================

        # 3.1 EMA5
        fields.append("EMA($close, 5)")
        names.append("JQ110_EMA5")

        # 3.2 EMAC 系列 (EMA/close-1)
        for w in [10, 12, 20, 26, 120]:
            fields.append(f"EMA($close,{w})/$close-1")
            names.append(f"JQ110_EMAC_{w:03d}")

        # 3.3 MAC 系列 (MA/close-1)
        for w in [5, 10, 20, 60, 120]:
            fields.append(f"Mean($close,{w})/$close-1")
            names.append(f"JQ110_MAC_{w:03d}")

        # 3.4 MACDC (MACD/close)
        fields.append("(EMA($close,12)-EMA($close,26))/$close")
        names.append("JQ110_MACDC")

        # 3.5 BOLL 上下轨 (相对于收盘价的偏离)
        fields.append("BOLL_UP($close,20,2.0)/$close-1")
        names.append("JQ110_boll_up")
        fields.append("BOLL_DN($close,20,2.0)/$close-1")
        names.append("JQ110_boll_down")

        # 3.6 MFI14
        fields.append("MFI($close, $high, $low, $volume, 14)")
        names.append("JQ110_MFI14")

        # =============================================================
        # 第4组：RISK 风险统计因子 (12个)
        # =============================================================

        for w in [20, 60, 120]:
            # Variance
            fields.append(f"Variance($close, {w})")
            names.append(f"JQ110_Variance_{w:03d}")
            # SharpeRatio (年化)
            fields.append(f"SharpeRatio($close, {w})")
            names.append(f"JQ110_sharpe_ratio_{w:03d}")
            # Skewness
            fields.append(f"Skewness($close, {w})")
            names.append(f"JQ110_Skewness_{w:03d}")
            # Kurtosis
            fields.append(f"Kurtosis($close, {w})")
            names.append(f"JQ110_Kurtosis_{w:03d}")

        # =============================================================
        # 第5组：STYLE 风格因子 (10个)
        # =============================================================

        # 5.1 daily_standard_deviation
        fields.append("Std($close/Ref($close,1)-1, 20)")
        names.append("JQ110_daily_std")

        # 5.2 historical_sigma
        fields.append("Std($close/Ref($close,1)-1, 120)")
        names.append("JQ110_hist_sigma")

        # 5.3 residual_volatility (简化: 60日收益标准差)
        fields.append("Std($close/Ref($close,1)-1, 60)")
        names.append("JQ110_residual_vol")

        # 5.4 cumulative_range
        fields.append("CumulativeRange($high, $low, $close, 20)")
        names.append("JQ110_cumulative_range")

        # 5.5 momentum (年动量)
        fields.append("$close/Ref($close, 250)-1")
        names.append("JQ110_momentum")

        # 5.6 liquidity (流动性代理: 20日均量/1000000)
        fields.append("Mean($volume, 20)/1000000")
        names.append("JQ110_liquidity")

        # 5.7 share_turnover_monthly (月度换手代理: 20日总成交量)
        fields.append("Sum($volume, 20)")
        names.append("JQ110_share_turnover_monthly")

        # 5.8 average_share_turnover_annual (年度均换手)
        fields.append("Mean($volume, 250)")
        names.append("JQ110_share_turnover_annual")

        # 5.9 average_share_turnover_quarterly (季度均换手)
        fields.append("Mean($volume, 60)")
        names.append("JQ110_share_turnover_quarterly")

        # 5.10 beta (需市场基准，用 Slope 近似)
        fields.append("Slope($close/Ref($close,1)-1, 120)")
        names.append("JQ110_beta")

        return fields, names

    def get_label_config(self):
        """默认标签：未来5日收益率"""
        return ["Ref($close, -5)/$close - 1"], ["LABEL5"]


class JQ110DL(QlibDataLoader):
    """JQ110 因子数据加载器（配合 JQ110DataHandler 使用）"""

    def __init__(self, config=None, **kwargs):
        _config = {"feature": self.get_feature_config()}
        if config is not None:
            _config.update(config)
        super().__init__(config=_config, **kwargs)

    @staticmethod
    def get_feature_config():
        """复用 JQ110DataHandler 的因子表达式"""
        return JQ110DataHandler.get_feature_config()


class Alpha360vwap(Alpha360):
    def get_label_config(self):
        return ["Ref($vwap, -2)/Ref($vwap, -1) - 1"], ["LABEL0"]

class Alpha158(DataHandlerLP):
    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        **kwargs,
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs,
        )

    def get_feature_config(self):
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {},
        }
        return Alpha158DL.get_feature_config(conf)

    def get_label_config(self):
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]


class Alpha158vwap(Alpha158):
    def get_label_config(self):
        return ["Ref($vwap, -2)/Ref($vwap, -1) - 1"], ["LABEL0"]

class Alpha101(DataHandlerLP):
    """Alpha101 Data Handler - 101 Formulaic Alphas (Kakushadze 2015).

    Provides 101 alpha factors from the WorldQuant paper as Qlib expressions.
    Compatible with Alpha158-style pipeline.

    Reference: Kakushadze (2015), "101 Formulaic Alphas", arXiv:1601.00991
    """

    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=None,
        learn_processors=None,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        **kwargs,
    ):
        if infer_processors is None:
            infer_processors = _DEFAULT_INFER_PROCESSORS
        if learn_processors is None:
            learn_processors = _DEFAULT_LEARN_PROCESSORS

        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs,
        )

    def get_feature_config(self):
        return Alpha101DL.get_feature_config()

    def get_label_config(self):
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]


class GTJA191(DataHandlerLP):
    """GTJA191 Data Handler - 191 Alphas from GTJA Research Report (2014).

    Provides 191 alpha factors from the GTJA research report as Qlib expressions.
    Compatible with Alpha158-style pipeline.

    Uses custom operators (SMA, TsArgmax, TsArgmin, Amount) registered
    automatically on import for precise factor computation.

    Reference: 国泰君安 191 alpha 研报 (2014)
    """

    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=None,
        learn_processors=None,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        **kwargs,
    ):
        if infer_processors is None:
            infer_processors = _DEFAULT_INFER_PROCESSORS
        if learn_processors is None:
            learn_processors = _DEFAULT_LEARN_PROCESSORS

        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs,
        )

    def get_feature_config(self):
        return GTJA191DL.get_feature_config()

    def get_label_config(self):
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]


class TDXGS(DataHandlerLP):
    """TDXGS Data Handler - 通达信/同花顺常用技术指标因子集

    基于 MyTT (https://github.com/mpquant/MyTT) 的算法逻辑，
    将通达信和同花顺A股软件中的常用技术指标拆解为独立因子，
    通过参数枚举固化，与 Alpha158 / WQ101 / GTJA191 / Alpha360 并列使用。

    因子分组（共 ~90 个因子）：
    - EMA 多周期：5/10/12/20/26/50/60
    - MA 简单均线：5/10/20/60
    - ATR 平均真实波幅：10/14/20/60
    - RSI 相对强弱：6/12/14/24
    - BIAS 乖离率：6/12/24
    - BBI 多空指数：(3,6,12,24)
    - WR 威廉指标：6/10
    - CCI 商品通道：14/20
    - DMI 动向指标：PDI/MDI/ADX/ADXR (14,6)/(7,3)
    - BOLL 布林带：上/中/下轨 (20,2.0)/(26,2.0)
    - PSY 心理线：12,6 / 20,10
    - ROC 变动率：12,6
    - MTM 动量：12,6
    - TRIX 三重指数平滑：(12,20)/(9,15)
    - VR 容量比率：26/12
    - CR 价格动量：20/26
    - ARBR 情绪指标：AR/BR 各26
    - OBV 能量潮
    - MFI 资金流量：14/9
    - DPO 区间震荡线：20,10,6
    - TAQ 唐安奇通道：上/中/下轨 (20)/(50)
    - KTN 肯特纳通道：上/中/下轨 (20,10)
    - EMV 简易波动：14,9
    - MASS 梅斯线：9,25,6
    - DFMA 平行线差：10,50,10
    - STD_TDX 通达信标准差：10/20
    - 加工因子：ATR比率、BOLL带宽、截面排序、量价相关性等
    """

    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=None,
        learn_processors=None,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        **kwargs,
    ):
        if infer_processors is None:
            infer_processors = _DEFAULT_INFER_PROCESSORS
        if learn_processors is None:
            learn_processors = _DEFAULT_LEARN_PROCESSORS

        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs,
        )

    def get_feature_config(self):
        """生成所有通达信/同花顺技术指标因子表达式列表。

        内置算子: Mean/Sum/Std/Max/Min/Rank/Delta/EMA/WMA/Corr/Cov/
                  Slope/Ref/Abs/Log/Sign/Power/If/Greater/Less
        自定义算子 (custom_ops.py):
                  ATR/RSV/RSI/BIAS/BBI/WR/CCI/PDI/MDI/ADX/ADXR/
                  BOLL_UP/BOLL_DN/BOLL_MID/PSY/PSYMA/ROC/MAROC/MTM/MTMMA/
                  TRIX/TRMA/VR/CR/AR/BR/OBV/MFI/DPO/MADPO/
                  TAQ_UP/TAQ_DN/TAQ_MID/KTN_UP/KTN_DN/KTN_MID/
                  EMV/MAEMV/MASS/MA_MASS/DFMA_DIF/DFMA_DIFMA/
                  MA/STD_TDX/SMA/TsArgmax/TsArgmin/Amount
        """
        fields = []
        names = []

        # =====================================================
        # 1. EMA 指数移动平均 (多周期)
        # =====================================================
        for n in [5, 10, 12, 20, 26, 50, 60]:
            fields.append(f"EMA($close, {n})")
            names.append(f"TDXGS_EMA_{n:02d}")

        # =====================================================
        # 2. MA 简单移动平均 (多周期)
        # =====================================================
        for n in [5, 10, 20, 60]:
            fields.append(f"Mean($close, {n})")
            names.append(f"TDXGS_MA_{n:02d}")

        # =====================================================
        # 3. ATR 平均真实波幅
        # =====================================================
        for n in [10, 14, 20, 60]:
            fields.append(f"ATR($close, $high, $low, {n})")
            names.append(f"TDXGS_ATR_{n:02d}")

        # =====================================================
        # 4. RSI 相对强弱指标
        # =====================================================
        for n in [6, 12, 14, 24]:
            fields.append(f"RSI($close, {n})")
            names.append(f"TDXGS_RSI_{n:02d}")

        # =====================================================
        # 5. BIAS 乖离率
        # =====================================================
        for n in [6, 12, 24]:
            fields.append(f"BIAS($close, {n})")
            names.append(f"TDXGS_BIAS_{n:02d}")

        # =====================================================
        # 6. BBI 多空指数
        # =====================================================
        fields.append("BBI($close, 3, 6, 12, 24)")
        names.append("TDXGS_BBI")

        # =====================================================
        # 7. WR 威廉指标
        # =====================================================
        for n in [6, 10]:
            fields.append(f"WR($close, $high, $low, {n})")
            names.append(f"TDXGS_WR_{n:02d}")

        # =====================================================
        # 8. CCI 商品通道指数
        # =====================================================
        for n in [14, 20]:
            fields.append(f"CCI($close, $high, $low, {n})")
            names.append(f"TDXGS_CCI_{n:02d}")

        # =====================================================
        # 9. DMI 动向指标 (PDI/MDI/ADX/ADXR)
        # =====================================================
        for m1, m2 in [(14, 6), (7, 3)]:
            fields.append(f"PDI($close, $high, $low, {m1}, {m2})")
            names.append(f"TDXGS_PDI_{m1}_{m2}")
            fields.append(f"MDI($close, $high, $low, {m1}, {m2})")
            names.append(f"TDXGS_MDI_{m1}_{m2}")
            fields.append(f"ADX($close, $high, $low, {m1}, {m2})")
            names.append(f"TDXGS_ADX_{m1}_{m2}")
            fields.append(f"ADXR($close, $high, $low, {m1}, {m2})")
            names.append(f"TDXGS_ADXR_{m1}_{m2}")

        # =====================================================
        # 10. BOLL 布林带 (上/中/下轨)
        # =====================================================
        for n in [20, 26]:
            fields.append(f"BOLL_UP($close, {n}, 2.0)")
            names.append(f"TDXGS_BOLL_UP_{n:02d}")
            fields.append(f"BOLL_MID($close, {n})")
            names.append(f"TDXGS_BOLL_MID_{n:02d}")
            fields.append(f"BOLL_DN($close, {n}, 2.0)")
            names.append(f"TDXGS_BOLL_DN_{n:02d}")

        # =====================================================
        # 11. PSY 心理线
        # =====================================================
        fields.append("PSY($close, 12)")
        names.append("TDXGS_PSY_12")
        fields.append("PSYMA($close, 12, 6)")
        names.append("TDXGS_PSYMA_12_6")
        fields.append("PSY($close, 20)")
        names.append("TDXGS_PSY_20")
        fields.append("PSYMA($close, 20, 10)")
        names.append("TDXGS_PSYMA_20_10")

        # =====================================================
        # 12. ROC 变动率
        # =====================================================
        fields.append("ROC($close, 12)")
        names.append("TDXGS_ROC_12")
        fields.append("MAROC($close, 12, 6)")
        names.append("TDXGS_MAROC_12_6")

        # =====================================================
        # 13. MTM 动量指标
        # =====================================================
        fields.append("MTM($close, 12)")
        names.append("TDXGS_MTM_12")
        fields.append("MTMMA($close, 12, 6)")
        names.append("TDXGS_MTMMA_12_6")

        # =====================================================
        # 14. TRIX 三重指数平滑
        # =====================================================
        for m1, m2 in [(12, 20), (9, 15)]:
            fields.append(f"TRIX($close, {m1})")
            names.append(f"TDXGS_TRIX_{m1}")
            fields.append(f"TRMA($close, {m1}, {m2})")
            names.append(f"TDXGS_TRMA_{m1}_{m2}")

        # =====================================================
        # 15. VR 容量比率
        # =====================================================
        for m1 in [26, 12]:
            fields.append(f"VR($close, $volume, {m1})")
            names.append(f"TDXGS_VR_{m1:02d}")

        # =====================================================
        # 16. CR 价格动量
        # =====================================================
        for n in [20, 26]:
            fields.append(f"CR($close, $high, $low, {n})")
            names.append(f"TDXGS_CR_{n:02d}")

        # =====================================================
        # 17. ARBR 情绪指标 (AR/BR)
        # =====================================================
        fields.append("AR($open, $close, $high, $low, 26)")
        names.append("TDXGS_AR_26")
        fields.append("BR($open, $close, $high, $low, 26)")
        names.append("TDXGS_BR_26")

        # =====================================================
        # 18. OBV 能量潮
        # =====================================================
        fields.append("OBV($close, $volume)")
        names.append("TDXGS_OBV")

        # =====================================================
        # 19. MFI 资金流量指标
        # =====================================================
        for n in [14, 9]:
            fields.append(f"MFI($close, $high, $low, $volume, {n})")
            names.append(f"TDXGS_MFI_{n:02d}")

        # =====================================================
        # 20. DPO 区间震荡线
        # =====================================================
        fields.append("DPO($close, 20, 10)")
        names.append("TDXGS_DPO_20_10")
        fields.append("MADPO($close, 20, 10, 6)")
        names.append("TDXGS_MADPO_20_10_6")

        # =====================================================
        # 21. TAQ 唐安奇通道 (海龟交易)
        # =====================================================
        for n in [20, 50]:
            fields.append(f"TAQ_UP($high, {n})")
            names.append(f"TDXGS_TAQ_UP_{n:02d}")
            fields.append(f"TAQ_MID($high, $low, {n})")
            names.append(f"TDXGS_TAQ_MID_{n:02d}")
            fields.append(f"TAQ_DN($low, {n})")
            names.append(f"TDXGS_TAQ_DN_{n:02d}")

        # =====================================================
        # 22. KTN 肯特纳通道
        # =====================================================
        fields.append("KTN_UP($close, $high, $low, 20, 10)")
        names.append("TDXGS_KTN_UP")
        fields.append("KTN_MID($close, $high, $low, 20)")
        names.append("TDXGS_KTN_MID")
        fields.append("KTN_DN($close, $high, $low, 20, 10)")
        names.append("TDXGS_KTN_DN")

        # =====================================================
        # 23. EMV 简易波动指标
        # =====================================================
        fields.append("EMV($high, $low, $volume, 14)")
        names.append("TDXGS_EMV_14")
        fields.append("MAEMV($high, $low, $volume, 14, 9)")
        names.append("TDXGS_MAEMV_14_9")

        # =====================================================
        # 24. MASS 梅斯线
        # =====================================================
        fields.append("MASS($high, $low, 9, 25)")
        names.append("TDXGS_MASS_9_25")
        fields.append("MA_MASS($high, $low, 9, 25, 6)")
        names.append("TDXGS_MA_MASS_9_25_6")

        # =====================================================
        # 25. DFMA 平行线差 (DMA/新DMA)
        # =====================================================
        fields.append("DFMA_DIF($close, 10, 50)")
        names.append("TDXGS_DFMA_DIF")
        fields.append("DFMA_DIFMA($close, 10, 50, 10)")
        names.append("TDXGS_DFMA_DIFMA")

        # =====================================================
        # 26. STD_TDX 通达信标准差 (ddof=0)
        # =====================================================
        for n in [10, 20]:
            fields.append(f"STD_TDX($close, {n})")
            names.append(f"TDXGS_STD_{n:02d}")

        # =====================================================
        # 27. 加工因子：比率/归一化/截面排序
        # =====================================================
        # ATR / Close 归一化波动率
        fields.append("ATR($close, $high, $low, 20)/($close + 1e-12)")
        names.append("TDXGS_ATR20_RATIO")

        # BOLL 带宽 (上轨-下轨)/中轨
        fields.append("(BOLL_UP($close, 20, 2.0)-BOLL_DN($close, 20, 2.0))/(BOLL_MID($close, 20)+1e-12)")
        names.append("TDXGS_BOLL_BANDWIDTH")

        # EMA20 截面排名
        fields.append("CsRank(EMA($close, 20))")
        names.append("TDXGS_EMA20_CSRANK")

        # 量价相关性
        fields.append("Corr($close, $volume, 20)")
        names.append("TDXGS_CORR_CV_20")

        # 换手率代理：volume/MA(volume,20)
        fields.append("$volume/(Mean($volume, 20)+1e-12)")
        names.append("TDXGS_VOL_RATIO_20")

        # 日内振幅
        fields.append("($high-$low)/($open+1e-12)")
        names.append("TDXGS_AMPLITUDE")

        # 趋势强度：|close - MA20| / ATR20
        fields.append("Abs($close-Mean($close,20))/(ATR($close,$high,$low,20)+1e-12)")
        names.append("TDXGS_TREND_STRENGTH")

        return fields, names

    def get_label_config(self):
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]


