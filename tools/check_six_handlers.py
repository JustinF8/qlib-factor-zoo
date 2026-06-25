# -*- coding: utf-8 -*-
"""
六大因子库抽样检查脚本
检查 Alpha360 / JQ110 / Alpha158 / Alpha101 / GTJA191 / TDXGS
是否能正常导入、实例化，以及因子表达式配置是否正确生成。
"""

import sys
import traceback

PASS = "[OK]"
FAIL = "[FAIL]"


def check_imports():
    """1. 检查所有模块导入"""
    print("=" * 60)
    print("  1. 模块导入检查")
    print("=" * 60)

    results = {}
    modules = {
        "handler": "qlib.contrib.data.handler",
        "loader": "qlib.contrib.data.loader",
        "loader_alpha101": "qlib.contrib.data.loader_alpha101",
        "loader_gtja191": "qlib.contrib.data.loader_gtja191",
        "custom_ops": "qlib.contrib.data.custom_ops",
    }

    for name, mod_path in modules.items():
        try:
            __import__(mod_path)
            print(f"  {PASS} {mod_path}")
            results[name] = True
        except Exception as e:
            print(f"  {FAIL} {mod_path} -> {e}")
            results[name] = False

    return all(results.values())


def check_handler_import():
    """2. 检查六大 Handler 类的导入"""
    print("\n" + "=" * 60)
    print("  2. Handler 类导入检查")
    print("=" * 60)

    classes = [
        ("Alpha360", "qlib.contrib.data.handler"),
        ("JQ110DataHandler", "qlib.contrib.data.handler"),
        ("Alpha158", "qlib.contrib.data.handler"),
        ("Alpha101", "qlib.contrib.data.handler"),
        ("GTJA191", "qlib.contrib.data.handler"),
        ("TDXGS", "qlib.contrib.data.handler"),
    ]

    imported = {}
    for cls_name, mod_path in classes:
        try:
            mod = __import__(mod_path, fromlist=[cls_name])
            getattr(mod, cls_name)
            print(f"  {PASS} {cls_name}")
            imported[cls_name] = getattr(mod, cls_name)
        except Exception as e:
            print(f"  {FAIL} {cls_name} -> {e}")
            imported[cls_name] = None

    return imported


def check_feature_configs(imported):
    """3. 检查因子表达式配置生成"""
    print("\n" + "=" * 60)
    print("  3. 因子表达式配置生成检查")
    print("=" * 60)

    from qlib.contrib.data.loader import Alpha158DL, Alpha360DL
    from qlib.contrib.data.loader_alpha101 import Alpha101DL
    from qlib.contrib.data.loader_gtja191 import GTJA191DL

    configs = {}

    # --- Alpha360 ---
    try:
        cfg = Alpha360DL.get_feature_config()
        fields, names = cfg
        print(f"  {PASS} Alpha360DL:  {len(fields)} 个表达式, {len(names)} 个名称")
        configs["Alpha360"] = (len(fields), len(names))
    except Exception as e:
        print(f"  {FAIL} Alpha360DL -> {e}")
        configs["Alpha360"] = None

    # --- Alpha158 ---
    try:
        conf = {
            "kbar": {},
            "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW", "VWAP"]},
            "rolling": {},
        }
        cfg = Alpha158DL.get_feature_config(conf)
        fields, names = cfg
        print(f"  {PASS} Alpha158DL:  {len(fields)} 个表达式, {len(names)} 个名称")
        configs["Alpha158"] = (len(fields), len(names))
    except Exception as e:
        print(f"  {FAIL} Alpha158DL -> {e}")
        configs["Alpha158"] = None

    # --- JQ110 ---
    try:
        cfg = imported["JQ110DataHandler"].get_feature_config()
        fields, names = cfg
        print(f"  {PASS} JQ110:       {len(fields)} 个表达式, {len(names)} 个名称")
        configs["JQ110"] = (len(fields), len(names))
        # 打印前5个因子名作为样本
        print(f"         样本: {names[:5]}")
    except Exception as e:
        print(f"  {FAIL} JQ110 -> {e}")
        traceback.print_exc()
        configs["JQ110"] = None

    # --- Alpha101 ---
    try:
        cfg = Alpha101DL.get_feature_config()
        fields, names = cfg
        print(f"  {PASS} Alpha101:    {len(fields)} 个表达式, {len(names)} 个名称")
        configs["Alpha101"] = (len(fields), len(names))
        print(f"         样本: {names[:5]}")
    except Exception as e:
        print(f"  {FAIL} Alpha101 -> {e}")
        traceback.print_exc()
        configs["Alpha101"] = None

    # --- GTJA191 ---
    try:
        cfg = GTJA191DL.get_feature_config()
        fields, names = cfg
        print(f"  {PASS} GTJA191:     {len(fields)} 个表达式, {len(names)} 个名称")
        configs["GTJA191"] = (len(fields), len(names))
        print(f"         样本: {names[:5]}")
    except Exception as e:
        print(f"  {FAIL} GTJA191 -> {e}")
        traceback.print_exc()
        configs["GTJA191"] = None

    # --- TDXGS ---
    try:
        TDXGS = imported["TDXGS"]
        # TDXGS.get_feature_config(self) 不依赖实例状态，传入类本身作为 self
        cfg = TDXGS.get_feature_config(TDXGS)
        fields, names = cfg
        print(f"  {PASS} TDXGS:       {len(fields)} 个表达式, {len(names)} 个名称")
        configs["TDXGS"] = (len(fields), len(names))
        print(f"         样本: {names[:5]}")
    except Exception as e:
        print(f"  {FAIL} TDXGS -> {e}")
        traceback.print_exc()
        configs["TDXGS"] = None

    return configs


def check_custom_ops():
    """4. 检查自定义算子注册"""
    print("\n" + "=" * 60)
    print("  4. 自定义算子注册检查")
    print("=" * 60)

    try:
        from qlib.contrib.data.custom_ops import TsArgmax, TsArgmin, SMA, Amount
        from qlib.contrib.data.custom_ops import (
            BIAS, CCI, WR, MTM, ROC, VR, CR,
            AR, BR, OBV, DPO, TAQ_UP, TAQ_DN, TAQ_MID, KTN_UP, KTN_DN, KTN_MID, EMV, MASS,
            DFMA_DIF, DFMA_DIFMA, STD_TDX, ATR, RSI,
        )
        ops = [
            "TsArgmax", "TsArgmin", "SMA", "Amount",
            "BIAS", "CCI", "WR", "MTM", "ROC", "VR", "CR",
            "AR", "BR", "OBV", "DPO", "TAQ_UP", "TAQ_DN", "TAQ_MID", "KTN_UP", "KTN_DN", "KTN_MID", "EMV", "MASS",
            "DFMA_DIF", "DFMA_DIFMA", "STD_TDX", "ATR", "RSI",
        ]
        print(f"  {PASS} 共 {len(ops)} 个自定义算子全部可用")
        print(f"         算子列表: {', '.join(ops[:10])} ...")
    except ImportError as e:
        print(f"  {FAIL} 自定义算子导入失败 -> {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"  {FAIL} 未知错误 -> {e}")
        traceback.print_exc()


def check_label_configs(imported):
    """5. 检查标签配置"""
    print("\n" + "=" * 60)
    print("  5. 标签配置检查")
    print("=" * 60)

    for name in ["Alpha360", "Alpha158", "Alpha101", "GTJA191", "TDXGS"]:
        cls = imported.get(name)
        if cls is None:
            print(f"  {FAIL} {name} (未导入)")
            continue
        try:
            # get_label_config(self) 不依赖实例状态，传入类本身作为 self
            labels, label_names = cls.get_label_config(cls)
            print(f"  {PASS} {name}: {label_names}")
        except Exception as e:
            print(f"  {FAIL} {name} -> {e}")

    # JQ110 单独检查
    try:
        JQ110 = imported["JQ110DataHandler"]
        labels, label_names = JQ110.get_label_config(JQ110)
        print(f"  {PASS} JQ110: {label_names}")
    except Exception as e:
        print(f"  {FAIL} JQ110 -> {e}")


def check_data_loader_registration():
    """6. 检查 DataLoader 是否在 Qlib 全局注册表中"""
    print("\n" + "=" * 60)
    print("  6. DataLoader 全局注册检查")
    print("=" * 60)

    try:
        from qlib.data.dataset.loader import QlibDataLoader
        print(f"  {PASS} QlibDataLoader 可用")

        from qlib.contrib.data.loader import Alpha158DL, Alpha360DL, JQ110DL
        print(f"  {PASS} Alpha158DL / Alpha360DL / JQ110DL 可用")

        from qlib.contrib.data.loader_alpha101 import Alpha101DL
        print(f"  {PASS} Alpha101DL 可用")

        from qlib.contrib.data.loader_gtja191 import GTJA191DL
        print(f"  {PASS} GTJA191DL 可用")
    except Exception as e:
        print(f"  {FAIL} -> {e}")
        traceback.print_exc()


def main():
    print("\n" + "█" * 60)
    print("  六大因子库抽样检查")
    print("█" * 60)

    all_ok = True

    # 1. 模块导入
    if not check_imports():
        all_ok = False

    # 2. Handler 类导入
    imported = check_handler_import()
    if any(v is None for v in imported.values()):
        all_ok = False

    # 3. 因子表达式配置
    configs = check_feature_configs(imported)
    if any(v is None for v in configs.values()):
        all_ok = False

    # 4. 自定义算子
    check_custom_ops()

    # 5. 标签配置
    check_label_configs(imported)

    # 6. DataLoader 注册
    check_data_loader_registration()

    # --- 汇总 ---
    print("\n" + "=" * 60)
    print("  汇总")
    print("=" * 60)
    print(f"  因子库        表达式数    名称数")
    print(f"  {'─' * 40}")
    expected = {
        "Alpha360": 360, "Alpha158": 158, "JQ110": 113,
        "Alpha101": 101, "GTJA191": 191, "TDXGS": 90,
    }
    for name, exp_count in expected.items():
        cfg = configs.get(name)
        if cfg:
            nf, nn = cfg
            flag = PASS if nf >= exp_count * 0.8 else "[WARN]"
            print(f"  {flag} {name:12s} {nf:5d}      {nn:5d}  (预期 ~{exp_count})")
        else:
            print(f"  {FAIL} {name:12s}  ---        ---")

    print()
    if all_ok:
        print("  [OK] 所有检查通过！六大因子库可以正常使用。")
    else:
        print("  [FAIL] 存在失败项，请检查上方报错信息。")


if __name__ == "__main__":
    main()
