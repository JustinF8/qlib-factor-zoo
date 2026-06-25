# -*- coding: utf-8 -*-
"""
通达信本地数据 → Qlib 格式转换工具

将通达信 TQ 接口的本地数据转换为 Qlib 标准 .bin 格式，
供 qlib.contrib.data.handler 中的六大因子库 (Alpha360/JQ110/Alpha158/
Alpha101/GTJA191/TDXGS) 直接使用。

数据流:
  通达信本地数据文件 (tqcenter.tq)
    -> 本脚本导出为 CSV 中间格式
    -> dump_bin.py 转换为 Qlib .bin 格式
    -> Qlib DataHandler 加载

使用方法:
  # 1. 先导出数据
  python tdx_data_to_qlib.py export --pool csi300 --start 20230101 --end 20240628

  # 2. 转换为 Qlib 格式
  python tdx_data_to_qlib.py dump --csv_dir ./tdx_csv_data --qlib_dir ./tdx_qlib_data

  # 3. 一步完成
  python tdx_data_to_qlib.py all --pool csi300 --qlib_dir ./tdx_qlib_data

依赖: tqcenter (通达信量化客户端), pandas, numpy
"""

import sys
import os
import time
import argparse
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# ============================================================
# 0. 初始化通达信 TQ 接口
# ============================================================

TDX_PLUGIN_PATH = 'd:/zd_zyb(x64 26011715)GA/PYPlugins/user'
if TDX_PLUGIN_PATH not in sys.path:
    sys.path.append(TDX_PLUGIN_PATH)

_global_tq = None
_global_tq_lock = threading.Lock()


def get_tq():
    """获取/初始化全局 TQ 实例（线程安全）"""
    global _global_tq
    if _global_tq is not None:
        return _global_tq
    with _global_tq_lock:
        if _global_tq is not None:
            return _global_tq
        from tqcenter import tq
        tq.initialize(TDX_PLUGIN_PATH)
        _global_tq = tq
        print(f"[TQ] 通达信数据接口初始化成功")
        return _global_tq


# ============================================================
# 1. 股票池获取
# ============================================================

# 沪深300成分股（2024年，前100只代表性股票）
CSI300_SAMPLE = [
    "000001", "000002", "000063", "000100", "000157", "000333", "000338", "000425",
    "000568", "000596", "000625", "000651", "000661", "000725", "000768", "000776",
    "000786", "000792", "000800", "000858", "000876", "000895", "000938", "000963",
    "000977", "002001", "002007", "002027", "002049", "002050", "002142", "002230",
    "002236", "002241", "002271", "002304", "002311", "002352", "002371", "002415",
    "002459", "002460", "002475", "002493", "002594", "002601", "002714", "002736",
    "002812", "002916",
    "600000", "600009", "600010", "600015", "600016", "600018", "600019", "600025",
    "600028", "600029", "600030", "600031", "600036", "600048", "600050", "600085",
    "600104", "600111", "600115", "600118", "600132", "600150", "600176", "600183",
    "600188", "600196", "600233", "600276", "600309", "600346", "600362", "600406",
    "600415", "600426", "600436", "600438", "600460", "600482", "600489", "600519",
    "600547", "600570", "600585", "600588", "600600", "600660", "600690", "600703",
    "600745", "600760", "600763", "600795", "600809", "600837", "600845", "600886",
    "600887", "600893", "600900", "600905", "600919", "600926", "600941", "600958",
    "600989", "601006", "601012", "601021", "601058", "601066", "601088", "601100",
    "601111", "601117", "601138", "601166", "601186", "601211", "601225", "601229",
    "601238", "601288", "601318", "601328", "601336", "601360", "601390", "601398",
    "601456", "601601", "601607", "601615", "601618", "601628", "601633", "601658",
    "601668", "601669", "601688", "601698", "601699", "601728", "601766", "601788",
    "601800", "601808", "601816", "601818", "601838", "601857", "601868", "601872",
    "601877", "601878", "601881", "601888", "601898", "601899", "601901", "601919",
    "601939", "601985", "601988", "601989", "601995", "601998",
    "603019", "603160", "603259", "603260", "603288", "603290", "603296", "603369",
    "603392", "603501", "603659", "603799", "603833", "603899", "603939",
    "688008", "688009", "688012", "688036", "688041", "688065", "688082", "688099",
    "688111", "688126", "688169", "688187", "688223", "688256", "688271", "688303",
    "688347", "688396", "688472", "688475", "688484", "688506", "688561", "688568",
    "688599", "688777", "688819", "688981",
]

# 中证500部分样本
CSI500_SAMPLE = [
    "000021", "000027", "000039", "000060", "000066", "000069", "000089", "000155",
    "000400", "000401", "000402", "000408", "000415", "000423", "000426", "000513",
    "000519", "000528", "000538", "000543", "000547", "000553", "000559", "000563",
    "000581", "000591", "000598", "000617", "000623", "000629", "000630", "000636",
    "000683", "000686", "000703", "000708", "000709", "000712", "000717", "000718",
    "000723", "000728", "000729", "000733", "000735", "000738", "000739", "000750",
    "000758", "000762",
]


def get_stock_pool(pool_name="csi300"):
    """获取股票池代码列表"""
    pools = {
        "csi300": CSI300_SAMPLE,
        "csi500": CSI500_SAMPLE,
        "hs300": CSI300_SAMPLE,
        "zz500": CSI500_SAMPLE,
    }
    if pool_name in pools:
        codes = pools[pool_name]
        print(f"股票池 {pool_name}: {len(codes)} 只股票")
        return codes

    # 尝试从 AKShare 获取
    try:
        import akshare as ak
        index_map = {
            "csi300": "000300",
            "csi500": "000905",
            "csi800": "000906",
            "csi1000": "000852",
        }
        idx = index_map.get(pool_name, pool_name)
        df = ak.index_stock_cons_csindex(symbol=idx)
        codes = df["成分券代码"].tolist()
        print(f"AKShare 获取 {pool_name}: {len(codes)} 只股票")
        return codes
    except Exception as e:
        print(f"AKShare 获取失败: {e}，使用预设池")
        return CSI300_SAMPLE


# ============================================================
# 2. 从通达信 TQ 获取行情数据
# ============================================================

def fetch_stock_data_tdx(code: str, start_date: str, end_date: str,
                         fields: List[str] = None) -> Optional[pd.DataFrame]:
    """
    从通达信 TQ 获取单只股票日线数据

    参数
    ----
    code : str
        股票代码，如 "600519"
    start_date : str
        开始日期，格式 YYYYMMDD
    end_date : str
        结束日期，格式 YYYYMMDD
    fields : list
        需要的字段，默认 ['open','high','low','close','volume','amount']

    返回
    ----
    pd.DataFrame or None
        包含 date + 各字段的 DataFrame
    """
    tq = get_tq()

    if fields is None:
        fields = ['open', 'high', 'low', 'close', 'volume', 'amount']

    # 通达信代码格式：加 .SH 或 .SZ
    code_tdx = _to_tdx_code(code)

    try:
        data = tq.get_market_data(
            field_list=fields,
            stock_list=[code_tdx],
            period='1d',
            start_time=start_date,
            end_time=end_date,
        )

        if data is None:
            return None

        # 解析返回数据
        df = _parse_tdx_response(data, code_tdx, fields)
        return df

    except Exception as e:
        # 静默处理，让上层决定是否重试
        return None


def _to_tdx_code(code: str) -> str:
    """将纯数字代码转为通达信格式"""
    code = code.strip()
    if '.' in code:
        return code
    if code.startswith(('6', '9')):
        return f"{code}.SH"
    elif code.startswith(('0', '3', '2')):
        return f"{code}.SZ"
    elif code.startswith('8'):
        if code.startswith('83'):
            return f"{code}.SZ"  # 创业板
        return f"{code}.SH"  # 科创板
    return f"{code}.SH"


def _parse_tdx_response(data, code_tdx: str, fields: List[str]) -> Optional[pd.DataFrame]:
    """解析 TQ 返回的多种数据格式"""
    tq = get_tq()
    df = pd.DataFrame()

    # 方式1: 字典 {field: DataFrame}
    if isinstance(data, dict):
        for field in fields:
            for fname in [field, field.capitalize()]:
                if fname in data:
                    field_data = data[fname]
                    if isinstance(field_data, pd.DataFrame):
                        if code_tdx in field_data.columns:
                            df[field] = field_data[code_tdx]
                        elif not field_data.empty:
                            df[field] = field_data.iloc[:, 0]
                    elif isinstance(field_data, pd.Series):
                        df[field] = field_data
                    elif isinstance(field_data, list):
                        df[field] = pd.Series(field_data)
                    break

    # 方式2: DataFrame 格式
    if df.empty and isinstance(data, pd.DataFrame):
        for field in fields:
            for fname in [field, field.capitalize(), field.upper(), field.lower()]:
                if fname in data.columns:
                    df[field] = data[fname]
                    break

    # 方式3: 使用 price_df
    if df.empty and hasattr(tq, 'price_df'):
        for field in fields:
            for fname in [field, field.capitalize()]:
                try:
                    field_df = tq.price_df(data, fname, column_names=[code_tdx])
                    if field_df is not None and not field_df.empty:
                        if code_tdx in field_df.columns:
                            df[field] = field_df[code_tdx]
                        else:
                            df[field] = field_df.iloc[:, 0]
                        break
                except Exception:
                    continue

    if df.empty:
        return None

    # 日期作为索引
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    return df


# ============================================================
# 3. 导出为 CSV 中间格式（Qlib dump_bin.py 能识别的格式）
# ============================================================

def export_to_csv(codes: List[str], start_date: str, end_date: str,
                  output_dir: str, retry: int = 2, delay: float = 0.5):
    """
    将通达信数据导出为每只股票一个 CSV 文件
    格式兼容 Qlib dump_bin.py: 包含 symbol, date 及 OHLCV 字段
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_codes = []

    for i, code in enumerate(codes):
        print(f"  [{i+1}/{len(codes)}] {code}", end=" ", flush=True)

        csv_file = output_path / f"{code.lower()}.csv"

        # 跳过已存在的
        if csv_file.exists():
            df_exist = pd.read_csv(csv_file)
            if len(df_exist) > 100:
                success_count += 1
                print("✓ (缓存)")
                continue

        # 重试获取
        df = None
        for attempt in range(retry + 1):
            df = fetch_stock_data_tdx(code, start_date, end_date)
            if df is not None and len(df) > 50:
                break
            if attempt < retry:
                time.sleep(delay)

        if df is None or len(df) < 50:
            fail_codes.append(code)
            print("✗ 数据不足")
            continue

        # 添加 symbol 和 date 列
        df = df.reset_index()
        df.rename(columns={'index': 'date'}, inplace=True)
        df['symbol'] = code.lower()

        # 确保列顺序
        cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
        if 'amount' in df.columns:
            cols.append('amount')
        df = df[[c for c in cols if c in df.columns]]

        df.to_csv(csv_file, index=False)
        success_count += 1
        print(f"✓ {len(df)}条")

        # 避免请求过快
        if i < len(codes) - 1:
            time.sleep(delay)

    print(f"\n导出完成: 成功 {success_count}/{len(codes)}, 失败 {len(fail_codes)}")
    if fail_codes:
        print(f"失败股票: {fail_codes}")
    return success_count


# ============================================================
# 4. 转换为 Qlib .bin 格式
# ============================================================

def dump_to_qlib(csv_dir: str, qlib_dir: str, freq: str = "day",
                 max_workers: int = 8, include_fields: str = "open,high,low,close,volume,amount"):
    """
    将 CSV 数据转换为 Qlib 标准 .bin 格式

    调用 qlib 自带的 dump_bin.py 进行转换
    """
    csv_path = Path(csv_dir)
    if not csv_path.exists():
        print(f"错误: CSV 目录不存在 {csv_dir}")
        return False

    # 统计 CSV 文件
    csv_files = list(csv_path.glob("*.csv"))
    print(f"CSV 文件: {len(csv_files)} 个")

    try:
        from qlib.scripts.dump_bin import DumpDataAll
    except ImportError:
        print("错误: 无法导入 qlib.scripts.dump_bin")
        print("请确保 qlib 已安装: pip install pyqlib")
        print("或添加 qlib-main 到 Python 路径")
        # 尝试手动添加路径
        qlib_path = Path(__file__).parent / "qlib-main"
        if qlib_path.exists():
            sys.path.insert(0, str(qlib_path))
            from qlib.scripts.dump_bin import DumpDataAll
        else:
            return False

    dumper = DumpDataAll(
        data_path=str(csv_path),
        qlib_dir=str(qlib_dir),
        freq=freq,
        max_workers=max_workers,
        date_field_name="date",
        symbol_field_name="symbol",
        include_fields=include_fields,
        file_suffix=".csv",
    )
    dumper.dump()
    print(f"\nQlib 数据已生成到: {qlib_dir}")
    return True


# ============================================================
# 5. 一键完成
# ============================================================

def do_all(pool: str, start_date: str, end_date: str, qlib_dir: str,
           csv_dir: str = None, max_stocks: int = 100):
    """一步完成: 导出 + 转换"""
    if csv_dir is None:
        csv_dir = "./tdx_csv_data"

    # Step 1: 获取股票池
    codes = get_stock_pool(pool)
    if max_stocks and len(codes) > max_stocks:
        codes = codes[:max_stocks]
        print(f"限制股票数量: {max_stocks}")

    # Step 2: 导出 CSV
    print(f"\n{'='*60}")
    print(f"[Step 1] 从通达信导出数据到 CSV")
    print(f"日期范围: {start_date} ~ {end_date}")
    print(f"{'='*60}")
    count = export_to_csv(codes, start_date, end_date, csv_dir)

    if count == 0:
        print("没有成功导出任何股票数据，退出")
        return

    # Step 3: 转换为 Qlib 格式
    print(f"\n{'='*60}")
    print(f"[Step 2] 转换为 Qlib .bin 格式")
    print(f"{'='*60}")
    dump_to_qlib(csv_dir, qlib_dir)

    print(f"\n{'='*60}")
    print(f"完成! Qlib 数据目录: {qlib_dir}")
    print(f"使用方式:")
    print(f"  import qlib")
    print(f"  from qlib.config import C")
    print(f"  C.set_uri('file://{qlib_dir}')")
    print(f"  # 然后即可使用 TDXGS / Alpha158 等 DataHandler")
    print(f"{'='*60}")


# ============================================================
# 6. 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="通达信本地数据 → Qlib 格式转换工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 导出通达信数据为 CSV
  python tdx_data_to_qlib.py export --pool csi300 --start 20230101 --end 20240628

  # CSV 转 Qlib .bin
  python tdx_data_to_qlib.py dump --csv_dir ./tdx_csv_data --qlib_dir ./tdx_qlib_data

  # 一键完成（最多100只股票）
  python tdx_data_to_qlib.py all --pool csi300 --qlib_dir ./tdx_qlib_data

  # 测试通达信连接
  python tdx_data_to_qlib.py test
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # export 子命令
    p_export = subparsers.add_parser("export", help="导出通达信数据为 CSV")
    p_export.add_argument("--pool", default="csi300", help="股票池 (csi300/csi500/自定义代码)")
    p_export.add_argument("--start", default="20230101", help="开始日期 YYYYMMDD")
    p_export.add_argument("--end", default="20240628", help="结束日期 YYYYMMDD")
    p_export.add_argument("--output", default="./tdx_csv_data", help="CSV 输出目录")
    p_export.add_argument("--max", type=int, default=100, help="最大股票数")

    # dump 子命令
    p_dump = subparsers.add_parser("dump", help="CSV 转 Qlib .bin")
    p_dump.add_argument("--csv_dir", default="./tdx_csv_data", help="CSV 目录")
    p_dump.add_argument("--qlib_dir", default="./tdx_qlib_data", help="Qlib 数据输出目录")
    p_dump.add_argument("--workers", type=int, default=8, help="并行线程数")

    # all 子命令
    p_all = subparsers.add_parser("all", help="一键完成导出+转换")
    p_all.add_argument("--pool", default="csi300", help="股票池")
    p_all.add_argument("--start", default="20230101", help="开始日期")
    p_all.add_argument("--end", default="20240628", help="结束日期")
    p_all.add_argument("--qlib_dir", default="./tdx_qlib_data", help="Qlib 输出目录")
    p_all.add_argument("--csv_dir", default="./tdx_csv_data", help="CSV 中间目录")
    p_all.add_argument("--max", type=int, default=100, help="最大股票数")

    # test 子命令
    p_test = subparsers.add_parser("test", help="测试通达信连接")

    args = parser.parse_args()

    if args.command == "export":
        codes = get_stock_pool(args.pool)
        if args.max:
            codes = codes[:args.max]
        print(f"导出 {len(codes)} 只股票, {args.start}~{args.end}")
        export_to_csv(codes, args.start, args.end, args.output)

    elif args.command == "dump":
        dump_to_qlib(args.csv_dir, args.qlib_dir, max_workers=args.workers)

    elif args.command == "all":
        do_all(args.pool, args.start, args.end, args.qlib_dir,
               csv_dir=args.csv_dir, max_stocks=args.max)

    elif args.command == "test":
        print("测试通达信 TQ 连接...")
        try:
            tq = get_tq()
            print("[OK] TQ 接口初始化成功")

            # 测试获取一只股票
            code = "600519.SH"
            print(f"测试获取 {code} 最近5天数据...")
            data = tq.get_market_data(
                field_list=['open', 'high', 'low', 'close', 'volume'],
                stock_list=[code],
                period='1d',
                count=5,
            )
            if data is not None:
                print(f"[OK] 数据获取成功")
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, pd.DataFrame):
                            print(f"  {k}: shape={v.shape}")
                else:
                    print(f"  type={type(data)}")
            else:
                print("[FAIL] 数据为空")
        except Exception as e:
            print(f"[FAIL] {e}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
