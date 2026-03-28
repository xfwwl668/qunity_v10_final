import sys
import numpy as np
import pandas as pd
from pathlib import Path

# 路径修复
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine.fast_runner_v10 import FastRunnerV10

def export_excel_audit():
    """
    导出含中间过程的白盒审计数据
    """
    print("--- EXPORTING WHITE-BOX AUDIT DATA TO EXCEL ---")

    cfg = {
        "npy_dir": "data/npy_physics_lab",
        "initial_cash": 1000000.0,
        "commission_rate": 0.0,
        "stamp_tax": 0.0,
        "slippage_rate": 0.0,
        "data": { "npy_dir": "data/npy_physics_lab" }
    }

    runner = FastRunnerV10(cfg)
    runner.load_data()

    # 1. 运行回测获取信号全矩阵
    class Params:
        rsrs_window = 18
        zscore_window = 300
        top_n = 3
        annual_factor = 252.0
        def to_dict(self): return {"top_n": 3}

    res = runner.run("sniper_v13_hfq", Params(), runner.dates[0], runner.dates[-1])

    # 2. 抓取 Stock 0 (sh.600000) 的全程细节
    stock_idx = 0
    code = runner.codes[stock_idx]

    # 物理数据
    df_raw = pd.DataFrame({
        'Date': runner.dates,
        'Open': runner._data_bundle.arrays['open'][stock_idx],
        'High': runner._data_bundle.arrays['high'][stock_idx],
        'Low': runner._data_bundle.arrays['low'][stock_idx],
        'Close': runner._data_bundle.arrays['close'][stock_idx],
        'Index': runner._data_bundle.arrays['market_index'][0]
    })

    # 由于 Sniper 的中间变量在策略函数内部，我们通过 replay 逻辑或直接从结果矩阵中还原
    # 这里我们构造一个 Trace 表，放最关键的 Z-Score 和 Signal

    # 注意：res.target_weights 记录的是 T 日收盘后的持仓目标
    df_trace = pd.DataFrame({
        'Date': runner.dates,
        'Raw_Weight': res.target_weights[stock_idx], # T 日收盘信号
        'Index_Value': runner._data_bundle.arrays['market_index'][0],
    })

    # 3. 抓取 Engine 的成交明细
    df_trades = res.trades_df[res.trades_df['code'] == code].copy()

    # 4. 写入 Excel
    excel_path = ROOT / "results" / "physics_lab_whitebox.xlsx"
    Path(excel_path.parent).mkdir(exist_ok=True)

    with pd.ExcelWriter(excel_path) as writer:
        df_raw.to_excel(writer, sheet_name='Raw_OHLCV', index=False)
        df_trace.to_excel(writer, sheet_name='Strategy_Signal_Trace', index=False)
        df_trades.to_excel(writer, sheet_name='Engine_Execution_Log', index=False)

    print(f"DONE: Excel export complete -> {excel_path}")

if __name__ == "__main__":
    export_excel_audit()
