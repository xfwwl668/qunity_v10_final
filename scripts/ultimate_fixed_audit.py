import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# 路径修复
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine.fast_runner_v10 import FastRunnerV10
from src.strategies.registry import list_vec_strategies

def run_ultimate_fixed_audit():
    print("\n--- FINAL PHYSICS AUDIT: OFFICE 2007 COMPATIBLE ---")

    cfg = {
        "npy_dir": "data/npy_physics_lab",
        "initial_cash": 1000000.0,
        "commission_rate": 0.0,
        "stamp_tax": 0.0,
        "slippage_rate": 0.0,
        "vol_multiplier": 100,
        "data": { "npy_dir": "data/npy_physics_lab" }
    }

    runner = FastRunnerV10(cfg)
    runner.load_data()

    # --- 核心修正：从数据包中获取物理变量 ---
    # 根据 V10 源码，日期和代码在 _data_bundle 中
    dates = list(runner._data_bundle.dates)
    codes = list(runner._data_bundle.codes)

    # 物理矩阵直接在 runner 根属性
    p_open = runner.open[0]
    p_close = runner.close[0]
    p_high = runner.high[0]
    p_low = runner.low[0]

    strats = list_vec_strategies()
    print(f"Loaded {len(strats)} strategies for auditing.")

    class Params:
        rsrs_window = 18
        zscore_window = 300
        top_n = 5
        annual_factor = 252.0
        def to_dict(self): return {"top_n": 5}

    excel_path = ROOT / "results" / "full_system_whitebox_audit_v3.xlsx"
    plot_path = ROOT / "results" / "physics_audit_plot_v3.png"
    Path(excel_path.parent).mkdir(exist_ok=True)

    # 强制使用 xlsxwriter 引擎以兼容 Office 2007
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        # Sheet 0: Physical Base Truth
        pd.DataFrame({
            'Date': dates,
            'Open': p_open, 'Close': p_close, 'High': p_high, 'Low': p_low
        }).to_excel(writer, sheet_name='PHYSICAL_BASE', index=False)

        fig, axes = plt.subplots(len(strats), 1, figsize=(10, 3*len(strats)))
        if len(strats) == 1: axes = [axes]

        for idx, sname in enumerate(strats):
            print(f" -> Auditing {sname}...")
            try:
                res = runner.run(sname, Params(), dates[0], dates[-1])

                # Signal Trace
                df_trace = pd.DataFrame({
                    'Date': dates,
                    'Target_Weight_T': res.target_weights[0]
                })

                # Execution Log
                t_df = res.trades_df[res.trades_df['code'] == codes[0]]
                if not t_df.empty:
                    trade_map = t_df.set_index('date')['price'].to_dict()
                    df_trace['Exec_Price_T1'] = df_trace['Date'].map(trade_map)

                df_trace.to_excel(writer, sheet_name=f'Audit_{sname[:20]}', index=False)

                # Graphics
                axes[idx].plot(res.nav_array, label=f"NAV: {sname}")
                axes[idx].set_title(sname)
                axes[idx].grid(True, alpha=0.2)
            except Exception as e:
                print(f"    ! Failed {sname}: {e}")

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"\n物理审计全量包已就绪！")
    print(f"1. Excel (Office 2007+): {excel_path}")
    print(f"2. 全景对比图: {plot_path}")

if __name__ == "__main__":
    run_ultimate_fixed_audit()
