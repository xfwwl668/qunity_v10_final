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

def run_fixed_system_audit():
    """
    修正版审计脚本：Office 2007 兼容 + 稳健属性访问
    """
    print("\n--- INITIATING ROBUST FULL-SYSTEM AUDIT ---")

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

    # 安全获取日期和代码
    # 查阅源码发现 FastRunnerV10 的日期列表在 self.dates 中
    dates = list(runner.dates)
    codes = list(runner.codes)
    print(f"Data Loaded: {len(codes)} stocks, {len(dates)} days.")

    strats = list_vec_strategies()
    print(f"Detected strategies: {strats}")

    class Params:
        rsrs_window = 18
        zscore_window = 300
        top_n = 5
        annual_factor = 252.0
        def to_dict(self): return {"top_n": 5}

    excel_path = ROOT / "results" / "full_system_whitebox_audit_v2.xlsx"
    plot_path = ROOT / "results" / "physics_audit_plot_v2.png"
    Path(excel_path.parent).mkdir(exist_ok=True)

    # 使用 xlsxwriter 确保 Office 2007 兼容性
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        # Sheet 1: Physical Constants
        df_phys = pd.DataFrame({
            'Date': dates,
            'Open': runner.open[0],
            'Close': runner.close[0],
            'High': runner.high[0],
            'Low': runner.low[0]
        })
        df_phys.to_excel(writer, sheet_name='PHYSICAL_TRUTH', index=False)

        # Audit Each Strategy
        fig, axes = plt.subplots(len(strats), 1, figsize=(12, 4*len(strats)))
        if len(strats) == 1: axes = [axes]

        for idx, sname in enumerate(strats):
            print(f"Auditing: {sname}...")
            try:
                res = runner.run(sname, Params(), dates[0], dates[-1])

                # Trace Data
                df_strat = pd.DataFrame({
                    'Date': dates,
                    'Target_Weight_Stock0_T': res.target_weights[0]
                })

                # Join trades for Stock 0
                t_df = res.trades_df[res.trades_df['code'] == codes[0]].copy()
                if not t_df.empty:
                    # 成交价映射
                    trade_map = t_df.set_index('date')['price'].to_dict()
                    df_strat['Exec_Price_T1'] = df_strat['Date'].map(trade_map)

                df_strat.to_excel(writer, sheet_name=f'Audit_{sname[:20]}', index=False)

                # Plotting
                ax = axes[idx]
                ax.plot(res.nav_array, label=f"{sname} NAV")
                ax.set_title(f"Simulation: {sname}")
                ax.grid(True, alpha=0.3)
                ax.legend()
            except Exception as e:
                print(f"Error auditing {sname}: {e}")

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"\n✅ SUCCESS: ")
    print(f"   Excel: {excel_path}")
    print(f"   Plot:  {plot_path}")

if __name__ == "__main__":
    run_fixed_system_audit()
