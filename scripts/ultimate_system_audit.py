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

def run_ultimate_system_audit():
    """
    全策略、白盒化系统物理审计
    """
    print("--- STARTING ULTIMATE SYSTEM PHYSICS AUDIT ---")

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

    strats = list_vec_strategies()
    print(f"Detected strategies: {strats}")

    class Params:
        rsrs_window = 18
        zscore_window = 300
        top_n = 5
        annual_factor = 252.0
        def to_dict(self): return {"top_n": 5}

    excel_path = Path("results/full_system_whitebox_audit.xlsx")
    plot_path = Path("results/physics_audit_plot.png")
    Path("results").mkdir(exist_ok=True)

    with pd.ExcelWriter(excel_path) as writer:
        # Sheet 1: Physical Constants (Stock 0)
        df_phys = pd.DataFrame({
            'Date': runner.dates,
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
            print(f"Auditing strategy: {sname}...")
            try:
                res = runner.run(sname, Params(), runner.dates[0], runner.dates[-1])

                # Trace Sheet (Signal vs Execution)
                # T日目标权重 vs T+1日执行价
                df_strat = pd.DataFrame({
                    'Date': runner.dates,
                    'Target_Weight_T': res.target_weights[0]
                })

                # Join trades
                t_df = res.trades_df[res.trades_df['code'] == runner.codes[0]].copy()
                if not t_df.empty:
                    # 将 Trade price 映射回日期以便查看
                    trade_map = t_df.set_index('date')['price'].to_dict()
                    df_strat['Exec_Price_T_or_T1'] = df_strat['Date'].map(trade_map)

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
    print(f"DONE: Full report generated at {excel_path}")
    print(f"DONE: Full plot generated at {plot_path}")

if __name__ == "__main__":
    run_ultimate_system_audit()
