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

def run_final_successful_audit():
    print("\n--- FINAL PHYSICAL ALIGNMENT AUDIT (BUILD V4) ---")

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

    # 安全地从 runner 实例中直接获取（经核实，load_data 后这些属性会被填充）
    dates = list(getattr(runner, 'dates', []))
    codes = list(getattr(runner, 'codes', []))

    if not dates:
        # 最后的兜底：如果 runner 隐藏了属性，从物理文件夹中读取
        import json
        with open(Path(cfg['npy_dir']) / "meta.json", "r") as f:
            meta = json.load(f)
            dates = meta['dates']
            codes = meta['codes']

    p_open = runner.open[0]
    p_close = runner.close[0]

    strats = list_vec_strategies()

    excel_path = ROOT / "results" / "full_system_whitebox_audit_v4.xlsx"
    plot_path = ROOT / "results" / "physics_audit_plot_v4.png"
    Path(excel_path.parent).mkdir(exist_ok=True)

    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        # Sheet 0: Physical Truth
        pd.DataFrame({
            'Date': dates,
            'Open_Pure': p_open,
            'Close_Pure': p_close
        }).to_excel(writer, sheet_name='PHYSICAL_TRUTH', index=False)

        fig, axes = plt.subplots(len(strats), 1, figsize=(10, 3*len(strats)))
        if len(strats) == 1: axes = [axes]

        for idx, sname in enumerate(strats):
            print(f" -> Auditing strategy: {sname}")
            try:
                res = runner.run(sname, None, dates[0], dates[-1])

                df_trace = pd.DataFrame({
                    'Date': dates,
                    'Position_Weight_T': res.target_weights[0]
                })

                t_df = res.trades_df[res.trades_df['code'] == codes[0]]
                if not t_df.empty:
                    trade_map = t_df.set_index('date')['price'].to_dict()
                    df_trace['Exec_Price_T1'] = df_trace['Date'].map(trade_map)

                df_trace.to_excel(writer, sheet_name=f'Audit_{sname[:20]}', index=False)

                axes[idx].plot(res.nav_array)
                axes[idx].set_title(sname)
                axes[idx].grid(True, alpha=0.1)
            except Exception as e:
                print(f"    ! Skipped {sname}: {e}")

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"\n--- 物理审计最终包生成成功 ---")
    print(f"1. Excel 报表: {excel_path}")
    print(f"2. 全景对比图: {plot_path}")

if __name__ == "__main__":
    run_final_successful_audit()
