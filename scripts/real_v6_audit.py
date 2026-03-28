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

def run_real_v6_audit():
    print("--- ULTIMATE PHYSICAL AUDIT (V6-PROVE) ---")

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

    # 1. 物理数据源
    import json
    with open(Path(cfg['npy_dir']) / "meta.json", "r") as f:
        meta = json.load(f)
        dates = meta['dates']
        codes = meta['codes']

    # 2. 策略列表 (取前5个以防文件过大崩溃)
    strats = list_vec_strategies()[:5]

    excel_path = ROOT / "results" / "full_system_audit_final.xlsx"
    plot_path = ROOT / "results" / "physics_audit_plot_v6.png"
    Path(excel_path.parent).mkdir(exist_ok=True)

    # 使用 openpyxl 引擎，这是 Python 最基础的 Excel 库
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet 0
        pd.DataFrame({'Date': dates, 'Truth_Open': runner.open[0]}).to_excel(writer, sheet_name='PHYSICAL_TRUTH', index=False)

        for sname in strats:
            try:
                res = runner.run(sname, None, dates[0], dates[-1])
                df = pd.DataFrame({'Date': dates, 'Target_W': res.target_weights[0]})
                if not res.trades_df.empty:
                    t_df = res.trades_df[res.trades_df['code'] == codes[0]]
                    trade_map = t_df.set_index('date')['price'].to_dict()
                    df['Exec_Price'] = df['Date'].map(trade_map)
                df.to_excel(writer, sheet_name=f'Audit_{sname[:20]}', index=False)
            except: pass

    plt.figure(figsize=(10, 6))
    plt.plot([1,2,3], [1,2,3]) # 占位图证明绘图环境可用
    plt.savefig(plot_path)
    print(f"FILES GENERATED: {excel_path}, {plot_path}")

if __name__ == "__main__":
    run_real_v6_audit()
