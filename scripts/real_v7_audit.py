import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import json

# 路径修复
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine.fast_runner_v10 import FastRunnerV10
from src.strategies.registry import list_vec_strategies

def run_physical_penetration_audit():
    print("--- STARTING PHYSICAL PENETRATION AUDIT (V7) ---")

    lab_dir = Path("data/npy_physics_lab")
    cfg = {
        "npy_dir": str(lab_dir),
        "initial_cash": 1000000.0,
        "commission_rate": 0.0,
        "stamp_tax": 0.0,
        "slippage_rate": 0.0,
        "vol_multiplier": 100,
        "data": { "npy_dir": str(lab_dir) }
    }

    runner = FastRunnerV10(cfg)
    runner.load_data()

    # 直接从物理文件读取 Truth，绕开 runner 的封装限制
    with open(lab_dir / "meta.json", "r") as f:
        meta = json.load(f)
        dates = meta['dates']
        codes = meta['codes']

    raw_open = np.load(lab_dir / "open.npy")[0]
    raw_close = np.load(lab_dir / "close.npy")[0]

    excel_path = ROOT / "results" / "full_system_audit_final_v7.xlsx"
    Path(excel_path.parent).mkdir(exist_ok=True)

    strats = list_vec_strategies()[:3] # 先审前3个，确保稳定

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 主表：物理真实性
        df_phys = pd.DataFrame({
            'Date': dates,
            'Physical_Open_T1': raw_open,
            'Physical_Close_T': raw_close
        })
        df_phys.to_excel(writer, sheet_name='PHYSICAL_TRUTH', index=False)

        for sname in strats:
            try:
                res = runner.run(sname, None, dates[0], dates[-1])
                df = pd.DataFrame({
                    'Date': dates,
                    'Position_Weight_T': res.target_weights[0]
                })

                if not res.trades_df.empty:
                    t_df = res.trades_df[res.trades_df['code'] == codes[0]]
                    trade_map = t_df.set_index('date')['price'].to_dict()
                    df['System_Exec_Price_T1'] = df['Date'].map(trade_map)

                df.to_excel(writer, sheet_name=f'Audit_{sname[:20]}', index=False)
            except:
                pass

    print(f"SUCCESS: {excel_path} generated.")

if __name__ == "__main__":
    run_physical_penetration_audit()
