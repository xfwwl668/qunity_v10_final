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

def run_v8_final_audit():
    print("\n--- [V8-FIX] EXTREME WHITE-BOX AUDIT: STARTING FINAL VALIDATION ---")

    lab_dir = ROOT / "data" / "npy_physics_lab"
    cfg = {
        "npy_dir": str(lab_dir),
        "initial_cash": 1000000.0,
        "commission_rate": 0.0,
        "stamp_tax": 0.0,
        "slippage_rate": 0.0,
        "vol_multiplier": 1,
        "data": { "npy_dir": str(lab_dir) },
        "hfq": True
    }

    runner = FastRunnerV10(cfg)
    runner.load_data()

    # 正确的数据访问路径：runner._data 字典
    dates = runner._meta.get('dates', [])
    codes = runner._meta.get('codes', [])
    raw_open = runner._data['open'][0]
    raw_close = runner._data['close'][0]

    print(f"Physical Data: {len(dates)} days, Code 0: {codes[0]}")
    print(f"Day 0 Open: {raw_open[0]:.4f}, Day 0 Close: {raw_close[0]:.4f}")

    strats = list_vec_strategies()
    print(f"Auditing {len(strats)} strategies...")

    class AuditParams:
        def __init__(self):
            self.rsrs_window = 18
            self.zscore_window = 600
            self.rsrs_threshold = 0.7
            self.mom_threshold = 0.03
            self.volume_confirm_ratio = 1.5
            self.top_n = 5
            self.annual_factor = 252.0
            self.max_single_pos = 0.2
            self.risk_budget = 0.02
            self.atr_multiplier = 2.0
            self.factor_ema_span = 5
            self.warmup_override = 100
        def to_dict(self):
            return {k: v for k, v in self.__dict__.items()}

    excel_path = ROOT / "results" / "full_system_audit_v8.xlsx"
    plot_path = ROOT / "results" / "physics_audit_plot_v8.png"
    Path(excel_path.parent).mkdir(exist_ok=True)

    nav_data = {}

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_phys = pd.DataFrame({
            'Date': dates,
            'Physical_Open_T1': raw_open,
            'Physical_Close_T': raw_close
        })
        df_phys.to_excel(writer, sheet_name='PHYSICAL_TRUTH', index=False)

        for sname in strats:
            try:
                res = runner.run(sname, AuditParams(), dates[0], dates[-1])
                nav_data[sname] = res.nav_array
                df_audit = pd.DataFrame({
                    'Date': dates,
                    'Weight_T': res.target_weights[0]
                })
                if not res.trades_df.empty:
                    t_df = res.trades_df[res.trades_df['code'] == codes[0]]
                    if not t_df.empty:
                        trade_map = t_df.set_index('date')['price'].to_dict()
                        df_audit['Exec_Price_T1'] = df_audit['Date'].map(trade_map)
                df_audit.to_excel(writer, sheet_name=f"Audit_{sname[:20]}", index=False)
            except Exception as e:
                print(f"  !! Failed {sname}: {e}")

    plt.figure(figsize=(12, 8))
    for sname, nav in nav_data.items():
        plt.plot(nav, label=sname, alpha=0.7)
    plt.title("Q-UNITY V10 Physical Integrity Audit (V8-FIX)")
    plt.axvline(x=600, color='r', linestyle='--', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"SUCCESS: {excel_path} and {plot_path} generated.")

if __name__ == "__main__":
    run_v8_final_audit()
