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
from src.strategies.registry import list_vec_strategies, get_alpha_fn

def plot_final_nav_curves():
    print("\n--- GENERATING FINAL EQUITY CURVES (NAV) FOR ALL STRATEGIES ---")

    lab_dir = ROOT / "data" / "npy_physics_lab"
    plot_path = ROOT / "results" / "complete_equity_curves_final.png"
    cfg = {"npy_dir": str(lab_dir), "initial_cash": 1000000.0, "data": {"npy_dir": str(lab_dir)}}
    runner = FastRunnerV10(cfg)
    runner.load_data()

    dates = runner._meta['dates']
    strats = list_vec_strategies()

    class AuditParams:
        def __init__(self):
            self.rsrs_window = 18; self.zscore_window = 100
            self.rsrs_threshold = 0.2; self.mom_threshold = 0.01; self.volume_confirm_ratio = 1.1
            self.top_n = 5; self.warmup_override = 50
            self.max_single_pos = 0.08 # 恢复标准 0.08 风控
        def to_dict(self): return {"top_n": 5}

    plt.figure(figsize=(15, 10))

    for sname in strats:
        try:
            res = runner.run(sname, AuditParams(), dates[0], dates[-1])
            if res.nav_array.max() > 1e-6:
                plt.plot(res.nav_array, label=f"{sname} (Sharpe: {res.sharpe_ratio:.2f})")
        except:
            pass

    plt.title("Q-UNITY V10 Ultimate Auditor View: Equity Curves (NAV)", fontsize=14)
    plt.xlabel("Days (Synthetic 1200 Day Sine Universe)")
    plt.ylabel("Normalized NAV (Initial: 1.0M)")
    plt.axvline(x=600, color='r', linestyle='--', alpha=0.3, label="HFQ Scale Jump (D600)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"SUCCESS: {plot_path} generated.")

if __name__ == "__main__":
    plot_final_nav_curves()
