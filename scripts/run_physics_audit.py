import sys
import numpy as np
import pandas as pd
from pathlib import Path
import json

# 路径修复
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine.fast_runner_v10 import FastRunnerV10
from src.engine.risk_config import RiskConfig

def run_physics_audit():
    """
    加载仿真数据，拷打引擎物理逻辑
    """
    print("\n" + "="*70)
    print("  Q-UNITY V10 PHYSICS ENGINE AUDIT (No-Unicode Version)")
    print("="*70)

    # 1. 仿真配置
    cfg = {
        "npy_dir": "data/npy_physics_lab",
        "initial_cash": 1000000.0,
        "commission_rate": 0.0003,
        "stamp_tax": 0.0005,
        "slippage_rate": 0.0,
        "vol_multiplier": 100,
        "data": {
            "npy_dir": "data/npy_physics_lab"
        }
    }

    # 2. 加载仿真数据
    print("[STEP 1] Loading synthetic matrix (N=10, T=1200)...")
    runner = FastRunnerV10(cfg)
    runner.load_data()

    # [FIX] V10 引擎数据存放在 self.arrays 中
    raw_open = runner.arrays.get("open")
    if raw_open is None:
        # 尝试从 runner 直接获取 (兼容性处理)
        raw_open = getattr(runner, "open", None)

    raw_dates = runner.dates
    raw_codes = runner.codes

    # 3. 运行 Sniper 策略测试
    class Params:
        rsrs_window = 18
        zscore_window = 300
        top_n = 5
        annual_factor = 252.0
        def to_dict(self): return {"top_n": 5, "rsrs_window": 18}

    print("[STEP 2] Running backtest on synthetic data...")
    start, end = raw_dates[0], raw_dates[-1]

    try:
        res = runner.run("sniper_v13_hfq", Params(), start, end)
    except Exception as e:
        print(f"[ERROR] Engine crashed: {e}")
        import traceback; traceback.print_exc()
        return

    # 4. 核心物理三项断言
    print("\n" + "-"*30 + " AUDIT REPORT " + "-"*30)

    # --- A. 预热期审计 ---
    target_weights = getattr(res, "target_weights", None)
    if target_weights is not None:
        pre_warmup_mask = target_weights[:, :300] != 0
        if np.any(pre_warmup_mask):
            leak_day = np.where(pre_warmup_mask)[1][0]
            print(f"[FAIL-01] Pre-warmup Leak: Signal found at Day {leak_day}. (Valid_mask violated)")
        else:
            print(f"[PASS-01] Pre-warmup Integrity: No trades before Day 300.")

    # --- B. T+1 执行价格审计 ---
    trades = getattr(res, "trades_df", None)
    if trades is not None and not trades.empty:
        err_list = []
        for idx, row in trades.iterrows():
            code = row['code']
            exec_date = str(row['date'])
            exec_price = float(row['price'])
            action = row.get('action', 'buy')

            n_idx = raw_codes.index(code)
            t_idx = raw_dates.index(exec_date)
            truth_open = float(raw_open[n_idx, t_idx])

            if abs(exec_price - truth_open) > 1e-4:
                err_list.append(f"Day {t_idx} {code} {action} mismatch: Realized={exec_price:.4f} vs SyntheticOpen={truth_open:.4f}")

        if err_list:
            print(f"[FAIL-02] Look-ahead Bias: {len(err_list)} price mismatches found.")
            print(f"      Example: {err_list[0]}")
        else:
            print(f"[PASS-02] T+1 Alignment: All executions strictly at Next-Day OPEN.")
    else:
        print("[WARN] No trades generated for audit.")

    # --- C. HFQ 平滑性审计 ---
    nav = res.nav_array
    if len(nav) > 602:
        max_daily_ret = np.max(np.abs(np.diff(nav) / nav[:-1]))
        if max_daily_ret > 0.5:
            print(f"[FAIL-03] HFQ Rupture: Day 601 return {max_daily_ret:.2%}. Bases shift contaminated NAV.")
        else:
            print(f"[PASS-03] Quant-Scale Adaptive: Day 600 base shift didn't contaminate NAV.")

    print("\n[RESULT SUMMARY]")
    print(f"  * Sharpe: {res.sharpe_ratio:.3f}")
    print(f"  * Annual Return: {res.annual_return:.1%}")
    print(f"  * Annual Turnover: {res.turnover:.1f}%/year")
    print(f"  * Final NAV: {nav[-1]:.4f}")
    print("="*70 + "\n")

if __name__ == "__main__":
    run_physics_audit()
