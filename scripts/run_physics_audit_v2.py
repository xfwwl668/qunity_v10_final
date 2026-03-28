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

def run_demonstrative_audit():
    """
    不仅审计，还要产出可视化的物理证据
    """
    print("\n" + "="*70)
    print("  Q-UNITY V10 PHYSICS AUDIT: VISUAL EVIDENCE GENERATOR")
    print("="*70)

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

    # 动态获取矩阵
    raw_close = runner._data_bundle.arrays['close']
    raw_open = runner._data_bundle.arrays['open']
    raw_dates = runner.dates
    raw_codes = runner.codes

    class Params:
        rsrs_window = 18
        zscore_window = 300
        top_n = 3
        annual_factor = 252.0
        def to_dict(self): return {"top_n": 3}

    res = runner.run("sniper_v13_hfq", Params(), raw_dates[0], raw_dates[-1])

    # --- 1. 打印物理成交对照日志 (前10次交易) ---
    print("\n[EVIDENCE 1] Physical Execution Log (Signal vs Execution)")
    print("-" * 85)
    print(f"{'Stock':<10} | {'Signal Date (T)':<15} | {'T Close':<10} | {'Exec Date (T+1)':<15} | {'Exec Price'}")
    print("-" * 85)

    trades = res.trades_df
    log_count = 0
    if not trades.empty:
        # 获取权重矩阵找出信号日
        # 信号在 T 日收盘计算，权重变动反映在 T。
        # 实际成交在 T+1 的 Open。
        for idx, row in trades.sort_values('date').head(10).iterrows():
            code = row['code']
            exec_date = str(row['date'])
            exec_price = float(row['price'])

            t_idx_exec = raw_dates.index(exec_date)
            t_idx_signal = t_idx_exec - 1 # 信号产生在执行日前一天

            n_idx = raw_codes.index(code)
            signal_date = raw_dates[t_idx_signal]
            signal_close = raw_close[n_idx, t_idx_signal]

            print(f"{code:<10} | {signal_date:<15} | {signal_close:<10.2f} | {exec_date:<15} | {exec_price:<10.4f}")
            log_count += 1

    # --- 2. 生成可视化大图 ---
    print("\n[EVIDENCE 2] Generating visual plots...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 子图1: 样本股价格 + 买卖点 + HFQ平移
    stock_idx = 0
    prices = raw_close[stock_idx]
    ax1.plot(prices, label=f"Price ({raw_codes[stock_idx]})", color='blue', alpha=0.6)

    # 标注成交点
    sample_trades = trades[trades['code'] == raw_codes[stock_idx]]
    for _, t in sample_trades.iterrows():
        t_pos = raw_dates.index(str(t['date']))
        ax1.scatter(t_pos, t['price'], color='red', s=30, zorder=5)

    ax1.set_title("Physical Price Scale & Execution Points (Note the jump at Day 600)")
    ax1.axvline(x=600, color='gray', linestyle='--', label="HFQ Shift (20x)")
    ax1.set_yscale('log') # 使用对数坐标看清平移后的细节
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2: 策略 NAV
    ax2.plot(res.nav_array, label="Strategy NAV", color='green', lw=2)
    ax2.set_title("Strategy Performance (Synthetic Lab)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = "results/physics_audit_plot.png"
    Path("results").mkdir(exist_ok=True)
    plt.savefig(plot_path)
    print(f"DONE: Evidence plot saved to {plot_path}")
    print("="*70 + "\n")

if __name__ == "__main__":
    run_demonstrative_audit()
