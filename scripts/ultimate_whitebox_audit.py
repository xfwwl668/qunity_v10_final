import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import json
import inspect

# 路径修复
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine.fast_runner_v10 import FastRunnerV10
from src.strategies.registry import list_vec_strategies, get_alpha_fn

def generate_extreme_whitebox_audit():
    print("\n--- [ULTIMATE] EXTREME WHITE-BOX AUDIT: FINAL PROOF ---")

    lab_dir = ROOT / "data" / "npy_physics_lab"
    excel_path = ROOT / "results" / "ultimate_whitebox_audit_final.xlsx"
    plot_path = ROOT / "results" / "ultimate_physics_plot.png"
    Path(excel_path.parent).mkdir(exist_ok=True)

    # 1. 加载物理引擎
    cfg = {
        "npy_dir": str(lab_dir),
        "initial_cash": 1000000.0,
        "data": { "npy_dir": str(lab_dir) }
    }
    runner = FastRunnerV10(cfg)
    runner.load_data()

    # 抽取物理真值
    dates = runner._meta['dates']
    codes = runner._meta['codes']
    p_open = runner._data['open'][0]
    p_close = runner._data['close'][0]

    # 万能参数
    class AuditParams:
        def __init__(self):
            self.rsrs_window = 18; self.zscore_window = 600
            self.top_n = 5; self.warmup_override = 100
        def to_dict(self): return {"top_n": 5}

    strats = list_vec_strategies()
    print(f"Auditing {len(strats)} strategies...")

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet 0: Physical Truth Table
        df_truth = pd.DataFrame({
            'Date': dates,
            'Physical_Open_T1': p_open,
            'Physical_Close_T': p_close
        })
        df_truth.to_excel(writer, sheet_name='PHYSICAL_TRUTH', index=False)

        plt.figure(figsize=(15, 10))

        for sname in strats:
            print(f" -> Processing {sname}...")
            try:
                # 运行引擎获取 NAV
                res = runner.run(sname, AuditParams(), dates[0], dates[-1])

                # 手动通过 Alpha 函数捕获原始权重（White-Box 特权）
                alpha_fn = get_alpha_fn(sname)
                # 构造策略参数（模拟 runner._build_strategy_kw 的核心部分）
                kw = {
                    "close": runner._data['close'],
                    "open_": runner._data['open'],
                    "high": runner._data['high'],
                    "low": runner._data['low'],
                    "volume": runner._data['volume'],
                    "params": AuditParams(),
                }
                # 检查是否需要 valid_mask
                sig = inspect.signature(alpha_fn)
                if "valid_mask" in sig.parameters:
                    kw["valid_mask"] = runner._data.get('valid_mask')

                alpha_obj = alpha_fn(**kw)
                weights = alpha_obj.raw_target_weights[0] # 取第0只股票

                # 构造审计表
                df_audit = pd.DataFrame({
                    'Date': dates,
                    'Price_Close_T': p_close,
                    'Weight_Signal_T': weights
                })

                # 计算买卖信号（权重变化）
                # Signal: 0 -> >0 为 Buy, >0 -> 0 为 Sell
                signal = []
                for i in range(len(weights)):
                    prev = weights[i-1] if i > 0 else 0
                    curr = weights[i]
                    if prev == 0 and curr > 0: signal.append("BUY_SIGNAL")
                    elif prev > 0 and curr == 0: signal.append("SELL_SIGNAL")
                    else: signal.append("")
                df_audit['Action_T'] = signal

                # T+1 物理成交对齐
                # 信号在第 T 日，成交在第 T+1 日开盘
                # 我们在 DataFrame 中将 T+1 的 Open 挪到 T 行展示，直观对比
                df_audit['Exec_Price_T1'] = df_audit['Date'].shift(-1).map(
                    pd.Series(p_open, index=dates).to_dict()
                )

                # 添加逻辑公式说明 (Text)
                summary_info = [f"Strategy: {sname}", f"Definition: See src/strategies/vectorized/{sname}_alpha.py"]
                summary_df = pd.DataFrame(summary_info, columns=["Logic"])
                summary_df.to_excel(writer, sheet_name=f"Audit_{sname[:20]}", index=False, startrow=0)
                df_audit.to_excel(writer, sheet_name=f"Audit_{sname[:20]}", index=False, startrow=4)

                plt.plot(res.nav_array, label=sname)

            except Exception as e:
                print(f"    Failed {sname}: {e}")

        plt.title("Extreme White-Box Physics Audit (v8-Final)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    print(f"SUCCESS: {excel_path} generated.")
    print(f"SUCCESS: {plot_path} generated.")

if __name__ == "__main__":
    generate_extreme_whitebox_audit()
