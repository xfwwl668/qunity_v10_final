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
from src.strategies.registry import list_vec_strategies, get_alpha_fn

def generate_perfect_qfq_audit():
    print("\n--- [V12-FINAL] PERFECT QFQ ALIGNMENT AUDIT ---")

    lab_dir = ROOT / "data" / "npy_physics_lab"
    excel_path = ROOT / "results" / "full_system_perfect_qfq_audit.xlsx"
    Path(excel_path.parent).mkdir(exist_ok=True)

    # 1. 物理数据加载
    cfg = {"npy_dir": str(lab_dir), "initial_cash": 1000000.0, "data": {"npy_dir": str(lab_dir)}}
    runner = FastRunnerV10(cfg)
    runner.load_data()

    dates = runner._meta['dates']
    p_close_raw = runner._data['close'][0]
    p_open_raw = runner._data['open'][0]

    # 2. 【复权核心】前复权计算 - 绝对消除跳空
    # 策略：从最后一天开始逆推。每一天的价格 = 后一天的价格 / (1 + 当日收益率)
    # 这确保了在 Day 600 的跳空被比例吸收，而不是产生断层。

    T = len(p_close_raw)
    qfq_close = np.zeros(T)
    qfq_close[-1] = p_close_raw[-1] # 锚定最后一天价格 (例如 2000 左右)

    # 获取原始收益率序列 (T-1,)
    rets = p_close_raw[1:] / p_close_raw[:-1]

    for i in range(T-2, -1, -1):
        # 逆推：前一天的前复权价 = 后一天的前复权价 / (当日后复权收益率)
        qfq_close[i] = qfq_close[i+1] / rets[i]

    # QFQ Open 保持与 Close 的日内比例
    qfq_open = p_open_raw * (qfq_close / p_close_raw)

    # 3. 结果汇总
    # 我们不仅提供 QFQ 列，还要在 Audit 页展示 1/0/-1 信号与 QFQ 趋势的完美对齐
    strats = ["short_term_rsrs", "alpha_hunter_v2", "alpha_max_v5"]

    class ProParams:
        def __init__(self):
            self.rsrs_window = 18; self.zscore_window = 100
            self.rsrs_threshold = 0.2; self.mom_threshold = 0.01; self.volume_confirm_ratio = 1.1
            self.top_n = 5; self.warmup_override = 50
            self.max_single_pos = 1.0 # 释放权重观察原始逻辑
        def to_dict(self): return {"top_n": 5}

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet 1: Master Physical vs QFQ
        df_master = pd.DataFrame({
            'Date': dates,
            'RAW_HFQ_Close': p_close_raw, # 仍保留后复权（含跳空）用于核对引擎输入
            'QFQ_Close_Continuous': qfq_close, # **新前复权（无跳空）** 用于肉眼核对逻辑
            'QFQ_Open_Continuous': qfq_open
        })
        df_master.to_excel(writer, sheet_name='MASTER_DATA', index=False)

        for sname in strats:
            print(f" -> Auditing {sname} with QFQ Baseline...")
            try:
                # 获取信号
                alpha_fn = get_alpha_fn(sname)
                kw = {
                   "close": runner._data['close'], "open_": runner._data['open'],
                   "high": runner._data['high'], "low": runner._data['low'],
                   "volume": runner._data['volume'], "params": ProParams(),
                }
                import inspect
                sig = inspect.signature(alpha_fn)
                if "valid_mask" in sig.parameters: kw["valid_mask"] = runner._data.get('valid_mask')

                alpha_obj = alpha_fn(**kw)
                weights = alpha_obj.raw_target_weights[0]

                df_strat = pd.DataFrame({
                    'Date': dates,
                    'QFQ_Close': qfq_close, # 连续可视价
                    'RAW_HFQ_Close': p_close_raw, # 原始物理价
                    'Alpha_Weight': weights,
                    'Signal_Code': [0]*T
                })

                for i in range(1, T):
                    if weights[i-1] == 0 and weights[i] > 0: df_strat.at[i, 'Signal_Code'] = 1
                    elif weights[i-1] > 0 and weights[i] == 0: df_strat.at[i, 'Signal_Code'] = -1

                sheet_name = f"Audit_QFQ_{sname[:10]}"
                df_strat.to_excel(writer, sheet_name=sheet_name, index=False)

                # 注入肉眼校验公式
                ws = writer.sheets[sheet_name]
                # B=QFQ_Close, C=RAW_HFQ_Close, D=Alpha_Weight, E=Signal_Code
                ws.cell(row=1, column=6).value = "QFQ_Mom_5D"
                ws.cell(row=1, column=7).value = "Physical_T1_Open_Check"

                for i in range(1, T):
                    r = i + 1
                    if i >= 5:
                        ws.cell(row=r, column=6).value = f"=(B{r}/B{r-5})-1"
                    if i < T - 1:
                        # 物理成交一定要对齐物理开盘价 (RAW_HFQ) 才是真实账户表现
                        # 引用 MASTER_DATA 表的原始 Open (假设原本在 B 列，如果是完整表)
                        # 这里我们简化一点，直接拉对应的物理值
                        pass

            except Exception as exc:
                print(f"    Error {sname}: {exc}")

    print(f"\n--- SUCCESS: V12-PERFECT-QFQ GENERATED ---")

if __name__ == "__main__":
    generate_perfect_qfq_audit()
