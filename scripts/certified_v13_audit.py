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

def generate_v13_certified_audit():
    print("\n--- [V13-PRO] CERTIFIED CONTINUOUS PRICE AUDIT ---")

    lab_dir = ROOT / "data" / "npy_physics_lab"
    excel_path = ROOT / "results" / "full_system_certified_v13_audit.xlsx"
    Path(excel_path.parent).mkdir(exist_ok=True)

    # 1. 物理数据
    cfg = {"npy_dir": str(lab_dir), "initial_cash": 1000000.0, "data": {"npy_dir": str(lab_dir)}}
    runner = FastRunnerV10(cfg)
    runner.load_data()

    dates = runner._meta['dates']
    p_close_raw = runner._data['close'][0]
    p_open_raw = runner._data['open'][0]

    # 2. 【核心审计技术】模拟 A 股前复权 (QFQ) —— 滤除人工跳空
    # 我们知道 Day 600 (Index 600) 有一个 20倍的人工跳空。
    # 为了肉眼核对趋势，我们必须在那个点将“逻辑收益率”与“价格跳空”剥离。

    T = len(p_close_raw)
    clean_returns = p_close_raw[1:] / p_close_raw[:-1]

    # 滤除 Day 600 的跳空干扰 (Index 599 -> 600)
    # 我们假设那一天的“真实经济回报”是 0%，跳空全是复权因子
    jump_idx = 599
    actual_jump_factor = clean_returns[jump_idx] # 应该是 ~20.0
    clean_returns[jump_idx] = 1.0 # 强制设为 0% 收益，用于视觉还原

    # 构造完全连续的价格序列
    qfq_continuous = np.zeros(T)
    qfq_continuous[-1] = p_close_raw[-1] # 以最后价格为基准
    for i in range(T-2, -1, -1):
        qfq_continuous[i] = qfq_continuous[i+1] / clean_returns[i]

    # 3. 准备审计报告
    strats = ["short_term_rsrs", "alpha_hunter_v2"]

    class CertParams:
        def __init__(self):
            self.rsrs_window = 18; self.zscore_window = 100
            self.rsrs_threshold = 0.2; self.mom_threshold = 0.01; self.volume_confirm_ratio = 1.1
            self.top_n = 5; self.warmup_override = 50; self.max_single_pos = 1.0
        def to_dict(self): return {"top_n": 5}

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet: 物理与视觉对照
        pd.DataFrame({
            'Date': dates,
            'RAW_HFQ_Price': p_close_raw, # 含跳空，引擎用
            'QFQ_Continuous_Price': qfq_continuous # **真正无缝连接**，肉眼审计用
        }).to_excel(writer, sheet_name='PRICE_AUDIT', index=False)

        for sname in strats:
            print(f" -> Certified Auditing {sname}...")
            alpha_fn = get_alpha_fn(sname)
            kw = {
               "close": runner._data['close'], "open_": runner._data['open'],
               "high": runner._data['high'], "low": runner._data['low'],
               "volume": runner._data['volume'], "params": CertParams(),
            }
            alpha_obj = alpha_fn(**kw)
            weights = alpha_obj.raw_target_weights[0]

            df = pd.DataFrame({
                'Date': dates,
                'QFQ_Price_T': qfq_continuous, # 这一列是连续的
                'Signal_Code': [0]*T,
                'Alpha_Weight_T': weights
            })

            # 计算 1/0/-1 信号
            for i in range(1, T):
                if weights[i-1] == 0 and weights[i] > 0: df.at[i, 'Signal_Code'] = 1
                elif weights[i-1] > 0 and weights[i] == 0: df.at[i, 'Signal_Code'] = -1

            sheet_name = f"Audit_{sname[:10]}"
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]

            # 注入肉眼校验公式
            # E: 5日趋势 (基于连续价)
            # F: 物理执行 (基于原价)
            ws.cell(row=1, column=5).value = "Mom_5D_Verify"
            ws.cell(row=1, column=6).value = "Exec_Physical_T1"
            for i in range(1, T):
                r = i + 1
                if i >= 5: ws.cell(row=r, column=5).value = f"=(B{r}/B{r-5})-1"
                if i < T - 1: ws.cell(row=r, column=6).value = f"=PRICE_AUDIT!A{r+1}"

    print(f"\n--- SUCCESS: V13-CERTIFIED REPORT READY ---")

if __name__ == "__main__":
    generate_v13_certified_audit()
