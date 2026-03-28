import sys
import numpy as np
import pandas as pd
from pathlib import Path
import json
import inspect

# 路径修复
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine.fast_runner_v10 import FastRunnerV10
from src.strategies.registry import list_vec_strategies, get_alpha_fn

def generate_qfq_style_audit():
    print("\n--- [V11-PRO] QFQ-NORMALIZED VISUAL AUDIT: ELIMINATING JUMPS ---")

    lab_dir = ROOT / "data" / "npy_physics_lab"
    excel_path = ROOT / "results" / "full_system_qfq_visual_audit.xlsx"
    Path(excel_path.parent).mkdir(exist_ok=True)

    # 1. 物理引擎初始化
    cfg = {"npy_dir": str(lab_dir), "initial_cash": 1000000.0, "data": {"npy_dir": str(lab_dir)}}
    runner = FastRunnerV10(cfg)
    runner.load_data()

    dates = runner._meta['dates']
    p_open = runner._data['open'][0]
    p_close = runner._data['close'][0]
    p_high = runner._data['high'][0]
    p_low = runner._data['low'][0]

    # 2. 【核心审计技术】前复权还原 (QFQ Normalization)
    # 计算每日涨跌幅矩阵，以最后一天的价格为基准，反向构造连续价格序列
    # 这样可以完全消除 Day 600 的跳空，方便肉眼直观查看趋势与信号的对应关系
    returns = p_close[1:] / p_close[:-1]
    qfq_close = np.ones_like(p_close)
    qfq_close[-1] = p_close[-1] # 以最后一天价格为准
    for i in range(len(p_close)-2, -1, -1):
        qfq_close[i] = qfq_close[i+1] / returns[i]

    # 同比例缩放 Open/High/Low，保持日内形态不变
    scale_factor = qfq_close / p_close
    qfq_open = p_open * scale_factor
    qfq_high = p_high * scale_factor
    qfq_low = p_low * scale_factor

    class AuditParams:
        def __init__(self, max_pos=1.0):
            self.rsrs_window = 18; self.zscore_window = 100
            self.rsrs_threshold = 0.2; self.mom_threshold = 0.01; self.volume_confirm_ratio = 1.1
            self.top_n = 5; self.warmup_override = 50
            self.max_single_pos = max_pos
        def to_dict(self): return {"top_n": 5}

    strats = list_vec_strategies()

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 物理真值（原始价 vs 前复权价）
        pd.DataFrame({
            'Date': dates,
            'RAW_Close_T': p_close,
            'QFQ_Close_T': qfq_close, # 用于肉眼校验趋势
            'QFQ_Open_T': qfq_open,
            'QFQ_High_T': qfq_high,
            'QFQ_Low_T': qfq_low
        }).to_excel(writer, sheet_name='PRICE_MASTER', index=False)

        for sname in strats[:10]:
            print(f" -> QFQ Auditing {sname}...")
            try:
                # 引擎运行获取真实成交价
                res = runner.run(sname, AuditParams(max_pos=0.08), dates[0], dates[-1])
                engine_weights = res.target_weights[0] if hasattr(res, 'target_weights') else np.zeros_like(p_close)

                # 获取原始信号描述
                alpha_fn = get_alpha_fn(sname)
                # 构造策略环境
                kw = {
                    "close": runner._data['close'], "open_": runner._data['open'],
                    "high": runner._data['high'], "low": runner._data['low'],
                    "volume": runner._data['volume'], "params": AuditParams(max_pos=1.0),
                }
                sig = inspect.signature(alpha_fn)
                if "valid_mask" in sig.parameters: kw["valid_mask"] = runner._data.get('valid_mask')

                alpha_obj = alpha_fn(**kw)
                raw_weights = alpha_obj.raw_target_weights[0]

                # 构造审计表
                df = pd.DataFrame({
                    'Date': dates,
                    'QFQ_Close_T': qfq_close, # 连续肉眼可见价
                    'Alpha_Raw_Weight_T': raw_weights,
                    'Signal_Code': [0]*len(dates),
                    'Engine_Final_Weight_T': engine_weights
                })

                # 标注 1 / 0 / -1 信号逻辑
                for i in range(1, len(dates)):
                    prev_w = raw_weights[i-1]
                    curr_w = raw_weights[i]
                    if prev_w == 0 and curr_w > 0: df.at[i, 'Signal_Code'] = 1
                    elif prev_w > 0 and curr_w == 0: df.at[i, 'Signal_Code'] = -1

                sheet_name = f"Audit_{sname[:15]}"
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                ws = writer.sheets[sheet_name]

                # 注入公式层 (F列: 5日趋势, G列: 物理对齐, H列: RSRS 模拟分)
                ws.cell(row=1, column=6).value = "Mom_5D_Normalized"
                ws.cell(row=1, column=7).value = "Trade_Price_T1_Open"

                for i in range(1, len(dates)):
                    dr = i + 1
                    # F: =IF(B{r}/B{r-5}>1.01, "UP", "FLAT") (针对前复权连续价)
                    if i >= 5:
                        ws.cell(row=dr, column=6).value = f"=(B{dr}/B{dr-5})-1"
                    # G: 映射回物理开盘价 (虽然行情按 QFQ 看，但成交必须是物理原价才对齐真实账户)
                    if i < len(dates) - 1:
                        ws.cell(row=dr, column=7).value = f"=PRICE_MASTER!B{dr+1}"

            except Exception as e:
                print(f"    Failed {sname}: {e}")

    print(f"\n--- SUCCESS: V11-QFQ VISUAL AUDIT REPORT GENERATED ---")

if __name__ == "__main__":
    generate_qfq_style_audit()
