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

def run_v14_professional_audit():
    print("\n--- [V14-FINAL] DEEP LOGIC PENTEST: EXECUTING ---")

    lab_dir = ROOT / "data" / "npy_physics_lab"
    excel_path = ROOT / "results" / "full_system_certified_v14_final.xlsx"
    Path(excel_path.parent).mkdir(exist_ok=True)

    # 1. 加载物理数据
    try:
        cfg = {"npy_dir": str(lab_dir), "initial_cash": 1000000.0, "data": {"npy_dir": str(lab_dir)}}
        runner = FastRunnerV10(cfg)
        runner.load_data()
        dates = runner._meta['dates']
        p_close = runner._data['close'][0]
        p_open = runner._data['open'][0]
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load physics lab data: {e}")
        return

    # 2. 前复权平滑还原 (QFQ)
    T = len(p_close)
    rets = p_close[1:] / p_close[:-1]
    rets[599] = 1.0 # 强制抹除 Day 600 跳空 (92->1852) 的视觉干扰
    qfq_price = np.zeros(T)
    qfq_price[-1] = p_close[-1]
    for i in range(T-2, -1, -1):
        qfq_price[i] = qfq_price[i+1] / rets[i]

    # 3. 参数解封 (释放 0.08 限制)
    class AuditParams:
        def __init__(self):
            self.rsrs_window = 18; self.zscore_window = 100
            self.rsrs_threshold = 0.2; self.mom_threshold = 0.01; self.volume_confirm_ratio = 1.1
            self.top_n = 5; self.warmup_override = 50
            self.max_single_pos = 1.0 # 【审计关键】强制释放仓位上限
        def to_dict(self): return {"top_n": 5}

    strats = ["short_term_rsrs", "alpha_hunter_v2", "alpha_max_v5"]

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet 1: Master Truth
        pd.DataFrame({
            'Date': dates,
            'RAW_HFQ_Price': p_close,
            'RAW_HFQ_Open': p_open,
            'QFQ_Visual_Price': qfq_price
        }).to_excel(writer, sheet_name='PRICE_MASTER', index=False)

        for sname in strats:
            print(f" -> Pentesting {sname}...")
            try:
                alpha_fn = get_alpha_fn(sname)
                # 构造注入环境
                kw = {
                   "close": runner._data['close'], "open_": runner._data['open'],
                   "high": runner._data['high'], "low": runner._data['low'],
                   "volume": runner._data['volume'], "params": AuditParams(),
                }
                sig = inspect.signature(alpha_fn)
                if "valid_mask" in sig.parameters: kw["valid_mask"] = runner._data.get('valid_mask')
                if "amount" in sig.parameters: kw["amount"] = runner._data.get('amount')

                alpha_obj = alpha_fn(**kw)
                raw_weights = alpha_obj.raw_target_weights[0]
                raw_scores = alpha_obj.score[0] if alpha_obj.score is not None else np.zeros_like(raw_weights)

                # 构造明细表 (包含前复权和后复权对比)
                df = pd.DataFrame({
                    'Date': dates,
                    'QFQ_Visual_Price': qfq_price, # 趋势平滑
                    'RAW_HFQ_Price': p_close,      # 物理跳空
                    'Alpha_Score_T': raw_scores,
                    'Uncapped_Alpha_Weight_T': raw_weights, # 这里的权重应该 > 0.08
                    'Signal_Code': [0]*T
                })

                # 计算 1/0/-1 信号
                for i in range(1, T):
                    if raw_weights[i-1] == 0 and raw_weights[i] > 0: df.at[i, 'Signal_Code'] = 1
                    elif raw_weights[i-1] > 0 and raw_weights[i] == 0: df.at[i, 'Signal_Code'] = -1

                sheet_name = f"Audit_{sname[:10]}"
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                ws = writer.sheets[sheet_name]

                # 注入公式 (G: T+1执行价, H: 动量校验)
                ws.cell(row=1, column=7).value = "Exec_Physical_T1"
                ws.cell(row=1, column=8).value = "Mom_5D_Check"
                for i in range(1, T):
                    r = i + 1
                    if i < T - 1:
                        # 强制指向下一行开盘物理价
                        ws.cell(row=r, column=7).value = f"=PRICE_MASTER!C{r+1}"
                    if i >= 5:
                        # 针对前复权平滑价算动量
                        ws.cell(row=r, column=8).value = f"=(B{r}/B{r-5})-1"

            except Exception as e:
                print(f"    Failed {sname}: {e}")

    print(f"SUCCESS: {excel_path} generated.")

if __name__ == "__main__":
    run_v14_professional_audit()
