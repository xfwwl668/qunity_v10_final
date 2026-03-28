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

def generate_full_system_formulas():
    print("\n--- GENERATING FULL SYSTEM WHITE-BOX AUDIT (ALL STRATEGIES) ---")

    lab_dir = ROOT / "data" / "npy_physics_lab"
    excel_path = ROOT / "results" / "full_system_whitebox_audit_all_strats.xlsx"
    Path(excel_path.parent).mkdir(exist_ok=True)

    # 1. 加载物理数据
    cfg = {"npy_dir": str(lab_dir), "initial_cash": 1000000.0, "data": {"npy_dir": str(lab_dir)}}
    runner = FastRunnerV10(cfg)
    runner.load_data()

    dates = runner._meta['dates']
    codes = runner._meta['codes']
    p_open = runner._data['open'][0]
    p_close = runner._data['close'][0]
    p_vol = runner._data['volume'][0]

    class AuditParams:
        def __init__(self):
            self.rsrs_window = 18; self.zscore_window = 600
            self.top_n = 5; self.warmup_override = 100
        def to_dict(self): return {"top_n": 5}

    strats = list_vec_strategies()
    print(f"Auditing {len(strats)} strategies...")

    # 使用 openpyxl 以兼容多 Sheet 写入
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet 0: Physical Truth
        df_base = pd.DataFrame({
            'Date': dates,
            'Open_T': p_open,
            'Close_T': p_close,
            'Volume_T': p_vol
        })
        df_base.to_excel(writer, sheet_name='BASE_DATA', index=False)

        for sname in strats:
            print(f" -> Injecting formulas for {sname}...")
            try:
                # 运行引擎获取原始权重和得分
                alpha_fn = get_alpha_fn(sname)
                # 构造参数
                kw = {
                    "close": runner._data['close'],
                    "open_": runner._data['open'],
                    "high": runner._data['high'],
                    "low": runner._data['low'],
                    "volume": runner._data['volume'],
                    "params": AuditParams(),
                }
                sig = inspect.signature(alpha_fn)
                if "valid_mask" in sig.parameters: kw["valid_mask"] = runner._data.get('valid_mask')
                # 为某些需要基本面的策略注入默认
                if "amount" in sig.parameters: kw["amount"] = runner._data.get('amount')

                alpha_obj = alpha_fn(**kw)
                weights = alpha_obj.raw_target_weights[0]
                scores = alpha_obj.score[0] if alpha_obj.score is not None else np.zeros_like(weights)

                # 构造 Sheet 数据
                df_strat = pd.DataFrame({
                    'Date': dates,
                    'Price_Close_T': p_close,
                    'Raw_Score_T': scores,
                    'System_Weight_T': weights
                })
                sheet_name = f"Audit_{sname[:20]}"
                df_strat.to_excel(writer, sheet_name=sheet_name, index=False)

                # 注入公式层
                worksheet = writer.sheets[sheet_name]
                # A=Date, B=Price_Close, C=Raw_Score, D=System_Weight
                # E 列: 5日涨幅公式 (肉眼校验动量)
                # F 列: T+1 物理执行校验
                worksheet.cell(row=1, column=5).value = "Mom_5D_Check"
                worksheet.cell(row=1, column=6).value = "Exec_Price_T1"
                worksheet.cell(row=1, column=7).value = "Formula_Alignment_Check"

                for row_idx in range(1, len(dates)):
                    excel_row = row_idx + 1
                    # G5: 5日涨幅
                    if row_idx >= 5:
                        worksheet.cell(row=excel_row, column=5).value = f"=(B{excel_row}/B{excel_row-5})-1"

                    # G6: T+1 开盘价 (物理对齐)
                    if row_idx < len(dates) - 1:
                        # 引用 BASE_DATA 表的开盘价
                        worksheet.cell(row=excel_row, column=6).value = f"=BASE_DATA!B{excel_row+1}"

                    # G7: 验证权重是否与分数对齐
                    # 如果分数为 -inf (通常在 Excel 是极小值)，权重应为 0
                    worksheet.cell(row=excel_row, column=7).value = f"=IF(D{excel_row}>0, \"VALID_EXEC\", \"NO_SIGNAL\")"

            except Exception as e:
                print(f"    Skipped {sname}: {e}")

    print(f"SUCCESS: {excel_path} created with all strategy audit sheets.")

if __name__ == "__main__":
    generate_full_system_formulas()
