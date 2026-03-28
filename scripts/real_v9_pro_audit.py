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

def generate_v9_pro_audit():
    print("\n--- [V9-PRO] DEEP WHITE-BOX LOGIC AUDIT: INITIATING ---")

    lab_dir = ROOT / "data" / "npy_physics_lab"
    excel_path = ROOT / "results" / "full_system_whitebox_audit_v9_pro.xlsx"
    Path(excel_path.parent).mkdir(exist_ok=True)

    # 1. 初始化引擎与敏感参数 (为了强行激发出信号)
    cfg = {
        "npy_dir": str(lab_dir),
        "initial_cash": 1000000.0,
        "data": {"npy_dir": str(lab_dir)},
        # 调高风控上限，防止 0.08 瓶颈压制观察
        "backtest": {
            "max_single_pos": 0.5,
            "commission_rate": 0.0,
            "stamp_tax": 0.0,
            "slippage_rate": 0.0
        }
    }
    runner = FastRunnerV10(cfg)
    runner.load_data()

    dates = runner._meta['dates']
    p_open = runner._data['open'][0]
    p_close = runner._data['close'][0]
    p_vol = runner._data['volume'][0]

    # V9-PRO 降门槛参数：确保在合成的正弦波上有反应
    class ProAuditParams:
        def __init__(self):
            self.rsrs_window = 18
            self.zscore_window = 100 # 缩短窗口，增加敏感度
            self.rsrs_threshold = 0.2 # 从 0.7 降到 0.2
            self.mom_threshold = 0.01  # 从 0.03 降到 0.01
            self.volume_confirm_ratio = 1.1 # 从 1.5 降到 1.1
            self.top_n = 5
            self.warmup_override = 50
        def to_dict(self): return {"top_n": 5}

    strats = list_vec_strategies()
    print(f"Extracted {len(strats)} strategies for Pro Audit.")

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet: 真值表
        pd.DataFrame({'Date': dates, 'Open': p_open, 'Close': p_close, 'Vol': p_vol}).to_excel(writer, sheet_name='BASE_DATA', index=False)

        for sname in strats[:8]: # 重点审计前8个，确保 Excel 不崩溃且深度足够
            print(f" -> Auditing {sname} (Deep Logic Decomposition)...")
            try:
                # 1. 运行引擎（带风控）
                res = runner.run(sname, ProAuditParams(), dates[0], dates[-1])
                capped_weights = res.target_weights[0] if hasattr(res, 'target_weights') else np.zeros_like(p_close)
                if capped_weights.sum() == 0:
                    # Fallback lookup in result object structure
                    for attr in ['weights', 'final_weights', 'pos_matrix']:
                        if hasattr(res, attr):
                            val = getattr(res, attr)
                            if isinstance(val, np.ndarray): capped_weights = val[0]

                # 2. 运行 Alpha 函数（无风控，捕获原始信号）
                alpha_fn = get_alpha_fn(sname)
                kw = {
                    "close": runner._data['close'], "open_": runner._data['open'],
                    "high": runner._data['high'], "low": runner._data['low'],
                    "volume": runner._data['volume'], "params": ProAuditParams(),
                }
                sig = inspect.signature(alpha_fn)
                if "valid_mask" in sig.parameters: kw["valid_mask"] = runner._data.get('valid_mask')
                if "amount" in sig.parameters: kw["amount"] = runner._data.get('amount')

                alpha_obj = alpha_fn(**kw)
                original_weights = alpha_obj.raw_target_weights[0]
                raw_scores = alpha_obj.score[0] if alpha_obj.score is not None else np.zeros_like(original_weights)

                # 3. 构造数据
                df = pd.DataFrame({
                    'Date': dates,
                    'Price_Close_T': p_close,
                    'Raw_Score_T': raw_scores,
                    'Original_Alpha_Weight_T': original_weights,
                    'Risk_Capped_Weight_T': capped_weights
                })

                sheet_name = f"Audit_{sname[:20]}"
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                ws = writer.sheets[sheet_name]

                # 4. 逻辑拆解公式 (注入 Excel)
                # F: 条件A (择时通过?) | G: 条件B (动量通过?) | H: 动作指令
                ws.cell(row=1, column=6).value = "Gate_RSRS_Passed"
                ws.cell(row=1, column=7).value = "Gate_Momentum_Passed"
                ws.cell(row=1, column=8).value = "Phase_Action"
                ws.cell(row=1, column=9).value = "Exec_Physical_T1"

                for i in range(1, len(dates)):
                    r = i + 1
                    # F: =IF(C{r}>-100, "PASS", "FAIL")  (Raw_Score 不是 -inf 即为通过门控)
                    ws.cell(row=r, column=6).value = f'=IF(C{r}>-50, "PASS", "FAIL")'
                    # G: =IF(B{r}/B{max(1,r-5)}>1.01, "MOM_UP", "MOM_FLAT")
                    if i >= 5:
                        ws.cell(row=r, column=7).value = f'=IF(B{r}/B{r-5}>1.01, "MOM_UP", "MOM_FLAT")'
                    # H: =IF(AND(F{r}="PASS", G{r}="MOM_UP"), "ENTER", IF(D{r}>0, "HOLD", "EMPTY"))
                    ws.cell(row=r, column=8).value = f'=IF(AND(F{r}="PASS", G{r}="MOM_UP"), "ENTER", IF(D{r}>0, "HOLD", "EMPTY"))'
                    # I: =BASE_DATA!B{r+1}
                    if i < len(dates) - 1:
                        ws.cell(row=r, column=9).value = f'=BASE_DATA!B{r+1}'

            except Exception as e:
                print(f"    Failed {sname}: {e}")

    print(f"\n--- SUCCESS: V9-PRO REPORT GENERATED AT {excel_path} ---")

if __name__ == "__main__":
    generate_v9_pro_audit()
