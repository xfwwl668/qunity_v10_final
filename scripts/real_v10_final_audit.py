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

def generate_v10_final_truth_audit():
    print("\n--- [V10-ULTIMATE] PROFESSIONAL AUDIT: SIGNAL DECONSTRUCTION ---")

    lab_dir = ROOT / "data" / "npy_physics_lab"
    excel_path = ROOT / "results" / "full_system_whitebox_audit_v10_final.xlsx"
    Path(excel_path.parent).mkdir(exist_ok=True)

    # 1. 物理引擎初始化
    # 我们故意设置不同的风控参数，用来在 Excel 中对比“原始信号”与“风控执行”
    cfg = {
        "npy_dir": str(lab_dir),
        "initial_cash": 1000000.0,
        "data": {"npy_dir": str(lab_dir)},
        "backtest": {
            "max_single_pos": 0.08, # 引擎风控：强制 8%
        }
    }
    runner = FastRunnerV10(cfg)
    runner.load_data()

    dates = runner._meta['dates']
    p_open = runner._data['open'][0]
    p_close = runner._data['close'][0]

    # 2. 构造“数据连续性”辅助列（除权还原）
    # 在 Day 600 有一个 20倍跳空。我们通过累积收益率构造一个“视觉连续价格”
    daily_ret = np.diff(p_close) / p_close[:-1]
    # 填充第0天，并计算累积净值用于视觉核对
    cont_price = np.ones(len(p_close))
    cont_price[1:] = np.cumprod(1 + daily_ret)

    # 3. 敏感审计参数
    class UltimateParams:
        def __init__(self, max_pos=1.0):
            self.rsrs_window = 18; self.zscore_window = 100
            self.rsrs_threshold = 0.2; self.mom_threshold = 0.01; self.volume_confirm_ratio = 1.1
            self.top_n = 5; self.warmup_override = 50
            self.max_single_pos = max_pos # 策略内置风控开关
        def to_dict(self): return {"top_n": 5}

    strats = list_vec_strategies()

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet: 物理真相（含视觉连续价）
        pd.DataFrame({
            'Date': dates,
            'Physical_Open_T1': p_open,
            'Physical_Close_T': p_close,
            'Visual_Continuous_Price': cont_price # 还原跳空后的连续曲线，供肉眼核对逻辑
        }).to_excel(writer, sheet_name='PHYSICAL_TRUTH', index=False)

        for sname in strats[:10]:
            print(f" -> Professional Auditing {sname}...")
            try:
                # A. 运行引擎（受 0.08 强力风控）
                res = runner.run(sname, UltimateParams(max_pos=0.08), dates[0], dates[-1])
                engine_weights = res.target_weights[0] if hasattr(res, 'target_weights') else np.zeros_like(p_close)

                # B. 运行 Alpha（完全释放 100% 仓位限制，暴露原始逻辑）
                alpha_fn = get_alpha_fn(sname)
                kw = {
                    "close": runner._data['close'], "open_": runner._data['open'],
                    "high": runner._data['high'], "low": runner._data['low'],
                    "volume": runner._data['volume'], "params": UltimateParams(max_pos=1.0),
                }
                sig = inspect.signature(alpha_fn)
                if "valid_mask" in sig.parameters: kw["valid_mask"] = runner._data.get('valid_mask')
                # 兼容 V10 财务注入
                for fund_key in ["amount", "pe_matrix", "roe_matrix", "mktcap_matrix"]:
                    if fund_key in sig.parameters: kw[fund_key] = runner._data.get(fund_key)

                alpha_obj = alpha_fn(**kw)
                raw_weights = alpha_obj.raw_target_weights[0]
                raw_scores = alpha_obj.score[0] if alpha_obj.score is not None else np.zeros_like(raw_weights)

                # 4. 构造审计 DataFrame
                df = pd.DataFrame({
                    'Date': dates,
                    'Close_T': p_close,
                    'Cont_Price_T': cont_price, # 连续价，方便算动量
                    'Raw_Score_T': raw_scores,
                    'Alpha_Raw_Weight': raw_weights, # 这里的权重应该是 0.25+
                    'Engine_Final_Weight': engine_weights # 这里应该是 0.08 封顶
                })

                # 5. 定义信号（符合用户要求的 1, 0, -1）
                # 1 = 买入信号 (ENTER)
                # 0 = 持仓或空仓 (HOLD/EMPTY)
                # -1 = 卖出信号 (EXIT)
                signals = [0] * len(dates)
                for i in range(1, len(dates)):
                    prev_w = raw_weights[i-1]
                    curr_w = raw_weights[i]
                    if prev_w == 0 and curr_w > 0: signals[i] = 1 # ENTER
                    elif prev_w > 0 and curr_w == 0: signals[i] = -1 # EXIT

                df['Signal_Code'] = signals

                sheet_name = f"Audit_{sname[:20]}"
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                ws = writer.sheets[sheet_name]

                # 6. 注入肉眼校验公式
                # G: 连续价动量 (Close_T / Close_T-5 - 1)
                # H: 执行价 (T+1 Open)
                # I: 逻辑校验
                ws.cell(row=1, column=8).value = "Mom_5D"
                ws.cell(row=1, column=9).value = "Exec_Price_T1"

                for i in range(1, len(dates)):
                    dr = i + 1
                    # G: 针对连续价格算动量，不受 Day 600 跳空干扰
                    if i >= 5:
                        ws.cell(row=dr, column=8).value = f"=(C{dr}/C{dr-5})-1"
                    if i < len(dates) - 1:
                        ws.cell(row=dr, column=9).value = f"=PHYSICAL_TRUTH!B{dr+1}"

            except Exception as e:
                print(f"    Failed {sname}: {e}")

    print(f"\n--- SUCCESS: V10-FINAL PROFESSIONAL AUDIT READY ---")

if __name__ == "__main__":
    generate_v10_final_truth_audit()
