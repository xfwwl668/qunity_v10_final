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

try:
    from src.engine.fast_runner_v10 import FastRunnerV10
    from src.strategies.registry import list_vec_strategies, get_alpha_fn
    from src.strategies.alpha_signal import _ema_smooth_factor # 为了修复某些脚本中的未定义引用
except ImportError:
    print("CRITICAL: src modules not found. Check ROOT path.")
    sys.exit(1)

def run_complete_v15_audit():
    print("\n--- [V15-FINAL] ULTIMATE COMPLETE SYSTEM AUDIT: EXECUTING ---")

    lab_dir = ROOT / "data" / "npy_physics_lab"
    excel_path = ROOT / "results" / "complete_system_audit_v15.xlsx"
    Path(excel_path.parent).mkdir(exist_ok=True)

    # 1. 加载物理数据
    cfg = {"npy_dir": str(lab_dir), "initial_cash": 1000000.0, "data": {"npy_dir": str(lab_dir)}}
    runner = FastRunnerV10(cfg)
    runner.load_data()

    dates = runner._meta['dates']
    p_close = runner._data['close'][0]
    p_open = runner._data['open'][0]
    p_high = runner._data['high'][0]
    p_low = runner._data['low'][0]
    p_vol = runner._data['volume'][0]

    # 2. 全量 QFQ 平滑还原
    T = len(p_close)
    rets = p_close[1:] / p_close[:-1]
    rets[599] = 1.0 # 抹除跳空
    qfq_price = np.zeros(T)
    qfq_price[-1] = p_close[-1]
    for i in range(T-2, -1, -1):
        qfq_price[i] = qfq_price[i+1] / rets[i]

    # 3. 审计参数: 开启暴力模式 (释放所有仓位限制)
    class ProAuditParams:
        def __init__(self):
            self.rsrs_window = 18; self.zscore_window = 100
            self.rsrs_threshold = 0.2; self.mom_threshold = 0.01; self.volume_confirm_ratio = 1.1
            self.top_n = 5; self.warmup_override = 50
            self.max_single_pos = 1.0 # 强制 100% 可见
            self.risk_budget = 0.02; self.atr_multiplier = 2.0; self.factor_ema_span = 5
        def to_dict(self): return {k: v for k, v in self.__dict__.items()}

    all_strats = list_vec_strategies()
    print(f"Detected {len(all_strats)} strategies for full audit.")

    summary_results = []

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet: 物理与视觉主基准 (1200行全数据)
        master_df = pd.DataFrame({
            'Date': dates,
            'RAW_HFQ_Close': p_close,
            'RAW_HFQ_Open': p_open,
            'QFQ_Visual_Price': qfq_price,
            'Volume_T': p_vol
        })
        master_df.to_excel(writer, sheet_name='MASTER_TRUTH', index=False)

        for sname in all_strats:
            print(f" -> Auditing {sname}...")
            try:
                # 运行引擎获取性能汇总
                res = runner.run(sname, ProAuditParams(), dates[0], dates[-1])
                summary_results.append({
                    'Strategy': sname,
                    'Ann_Return': res.annual_return,
                    'Sharpe': res.sharpe_ratio,
                    'MaxDD': res.max_drawdown
                })

                # 手动获取详细列
                alpha_fn = get_alpha_fn(sname)
                # 构造全维度注入 (修复之前的 missing argument)
                kw = {
                   "close": runner._data['close'], "open_": runner._data['open'],
                   "high": runner._data['high'], "low": runner._data['low'],
                   "volume": runner._data['volume'], "params": ProAuditParams(),
                }
                sig = inspect.signature(alpha_fn)
                # 自动注入所有缺失的数据
                if "valid_mask" in sig.parameters: kw["valid_mask"] = runner._data.get('valid_mask')
                if "amount" in sig.parameters: kw["amount"] = runner._data.get('volume') # 模拟成交额
                if "amount_matrix" in sig.parameters: kw["amount_matrix"] = runner._data.get('volume')

                # 特殊：sniper_v13_hfq 可能需要全局 ema 工具
                import src.strategies.alpha_signal as alpha_signal
                # 将全局函数注入到策略函数的作用域中，防止 NameError
                alpha_fn.__globals__['_ema_smooth_factor'] = alpha_signal._ema_smooth_factor

                alpha_obj = alpha_fn(**kw)
                raw_w = alpha_obj.raw_target_weights[0]
                raw_s = alpha_obj.score[0] if alpha_obj.score is not None else np.zeros_like(raw_w)

                df_detail = pd.DataFrame({
                    'Date': dates,
                    'QFQ_Visual_Price': qfq_price,
                    'RAW_HFQ_Close': p_close,
                    'Alpha_Score_T': raw_s,
                    'Alpha_Uncapped_Weight_T': raw_w,
                    'Signal_Code': [0]*T
                })

                for i in range(1, T):
                    if raw_w[i-1] == 0 and raw_w[i] > 0: df_detail.at[i, 'Signal_Code'] = 1
                    elif raw_w[i-1] > 0 and raw_w[i] == 0: df_detail.at[i, 'Signal_Code'] = -1

                sheet_short_name = f"Audit_{sname[:10]}"
                df_detail.to_excel(writer, sheet_name=sheet_short_name, index=False)

                # 注入审计公式
                ws = writer.sheets[sheet_short_name]
                ws.cell(row=1, column=7).value = "Exec_Physical_T1"
                ws.cell(row=1, column=8).value = "Trend_Align"
                for i in range(1, T):
                    r = i + 1
                    if i < T - 1: ws.cell(row=r, column=7).value = f"=MASTER_TRUTH!C{r+1}"
                    if i >= 1: ws.cell(row=r, column=8).value = f"=IF(B{r}>B{r-1}, \"UP\", \"DOWN\")"

            except Exception as e:
                print(f"    [FAILED] {sname}: {e}")

        # Summary Sheet
        pd.DataFrame(summary_results).to_excel(writer, sheet_name='AUDIT_SUMMARY', index=False)

    print(f"\n--- SUCCESS: COMPLETE V15 AUDIT PACKAGE GENERATED AT {excel_path} ---")

if __name__ == "__main__":
    run_complete_v15_audit()
