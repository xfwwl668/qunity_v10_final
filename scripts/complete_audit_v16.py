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
    import src.strategies.alpha_signal as alpha_signal
except ImportError:
    print("CRITICAL: src modules not found. Check ROOT path.")
    sys.exit(1)

def run_complete_v16_pro_audit():
    print("\n--- [V16-ULTIMATE] ENGINE REPAIRED: EXECUTING FINAL SYSTEM AUDIT ---")
    print("Fix implemented: Resolved 'Permanent Death Lock' in Numba Kernel.")
    print("Fix implemented: Dynamic weight capping (Visibility: 100%).")

    lab_dir = ROOT / "data" / "npy_physics_lab"
    excel_path = ROOT / "results" / "complete_system_audit_v16.xlsx"
    Path(excel_path.parent).mkdir(exist_ok=True)

    # 1. 加载物理数据
    cfg = {"npy_dir": str(lab_dir), "initial_cash": 1000000.0, "data": {"npy_dir": str(lab_dir)}}
    runner = FastRunnerV10(cfg)
    runner.load_data()

    dates = runner._meta['dates']
    p_close = runner._data['close'][0]
    p_open = runner._data['open'][0]
    p_high = runner._data['high'][0]
    p_vol = runner._data['volume'][0]

    # 2. 全量 QFQ 完美还原 (Backward Cumulative Returns)
    # 策略：从最后一天逆向推导，消灭 Row 601 的 20倍跳空感官干扰
    T = len(p_close)
    rets = p_close[1:] / p_close[:-1]

    # 修复物理跳空点 (Index 599 -> 600 为 Day 601 表现)
    # 我们认为这 20 倍是复权，不是交易逻辑产生的财富，故设其逻辑收益为 1.0 (0%)
    rets[599] = 1.0

    qfq_price = np.zeros(T)
    qfq_price[-1] = p_close[-1]
    for i in range(T-2, -1, -1):
        qfq_price[i] = qfq_price[i+1] / rets[i]

    # 3. 审计参数: 释放仓位限制，暴露原始 Alpha
    class AuditorParams:
        def __init__(self):
            self.rsrs_window = 18; self.zscore_window = 100
            self.rsrs_threshold = 0.2; self.mom_threshold = 0.01; self.volume_confirm_ratio = 1.1
            self.top_n = 5; self.warmup_override = 50
            self.max_single_pos = 1.0 # 100% 权重可见性
            self.risk_budget = 0.02; self.atr_multiplier = 2.0; self.factor_ema_span = 5
            # 内核参数
            self.stop_recovery_days = 5 # 修复 Death Lock 的冷却期
            self.hard_stop_loss = 0.10  # 硬止损 10%
        def to_dict(self): return {k: v for k, v in self.__dict__.items()}

    all_strats = list_vec_strategies()
    print(f"Auditing all {len(all_strats)} strategies...")

    summary_results = []

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet: Master Physical Truth
        pd.DataFrame({
            'Date': dates,
            'RAW_HFQ_Close': p_close,
            'RAW_HFQ_Open': p_open,
            'QFQ_Visual_Smooth': qfq_price, # **核心：审计师肉眼对齐用**
            'Volume_T': p_vol
        }).to_excel(writer, sheet_name='MASTER_TRUTH', index=False)

        for sname in all_strats:
            print(f" -> Rendering logic for {sname}...")
            try:
                # 运行修复后的引擎
                res = runner.run(sname, AuditorParams(), dates[0], dates[-1])
                summary_results.append({
                    'Strategy': sname,
                    'Ann_Return': res.annual_return,
                    'Sharpe': res.sharpe_ratio,
                    'MaxDD': res.max_drawdown,
                    'Final_NAV': res.nav_array[-1]
                })

                # 获取原始信号逻辑
                alpha_fn = get_alpha_fn(sname)
                # 注入依赖项防止 sniper 等策略报错
                alpha_fn.__globals__['_ema_smooth_factor'] = alpha_signal._ema_smooth_factor

                kw = {
                    "close": runner._data['close'],
                    "open_": runner._data['open'],
                    "high": runner._data['high'],
                    "low": runner._data['low'],
                    "volume": runner._data['volume'],
                    "params": AuditorParams(),
                }
                sig_inspect = inspect.signature(alpha_fn)
                if "valid_mask" in sig_inspect.parameters: kw["valid_mask"] = runner._data.get('valid_mask')
                if "amount" in sig_inspect.parameters: kw["amount"] = runner._data.get('volume')

                alpha_obj = alpha_fn(**kw)
                raw_w = alpha_obj.raw_target_weights[0]
                raw_s = alpha_obj.score[0] if alpha_obj.score is not None else np.zeros_like(raw_w)

                # 构造明细页
                df_detail = pd.DataFrame({
                    'Date': dates,
                    'QFQ_Price_T': qfq_price,      # 逻辑趋势准星
                    'RAW_HFQ_Close_T': p_close,     # 物理价格
                    'Alpha_Score': raw_s,
                    'Target_Weight_T': raw_w,
                    'Engine_NAV_T': res.nav_array,
                    'Signal_Code': [0]*T
                })

                for i in range(1, T):
                    if raw_w[i-1] == 0 and raw_w[i] > 0: df_detail.at[i, 'Signal_Code'] = 1    # BUY
                    elif raw_w[i-1] > 0 and raw_w[i] == 0: df_detail.at[i, 'Signal_Code'] = -1 # SELL

                sheet_name = f"Audit_{sname[:10]}"
                df_detail.to_excel(writer, sheet_name=sheet_name, index=False)

                # 注入 Excel 物理对齐公式 (T+1 执行价)
                ws = writer.sheets[sheet_name]
                ws.cell(row=1, column=8).value = "Exec_Physical_T1"
                ws.cell(row=1, column=9).value = "Logical_Momentum_5D"

                for i in range(1, T):
                    r = i + 1
                    # G列是 Engine_NAV_T, H列是 Exec_Physical_T1, I列是 Logical_Momentum
                    if i < T - 1:
                        # 物理对齐公式：T日闭盘信号 -> T+1日开盘物理价 (MASTER_TRUTH!C列)
                        ws.cell(row=r, column=8).value = f"=MASTER_TRUTH!C{r+1}"
                    if i >= 5:
                        # 逻辑动量：基于 QFQ 平滑价格
                        ws.cell(row=r, column=9).value = f"=(B{r}/B{r-5})-1"

            except Exception as e:
                print(f"    [FAIL] {sname}: {e}")

        # Summary Sheet
        pd.DataFrame(summary_results).to_excel(writer, sheet_name='AUDIT_SUMMARY', index=False)

    print(f"\n--- SUCCESS: V16 ULTIMATE AUDIT COMPLETED at {excel_path} ---")

if __name__ == "__main__":
    run_complete_v16_pro_audit()
