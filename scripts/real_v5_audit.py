import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# 路径修复
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine.fast_runner_v10 import FastRunnerV10
from src.strategies.registry import list_vec_strategies

def run_real_v5_audit():
    print("\n--- INITIATING SYSTEM PROBE & FINAL AUDIT (V5-REAL) ---")

    cfg = {
        "npy_dir": "data/npy_physics_lab",
        "initial_cash": 1000000.0,
        "commission_rate": 0.0,
        "stamp_tax": 0.0,
        "slippage_rate": 0.0,
        "vol_multiplier": 100,
        "data": { "npy_dir": "data/npy_physics_lab" }
    }

    runner = FastRunnerV10(cfg)
    runner.load_data()

    # --- 探测真实物理路径 ---
    # 根据 V10 DataBundle 逻辑，实际数据存放在实例字典或 bundle 属性中
    # 我们遍历找 close
    potential_data = {}
    if hasattr(runner, '_data_bundle'):
        # 如果是 V10 标准结构
        bundle = runner._data_bundle
        potential_data['open'] = getattr(bundle, 'open', None)
        potential_data['close'] = getattr(bundle, 'close', None)
        potential_data['dates'] = getattr(bundle, 'dates', [])
        potential_data['codes'] = getattr(bundle, 'codes', [])

    if not potential_data.get('close'):
        # 回退策略：检查 instance dict
        for k, v in runner.__dict__.items():
            if k == 'close' or k == '_close':
                potential_data['close'] = v
            if k == 'open' or k == '_open':
                potential_data['open'] = v
            if k == 'dates' or k == '_dates':
                potential_data['dates'] = v

    # 最后的绝对兜底：直接从物理 meta.json 拿日期和代码
    import json
    with open(Path(cfg['npy_dir']) / "meta.json", "r") as f:
        meta = json.load(f)
        dates = meta['dates']
        codes = meta['codes']

    # 物理矩阵如果还拿不到，直接读取生成的 npy
    p_open = potential_data.get('open')
    if p_open is None:
        p_open = np.load(Path(cfg['npy_dir']) / "open.npy")[0]
    else:
        p_open = p_open[0]

    p_close = potential_data.get('close')
    if p_close is None:
        p_close = np.load(Path(cfg['npy_dir']) / "close.npy")[0]
    else:
        p_close = p_close[0]

    strats = list_vec_strategies()
    print(f"Probe complete. Detected {len(strats)} strategies.")

    excel_path = ROOT / "results" / "full_system_whitebox_audit_v5_final.xlsx"
    plot_path = ROOT / "results" / "physics_audit_plot_v5_final.png"
    Path(excel_path.parent).mkdir(exist_ok=True)

    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        # Physical Base
        pd.DataFrame({
            'Date': dates,
            'Price_Open': p_open,
            'Price_Close': p_close
        }).to_excel(writer, sheet_name='PHYSICAL_TRUTH', index=False)

        fig, axes = plt.subplots(len(strats), 1, figsize=(10, 3*len(strats)))
        if len(strats) == 1: axes = [axes]

        for idx, sname in enumerate(strats):
            print(f"  -> Processing: {sname}")
            try:
                # 强制传递参数对象以绕过可能的 API 变动
                class DummyParams:
                    rsrs_window = 18; zscore_window = 300; top_n = 5; annual_factor = 252.0
                    def to_dict(self): return {"top_n": 5}

                res = runner.run(sname, DummyParams(), dates[0], dates[-1])

                df_trace = pd.DataFrame({'Date': dates, 'Weight_T': res.target_weights[0]})

                # 物理对齐检查
                if not res.trades_df.empty:
                    t_df = res.trades_df[res.trades_df['code'] == codes[0]]
                    trade_map = t_df.set_index('date')['price'].to_dict()
                    df_trace['Exec_Price_T1'] = df_trace['Date'].map(trade_map)

                df_trace.to_excel(writer, sheet_name=f'Audit_{sname[:20]}', index=False)
                axes[idx].plot(res.nav_array); axes[idx].set_title(sname)
            except Exception as e:
                print(f"     Failed {sname}: {e}")

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"\n--- SUCCESS ---")
    print(f"Excel: {excel_path}")
    print(f"Plot : {plot_path}")

if __name__ == "__main__":
    run_real_v5_audit()
