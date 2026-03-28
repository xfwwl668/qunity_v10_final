import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import datetime

def generate_physics_lab_data(output_dir="data/npy_physics_lab"):
    """
    Q-UNITY V10 物理实验室数据生成器 - (移除特殊字符兼容版)
    目标：1200天, 10只股票, 3个完整周期, HFQ物理平移
    """
    N, T = 10, 1200
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"--- START GENERATING LAB DATA: N={N}, T={T} ---")

    # 1. 交易日期
    all_dates = []
    curr = datetime.date(2019, 1, 1)
    while len(all_dates) < T:
        if curr.weekday() < 5:
            all_dates.append(curr.isoformat())
        curr += datetime.timedelta(days=1)

    # 2. 市场指数
    t_idx = np.arange(T)
    index_wave = 3000 * (1 + 0.3 * np.sin(2 * np.pi * t_idx / 400))
    market_index = index_wave.reshape(1, T).astype(np.float32)

    # 3. 价格生成 (带 HFQ 平移)
    close = np.zeros((N, T), dtype=np.float32)
    hfq_shift = np.ones(T, dtype=np.float32)
    hfq_shift[600:] = 20.0

    np.random.seed(42)
    for i in range(N):
        noise = np.random.normal(0, 0.01, T).cumsum()
        stock_wave = 100 * (1 + 0.2 * np.sin(2 * np.pi * t_idx / 400 + i * 0.1)) * (1 + noise)
        close[i] = (stock_wave * hfq_shift).astype(np.float32)

    open_ = close * (1 + np.random.uniform(-0.005, 0.005, (N, T)))
    high = np.maximum(close, open_) * 1.01
    low  = np.minimum(close, open_) * 0.99

    # 4. 成交量与成交额
    volume = np.random.uniform(1000, 5000, (N, T)).astype(np.float32)
    amount = (close * volume * 100).astype(np.float32)

    # 5. 有效掩码 (预热期验证)
    valid_mask = np.ones((N, T), dtype=bool)
    valid_mask[:, :300] = False

    # 6. 基本面数据
    roe = np.full((N, T), 0.05, dtype=np.float32)
    roe[:3, :] = 0.18
    pe = (close / 10.0).astype(np.float32)
    mcap = (close * 100000000).astype(np.float32)
    concept_ids = np.zeros((N, T), dtype=np.int32)
    concept_ids[5:, :] = 1
    sue = np.zeros((N, T), dtype=np.float32)
    for q in range(350, T, 100):
        sue[:2, q:q+3] = 3.0

    # 7. 保存文件
    np.save(out / "close.npy", close)
    np.save(out / "open.npy", open_)
    np.save(out / "high.npy", high)
    np.save(out / "low.npy", low)
    np.save(out / "volume.npy", volume)
    np.save(out / "amount.npy", amount)
    np.save(out / "valid_mask.npy", valid_mask)
    np.save(out / "market_index.npy", market_index)
    np.save(out / "market_index_dates.npy", np.array(all_dates, dtype=object))
    np.save(out / "fundamental_roe.npy", roe)
    np.save(out / "pe_ttm.npy", pe)
    np.save(out / "valuation_peTTM.npy", pe)
    np.save(out / "market_cap_total.npy", mcap)
    np.save(out / "sue.npy", sue)
    np.save(out / "concept_ids.npy", concept_ids)

    # 8. meta.json
    meta = {
        "shape": [N, T],
        "codes": [f"sh.60000{i}" for i in range(N)],
        "dates": all_dates,
        "fields": ["close", "open", "high", "low", "volume", "amount", "valid_mask"],
        "adj_type": "hfq",
        "build_time": datetime.datetime.now().isoformat(),
        "extra": {
            "is_physics_lab": True,
            "hfq_shift_day": 600,
            "warmup_days": 300
        }
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"DONE: Data generated at {out}")

if __name__ == "__main__":
    generate_physics_lab_data()
