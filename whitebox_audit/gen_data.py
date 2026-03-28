#!/usr/bin/env python3
import numpy as np
import pandas as pd
from datetime import datetime

# 生成1500天数据
n_days = 1500
n_stocks = 50

rng = np.random.default_rng(42)
dates = pd.date_range('2020-01-01', periods=n_days, freq='D')

# 3周期正弦波
t = np.arange(n_days)
sin1 = 20 * np.sin(2 * np.pi * t / 252)
sin2 = 10 * np.sin(2 * np.pi * t / 84)
sin3 = 5 * np.sin(2 * np.pi * t / 30)
trend = 0.5 * t / n_days * 20
noise = rng.normal(0, 2, n_days)
base_price = 100 + sin1 + sin2 + sin3 + trend + noise

# 除权处理
splits = [(300, 0.5), (600, 0.8), (900, 0.7), (1200, 0.9), (1400, 0.95)]
adj_factor = np.ones(n_days)
for split_day, split_ratio in splits:
    if split_day < n_days:
        adj_factor[split_day:] *= split_ratio

# 后复权价格
close_hfq = base_price * adj_factor

# 基础数据CSV
df_base = pd.DataFrame({
    'Date': [d.strftime('%Y-%m-%d') for d in dates],
    'Close_HFQ': [f'{p:.2f}' for p in close_hfq],
    'Adj_Factor': [f'{f:.4f}' for f in adj_factor],
    'Split_Day': [1 if any(sd == d for sd, _ in splits) else 0 for d in range(n_days)],
})

# 输出前10行到stdout
print("=== 基础数据 (前10行) ===")
print(df_base.head(10).to_string(index=False))
print(f"\n总行数: {len(df_base)}")

# 获取完整CSV内容
csv_content = df_base.to_csv(index=False)

# 输出CSV内容到stdout供保存
print("\n=== 完整CSV内容 ===")
print(csv_content)

# 同时写到文件
import sys
import os
sys.path.insert(0, '/vercel/share/v0-project')
os.makedirs('/vercel/share/v0-project/whitebox_audit', exist_ok=True)
output_path = '/vercel/share/v0-project/whitebox_audit/01_base_data.csv'
df_base.to_csv(output_path, index=False)
print(f"\n已写入文件: {output_path}")
