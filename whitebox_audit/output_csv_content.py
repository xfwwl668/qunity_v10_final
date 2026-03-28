#!/usr/bin/env python3
import numpy as np
import pandas as pd

# 生成1500天数据
n_days = 1500
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

# 生成CSV内容
csv_lines = ['Date,Close_HFQ,Adj_Factor,Split_Day']
for d in range(n_days):
    date_str = dates[d].strftime('%Y-%m-%d')
    close_str = f'{close_hfq[d]:.2f}'
    adj_str = f'{adj_factor[d]:.4f}'
    split_str = '1' if any(sd == d for sd, _ in splits) else '0'
    csv_lines.append(f'{date_str},{close_str},{adj_str},{split_str}')

csv_content = '\n'.join(csv_lines)
print(csv_content)
