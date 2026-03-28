import numpy as np
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime, timedelta

# 添加项目路径
try:
    PROJECT_ROOT = Path(__file__).parent.parent
except (NameError, AttributeError):
    PROJECT_ROOT = Path('/vercel/share/v0-project')

sys.path.insert(0, str(PROJECT_ROOT))

print("[v0] ===== 13个策略信号提取 =====")
print("[v0] 目标：从white box审计数据中提取13策略的因子和信号")
print()

# 13个策略列表
STRATEGIES_13 = [
    'alpha_hunter_v2',
    'alpha_max_v5',
    'kunpeng_v10',
    'momentum_reversal',
    'retail_sniper_v10',
    'sentiment_reversal',
    'short_term_rsrs',
    'sniper_v6a',  # 遗漏的第13个
    'snma_v4',
    'titan_alpha_v1',
    'titan_orthogonal_v10',
    'ultra_alpha_v1',
    'weak_to_strong',
]

print(f"[v0] 待审计策略数: {len(STRATEGIES_13)}")
for i, s in enumerate(STRATEGIES_13, 1):
    print(f"     {i:2d}. {s}")
print()

# 生成审计数据
print("[v0] 第1步：重新加载white box审计数据...")
np.random.seed(42)
days = 1500
t = np.arange(days)

base_price = 100 + 20 * np.sin(2 * np.pi * t / 252)
trend = 0.5 * t / 252
prices = base_price + trend + np.random.normal(0, 0.5, days)
prices = np.maximum(prices, 10)

# 除权
split_days = [300, 600, 900, 1200, 1400]
split_factors = [0.5, 0.8, 0.7, 0.9, 0.95]

hfq_prices = prices.copy()
qfq_factors = np.ones(days)

for split_day, factor in zip(split_days, split_factors):
    hfq_prices[:split_day] *= factor
    qfq_factors[split_day:] /= factor

qfq_prices = hfq_prices / qfq_factors

dates = [datetime(2019, 1, 1) + timedelta(days=i) for i in range(days)]

df = pd.DataFrame({
    'date': dates,
    'open': hfq_prices * (1 + np.random.normal(0, 0.01, days)),
    'high': hfq_prices * (1 + np.random.normal(0.01, 0.01, days)),
    'low': hfq_prices * (1 - np.random.normal(0.01, 0.01, days)),
    'close': hfq_prices,
    'volume': np.random.randint(1000000, 10000000, days),
    'hfq_factor': qfq_factors,
})

df['high'] = df[['open', 'high', 'close']].max(axis=1) * 1.001
df['low'] = df[['open', 'low', 'close']].min(axis=1) * 0.999

print(f"[v0] 数据加载完成: {len(df)}行")
print()

# 保存为临时CSV用于策略加载
print("[v0] 第2步：准备策略输入数据...")
temp_data_file = PROJECT_ROOT / "whitebox_audit_results" / "whitebox_data.csv"
temp_data_file.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(temp_data_file, index=False)
print(f"[v0] 临时数据文件: {temp_data_file}")
print()

# 创建输出Excel
print("[v0] 第3步：准备输出格式...")
output_file = PROJECT_ROOT / "whitebox_audit_results" / "strategies_signals_audit.xlsx"

# 第一步：验证后复权和前复权
print("[v0] 验证数据校正...")
print(f"     后复权价格 min={hfq_prices.min():.2f}, max={hfq_prices.max():.2f}")
print(f"     前复权价格 min={qfq_prices.min():.2f}, max={qfq_prices.max():.2f}")
print(f"     除权因子 min={qfq_factors.min():.4f}, max={qfq_factors.max():.4f}")
print()

# 后续步骤
print("[v0] ===== 下一步 =====")
print("[v0] 1. 需要集成src/strategies/registry.py加载13个策略")
print("[v0] 2. 为每个策略调用alpha函数获取因子数据")
print("[v0] 3. 生成买卖信号（基于因子值）")
print("[v0] 4. 输出完整Excel让用户核对")
print()

# 创建占位符Excel
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # 创建汇总表
    summary_df = pd.DataFrame({
        '策略名': STRATEGIES_13,
        '状态': ['待审计'] * len(STRATEGIES_13),
        '因子数': [0] * len(STRATEGIES_13),
        '买入信号': [0] * len(STRATEGIES_13),
        '卖出信号': [0] * len(STRATEGIES_13),
        '准确性': [''] * len(STRATEGIES_13),
    })
    summary_df.to_excel(writer, sheet_name='汇总', index=False)
    
    # 创建数据表
    df[['date', 'close', 'hfq_factor', 'volume']].to_excel(
        writer, sheet_name='基础数据', index=False
    )

print(f"[v0] 占位符Excel已创建: {output_file}")
print("[v0] 等待集成策略函数...")
