import numpy as np
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime, timedelta

# 添加项目路径（兼容exec执行）
try:
    PROJECT_ROOT = Path(__file__).parent.parent
except (NameError, AttributeError):
    PROJECT_ROOT = Path('/vercel/share/v0-project')

sys.path.insert(0, str(PROJECT_ROOT))

print("[v0] ===== Q-UNITY V10 深度白盒审计框架 =====")
print("[v0] 目标：13个策略 × 1500天正弦波数据 → Excel报告")
print()

# ============================================================================
# 第1步：生成1500天正弦波+除权数据
# ============================================================================
print("[v0] 第1步：生成1500天正弦波数据...")

np.random.seed(42)
days = 1500
t = np.arange(days)

# 基础价格：正弦波 + 趋势
base_price = 100 + 20 * np.sin(2 * np.pi * t / 252)  # 年周期
trend = 0.5 * t / 252  # 缓慢上升趋势
prices = base_price + trend + np.random.normal(0, 0.5, days)
prices = np.maximum(prices, 10)  # 确保价格>0

print(f"[v0] 生成{days}天价格数据")
print(f"[v0] 价格范围: {prices.min():.2f} - {prices.max():.2f}")

# 生成除权事件（模拟5次除权）
split_days = [300, 600, 900, 1200, 1400]
split_factors = [0.5, 0.8, 0.7, 0.9, 0.95]  # 除权倍数

hfq_prices = prices.copy()  # 后复权价格
qfq_factors = np.ones(days)  # 前复权因子

# 应用除权到后复权价格
for split_day, factor in zip(split_days, split_factors):
    hfq_prices[:split_day] *= factor
    qfq_factors[split_day:] /= factor

print(f"[v0] 添加{len(split_days)}次除权事件")

# 生成前复权价格
qfq_prices = hfq_prices / qfq_factors

print(f"[v0] 后复权价格范围: {hfq_prices.min():.2f} - {hfq_prices.max():.2f}")
print(f"[v0] 前复权价格范围: {qfq_prices.min():.2f} - {qfq_prices.max():.2f}")
print()

# ============================================================================
# 第2步：创建DataFrame准备
# ============================================================================
print("[v0] 第2步：创建交易数据DataFrame...")

# 生成OHLCV数据
dates = [datetime(2019, 1, 1) + timedelta(days=i) for i in range(days)]

df = pd.DataFrame({
    'date': dates,
    'open': hfq_prices * (1 + np.random.normal(0, 0.01, days)),
    'high': hfq_prices * (1 + np.random.normal(0.01, 0.01, days)),
    'low': hfq_prices * (1 - np.random.normal(0.01, 0.01, days)),
    'close': hfq_prices,
    'volume': np.random.randint(1000000, 10000000, days),
    'hfq_factor': qfq_factors,  # 后复权因子
})

# 确保OHLC关系合理
df['high'] = df[['open', 'high', 'close']].max(axis=1) * 1.001
df['low'] = df[['open', 'low', 'close']].min(axis=1) * 0.999

print(f"[v0] 创建DataFrame: {len(df)}行 × {len(df.columns)}列")
print(f"[v0] 日期范围: {df['date'].iloc[0].date()} - {df['date'].iloc[-1].date()}")
print()

# ============================================================================
# 第3步：计算技术指标
# ============================================================================
print("[v0] 第3步：计算技术指标...")

# RSI
def calc_rsi(prices, period=14):
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = pd.Series(gains).rolling(period).mean().values
    avg_loss = pd.Series(losses).rolling(period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    # 补齐长度
    rsi_full = np.full(len(prices), np.nan)
    rsi_full[len(deltas)-len(rsi)+period:] = rsi[-(len(prices)-period):]
    # 更简单的方法：直接返回正确长度
    rsi_series = pd.Series(np.nan, index=range(len(prices)))
    for i in range(period, len(prices)):
        gains_slice = gains[max(0,i-period):i]
        losses_slice = losses[max(0,i-period):i]
        avg_g = np.mean(gains_slice)
        avg_l = np.mean(losses_slice)
        rs_val = avg_g / (avg_l + 1e-10)
        rsi_series.iloc[i] = 100 - (100 / (1 + rs_val))
    return rsi_series.values

# MACD
def calc_macd(prices):
    ema12 = pd.Series(prices).ewm(span=12).mean().values
    ema26 = pd.Series(prices).ewm(span=26).mean().values
    macd = ema12 - ema26
    signal = pd.Series(macd).ewm(span=9).mean().values
    hist = macd - signal
    return macd, signal, hist

# 布林带
def calc_bollinger(prices, period=20):
    sma = pd.Series(prices).rolling(period).mean().values
    std = pd.Series(prices).rolling(period).std().values
    upper = sma + 2 * std
    lower = sma - 2 * std
    return upper, sma, lower

df['rsi'] = calc_rsi(hfq_prices)
df['macd'], df['macd_signal'], df['macd_hist'] = calc_macd(hfq_prices)
df['bb_upper'], df['bb_middle'], df['bb_lower'] = calc_bollinger(hfq_prices)

print("[v0] 已计算: RSI, MACD, 布林带")
print()

# ============================================================================
# 第4步：显示数据样本
# ============================================================================
print("[v0] 第4步：数据样本（前10行）...")
print(df[['date', 'close', 'hfq_factor', 'rsi', 'macd', 'bb_upper']].head(10).to_string())
print()

# ============================================================================
# 第5步：显示关键指标
# ============================================================================
print("[v0] 第5步：关键指标统计...")
print(f"[v0] 后复权价格统计:")
print(f"     均值: {hfq_prices.mean():.2f}")
print(f"     标差: {hfq_prices.std():.2f}")
print(f"     变化率: {(hfq_prices[-1] - hfq_prices[0]) / hfq_prices[0] * 100:.2f}%")
print()
print(f"[v0] 技术指标统计:")
print(f"     RSI均值: {df['rsi'].mean():.2f}")
print(f"     MACD均值: {df['macd'].mean():.4f}")
print()

# ============================================================================
# 第6步：准备输出Excel
# ============================================================================
print("[v0] 第6步：准备输出Excel...")

# 创建输出目录
output_dir = PROJECT_ROOT / "whitebox_audit_results"
output_dir.mkdir(parents=True, exist_ok=True)

# 保存基础数据
output_file = output_dir / "whitebox_audit_data.xlsx"
print(f"[v0] 输出文件: {output_file}")

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # Sheet1: 基础数据 (后复权 + 前复权平滑)
    export_df = df[[
        'date', 'close', 'hfq_factor', 
        'rsi', 'macd', 'macd_signal', 
        'bb_upper', 'bb_middle', 'bb_lower'
    ]].copy()
    
    export_df.rename(columns={
        'close': '后复权价格',
        'hfq_factor': '除权因子',
        'rsi': 'RSI(技术指标)',
        'macd': 'MACD',
        'macd_signal': 'MACD_Signal',
        'bb_upper': '布林_上轨',
        'bb_middle': '布林_中轨',
        'bb_lower': '布林_下轨'
    }, inplace=True)
    
    export_df.to_excel(writer, sheet_name='数据概览', index=False)
    
    print("[v0] 已导出: Sheet1 '数据概览' (基础数据)")

print("[v0] ✓ Excel文件已生成!")
print()

# ============================================================================
# 总结
# ============================================================================
print("[v0] ===== 白盒审计框架已就绪 =====")
print("[v0] 下一步:")
print("[v0] 1. 手工检查Excel中的后复权/前复权/除权数据")
print("[v0] 2. 集成13个策略，提取各自的因子和信号")
print("[v0] 3. 输出完整Excel供人工核对")
print()
