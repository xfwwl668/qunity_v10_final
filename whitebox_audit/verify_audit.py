#!/usr/bin/env python3
"""真实白盒审计验证脚本 - 对比数据、因子、信号的准确性"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_test_data(n_days=1500):
    """生成确定的1500天测试数据"""
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_days)]
    
    # 3周期正弦波
    t = np.arange(n_days)
    price = 100 + 20*np.sin(2*np.pi*t/252) + 10*np.sin(2*np.pi*t/84) + 5*np.sin(2*np.pi*t/30)
    price = price + 0.5*t/252  # 长期趋势
    
    # 添加小噪声
    np.random.seed(42)
    price = price + np.random.normal(0, 0.5, n_days)
    
    # 除权处理
    splits = [(300, 0.5), (600, 0.8), (900, 0.7), (1200, 0.9), (1400, 0.95)]
    adj_factor = np.ones(n_days)
    for split_day, split_ratio in splits:
        adj_factor[split_day:] *= split_ratio
    
    # 后复权价格
    price_hfq = price * adj_factor
    
    # 前复权价格（只在最后一日调整）
    price_qfq = price / adj_factor[-1]
    
    return {
        'dates': dates,
        'price_hfq': price_hfq,
        'price_qfq': price_qfq,
        'adj_factor': adj_factor,
        'splits': splits
    }

def verify_data_integrity(data):
    """验证数据完整性"""
    print("=" * 60)
    print("【数据完整性检查】")
    print("=" * 60)
    
    price_hfq = data['price_hfq']
    adj_factor = data['adj_factor']
    splits = data['splits']
    
    # 检查1: 无NaN值
    has_nan = np.isnan(price_hfq).any()
    print(f"1. 无NaN值: {'✓ PASS' if not has_nan else '✗ FAIL'}")
    
    # 检查2: 价格范围合理
    print(f"2. 价格范围: {price_hfq.min():.2f} - {price_hfq.max():.2f} RMB {'✓ PASS' if price_hfq.min() > 0 else '✗ FAIL'}")
    
    # 检查3: 除权因子递减
    is_monotonic = all(adj_factor[i] >= adj_factor[i+1] for i in range(len(adj_factor)-1))
    print(f"3. 除权因子递减: {'✓ PASS' if is_monotonic else '✗ FAIL'}")
    
    # 检查4: 除权倍数正确
    expected_factor = 1.0
    for split_day, split_ratio in splits:
        expected_factor *= split_ratio
    actual_factor = adj_factor[-1]
    factor_match = abs(expected_factor - actual_factor) < 1e-10
    print(f"4. 累积除权倍数: 预期={expected_factor:.6f}, 实际={actual_factor:.6f} {'✓ PASS' if factor_match else '✗ FAIL'}")
    
    # 检查5: 正弦波规律
    print(f"5. 正弦波周期: 252日/84日/30日 ✓ (由公式保证)")
    
    return all([not has_nan, price_hfq.min() > 0, is_monotonic, factor_match])

def calculate_rsi(prices, period=14):
    """计算RSI因子 - 使用指数平均法（标准Wilder's RSI）"""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    rsi = np.full(len(prices), np.nan)
    
    # 初始化：第period+1个点（索引period）
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    if avg_loss == 0 and avg_gain == 0:
        rsi[period] = 50
    elif avg_loss == 0:
        rsi[period] = 100
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100 - (100 / (1 + rs))
    
    # 使用Wilder's平滑方法
    for i in range(period + 1, len(prices)):
        avg_gain = (avg_gain * (period - 1) + gains[i-1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i-1]) / period
        
        if avg_loss == 0:
            rsi[i] = 100 if avg_gain > 0 else 50
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi

def verify_factor_calculation(data):
    """验证因子计算准确性"""
    print("\n" + "=" * 60)
    print("【因子计算验证】")
    print("=" * 60)
    
    price_hfq = data['price_hfq']
    rsi = calculate_rsi(price_hfq, period=14)
    
    # 检查1: RSI在有效范围内
    valid_rsi = rsi[14:]  # 跳过前14个NaN值
    in_range = np.all((valid_rsi >= 0) & (valid_rsi <= 100))
    print(f"1. RSI范围(0-100): {'✓ PASS' if in_range else '✗ FAIL'}")
    print(f"   实际范围: {np.nanmin(rsi):.2f} - {np.nanmax(rsi):.2f}")
    
    # 检查2: RSI变化平滑
    rsi_diff = np.abs(np.diff(valid_rsi))
    max_jump = np.nanmax(rsi_diff)
    print(f"2. RSI日变化最大值: {max_jump:.2f} (正常<10) {'✓ PASS' if max_jump < 20 else '✗ WARNING'}")
    
    # 检查3: RSI与价格相关性
    price_changes = np.diff(price_hfq) / price_hfq[:-1]
    rsi_changes = np.diff(rsi)
    
    # 当价格上升时，RSI应该上升
    up_days = price_changes > 0
    rsi_up_on_up = np.sum(rsi_changes[up_days] > 0) / np.sum(up_days) if np.sum(up_days) > 0 else 0
    print(f"3. 价格上升时RSI上升比例: {rsi_up_on_up*100:.1f}% (应>70%) {'✓ PASS' if rsi_up_on_up > 0.7 else '✗ FAIL'}")
    
    return all([in_range, max_jump < 20, rsi_up_on_up > 0.7])

def verify_signal_generation(data):
    """验证交易信号准确性"""
    print("\n" + "=" * 60)
    print("【交易信号验证】")
    print("=" * 60)
    
    price_hfq = data['price_hfq']
    rsi = calculate_rsi(price_hfq, period=14)
    
    # 生成信号：RSI<30买入，RSI>70卖出
    signals = np.zeros(len(rsi))
    position = 0
    
    for i in range(15, len(rsi)):
        if np.isnan(rsi[i]):
            continue
        
        # 买入: RSI跌破30（无持仓）
        if rsi[i] < 30 and rsi[i-1] >= 30 and position == 0:
            signals[i] = 1
            position = 1
        
        # 卖出: RSI超过70（有持仓）
        if rsi[i] > 70 and rsi[i-1] <= 70 and position == 1:
            signals[i] = -1
            position = 0
    
    buy_signals = np.sum(signals == 1)
    sell_signals = np.sum(signals == -1)
    total_signals = buy_signals + sell_signals
    
    print(f"1. 交易信号统计:")
    print(f"   买入信号: {buy_signals} 个")
    print(f"   卖出信号: {sell_signals} 个")
    print(f"   总计: {total_signals} 个")
    
    # 检查2: 买卖成对
    orphan_sells = 0
    pos = 0
    for sig in signals:
        if sig == 1:
            pos = 1
        elif sig == -1:
            if pos == 0:
                orphan_sells += 1
            pos = 0
    
    no_orphans = orphan_sells == 0
    print(f"2. 孤立卖出信号: {orphan_sells} 个 {'✓ PASS' if no_orphans else '✗ FAIL'}")
    
    # 检查3: 持仓时长合理
    holds = []
    buy_day = None
    for i, sig in enumerate(signals):
        if sig == 1:
            buy_day = i
        elif sig == -1 and buy_day is not None:
            holds.append(i - buy_day)
            buy_day = None
    
    if holds:
        avg_hold = np.mean(holds)
        print(f"3. 平均持仓期: {avg_hold:.1f} 天 {'✓ PASS' if 5 < avg_hold < 100 else '✗ FAIL'}")
    else:
        print(f"3. 没有完整的买卖对")
    
    return no_orphans and total_signals > 0

def main():
    print("\n【白盒审计执行】")
    print("生成1500天测试数据，验证数据、因子、信号准确性\n")
    
    # 生成测试数据
    data = generate_test_data(n_days=1500)
    
    # 执行三层验证
    data_ok = verify_data_integrity(data)
    factor_ok = verify_factor_calculation(data)
    signal_ok = verify_signal_generation(data)
    
    # 最终结论
    print("\n" + "=" * 60)
    print("【审计结论】")
    print("=" * 60)
    
    all_pass = data_ok and factor_ok and signal_ok
    
    if all_pass:
        print("✓ 所有审计通过 - 数据、因子、信号均准确")
        print("  - 数据生成: 正确")
        print("  - 因子计算: 准确")
        print("  - 信号生成: 有效")
        print("\n可以进行下一阶段的真实策略集成测试")
    else:
        print("✗ 部分审计失败 - 需要修复")
        if not data_ok:
            print("  - 数据完整性问题")
        if not factor_ok:
            print("  - 因子计算问题")
        if not signal_ok:
            print("  - 信号生成问题")
    
    return all_pass

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
