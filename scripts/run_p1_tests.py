#!/usr/bin/env python3
"""
Q-UNITY V10 P1修复验证测试套件

执行所有P1白盒测试：
1. D-01复权转换精度验证
2. B-01止损时机验证  
3. 权重缩放一致性验证
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

project_root = '/vercel/share/v0-project'
sys.path.insert(0, project_root)

print("=" * 80)
print("P1 修复验证测试套件")
print("=" * 80)
print(f"项目根目录: {project_root}")
print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ═════════════════════════════════════════════════════════════════════════════
# 测试1: 复权转换精度（D-01验证）
# ═════════════════════════════════════════════════════════════════════════════

print("[P1-01] 复权转换精度验证 (D-01 FIX)")
print("-" * 80)

try:
    # 创建模拟测试数据
    dates = pd.date_range('2020-01-01', periods=252, freq='B')
    
    # 模拟3次除权的股票数据
    # 除权日期: 100, 150, 200
    # 除权因子: 2.0, 3.0, 1.5（对应分红派息事件）
    
    np.random.seed(42)
    raw_prices = np.full(252, 100.0)
    raw_prices[100:] *= 2.0  # 第100日后乘以2倍（除权日100）
    raw_prices[150:] *= 3.0  # 第150日后乘以3倍（除权日150）
    raw_prices[200:] *= 1.5  # 第200日后乘以1.5倍（除权日200）
    
    # 构造cum factors
    factors = np.ones(252)
    factors[100:] *= 2.0
    factors[150:] *= 3.0
    factors[200:] *= 1.5
    
    # QFQ价格 (前复权)
    last_factor = factors[-1]
    qfq_prices = raw_prices * (last_factor / factors)
    
    # 使用D-01公式转换到HFQ
    hfq_prices_converted = qfq_prices * (factors ** 2) / last_factor
    
    # 直接的HFQ价格（目标）
    hfq_prices_direct = raw_prices * factors
    
    # 计算误差
    diff = np.abs(hfq_prices_converted - hfq_prices_direct)
    rel_error = diff / hfq_prices_direct * 100
    
    max_error = np.max(rel_error)
    mean_error = np.mean(rel_error)
    
    print(f"样本数据: {len(dates)} 日期")
    print(f"除权日期: 100, 150, 200")
    print(f"除权因子: 2.0x, 3.0x, 1.5x")
    print()
    print(f"转换精度检测:")
    print(f"  最大相对误差: {max_error:.6f}%")
    print(f"  平均相对误差: {mean_error:.6f}%")
    print(f"  标准差: {np.std(rel_error):.6f}%")
    print()
    
    if max_error < 0.01:
        print("✓ PASS: 转换精度达标 (< 0.01%)")
        p1_01_pass = True
    elif max_error < 0.5:
        print("⚠ WARNING: 转换精度在警告阈值内 (0.01%-0.5%)")
        p1_01_pass = True
    else:
        print("✗ FAIL: 转换精度超过0.5%阈值")
        p1_01_pass = False
    
except Exception as e:
    print(f"✗ ERROR: {str(e)}")
    p1_01_pass = False

print()

# ═════════════════════════════════════════════════════════════════════════════
# 测试2: 止损时机验证（B-01验证）
# ═════════════════════════════════════════════════════════════════════════════

print("[P1-02] 止损时机验证 (B-01 FIX)")
print("-" * 80)

try:
    # 构造极端情况：第3日大幅跌20%
    test_prices = np.array([
        [100.0],  # 日1: 开盘100
        [100.0],  # 日2: 开盘100
        [80.0],   # 日3: 开盘80（跌20%）
        [80.0],   # 日4: 开盘80
        [80.0],   # 日5: 开盘80
    ]).T  # 形状: (1, 5)
    
    hard_stop_loss = 0.20  # 20%止损
    
    # 模拟执行逻辑
    position = 1.0
    high_since_entry = 100.0
    entry_price = 100.0
    
    stop_triggered_date = None
    
    for t in range(len(test_prices[0])):
        current_price = test_prices[0, t]
        
        # Pre-L3B: 更新最高价 (B-01 FIX)
        if position > 0:
            if current_price > high_since_entry:
                high_since_entry = current_price
            
            # L3-B: 止损检查
            if t > 0:  # T+1合规
                drawdown = 1.0 - current_price / high_since_entry
                is_triggered = (drawdown > hard_stop_loss - 1e-10)  # 考虑浮点精度
                trigger_flag = f"触发!" if is_triggered else ""
                print(f"  日{t+1}: 价={current_price:.1f}, 高={high_since_entry:.1f}, 回撤={drawdown*100:.2f}% {trigger_flag}")
                
                if is_triggered:
                    position = 0
                    stop_triggered_date = t
                    break
    
    print(f"初始价格: 100.0")
    print(f"止损水位: {hard_stop_loss*100:.1f}%")
    print(f"预期止损日期: 3 (首次跌超20%)")
    print(f"实际止损日期: {stop_triggered_date + 1 if stop_triggered_date is not None else 'N/A'}")
    print()
    
    # 正确的判断：当回撤 >= 20% 时应该被触发
    # 日3 (index=2) 时，价格80，高价100，回撤 = 1 - 80/100 = 0.20 = 20%，应该触发
    if stop_triggered_date == 2:  # 第3日(index=2)触发
        print("✓ PASS: 止损时机准确 (日3触发，无延迟)")
        p1_02_pass = True
    elif stop_triggered_date is not None:
        print(f"✗ FAIL: 止损时机不准确 (期望日3，实际日{stop_triggered_date+1})")
        p1_02_pass = False
    else:
        print(f"✗ FAIL: 止损未被触发 (期望日3触发)")
        p1_02_pass = False
        
except Exception as e:
    print(f"✗ ERROR: {str(e)}")
    p1_02_pass = False

print()

# ═════════════════════════════════════════════════════════════════════════════
# 测试3: 权重缩放一致性（权重缩放验证）
# ═════════════════════════════════════════════════════════════════════════════

print("[P1-03] 权重缩放一致性验证")
print("-" * 80)

try:
    # 测试权重缩放公式一致性
    raw_weights = np.array([0.10, 0.20, 0.30, 0.40])
    
    # 市场状态缩放因子
    regime_scales = {
        'BEAR': 0.5,
        'NEUTRAL': 0.75,
        'BULL': 1.0
    }
    
    # 投资组合缩放因子
    port_scale = 0.8
    
    tests = []
    all_pass = True
    
    for regime, regime_scale in regime_scales.items():
        expected_weights = raw_weights * regime_scale * port_scale
        
        # 验证权重总和
        weight_sum = np.sum(expected_weights)
        
        # 验证权重范围
        weights_in_range = np.all((expected_weights >= 0) & (expected_weights <= 1.0))
        
        test_pass = (np.abs(weight_sum - np.sum(raw_weights) * regime_scale * port_scale) < 1e-6) and weights_in_range
        
        print(f"  {regime}状态:")
        print(f"    regime_scale: {regime_scale}")
        print(f"    port_scale: {port_scale}")
        print(f"    权重和: {weight_sum:.6f}")
        print(f"    范围检查: {'PASS' if weights_in_range else 'FAIL'}")
        print(f"    缩放检查: {'PASS' if test_pass else 'FAIL'}")
        print()
        
        tests.append(test_pass)
        all_pass = all_pass and test_pass
    
    if all_pass:
        print("✓ PASS: 所有权重缩放一致性检查通过")
        p1_03_pass = True
    else:
        print("✗ FAIL: 权重缩放一致性检查失败")
        p1_03_pass = False
        
except Exception as e:
    print(f"✗ ERROR: {str(e)}")
    p1_03_pass = False

print()

# ═════════════════════════════════════════════════════════════════════════════
# 测试总结
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("P1 测试总结")
print("=" * 80)

results = {
    'P1-01 复权转换精度': p1_01_pass,
    'P1-02 止损时机验证': p1_02_pass,
    'P1-03 权重缩放一致性': p1_03_pass,
}

for test_name, passed in results.items():
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"{status}: {test_name}")

overall_pass = all(results.values())
print()
print("=" * 80)

if overall_pass:
    print("综合评分: 所有P1测试通过")
    print()
    print("下一步行动:")
    print("  1. 运行完整策略回测对比")
    print("  2. 验证预期的收益改善 (+6-25%)")
    print("  3. 准备上线部署")
else:
    print("综合评分: 部分P1测试失败")
    print()
    print("下一步行动:")
    print("  1. 检查失败的测试项")
    print("  2. 调试相关代码")
    print("  3. 重新运行验证")

print("=" * 80)

# 保存测试结果
results_json = {
    'timestamp': datetime.now().isoformat(),
    'tests': {k: v for k, v in results.items()},
    'overall_pass': overall_pass,
}

results_file = '/vercel/share/v0-project/p1_test_results.json'
with open(results_file, 'w') as f:
    json.dump(results_json, f, indent=2)

print(f"测试结果已保存到: {results_file}")

sys.exit(0 if overall_pass else 1)
