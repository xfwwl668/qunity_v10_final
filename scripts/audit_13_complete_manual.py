#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
13个策略完整白盒审计报告 - 手动模式
==================================
"""

import os

PROJECT_ROOT = '/vercel/share/v0-project'

# 直接读取所有13个策略文件并生成报告
STRATEGIES = {
    '1. kunpeng_v10_alpha': 'src/strategies/vectorized/kunpeng_v10_alpha.py',
    '2. titan_orthogonal_v10_alpha': 'src/strategies/vectorized/titan_orthogonal_v10_alpha.py',
    '3. snma_v4_alpha': 'src/strategies/vectorized/snma_v4_alpha.py',
    '4. momentum_reversal_alpha': 'src/strategies/vectorized/momentum_reversal_alpha.py',
    '5. short_term_rsrs_alpha': 'src/strategies/vectorized/short_term_rsrs_alpha.py',
    '6. sentiment_reversal_alpha': 'src/strategies/vectorized/sentiment_reversal_alpha.py',
    '7. alpha_hunter_v2_alpha': 'src/strategies/vectorized/alpha_hunter_v2_alpha.py',
    '8. alpha_max_v5_alpha': 'src/strategies/vectorized/alpha_max_v5_alpha.py',
    '9. retail_sniper_v10_alpha': 'src/strategies/vectorized/retail_sniper_v10_alpha.py',
    '10. sniper_v6a_alpha': 'src/strategies/vectorized/sniper_v6a_alpha.py',
    '11. titan_alpha_v1_alpha': 'src/strategies/vectorized/titan_alpha_v1_alpha.py',
    '12. ultra_alpha_v1_alpha': 'src/strategies/vectorized/ultra_alpha_v1_alpha.py',
    '13. weak_to_strong_alpha': 'src/strategies/vectorized/weak_to_strong_alpha.py',
}

print("\n" + "="*90)
print("Q-UNITY V10 - 全13个策略白盒审计最终报告")
print("="*90 + "\n")

results = []
total_size = 0
total_funcs = 0

for display_name, rel_path in STRATEGIES.items():
    filepath = os.path.join(PROJECT_ROOT, rel_path)
    
    print(f"[{display_name}] 审计中...", end='', flush=True)
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = len(content.split('\n'))
        size_kb = len(content) / 1024
        func_count = content.count('def ')
        
        # 提取因子信息
        factors = []
        if 'smartmoney' in content.lower():
            factors.append('SmartMoney')
        if 'volatility' in content.lower():
            factors.append('Volatility')
        if 'reversal' in content.lower():
            factors.append('Reversal')
        if 'momentum' in content.lower():
            factors.append('Momentum')
        if 'ma' in content.lower() or 'moving' in content.lower():
            factors.append('MA')
        if 'rsi' in content.lower():
            factors.append('RSI')
        if 'rsrs' in content.lower():
            factors.append('RSRS')
        if 'sentiment' in content.lower():
            factors.append('Sentiment')
        if 'breadth' in content.lower():
            factors.append('Breadth')
        if 'zscore' in content.lower():
            factors.append('ZScore')
        
        # 检查买卖逻辑
        has_buy = 'buy' in content.lower() or 'long' in content.lower()
        has_sell = 'sell' in content.lower() or 'short' in content.lower()
        has_weights = 'weights' in content.lower()
        
        # 检查前视偏差
        lookahead_markers = content.count('[t+1]') + content.count('[t+2]') + \
                           content.count('shift(-') + content.count('[t +')
        
        results.append({
            'name': display_name.split('. ')[1],
            'lines': lines,
            'size_kb': size_kb,
            'functions': func_count,
            'factors': factors,
            'has_buy': has_buy,
            'has_sell': has_sell,
            'has_weights': has_weights,
            'lookahead_markers': lookahead_markers,
            'safe': lookahead_markers == 0
        })
        
        total_size += size_kb
        total_funcs += func_count
        
        status = "✓" if lookahead_markers == 0 else "⚠️"
        print(f" {status} {lines:5d}行 {size_kb:5.1f}KB {len(factors):2d}个因子")
        
    except Exception as e:
        print(f" ✗ 错误: {e}")

# 生成汇总
print("\n" + "="*90)
print("审计汇总")
print("="*90 + "\n")

if results:
    print(f"代码规模:")
    print(f"  • 策略总数:        {len(results)}")
    print(f"  • 代码总大小:      {total_size:.1f} KB")
    print(f"  • 平均策略大小:    {total_size/len(results):.1f} KB")
    print(f"  • 平均代码行数:    {sum(r['lines'] for r in results)//len(results)}")
    print(f"  • 函数总数:        {total_funcs}")
    
    print(f"\n信号逻辑:")
    buy_count = sum(1 for r in results if r['has_buy'])
    sell_count = sum(1 for r in results if r['has_sell'])
    weight_count = sum(1 for r in results if r['has_weights'])
    print(f"  • 带买入逻辑:      {buy_count}/{len(results)}")
    print(f"  • 带卖出逻辑:      {sell_count}/{len(results)}")
    print(f"  • 权重分配:        {weight_count}/{len(results)}")
    
    print(f"\n因子多样性:")
    all_factors = set()
    for r in results:
        all_factors.update(r['factors'])
    print(f"  • 全局独特因子:    {len(all_factors)}")
    print(f"  • 因子列表:        {', '.join(sorted(all_factors))}")
    print(f"  • 平均因子/策略:   {sum(len(r['factors']) for r in results)/len(results):.1f}")
    
    print(f"\n数据安全性:")
    safe_count = sum(1 for r in results if r['safe'])
    print(f"  • 无前视偏差:      {safe_count}/{len(results)}")
    print(f"  • 前视偏差风险:    {len(results) - safe_count}/{len(results)}")
    
    print(f"\n策略详情表:")
    print("-"*90)
    print(f"{'#':>2} {'策略名称':<35} {'行数':>6} {'大小':>7} {'因子':>4} {'买':>3} {'卖':>3} {'安全':>4}")
    print("-"*90)
    
    for i, r in enumerate(results, 1):
        buy_marker = "✓" if r['has_buy'] else " "
        sell_marker = "✓" if r['has_sell'] else " "
        safe_marker = "✓" if r['safe'] else "⚠️"
        
        factor_str = f"{len(r['factors'])}"
        print(f"{i:2d} {r['name']:<35} {r['lines']:6d} {r['size_kb']:6.1f}KB {factor_str:>4} {buy_marker:>3} {sell_marker:>3} {safe_marker:>4}")
    
    print("\n因子使用频率Top-10:")
    factor_freq = {}
    for r in results:
        for f in r['factors']:
            factor_freq[f] = factor_freq.get(f, 0) + 1
    
    for factor, freq in sorted(factor_freq.items(), key=lambda x: -x[1])[:10]:
        print(f"  {factor:<20}: {freq:2d}/13 策略")

print("\n" + "="*90)
print("白盒审计完成!")
print("="*90 + "\n")

# 保存报告
output_path = os.path.join(PROJECT_ROOT, 'whitebox_audit/output/13_strategies_complete_audit.txt')
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, 'w', encoding='utf-8') as f:
    f.write("="*90 + "\n")
    f.write("Q-UNITY V10 - 全13个策略白盒审计最终报告\n")
    f.write("="*90 + "\n\n")
    
    f.write("汇总统计:\n")
    f.write(f"  策略总数: {len(results)}\n")
    f.write(f"  代码总大小: {total_size:.1f} KB\n")
    f.write(f"  无前视偏差: {safe_count}/{len(results)}\n\n")
    
    f.write("详细列表:\n")
    for r in results:
        f.write(f"\n{r['name']}\n")
        f.write(f"  行数: {r['lines']}\n")
        f.write(f"  大小: {r['size_kb']:.1f} KB\n")
        f.write(f"  因子: {', '.join(r['factors'])}\n")
        f.write(f"  买入: {'是' if r['has_buy'] else '否'}\n")
        f.write(f"  卖出: {'是' if r['has_sell'] else '否'}\n")
        f.write(f"  权重: {'是' if r['has_weights'] else '否'}\n")
        f.write(f"  安全: {'是' if r['safe'] else '否'}\n")

print(f"✓ 报告已保存: {output_path}\n")
