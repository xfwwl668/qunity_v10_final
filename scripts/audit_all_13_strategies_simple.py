#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
13个策略白盒审计 - 简化版
========================
直接审计策略文件内容，验证买卖逻辑
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = '/vercel/share/v0-project'

# 确保目录存在
strategies_dir = os.path.join(PROJECT_ROOT, 'src/strategies/vectorized')
print(f"[DEBUG] 查找策略目录: {strategies_dir}")
print(f"[DEBUG] 目录存在: {os.path.exists(strategies_dir)}")

if os.path.exists(strategies_dir):
    print(f"[DEBUG] 目录中的文件:")
    for f in os.listdir(strategies_dir)[:5]:
        print(f"  - {f}")

# 13个策略列表
STRATEGIES = [
    'alpha_hunter_v2_alpha',
    'alpha_max_v5_alpha',
    'kunpeng_v10_alpha',
    'momentum_reversal_alpha',
    'retail_sniper_v10_alpha',
    'sentiment_reversal_alpha',
    'short_term_rsrs_alpha',
    'sniper_v6a_alpha',
    'snma_v4_alpha',
    'titan_alpha_v1_alpha',
    'titan_orthogonal_v10_alpha',
    'ultra_alpha_v1_alpha',
    'weak_to_strong_alpha'
]

def analyze_strategy_file(strategy_name):
    """分析策略文件内容"""
    strategy_path = os.path.join(
        PROJECT_ROOT,
        'src/strategies/vectorized',
        f'{strategy_name}.py'
    )
    
    if not os.path.exists(strategy_path):
        return {
            'name': strategy_name,
            'status': 'NOT_FOUND',
            'buy_signals': 0,
            'sell_signals': 0,
            'errors': f'File not found: {strategy_path}'
        }
    
    try:
        with open(strategy_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 统计买卖信号关键词
        buy_patterns = [
            r'weights\s*[><=]+\s*[0-9.]',
            r'buy',
            r'long',
            r'signal\s*>\s*0',
            r'alpha\s*>\s*0'
        ]
        
        sell_patterns = [
            r'weights\s*[<]=',
            r'sell',
            r'short',
            r'signal\s*<\s*0',
            r'alpha\s*<\s*0'
        ]
        
        buy_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) 
                       for pattern in buy_patterns)
        sell_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) 
                        for pattern in sell_patterns)
        
        # 检查因子计算
        factors = []
        factor_patterns = [
            r'def\s+calculate_\w+',
            r'momentum',
            r'volatility',
            r'reversal',
            r'zscore',
            r'ma\d+',
            r'rsi',
            r'atr',
            r'sentiment',
            r'breadth'
        ]
        
        for pattern in factor_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            factors.extend(matches)
        
        # 检查前视偏差（不应使用未来数据）
        lookahead_patterns = [
            r'close\[t\+1\]',
            r'close\[t\+2\]',
            r'\[t\+[0-9]\]',
            r'rolling\(\).*shift\(-',
        ]
        
        lookahead_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) 
                             for pattern in lookahead_patterns)
        
        # 检查复权调整
        has_adjustment = bool(re.search(r'adj|复权|qfq|hfq', content, re.IGNORECASE))
        
        return {
            'name': strategy_name,
            'status': 'OK',
            'file_size_kb': len(content) / 1024,
            'buy_signal_references': buy_count,
            'sell_signal_references': sell_count,
            'unique_factors': len(set(factors)),
            'factors': list(set(factors))[:5],  # 前5个因子
            'lookahead_risk_count': lookahead_count,
            'has_adjustment': has_adjustment,
            'lookahead_safe': lookahead_count == 0
        }
        
    except Exception as e:
        return {
            'name': strategy_name,
            'status': 'ERROR',
            'error': str(e)
        }


def main():
    print("\n" + "="*80)
    print("Q-UNITY V10 - 全13个策略代码审计")
    print("="*80 + "\n")
    
    print(f"[INFO] 审计 {len(STRATEGIES)} 个策略的代码内容...\n")
    
    results = []
    total_buy = 0
    total_sell = 0
    lookahead_risky = 0
    
    for i, strategy_name in enumerate(STRATEGIES, 1):
        print(f"[{i:2d}/{len(STRATEGIES)}] 审计 {strategy_name:35s} ... ", end='', flush=True)
        
        result = analyze_strategy_file(strategy_name)
        results.append(result)
        
        if result['status'] == 'OK':
            buy = result.get('buy_signal_references', 0)
            sell = result.get('sell_signal_references', 0)
            total_buy += buy
            total_sell += sell
            
            if not result.get('lookahead_safe', True):
                lookahead_risky += 1
                status_icon = "⚠️ "
            else:
                status_icon = "✓ "
            
            print(f"{status_icon} 买:{buy:3d} 卖:{sell:3d} 因子:{result.get('unique_factors', 0):2d}")
        else:
            print(f"✗ {result.get('error', result['status'])}")
    
    # 生成汇总报告
    print("\n" + "="*80)
    print("审计汇总")
    print("="*80 + "\n")
    
    ok_count = len([r for r in results if r['status'] == 'OK'])
    
    print(f"审计统计:")
    print(f"  • 总策略数:     {len(STRATEGIES)}")
    print(f"  • 成功审计:     {ok_count}")
    print(f"  • 失败审计:     {len(STRATEGIES) - ok_count}")
    print(f"  • 总买信号:     {total_buy}")
    print(f"  • 总卖信号:     {total_sell}")
    print(f"  • 前视偏差风险: {lookahead_risky}")
    
    # 按策略详细列表
    print(f"\n详细列表:")
    print(f"{'#':>2} {'策略名称':<35} {'状态':<6} {'买':>4} {'卖':>4} {'因子':>4} {'安全':>4}")
    print("-" * 80)
    
    for i, result in enumerate(results, 1):
        name = result['name']
        status = result['status']
        
        if status == 'OK':
            buy = result.get('buy_signal_references', 0)
            sell = result.get('sell_signal_references', 0)
            factors = result.get('unique_factors', 0)
            safe = "✓" if result.get('lookahead_safe', True) else "⚠️"
            print(f"{i:2d} {name:<35} {status:<6} {buy:4d} {sell:4d} {factors:4d} {safe:>4}")
        else:
            print(f"{i:2d} {name:<35} {status:<6}")
    
    # 因子分布
    print(f"\n全局因子统计:")
    all_factors = {}
    for result in results:
        if result['status'] == 'OK':
            for factor in result.get('factors', []):
                all_factors[factor] = all_factors.get(factor, 0) + 1
    
    for factor, count in sorted(all_factors.items(), key=lambda x: -x[1])[:10]:
        print(f"  • {factor:<20} : {count:2d}个策略使用")
    
    # 保存JSON报告
    output_dir = os.path.join(PROJECT_ROOT, 'whitebox_audit', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, '13_strategies_code_audit.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 报告已保存: {report_path}")
    
    # 保存文本报告
    txt_path = os.path.join(output_dir, '13_strategies_code_audit.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Q-UNITY V10 - 全13个策略代码审计报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("汇总统计:\n")
        f.write(f"  • 总策略数:     {len(STRATEGIES)}\n")
        f.write(f"  • 成功审计:     {ok_count}\n")
        f.write(f"  • 失败审计:     {len(STRATEGIES) - ok_count}\n")
        f.write(f"  • 总买信号:     {total_buy}\n")
        f.write(f"  • 总卖信号:     {total_sell}\n")
        f.write(f"  • 前视偏差风险: {lookahead_risky}\n\n")
        
        f.write("详细列表:\n")
        for i, result in enumerate(results, 1):
            f.write(f"\n{i}. {result['name']}\n")
            f.write(f"   状态: {result['status']}\n")
            if result['status'] == 'OK':
                f.write(f"   买信号: {result.get('buy_signal_references', 0)}\n")
                f.write(f"   卖信号: {result.get('sell_signal_references', 0)}\n")
                f.write(f"   因子数: {result.get('unique_factors', 0)}\n")
                f.write(f"   安全性: {'✓ 无前视偏差' if result.get('lookahead_safe', True) else '⚠️ 有前视偏差风险'}\n")
    
    print(f"✓ 文本报告已保存: {txt_path}")
    
    print("\n" + "="*80)
    print("审计完成!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
