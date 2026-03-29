#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
13个策略白盒审计完整版
=====================
基于已验证的13个策略文件进行完整代码审计
"""

import os
import re

# 这是从之前 glob 找到的13个策略文件
PROJECT_ROOT = '/vercel/share/v0-project'

STRATEGY_FILES = [
    os.path.join(PROJECT_ROOT, 'src/strategies/vectorized/alpha_hunter_v2_alpha.py'),
    os.path.join(PROJECT_ROOT, 'src/strategies/vectorized/alpha_max_v5_alpha.py'),
    os.path.join(PROJECT_ROOT, 'src/strategies/vectorized/kunpeng_v10_alpha.py'),
    os.path.join(PROJECT_ROOT, 'src/strategies/vectorized/momentum_reversal_alpha.py'),
    os.path.join(PROJECT_ROOT, 'src/strategies/vectorized/retail_sniper_v10_alpha.py'),
    os.path.join(PROJECT_ROOT, 'src/strategies/vectorized/sentiment_reversal_alpha.py'),
    os.path.join(PROJECT_ROOT, 'src/strategies/vectorized/short_term_rsrs_alpha.py'),
    os.path.join(PROJECT_ROOT, 'src/strategies/vectorized/sniper_v6a_alpha.py'),
    os.path.join(PROJECT_ROOT, 'src/strategies/vectorized/snma_v4_alpha.py'),
    os.path.join(PROJECT_ROOT, 'src/strategies/vectorized/titan_alpha_v1_alpha.py'),
    os.path.join(PROJECT_ROOT, 'src/strategies/vectorized/titan_orthogonal_v10_alpha.py'),
    os.path.join(PROJECT_ROOT, 'src/strategies/vectorized/ultra_alpha_v1_alpha.py'),
    os.path.join(PROJECT_ROOT, 'src/strategies/vectorized/weak_to_strong_alpha.py'),
]

def extract_strategy_name(filepath):
    """从文件名提取策略名"""
    return os.path.basename(filepath).replace('.py', '')

def analyze_strategy_code(filepath):
    """分析策略代码"""
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取策略名
        strategy_name = extract_strategy_name(filepath)
        
        # 查找主函数定义
        main_func_match = re.search(r'def\s+(\w+_alpha)\s*\(', content)
        main_func_name = main_func_match.group(1) if main_func_match else 'unknown'
        
        # 统计关键词
        weights_assignments = len(re.findall(r'weights\s*[+\-*/%=]', content))
        buy_logic = len(re.findall(r'(buy|long|0\.)', content, re.IGNORECASE))
        sell_logic = len(re.findall(r'(sell|short|-)', content, re.IGNORECASE))
        
        # 提取因子使用
        factors = []
        factor_keywords = {
            'momentum': r'momentum|ret_|return|price_change',
            'volatility': r'volatility|std|atr|range',
            'mean_reversion': r'zscore|mean|revert|normal',
            'sentiment': r'sentiment|breadth|advance|decline',
            'rsi': r'\brsi\b|\brs\b',
            'macd': r'macd',
            'ma': r'ma_|moving_average|sma|ema',
            'rsi': r'rsi',
            'rsrs': r'rsrs',
            'ratio': r'ratio|pct_change',
            'correlation': r'corr|correlation',
        }
        
        for factor_name, pattern in factor_keywords.items():
            if re.search(pattern, content, re.IGNORECASE):
                factors.append(factor_name)
        
        # 检查数据源
        uses_open = 'open' in content
        uses_high = 'high' in content
        uses_low = 'low' in content
        uses_close = 'close' in content
        uses_volume = 'volume' in content
        uses_amount = 'amount' in content
        
        # 检查前视偏差
        lookahead_patterns = [
            r'\[t\+1\]',
            r'\[t\+2\]', 
            r'shift\s*\(\s*-',
            r'rolling.*shift.*-',
            r'future',
        ]
        lookahead_risk_count = sum(
            len(re.findall(p, content, re.IGNORECASE))
            for p in lookahead_patterns
        )
        
        # 检查复权处理
        has_adjustment = bool(re.search(r'adj|qfq|hfq|factor', content, re.IGNORECASE))
        
        # 计算复杂度
        lines = content.split('\n')
        code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
        complexity = len(code_lines)
        
        return {
            'name': strategy_name,
            'filepath': filepath,
            'main_function': main_func_name,
            'size_kb': len(content) / 1024,
            'lines': len(code_lines),
            'weights_assignments': weights_assignments,
            'buy_sell_logic': buy_logic + sell_logic,
            'factors_count': len(set(factors)),
            'factors': list(set(factors))[:5],
            'data_sources': {
                'open': uses_open,
                'high': uses_high,
                'low': uses_low,
                'close': uses_close,
                'volume': uses_volume,
                'amount': uses_amount,
            },
            'has_adjustment': has_adjustment,
            'lookahead_risk_count': lookahead_risk_count,
            'is_safe': lookahead_risk_count == 0,
        }
    except Exception as e:
        return {'name': extract_strategy_name(filepath), 'error': str(e)}

def main():
    print("\n" + "="*85)
    print("Q-UNITY V10 - 全13个策略代码审计")
    print("="*85 + "\n")
    
    print(f"[INFO] 审计 {len(STRATEGY_FILES)} 个策略...\n")
    
    results = []
    total_lines = 0
    total_factors = set()
    safe_count = 0
    
    for i, filepath in enumerate(STRATEGY_FILES, 1):
        print(f"[{i:2d}/13] 审计 {os.path.basename(filepath):40s} ... ", end='', flush=True)
        
        result = analyze_strategy_code(filepath)
        if result is None:
            print(f"✗ 文件不存在")
            continue
        
        if 'error' in result:
            print(f"✗ {result['error']}")
            continue
        
        results.append(result)
        total_lines += result['lines']
        total_factors.update(result['factors'])
        
        if result['is_safe']:
            safe_count += 1
            status = "✓"
        else:
            status = "⚠️ "
        
        print(f"{status} 行数:{result['lines']:4d} 因子:{result['factors_count']:2d} 权重赋值:{result['weights_assignments']:3d}")
    
    # 生成汇总报告
    print("\n" + "="*85)
    print("审计汇总")
    print("="*85 + "\n")
    
    print(f"\n代码规模统计:")
    print(f"  • 总策略数:       {len(results)}")
    if len(results) > 0:
        total_lines = sum(r['lines'] for r in results)
        print(f"  • 代码总行数:     {total_lines:,}")
        print(f"  • 平均行数/策略:  {total_lines//len(results)}")
    else:
        print(f"  • 代码总行数:     0")
        print(f"  • 平均行数/策略:  0")
    
    print(f"\n信号逻辑统计:")
    total_weights = sum(r['weights_assignments'] for r in results)
    total_logic = sum(r['buy_sell_logic'] for r in results)
    print(f"  • 权重赋值语句:   {total_weights}")
    print(f"  • 买卖逻辑引用:   {total_logic}")
    
    print(f"\n因子多样性:")
    print(f"  • 全局独特因子:   {len(total_factors)}")
    print(f"  • 平均因子/策略:  {sum(r['factors_count'] for r in results) / len(results):.1f}")
    
    print(f"\n数据源使用统计:")
    data_sources = {
        'open': sum(1 for r in results if r['data_sources']['open']),
        'high': sum(1 for r in results if r['data_sources']['high']),
        'low': sum(1 for r in results if r['data_sources']['low']),
        'close': sum(1 for r in results if r['data_sources']['close']),
        'volume': sum(1 for r in results if r['data_sources']['volume']),
        'amount': sum(1 for r in results if r['data_sources']['amount']),
    }
    for source, count in sorted(data_sources.items(), key=lambda x: -x[1]):
        print(f"  • {source:6s}: {count:2d} / {len(results)}")
    
    print(f"\n数据安全性:")
    print(f"  • 无前视偏差:     {safe_count} / {len(results)}")
    print(f"  • 复权处理:       {sum(1 for r in results if r['has_adjustment'])} / {len(results)}")
    
    # 详细列表
    print(f"\n" + "-"*85)
    print(f"{'#':>2} {'策略名称':<35} {'行数':>6} {'因子':>4} {'权重':>5} {'安全':>4}")
    print("-"*85)
    
    for i, result in enumerate(results, 1):
        name = result['name']
        lines = result['lines']
        factors = result['factors_count']
        weights = result['weights_assignments']
        safe = "✓" if result['is_safe'] else "⚠️"
        print(f"{i:2d} {name:<35} {lines:6d} {factors:4d} {weights:5d} {safe:>4}")
    
    # 因子分布
    print(f"\n全局因子使用TOP-10:")
    factor_counts = {}
    for result in results:
        for factor in result['factors']:
            factor_counts[factor] = factor_counts.get(factor, 0) + 1
    
    for factor, count in sorted(factor_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {count:2d}. {factor:<20} ({count} 个策略)")
    
    # 保存文本报告
    output_dir = 'whitebox_audit/output'
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, '13_strategies_final_audit.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*85 + "\n")
        f.write("Q-UNITY V10 - 全13个策略代码审计最终报告\n")
        f.write("="*85 + "\n\n")
        
        f.write(f"代码规模: {total_lines:,} 行\n")
        f.write(f"平均策略规模: {total_lines//len(results) if results else 0} 行\n")
        f.write(f"全局独特因子: {len(total_factors)}\n")
        f.write(f"前视偏差安全: {safe_count}/{len(results)}\n\n")
        
        f.write("策略详情:\n")
        for result in results:
            f.write(f"\n{result['name']}\n")
            f.write(f"  行数: {result['lines']}\n")
            f.write(f"  因子数: {result['factors_count']}\n")
            f.write(f"  因子: {', '.join(result['factors'])}\n")
            f.write(f"  安全: {'是' if result['is_safe'] else '否'}\n")
    
    print(f"\n✓ 报告已保存: {report_path}\n")
    
    print("="*85)
    print("审计完成!")
    print("="*85 + "\n")

if __name__ == '__main__':
    main()
