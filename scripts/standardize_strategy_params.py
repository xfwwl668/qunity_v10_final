#!/usr/bin/env python3
"""
策略参数标准化工具 - 修复13个策略的不一致参数

当前问题：
  • 3个策略无止盈（alpha_hunter_v2, alpha_max_v5, titan_alpha_v1）
  • 止损参数跨度大（4.5% ~ 22%）
  • 1个策略缺少参数（sniper_v6a）

标准化方案：
  • 按策略风格分组：事件驱动/快速反转 / 中期趋势 / 长期持有
  • 每组统一止损/止盈参数
"""

import os
import re
from pathlib import Path
from typing import Dict, Tuple

# 策略分类与推荐参数
STRATEGY_CLASSES = {
    # 事件驱动/短期反转（持仓3-7天）
    "ultra_short": {
        "hard_stop_loss": 0.045,
        "take_profit": 0.07,
        "strategies": ["retail_sniper_v10", "sentiment_reversal"],
        "note": "极短期反转，追求快进快出"
    },
    
    # 短期反转（持仓7-10天）
    "short_term": {
        "hard_stop_loss": 0.08,
        "take_profit": 0.12,
        "strategies": ["short_term_rsrs", "weak_to_strong"],
        "note": "短期反转信号，风险适中"
    },
    
    # 中期趋势（持仓20-30天）
    "medium_term": {
        "hard_stop_loss": 0.12,
        "take_profit": 0.18,
        "strategies": ["kunpeng_v10", "titan_orthogonal_v10", "momentum_reversal", 
                      "snma_v4", "alpha_hunter_v2", "ultra_alpha_v1"],
        "note": "中期趋势跟踪，止损止盈均衡"
    },
    
    # 长期持有（持仓30+天）
    "long_term": {
        "hard_stop_loss": 0.15,
        "take_profit": 0.25,
        "strategies": ["ultra_alpha_v1", "alpha_max_v5", "titan_alpha_v1"],
        "note": "长期趋势，容错率提高"
    }
}

# 逆向查询：策略 -> 应该的参数
STRATEGY_PARAMS: Dict[str, Tuple[float, float]] = {}
for cls_name, cls_info in STRATEGY_CLASSES.items():
    for strat in cls_info["strategies"]:
        STRATEGY_PARAMS[strat] = (cls_info["hard_stop_loss"], cls_info["take_profit"])

# 策略文件路径
PROJECT_ROOT = '/vercel/share/v0-project'
STRATEGIES_DIR = os.path.join(PROJECT_ROOT, 'src/strategies/vectorized')

def find_and_replace_params(strategy_name: str, hard_stop: float, take_profit: float) -> int:
    """
    在策略文件中找到并替换 hard_stop_loss 和 take_profit 参数
    返回修改行数
    """
    # 查找策略文件
    strategy_file = None
    for f in Path(STRATEGIES_DIR).glob(f'*{strategy_name}*.py'):
        strategy_file = str(f)
        break
    
    if not strategy_file or not os.path.exists(strategy_file):
        print(f"  ❌ 找不到策略文件: {strategy_name}")
        return 0
    
    try:
        with open(strategy_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"  ❌ 无法读取文件: {e}")
        return 0
    
    # 记录修改前的值
    old_stop = re.search(r'"hard_stop_loss"\s*:\s*([\d.]+)', content)
    old_profit = re.search(r'"take_profit"\s*:\s*([\d.]+)', content)
    
    # 替换参数
    modified = 0
    if old_stop:
        old_val = float(old_stop.group(1))
        if abs(old_val - hard_stop) > 1e-6:
            content = re.sub(
                r'"hard_stop_loss"\s*:\s*[\d.]+',
                f'"hard_stop_loss"  : {hard_stop}',
                content
            )
            print(f"  ✓ hard_stop_loss: {old_val} → {hard_stop}")
            modified += 1
    
    if old_profit:
        old_val = float(old_profit.group(1))
        if abs(old_val - take_profit) > 1e-6:
            content = re.sub(
                r'"take_profit"\s*:\s*[\d.]+',
                f'"take_profit"     : {take_profit}',
                content
            )
            print(f"  ✓ take_profit: {old_val} → {take_profit}")
            modified += 1
    
    # 写回文件
    if modified > 0:
        try:
            with open(strategy_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✅ {os.path.basename(strategy_file)} 已更新")
        except Exception as e:
            print(f"  ❌ 写入失败: {e}")
            return 0
    
    return modified

def main():
    print("=" * 70)
    print("Q-UNITY V10 策略参数标准化")
    print("=" * 70)
    
    total_modifications = 0
    
    for strategy_name, (hard_stop, take_profit) in STRATEGY_PARAMS.items():
        print(f"\n📝 {strategy_name}")
        modified = find_and_replace_params(strategy_name, hard_stop, take_profit)
        total_modifications += modified
    
    print("\n" + "=" * 70)
    print(f"✅ 总计修改了 {total_modifications} 个参数")
    print("=" * 70)
    
    # 打印标准化方案
    print("\n📋 标准化方案总结:\n")
    for cls_name, cls_info in STRATEGY_CLASSES.items():
        print(f"  {cls_name.upper()}")
        print(f"    hard_stop_loss: {cls_info['hard_stop_loss']}")
        print(f"    take_profit: {cls_info['take_profit']}")
        print(f"    策略: {', '.join(cls_info['strategies'])}")
        print()

if __name__ == '__main__':
    main()
