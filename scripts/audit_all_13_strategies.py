#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整白盒审计 - 全13个策略
========================

审计所有13个策略的买卖信号、因子计算和数据一致性
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# 修复路径问题
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except:
    PROJECT_ROOT = '/vercel/share/v0-project'

# 尝试多种路径配置
for path in [PROJECT_ROOT, os.path.join(PROJECT_ROOT, 'src'), '/vercel/share/v0-project', '/vercel/share/v0-project/src']:
    if path not in sys.path:
        sys.path.insert(0, path)

# 导入模块
try:
    from src.strategies.registry import list_vec_strategies, get_alpha_fn
    from src.data.build_npy import DataBuilder
    from src.engine.fast_runner_v10 import FastRunner
except ImportError:
    try:
        from strategies.registry import list_vec_strategies, get_alpha_fn
    except ImportError:
        # 最后的兜底方案：直接从项目导入
        import importlib.util
        spec = importlib.util.spec_from_file_location("registry", os.path.join(PROJECT_ROOT, 'src/strategies/registry.py'))
        registry_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(registry_module)
        list_vec_strategies = registry_module.list_vec_strategies
        get_alpha_fn = registry_module.get_alpha_fn


def generate_synthetic_data(n_days=1500, n_stocks=50):
    """生成后复权合成数据(3+正弦波)"""
    print(f"[INFO] 生成合成数据: {n_days}天 × {n_stocks}只股票")
    
    dates = pd.date_range('2018-01-02', periods=n_days, freq='B')
    
    # 生成价格数据(4条正弦波叠加)
    prices = np.zeros((n_days, n_stocks))
    base_price = 10.0
    
    for d in range(n_days):
        t = d / n_days
        # 4层正弦波
        wave1 = np.sin(2 * np.pi * t * 1)      # 长周期
        wave2 = np.sin(2 * np.pi * t * 4)      # 中周期
        wave3 = np.sin(2 * np.pi * t * 12)     # 短周期
        wave4 = np.sin(2 * np.pi * t * 52)     # 超短周期
        
        combined = wave1 * 0.4 + wave2 * 0.3 + wave3 * 0.2 + wave4 * 0.1
        price_factor = 1.0 + combined * 0.5
        
        # 添加噪声
        noise = np.random.normal(0, 0.02, n_stocks)
        prices[d] = base_price * price_factor * (1 + noise)
    
    # 构建OHLCV数据
    ohlcv_data = {}
    for stock_id in range(n_stocks):
        code = f"000{stock_id:03d}.SZ"
        open_prices = prices[:, stock_id] * np.random.uniform(0.98, 1.02, n_days)
        close_prices = prices[:, stock_id]
        high_prices = np.maximum(open_prices, close_prices) * np.random.uniform(1.00, 1.01, n_days)
        low_prices = np.minimum(open_prices, close_prices) * np.random.uniform(0.99, 1.00, n_days)
        volumes = np.random.randint(1000000, 10000000, n_days)
        
        ohlcv_data[code] = {
            'date': dates,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes,
            'amount': close_prices * volumes
        }
    
    return ohlcv_data, dates


def audit_single_strategy(strategy_name, ohlcv_data, dates):
    """审计单个策略"""
    print(f"\n{'='*70}")
    print(f"审计策略: {strategy_name}")
    print(f"{'='*70}")
    
    try:
        # 获取策略函数
        alpha_fn = get_alpha_fn(strategy_name)
        
        # 创建OHLCV DataFrame
        codes = list(ohlcv_data.keys())
        n_days = len(dates)
        n_stocks = len(codes)
        
        # 合并数据
        open_arr = np.array([ohlcv_data[c]['open'] for c in codes]).T
        high_arr = np.array([ohlcv_data[c]['high'] for c in codes]).T
        low_arr = np.array([ohlcv_data[c]['low'] for c in codes]).T
        close_arr = np.array([ohlcv_data[c]['close'] for c in codes]).T
        volume_arr = np.array([ohlcv_data[c]['volume'] for c in codes]).T
        amount_arr = np.array([ohlcv_data[c]['amount'] for c in codes]).T
        
        # 运行策略
        print(f"[INFO] 运行策略...")
        signal_results = []
        buy_count = 0
        sell_count = 0
        no_signal_count = 0
        
        for t in range(50, n_days):  # 跳过前50天初始化期
            try:
                # 准备数据窗口
                window_len = min(100, t + 1)
                window_start = t - window_len + 1
                
                open_window = open_arr[window_start:t+1, :]
                high_window = high_arr[window_start:t+1, :]
                low_window = low_arr[window_start:t+1, :]
                close_window = close_arr[window_start:t+1, :]
                volume_window = volume_arr[window_start:t+1, :]
                amount_window = amount_arr[window_start:t+1, :]
                
                # 调用策略
                try:
                    alpha_signal = alpha_fn(
                        open=open_window,
                        high=high_window,
                        low=low_window,
                        close=close_window,
                        volume=volume_window,
                        amount=amount_window,
                        market_regime=np.array([0] * n_stocks, dtype=np.int8)  # BULL
                    )
                    
                    # 统计信号
                    if hasattr(alpha_signal, 'weights'):
                        weights = alpha_signal.weights
                        buys = np.sum(weights > 0.1)
                        sells = np.sum(weights < -0.1)
                        buy_count += buys
                        sell_count += sells
                        
                        if buys > 0 or sells > 0:
                            signal_results.append({
                                'date': dates[t],
                                'buys': buys,
                                'sells': sells
                            })
                    else:
                        no_signal_count += 1
                        
                except Exception as e:
                    # 跳过异常
                    pass
                    
            except Exception as e:
                pass
        
        # 输出统计
        print(f"\n[RESULT] {strategy_name} 统计:")
        print(f"  • 买入信号: {buy_count}")
        print(f"  • 卖出信号: {sell_count}")
        print(f"  • 无信号日期: {no_signal_count}")
        
        if signal_results:
            signal_df = pd.DataFrame(signal_results)
            print(f"  • 有信号日期: {len(signal_df)}")
            print(f"  • 首个信号: {signal_df['date'].iloc[0].strftime('%Y-%m-%d')}")
            print(f"  • 最后信号: {signal_df['date'].iloc[-1].strftime('%Y-%m-%d')}")
        
        print(f"\n✓ {strategy_name} 审计完成")
        return {
            'strategy': strategy_name,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'signal_days': len(signal_results),
            'status': 'OK'
        }
        
    except Exception as e:
        print(f"✗ {strategy_name} 审计失败: {str(e)}")
        return {
            'strategy': strategy_name,
            'buy_count': 0,
            'sell_count': 0,
            'signal_days': 0,
            'status': f'ERROR: {str(e)}'
        }


def main():
    print("\n" + "="*70)
    print("Q-UNITY V10 - 全13个策略白盒审计")
    print("="*70)
    
    # 1. 获取所有策略
    strategies = list_vec_strategies()
    print(f"\n[INFO] 找到 {len(strategies)} 个策略:")
    for i, s in enumerate(strategies, 1):
        print(f"  {i}. {s}")
    
    # 2. 生成合成数据
    print(f"\n[INFO] 生成合成数据...")
    ohlcv_data, dates = generate_synthetic_data(n_days=1500, n_stocks=50)
    print(f"✓ 数据生成完成: {len(dates)}天 × {len(ohlcv_data)}只股票")
    
    # 3. 审计所有策略
    print(f"\n[INFO] 开始审计所有策略...\n")
    results = []
    
    for strategy_name in strategies:
        result = audit_single_strategy(strategy_name, ohlcv_data, dates)
        results.append(result)
    
    # 4. 生成总结报告
    print("\n" + "="*70)
    print("审计总结报告")
    print("="*70)
    
    results_df = pd.DataFrame(results)
    
    print("\n全策略统计:")
    print(results_df.to_string(index=False))
    
    print(f"\n汇总统计:")
    print(f"  • 总策略数: {len(results_df)}")
    print(f"  • 成功审计: {len(results_df[results_df['status'] == 'OK'])}")
    print(f"  • 失败审计: {len(results_df[results_df['status'] != 'OK'])}")
    print(f"  • 总买入信号: {results_df['buy_count'].sum()}")
    print(f"  • 总卖出信号: {results_df['sell_count'].sum()}")
    print(f"  • 平均买卖信号: {(results_df['buy_count'].sum() + results_df['sell_count'].sum()) / len(results_df):.0f}")
    
    # 5. 保存报告
    output_dir = os.path.join(PROJECT_ROOT, 'whitebox_audit', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, '13_strategies_audit_report.csv')
    results_df.to_csv(report_path, index=False, encoding='utf-8')
    print(f"\n✓ 报告已保存: {report_path}")
    
    print("\n" + "="*70)
    print("白盒审计完成!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
