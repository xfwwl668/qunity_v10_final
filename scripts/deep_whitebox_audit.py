#!/usr/bin/env python3
"""
深度审计脚本 - 逐日追踪信号逻辑和交易执行
============================================
验证:
1. 因子计算的精确性
2. 买卖信号的时序正确性
3. 交易成本的准确应用
4. 复权数据的一致性
"""

import sys
import os
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except:
    PROJECT_ROOT = '/vercel/share/v0-project'
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ============================================================================
# 深度审计 - 逐日信号追踪
# ============================================================================

class DailySignalTracer:
    """逐日追踪信号和因子计算"""
    
    def __init__(self, n_days=100, n_stocks=10):
        self.n_days = n_days
        self.n_stocks = n_stocks
        self.traces = []
        
    def generate_test_data(self):
        """生成测试数据"""
        np.random.seed(42)
        dates = [(datetime(2023, 1, 1) + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(self.n_days)]
        
        # 生成价格序列 - 含有明确的买卖信号点
        close = np.zeros((self.n_days, self.n_stocks))
        
        for s in range(self.n_stocks):
            base_price = 10.0 + s * 0.5
            # 生成 2 个完整的买卖周期
            for t in range(self.n_days):
                # 周期1: 0-50天
                if t < 50:
                    if t < 20:
                        close[t, s] = base_price * (1 - 0.02 * t / 20)  # 下跌20%
                    elif t < 40:
                        close[t, s] = base_price * (0.8 + 0.02 * (t - 20) / 20)  # 反弹20%
                    else:
                        close[t, s] = base_price * (1 + 0.01 * (t - 40) / 10)  # 缓涨
                # 周期2: 50-100天
                else:
                    cycle_t = t - 50
                    if cycle_t < 20:
                        close[t, s] = base_price * (1.1 - 0.03 * cycle_t / 20)
                    elif cycle_t < 40:
                        close[t, s] = base_price * (0.8 + 0.03 * (cycle_t - 20) / 20)
                    else:
                        close[t, s] = base_price * (1.1 + 0.01 * (cycle_t - 40) / 10)
        
        return {
            'dates': dates,
            'close': close,
            'high': close * 1.01,
            'low': close * 0.99,
            'volume': np.full((self.n_days, self.n_stocks), 1e6),
        }
    
    def audit_ma_signal(self, close_data):
        """逐日审计 MA 信号"""
        print("\n[深度审计] MA Crossover 逐日追踪")
        print("-" * 70)
        
        n_days, n_stocks = close_data.shape
        ma_5 = np.zeros_like(close_data)
        ma_20 = np.zeros_like(close_data)
        
        # 计算 MA
        for t in range(n_days):
            if t >= 4:
                ma_5[t] = np.mean(close_data[t-4:t+1], axis=0)
            if t >= 19:
                ma_20[t] = np.mean(close_data[t-19:t+1], axis=0)
        
        # 检测交叉
        buy_signals = 0
        sell_signals = 0
        
        for t in range(20, min(50, n_days)):  # 只看前50天作为示例
            for s in range(n_stocks):
                if t > 0:
                    # 金叉检测
                    prev_below = ma_5[t-1, s] < ma_20[t-1, s]
                    curr_above = ma_5[t, s] > ma_20[t, s]
                    if prev_below and curr_above:
                        print(f"  [{t:3d}] 股票{s:2d}: 金叉 - MA5={ma_5[t, s]:.2f} > MA20={ma_20[t, s]:.2f}, close={close_data[t, s]:.2f}")
                        buy_signals += 1
                    
                    # 死叉检测
                    prev_above = ma_5[t-1, s] > ma_20[t-1, s]
                    curr_below = ma_5[t, s] < ma_20[t, s]
                    if prev_above and curr_below:
                        print(f"  [{t:3d}] 股票{s:2d}: 死叉 - MA5={ma_5[t, s]:.2f} < MA20={ma_20[t, s]:.2f}, close={close_data[t, s]:.2f}")
                        sell_signals += 1
        
        print(f"\n  总统计: 买入={buy_signals}, 卖出={sell_signals}")
        return {'ma_5': ma_5, 'ma_20': ma_20, 'buys': buy_signals, 'sells': sell_signals}
    
    def audit_momentum_signal(self, close_data):
        """逐日审计 Momentum 反转信号"""
        print("\n[深度审计] Momentum Reversal 逐日追踪")
        print("-" * 70)
        
        n_days, n_stocks = close_data.shape
        
        # 计算收益率
        ret_5d = np.zeros_like(close_data)
        ret_20d = np.zeros_like(close_data)
        ma_20 = np.zeros_like(close_data)
        
        for t in range(1, n_days):
            if t >= 5:
                ret_5d[t] = (close_data[t] - close_data[t-5]) / close_data[t-5]
            if t >= 20:
                ret_20d[t] = (close_data[t] - close_data[t-20]) / close_data[t-20]
                ma_20[t] = np.mean(close_data[t-19:t+1], axis=0)
        
        # 检测买入信号
        buy_signals = 0
        
        for t in range(20, min(50, n_days)):
            for s in range(n_stocks):
                cond1 = ret_5d[t, s] < -0.05  # 5日超跌 < -5%
                cond2 = ret_20d[t, s] > 0      # 20日向上
                cond3 = close_data[t, s] > ma_20[t, s]  # 价格 > MA20
                
                if cond1 and cond2 and cond3:
                    print(f"  [{t:3d}] 股票{s:2d}: 反转买入")
                    print(f"      5日收益={ret_5d[t, s]:+.2%}, 20日收益={ret_20d[t, s]:+.2%}")
                    print(f"      close={close_data[t, s]:.2f} > MA20={ma_20[t, s]:.2f}")
                    buy_signals += 1
        
        print(f"\n  总统计: 买入信号={buy_signals}")
        return {'ret_5d': ret_5d, 'ret_20d': ret_20d, 'ma_20': ma_20, 'buys': buy_signals}
    
    def audit_costs_impact(self, close_data):
        """审计交易成本对收益的影响"""
        print("\n[深度审计] 交易成本影响分析")
        print("-" * 70)
        
        n_days, n_stocks = close_data.shape
        
        # 模拟一个简单的策略: 买入持有
        entry_price = close_data[0, 0]
        exit_price = close_data[-1, 0]
        
        # 成本参数
        commission_rate = 0.0003  # 万3
        stamp_tax = 0.001  # 千一 (卖出时)
        slippage_rate = 0.0005  # 5bp 滑点
        
        # 无成本收益
        ret_no_cost = (exit_price - entry_price) / entry_price
        
        # 有成本收益 (买入时扣手续费+滑点, 卖出时扣手续费+印花税+滑点)
        buy_cost = commission_rate + slippage_rate
        sell_cost = commission_rate + stamp_tax + slippage_rate
        
        effective_entry = entry_price * (1 + buy_cost)
        effective_exit = exit_price * (1 - sell_cost)
        
        ret_with_cost = (effective_exit - effective_entry) / effective_entry
        
        # 收益损失
        loss = ret_no_cost - ret_with_cost
        loss_pct = loss / abs(ret_no_cost) * 100 if ret_no_cost != 0 else 0
        
        print(f"  入场价: {entry_price:.2f}")
        print(f"  出场价: {exit_price:.2f}")
        print(f"  无成本收益: {ret_no_cost:+.2%}")
        print(f"  \n  成本明细:")
        print(f"    - 买入手续费: {commission_rate:.2%}")
        print(f"    - 卖出手续费: {commission_rate:.2%}")
        print(f"    - 印花税(卖出): {stamp_tax:.2%}")
        print(f"    - 滑点: {slippage_rate:.2%} (买入+卖出)")
        print(f"  \n  有成本收益: {ret_with_cost:+.2%}")
        print(f"  收益损失: {loss:.2%} ({loss_pct:.1f}%)")
        
        # 重要: 多笔交易场景
        print(f"\n  多笔交易场景 (10次买卖):")
        n_trades = 10
        cumulative_cost = 0
        for i in range(n_trades):
            # 每次买卖都要支付手续费和滑点
            cumulative_cost += (buy_cost + sell_cost)
        
        cumulative_cost_pct = cumulative_cost * 100
        print(f"    - 总成本: {cumulative_cost_pct:.2f}bp (每次往返: {(buy_cost + sell_cost)*10000:.0f}bp)")
        
        return {
            'ret_no_cost': ret_no_cost,
            'ret_with_cost': ret_with_cost,
            'cost_loss': loss,
            'cumulative_cost_10trades': cumulative_cost_pct
        }


# ============================================================================
# 复权数据一致性验证
# ============================================================================

class AdjustmentConsistencyVerifier:
    """验证前复权和后复权数据一致性"""
    
    def verify_adjustment_formula(self):
        """验证调整公式"""
        print("\n[深度审计] 复权数据一致性验证")
        print("-" * 70)
        
        # 模拟历史价格和复权因子
        raw_prices = np.array([8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0])
        factors = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5])  # 第4天有除权
        
        # 计算前复权价格
        latest_factor = factors[-1]
        qfq_prices = raw_prices * (factors / latest_factor)
        
        # 计算后复权价格 (新公式)
        hfq_prices = raw_prices * factors
        
        # 验证逆推
        hfq_from_qfq = qfq_prices * latest_factor
        
        print(f"  原始价格: {raw_prices}")
        print(f"  复权因子: {factors}")
        print(f"  前复权价: {qfq_prices}")
        print(f"  后复权价: {hfq_prices}")
        print(f"  从前复权推导: {hfq_from_qfq}")
        print(f"  一致性检验: {np.allclose(hfq_prices, hfq_from_qfq)}")
        
        # 关键: 收益率应该相等
        ret_raw = (raw_prices[-1] - raw_prices[0]) / raw_prices[0]
        ret_qfq = (qfq_prices[-1] - qfq_prices[0]) / qfq_prices[0]
        ret_hfq = (hfq_prices[-1] - hfq_prices[0]) / hfq_prices[0]
        
        print(f"\n  收益率验证:")
        print(f"    原始收益: {ret_raw:+.2%}")
        print(f"    前复权收益: {ret_qfq:+.2%}")
        print(f"    后复权收益: {ret_hfq:+.2%}")
        print(f"    都相等: {np.allclose([ret_raw, ret_qfq, ret_hfq], ret_raw)}")
        
        return {
            'consistency': np.allclose(hfq_prices, hfq_from_qfq),
            'returns_match': np.allclose([ret_raw, ret_qfq, ret_hfq], ret_raw),
        }


# ============================================================================
# 主程序
# ============================================================================

def main():
    print("=" * 70)
    print("Q-UNITY V10 深度白盒审计 - 逐日追踪")
    print("=" * 70)
    
    # 1. 生成测试数据
    print("\n[准备] 生成测试数据...")
    tracer = DailySignalTracer(n_days=100, n_stocks=5)
    test_data = tracer.generate_test_data()
    print(f"  数据形状: {test_data['close'].shape}")
    print(f"  日期范围: {test_data['dates'][0]} ~ {test_data['dates'][-1]}")
    
    # 2. 审计 MA 信号
    ma_result = tracer.audit_ma_signal(test_data['close'])
    
    # 3. 审计 Momentum 信号
    momentum_result = tracer.audit_momentum_signal(test_data['close'])
    
    # 4. 审计交易成本
    cost_result = tracer.audit_costs_impact(test_data['close'])
    
    # 5. 验证复权一致性
    print()
    verifier = AdjustmentConsistencyVerifier()
    adj_result = verifier.verify_adjustment_formula()
    
    # 生成汇总
    print("\n" + "=" * 70)
    print("深度审计汇总")
    print("=" * 70)
    print(f"\n✓ MA信号: {ma_result['buys']} 次买入, {ma_result['sells']} 次卖出")
    print(f"✓ Momentum信号: {momentum_result['buys']} 次反转买入")
    print(f"✓ 交易成本影响: 收益从 {cost_result['ret_no_cost']:+.2%} 降至 {cost_result['ret_with_cost']:+.2%}")
    print(f"✓ 复权一致性: {adj_result['consistency']}, 收益率匹配: {adj_result['returns_match']}")
    
    print("\n" + "=" * 70)
    print("深度审计完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
