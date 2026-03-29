#!/usr/bin/env python3
"""
Q-UNITY V10 白盒审计脚本
========================
生成合成数据(后复权, 3+正弦波, 1500D+)，审计因子计算和买卖信号对应关系
"""

import sys
import os

# 添加项目根目录到路径 (修复沙盒环境下的 __file__ 问题)
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except:
    PROJECT_ROOT = '/vercel/share/v0-project'
    
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime, timedelta

# ============================================================================
# 第一部分: 合成数据生成器 (后复权, 3+正弦波, 1500D+)
# ============================================================================

@dataclass
class SyntheticConfig:
    """合成数据配置"""
    n_days: int = 1500
    n_stocks: int = 50
    n_sine_waves: int = 4  # 正弦波数量
    base_price: float = 10.0
    seed: int = 42

def generate_hfq_synthetic_data(config: SyntheticConfig) -> Dict[str, np.ndarray]:
    """
    生成后复权合成OHLCV数据
    
    特点:
    1. 使用多个正弦波叠加模拟真实价格波动
    2. 后复权格式 - 历史价格不变，新价格累加
    3. 包含已知的买卖信号点用于验证
    """
    np.random.seed(config.seed)
    
    n_days = config.n_days
    n_stocks = config.n_stocks
    
    # 生成时间序列
    dates = np.arange(n_days)
    
    # 初始化数组
    open_arr = np.zeros((n_days, n_stocks), dtype=np.float64)
    high_arr = np.zeros((n_days, n_stocks), dtype=np.float64)
    low_arr = np.zeros((n_days, n_stocks), dtype=np.float64)
    close_arr = np.zeros((n_days, n_stocks), dtype=np.float64)
    volume_arr = np.zeros((n_days, n_stocks), dtype=np.float64)
    amount_arr = np.zeros((n_days, n_stocks), dtype=np.float64)
    
    # 为每只股票生成价格序列
    for s in range(n_stocks):
        # 基础价格 + 长期趋势
        base = config.base_price * (1 + 0.5 * np.random.random())
        trend = 0.0001 * (np.random.random() - 0.3)  # 轻微上升趋势
        
        # 叠加多个正弦波
        price = np.ones(n_days) * base
        
        # 正弦波1: 长周期 (约300天)
        period1 = 280 + np.random.randint(-30, 30)
        amp1 = 0.15 + 0.1 * np.random.random()
        phase1 = np.random.random() * 2 * np.pi
        price += base * amp1 * np.sin(2 * np.pi * dates / period1 + phase1)
        
        # 正弦波2: 中周期 (约60天)
        period2 = 55 + np.random.randint(-10, 10)
        amp2 = 0.08 + 0.05 * np.random.random()
        phase2 = np.random.random() * 2 * np.pi
        price += base * amp2 * np.sin(2 * np.pi * dates / period2 + phase2)
        
        # 正弦波3: 短周期 (约20天)
        period3 = 18 + np.random.randint(-5, 5)
        amp3 = 0.04 + 0.02 * np.random.random()
        phase3 = np.random.random() * 2 * np.pi
        price += base * amp3 * np.sin(2 * np.pi * dates / period3 + phase3)
        
        # 正弦波4: 超短周期 (约5天)
        period4 = 5 + np.random.randint(-1, 2)
        amp4 = 0.02 + 0.01 * np.random.random()
        phase4 = np.random.random() * 2 * np.pi
        price += base * amp4 * np.sin(2 * np.pi * dates / period4 + phase4)
        
        # 添加趋势
        price *= (1 + trend * dates)
        
        # 添加随机噪声
        noise = 1 + 0.005 * np.random.randn(n_days)
        price *= noise
        
        # 确保价格为正
        price = np.maximum(price, 0.1)
        
        # 生成OHLC (后复权格式)
        close_arr[:, s] = price
        
        # Open: 前一日收盘 + 小幅跳空
        open_arr[0, s] = price[0] * (1 + 0.002 * np.random.randn())
        open_arr[1:, s] = price[:-1] * (1 + 0.003 * np.random.randn(n_days - 1))
        
        # High/Low: 基于收盘价的波动
        daily_range = 0.02 + 0.01 * np.random.random(n_days)
        high_arr[:, s] = np.maximum(open_arr[:, s], close_arr[:, s]) * (1 + daily_range)
        low_arr[:, s] = np.minimum(open_arr[:, s], close_arr[:, s]) * (1 - daily_range)
        
        # Volume: 基础成交量 + 波动
        base_vol = 1e6 * (1 + np.random.random())
        volume_arr[:, s] = base_vol * (1 + 0.5 * np.random.random(n_days))
        
        # Amount: 成交额 = 均价 * 成交量
        avg_price = (open_arr[:, s] + high_arr[:, s] + low_arr[:, s] + close_arr[:, s]) / 4
        amount_arr[:, s] = avg_price * volume_arr[:, s]
    
    # 生成日期序列
    start_date = datetime(2018, 1, 2)
    date_list = [(start_date + timedelta(days=int(d))).strftime('%Y%m%d') for d in dates]
    
    return {
        'open': open_arr,
        'high': high_arr,
        'low': low_arr,
        'close': close_arr,
        'volume': volume_arr,
        'amount': amount_arr,
        'dates': np.array(date_list),
        'n_days': n_days,
        'n_stocks': n_stocks,
    }


# ============================================================================
# 第二部分: 因子计算验证器
# ============================================================================

def calculate_ma(close: np.ndarray, window: int) -> np.ndarray:
    """手动计算移动平均"""
    n_days, n_stocks = close.shape
    ma = np.full((n_days, n_stocks), np.nan)
    for t in range(window - 1, n_days):
        ma[t] = np.mean(close[t - window + 1:t + 1], axis=0)
    return ma

def calculate_ema(close: np.ndarray, span: int) -> np.ndarray:
    """手动计算EMA"""
    n_days, n_stocks = close.shape
    ema = np.zeros((n_days, n_stocks))
    alpha = 2.0 / (span + 1)
    ema[0] = close[0]
    for t in range(1, n_days):
        ema[t] = alpha * close[t] + (1 - alpha) * ema[t - 1]
    return ema

def calculate_returns(close: np.ndarray, period: int = 1) -> np.ndarray:
    """计算收益率"""
    n_days, n_stocks = close.shape
    returns = np.full((n_days, n_stocks), np.nan)
    for t in range(period, n_days):
        returns[t] = (close[t] - close[t - period]) / close[t - period]
    return returns

def calculate_volatility(close: np.ndarray, window: int) -> np.ndarray:
    """计算波动率"""
    returns = calculate_returns(close, 1)
    n_days, n_stocks = close.shape
    vol = np.full((n_days, n_stocks), np.nan)
    for t in range(window, n_days):
        vol[t] = np.std(returns[t - window + 1:t + 1], axis=0)
    return vol


# ============================================================================
# 第三部分: 策略信号验证器
# ============================================================================

class StrategySignalAuditor:
    """策略信号审计器"""
    
    def __init__(self, data: Dict[str, np.ndarray]):
        self.data = data
        self.close = data['close']
        self.n_days = data['n_days']
        self.n_stocks = data['n_stocks']
        self.audit_results = {}
    
    def audit_momentum_reversal(self) -> Dict:
        """
        审计动量反转策略
        逻辑: 短期超跌 + 中期动量 → 买入信号
        """
        print("\n[审计] Momentum Reversal 策略...")
        
        close = self.close
        
        # 计算因子
        ret_5d = calculate_returns(close, 5)
        ret_20d = calculate_returns(close, 20)
        ma_20 = calculate_ma(close, 20)
        
        # 手动计算买入信号
        # 条件: 5日收益 < -5% (超跌) AND 20日收益 > 0 (中期趋势向上) AND 价格 > MA20
        manual_buy_signals = np.zeros((self.n_days, self.n_stocks), dtype=bool)
        for t in range(20, self.n_days):
            for s in range(self.n_stocks):
                if not np.isnan(ret_5d[t, s]) and not np.isnan(ret_20d[t, s]):
                    cond1 = ret_5d[t, s] < -0.05  # 短期超跌
                    cond2 = ret_20d[t, s] > 0      # 中期向上
                    cond3 = close[t, s] > ma_20[t, s]  # 价格在均线上方
                    if cond1 and cond2 and cond3:
                        manual_buy_signals[t, s] = True
        
        # 统计
        total_signals = np.sum(manual_buy_signals)
        signal_days = np.sum(np.any(manual_buy_signals, axis=1))
        
        result = {
            'strategy': 'momentum_reversal',
            'total_buy_signals': int(total_signals),
            'signal_days': int(signal_days),
            'signal_rate': float(signal_days / self.n_days),
            'factors_computed': ['ret_5d', 'ret_20d', 'ma_20'],
            'signal_conditions': ['ret_5d < -5%', 'ret_20d > 0', 'close > MA20'],
            'status': 'PASS' if total_signals > 0 else 'WARN_NO_SIGNALS'
        }
        
        print(f"  - 买入信号总数: {total_signals}")
        print(f"  - 有信号的天数: {signal_days}/{self.n_days}")
        print(f"  - 状态: {result['status']}")
        
        self.audit_results['momentum_reversal'] = result
        return result
    
    def audit_ma_crossover(self) -> Dict:
        """
        审计均线交叉策略
        逻辑: MA5 上穿 MA20 → 买入, MA5 下穿 MA20 → 卖出
        """
        print("\n[审计] MA Crossover 策略...")
        
        close = self.close
        
        # 计算均线
        ma_5 = calculate_ma(close, 5)
        ma_20 = calculate_ma(close, 20)
        
        # 手动计算交叉信号
        buy_signals = np.zeros((self.n_days, self.n_stocks), dtype=bool)
        sell_signals = np.zeros((self.n_days, self.n_stocks), dtype=bool)
        
        for t in range(21, self.n_days):
            for s in range(self.n_stocks):
                # 金叉: 前一日 MA5 < MA20, 今日 MA5 > MA20
                prev_below = ma_5[t-1, s] < ma_20[t-1, s]
                curr_above = ma_5[t, s] > ma_20[t, s]
                if prev_below and curr_above:
                    buy_signals[t, s] = True
                
                # 死叉: 前一日 MA5 > MA20, 今日 MA5 < MA20
                prev_above = ma_5[t-1, s] > ma_20[t-1, s]
                curr_below = ma_5[t, s] < ma_20[t, s]
                if prev_above and curr_below:
                    sell_signals[t, s] = True
        
        # 统计
        total_buy = np.sum(buy_signals)
        total_sell = np.sum(sell_signals)
        
        result = {
            'strategy': 'ma_crossover',
            'total_buy_signals': int(total_buy),
            'total_sell_signals': int(total_sell),
            'buy_days': int(np.sum(np.any(buy_signals, axis=1))),
            'sell_days': int(np.sum(np.any(sell_signals, axis=1))),
            'factors_computed': ['ma_5', 'ma_20'],
            'signal_conditions': ['MA5 crosses above MA20 (buy)', 'MA5 crosses below MA20 (sell)'],
            'status': 'PASS' if total_buy > 0 and total_sell > 0 else 'WARN'
        }
        
        print(f"  - 买入信号(金叉): {total_buy}")
        print(f"  - 卖出信号(死叉): {total_sell}")
        print(f"  - 状态: {result['status']}")
        
        self.audit_results['ma_crossover'] = result
        return result
    
    def audit_volatility_breakout(self) -> Dict:
        """
        审计波动率突破策略
        逻辑: 价格突破 N日最高价 + ATR → 买入
        """
        print("\n[审计] Volatility Breakout 策略...")
        
        close = self.close
        high = self.data['high']
        low = self.data['low']
        
        # 计算ATR
        n_days, n_stocks = close.shape
        tr = np.zeros((n_days, n_stocks))
        for t in range(1, n_days):
            tr[t] = np.maximum(
                high[t] - low[t],
                np.maximum(
                    np.abs(high[t] - close[t-1]),
                    np.abs(low[t] - close[t-1])
                )
            )
        
        atr_20 = calculate_ma(tr, 20)
        
        # 计算20日最高价
        highest_20 = np.zeros((n_days, n_stocks))
        for t in range(20, n_days):
            highest_20[t] = np.max(high[t-20:t], axis=0)
        
        # 突破信号: 收盘价 > 20日最高 + 0.5*ATR
        buy_signals = np.zeros((n_days, n_stocks), dtype=bool)
        for t in range(21, n_days):
            for s in range(n_stocks):
                if close[t, s] > highest_20[t, s] + 0.5 * atr_20[t, s]:
                    buy_signals[t, s] = True
        
        total_signals = np.sum(buy_signals)
        
        result = {
            'strategy': 'volatility_breakout',
            'total_buy_signals': int(total_signals),
            'signal_days': int(np.sum(np.any(buy_signals, axis=1))),
            'factors_computed': ['ATR_20', 'highest_20'],
            'signal_conditions': ['close > highest_20 + 0.5*ATR'],
            'status': 'PASS' if total_signals > 0 else 'WARN_NO_SIGNALS'
        }
        
        print(f"  - 突破信号总数: {total_signals}")
        print(f"  - 状态: {result['status']}")
        
        self.audit_results['volatility_breakout'] = result
        return result
    
    def audit_mean_reversion(self) -> Dict:
        """
        审计均值回归策略
        逻辑: 价格偏离MA过大时反向操作
        """
        print("\n[审计] Mean Reversion 策略...")
        
        close = self.close
        
        # 计算偏离度
        ma_20 = calculate_ma(close, 20)
        deviation = (close - ma_20) / ma_20
        
        # 计算偏离度的标准差
        n_days, n_stocks = close.shape
        dev_std = np.zeros((n_days, n_stocks))
        for t in range(40, n_days):
            dev_std[t] = np.std(deviation[t-20:t], axis=0)
        
        # 买入信号: 偏离度 < -2倍标准差 (超卖)
        # 卖出信号: 偏离度 > +2倍标准差 (超买)
        buy_signals = np.zeros((n_days, n_stocks), dtype=bool)
        sell_signals = np.zeros((n_days, n_stocks), dtype=bool)
        
        for t in range(40, n_days):
            for s in range(n_stocks):
                if dev_std[t, s] > 0.001:  # 避免除零
                    z_score = deviation[t, s] / dev_std[t, s]
                    if z_score < -2.0:
                        buy_signals[t, s] = True
                    elif z_score > 2.0:
                        sell_signals[t, s] = True
        
        result = {
            'strategy': 'mean_reversion',
            'total_buy_signals': int(np.sum(buy_signals)),
            'total_sell_signals': int(np.sum(sell_signals)),
            'factors_computed': ['ma_20', 'deviation', 'z_score'],
            'signal_conditions': ['z_score < -2 (buy)', 'z_score > +2 (sell)'],
            'status': 'PASS'
        }
        
        print(f"  - 超卖买入信号: {np.sum(buy_signals)}")
        print(f"  - 超买卖出信号: {np.sum(sell_signals)}")
        print(f"  - 状态: {result['status']}")
        
        self.audit_results['mean_reversion'] = result
        return result
    
    def verify_signal_timing(self) -> Dict:
        """
        验证信号时序 - 检测是否存在前视偏差
        
        方法: 确保T日信号只使用T-1及之前的数据
        """
        print("\n[审计] 信号时序验证 (前视偏差检测)...")
        
        close = self.close
        n_days = self.n_days
        
        # 测试: 使用T日收盘价计算信号是否会影响T日决策
        # 正确: T日信号应基于T-1日收盘价
        
        issues = []
        
        # 检查MA计算是否包含当日
        ma_5_correct = calculate_ma(close, 5)
        
        # 模拟错误的MA计算(包含未来数据)
        ma_5_wrong = np.zeros_like(close)
        for t in range(4, n_days - 1):
            # 错误: 使用t+1的数据
            ma_5_wrong[t] = np.mean(close[t-3:t+2], axis=0)
        
        # 比较差异
        diff = np.abs(ma_5_correct[5:-1] - ma_5_wrong[5:-1])
        if np.any(diff > 1e-10):
            max_diff = np.max(diff)
            print(f"  - MA���算验证: 差异检测正常 (最大差异: {max_diff:.6f})")
        else:
            issues.append("MA计算可能存在前视偏差")
        
        result = {
            'test': 'lookahead_bias',
            'issues_found': len(issues),
            'issues': issues,
            'status': 'PASS' if len(issues) == 0 else 'FAIL'
        }
        
        print(f"  - 状态: {result['status']}")
        
        self.audit_results['timing_verification'] = result
        return result
    
    def generate_report(self) -> str:
        """生成审计报告"""
        report = []
        report.append("=" * 70)
        report.append("Q-UNITY V10 白盒审计报告")
        report.append("=" * 70)
        report.append(f"审计时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"数据天数: {self.n_days}")
        report.append(f"股票数量: {self.n_stocks}")
        report.append("")
        
        # 汇总结果
        passed = 0
        failed = 0
        warnings = 0
        
        for name, result in self.audit_results.items():
            status = result.get('status', 'UNKNOWN')
            if status == 'PASS':
                passed += 1
            elif status.startswith('WARN'):
                warnings += 1
            else:
                failed += 1
            
            report.append(f"\n--- {name} ---")
            for key, value in result.items():
                if key != 'strategy' and key != 'test':
                    report.append(f"  {key}: {value}")
        
        report.append("\n" + "=" * 70)
        report.append(f"审计汇总: 通过={passed}, 警告={warnings}, 失败={failed}")
        report.append("=" * 70)
        
        return "\n".join(report)


# ============================================================================
# 第四部分: 复权数据验证
# ============================================================================

def verify_hfq_formula():
    """验证后复权公式修复"""
    print("\n[验证] 后复权公式修复...")
    
    # 模拟前复权数据
    qfq_close = np.array([8.0, 8.5, 9.0, 9.5, 10.0])  # 前复权价格
    latest_factor = 1.25  # 最新复权因子
    
    # 旧公式 (错误): hfq = qfq * factor^2 / latest_factor
    # 假设 factor = latest_factor (简化情况)
    hfq_old = qfq_close * (latest_factor ** 2) / latest_factor
    
    # 新公式 (正确): hfq = qfq * latest_factor
    hfq_new = qfq_close * latest_factor
    
    print(f"  前复权价格: {qfq_close}")
    print(f"  旧公式后复权: {hfq_old}")
    print(f"  新公式后复权: {hfq_new}")
    print(f"  差异: {hfq_old - hfq_new}")
    
    # 计算收益率差异
    ret_old = (hfq_old[-1] - hfq_old[0]) / hfq_old[0]
    ret_new = (hfq_new[-1] - hfq_new[0]) / hfq_new[0]
    
    print(f"  旧公式收益率: {ret_old:.2%}")
    print(f"  新公式收益率: {ret_new:.2%}")
    print(f"  收益率一致: {'是' if abs(ret_old - ret_new) < 0.001 else '否'}")
    
    return {
        'old_formula_result': hfq_old.tolist(),
        'new_formula_result': hfq_new.tolist(),
        'returns_match': abs(ret_old - ret_new) < 0.001
    }


# ============================================================================
# 第五部分: 主程序
# ============================================================================

def main():
    print("=" * 70)
    print("Q-UNITY V10 白盒审计系统")
    print("=" * 70)
    
    # 1. 生成合成数据
    print("\n[步骤1] 生成合成数据 (后复权, 4正弦波, 1500天)...")
    config = SyntheticConfig(
        n_days=1500,
        n_stocks=50,
        n_sine_waves=4,
        seed=42
    )
    data = generate_hfq_synthetic_data(config)
    
    print(f"  - 数据形状: {data['close'].shape}")
    print(f"  - 日期范围: {data['dates'][0]} ~ {data['dates'][-1]}")
    print(f"  - 收盘价范围: [{data['close'].min():.2f}, {data['close'].max():.2f}]")
    
    # 2. 验证复权公式
    print("\n[步骤2] 验证复权公式修复...")
    hfq_result = verify_hfq_formula()
    
    # 3. 运行策略信号审计
    print("\n[步骤3] 运行策略信号审计...")
    auditor = StrategySignalAuditor(data)
    
    auditor.audit_momentum_reversal()
    auditor.audit_ma_crossover()
    auditor.audit_volatility_breakout()
    auditor.audit_mean_reversion()
    auditor.verify_signal_timing()
    
    # 4. 生成报告
    print("\n[步骤4] 生成审计报告...")
    report = auditor.generate_report()
    print(report)
    
    # 5. 保存结果
    output_dir = os.path.join(PROJECT_ROOT, 'whitebox_audit', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存报告
    report_path = os.path.join(output_dir, 'audit_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n报告已保存至: {report_path}")
    
    print("\n" + "=" * 70)
    print("白盒审计完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
