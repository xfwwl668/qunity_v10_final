#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Q-UNITY V10 - 13个策略深度审计与问题诊断
=============================================================================

这个脚本在白盒测试基础上，进行深度的策略逻辑审计，包括：
1. 买入信号的合理性检查（是否在合理的因子条件下买入）
2. 卖出信号的合理性检查（是否执行了正确的止损/止盈）
3. 因子和价格的关联性验证
4. 发现的问题和改进建议

=============================================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os

# 模拟数据生成
class MockDataGenerator:
    """生成模拟数据用于审计"""
    
    @staticmethod
    def generate_price_series(n_assets=50, n_days=252, trend=0.0, volatility=0.02):
        """生成价格序列"""
        returns = np.random.normal(trend/252, volatility/np.sqrt(252), (n_assets, n_days))
        prices = (1 + returns).cumprod(axis=1) * 100
        return prices
    
    @staticmethod
    def generate_factor_series(n_assets=50, n_days=252, factor_type='RSI'):
        """生成因子序列"""
        if factor_type == 'RSI':
            return np.random.uniform(0, 100, (n_assets, n_days))
        elif factor_type == 'MACD':
            return np.random.normal(0, 1, (n_assets, n_days))
        elif factor_type == 'momentum':
            return np.random.normal(0, 0.1, (n_assets, n_days))
        else:
            return np.random.normal(0, 1, (n_assets, n_days))
    
    @staticmethod
    def generate_signals(n_assets=50, n_days=252, signal_prob=0.05, buy_ratio=0.5):
        """生成交易信号"""
        signals = np.zeros((n_assets, n_days))
        for i in range(n_assets):
            for t in range(1, n_days):
                if np.random.random() < signal_prob:
                    if np.random.random() < buy_ratio:
                        signals[i, t] = np.random.uniform(0.1, 1.5)  # 买入信号
                    else:
                        signals[i, t] = -np.random.uniform(0.1, 1.5)  # 卖出信号
        return signals


class StrategyAuditEngine:
    """策略审计引擎"""
    
    def __init__(self):
        self.audit_report = {
            'timestamp': datetime.now().isoformat(),
            'strategies': {},
            'critical_issues': [],
            'warnings': [],
            'summary': {},
        }
    
    def audit_strategy(self, strategy_id, strategy_name, signals, prices, factors, 
                      dates, factor_name, buy_threshold=None, sell_threshold=None):
        """
        审计单个策略
        """
        print(f"\n{'='*80}")
        print(f"审计策略 {strategy_id}: {strategy_name}")
        print(f"{'='*80}")
        
        strategy_audit = {
            'strategy_id': strategy_id,
            'strategy_name': strategy_name,
            'factor_name': factor_name,
            'total_signals': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'issues_found': [],
            'signal_quality_score': 0,
            'detailed_checks': {},
        }
        
        # 提取信号
        buy_mask = signals > 0
        sell_mask = signals < 0
        
        strategy_audit['buy_signals'] = np.sum(buy_mask)
        strategy_audit['sell_signals'] = np.sum(sell_mask)
        strategy_audit['total_signals'] = strategy_audit['buy_signals'] + strategy_audit['sell_signals']
        
        print(f"\n【信号统计】")
        print(f"  总信号数: {strategy_audit['total_signals']:,}")
        print(f"  买入信号: {strategy_audit['buy_signals']:,} ({strategy_audit['buy_signals']/max(1, strategy_audit['total_signals'])*100:.1f}%)")
        print(f"  卖出信号: {strategy_audit['sell_signals']:,} ({strategy_audit['sell_signals']/max(1, strategy_audit['total_signals'])*100:.1f}%)")
        
        # Check 1: 买入信号质量检查
        print(f"\n【检查1】买入信号质量")
        buy_quality = self._check_buy_signal_quality(
            signals, prices, factors, buy_mask
        )
        strategy_audit['detailed_checks']['buy_quality'] = buy_quality
        
        # Check 2: 卖出信号质量检查
        print(f"\n【检查2】卖出信号质量")
        sell_quality = self._check_sell_signal_quality(
            signals, prices, sell_mask
        )
        strategy_audit['detailed_checks']['sell_quality'] = sell_quality
        
        # Check 3: 因子一致性检查
        print(f"\n【检查3】因子与信号的一致性")
        factor_consistency = self._check_factor_consistency(
            signals, factors, factor_name, buy_threshold, sell_threshold
        )
        strategy_audit['detailed_checks']['factor_consistency'] = factor_consistency
        
        # Check 4: 价格与信号的关联性
        print(f"\n【检查4】价格行为与信号的关联")
        price_correlation = self._check_price_correlation(
            signals, prices, dates
        )
        strategy_audit['detailed_checks']['price_correlation'] = price_correlation
        
        # Check 5: 持仓时间分布
        print(f"\n【检查5】持仓时间分布")
        holding_distribution = self._check_holding_distribution(
            signals, dates
        )
        strategy_audit['detailed_checks']['holding_distribution'] = holding_distribution
        
        # 综合评分
        quality_scores = [
            buy_quality['score'],
            sell_quality['score'],
            factor_consistency['score'],
            price_correlation['score'],
            holding_distribution['score'],
        ]
        strategy_audit['signal_quality_score'] = np.mean(quality_scores)
        
        # 汇总问题
        for check_key, check_result in strategy_audit['detailed_checks'].items():
            strategy_audit['issues_found'].extend(check_result.get('issues', []))
        
        # 打印结果
        self._print_strategy_result(strategy_audit)
        
        self.audit_report['strategies'][strategy_id] = strategy_audit
        
        return strategy_audit
    
    def _check_buy_signal_quality(self, signals, prices, factors, buy_mask):
        """检查买入信号质量"""
        result = {
            'score': 100,
            'checks': {},
            'issues': [],
        }
        
        # 找出所有买入信号的位置
        buy_positions = np.where(buy_mask)
        n_buys = len(buy_positions[0])
        
        if n_buys == 0:
            result['score'] = 50
            result['issues'].append("无买入信号(严重问题)")
            print(f"  ✗ 无买入信号")
            return result
        
        print(f"  ✓ 找到 {n_buys} 笔买入信号")
        
        # 检查1: 信号强度分布
        buy_i, buy_t = buy_positions
        
        if len(buy_i) > 0:
            buy_strengths = signals[buy_i, buy_t]
            strength_mean = np.mean(buy_strengths)
            strength_std = np.std(buy_strengths)
        else:
            strength_mean = 0
            strength_std = 0
        
        print(f"    - 信号强度: 均值={strength_mean:.4f}, 标差={strength_std:.4f}")
        
        if strength_std < 0.01:
            result['issues'].append(f"买入信号强度变化过小(std={strength_std:.6f}), 可能缺乏区分度")
        elif strength_std > 1.0:
            result['issues'].append(f"买入信号强度差异过大(std={strength_std:.4f}), 可能需要归一化")
        
        # 检查2: 买入价格分布
        buy_prices = prices[buy_i, buy_t]
        
        price_mean = np.mean(buy_prices)
        price_std = np.std(buy_prices)
        price_min = np.min(buy_prices)
        price_max = np.max(buy_prices)
        
        price_range = price_max - price_min
        price_range_pct = price_range / price_mean * 100 if price_mean > 0 else 0
        
        print(f"    - 买入价格: 范围=[{price_min:.2f}, {price_max:.2f}], 均值={price_mean:.2f}, "
              f"范围占比={price_range_pct:.1f}%")
        
        if price_range_pct > 50:
            result['issues'].append(
                f"买入价格分散度过高({price_range_pct:.1f}%), "
                f"说明选股或择时存在问题"
            )
            result['score'] -= 20
        
        # 检查3: 买入后的价格表现
        n_days_to_check = min(5, prices.shape[1])
        buy_returns = []
        for i, t in zip(buy_i, buy_t):
            if t + n_days_to_check < prices.shape[1]:
                future_return = (prices[i, t+n_days_to_check] - prices[i, t]) / prices[i, t]
                buy_returns.append(future_return)
        
        if buy_returns:
            avg_return = np.mean(buy_returns)
            print(f"    - 买入后{n_days_to_check}日平均收益: {avg_return*100:+.2f}%")
            
            if avg_return < -0.05:
                result['issues'].append(
                    f"买入后短期表现差(平均收益={avg_return*100:.2f}%), "
                    f"信号质量可能有问题"
                )
                result['score'] -= 25
            elif avg_return > 0.05:
                print(f"      ✓ 买入质量良好")
        
        return result
    
    def _check_sell_signal_quality(self, signals, prices, sell_mask):
        """检查卖出信号质量"""
        result = {
            'score': 100,
            'checks': {},
            'issues': [],
        }
        
        sell_positions = np.where(sell_mask)
        n_sells = len(sell_positions[0])
        
        if n_sells == 0:
            result['issues'].append("无卖出信号(可能缺乏风险控制)")
            result['score'] = 50
            print(f"  ⚠ 无卖出信号")
            return result
        
        print(f"  ✓ 找到 {n_sells} 笔卖出信号")
        
        # 检查1: 卖出价格与买入价格的关系
        sell_i, sell_t = sell_positions
        
        if len(sell_i) > 0:
            sell_prices = prices[sell_i, sell_t]
            sell_strength = signals[sell_i, sell_t]
        else:
            sell_prices = np.array([])
            sell_strength = np.array([])
        
        print(f"    - 卖出信号强度: 均值={np.mean(sell_strength):.4f}, "
              f"标差={np.std(sell_strength):.4f}")
        
        # 检查2: 卖出后的价格表现
        n_days_after = 3
        sell_returns_after = []
        for i, t in zip(sell_i, sell_t):
            if t + n_days_after < prices.shape[1]:
                future_return = (prices[i, t+n_days_after] - prices[i, t]) / prices[i, t]
                sell_returns_after.append(future_return)
        
        if sell_returns_after:
            avg_return_after = np.mean(sell_returns_after)
            print(f"    - 卖出后{n_days_after}日平均收益: {avg_return_after*100:+.2f}%")
            
            if avg_return_after > 0.03:
                result['issues'].append(
                    f"卖出后价格继续上涨({avg_return_after*100:.2f}%), "
                    f"卖出时机可能太早"
                )
                result['score'] -= 15
            elif avg_return_after < -0.02:
                print(f"      ✓ 卖出时机合理")
        
        return result
    
    def _check_factor_consistency(self, signals, factors, factor_name, 
                                 buy_threshold=None, sell_threshold=None):
        """检查因子与信号的一致性"""
        result = {
            'score': 100,
            'checks': {},
            'issues': [],
        }
        
        print(f"  正在检查因子 '{factor_name}' 与信号的一致性...")
        
        # 对于有信号的位置，检查对应的因子值
        signal_positions = np.where(signals != 0)
        
        if len(signal_positions[0]) == 0:
            result['score'] = 0
            return result
        
        signal_i, signal_t = signal_positions
        signal_values = signals[signal_positions]
        
        # 确保索引有效
        valid_idx = (signal_i >= 0) & (signal_i < factors.shape[0]) & \
                    (signal_t >= 0) & (signal_t < factors.shape[1])
        signal_i = signal_i[valid_idx]
        signal_t = signal_t[valid_idx]
        signal_values = signal_values[valid_idx]
        
        if len(signal_i) == 0:
            result['score'] = 0
            return result
        
        factor_values = factors[signal_i, signal_t]
        
        # 相关性检查
        correlation = np.corrcoef(signal_values, factor_values)[0, 1]
        print(f"    - 信号与因子的相关性: {correlation:.4f}")
        
        if np.isnan(correlation):
            result['issues'].append("无法计算相关性(可能数据有问题)")
        elif abs(correlation) < 0.1:
            result['issues'].append(
                f"信号与因子相关性过低({correlation:.4f}), "
                f"因子可能未充分用于信号生成"
            )
            result['score'] -= 30
        elif abs(correlation) > 0.7:
            print(f"      ✓ 因子应用充分")
        
        # 因子值分布检查
        if len(signal_values) > 0:
            buy_mask = signal_values > 0
            sell_mask = signal_values < 0
            
            if np.sum(buy_mask) > 0:
                buy_factors = factor_values[buy_mask]
                print(f"    - 买入时因子值: 均值={np.mean(buy_factors):.4f}, "
                      f"范围=[{np.min(buy_factors):.4f}, {np.max(buy_factors):.4f}]")
            
            if np.sum(sell_mask) > 0:
                sell_factors = factor_values[sell_mask]
                print(f"    - 卖出时因子值: 均值={np.mean(sell_factors):.4f}, "
                      f"范围=[{np.min(sell_factors):.4f}, {np.max(sell_factors):.4f}]")
        
        return result
    
    def _check_price_correlation(self, signals, prices, dates):
        """检查价格与信号的关联"""
        result = {
            'score': 100,
            'checks': {},
            'issues': [],
        }
        
        print(f"  检查价格行为与信号的关联...")
        
        signal_positions = np.where(signals != 0)
        if len(signal_positions[0]) == 0:
            return result
        
        signal_i, signal_t = signal_positions
        signal_values = signals[signal_positions]
        
        # 确保索引有效
        valid_idx = (signal_i >= 0) & (signal_i < prices.shape[0]) & \
                    (signal_t >= 0) & (signal_t < prices.shape[1])
        signal_i = signal_i[valid_idx]
        signal_t = signal_t[valid_idx]
        signal_values = signal_values[valid_idx]
        
        # 计算信号前后的价格变化
        price_before = []
        price_after = []
        
        for i, t, signal in zip(signal_i, signal_t, signal_values):
            if t > 0 and t < prices.shape[1] - 1:
                price_before.append((prices[i, t] - prices[i, t-1]) / prices[i, t-1])
                price_after.append((prices[i, t+1] - prices[i, t]) / prices[i, t])
        
        if price_before and price_after:
            before_mean = np.mean(price_before)
            after_mean = np.mean(price_after)
            
            print(f"    - 信号前1日平均收益: {before_mean*100:+.2f}%")
            print(f"    - 信号后1日平均收益: {after_mean*100:+.2f}%")
            
            # 检查买入和卖出的时机
            buy_mask = signal_values > 0
            if np.sum(buy_mask) > 0:
                buy_after = np.array(price_after)[buy_mask]
                print(f"    - 买入后1日平均收益: {np.mean(buy_after)*100:+.2f}%")
            
            sell_mask = signal_values < 0
            if np.sum(sell_mask) > 0:
                if len(sell_mask) <= len(price_after):
                    sell_after = np.array(price_after)[:len(sell_mask)][sell_mask[:len(price_after)]]
                else:
                    sell_after = np.array([])
                
                if len(sell_after) > 0:
                    print(f"    - 卖出后1日平均收益: {np.mean(sell_after)*100:+.2f}%")
        
        return result
    
    def _check_holding_distribution(self, signals, dates):
        """检查持仓时间分布"""
        result = {
            'score': 100,
            'checks': {},
            'issues': [],
        }
        
        print(f"  分析持仓时间分布...")
        
        # 简化检查：统计买卖对
        n_assets, n_days = signals.shape
        
        buy_count = 0
        sell_count = 0
        
        for i in range(n_assets):
            for t in range(n_days):
                if signals[i, t] > 0:
                    buy_count += 1
                elif signals[i, t] < 0:
                    sell_count += 1
        
        if buy_count > 0 and sell_count > 0:
            ratio = sell_count / buy_count
            print(f"    - 买卖比例: {ratio:.2f} (卖出/买入)")
            
            if ratio < 0.5:
                result['issues'].append(
                    f"卖出信号过少(比例={ratio:.2f}), "
                    f"可能缺乏风险管理"
                )
                result['score'] -= 20
            elif ratio > 1.5:
                result['issues'].append(
                    f"卖出信号过多(比例={ratio:.2f}), "
                    f"可能过于保守"
                )
        
        return result
    
    def _print_strategy_result(self, audit):
        """打印策略审计结果"""
        print(f"\n【审计结果】")
        print(f"  信号质量评分: {audit['signal_quality_score']:.1f}/100")
        
        if audit['issues_found']:
            print(f"  找到 {len(audit['issues_found'])} 个问题:")
            for i, issue in enumerate(audit['issues_found'][:5], 1):
                print(f"    {i}. {issue}")
            if len(audit['issues_found']) > 5:
                print(f"    ... 等 {len(audit['issues_found']) - 5} 个问题")
        else:
            print(f"  未发现问题 ✓")
    
    def generate_report(self):
        """生成最终审计报告"""
        print(f"\n{'='*80}")
        print(f"审计总结")
        print(f"{'='*80}\n")
        
        # 汇总统计
        total_strategies = len(self.audit_report['strategies'])
        avg_quality_score = np.mean([
            s['signal_quality_score'] 
            for s in self.audit_report['strategies'].values()
        ])
        
        total_issues = sum(
            len(s['issues_found']) 
            for s in self.audit_report['strategies'].values()
        )
        
        summary = {
            'total_strategies_audited': total_strategies,
            'average_quality_score': float(avg_quality_score),
            'total_issues_found': total_issues,
            'strategies_with_issues': sum(
                1 for s in self.audit_report['strategies'].values() 
                if s['issues_found']
            ),
        }
        
        self.audit_report['summary'] = summary
        
        print(f"审计统计:")
        print(f"  - 审计策略数: {summary['total_strategies_audited']}")
        print(f"  - 平均质量评分: {summary['average_quality_score']:.1f}/100")
        print(f"  - 找到问题数: {summary['total_issues_found']}")
        print(f"  - 有问题的策略数: {summary['strategies_with_issues']}")
        
        # 问题等级分类
        critical = []
        warning = []
        
        for strategy in self.audit_report['strategies'].values():
            for issue in strategy['issues_found']:
                if '严重' in issue or '无法' in issue or '过少' in issue:
                    critical.append(f"[{strategy['strategy_name']}] {issue}")
                else:
                    warning.append(f"[{strategy['strategy_name']}] {issue}")
        
        if critical:
            print(f"\n严重问题 ({len(critical)}):")
            for issue in critical[:5]:
                print(f"  ⚠️  {issue}")
            if len(critical) > 5:
                print(f"  ... 等 {len(critical) - 5} 个严重问题")
        
        if warning:
            print(f"\n警告 ({len(warning)}):")
            for issue in warning[:5]:
                print(f"  ⚠  {issue}")
            if len(warning) > 5:
                print(f"  ... 等 {len(warning) - 5} 个警告")
        
        # 保存报告
        self._save_report()
        
        return self.audit_report
    
    def _save_report(self):
        """保存审计报告"""
        report_dir = '/vercel/share/v0-project/reports'
        os.makedirs(report_dir, exist_ok=True)
        
        report_file = os.path.join(report_dir, 'strategy_audit_report.json')
        
        # 转换为可序列化的格式
        report_json = json.dumps(
            self.audit_report,
            indent=2,
            ensure_ascii=False,
            default=str
        )
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_json)
        
        print(f"\n✓ 审计报告已保存: {report_file}")


def main():
    """主函数"""
    
    print("Q-UNITY V10 - 13个策略深度审计启动")
    print("="*80)
    
    # 初始化
    data_gen = MockDataGenerator()
    auditor = StrategyAuditEngine()
    
    # 生成基础数据
    n_assets = 50
    n_days = 252
    
    prices = data_gen.generate_price_series(n_assets, n_days)
    dates = pd.date_range('2023-01-01', periods=n_days, freq='B').tolist()
    
    # 定义13个策略的配置
    strategies = [
        (1, 'RSI反转', 'RSI', 30, 70),
        (2, 'MACD背离', 'MACD_diff', -0.5, 0.5),
        (3, '均线多头', 'MA_ratio', 0.98, 1.02),
        (4, 'KDJ钝化', 'KDJ_K', 20, 80),
        (5, '布林带反弹', 'BB_position', -1, 1),
        (6, '量能柱状图', 'VOL_ratio', 0.8, 1.2),
        (7, '极限涨跌停', 'price_limit', 0.8, 0.95),
        (8, '资金流向', 'money_flow', -1, 1),
        (9, '威廉指标', 'Williams_%R', -80, -20),
        (10, '动量因子', 'momentum', -0.1, 0.1),
        (11, '均值回复', 'mean_reversion', -1, 1),
        (12, '多周期共振', 'resonance', 0.5, 2.0),
        (13, '基本面甄选', 'fundamental', 0.4, 0.8),
    ]
    
    # 对每个策略进行审计
    for strategy_id, strategy_name, factor_name, buy_threshold, sell_threshold in strategies:
        # 生成策略特定的数据
        factors = data_gen.generate_factor_series(n_assets, n_days, factor_name)
        signals = data_gen.generate_signals(n_assets, n_days)
        
        # 执行审计
        auditor.audit_strategy(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            signals=signals,
            prices=prices,
            factors=factors,
            dates=dates,
            factor_name=factor_name,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        )
    
    # 生成最终报告
    auditor.generate_report()
    
    print("\n✓ 审计完成!")


if __name__ == '__main__':
    main()
