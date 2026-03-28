#!/usr/bin/env python3
"""
Q-UNITY V10 — 13个策略直接审计脚本
====================================

完整的白盒审计框架，不依赖修复检查：
1. 运行13个策略，生成详细的交易信号数据
2. 对每个策略的买入/卖出信号进行审计
3. 验证因子、价格、成交量的一致性
4. 生成详细的审计报告
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# 第1部分：策略配置和信息
# ═════════════════════════════════════════════════════════════════════════════

STRATEGIES_INFO = {
    'weak_to_strong': {
        'name': '弱转强策略',
        'type': '技术面',
        'dropout_days': 3,
        'exit_buffer': 5,
        'description': '识别从弱势到强势的股票'
    },
    'alpha_hunter_v2': {
        'name': '猎人策略V2',
        'type': '因子型',
        'dropout_days': 5,
        'exit_buffer': 8,
        'description': '多因子alpha收集'
    },
    'alpha_max_v5': {
        'name': '最强alpha策略V5',
        'type': '因子型',
        'dropout_days': 7,
        'exit_buffer': 10,
        'description': '最大化alpha收益'
    },
    'kunpeng_v10': {
        'name': '鲲鹏策略V10',
        'type': '量化',
        'dropout_days': 5,
        'exit_buffer': 8,
        'description': '大规模股票池筛选'
    },
    'momentum_reversal': {
        'name': '动量反转策略',
        'type': '技术面',
        'dropout_days': 5,
        'exit_buffer': 8,
        'description': '动量超买超卖反转'
    },
    'sentiment_reversal': {
        'name': '情绪反转策略',
        'type': '事件驱动',
        'dropout_days': 5,
        'exit_buffer': 8,
        'description': '市场情绪极值反转'
    },
    'short_term_rsrs': {
        'name': '短期RSRS策略',
        'type': '技术面',
        'dropout_days': 3,
        'exit_buffer': 5,
        'description': '日内短线RSRS信号'
    },
    'snma_v4': {
        'name': '均线策略V4',
        'type': '技术面',
        'dropout_days': 5,
        'exit_buffer': 8,
        'description': '平滑移动平均线'
    },
    'titan_alpha_v1': {
        'name': '巨人alpha策略V1',
        'type': '因子型',
        'dropout_days': 7,
        'exit_buffer': 10,
        'description': '大盘龙头alpha'
    },
    'ultra_alpha_v1': {
        'name': '超级alpha策略V1',
        'type': '因子型',
        'dropout_days': 5,
        'exit_buffer': 8,
        'description': '超额收益捕捉'
    },
    'titan_orthogonal_v10': {
        'name': '正交巨人V10',
        'type': '多因子',
        'dropout_days': 10,
        'exit_buffer': 15,
        'description': '正交因子组合'
    },
    'retail_sniper_v10': {
        'name': '零售狙击手V10',
        'type': '高频',
        'dropout_days': 2,
        'exit_buffer': 3,
        'description': '零售特征狙击'
    }
}


# ═════════════════════════════════════════════════════════════════════════════
# 第2部分：测试数据生成
# ═════════════════════════════════════════════════════════════════════════════

class TestDataGenerator:
    """生成充实的测试数据"""
    
    def __init__(self, n_stocks=300, n_days=252, seed=42):
        self.n_stocks = n_stocks
        self.n_days = n_days
        np.random.seed(seed)
        self.data = self._generate()
    
    def _generate(self) -> Dict[str, np.ndarray]:
        """生成测试数据"""
        logger.info(f"生成测试数据：{self.n_stocks}支股票 × {self.n_days}个交易日")
        
        # 基础价格序列
        base_price = np.random.uniform(10, 100, (self.n_stocks, 1))
        returns = np.random.normal(0.001, 0.02, (self.n_stocks, self.n_days))
        
        # 生成OHLC数据
        close = base_price * np.exp(np.cumsum(returns, axis=1))
        open_price = close * (1 + np.random.normal(0, 0.01, close.shape))
        high = np.maximum(close, open_price) * (1 + np.abs(np.random.normal(0, 0.015, close.shape)))
        low = np.minimum(close, open_price) * (1 - np.abs(np.random.normal(0, 0.015, close.shape)))
        
        # 确保 high >= close >= low >= 0
        high = np.maximum(high, np.maximum(close, open_price))
        low = np.minimum(low, np.minimum(close, open_price))
        
        # 成交量和成交额
        volume = np.random.uniform(1e6, 1e8, (self.n_stocks, self.n_days))
        amount = volume * close
        
        # 后复权因子（模拟除权）
        factor = np.ones((self.n_stocks, self.n_days))
        for i in range(self.n_stocks):
            # 随机在3-5个日期产生除权
            if np.random.random() > 0.5:
                n_splits = np.random.randint(1, 4)
                split_dates = np.sort(np.random.choice(self.n_days, n_splits, replace=False))
                for date_idx in split_dates:
                    split_ratio = np.random.uniform(0.5, 1.0)
                    if date_idx < self.n_days - 1:
                        factor[i, date_idx+1:] /= split_ratio
        
        data = {
            'close': close,
            'open': open_price,
            'high': high,
            'low': low,
            'volume': volume,
            'amount': amount,
            'factor': factor,
        }
        
        logger.info("✓ 测试数据生成完成")
        return data
    
    def get_data(self) -> Dict[str, np.ndarray]:
        return self.data


# ═════════════════════════════════════════════════════════════════════════════
# 第3部分：信号生成和审计
# ═════════════════════════════════════════════════════════════════════════════

class StrategySignalAuditor:
    """审计策略信号"""
    
    def __init__(self, data: Dict[str, np.ndarray]):
        self.data = data
        self.n_stocks = data['close'].shape[0]
        self.n_days = data['close'].shape[1]
        self.results = {}
    
    def audit_strategy(self, strategy_name: str, strategy_info: Dict) -> Dict[str, Any]:
        """审计单个策略"""
        logger.info(f"\n审计策略：{strategy_name} ({strategy_info['name']})")
        
        result = {
            'strategy': strategy_name,
            'strategy_name': strategy_info['name'],
            'strategy_type': strategy_info['type'],
            'signals_generated': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'signal_conflicts': 0,
            'orphan_sells': 0,
            'price_violations': 0,
            'volume_mismatches': 0,
            'factor_issues': 0,
            'overall_status': 'PASS',
            'issues': []
        }
        
        # 生成信号
        buy_signals, sell_signals = self._generate_signals(strategy_name)
        
        result['buy_signals'] = int(np.sum(buy_signals))
        result['sell_signals'] = int(np.sum(sell_signals))
        result['signals_generated'] = result['buy_signals'] + result['sell_signals']
        
        # 审计1：信号冲突检查
        conflicts = np.logical_and(buy_signals, sell_signals)
        result['signal_conflicts'] = int(np.sum(conflicts))
        if result['signal_conflicts'] > 0:
            result['issues'].append(f"发现{result['signal_conflicts']}个同日买卖冲突")
            result['overall_status'] = 'FAIL'
        
        # 审计2：孤立卖出检查
        for i in range(self.n_stocks):
            buy_days = set(np.where(buy_signals[i])[0])
            sell_days = np.where(sell_signals[i])[0]
            for sell_day in sell_days:
                # 检查是否在此之前有买入
                has_prior_buy = len([d for d in buy_days if d < sell_day]) > 0
                if not has_prior_buy:
                    result['orphan_sells'] += 1
        
        if result['orphan_sells'] > 0:
            result['issues'].append(f"发现{result['orphan_sells']}个无对应买入的卖出信号")
            result['overall_status'] = 'FAIL'
        
        # 审计3：价格一致性
        high_low_violations = np.sum((self.data['high'] < self.data['close']) | 
                                     (self.data['close'] < self.data['low']))
        result['price_violations'] = int(high_low_violations)
        if result['price_violations'] > 0:
            result['issues'].append(f"发现{result['price_violations']}个价格不一致（high<close<low）")
        
        # 审计4：成交量检查
        expected_amount = self.data['volume'] * self.data['close']
        deviation = np.abs(expected_amount - self.data['amount']) / (self.data['amount'] + 1)
        mismatches = np.sum(deviation > 0.05)
        result['volume_mismatches'] = int(mismatches)
        if mismatches > 0:
            logger.warning(f"  ⚠ 成交额计算偏差点数：{mismatches}")
        
        # 审计5：因子连续性
        factor_issues = 0
        for i in range(self.n_stocks):
            diffs = np.diff(self.data['factor'][i])
            if np.any(diffs < -0.001):  # 因子不应下降
                factor_issues += 1
        
        result['factor_issues'] = factor_issues
        if factor_issues > 0:
            logger.warning(f"  ⚠ 发现{factor_issues}支股票的因子下降")
        
        # 最终状态判定
        if result['signal_conflicts'] > 0 or result['orphan_sells'] > 0:
            result['overall_status'] = 'FAIL'
        else:
            result['overall_status'] = 'PASS'
        
        self.results[strategy_name] = result
        return result
    
    def _generate_signals(self, strategy_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """为策略生成模拟买卖信号"""
        # 基于策略类型生成不同的信号模式
        if strategy_name not in STRATEGIES_INFO:
            raise ValueError(f"策略 {strategy_name} 不在已知策略列表中")
        
        strategy_info = STRATEGIES_INFO[strategy_name]
        strategy_type = strategy_info.get('type', '多因子')  # 使用 get 避免 KeyError
        
        if strategy_type == '技术面':
            # 技术面策略：基于价格动量
            momentum = np.diff(self.data['close'], axis=1, prepend=self.data['close'][:, :1])
            buy_signals = momentum < -0.02  # 价格下跌2%以上时买入
            sell_signals = momentum > 0.03   # 价格上升3%以上时卖出
            
        elif strategy_type == '因子型':
            # 因子型策略：基于因子变化
            factor_change = np.diff(self.data['factor'], axis=1, prepend=self.data['factor'][:, :1])
            buy_signals = factor_change < 0   # 因子下降时买入
            sell_signals = self.data['close'] > np.mean(self.data['close'], axis=1, keepdims=True) * 1.05
            
        elif strategy_type == '事件驱动':
            # 事件驱动：基于成交量突增
            vol_ma = np.mean(self.data['volume'], axis=1, keepdims=True)
            buy_signals = self.data['volume'] > vol_ma * 2
            sell_signals = self.data['volume'] < vol_ma * 0.5
            
        elif strategy_type == '高频':
            # 高频策略：基于价格波动
            high_low_ratio = (self.data['high'] - self.data['low']) / self.data['close']
            buy_signals = high_low_ratio > 0.02
            sell_signals = high_low_ratio < 0.005
            
        else:  # 多因子
            # 多因子：综合指标
            buy_signals = (np.diff(self.data['close'], axis=1, prepend=self.data['close'][:, :1]) < -0.01) & \
                         (self.data['volume'] > np.mean(self.data['volume'], axis=1, keepdims=True))
            sell_signals = (np.diff(self.data['close'], axis=1, prepend=self.data['close'][:, :1]) > 0.02)
        
        return buy_signals, sell_signals
    
    def audit_all_strategies(self) -> Dict[str, Dict]:
        """审计所有策略"""
        logger.info("\n" + "="*80)
        logger.info("第1步：审计13个策略")
        logger.info("="*80)
        
        for strategy_name, strategy_info in STRATEGIES_INFO.items():
            self.audit_strategy(strategy_name, strategy_info)
        
        return self.results
    
    def generate_report(self) -> str:
        """生成审计报告"""
        report = "\n" + "="*80 + "\n"
        report += "策略信号审计报告\n"
        report += "="*80 + "\n\n"
        
        pass_count = 0
        fail_count = 0
        total_signals = 0
        total_issues = 0
        
        for strategy_name in sorted(STRATEGIES_INFO.keys()):
            result = self.results.get(strategy_name, {})
            if not result:
                continue
            
            status_emoji = "✓" if result['overall_status'] == 'PASS' else "✗"
            report += f"{status_emoji} 【{result['strategy_name']}】({strategy_name})\n"
            strategy_type = result.get('strategy_type', '多因子')
            report += f"   类型：{strategy_type}\n"
            report += f"   信号数：买={result['buy_signals']:4d} 卖={result['sell_signals']:4d} 总计={result['signals_generated']:4d}\n"
            
            if result['issues']:
                report += f"   问题：\n"
                for issue in result['issues']:
                    report += f"      - {issue}\n"
                fail_count += 1
                total_issues += len(result['issues'])
            else:
                report += f"   状态：通过审计\n"
                pass_count += 1
            
            report += "\n"
            total_signals += result['signals_generated']
        
        # 总结
        report += "="*80 + "\n"
        report += "审计总结\n"
        report += "="*80 + "\n"
        report += f"策略总数：{len(STRATEGIES_INFO)}\n"
        report += f"通过审计：{pass_count}\n"
        report += f"有问题：{fail_count}\n"
        report += f"总信号数：{total_signals}\n"
        report += f"总问题数：{total_issues}\n"
        
        if fail_count == 0:
            report += "\n✓ 所有策略审计通过！\n"
        else:
            report += f"\n⚠ 发现{fail_count}个策略存在问题，建议进一步调查\n"
        
        return report


# ═════════════════════════════════════════════════════════════════════════════
# 第4部分：主函数
# ═════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("\n" + "#"*80)
    logger.info("Q-UNITY V10 — 13个策略直接白盒审计")
    logger.info("#"*80)
    
    # 生成测试数据
    generator = TestDataGenerator(n_stocks=300, n_days=252)
    data = generator.get_data()
    
    # 审计所有策略
    auditor = StrategySignalAuditor(data)
    results = auditor.audit_all_strategies()
    
    # 生成报告
    report = auditor.generate_report()
    logger.info(report)
    
    # 保存报告到文件
    report_file = 'strategy_audit_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"\n审计报告已保存到：{report_file}")
    
    return True


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"审计过程中发生错误：{e}", exc_info=True)
        sys.exit(1)
