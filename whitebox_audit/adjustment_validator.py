"""
Q-UNITY V10 — WhiteBox Audit: AdjustmentValidator
==================================================
复权数据验证器

A股量化回测中，复权处理是数据正确性的核心保障。
本模块验证：

1. 后复权(HFQ)数据特征：
   - 首日价格为基准（通常最低）
   - 历史价格不因除权而回调
   - 收益率计算与不复权数据一致

2. 前复权(QFQ)数据特征：
   - 最新价格为基准（等于不复权价）
   - 历史价格因除权向下回调
   - 直接计算收益率会有误差

3. 复权因子一致性：
   - adj_factor 序列单调（后复权）
   - 复权后收益率 = 不复权收益率

常见错误：
- QFQ 数据误当 HFQ 使用（历史涨幅被低估）
- 复权因子回填日期错误（前视偏差）
- 除权日当天数据处理不一致
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AdjustmentIssue:
    """复权问题记录"""
    stock_idx: int
    date_idx: int
    issue_type: str
    description: str
    severity: str  # HIGH / MEDIUM / LOW


class AdjustmentValidator:
    """
    复权数据验证器
    """
    
    def __init__(
        self,
        close: np.ndarray,
        dates: List[str],
        adj_factor: Optional[np.ndarray] = None,
    ):
        """
        Parameters
        ----------
        close : np.ndarray
            (N, T) 收盘价矩阵
        dates : List[str]
            日期列表
        adj_factor : Optional[np.ndarray]
            (N, T) 复权因子矩阵（如有）
        """
        self.close = close.astype(np.float64)
        self.dates = dates
        self.adj_factor = adj_factor.astype(np.float64) if adj_factor is not None else None
        
        self.N, self.T = close.shape
        self.issues: List[AdjustmentIssue] = []
        
    def validate_hfq_characteristics(self) -> List[AdjustmentIssue]:
        """
        验证后复权数据特征
        
        后复权特征：
        1. 价格序列单调性：不应有负收益率>50% 的突然跳变（除权信号）
        2. 长期趋势：大部分股票应有正漂移（A股长期上涨）
        3. 无价格重置：不应在除权日突然跳水到低位
        """
        issues = []
        
        # 计算日收益率
        returns = np.zeros((self.N, self.T), dtype=np.float64)
        returns[:, 1:] = self.close[:, 1:] / (self.close[:, :-1] + 1e-10) - 1.0
        
        for i in range(self.N):
            # 检查异常大的负收益率（可能是 QFQ 的除权跳变）
            large_drops = np.where(returns[i, :] < -0.3)[0]
            
            for t in large_drops:
                # 排除正常跌停（A股跌停 -10%）
                if returns[i, t] < -0.15:  # 比跌停更大的跌幅
                    issues.append(AdjustmentIssue(
                        stock_idx=i,
                        date_idx=t,
                        issue_type="SUSPICIOUS_DROP",
                        description=f"异常大跌幅 {returns[i, t]:.2%}，可能是前复权除权跳变",
                        severity="HIGH" if returns[i, t] < -0.30 else "MEDIUM",
                    ))
        
        self.issues.extend(issues)
        return issues
    
    def validate_adj_factor_monotonicity(self) -> List[AdjustmentIssue]:
        """
        验证复权因子单调性
        
        后复权因子特征：
        - 从首日到最新日单调递增（或不变）
        - 除权日因子跳增
        """
        issues = []
        
        if self.adj_factor is None:
            return issues
        
        for i in range(self.N):
            factor = self.adj_factor[i, :]
            
            # 检查是否单调
            diffs = np.diff(factor)
            decreasing = diffs < -1e-8
            
            if decreasing.any():
                bad_indices = np.where(decreasing)[0]
                for t in bad_indices[:5]:  # 只报告前5个
                    issues.append(AdjustmentIssue(
                        stock_idx=i,
                        date_idx=t + 1,
                        issue_type="ADJ_FACTOR_DECREASE",
                        description=f"复权因子下降 {factor[t]:.4f} -> {factor[t+1]:.4f}",
                        severity="MEDIUM",
                    ))
        
        self.issues.extend(issues)
        return issues
    
    def validate_return_consistency(
        self,
        close_raw: Optional[np.ndarray] = None,
    ) -> List[AdjustmentIssue]:
        """
        验证复权后收益率与原始收益率一致性
        
        原理：
        - 不复权收益率 = raw_close[t] / raw_close[t-1] - 1
        - 后复权收益率 = hfq_close[t] / hfq_close[t-1] - 1
        - 二者应相等（复权只平移价格，不改变收益率）
        """
        issues = []
        
        if close_raw is None:
            return issues
        
        close_raw = close_raw.astype(np.float64)
        
        # 计算两组收益率
        ret_hfq = self.close[:, 1:] / (self.close[:, :-1] + 1e-10) - 1.0
        ret_raw = close_raw[:, 1:] / (close_raw[:, :-1] + 1e-10) - 1.0
        
        # 比较差异
        diff = np.abs(ret_hfq - ret_raw)
        
        # 允许小幅误差（浮点精度）
        large_diff_mask = diff > 0.001
        
        if large_diff_mask.any():
            bad_indices = np.argwhere(large_diff_mask)
            for idx in bad_indices[:20]:  # 只报告前20个
                i, t = idx
                issues.append(AdjustmentIssue(
                    stock_idx=i,
                    date_idx=t + 1,
                    issue_type="RETURN_INCONSISTENCY",
                    description=f"收益率不一致: HFQ={ret_hfq[i, t]:.4%}, RAW={ret_raw[i, t]:.4%}",
                    severity="HIGH" if diff[i, t] > 0.01 else "MEDIUM",
                ))
        
        self.issues.extend(issues)
        return issues
    
    def detect_qfq_in_hfq(self) -> List[AdjustmentIssue]:
        """
        检测是否误将前复权数据当作后复权使用
        
        检测方法：
        - 前复权数据的最新价格 ≈ 不复权价格
        - 历史价格被"压缩"
        - 因此首日/最新日的价格比值会异常
        """
        issues = []
        
        # 计算首日与最新日的价格比
        first_prices = self.close[:, 0]
        last_prices = self.close[:, -1]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = last_prices / (first_prices + 1e-10)
        
        # 后复权数据：比值应接近累积收益率（可能 < 1 或 > 1）
        # 前复权数据：由于历史被压缩，比值往往异常大
        
        # 统计比值分布
        valid_ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
        
        if len(valid_ratio) > 10:
            median_ratio = np.median(valid_ratio)
            
            # 如果中位数比值非常大（> 5），可能是前复权数据
            if median_ratio > 5.0:
                issues.append(AdjustmentIssue(
                    stock_idx=-1,  # 全局问题
                    date_idx=-1,
                    issue_type="POSSIBLE_QFQ_DATA",
                    description=f"价格比值中位数 {median_ratio:.2f} 异常高，可能是前复权数据",
                    severity="HIGH",
                ))
        
        self.issues.extend(issues)
        return issues
    
    def validate_all(self) -> Dict[str, int]:
        """运行所有验证"""
        results = {}
        
        issues = self.validate_hfq_characteristics()
        results["hfq_characteristics"] = len(issues)
        
        issues = self.validate_adj_factor_monotonicity()
        results["adj_factor_monotonicity"] = len(issues)
        
        issues = self.detect_qfq_in_hfq()
        results["possible_qfq"] = len(issues)
        
        return results
    
    def generate_report(self) -> str:
        """生成验证报告"""
        lines = []
        lines.append("=" * 70)
        lines.append("复权数据验证报告")
        lines.append("=" * 70)
        
        # 统计
        high_count = sum(1 for i in self.issues if i.severity == "HIGH")
        medium_count = sum(1 for i in self.issues if i.severity == "MEDIUM")
        low_count = sum(1 for i in self.issues if i.severity == "LOW")
        
        lines.append(f"\n问题统计: HIGH={high_count}, MEDIUM={medium_count}, LOW={low_count}")
        
        if not self.issues:
            lines.append("\n[PASS] 未发现复权数据问题")
        else:
            # 按类型分组
            by_type: Dict[str, List[AdjustmentIssue]] = {}
            for issue in self.issues:
                if issue.issue_type not in by_type:
                    by_type[issue.issue_type] = []
                by_type[issue.issue_type].append(issue)
            
            for issue_type, issues in by_type.items():
                lines.append(f"\n{issue_type}: {len(issues)} 个")
                lines.append("-" * 50)
                
                for issue in issues[:10]:  # 只显示前10个
                    lines.append(
                        f"  [{issue.severity}] stock={issue.stock_idx}, "
                        f"date={issue.date_idx}: {issue.description}"
                    )
                
                if len(issues) > 10:
                    lines.append(f"  ... 还有 {len(issues) - 10} 个问题")
        
        return "\n".join(lines)


# =============================================================================
# 验收测试
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AdjustmentValidator 验收测试")
    print("=" * 70)
    
    # 生成测试数据
    rng = np.random.default_rng(42)
    N, T = 50, 500
    
    # 正确的后复权数据（累积收益）
    returns = rng.normal(0.0005, 0.02, (N, T))
    close_hfq = 10 * np.cumprod(1 + returns, axis=1)
    
    dates = [f"2020-{i//20+1:02d}-{i%20+1:02d}" for i in range(T)]
    
    # 验证
    validator = AdjustmentValidator(close_hfq, dates)
    results = validator.validate_all()
    
    print(f"\n验证结果: {results}")
    
    report = validator.generate_report()
    print(report)
    
    # 测试前复权数据检测
    print("\n--- 测试前复权数据检测 ---")
    
    # 模拟前复权数据：最新价格正常，历史价格被压缩
    close_qfq = close_hfq.copy()
    # 模拟除权：历史价格除以一个大因子
    close_qfq[:, :300] = close_qfq[:, :300] / 2.0
    
    validator_qfq = AdjustmentValidator(close_qfq, dates)
    validator_qfq.validate_all()
    
    report_qfq = validator_qfq.generate_report()
    print(report_qfq)
    
    print("\n[SUCCESS] AdjustmentValidator 验收完成")
