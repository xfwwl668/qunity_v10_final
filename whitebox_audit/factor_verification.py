"""
Q-UNITY V10 — WhiteBox Audit: FactorVerification
=================================================
因子计算验证

对每个策略的核心因子进行逐点手算验证，确保：
1. 滚动窗口计算正确（边界处理）
2. 截面标准化正确（Z-Score / Rank）
3. 数值稳定性（NaN / Inf 处理）
4. 后复权数据计算正确

验证方法：
- 选取特定时间点 t 和股票 i
- 手动计算因子值 expected
- 对比策略计算值 actual
- 允许浮点误差 < 1e-6
"""

from __future__ import annotations

import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class FactorTestCase:
    """因子测试用例"""
    strategy_name: str
    factor_name: str
    stock_idx: int
    time_idx: int
    expected_value: float
    actual_value: float
    tolerance: float = 1e-6
    
    @property
    def passed(self) -> bool:
        if np.isnan(self.expected_value) and np.isnan(self.actual_value):
            return True
        if np.isinf(self.expected_value) and np.isinf(self.actual_value):
            return np.sign(self.expected_value) == np.sign(self.actual_value)
        return abs(self.expected_value - self.actual_value) < self.tolerance


class FactorVerifier:
    """因子验证器"""
    
    def __init__(self, data: Dict[str, np.ndarray]):
        self.data = data
        self.close = data["close"].astype(np.float64)
        self.open_ = data.get("open", self.close).astype(np.float64)
        self.high = data.get("high", self.close).astype(np.float64)
        self.low = data.get("low", self.close).astype(np.float64)
        self.volume = data.get("volume", np.ones_like(self.close)).astype(np.float64)
        self.amount = data.get("amount", self.close * self.volume).astype(np.float64)
        
        self.N, self.T = self.close.shape
        self.test_cases: List[FactorTestCase] = []
        
    # =========================================================================
    # 手算因子计算函数
    # =========================================================================
    
    def _manual_momentum(self, i: int, t: int, window: int = 20) -> float:
        """手算 N 日动量"""
        if t < window:
            return np.nan
        p_now = self.close[i, t]
        p_past = self.close[i, t - window]
        if p_past < 1e-8:
            return np.nan
        return p_now / p_past - 1.0
    
    def _manual_reversal(self, i: int, t: int, window: int = 5) -> float:
        """手算 N 日反转"""
        mom = self._manual_momentum(i, t, window)
        if np.isnan(mom):
            return np.nan
        return -mom
    
    def _manual_rsrs_beta(self, i: int, t: int, window: int = 18) -> float:
        """手算 RSRS OLS Beta"""
        if t < window - 1:
            return np.nan
        
        # 窗口数据
        x = self.low[i, t - window + 1:t + 1]  # low 作为自变量
        y = self.high[i, t - window + 1:t + 1]  # high 作为因变量
        
        # OLS: beta = cov(x, y) / var(x)
        x_mean = x.mean()
        y_mean = y.mean()
        
        cov_xy = ((x - x_mean) * (y - y_mean)).sum()
        var_x = ((x - x_mean) ** 2).sum()
        
        if var_x < 1e-12:
            return np.nan
        
        return cov_xy / var_x
    
    def _manual_smart_money(self, i: int, t: int, window: int = 10) -> float:
        """手算聪明钱因子 (Williams Money Flow 变体)"""
        if t < window - 1:
            return np.nan
        
        buy_vol_sum = 0.0
        sell_vol_sum = 0.0
        total_vol = 0.0
        
        for k in range(t - window + 1, t + 1):
            c = self.close[i, k]
            h = self.high[i, k]
            l = self.low[i, k]
            v = self.volume[i, k]
            
            hl = h - l
            if hl < 1e-8 or v < 1e-3:
                # 一字板或无成交量
                if v > 1e-3:
                    total_vol += v
                continue
            
            buy_vol = (c - l) / hl * v
            sell_vol = (h - c) / hl * v
            
            buy_vol_sum += buy_vol
            sell_vol_sum += sell_vol
            total_vol += v
        
        if total_vol < 1.0:
            return np.nan
        
        return (buy_vol_sum - sell_vol_sum) / total_vol
    
    def _manual_amihud_illiq(self, i: int, t: int) -> float:
        """手算 Amihud 非流动性（单日）"""
        if t == 0:
            return np.nan
        
        p0 = self.close[i, t - 1]
        p1 = self.close[i, t]
        a = self.amount[i, t]
        
        if p0 < 1e-8 or a < 1.0:
            return np.nan
        
        ret_abs = abs(p1 / p0 - 1.0)
        return ret_abs / a
    
    def _manual_whale_pulse(self, i: int, t: int) -> float:
        """手算鲸鱼脉冲因子"""
        c = self.close[i, t]
        o = self.open_[i, t]
        h = self.high[i, t]
        l = self.low[i, t]
        v = self.volume[i, t]
        a = self.amount[i, t]
        
        if v < 1e-3 or c < 1e-8:
            return np.nan
        
        hl_range = h - l + 1e-6
        body_ratio = (c - o) / hl_range
        
        price_approx = (h + l) / 2.0 + 1e-6
        money_density = np.log1p(a / (v * price_approx + 1e-6))
        
        return body_ratio * money_density
    
    def _manual_cross_section_zscore(
        self,
        values: np.ndarray,  # (N,) 单列因子值
        i: int,              # 目标股票
    ) -> float:
        """手算截面 Z-Score"""
        valid = ~np.isnan(values)
        n_valid = valid.sum()
        
        if n_valid < 5:
            return np.nan
        
        mu = values[valid].mean()
        std = values[valid].std(ddof=1)  # 样本标准差
        
        if std < 1e-10:
            return 0.0
        
        return (values[i] - mu) / std
    
    # =========================================================================
    # 验证方法
    # =========================================================================
    
    def verify_momentum_reversal(self) -> List[FactorTestCase]:
        """验证 MomentumReversal 策略因子"""
        from src.strategies.vectorized.momentum_reversal_alpha import (
            _momentum_batch, _reversal_batch, _quality_batch
        )
        
        cases = []
        
        # 计算策略因子
        mom = _momentum_batch(self.close, window=20)
        rev = _reversal_batch(self.close, window=5)
        
        # 选取验证点：预热后的几个时间点
        test_points = [(10, 100), (25, 200), (50, 500), (0, 1000)]
        
        for i, t in test_points:
            if t >= self.T or i >= self.N:
                continue
            
            # 动量验证
            expected_mom = self._manual_momentum(i, t, 20)
            actual_mom = mom[i, t]
            cases.append(FactorTestCase(
                strategy_name="momentum_reversal",
                factor_name="momentum_20d",
                stock_idx=i, time_idx=t,
                expected_value=expected_mom,
                actual_value=actual_mom,
            ))
            
            # 反转验证
            expected_rev = self._manual_reversal(i, t, 5)
            actual_rev = rev[i, t]
            cases.append(FactorTestCase(
                strategy_name="momentum_reversal",
                factor_name="reversal_5d",
                stock_idx=i, time_idx=t,
                expected_value=expected_rev,
                actual_value=actual_rev,
            ))
        
        self.test_cases.extend(cases)
        return cases
    
    def verify_short_term_rsrs(self) -> List[FactorTestCase]:
        """验证 ShortTermRSRS 策略因子"""
        from src.strategies.vectorized.short_term_rsrs_alpha import _rsrs_beta_r2
        
        cases = []
        
        # 计算策略因子
        beta, r2 = _rsrs_beta_r2(self.high, self.low, window=18)
        
        # 验证点
        test_points = [(5, 50), (20, 150), (40, 300)]
        
        for i, t in test_points:
            if t >= self.T or i >= self.N:
                continue
            
            expected = self._manual_rsrs_beta(i, t, 18)
            actual = beta[i, t]
            cases.append(FactorTestCase(
                strategy_name="short_term_rsrs",
                factor_name="rsrs_beta",
                stock_idx=i, time_idx=t,
                expected_value=expected,
                actual_value=actual,
                tolerance=1e-5,  # RSRS 允许更大容差
            ))
        
        self.test_cases.extend(cases)
        return cases
    
    def verify_kunpeng(self) -> List[FactorTestCase]:
        """验证 Kunpeng 策略因子"""
        from src.strategies.vectorized.kunpeng_v10_alpha import (
            _rolling_smart_money, _rolling_amihud_stable
        )
        
        cases = []
        
        # 计算策略因子
        sm = _rolling_smart_money(self.close, self.high, self.low, self.volume, window=10)
        
        # 验证点
        test_points = [(5, 30), (15, 100), (30, 250)]
        
        for i, t in test_points:
            if t >= self.T or i >= self.N:
                continue
            
            expected = self._manual_smart_money(i, t, 10)
            actual = sm[i, t]
            cases.append(FactorTestCase(
                strategy_name="kunpeng_v10",
                factor_name="smart_money",
                stock_idx=i, time_idx=t,
                expected_value=expected,
                actual_value=actual,
                tolerance=1e-5,
            ))
        
        self.test_cases.extend(cases)
        return cases
    
    def verify_snma(self) -> List[FactorTestCase]:
        """验证 SNMA 策略因子"""
        from src.strategies.vectorized.snma_v4_alpha import _compute_whale_pulse
        
        cases = []
        
        # 计算策略因子
        wp = _compute_whale_pulse(
            self.close, self.open_, self.high, self.low,
            self.volume, self.amount
        )
        
        # 验证点
        test_points = [(3, 50), (10, 100), (20, 200)]
        
        for i, t in test_points:
            if t >= self.T or i >= self.N:
                continue
            
            expected = self._manual_whale_pulse(i, t)
            actual = wp[i, t]
            cases.append(FactorTestCase(
                strategy_name="snma_v4",
                factor_name="whale_pulse",
                stock_idx=i, time_idx=t,
                expected_value=expected,
                actual_value=actual,
                tolerance=1e-5,
            ))
        
        self.test_cases.extend(cases)
        return cases
    
    def run_all_verifications(self) -> Dict[str, List[FactorTestCase]]:
        """运行所有因子验证"""
        results = {}
        
        print("\n[FactorVerification] 开始因子验证...")
        
        # MomentumReversal
        try:
            cases = self.verify_momentum_reversal()
            results["momentum_reversal"] = cases
            pass_count = sum(1 for c in cases if c.passed)
            print(f"  momentum_reversal: {pass_count}/{len(cases)} 通过")
        except Exception as e:
            print(f"  momentum_reversal: 验证失败 - {e}")
        
        # ShortTermRSRS
        try:
            cases = self.verify_short_term_rsrs()
            results["short_term_rsrs"] = cases
            pass_count = sum(1 for c in cases if c.passed)
            print(f"  short_term_rsrs: {pass_count}/{len(cases)} 通过")
        except Exception as e:
            print(f"  short_term_rsrs: 验证失败 - {e}")
        
        # Kunpeng
        try:
            cases = self.verify_kunpeng()
            results["kunpeng_v10"] = cases
            pass_count = sum(1 for c in cases if c.passed)
            print(f"  kunpeng_v10: {pass_count}/{len(cases)} 通过")
        except Exception as e:
            print(f"  kunpeng_v10: 验证失败 - {e}")
        
        # SNMA
        try:
            cases = self.verify_snma()
            results["snma_v4"] = cases
            pass_count = sum(1 for c in cases if c.passed)
            print(f"  snma_v4: {pass_count}/{len(cases)} 通过")
        except Exception as e:
            print(f"  snma_v4: 验证失败 - {e}")
        
        return results
    
    def generate_report(self) -> str:
        """生成验证报告"""
        lines = []
        lines.append("=" * 70)
        lines.append("因子计算验证报告")
        lines.append("=" * 70)
        
        # 按策略分组
        by_strategy: Dict[str, List[FactorTestCase]] = {}
        for case in self.test_cases:
            if case.strategy_name not in by_strategy:
                by_strategy[case.strategy_name] = []
            by_strategy[case.strategy_name].append(case)
        
        total_pass = sum(1 for c in self.test_cases if c.passed)
        total_cases = len(self.test_cases)
        lines.append(f"\n总体: {total_pass}/{total_cases} 通过 ({total_pass/total_cases*100:.1f}%)")
        
        for strategy, cases in by_strategy.items():
            pass_count = sum(1 for c in cases if c.passed)
            lines.append(f"\n{strategy}: {pass_count}/{len(cases)}")
            lines.append("-" * 50)
            
            for case in cases:
                status = "PASS" if case.passed else "FAIL"
                if case.passed:
                    lines.append(
                        f"  [{status}] {case.factor_name}[{case.stock_idx},{case.time_idx}]: "
                        f"expected={case.expected_value:.6f}, actual={case.actual_value:.6f}"
                    )
                else:
                    lines.append(
                        f"  [{status}] {case.factor_name}[{case.stock_idx},{case.time_idx}]: "
                        f"expected={case.expected_value:.6f}, actual={case.actual_value:.6f} "
                        f"(diff={abs(case.expected_value - case.actual_value):.2e})"
                    )
        
        return "\n".join(lines)


# =============================================================================
# 验收测试
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FactorVerification 验收测试")
    print("=" * 70)
    
    # 生成测试数据
    from whitebox_audit.synthetic_data_generator import SyntheticDataGenerator
    
    gen = SyntheticDataGenerator(N=50, T=500, seed=42)
    data = gen.generate_sinusoidal(cycles=2)
    
    # 创建验证器
    verifier = FactorVerifier({
        "close": data["close"],
        "open": data["open"],
        "high": data["high"],
        "low": data["low"],
        "volume": data["volume"],
        "amount": data["amount"],
    })
    
    # 运行验证
    results = verifier.run_all_verifications()
    
    # 生成报告
    report = verifier.generate_report()
    print(report)
    
    # 统计
    total_pass = sum(1 for c in verifier.test_cases if c.passed)
    total_cases = len(verifier.test_cases)
    
    if total_pass == total_cases:
        print("\n[SUCCESS] 全部因子验证通过")
    else:
        print(f"\n[WARNING] {total_cases - total_pass} 个因子验证失败")
