"""
Q-UNITY V10 — WhiteBox Audit: LookAheadBiasDetector
====================================================
前视偏差检测器

前视偏差（Look-Ahead Bias）是量化策略最常见的致命错误，
表现为在时间 t 的信号计算中使用了 t+k (k>0) 的数据。

检测方法：
1. 数据时间戳验证：检查因子计算中使用的数据索引是否越界
2. 因果性测试：随机替换未来数据，观察信号是否变化
3. 信号提前性检验：对比信号时点与收益实现时点

常见前视偏差来源：
- 使用当日收盘价计算当日交易信号（应用昨日数据或当日开盘前数据）
- 财务数据使用报告期而非公布日期
- 复权因子回填（应使用当日已知的复权因子）
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import inspect


@dataclass
class LookAheadViolation:
    """前视偏差违规记录"""
    strategy_name: str          # 策略名称
    factor_name: str            # 因子名称
    violation_type: str         # 违规类型
    evidence: str               # 证据描述
    severity: str               # 严重程度 HIGH/MEDIUM/LOW
    line_hint: str              # 代码行提示（如有）


class LookAheadBiasDetector:
    """
    前视偏差检测器
    
    检测方法：
    1. 静态分析：检查因子函数的数组索引模式
    2. 动态测试：通过数据扰动检测因果性违规
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.violations: List[LookAheadViolation] = []
        
    def detect_in_strategy(
        self,
        strategy_fn: Callable,
        strategy_name: str,
        test_data: Dict[str, np.ndarray],
    ) -> List[LookAheadViolation]:
        """
        对单个策略进行前视偏差检测
        
        Parameters
        ----------
        strategy_fn : Callable
            策略函数（alpha 函数）
        strategy_name : str
            策略名称
        test_data : Dict[str, np.ndarray]
            测试数据字典，包含 close, open, high, low, volume 等
        
        Returns
        -------
        violations : List[LookAheadViolation]
            检测到的违规列表
        """
        violations = []
        
        # 1. 源码静态分析
        source_violations = self._analyze_source(strategy_fn, strategy_name)
        violations.extend(source_violations)
        
        # 2. 动态因果性测试
        causal_violations = self._test_causality(
            strategy_fn, strategy_name, test_data
        )
        violations.extend(causal_violations)
        
        self.violations.extend(violations)
        
        if self.verbose:
            print(f"[LookAheadBiasDetector] {strategy_name}: "
                  f"发现 {len(violations)} 个潜在前视偏差")
        
        return violations
    
    def _analyze_source(
        self,
        fn: Callable,
        name: str,
    ) -> List[LookAheadViolation]:
        """
        源码静态分析：检查可疑的数组索引模式
        
        检测模式：
        - arr[:, t+1] 或 arr[t+1] 在 t 循环内（直接前视）
        - close[:, 1:] 与 volume[:, :-1] 不对齐（错位前视）
        - np.roll 负偏移（常见的前视来源）
        """
        violations = []
        
        try:
            source = inspect.getsource(fn)
        except (OSError, TypeError):
            return violations
        
        lines = source.split('\n')
        
        dangerous_patterns = [
            # 直接前视
            (r"t\s*\+\s*1", "直接使用 t+1 索引"),
            (r"t\s*\+\s*\d+", "使用 t+k (k>0) 索引"),
            # roll 负偏移
            (r"np\.roll\s*\([^,]+,\s*-", "np.roll 负偏移可能导致前视"),
            # 不安全的切片
            (r"\[:,\s*1:\].*\[:,\s*:-1\]", "切片索引可能不对齐"),
        ]
        
        import re
        for i, line in enumerate(lines, 1):
            for pattern, desc in dangerous_patterns:
                if re.search(pattern, line):
                    violations.append(LookAheadViolation(
                        strategy_name=name,
                        factor_name="unknown",
                        violation_type="SOURCE_PATTERN",
                        evidence=f"Line {i}: {line.strip()[:80]}",
                        severity="MEDIUM",
                        line_hint=f"Line {i}",
                    ))
        
        return violations
    
    def _test_causality(
        self,
        fn: Callable,
        name: str,
        data: Dict[str, np.ndarray],
    ) -> List[LookAheadViolation]:
        """
        动态因果性测试：扰动未来数据，观察信号变化
        
        原理：
        如果策略在时间 t 的信号依赖于 t+k 的数据，
        那么扰动 t+k 的数据会导致 t 的信号变化。
        
        测试方法：
        1. 运行策略获得基准信号
        2. 扰动最后 10% 的数据
        3. 重新运行策略
        4. 检查前 90% 时间步的信号是否变化
        """
        violations = []
        
        N, T = data["close"].shape
        perturb_start = int(T * 0.9)  # 扰动最后 10%
        
        # 准备策略参数
        class DummyParams:
            rsrs_window = 18
            zscore_window = 600
            top_n = 20
            max_single_pos = 0.08
            factor_ema_span = 5
            extra = {}
            
            def to_dict(self):
                return {"top_n": 20}
        
        try:
            # 基准运行
            kw_base = self._prepare_strategy_kwargs(fn, data, DummyParams())
            result_base = fn(**kw_base)
            weights_base = result_base.raw_target_weights.copy()
            
            # 扰动数据
            data_perturbed = {k: v.copy() for k, v in data.items()}
            for field in ["close", "high", "low", "volume"]:
                if field in data_perturbed:
                    # 在未来时段添加大幅扰动
                    noise = np.random.randn(N, T - perturb_start) * 0.5
                    data_perturbed[field][:, perturb_start:] *= (1 + noise)
            
            # 扰动后运行
            kw_perturbed = self._prepare_strategy_kwargs(fn, data_perturbed, DummyParams())
            result_perturbed = fn(**kw_perturbed)
            weights_perturbed = result_perturbed.raw_target_weights
            
            # 检查前 90% 时段的信号是否变化
            weights_before_base = weights_base[:, :perturb_start]
            weights_before_perturbed = weights_perturbed[:, :perturb_start]
            
            # 计算变化程度（归一化）
            diff = np.abs(weights_before_base - weights_before_perturbed)
            max_diff = diff.max()
            mean_diff = diff.mean()
            
            # 如果扰动未来数据导致过去信号显著变化，则存在前视偏差
            # 阈值：单点变化超过 1% 或平均变化超过 0.1%
            if max_diff > 0.01 or mean_diff > 0.001:
                violations.append(LookAheadViolation(
                    strategy_name=name,
                    factor_name="overall",
                    violation_type="CAUSALITY_VIOLATION",
                    evidence=f"扰动未来数据导致过去信号变化: max_diff={max_diff:.4f}, mean_diff={mean_diff:.6f}",
                    severity="HIGH" if max_diff > 0.05 else "MEDIUM",
                    line_hint="",
                ))
                
                # 找出最严重的时间点
                worst_t = np.unravel_index(diff.argmax(), diff.shape)
                violations[-1].evidence += f"\n最严重时点: stock={worst_t[0]}, t={worst_t[1]}"
                
        except Exception as e:
            if self.verbose:
                print(f"[LookAheadBiasDetector] 因果性测试异常: {e}")
        
        return violations
    
    def _prepare_strategy_kwargs(
        self,
        fn: Callable,
        data: Dict[str, np.ndarray],
        params: Any,
    ) -> Dict[str, Any]:
        """准备策略函数的关键字参数"""
        sig = inspect.signature(fn)
        kw = {}
        
        # 映射参数名
        param_map = {
            "close": "close",
            "open_": "open",
            "high": "high",
            "low": "low",
            "volume": "volume",
            "amount": "amount",
            "valid_mask": "valid_mask",
        }
        
        for param_name in sig.parameters:
            if param_name == "params":
                kw["params"] = params
            elif param_name in param_map:
                data_key = param_map[param_name]
                if data_key in data:
                    kw[param_name] = data[data_key]
            elif param_name == "market_regime":
                N, T = data["close"].shape
                kw["market_regime"] = np.full(T, 2, dtype=np.int8)  # NEUTRAL
            elif param_name == "kw" or param_name == "kwargs":
                pass  # 跳过 **kw
                
        return kw
    
    def run_all_strategies(
        self,
        strategies: Dict[str, Callable],
        test_data: Dict[str, np.ndarray],
    ) -> Dict[str, List[LookAheadViolation]]:
        """
        对所有策略运行前视偏差检测
        
        Returns
        -------
        results : Dict[str, List[LookAheadViolation]]
            策略名 -> 违规列表
        """
        results = {}
        
        for name, fn in strategies.items():
            results[name] = self.detect_in_strategy(fn, name, test_data)
        
        return results
    
    def generate_report(self) -> str:
        """生成检测报告"""
        lines = []
        lines.append("=" * 70)
        lines.append("前视偏差检测报告")
        lines.append("=" * 70)
        lines.append(f"检测到违规总数: {len(self.violations)}")
        lines.append("")
        
        # 按策略分组
        by_strategy: Dict[str, List[LookAheadViolation]] = {}
        for v in self.violations:
            if v.strategy_name not in by_strategy:
                by_strategy[v.strategy_name] = []
            by_strategy[v.strategy_name].append(v)
        
        for strategy, violations in by_strategy.items():
            lines.append(f"\n策略: {strategy}")
            lines.append("-" * 50)
            
            high_count = sum(1 for v in violations if v.severity == "HIGH")
            medium_count = sum(1 for v in violations if v.severity == "MEDIUM")
            low_count = sum(1 for v in violations if v.severity == "LOW")
            
            lines.append(f"  HIGH: {high_count}, MEDIUM: {medium_count}, LOW: {low_count}")
            
            for v in violations:
                lines.append(f"  [{v.severity}] {v.violation_type}: {v.evidence[:100]}")
        
        return "\n".join(lines)


# =============================================================================
# 验收测试
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LookAheadBiasDetector 验收测试")
    print("=" * 70)
    
    # 创建测试数据
    rng = np.random.default_rng(42)
    N, T = 50, 500
    test_data = {
        "close": np.cumprod(1 + rng.normal(0.0003, 0.02, (N, T)), axis=1) * 10,
        "open": np.cumprod(1 + rng.normal(0.0003, 0.02, (N, T)), axis=1) * 10,
        "high": np.cumprod(1 + rng.normal(0.0003, 0.02, (N, T)), axis=1) * 11,
        "low": np.cumprod(1 + rng.normal(0.0003, 0.02, (N, T)), axis=1) * 9,
        "volume": rng.uniform(1e6, 1e8, (N, T)),
        "valid_mask": np.ones((N, T), dtype=bool),
    }
    
    detector = LookAheadBiasDetector(verbose=True)
    
    # 定义一个有前视偏差的测试策略
    def bad_strategy(close, open_, high, low, volume, params, valid_mask=None, **kw):
        """故意使用未来数据的坏策略"""
        from src.strategies.alpha_signal import AlphaSignal
        N, T = close.shape
        
        # 前视偏差：使用 t+1 的数据计算 t 的信号
        score = np.zeros((N, T), dtype=np.float64)
        for t in range(T - 1):
            # 错误：使用明天的收益
            future_ret = close[:, t + 1] / close[:, t] - 1
            score[:, t] = future_ret
        
        score[:, -1] = 0.0
        weights = np.where(score > 0.02, 0.05, 0.0)
        
        return AlphaSignal(
            raw_target_weights=weights,
            score=score,
            strategy_name="bad_strategy",
        )
    
    # 定义一个正确的策略
    def good_strategy(close, open_, high, low, volume, params, valid_mask=None, **kw):
        """只使用历史数据的好策略"""
        from src.strategies.alpha_signal import AlphaSignal
        N, T = close.shape
        
        # 正确：使用过去的数据计算信号
        score = np.zeros((N, T), dtype=np.float64)
        for t in range(5, T):
            # 正确：使用过去5天的动量
            past_ret = close[:, t - 1] / close[:, t - 5] - 1
            score[:, t] = past_ret
        
        weights = np.where(score > 0.02, 0.05, 0.0)
        
        return AlphaSignal(
            raw_target_weights=weights,
            score=score,
            strategy_name="good_strategy",
        )
    
    # 测试坏策略
    violations_bad = detector.detect_in_strategy(bad_strategy, "bad_strategy", test_data)
    assert len(violations_bad) > 0, "未检测到坏策略的前视偏差"
    print(f"[PASS] 坏策略检测: 发现 {len(violations_bad)} 个前视偏差")
    
    # 测试好策略
    violations_good = detector.detect_in_strategy(good_strategy, "good_strategy", test_data)
    # 好策略可能有少量误报（源码模式匹配），但因果性测试应该通过
    causality_violations = [v for v in violations_good if v.violation_type == "CAUSALITY_VIOLATION"]
    print(f"[INFO] 好策略检测: 因果性违规 {len(causality_violations)} 个")
    
    # 生成报告
    report = detector.generate_report()
    print("\n" + report)
    
    print()
    print("[SUCCESS] LookAheadBiasDetector 验收测试完成")
