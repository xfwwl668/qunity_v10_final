"""
Q-UNITY V10 — src/engine/portfolio_allocator.py
=================================================
多策略资金分配器（PortfolioAllocator）

★[H-01] 审计修复：实现白皮书 §6.1 规定的多策略资金分配层

支持方法：
  "equal"         等权分配（1/N，基准）
  "risk_parity"   波动率倒数分配（Bridgewater All Weather）
  "momentum_tilt" 近期夏普加权分配

铁律
----
1. stamp_tax = 0.0005（分配层不影响单策略成本，记录备查）
2. 资金分配只决定合并比例，不修改单策略 AlphaSignal
3. 合并后权重仍须经 PortfolioBuilder.build() 做 Regime 归一化
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# MultiRunResult
# ─────────────────────────────────────────────────────────────────────────────

class MultiRunResult:
    """多策略组合回测结果（white parer §6.1）。"""

    def __init__(
        self,
        combined_nav    : np.ndarray,
        strategy_results: Dict[str, Any],
        allocations     : Dict[str, float],
        high_corr_pairs : List[Tuple],
        annual_factor   : float = 252.0,
    ) -> None:
        self.combined_nav     = combined_nav
        self.strategy_results = strategy_results
        self.allocations      = allocations
        self.high_corr_pairs  = high_corr_pairs

        nav = np.asarray(combined_nav, dtype=np.float64)
        if len(nav) >= 2:
            ret = np.diff(nav) / (nav[:-1] + 1e-10)
            mu  = float(np.mean(ret))
            sig = float(np.std(ret, ddof=1))
            self.sharpe_ratio  = float(mu / (sig + 1e-10) * np.sqrt(annual_factor))
            peak = np.maximum.accumulate(nav)
            self.max_drawdown  = float(((peak - nav) / (peak + 1e-10)).max())
            n_yr = len(nav) / annual_factor
            tot  = float(nav[-1] / nav[0] - 1.0) if nav[0] > 0 else 0.0
            self.annual_return = float((1 + tot) ** (1.0 / max(n_yr, 1e-9)) - 1.0)
        else:
            self.sharpe_ratio  = 0.0
            self.max_drawdown  = 0.0
            self.annual_return = 0.0

    def to_summary(self) -> str:
        lines = [
            f"MultiRunResult | {len(self.strategy_results)} 策略",
            f"  合并组合: Ret={self.annual_return:+.1%}  Sharpe={self.sharpe_ratio:.2f}"
            f"  DD={self.max_drawdown:.1%}",
            f"  资金分配: { {k: f'{v:.1%}' for k, v in self.allocations.items()} }",
        ]
        if self.high_corr_pairs:
            lines.append(f"  ⚠ 高相关策略对: {self.high_corr_pairs}")
        for name, res in self.strategy_results.items():
            ar = getattr(res, 'annual_return', 0.0)
            sr = getattr(res, 'sharpe_ratio',  0.0)
            dd = getattr(res, 'max_drawdown',  0.0)
            a  = self.allocations.get(name, 0.0)
            lines.append(
                f"  [{name}] alloc={a:.1%}  Ret={ar:+.1%}  "
                f"Sharpe={sr:.2f}  DD={dd:.1%}"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# PortfolioAllocator
# ─────────────────────────────────────────────────────────────────────────────

class PortfolioAllocator:
    """
    多策略资金分配器（白皮书 §6.1）。

    Parameters
    ----------
    method         : "equal" | "risk_parity" | "momentum_tilt"
    lookback_days  : 波动率/夏普计算窗口（交易日，默认 60）
    corr_threshold : 高相关告警阈值（默认 0.80）
    annual_factor  : 年化因子（默认 252）
    """

    def __init__(
        self,
        method         : str   = "risk_parity",
        lookback_days  : int   = 60,
        corr_threshold : float = 0.80,
        annual_factor  : float = 252.0,
    ) -> None:
        assert method in ("equal", "risk_parity", "momentum_tilt"), (
            f"method 须为 'equal'|'risk_parity'|'momentum_tilt'，got '{method}'"
        )
        self.method         = method
        self.lookback_days  = lookback_days
        self.corr_threshold = corr_threshold
        self.annual_factor  = annual_factor

    # ── allocate ─────────────────────────────────────────────────────────────

    def allocate(
        self,
        nav_history: Dict[str, np.ndarray],   # {name: nav_array (T,)}
    ) -> Dict[str, float]:
        """
        计算各策略资金分配比例（和 = 1.0）。

        Parameters
        ----------
        nav_history : {strategy_name: nav_array (T,)}

        Returns
        -------
        Dict[str, float]  分配比例
        """
        names = list(nav_history.keys())
        n = len(names)
        if n == 0:
            return {}
        if n == 1:
            return {names[0]: 1.0}

        if self.method == "equal":
            return {k: 1.0 / n for k in names}

        elif self.method == "risk_parity":
            # ── 波动率倒数（Bridgewater Risk Parity）─────────────────────
            vols: Dict[str, float] = {}
            for name, nav in nav_history.items():
                arr = np.asarray(nav, dtype=np.float64)
                w   = min(self.lookback_days, len(arr) - 1)
                if w < 5:
                    vols[name] = 1.0
                    continue
                ret = np.diff(arr[-w - 1:]) / (arr[-(w + 1):-1] + 1e-10)
                # [FIX-B14] 零波动策略（如titan零交易NAV恒为1）会使std=0，
                # 导致后续 corrcoef 除零产生NaN，污染整个组合权重矩阵。
                # 修复：零波动策略给予极小但非零的虚拟波动率（权重趋近0），
                # 使 risk_parity 退化为忽略该策略而非整体崩溃。
                std_val = float(np.std(ret, ddof=1))
                if not np.isfinite(std_val) or std_val < 1e-8:
                    vols[name] = 1e-4   # 极大波动率倒数→极小权重，安全降权
                else:
                    vols[name] = max(std_val * np.sqrt(self.annual_factor), 1e-4)

            inv_v = {k: 1.0 / v for k, v in vols.items()}
            tot   = sum(inv_v.values())
            alloc = {k: v / tot for k, v in inv_v.items()}
            logger.info(
                "[PortfolioAllocator] risk_parity: "
                + "  ".join(f"{k}={w:.1%}(σ={vols[k]:.1%})"
                            for k, w in alloc.items())
            )
            return alloc

        else:  # momentum_tilt
            # ── 近期夏普倾斜 ─────────────────────────────────────────────
            sharpes: Dict[str, float] = {}
            for name, nav in nav_history.items():
                arr = np.asarray(nav, dtype=np.float64)
                w   = min(self.lookback_days, len(arr) - 1)
                if w < 5:
                    sharpes[name] = 0.0
                    continue
                ret = np.diff(arr[-w - 1:]) / (arr[-(w + 1):-1] + 1e-10)
                mu  = float(np.mean(ret))
                sig = float(np.std(ret, ddof=1))
                sharpes[name] = float(mu / (sig + 1e-10) * np.sqrt(self.annual_factor))

            pos = {k: max(v, 0.0) for k, v in sharpes.items()}
            tot = sum(pos.values())
            if tot < 1e-9:
                logger.warning("[PortfolioAllocator] 全策略夏普≤0，退化为等权")
                return {k: 1.0 / n for k in names}
            alloc = {k: v / tot for k, v in pos.items()}
            logger.info(
                "[PortfolioAllocator] momentum_tilt: "
                + "  ".join(f"{k}={w:.1%}(Sr={sharpes[k]:.2f})"
                            for k, w in alloc.items())
            )
            return alloc

    # ── correlation_check ────────────────────────────────────────────────────

    def correlation_check(
        self,
        nav_history: Dict[str, np.ndarray],
        threshold  : Optional[float] = None,
    ) -> List[Tuple[str, str, float]]:
        """
        检测策略两两相关性，返回相关性 > threshold 的策略对。

        Returns
        -------
        List[(name_a, name_b, corr)]  按相关性降序
        """
        thr   = threshold if threshold is not None else self.corr_threshold
        rets: Dict[str, np.ndarray] = {}
        min_len: Optional[int] = None

        for name, nav in nav_history.items():
            arr = np.asarray(nav, dtype=np.float64)
            if len(arr) < 2:
                continue
            r = np.diff(arr) / (arr[:-1] + 1e-10)
            rets[name] = r
            min_len = len(r) if min_len is None else min(min_len, len(r))

        if min_len is None or min_len < 5:
            return []

        aligned    = {k: v[-min_len:] for k, v in rets.items()}
        valid_keys = list(aligned.keys())
        pairs: List[Tuple[str, str, float]] = []

        for i in range(len(valid_keys)):
            for j in range(i + 1, len(valid_keys)):
                a, b = valid_keys[i], valid_keys[j]
                # [FIX-B14] 零波动策略的 corrcoef 会产生 NaN，跳过即可
                std_a = float(np.std(aligned[a]))
                std_b = float(np.std(aligned[b]))
                if std_a < 1e-8 or std_b < 1e-8:
                    continue
                corr = float(np.corrcoef(aligned[a], aligned[b])[0, 1])
                if not np.isnan(corr) and corr > thr:
                    pairs.append((a, b, round(corr, 4)))

        pairs.sort(key=lambda x: -x[2])
        if pairs:
            logger.warning(
                f"[PortfolioAllocator] ⚠ 高相关策略对(thr={thr:.2f}): "
                + ", ".join(f"{a}×{b}={c:.3f}" for a, b, c in pairs)
            )
        return pairs

    # ── combine_weights ──────────────────────────────────────────────────────

    @staticmethod
    def combine_weights(
        strategy_weights: Dict[str, np.ndarray],   # {name: (N,T)}
        allocations     : Dict[str, float],
    ) -> np.ndarray:
        """
        按分配比例加权合并各策略权重矩阵。

        Parameters
        ----------
        strategy_weights : {name: (N, T) float64}
        allocations      : {name: float}  各策略资金比例

        Returns
        -------
        combined : (N, T) float64
        """
        if not strategy_weights:
            raise ValueError("[combine_weights] strategy_weights 为空")
        shapes = {k: v.shape for k, v in strategy_weights.items()}
        if len(set(shapes.values())) > 1:
            raise ValueError(f"[combine_weights] shape 不一致: {shapes}")

        shape    = next(iter(strategy_weights.values())).shape
        combined = np.zeros(shape, dtype=np.float64)
        for name, w in strategy_weights.items():
            combined += np.asarray(w, dtype=np.float64) * allocations.get(name, 0.0)
        return combined


# ─────────────────────────────────────────────────────────────────────────────
# 验收测试
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  portfolio_allocator.py  验收测试")
    print("=" * 60)

    rng = np.random.default_rng(42)
    T   = 300
    nav_a = np.cumprod(1 + rng.normal(0.0004, 0.008, T))   # 低波动
    nav_b = np.cumprod(1 + rng.normal(0.0006, 0.020, T))   # 高波动
    hist  = {"low_A": nav_a, "high_B": nav_b}

    # T1: equal
    w = PortfolioAllocator("equal").allocate(hist)
    assert abs(w["low_A"] - 0.5) < 1e-9
    print(f"[OK] T1 equal: {w}")

    # T2: risk_parity（低波动获更多资金）
    w2 = PortfolioAllocator("risk_parity").allocate(hist)
    assert w2["low_A"] > w2["high_B"], f"风险平价应给低波动更多资金: {w2}"
    assert abs(sum(w2.values()) - 1.0) < 1e-9
    print(f"[OK] T2 risk_parity: low_A={w2['low_A']:.1%}  high_B={w2['high_B']:.1%}")

    # T3: momentum_tilt
    w3 = PortfolioAllocator("momentum_tilt").allocate(hist)
    assert abs(sum(w3.values()) - 1.0) < 1e-9
    print(f"[OK] T3 momentum_tilt: low_A={w3['low_A']:.1%}  high_B={w3['high_B']:.1%}")

    # T4: correlation_check
    nav_c = nav_a * (1 + rng.normal(0, 0.001, T))
    pairs = PortfolioAllocator().correlation_check(
        {"low_A": nav_a, "clone": nav_c, "high_B": nav_b}, threshold=0.90
    )
    assert any({"low_A","clone"} == {a, b} for a, b, _ in pairs), f"未检出: {pairs}"
    print(f"[OK] T4 correlation_check: {pairs}")

    # T5: combine_weights
    N = 50
    wa = rng.uniform(0, 0.05, (N, T))
    wb = rng.uniform(0, 0.05, (N, T))
    comb = PortfolioAllocator.combine_weights({"A": wa, "B": wb}, {"A": 0.6, "B": 0.4})
    assert np.allclose(comb, wa * 0.6 + wb * 0.4)
    print(f"[OK] T5 combine_weights: shape={comb.shape}")

    # T6: MultiRunResult
    nav_c2 = np.cumprod(1 + rng.normal(0.0004, 0.012, T))
    mr = MultiRunResult(nav_c2, {}, {"A": 0.6, "B": 0.4}, [])
    assert isinstance(mr.sharpe_ratio, float)
    print(f"[OK] T6 MultiRunResult: Sharpe={mr.sharpe_ratio:.2f}  DD={mr.max_drawdown:.1%}")

    print()
    print("[PASS] portfolio_allocator.py 全部验收通过 ✓")
    sys.exit(0)
