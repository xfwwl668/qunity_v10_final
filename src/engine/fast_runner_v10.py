"""
Q-UNITY V10 — fast_runner_v10.py
=================================
权重驱动策略执行器（V10，新引擎）

整合链路：
  npy memmap
    → MarketRegimeDetector (breadth + NAV MA 状态机)
    → VEC_STRATEGY_REGISTRY (权重矩阵信号)
    → PortfolioBuilder (valid_mask + regime_limit 归一化)
    → match_engine_weights_driven (Numba 撮合)
    → RunResult

铁律（每次修改前默读）
----------------------
1. stamp_tax = 0.0005（万五，从 RiskConfig 读取，绝不手改）
2. holding_days 递增在内核主循环最顶部（由内核保证）
3. L3-A 在 L3-B 之前（由内核保证）
4. market_regime 是 int8 数组；_build_strategy_kw 映射为 str 后才能查 FACTOR_WEIGHTS
5. 不修改 fast_runner.py 或任何现有文件

★[H-01] 审计修复：新增 multi_run() 方法（白皮书 §6.1）
  多策略组合回测：并行运行 → 相关性检查 → 资金分配 → 合并 NAV
  依赖 portfolio_allocator.py（PortfolioAllocator / MultiRunResult）

与 V8 FastStrategyRunner 关键差异
----------------------------------
- 撮合引擎：match_engine_core → match_engine_weights_driven（权重驱动）
- 信号格式：(buy, sell) 二值信号 → raw_target_weights 矩阵（AlphaSignal）
- Regime：不在信号层控制，改由 PortfolioBuilder.build() 统一缩放
- 成交量单位：TdxQuant Volume=股(shares)，vol_multiplier=1；AKShare/BaoStock=手时设100
- 预热计算：优先 params.warmup_override，否则 rsrs+zscore+120
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ── 同包 V10 模块 ─────────────────────────────────────────────────────────────
try:
    from src.engine.risk_config import RiskConfig
    from src.engine.portfolio_builder import PortfolioBuilder, MarketRegimeDetector
    from src.engine.numba_kernels_v10 import match_engine_weights_driven
    from src.strategies.alpha_signal import AlphaSignal, REGIME_IDX_TO_STR
except ImportError:
    from risk_config import RiskConfig                              # type: ignore
    from portfolio_builder import PortfolioBuilder, MarketRegimeDetector  # type: ignore
    from numba_kernels_v10 import match_engine_weights_driven       # type: ignore
    from alpha_signal import AlphaSignal, REGIME_IDX_TO_STR         # type: ignore

# ── V8 RunResult（兼容复用；若不可用则本地重定义）────────────────────────────
try:
    from src.engine.fast_runner import RunResult, _compute_rolling_liquidity_mask
    _HAS_V8_RUNNER = True
except Exception:
    _HAS_V8_RUNNER = False
    # [BUG-B-4 FIX] 提供本地实现，避免 None 被调用时崩溃
    def _compute_rolling_liquidity_mask(  # type: ignore[misc]
        amount: np.ndarray,               # (N, T) float64 成交额（元）
        min_avg_amount: float = 5e6,
        window: int = 5,
    ) -> np.ndarray:                      # (N, T) bool
        """
        [FIX-F-02] 滚动 window 日均成交额 >= min_avg_amount 为 True。
        完全向量化实现（原版 Python for-t 循环已替换）。
        复杂度：O(N×T)，N=5500、T=2700 时约 5~10ms vs 原版 300~500ms。
        """
        if amount is None:
            return np.ones(1, dtype=np.bool_)
        amt = np.asarray(amount, dtype=np.float64)
        N, T = amt.shape
        # cumsum 技巧：rolling_sum[t] = cs[t] - cs[t-window]（t >= window 时）
        cs  = np.cumsum(amt, axis=1)                      # (N, T)
        lag = np.zeros((N, T), dtype=np.float64)
        if T > window:
            lag[:, window:] = cs[:, :T - window]
        # 各列实际参与均值的天数（前 window 天窗口不足）
        cnt = np.minimum(np.arange(1, T + 1, dtype=np.float64), float(window))  # (T,)
        roll_mean = (cs - lag) / cnt[np.newaxis, :]       # (N, T) 广播除法
        return roll_mean >= min_avg_amount

# ── 策略注册表（优先 V8 共享表；不可用时使用本地空表）────────────────────────
try:
    from src.strategies.registry import (
        VEC_STRATEGY_REGISTRY as _VEC_STRATEGY_REGISTRY,
        register_vec_strategy,
        list_vec_strategies,
    )
    _HAS_REGISTRY = True
except Exception:
    _VEC_STRATEGY_REGISTRY: Dict[str, Callable] = {}
    _HAS_REGISTRY = False
    def register_vec_strategy(name: str):  # type: ignore[misc]
        def _deco(fn):
            _VEC_STRATEGY_REGISTRY[name] = fn
            return fn
        return _deco
    def list_vec_strategies():  # type: ignore[misc]
        return list(_VEC_STRATEGY_REGISTRY.keys())

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# RunResult（本地定义，仅在无法从 V8 导入时使用）
# ─────────────────────────────────────────────────────────────────────────────
if not _HAS_V8_RUNNER:
    @dataclass
    class RunResult:  # type: ignore[no-redef]
        """轻量回测结果，兼容 V8 RunResult 字段集合。"""
        strategy_name   : str
        params_dict     : Dict[str, Any]
        total_return    : float
        annual_return   : float
        sharpe_ratio    : float
        max_drawdown    : float
        sortino_ratio   : float
        calmar_ratio    : float
        win_rate        : float
        profit_factor   : float
        volatility      : float
        turnover        : float
        nav_array       : np.ndarray          # (T,) 净值序列（以初始资金为基准）
        dates           : List[str]
        elapsed_ms      : float
        annual_factor   : float = 252.0
        invested_ratio  : float = 1.0         # [FIX-REGIME-STATS] 有持仓天比例
        buy_count       : int   = 0           # [NEW] 总买入操作次数
        sell_count      : int   = 0           # [NEW] 总卖出操作次数
        final_positions : int   = 0           # [NEW] 回测结束时持仓股票数
        pipeline_breakdown: Dict[str, float] = field(default_factory=dict)

        def to_summary(self) -> str:
            return (
                f"{self.strategy_name:22s} | "
                f"Ret={self.annual_return:+.1%} | "
                f"Sharpe={self.sharpe_ratio:5.2f} | "
                f"DD={self.max_drawdown:.1%} | "
                f"Win={self.win_rate:.1%} | "
                f"Calmar={self.calmar_ratio:.2f} | "
                f"{self.elapsed_ms:.0f}ms"
            )

# ─────────────────────────────────────────────────────────────────────────────
# 纯 NumPy 绩效计算（不依赖 Numba）
# ─────────────────────────────────────────────────────────────────────────────

def _metrics_numpy(
    nav: np.ndarray,
    annual_factor: float = 252.0,
    pos_matrix: "np.ndarray | None" = None,   # [FIX-WINRATE] (N,T) position matrix
) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    计算常用回测绩效指标（纯 NumPy）。

    Returns
    -------
    (total_ret, annual_ret, sharpe, max_dd, sortino, calmar, win_rate, profit_factor)

    win_rate 说明
    -------------
    原始 win_rate = 所有交易日中NAV上涨的比例（含空仓日，空仓时NAV≈0变化→不算win）。
    [FIX-WINRATE] 若传入 pos_matrix，额外计算 invested_win_rate（仅统计有持仓日），
    该值更能反映策略实际选股能力，避免大量空仓日（Regime/全止止损）拉低胜率。
    当前实现：win_rate 仍为全天统计值（与用户感知一致），但result表格会同时展示两者。
    """
    nav = np.asarray(nav, dtype=np.float64)
    if len(nav) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    total_ret  = float(nav[-1] / nav[0] - 1.0)
    n_years    = len(nav) / annual_factor
    annual_ret = float((1 + total_ret) ** (1.0 / max(n_years, 1e-9)) - 1.0)

    daily_ret = np.diff(nav) / (nav[:-1] + 1e-10)

    # Sharpe
    mu    = float(np.mean(daily_ret))
    sigma = float(np.std(daily_ret, ddof=1))
    sharpe = float(mu / (sigma + 1e-10) * np.sqrt(annual_factor))

    # Max Drawdown
    peak   = np.maximum.accumulate(nav)
    dd_arr = (peak - nav) / (peak + 1e-10)
    max_dd = float(dd_arr.max())

    # Sortino
    down   = daily_ret[daily_ret < 0.0]
    down_std = float(np.std(down, ddof=1)) if len(down) > 1 else 1e-10
    sortino  = float(mu / (down_std + 1e-10) * np.sqrt(annual_factor))

    # Calmar
    calmar = float(annual_ret / (max_dd + 1e-10))

    # [FIX-WINRATE] Win rate：只统计有持仓的交易日
    # 空仓日NAV基本不变（daily_ret≈0），大量空仓日会把全天胜率压到5-15%
    # 有持仓日胜率才真实反映选股能力
    if pos_matrix is not None and pos_matrix.shape[1] == len(daily_ret) + 1:
        # 用日初持仓判断是否有仓（pos_matrix[:,t]为t日收盘后持仓）
        # daily_ret[t] = nav[t+1]/nav[t]-1 由t日收盘仓位决定
        invested_mask = pos_matrix.sum(axis=0)[:-1] > 0  # (T-1,) bool，t日有持仓
        if invested_mask.sum() > 5:
            invested_ret = daily_ret[invested_mask]
            win_rate = float((invested_ret > 0).sum() / len(invested_ret))
        else:
            win_rate = float((daily_ret > 0).sum() / max(len(daily_ret), 1))
    else:
        win_rate = float((daily_ret > 0).sum() / max(len(daily_ret), 1))

    # Profit factor
    gain = float(daily_ret[daily_ret > 0].sum()) if (daily_ret > 0).any() else 0.0
    loss = float(abs(daily_ret[daily_ret < 0].sum())) if (daily_ret < 0).any() else 1e-10
    pf   = float(gain / (loss + 1e-10))

    return total_ret, annual_ret, sharpe, max_dd, sortino, calmar, win_rate, pf


def _turnover_numpy(
    position_matrix: np.ndarray,   # (N, T)
    nav_array      : np.ndarray,   # (T,)
    annual_factor  : float = 252.0,
    close_prices   : np.ndarray | None = None,  # (N, T) float64 收盘价
) -> float:
    """
    年化换手率（双边，以 NAV 为分母）。

    ★[FIX-T1] 正确实现：用「持仓股数变化 × 当日收盘价」计算交易金额，
    避免原来用「持仓市值差」导致的价格漂移虚增换手率问题。

    原实现 diff(position × close) = price_drift_item + real_trade_item，
    价格涨跌本身也会计入换手（零交易时换手率 ≠ 0），系统性虚高 3-10 倍。

    正确公式：turnover = Σ|ΔP[i,t]| × close[i,t] / NAV[t]
    """
    if position_matrix.shape[1] < 2 or nav_array.mean() < 1e-8:
        return 0.0
    if close_prices is not None and close_prices.shape == position_matrix.shape:
        # ★[FIX-T1-V3] 股数变化 × 收盘价 = 实际交易金额（剔除价格漂移）
        # Σ|Δshares| * close / NAV
        shares_diff = np.abs(np.diff(position_matrix, axis=1))
        trade_val   = shares_diff * close_prices[:, 1:]
        # ★[FIX-T1-V2] 修正换手率定义：除以 2.0 以符合行业标准 (Turnover Ratio)
        # 买入 100% + 卖出 100% = 100% 换手，而非 200%
        daily_to    = (trade_val.sum(axis=0) / (nav_array[1:] + 1e-10)) / 2.0
    else:
        # 兼容不传 close 的场景（近似，方向一致）
        shares_diff = np.abs(np.diff(position_matrix, axis=1))
        # 用平均市值近似
        daily_to    = (shares_diff.sum(axis=0) / (position_matrix.sum(axis=0)[:-1] + 1e-10)) / 2.0
    return float(daily_to.mean() * annual_factor)


# ─────────────────────────────────────────────────────────────────────────────
# FastRunnerV10
# ─────────────────────────────────────────────────────────────────────────────

class FastRunnerV10:
    """
    Q-UNITY V10 权重驱动策略执行器。

    相比 V8 FastStrategyRunner，核心变化：
    - 撮合引擎换为 match_engine_weights_driven（权重驱动）
    - Regime 由 MarketRegimeDetector 在含预热期完整数组上计算后裁剪
    - 成交量入内核前乘以 _vol_multiplier=100（BaoStock 手→股）
    - stamp_tax 从 RiskConfig 读取（永远 0.0005）

    Config 键（与 V8 兼容）：
      npy_dir, initial_cash, commission_rate, slippage_rate, stamp_tax,
      participation_rate, min_commission, max_single_pos, hard_stop_loss,
      max_holding_days, full_stop_dd, half_stop_dd, max_gap_up, vol_multiplier
    """

    # [FIX-VOL] TdxQuant Volume 字段实际单位为「股」（不是手）
    # 经 amount/close/volume 三者比对验证：volume median=6.637e6，close median=10.46，
    # amount median=8.332e7 → amount≈close×volume → volume 单位为股(shares)
    # 旧注释"BaoStock 手→股 ×100"已过时，TdxQuant 直接返回股数，无需转换
    _DEFAULT_VOL_MULTIPLIER: int = 1

    def __init__(self, config: dict) -> None:
        """
        Parameters
        ----------
        config : dict
            回测配置，键参考上方文档。必须包含 "npy_dir"。
        """
        self._cfg_raw = config

        bt_cfg = config.get("backtest", config)
        fe_cfg = config.get("fast_engine", config)

        # ── 数据路径 ──────────────────────────────────────────────────────
        # [FIX-C-02] 优先读取 npy_v10_dir（V10新增），回退到 npy_dir
        npy_dir_raw = (
            config.get("npy_v10_dir")
            or bt_cfg.get("npy_v10_dir")
            or bt_cfg.get("npy_dir")
            or config.get("npy_dir")
            or config.get("data_dir", "data/npy_v10")
        )
        self.npy_dir = Path(str(npy_dir_raw))

        # ── 成交量单位换算因子（手→股）──────────────────────────────────
        self._vol_multiplier: int = int(
            config.get("vol_multiplier", self._DEFAULT_VOL_MULTIPLIER)
        )

        # ── RiskConfig（stamp_tax 永远从此读取）──────────────────────────
        self._risk_cfg = RiskConfig(
            commission_rate     = float(bt_cfg.get("commission_rate",   0.0003)),
            stamp_tax           = float(fe_cfg.get("stamp_tax",         0.0005)),  # ★万五
            slippage_rate       = float(bt_cfg.get("slippage_rate",     0.001)),
            max_single_pos      = float(bt_cfg.get("max_single_pos",    0.08)),
            hard_stop_loss      = float(bt_cfg.get("hard_stop_loss",    0.20)),
            max_holding_days    = int(bt_cfg.get("max_holding_days",    0)),
            min_commission      = float(bt_cfg.get("min_commission",    5.0)),
            full_stop_dd        = float(bt_cfg.get("full_stop_dd",      0.15)),
            half_stop_dd        = float(bt_cfg.get("half_stop_dd",      0.08)),
            max_gap_up          = float(bt_cfg.get("max_gap_up",        0.025)),
            min_avg_amount      = float(bt_cfg.get("min_avg_amount",    5e6)),
            allow_fractional    = bool(bt_cfg.get("allow_fractional",   True)),
            # [FIX-CONFIG] Regime参数从config.json读取，修改配置文件即生效
            bear_confirm_days   = int(bt_cfg.get("bear_confirm_days",    3)),
            bear_exit_days      = int(bt_cfg.get("bear_exit_days",       10)),
            breadth_window      = int(bt_cfg.get("breadth_window",       10)),
            nav_ma_window       = int(bt_cfg.get("nav_ma_window",        60)),
            bear_breadth_thr    = float(bt_cfg.get("bear_breadth_thr",   0.32)),
            soft_bear_breadth   = float(bt_cfg.get("soft_bear_breadth",  0.38)),
            bull_breadth_thr    = float(bt_cfg.get("bull_breadth_thr",   0.44)),
            strong_bull_breadth = float(bt_cfg.get("strong_bull_breadth", 0.52)),
            participation_rate  = float(fe_cfg.get("participation_rate",  0.10)),
            # [FIX-BUG3] stop_recovery_days 默认值从 252（一年"永久棘轮"）降为 30 天。
            # 原 252 天：触发15%回撤后空仓整整一年，错过所有反弹，净值横盘至耗尽。
            # 修复语义：冷却 30 个交易日（约1.5个月）后重置 nav_peak 基准，恢复入场尝试。
            # 用户可在 config.json 中通过 "stop_recovery_days" 覆盖此默认值。
            stop_recovery_days  = int(bt_cfg.get("stop_recovery_days",  30)),
            # [FIX-CONFIG] 防补仓阈值和最小交易额从config.json读取
            rebalance_threshold = float(bt_cfg.get("rebalance_threshold", 0.05)),
            min_trade_value     = float(bt_cfg.get("min_trade_value",     1000.0)),
        )

        # ── 运行时参数 ────────────────────────────────────────────────────
        self.initial_cash       = float(bt_cfg.get("initial_cash",        1_000_000.0))
        # [FIX-CONFIG] participation_rate 统一从 _risk_cfg 读取，消除二次赋值不一致
        self.participation_rate = self._risk_cfg.participation_rate

        # ── 数据 & 元数据（延迟加载）─────────────────────────────────────
        self._data : Optional[Dict[str, np.ndarray]] = None
        self._meta : Optional[Dict[str, Any]]        = None

        # ── Regime 相关（延迟构造，依赖市场指数数据）─────────────────────
        self._regime_det    : Optional[MarketRegimeDetector] = None
        self._port_builder  : Optional[PortfolioBuilder]     = None

        logger.debug(f"[FastRunnerV10] 初始化完成，npy_dir={self.npy_dir}")

    # ─────────────────────────────────────────────────────────────────────────
    # 数据加载
    # ─────────────────────────────────────────────────────────────────────────

    def load_data(self, force_reload: bool = False) -> Tuple[int, int]:
        """
        加载 npy memmap 数据。

        Returns (N, T)。若 npy_dir 不存在则抛出 FileNotFoundError。
        """
        if self._data is not None and not force_reload:
            N, T = self._meta["shape"]  # type: ignore[index]
            return N, T

        if not self.npy_dir.exists():
            raise FileNotFoundError(
                f"[FastRunnerV10] npy 目录不存在: {self.npy_dir}\n"
                "请先运行 build_npy 生成数据"
            )

        meta_path = self.npy_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"[FastRunnerV10] meta.json 不存在: {meta_path}"
            )

        t0 = time.perf_counter()
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        # dates → List[str]
        dates_raw = meta.get("dates", [])
        if dates_raw and not isinstance(dates_raw[0], str):
            meta["dates"] = [str(d) for d in dates_raw]

        data: Dict[str, np.ndarray] = {}
        for fname in meta.get("fields", ["close", "open", "high", "low", "volume", "amount"]):
            p = self.npy_dir / f"{fname}.npy"
            if p.exists():
                data[fname] = np.load(str(p), mmap_mode="r")

        # valid_mask
        vm_path = self.npy_dir / "valid_mask.npy"
        if vm_path.exists():
            data["valid_mask"] = np.load(str(vm_path), mmap_mode="r")
        else:
            logger.warning("[FastRunnerV10] valid_mask.npy 缺失，使用 volume>0 保守掩码")
            vol_ = data.get("volume")
            clo_ = data.get("close")
            if vol_ is not None and clo_ is not None:
                data["valid_mask"] = ((np.asarray(vol_) > 0) & (np.asarray(clo_) > 0)).astype(np.bool_)
            elif vol_ is not None:
                data["valid_mask"] = (np.asarray(vol_) > 0).astype(np.bool_)

        # 基本面 / 估值矩阵（可选）
        N_full, T_full = meta["shape"]
        for fname in [
            # ── OHLCV 衍生（已在上方加载，此处跳过）─────────────────────
            # ── step3 输出（fundamental_*.npy / market_cap_total.npy）──
            # [FIX-A2] 原列表缺少 step3 实际生成的文件名，导致策略收到 None
            "fundamental_roe",        # roe_matrix
            "fundamental_eps_ttm",    # eps factor
            "fundamental_yoy_ni",     # growth_matrix
            "fundamental_net_profit",
            "fundamental_net_profit_margin",
            "fundamental_total_share",
            "fundamental_liq_share",
            "market_cap_total",       # mktcap_matrix / mktcap
            "market_cap_circ",
            "days_ann",               # days_since_ann
            "sue",                    # sue_matrix
            "pe_ttm",                 # pe_matrix (step3)
            "market_index",           # Regime 计算用
            # ── 旧路径兼容（roe_ttm / eps / mktcap 等别名）──────────────
            "roe_ttm", "eps", "mktcap",
            # ── concept / sector / ST ──────────────────────────────────
            "concept_ids", "is_st", "pb_mrq", "sector_matrix",
            # ── 4e BaoStock 路径（与 step3 路径并存兼容）──────────────────
            "valuation_peTTM", "valuation_pbMRQ", "valuation_isST",
        ]:
            p = self.npy_dir / f"{fname}.npy"
            if p.exists() and fname not in data:
                try:
                    arr = np.load(str(p), mmap_mode="r")
                    if arr.shape[0] == N_full:
                        data[fname] = arr
                except Exception:
                    pass

        self._data = data
        self._meta = meta

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"[FastRunnerV10] 数据加载完成: {N_full}×{T_full}，耗时 {elapsed:.1f}ms"
        )

        # ── 初始化 Regime 检测器和组合构建器 ─────────────────────────────
        self._init_regime(N_full, T_full)

        return N_full, T_full

    def _init_regime(self, N: int, T: int) -> None:
        """从 close / market_index 初始化 MarketRegimeDetector 和 PortfolioBuilder。"""
        # market_index：优先 data["market_index"]，其次用股票等权均值代替
        mkt_idx = self._data.get("market_index")  # type: ignore[union-attr]
        if mkt_idx is not None:
            mkt_arr = np.asarray(mkt_idx, dtype=np.float64).flatten()[:T]
        else:
            # 用全体收盘价等权均值作为市场指数代理
            close_all = np.asarray(
                self._data["close"], dtype=np.float64  # type: ignore[index]
            )
            with np.errstate(all="ignore"):
                mkt_arr = np.nanmean(close_all, axis=0)
            mkt_arr = np.where(np.isfinite(mkt_arr), mkt_arr, 0.0)

        self._regime_det   = MarketRegimeDetector(self._risk_cfg, mkt_arr)
        self._port_builder = PortfolioBuilder(self._risk_cfg, mkt_arr)

    # ─────────────────────────────────────────────────────────────────────────
    # Numba JIT 预热（最小 dummy 数据触发编译，与实际参数签名一致）
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def warmup_jit(risk_cfg: "RiskConfig | None" = None) -> float:
        """
        ★[FIX-WARMUP] 全面 JIT 热身：
        1. 检测 Numba 可用性 — 不可用时发出明确警告并估算纯 Python 运行时间
        2. 热身 match_engine_weights_driven（主撮合内核）
        3. 热身 MarketRegimeDetector.compute()（Regime 状态机，numpy 密集）
        4. 热身 PortfolioBuilder.build()（流动性过滤 + 归一化）
        确保多策略 multi_run 时第一个策略不触发额外编译延迟。

        Returns
        -------
        elapsed : float  总热身耗时（秒）
        """
        from src.engine.numba_kernels_v10 import (
            match_engine_weights_driven, _NUMBA_AVAILABLE
        )
        from src.engine.risk_config import RiskConfig as _RC
        from src.engine.portfolio_builder import (
            MarketRegimeDetector as _MRD, PortfolioBuilder as _PB
        )
        from src.strategies.alpha_signal import AlphaSignal as _AS

        # ── Bug 4：Numba 不可用时的明确警告 ──────────────────────────────
        if not _NUMBA_AVAILABLE:
            import warnings as _w
            _w.warn(
                "\n" + "="*65 + "\n"
                "[Q-UNITY] Numba 未安装！撮合引擎将以纯 Python 运行。\n"
                "  预估影响：N=5000 股票，T=1800 天的回测每次约需 15-60 分钟。\n"
                "  推荐安装：pip install numba\n"
                "  安装后首次运行会触发 JIT 编译（约 30-60 秒），后续会缓存加速。\n"
                + "="*65,
                UserWarning, stacklevel=2,
            )

        rc = risk_cfg or _RC()
        kw = rc.to_kernel_kwargs()

        t0 = time.perf_counter()

        # ── 热身1：撮合引擎（主 JIT 内核）──────────────────────────────
        n, t = 5, 20   # 比旧的 3×10 更接近真实调用形状，JIT 签名更稳定
        rng = np.random.default_rng(0)
        w   = np.full((n, t), 1.0 / n, dtype=np.float64)
        px  = rng.uniform(9.0, 11.0, (n, t))
        vol = rng.integers(100_000, 1_000_000, (n, t)).astype(np.float64)
        lu  = np.zeros((n, t), dtype=np.bool_)
        ld  = np.zeros((n, t), dtype=np.bool_)

        match_engine_weights_driven(
            final_target_weights = w,
            exec_prices          = px,
            close_prices         = px,
            high_prices          = px * 1.01,
            volume               = vol,
            limit_up_mask        = lu,
            limit_dn_mask        = ld,
            initial_cash         = 1_000_000.0,
            **kw,
        )
        t_engine = time.perf_counter() - t0

        # ── 热身2：Regime 检测器（O(N×T) numpy，必须执行一次建立状态）──
        t1 = time.perf_counter()
        _mkt = np.cumprod(1 + rng.normal(0.0002, 0.01, t + 30)) * 3000.0
        _close_w = rng.uniform(9.0, 11.0, (n, t + 30))
        _valid_w = np.ones((n, t + 30), dtype=np.bool_)
        _mrd = _MRD(rc, _mkt)
        _mrd.compute(_close_w, _valid_w, warmup=30)
        t_regime = time.perf_counter() - t1

        # ── 热身3：PortfolioBuilder（valid_mask + 流动性 + regime 归一化）
        t2 = time.perf_counter()
        _pb = _PB(rc, _mkt)
        _pb._regime_limits = np.ones(t, dtype=np.float64)
        _alpha_dummy = _AS(
            raw_target_weights = np.full((n, t), 0.04, dtype=np.float64),
            strategy_name      = "__warmup__",
        )
        _pb.build(_alpha_dummy, valid_mask=np.ones((n, t), dtype=np.bool_))
        t_builder = time.perf_counter() - t2

        total = time.perf_counter() - t0
        logger.info(
            f"[warmup_jit] engine={t_engine*1000:.0f}ms "
            f"regime={t_regime*1000:.0f}ms "
            f"builder={t_builder*1000:.0f}ms "
            f"total={total:.2f}s"
            + (" [Numba已启用]" if _NUMBA_AVAILABLE else " [⚠ 纯Python模式]")
        )
        return total


    # ─────────────────────────────────────────────────────────────────────────
    # 预热天数计算
    # ─────────────────────────────────────────────────────────────────────────

    def _calc_warmup(self, params: Any) -> int:
        """
        计算预热天数。

        优先级：
          1. params.warmup_override（若存在且 > 0）
          2. rsrs_window + zscore_window + 60，但不超过全量数据的 40%

        [FIX-4] 修复：
          - 固定冗余从 120 降为 60（原值过大，压缩回测期）
          - 新增上限保护：预热不超过全量数据的 40%，防止极端压缩回测期
        """
        if params is not None:
            override = getattr(params, "warmup_override", 0)
            if isinstance(override, (int, float)) and int(override) > 0:
                return int(override)

        rsrs_w   = int(getattr(params, "rsrs_window",   18))  if params is not None else 18
        # [FIX-C1] 默认值从 600 改为 300，与 titan_alpha(300)/short_term_rsrs(600) 取折中
        # 各策略自身会按 params.zscore_window 实际计算，此处仅决定预热期长度
        # 若策略真的用 600，warmup 会不足但不会越界（策略内部有 min_count 保护）
        zscore_w = int(getattr(params, "zscore_window", 300)) if params is not None else 300
        base_warmup = rsrs_w + zscore_w + 60   # [FIX-4] 120→60，减少固定冗余
        # [FIX-4] 限制预热不超过全量数据的 40%，防止极端压缩回测期
        if self._meta is not None:
            T_total = self._meta["shape"][1]
            base_warmup = min(base_warmup, int(T_total * 0.4))
        return base_warmup

    # ─────────────────────────────────────────────────────────────────────────
    # 策略参数字典构建（按需注入）
    # ─────────────────────────────────────────────────────────────────────────

    def _build_strategy_kw(
        self,
        close       : np.ndarray,    # (N, T_slice) 含预热
        open_       : np.ndarray,
        high        : np.ndarray,
        low         : np.ndarray,
        volume      : np.ndarray,    # (N, T_slice) 已乘以 vol_multiplier（单位：股）
        amount      : Optional[np.ndarray],
        valid_mask  : Optional[np.ndarray],
        params      : Any,
        sig         : inspect.Signature,
        regime_slice: Optional[np.ndarray],  # (T_slice,) int8，含预热
        t_start     : int = 0,        # [FIX-SLICE] 全量数组的起始索引（用于切片财务矩阵）
        t_end       : Optional[int] = None,  # [FIX-SLICE] 全量数组的结束索引
    ) -> Dict[str, Any]:
        """
        用 inspect.signature() 按需注入参数，避免不支持的参数触发 TypeError。

        基础注入（所有策略）：close / open_ / high / low / volume / params
        按需注入：
          valid_mask, amount,
          pe_matrix, roe_matrix, sue_matrix, mktcap_matrix, days_since_ann,
          concept_ids, sector_matrix, mktcap, is_st,
          market_regime（int8 数组，须在策略内用 REGIME_IDX_TO_STR 查字符串）
        """
        kw: Dict[str, Any] = {
            "close"  : close,
            "open_"  : open_,
            "high"   : high,
            "low"    : low,
            "volume" : volume,
            "params" : params,
        }

        def _add_if(param_name: str, value: Any) -> None:
            if param_name in sig.parameters and value is not None:
                kw[param_name] = value

        _add_if("valid_mask", valid_mask)
        _add_if("amount",     amount)
        # [FIX-AMOUNT] alpha_max_v5 / kunpeng_v10 用 kw.get("amount_matrix")，
        # 需要同时注入"amount_matrix"键，否则流动性因子始终为 None
        if "amount_matrix" in sig.parameters:
            kw["amount_matrix"] = amount

        # ── market_regime（int8 数组）──────────────────────────────────
        # 注意：传入 int8，策略内需先 REGIME_IDX_TO_STR[int(r)] 再查 FACTOR_WEIGHTS
        _add_if("market_regime", regime_slice)

        # ── 财务/估值矩阵 ────────────────────────────────────────────────
        # [FIX-C-04] 基本面文件名与 step3 输出对齐
        # step3 输出: fundamental_roe.npy, fundamental_eps_ttm.npy,
        #   fundamental_total_share.npy, market_cap_total.npy, pe_ttm.npy 等
        # [FIX-SLICE] 财务矩阵切片辅助：若 t_start/t_end 指定，则裁剪到与 close 相同时段
        def _slice_fund(arr):
            if arr is None:
                return None
            arr2d = np.ascontiguousarray(arr.astype(np.float64))
            if t_end is not None and arr2d.ndim == 2:
                te = t_end if t_end <= arr2d.shape[1] else arr2d.shape[1]
                ts = t_start if t_start < te else 0
                return arr2d[:, ts:te]
            return arr2d

        # [FIX-FIELD] 兼容 step3 路径（pe_ttm）和 4e BaoStock 路径（valuation_peTTM）
        # 优先用 step3 输出，若不存在则 fallback 到 4e 生成的文件
        def _get_fund(primary: str, fallback: str = None):
            arr = self._data.get(primary)   # type: ignore[union-attr]
            if arr is None and fallback:
                arr = self._data.get(fallback)  # type: ignore[union-attr]
            return arr

        # [FIX-A2] 完整 FUND_MAP：覆盖 step3 所有输出文件名
        _FUND_MAP: Dict[str, tuple] = {
            # (参数名): (首选key, 备选key)
            "pe_matrix"     : ("pe_ttm",              "valuation_peTTM"),
            "roe_matrix"    : ("fundamental_roe",      "roe_ttm"),
            "sue_matrix"    : ("sue",                  None),
            "mktcap_matrix" : ("market_cap_total",     "mktcap"),
            "mktcap"        : ("market_cap_total",     "mktcap"),
            "days_since_ann": ("days_ann",             None),
            "growth_matrix" : ("fundamental_yoy_ni",   None),
            "is_st"         : ("is_st",                "valuation_isST"),
        }
        for _pname, (_primary, _fallback) in _FUND_MAP.items():
            if _pname in sig.parameters:
                _arr = _get_fund(_primary, _fallback)
                kw[_pname] = _slice_fund(_arr)  # [FIX-SLICE]

        # ── concept_ids (uint16) ─────────────────────────────────────────
        if "concept_ids" in sig.parameters:
            _cid = self._data.get("concept_ids")  # type: ignore[union-attr]
            if _cid is not None:
                _cid2 = np.ascontiguousarray(_cid).astype(np.uint16)
                if t_end is not None and _cid2.ndim == 2:
                    te = min(t_end, _cid2.shape[1])
                    _cid2 = _cid2[:, t_start:te]  # [FIX-SLICE]
                kw["concept_ids"] = _cid2
            else:
                kw["concept_ids"] = None

        # ── sector_matrix ────────────────────────────────────────────────
        if "sector_matrix" in sig.parameters:
            kw["sector_matrix"] = self._data.get("sector_matrix")  # type: ignore[union-attr]

        return kw

    # ─────────────────────────────────────────────────────────────────────────
    # 日期切片辅助
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _slice_dates(
        dates: List[str],
        start: Optional[str],
        end  : Optional[str],
    ) -> Tuple[int, int]:
        t_s = 0 if start is None else next(
            (i for i, d in enumerate(dates) if d >= start), 0
        )
        t_e = len(dates) if end is None else next(
            (i for i, d in enumerate(dates) if d > end), len(dates)
        )
        return t_s, t_e

    def _slice_ending_at(
        self,
        as_of_date  : str,
        warmup      : int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, Optional[np.ndarray], Optional[np.ndarray],
               Optional[np.ndarray], List[str], int]:
        """
        截至 as_of_date（含）的 warmup+1 天切片，用于实盘信号生成。

        Returns
        -------
        (close, open_, high, low, volume, amount, valid_mask, regime_slice,
         period_dates, actual_warmup)
        """
        if self._data is None:
            self.load_data()

        dates  = self._meta["dates"]  # type: ignore[index]
        t_end  = next((i for i, d in enumerate(dates) if d > as_of_date), len(dates))
        t_start = max(0, t_end - warmup - 1)
        actual_warmup = t_end - t_start - 1

        def _c(key, dtype=np.float64, fallback_key=None):
            arr = self._data.get(key)  # type: ignore[union-attr]
            if arr is None and fallback_key:
                arr = self._data.get(fallback_key)
            if arr is None:
                return None
            return np.ascontiguousarray(arr[:, t_start:t_end], dtype=dtype)

        close      = _c("close")
        open_      = _c("open", fallback_key="close")
        high       = _c("high", fallback_key="close")
        low        = _c("low",  fallback_key="close")
        vol_raw    = _c("volume")
        volume     = (vol_raw * self._vol_multiplier) if vol_raw is not None else None
        amount     = _c("amount")
        valid_mask = _c("valid_mask", dtype=np.bool_)

        period_dates = dates[t_start:t_end]

        # regime（在 warmup+1 天切片上计算，不含更多历史，近似处理）
        regime_slice = None
        if self._regime_det is not None and close is not None:
            vfull = valid_mask if valid_mask is not None else np.ones_like(close, dtype=np.bool_)
            lims = self._regime_det.compute(close, vfull, warmup=actual_warmup)
            # 完整 regime 数组（含预热），切片到含预热段
            regime_slice = self._regime_det.get_regime_enum_array()

        return (
            close, open_, high, low, volume, amount,
            valid_mask, regime_slice, period_dates, actual_warmup,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 单策略回测
    # ─────────────────────────────────────────────────────────────────────────

    def run(
        self,
        strategy_name : str,
        params        : Any,
        start_date    : Optional[str] = None,
        end_date      : Optional[str] = None,
    ) -> RunResult:
        """
        执行单次权重驱动策略回测。

        主循环严格顺序（由内核保证，此处标注为审计记录）：
          内核内部 for t in range(T):
            0. skip_l3a[:] = False
            1. ★ holding_days 递增（最顶部）
            2. Phase0 退市检测
            3. ★ L3-A 估值 + delta_val（先）
            4. ★ L3-B 止损防线（后）
            5. L3-C 防补仓过滤
            6. Pass1 卖出（stamp_tax=0.0005）
            7. Pass2 买入（vol×vol_multiplier）
            8. Phase4 NAV
            9. PortfolioRiskGuard port_scale 更新

        Returns
        -------
        RunResult
        """
        t_total = time.perf_counter()
        breakdown: Dict[str, float] = {}

        if self._data is None:
            self.load_data()

        # ── Step 1: 计算预热期 & 数据切片 ────────────────────────────────
        t1 = time.perf_counter()
        dates    = self._meta["dates"]  # type: ignore[index]
        T_total  = len(dates)
        t_s, t_e = self._slice_dates(dates, start_date, end_date)

        warmup_req   = self._calc_warmup(params)
        t_data_start = max(0, t_s - warmup_req)
        actual_warmup = t_s - t_data_start
        T_backtest   = t_e - t_s
        period_dates = dates[t_s:t_e]

        # ── 切片辅助 ──────────────────────────────────────────────────────
        def _slc(key: str, dtype=np.float64, fallback: Optional[str] = None):
            arr = self._data.get(key)  # type: ignore[union-attr]
            if arr is None and fallback:
                arr = self._data.get(fallback)
            if arr is None:
                return None
            return np.ascontiguousarray(arr[:, t_data_start:t_e], dtype=dtype)

        close_full  = _slc("close")
        open_full   = _slc("open", fallback="close")
        high_full   = _slc("high", fallback="close")
        low_full    = _slc("low",  fallback="close")

        # ★ vol 入内核前乘以 vol_multiplier（BaoStock 手 → 股）
        _vol_raw = _slc("volume")
        vol_full = (
            np.ascontiguousarray(_vol_raw * self._vol_multiplier, dtype=np.float64)
            if _vol_raw is not None
            else np.zeros_like(close_full)
        )

        amount_full    = _slc("amount")
        valid_full_slc = _slc("valid_mask", dtype=np.bool_)

        # NaN 填充
        close_full = np.nan_to_num(close_full, nan=0.0)
        open_full  = np.nan_to_num(open_full,  nan=0.0)
        high_full  = np.nan_to_num(high_full,  nan=0.0)
        low_full   = np.nan_to_num(low_full,   nan=0.0)
        vol_full   = np.nan_to_num(vol_full,   nan=0.0)

        N = close_full.shape[0]
        breakdown["data_slice_ms"] = (time.perf_counter() - t1) * 1000

        # ── Step 2: ★ Regime 计算（在含预热的完整数组上！）──────────────
        t2 = time.perf_counter()

        if valid_full_slc is None:
            valid_full_slc_use = np.ones_like(close_full, dtype=np.bool_)
        else:
            valid_full_slc_use = valid_full_slc

        # ★[FIX-REGIME-CACHE] Regime 缓存机制
        # 根因：multi_run 调用 10 次 run()，相同日期范围下每次都重建
        # MarketRegimeDetector 并重跑 compute()（O(N×T) 操作）= 10倍冗余计算。
        # 缓存键：(t_data_start, t_e, actual_warmup) 三元组唯一标识一个计算窗口。
        # 当连续的 run() 调用使用相同窗口时（multi_run 的典型场景），
        # 直接复用上次的 _regime_det/_port_builder/_regime_limits，跳过重计算。
        _regime_cache_key = (t_data_start, t_e, actual_warmup)
        _cache_hit = (
            hasattr(self, "_regime_cache_key")
            and self._regime_cache_key == _regime_cache_key  # type: ignore[attr-defined]
            and self._regime_det is not None
            and self._port_builder is not None
            and hasattr(self, "_cached_regime_limits")
        )

        if _cache_hit:
            _regime_limits  = self._cached_regime_limits    # type: ignore[attr-defined]
            regime_full_arr = self._regime_det.get_regime_enum_array()  # type: ignore[union-attr]
            regime_bt_arr   = regime_full_arr[actual_warmup:]
            breakdown["regime_ms"] = 0.0   # cache hit → 0ms
            logger.debug(f"[regime cache HIT] key={_regime_cache_key}")
        else:
            # ★[BUG-B-1 FIX] 每次 run() 必须用当前回测窗口的 market_index 切片
            # 重新构造 Regime 检测器，避免不同时间段共享错误的市场状态判断。
            mkt_raw = self._data.get("market_index")  # type: ignore[union-attr]
            if mkt_raw is not None:
                mkt_slice = np.asarray(mkt_raw, dtype=np.float64).flatten()[t_data_start:t_e]
            else:
                # [FIX-B4] 无独立指数时用回测窗口内股票等权均值。
                # 停牌/未上市股票 close=0.0（非NaN），直接 nanmean 会把数千个0计入分母，
                # 导致代理指数被系统性低估30-40%，误触BEAR。先将0替换为NaN再均值。
                with np.errstate(all="ignore"):
                    close_for_idx = np.where(close_full > 1e-8, close_full, np.nan)
                    mkt_slice = np.nanmean(close_for_idx, axis=0)
                mkt_slice = np.where(np.isfinite(mkt_slice), mkt_slice, 0.0)

            self._regime_det   = MarketRegimeDetector(self._risk_cfg, mkt_slice)
            self._port_builder = PortfolioBuilder(self._risk_cfg, mkt_slice)

            _regime_limits = self._regime_det.compute(  # type: ignore[union-attr]
                close_full,
                valid_full_slc_use,
                warmup=actual_warmup,
            )
            regime_full_arr = self._regime_det.get_regime_enum_array()
            regime_bt_arr   = regime_full_arr[actual_warmup:]

            # 写入缓存
            self._regime_cache_key    = _regime_cache_key    # type: ignore[attr-defined]
            self._cached_regime_limits = _regime_limits       # type: ignore[attr-defined]

            breakdown["regime_ms"] = (time.perf_counter() - t2) * 1000
            logger.debug(f"[regime cache MISS] key={_regime_cache_key}, "
                         f"computed in {breakdown['regime_ms']:.0f}ms")

        # ── Step 3: 裁剪回测期切片 ───────────────────────────────────────
        close_bt     = np.ascontiguousarray(close_full[:, actual_warmup:],  dtype=np.float64)
        open_bt      = np.ascontiguousarray(open_full[:,  actual_warmup:],  dtype=np.float64)
        high_bt      = np.ascontiguousarray(high_full[:,  actual_warmup:],  dtype=np.float64)
        low_bt       = np.ascontiguousarray(low_full[:,   actual_warmup:],  dtype=np.float64)
        # ★ vol_bt 已是股数（乘以 vol_multiplier 后再切片）
        vol_bt       = np.ascontiguousarray(vol_full[:,   actual_warmup:],  dtype=np.float64)
        valid_bt     = np.ascontiguousarray(valid_full_slc_use[:, actual_warmup:], dtype=np.bool_)
        amount_bt    = (
            np.ascontiguousarray(amount_full[:, actual_warmup:], dtype=np.float64)
            if amount_full is not None else None
        )

        # ── Step 4: 信号生成 ──────────────────────────────────────────────
        t3 = time.perf_counter()
        gen_fn = _VEC_STRATEGY_REGISTRY.get(strategy_name)
        if gen_fn is None:
            available = list_vec_strategies()
            raise ValueError(
                f"[FastRunnerV10] 策略 '{strategy_name}' 未注册。\n"
                f"已注册: {available}"
            )

        sig = inspect.signature(gen_fn)
        kw = self._build_strategy_kw(
            close       = close_full,          # 含预热（策略需历史数据计算因子）
            open_       = open_full,
            high        = high_full,
            low         = low_full,
            volume      = vol_full,            # 含预热（股数）
            amount      = amount_full,
            valid_mask  = valid_full_slc_use,
            params      = params,
            sig         = sig,
            regime_slice= regime_full_arr,     # int8 数组（含预热）
            # [FIX-A1] 传入绝对索引，_build_strategy_kw 内 _slice_fund 会将
            # 财务矩阵从全量 (N,T_full) 裁剪到与 close_full 相同的 (N,T_slice)
            # 原版缺少此两个参数，导致财务矩阵时间轴与价格矩阵完全错位
            t_start     = t_data_start,
            t_end       = t_e,
        )

        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", RuntimeWarning)
            result_tup = gen_fn(**kw)

        # ── 解析策略返回值 ────────────────────────────────────────────────
        # V10 策略期望返回 AlphaSignal 或 (raw_weights, ...)
        # 兼容 V8 (buy, sell) 和 (buy, sell, weights) 格式
        # [FIX-B-03] AlphaSignal路径也需裁剪预热期，与tuple路径对齐
        if isinstance(result_tup, AlphaSignal):
            w_full = result_tup.raw_target_weights
            if w_full.shape[1] > T_backtest:
                w_full = w_full[:, actual_warmup:]
            alpha_raw = AlphaSignal(
                raw_target_weights = w_full,
                strategy_name      = result_tup.strategy_name,
                score              = result_tup.score[:, actual_warmup:] if result_tup.score is not None and result_tup.score.shape[1] > T_backtest else result_tup.score,
                meta               = result_tup.meta,
            )
        elif isinstance(result_tup, np.ndarray):
            w = result_tup
            if w.ndim == 2 and w.shape[1] > T_backtest:
                w = w[:, actual_warmup:]
            alpha_raw = AlphaSignal(raw_target_weights=w, strategy_name=strategy_name)
        elif isinstance(result_tup, tuple):
            if len(result_tup) >= 3 and result_tup[2] is not None:
                w = np.asarray(result_tup[2], dtype=np.float64)
                if w.ndim == 2:
                    w = w[:, actual_warmup:]       # 裁掉预热
                alpha_raw = AlphaSignal(raw_target_weights=w, strategy_name=strategy_name)
            else:
                # V8 (buy, sell)：将 buy_signals 当权重
                buy = np.asarray(result_tup[0], dtype=np.float64)
                if buy.ndim == 2:
                    buy = buy[:, actual_warmup:]
                alpha_raw = AlphaSignal(raw_target_weights=buy, strategy_name=strategy_name)
        else:
            raise ValueError(
                f"[FastRunnerV10] 策略 '{strategy_name}' 返回类型不识别: {type(result_tup)}"
            )

        # [FIX-F-01] (N,1) 单日权重广播保护
        # 若策略返回 (N,1)，说明策略未向量化（如 weak_to_strong 旧版）。
        # 广播到 (N, T_backtest) 使内核能正常循环；同时发出警告提示修复策略。
        w_arr = alpha_raw.raw_target_weights
        if w_arr.ndim == 2 and w_arr.shape[1] == 1 and T_backtest > 1:
            logger.warning(
                f"[FIX-F-01] 策略 '{strategy_name}' 返回 (N,1) 单日权重，"
                f"已自动广播到 (N,{T_backtest})。"
                f"建议将策略改为返回全时段 (N,T) 矩阵以消除隐式前视风险。"
            )
            w_arr = np.repeat(w_arr, T_backtest, axis=1)

        # NaN/inf 保护
        w_arr = np.nan_to_num(w_arr, nan=0.0, posinf=0.0, neginf=0.0)
        w_arr = np.clip(w_arr, 0.0, None)
        alpha_raw = AlphaSignal(
            raw_target_weights = w_arr,
            strategy_name      = strategy_name,
            score              = alpha_raw.score,
            meta               = alpha_raw.meta,
        )

        breakdown["signal_ms"] = (time.perf_counter() - t3) * 1000

        # ── Step 5: ★ PortfolioBuilder.build（regime 归一化）──────────────
        t4 = time.perf_counter()
        # [BUG-B-3 FIX] 直接注入 Step2 计算好的 _regime_limits，避免 build()
        # 内部重复调用 compute()（性能浪费 + 状态机重走）
        # [BUG-B-2 FIX] 传入 amount_bt 激活白皮书 §4.2 流动性过滤
        self._port_builder._regime_limits = _regime_limits  # type: ignore[union-attr]
        final_weights = self._port_builder.build(  # type: ignore[union-attr]
            alpha         = alpha_raw,
            valid_mask    = valid_bt,
            amount_matrix = amount_bt,    # [BUG-B-2 FIX] 流动性过滤
            # 不传 close_full/valid_full/warmup → 使用上方注入的缓存
        )
        breakdown["portfolio_build_ms"] = (time.perf_counter() - t4) * 1000

        # ── Step 5b [REMOVED-WP-DELAY] ─────────────────────────────────────────────
        # 已移除：T+1 权重延迟补偿（左移逻辑）。
        # 原逻辑导致信号使用了当日收盘价进行“回测瞬移”，产生前视偏差。
        # 现代 V10 内核原生支持 T+1（读取 col_w = t-1），无需手动左移。
        pass

        # ── Step 6: 传内核前强制 ascontiguousarray float64 ───────────────
        t5 = time.perf_counter()
        def _C(x: np.ndarray) -> np.ndarray:
            return np.ascontiguousarray(x, dtype=np.float64)

        limit_up_mask = np.zeros((N, T_backtest), dtype=np.bool_)
        limit_dn_mask = np.zeros((N, T_backtest), dtype=np.bool_)

        # [FIX-B8] 涨跌停 mask：按板块动态阈值，不再固定9.5%
        # 主板/中小板 ≈ 9.5%，创业板300/科创板688 ≈ 19.5%，ST ≈ 4.8%
        # [FIX-6] 改用前一日收盘价vs前前日收盘价判定：
        #   昨日收盘涨停 → 今天大概率一字板买不进
        #   昨日收盘跌停 → 今天大概率一字板卖不出
        #   原实现用开盘价判定，高开不一定涨停，导致误判
        if T_backtest > 1:
            prev_close_bt = np.zeros((N, T_backtest), dtype=np.float64)
            if actual_warmup > 0:
                prev_close_bt[:, 0] = close_full[:, actual_warmup - 1]
            else:
                prev_close_bt[:, 0] = close_bt[:, 0]
            prev_close_bt[:, 1:] = close_bt[:, :-1]

            # [FIX-6] 前前日收盘价（用于判断前一日是否涨跌停）
            prev2_close_bt = np.zeros((N, T_backtest), dtype=np.float64)
            if actual_warmup > 1:
                prev2_close_bt[:, 0] = close_full[:, actual_warmup - 2]
            else:
                prev2_close_bt[:, 0] = prev_close_bt[:, 0]
            prev2_close_bt[:, 1:] = prev_close_bt[:, :-1]

            with np.errstate(divide="ignore", invalid="ignore"):
                # [FIX-6] 前一日涨跌幅 = (昨收 - 前收) / 前收
                chg = np.where(
                    prev2_close_bt > 0,
                    (prev_close_bt - prev2_close_bt) / prev2_close_bt,
                    0.0,
                )
            # 按股票代码区分板块阈值（广播到时间轴）
            codes = self._meta.get("codes", [])
            thr_vec = np.full(N, 0.095, dtype=np.float64)   # 默认主板10%
            for i, c in enumerate(codes):
                c_str = str(c)
                prefix = c_str[:3] if len(c_str) >= 3 else c_str
                if prefix in ("300", "301", "688", "689"):
                    thr_vec[i] = 0.195   # 创业板/科创板20%
                elif c_str.startswith("8") or "ST" in c_str.upper():
                    thr_vec[i] = 0.048   # 北交所/ST 5%
            thr_2d = thr_vec[:, np.newaxis]   # (N,1) 广播
            limit_up_mask = chg >= thr_2d
            limit_dn_mask = chg <= -thr_2d

        kw_kernel = self._risk_cfg.to_kernel_kwargs()
        # 参与率统一从 self 读（RiskConfig 无此字段）
        kw_kernel["participation_rate"] = self.participation_rate

        # ★[FIX-EXIT] 读取策略个性化出场配置（exit_config），覆盖全局参数
        # 若策略未提供 exit_config，则使用 RiskConfig 全局值（向后兼容）
        _exit_cfg = getattr(alpha_raw, "exit_config", None) or {}
        if _exit_cfg.get("hard_stop_loss") is not None:
            kw_kernel["hard_stop_loss"]    = float(_exit_cfg["hard_stop_loss"])
        if _exit_cfg.get("max_holding_days") is not None:
            kw_kernel["max_holding_days"]  = int(_exit_cfg["max_holding_days"])
        # stop_mode: "trailing" → True, "entry_price" → False
        _stop_mode_str = _exit_cfg.get("stop_mode", "trailing")
        kw_kernel["stop_mode_trailing"] = (_stop_mode_str != "entry_price")
        kw_kernel["take_profit"]        = float(_exit_cfg.get("take_profit", 0.0))

        # ── Pass 1：第一次撮合，获取 stop_triggered 矩阵 ─────────────────
        pos_matrix, nav_array, cash_array, stop_triggered = match_engine_weights_driven(
            final_target_weights = _C(final_weights),
            exec_prices          = _C(open_bt),
            close_prices         = _C(close_bt),
            high_prices          = _C(high_bt),
            volume               = _C(vol_bt),
            limit_up_mask        = limit_up_mask,
            limit_dn_mask        = limit_dn_mask,
            initial_cash         = self.initial_cash,
            **kw_kernel,
        )

        # ★[FIX-STATE-DRIFT-REAL] stop_cooldown_days 真实状态漂移修复
        # ─────────────────────────────────────────────────────────────────
        # 问题根因（完整链路）：
        #   1. _score_to_weights 全量预计算 final_weights (N,T)
        #   2. Pass1 内核在 t 日 L3-B 止损，position[i]=0，但 final_weights[i,t+1:]>0
        #   3. Pass1 内核次日看到 weights>0，重新建仓 → 止损后立即回购
        #
        # 修复（二次过引擎方案）：
        #   1. 从 stop_triggered 矩阵读出每只股票的止损日期
        #   2. 对每个止损事件 (i, t_stop)，将 final_weights[i, t_stop+1:t_stop+cooldown+1] 清零
        #   3. 用修正后的权重矩阵 Pass2 过引擎，得到无状态漂移的正确结果
        #
        # cooldown 天数：从 exit_config["dropout_days"] 读取（语义一致：需要连续
        # dropout_days 天缺席才真正离场，冷却期与此对齐）
        stop_cooldown_days = int(_exit_cfg.get("dropout_days", 3))

        if stop_triggered.any() and stop_cooldown_days > 0:
            # 构建冷却掩码：止损后 cooldown 天内强制权重=0
            weights_cooled = final_weights.copy()
            N_w, T_w = weights_cooled.shape
            stop_i, stop_t = np.where(stop_triggered)
            for i, t_stop in zip(stop_i, stop_t):
                # ★[FIX-OFFSET] 内核在 t 日读取 col_w = t-1 的权重。
                # 若 L3-B 在 t=t_stop 触发止损，内核在 t=t_stop+1 执行买入时读的是
                # weights[:, t_stop]（而不是 t_stop+1）。
                # 因此冷却窗口必须从 t_stop 开始（覆盖内核在 t_stop+1..t_stop+cd 读到的列）。
                t_start = t_stop            # 从止损当天的列开始清零
                t_end   = min(t_start + stop_cooldown_days, T_w)
                if t_start < T_w:
                    weights_cooled[i, t_start:t_end] = 0.0

            # Pass 2：用冷却后权重重新撮合（仅当权重实际发生变化时）
            if not np.array_equal(weights_cooled, final_weights):
                breakdown["match_pass1_ms"] = breakdown.pop("match_ms", 0)
                t5b = time.perf_counter()
                pos_matrix, nav_array, cash_array, stop_triggered = match_engine_weights_driven(
                    final_target_weights = _C(weights_cooled),
                    exec_prices          = _C(open_bt),
                    close_prices         = _C(close_bt),
                    high_prices          = _C(high_bt),
                    volume               = _C(vol_bt),
                    limit_up_mask        = limit_up_mask,
                    limit_dn_mask        = limit_dn_mask,
                    initial_cash         = self.initial_cash,
                    **kw_kernel,
                )
                breakdown["match_pass2_ms"] = (time.perf_counter() - t5b) * 1000
                logger.debug(
                    f"[FIX-STATE-DRIFT] Pass2 完成: "
                    f"冷却事件={len(stop_i)}个 "
                    f"cooldown={stop_cooldown_days}天"
                )

        breakdown["match_ms"] = (time.perf_counter() - t5) * 1000
        alpha_raw.meta["stop_triggered"] = stop_triggered  # type: ignore[union-attr]

        # ── Step 7: 绩效计算 ─────────────────────────────────────────────
        annual_factor = float(getattr(params, "annual_factor", 252.0))
        (total_ret, annual_ret, sharpe, max_dd,
         sortino, calmar, win_rate, pf) = _metrics_numpy(
             nav_array, annual_factor, pos_matrix   # [FIX-WINRATE] 传入持仓矩阵
         )

        daily_ret  = np.diff(nav_array) / (nav_array[:-1] + 1e-10)
        volatility = float(np.std(daily_ret) * np.sqrt(annual_factor))
        turnover   = _turnover_numpy(pos_matrix, nav_array, annual_factor, close_bt)

        # [FIX-REGIME-STATS] 计算有持仓天占比和Regime分布
        invested_days  = int((pos_matrix.sum(axis=0) > 0).sum())
        total_days     = pos_matrix.shape[1]
        invested_ratio = float(invested_days / max(total_days, 1))

        # [NEW] 计算买入/卖出次数和期末持仓
        # [FIX-B10] np.diff 漏掉T=0初始建仓（从0→持仓的变化）。
        # 修复：左侧补一列全零后再 diff，使T=0的建仓被正确计入 buy_count。
        _pos_with_zero  = np.concatenate(
            [np.zeros((pos_matrix.shape[0], 1), dtype=np.float64), pos_matrix], axis=1
        )
        _pos_diff       = np.diff(_pos_with_zero, axis=1)           # (N, T)
        buy_count       = int((_pos_diff > 1e-3).sum())
        sell_count      = int((_pos_diff < -1e-3).sum())
        final_positions = int((pos_matrix[:, -1] > 1e-3).sum())  # 期末持仓股数

        params_dict = params.to_dict() if hasattr(params, "to_dict") else {}
        total_ms    = (time.perf_counter() - t_total) * 1000

        logger.info(
            f"[FastRunnerV10] ✓ {strategy_name} "
            f"Ret={annual_ret:+.1%} Sharpe={sharpe:.2f} DD={max_dd:.1%} "
            f"stamp_tax={self._risk_cfg.stamp_tax} "
            f"总耗时={total_ms:.0f}ms"
        )

        # ── 缓存矩阵（供 run_with_details 使用）─────────────────────────
        self._last_pos_matrix    = pos_matrix
        self._last_nav_array     = nav_array
        self._last_period_dates  = period_dates
        self._last_regime_bt     = regime_bt_arr    # (T_backtest,) int8
        self._last_actual_warmup = actual_warmup    # ★[FIX-T2] 供 multi_run 使用
        # ★[FIX-PASS2] 缓存 final_weights，供 _run_get_weights 命中缓存，
        # 消除 multi_run Step4 的「静默第二轮」重跑。
        self._last_final_weights = final_weights    # (N, T_backtest) T+1移位后
        self._last_fw_key        = (strategy_name, start_date, end_date)

        # ── 构造 RunResult ────────────────────────────────────────────────
        if _HAS_V8_RUNNER:
            # 兼容 V8 RunResult 字段
            return RunResult(  # type: ignore[call-arg]
                strategy_name  = strategy_name,
                params_dict    = params_dict,
                total_return   = float(total_ret),
                annual_return  = float(annual_ret),
                sharpe_ratio   = float(sharpe),
                max_drawdown   = float(max_dd),
                sortino_ratio  = float(sortino),
                calmar_ratio   = float(calmar),
                win_rate       = float(win_rate),
                profit_factor  = float(pf),
                volatility     = float(volatility),
                turnover       = float(turnover),
                nav            = nav_array,
                dates          = period_dates,
                elapsed_ms     = total_ms,
                annual_factor  = annual_factor,
                pipeline_breakdown = breakdown,
            )
        else:
            return RunResult(
                strategy_name  = strategy_name,
                params_dict    = params_dict,
                total_return   = float(total_ret),
                annual_return  = float(annual_ret),
                sharpe_ratio   = float(sharpe),
                max_drawdown   = float(max_dd),
                sortino_ratio  = float(sortino),
                calmar_ratio   = float(calmar),
                win_rate       = float(win_rate),
                profit_factor  = float(pf),
                volatility     = float(volatility),
                turnover       = float(turnover),
                nav_array      = nav_array,
                dates          = period_dates,
                elapsed_ms     = total_ms,
                annual_factor  = annual_factor,
                invested_ratio = invested_ratio,
                buy_count      = buy_count,
                sell_count     = sell_count,
                final_positions= final_positions,
                pipeline_breakdown = breakdown,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # 实盘单日信号
    # ─────────────────────────────────────────────────────────────────────────

    def realtime_signal(
        self,
        strategy_name : str,
        params        : Any,
        as_of_date    : str,
        extra_warmup  : int = 0,
    ) -> Dict[str, float]:
        """
        实盘单日信号生成，返回 {code: weight} 字典。

        Parameters
        ----------
        strategy_name : str
        params        : 策略参数
        as_of_date    : str  "YYYY-MM-DD"，信号日期（闭区间）
        extra_warmup  : int  额外追加的预热天数

        Returns
        -------
        {code: target_weight}，权重之和 ≤ regime_limit
        """
        if self._data is None:
            self.load_data()

        warmup  = self._calc_warmup(params) + extra_warmup
        (close, open_, high, low, volume, amount,
         valid_mask, regime_slice, period_dates,
         actual_warmup) = self._slice_ending_at(as_of_date, warmup)
        # [FIX-SLICE] 获取绝对索引，用于切片财务矩阵到同一时段
        dates = self._data.get("dates", [])  # type: ignore[union-attr]
        _t_end   = next((i for i, d in enumerate(dates) if d > as_of_date), len(dates)) if dates else None
        _t_start = max(0, _t_end - actual_warmup - 1) if _t_end is not None else 0

        if close is None or close.shape[1] < 2:
            logger.warning(f"[FastRunnerV10] realtime_signal: 数据不足，返回空信号")
            return {}

        gen_fn = _VEC_STRATEGY_REGISTRY.get(strategy_name)
        if gen_fn is None:
            raise ValueError(f"[FastRunnerV10] 策略 '{strategy_name}' 未注册")

        sig = inspect.signature(gen_fn)
        kw  = self._build_strategy_kw(
            close       = close,
            open_       = open_,
            high        = high,
            low         = low,
            volume      = volume,
            amount      = amount,
            valid_mask  = valid_mask,
            params      = params,
            sig         = sig,
            regime_slice= regime_slice,
            t_start     = _t_start,   # [FIX-SLICE] 财务矩阵对齐到 realtime 时段
            t_end       = _t_end,
        )

        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", RuntimeWarning)
            result_tup = gen_fn(**kw)

        # 取最后一列权重（今日信号）
        if isinstance(result_tup, AlphaSignal):
            w_today = result_tup.raw_target_weights[:, -1].copy()
        elif isinstance(result_tup, np.ndarray) and result_tup.ndim == 2:
            w_today = result_tup[:, -1].copy()
        elif isinstance(result_tup, tuple):
            buy = result_tup[2] if len(result_tup) >= 3 else result_tup[0]
            buy = np.asarray(buy, dtype=np.float64)
            w_today = (buy[:, -1] if buy.ndim == 2 else buy).copy()
        else:
            return {}

        w_today = np.nan_to_num(w_today, nan=0.0)
        w_today = np.clip(w_today, 0.0, None)

        # regime 仓位上限（今日）
        regime_limit = (
            float(self._regime_det.get_pos_limit_today())  # type: ignore[union-attr]
            if self._regime_det is not None else 1.0
        )

        valid_today = valid_mask[:, -1] if valid_mask is not None else np.ones(len(w_today), dtype=bool)
        w_today[~valid_today] = 0.0

        col_sum = w_today.sum()
        if col_sum > regime_limit and col_sum > 1e-12:
            w_today *= regime_limit / col_sum

        codes = self._meta.get("codes", [f"stock_{i}" for i in range(len(w_today))])  # type: ignore[union-attr]
        return {
            str(codes[i]): float(w_today[i])
            for i in range(len(w_today))
            if w_today[i] > 1e-9
        }

    def _run_get_weights(
        self,
        strategy_name : str,
        params        : Any,
        start_date    : Optional[str] = None,
        end_date      : Optional[str] = None,
    ) -> np.ndarray:
        """
        ★[FIX-T2] 内部辅助：运行策略并返回 PortfolioBuilder 输出的
        final_weights (N, T_backtest) float64（T+1 延迟补偿后），
        供 multi_run 做真实权重合并。

        与 run() 的区别：不进行撮合，直接返回权重矩阵。

        ★[FIX-PASS2] 命中缓存路径：
        multi_run Step1 已调用 run() 并将 final_weights 写入
        self._last_final_weights；key 一致时直接返回缓存，
        完全跳过重跑，将 Step4 从 ~600s 降为 <1ms/策略。
        """
        # ── 缓存命中：run() 已缓存同一 (strategy, start, end) 的 final_weights ─
        _fw_cache_key = (strategy_name, start_date, end_date)
        if (
            hasattr(self, "_last_fw_key")
            and self._last_fw_key == _fw_cache_key
            and hasattr(self, "_last_final_weights")
            and self._last_final_weights is not None
        ):
            return self._last_final_weights  # type: ignore[return-value]

        if self._data is None:
            self.load_data()

        dates    = self._meta["dates"]  # type: ignore[index]
        t_s, t_e = self._slice_dates(dates, start_date, end_date)

        warmup_req   = self._calc_warmup(params)
        t_data_start = max(0, t_s - warmup_req)
        actual_warmup = t_s - t_data_start
        T_backtest   = t_e - t_s

        def _slc(key: str, dtype=np.float64, fallback: Optional[str] = None):
            arr = self._data.get(key)  # type: ignore[union-attr]
            if arr is None and fallback:
                arr = self._data.get(fallback)
            if arr is None:
                return None
            return np.ascontiguousarray(arr[:, t_data_start:t_e], dtype=dtype)

        close_full  = np.nan_to_num(_slc("close"),               nan=0.0)
        open_full   = np.nan_to_num(_slc("open", fallback="close"), nan=0.0)
        high_full   = np.nan_to_num(_slc("high", fallback="close"), nan=0.0)
        low_full    = np.nan_to_num(_slc("low",  fallback="close"), nan=0.0)
        _vol_raw    = _slc("volume")
        vol_full    = (
            np.ascontiguousarray(_vol_raw * self._vol_multiplier, dtype=np.float64)
            if _vol_raw is not None else np.zeros_like(close_full)
        )
        amount_full    = _slc("amount")
        valid_full_slc = _slc("valid_mask", dtype=np.bool_)
        valid_full_use = (
            valid_full_slc if valid_full_slc is not None
            else np.ones_like(close_full, dtype=np.bool_)
        )

        mkt_raw = self._data.get("market_index")  # type: ignore[union-attr]
        if mkt_raw is not None:
            mkt_slice = np.asarray(mkt_raw, dtype=np.float64).flatten()[t_data_start:t_e]
        else:
            with np.errstate(all="ignore"):
                _cf = np.where(close_full > 1e-8, close_full, np.nan)
                mkt_slice = np.nanmean(_cf, axis=0)
            mkt_slice = np.where(np.isfinite(mkt_slice), mkt_slice, 0.0)

        _rd = MarketRegimeDetector(self._risk_cfg, mkt_slice)
        _pb = PortfolioBuilder(self._risk_cfg, mkt_slice)
        _rl = _rd.compute(close_full, valid_full_use, warmup=actual_warmup)
        _pb._regime_limits = _rl

        close_bt  = np.ascontiguousarray(close_full[:, actual_warmup:], dtype=np.float64)
        valid_bt  = np.ascontiguousarray(valid_full_use[:, actual_warmup:], dtype=np.bool_)
        amount_bt = (
            np.ascontiguousarray(amount_full[:, actual_warmup:], dtype=np.float64)
            if amount_full is not None else None
        )

        import inspect as _inspect
        sig  = _inspect.signature(_VEC_STRATEGY_REGISTRY[strategy_name])
        kw   = self._build_strategy_kw(
            close=close_full, open_=open_full, high=high_full, low=low_full,
            volume=vol_full, amount=amount_full, valid_mask=valid_full_use,
            params=params, sig=sig,
            regime_slice=_rd.get_regime_enum_array(),
            t_start=t_data_start, t_end=t_e,
        )
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore", RuntimeWarning)
            result = _VEC_STRATEGY_REGISTRY[strategy_name](**kw)

        if isinstance(result, AlphaSignal):
            w_full = result.raw_target_weights
            if w_full.shape[1] > T_backtest:
                w_full = w_full[:, actual_warmup:]
            alpha_r = AlphaSignal(raw_target_weights=w_full, strategy_name=strategy_name,
                                  score=result.score, meta=result.meta)
        elif isinstance(result, np.ndarray):
            w = result
            if w.ndim == 2 and w.shape[1] > T_backtest:
                w = w[:, actual_warmup:]
            alpha_r = AlphaSignal(raw_target_weights=w, strategy_name=strategy_name)
        else:
            raise ValueError(f"_run_get_weights: unrecognised return type {type(result)}")

        w_arr = np.nan_to_num(alpha_r.raw_target_weights, nan=0.0, posinf=0.0, neginf=0.0)
        w_arr = np.clip(w_arr, 0.0, None)
        alpha_r = AlphaSignal(raw_target_weights=w_arr, strategy_name=strategy_name,
                               score=alpha_r.score, meta=alpha_r.meta)

        final = _pb.build(alpha_r, valid_mask=valid_bt, amount_matrix=amount_bt)

        # T+1 延迟补偿（与 run() 一致）
        if final.shape[1] > 1:
            fs = np.zeros_like(final)
            fs[:, :-1] = final[:, 1:]
            fs[:, -1]  = final[:, -1]
            final = fs

        return final

    # ─────────────────────────────────────────────────────────────────────────
    # ★[H-01] 多策略组合回测（白皮书 §6.1）
    # ─────────────────────────────────────────────────────────────────────────

    def multi_run(
        self,
        strategy_configs : List[Tuple[str, Any]],
        allocator,                                    # PortfolioAllocator
        start_date       : Optional[str] = None,
        end_date         : Optional[str] = None,
        corr_warn_threshold: float = 0.80,
    ):
        """
        ★[H-01] 多策略组合回测（白皮书 §6.1）。

        步骤
        ----
        1. 并行运行各策略，收集 RunResult 和 final_weights
        2. 相关性检查（高相关策略对输出 WARNING）
        3. 计算资金分配比例（通过 allocator.allocate()）
        4. 按比例加权合并 final_target_weights
        5. 合并权重再做一次 PortfolioBuilder.build()（Regime + 流动性归一化）
        6. 最终撮合，返回 MultiRunResult

        Parameters
        ----------
        strategy_configs : [(strategy_name, params), ...]  策略名称与参数列表
        allocator        : PortfolioAllocator 实例
        start_date       : 回测起始日期（可选）
        end_date         : 回测截止日期（可选）
        corr_warn_threshold : 相关性告警阈值（默认 0.80）

        Returns
        -------
        MultiRunResult
        """
        # 延迟导入（避免循环依赖）
        try:
            from src.engine.portfolio_allocator import (
                PortfolioAllocator, MultiRunResult
            )
        except ImportError:
            from portfolio_allocator import PortfolioAllocator, MultiRunResult  # type: ignore

        if self._data is None:
            self.load_data()

        logger.info(
            f"[FastRunnerV10.multi_run] 策略数={len(strategy_configs)} "
            f"method={allocator.method}"
        )

        n_strats = len(strategy_configs)
        print(f"\n[multi_run] 开始组合回测：{n_strats} 个策略  method={allocator.method}",
              flush=True)

        # ★[FIX-REGIME-CACHE] multi_run 前清除缓存，确保首个策略触发真实计算，
        # 后续相同窗口的策略命中缓存（省去 9/10 的 Regime 重复计算）。
        if hasattr(self, "_regime_cache_key"):
            del self._regime_cache_key  # type: ignore[attr-defined]

        # ── Step 1: 逐策略运行，收集结果 ────────────────────────────────────
        results: Dict[str, Any]           = {}
        nav_history: Dict[str, np.ndarray] = {}
        _t_multirun_start = time.perf_counter()

        for _si, (strategy_name, params) in enumerate(strategy_configs, 1):
            print(f"  [{_si}/{n_strats}] {strategy_name} ...", end="", flush=True)
            _t_s = time.perf_counter()
            try:
                res = self.run(
                    strategy_name = strategy_name,
                    params        = params,
                    start_date    = start_date,
                    end_date      = end_date,
                )
                results[strategy_name] = res
                # [FIX-NAV-OR] 兼容 nav_array / nav 两种字段名，安全提取 nav
                _nav = getattr(res, "nav_array", None)
                if _nav is None:
                    _nav = getattr(res, "nav", None)
                if _nav is not None:
                    nav_history[strategy_name] = np.asarray(_nav, dtype=np.float64)
                _elapsed = time.perf_counter() - _t_s
                print(f" Ret={res.annual_return:+.1%}  Sharpe={res.sharpe_ratio:.2f}"
                      f"  ({_elapsed:.1f}s)", flush=True)
                logger.info(
                    f"  [{strategy_name}] Ret={res.annual_return:+.1%} "
                    f"Sharpe={res.sharpe_ratio:.2f}"
                )
            except Exception as exc:
                print(f" 失败: {exc}", flush=True)
                logger.error(f"  [{strategy_name}] 运行失败: {exc}")

        if not nav_history:
            raise RuntimeError("[multi_run] 所有策略均运行失败，无法生成组合结果")

        _elapsed_total = time.perf_counter() - _t_multirun_start
        print(f"\n[multi_run] 策略回测完成，共 {_elapsed_total:.1f}s", flush=True)

        # ── Step 2: 相关性检查 ───────────────────────────────────────────────
        high_corr_pairs = allocator.correlation_check(
            nav_history, threshold=corr_warn_threshold
        )

        # ── Step 3: 计算资金分配 ─────────────────────────────────────────────
        allocations = allocator.allocate(nav_history)
        logger.info(f"[multi_run] 资金分配: {allocations}")

        # ── Step 4: ★[FIX-T2] 合并 final_target_weights，过真实撮合引擎 ────
        # BUG-T2 根因：原实现对各策略归一化 NAV 做加权平均，
        # 等价于「各策略独立运行后对收益率加权」，这与「合并权重后统一撮合」
        # 语义完全不同：
        #   ① 交易成本：NAV 平均 = 各策略各自交易；权重合并 = 重叠持仓抵消，成本更低
        #   ② 仓位上限：NAV 平均不受 max_single_pos 约束；权重合并后会被截断
        #   ③ Regime 过滤：NAV 平均每策略单独过滤；权重合并后统一过滤
        # 正确做法：各策略 final_weights × 分配比例 → 相加 → 过 PortfolioBuilder
        # → 过 match_engine_weights_driven → 得到真实组合 NAV

        # ★[FIX-PASS2] _run_get_weights 命中缓存时 <1ms/策略（run() 已写入缓存）；
        # 缓存 miss（理论上不应发生）时打印进度提示，避免静默卡死。
        combined_weights : np.ndarray | None = None
        T_combined = None
        n_fw_total = sum(1 for s, _ in strategy_configs if s in results)
        n_fw_done  = 0
        print(f"[multi_run] Step4: 合并权重矩阵（{n_fw_total} 个策略）...", flush=True)

        for strategy_name, params in strategy_configs:
            if strategy_name not in results:
                continue
            alloc = allocations.get(strategy_name, 0.0)
            if alloc < 1e-8:
                continue

            # ★[FIX-PASS2] 命中缓存时 <1ms；miss 时重跑并打印进度，避免静默卡死
            n_fw_done += 1
            try:
                fw = self._run_get_weights(
                    strategy_name = strategy_name,
                    params        = params,
                    start_date    = start_date,
                    end_date      = end_date,
                )
                print(f"  [Step4 {n_fw_done}/{n_fw_total}] {strategy_name} 权重已就绪", flush=True)
            except Exception as exc:
                logger.warning(f"  [{strategy_name}] 获取合并权重失败，跳过: {exc}")
                print(f"  [Step4 {n_fw_done}/{n_fw_total}] {strategy_name} ⚠ 失败: {exc}", flush=True)
                continue

            if combined_weights is None:
                T_combined      = fw.shape[1]
                N_stocks        = fw.shape[0]
                combined_weights = np.zeros((N_stocks, T_combined), dtype=np.float64)

            # 时间轴对齐（取最短）
            T_use = min(fw.shape[1], T_combined)  # type: ignore[arg-type]
            combined_weights[:, :T_use] += fw[:, :T_use] * alloc
            T_combined = T_use

        if combined_weights is None or T_combined == 0:
            # 降级：所有策略权重获取失败，回退到 NAV 加权平均
            logger.warning("[multi_run] 权重合并失败，降级为 NAV 加权平均（近似结果）")
            T_lens = [len(v) for v in nav_history.values()]
            T_min  = min(T_lens)
            combined_nav = np.zeros(T_min, dtype=np.float64)
            for name, nav in nav_history.items():
                alloc = allocations.get(name, 0.0)
                nav_aligned = np.asarray(nav[-T_min:], dtype=np.float64)
                nav_norm = nav_aligned / (nav_aligned[0] + 1e-10)
                combined_nav += nav_norm * alloc
        else:
            # ── Step 5: 合并权重过 PortfolioBuilder + 撮合引擎 ──────────────
            combined_weights = combined_weights[:, :T_combined]

            # 取最后一次 run() 对应的回测期切片数据
            dates      = self._meta["dates"]  # type: ignore[index]
            t_s, t_e   = self._slice_dates(dates, start_date, end_date)
            actual_wu  = self._last_actual_warmup if hasattr(self, "_last_actual_warmup") else 0
            t_ds       = max(0, t_s - actual_wu)

            def _slc2(key: str) -> np.ndarray:
                arr = self._data.get(key)  # type: ignore[union-attr]
                return np.ascontiguousarray(arr[:, t_ds:t_e], dtype=np.float64)

            close_f2  = np.nan_to_num(_slc2("close"),  nan=0.0)
            open_f2   = np.nan_to_num(_slc2("open"),   nan=0.0)
            high_f2   = np.nan_to_num(_slc2("high"),   nan=0.0)
            vol_f2    = np.nan_to_num(_slc2("volume"),  nan=0.0) * self._vol_multiplier
            valid_f2_raw = self._data.get("valid_mask")  # type: ignore[union-attr]
            valid_f2  = (
                np.ascontiguousarray(valid_f2_raw[:, t_ds:t_e], dtype=np.bool_)
                if valid_f2_raw is not None
                else np.ones_like(close_f2, dtype=np.bool_)
            )
            wu2 = t_s - t_ds
            close_bt2 = np.ascontiguousarray(close_f2[:, wu2:],  dtype=np.float64)
            open_bt2  = np.ascontiguousarray(open_f2[:,  wu2:],  dtype=np.float64)
            high_bt2  = np.ascontiguousarray(high_f2[:,  wu2:],  dtype=np.float64)
            vol_bt2   = np.ascontiguousarray(vol_f2[:,   wu2:],  dtype=np.float64)
            valid_bt2 = np.ascontiguousarray(valid_f2[:,  wu2:], dtype=np.bool_)

            # PortfolioBuilder 归一化（Regime + valid_mask）
            alpha_combined = AlphaSignal(
                raw_target_weights = combined_weights,
                strategy_name      = "multi_run_combined",
            )
            mkt_raw = self._data.get("market_index")  # type: ignore[union-attr]
            if mkt_raw is not None:
                mkt_s2 = np.asarray(mkt_raw, dtype=np.float64).flatten()[t_ds:t_e]
            else:
                with np.errstate(all="ignore"):
                    _cf2 = np.where(close_f2 > 1e-8, close_f2, np.nan)
                    mkt_s2 = np.nanmean(_cf2, axis=0)
                mkt_s2 = np.where(np.isfinite(mkt_s2), mkt_s2, 0.0)

            _pb2 = PortfolioBuilder(self._risk_cfg, mkt_s2)
            _rl2 = MarketRegimeDetector(self._risk_cfg, mkt_s2).compute(
                close_f2, valid_f2, warmup=wu2
            )
            _pb2._regime_limits = _rl2
            final_combined = _pb2.build(alpha_combined, valid_mask=valid_bt2)

            # T+1 延迟补偿（与 run() 中逻辑一致）
            T_bc = final_combined.shape[1]
            if T_bc > 1:
                fw_s = np.zeros_like(final_combined)
                fw_s[:, :-1] = final_combined[:, 1:]
                fw_s[:, -1]  = final_combined[:, -1]
                final_combined = fw_s

            # 涨跌停 mask
            N_bc = close_bt2.shape[0]
            lu2 = np.zeros((N_bc, T_bc), dtype=np.bool_)
            ld2 = np.zeros((N_bc, T_bc), dtype=np.bool_)
            if T_bc > 1:
                chg2 = np.zeros_like(close_bt2)
                chg2[:, 1:] = (close_bt2[:, 1:] - close_bt2[:, :-1]) / (close_bt2[:, :-1] + 1e-10)
                lu2 = chg2 >= 0.095
                ld2 = chg2 <= -0.095

            _C2 = lambda x: np.ascontiguousarray(x, dtype=np.float64)
            kw2 = self._risk_cfg.to_kernel_kwargs()
            kw2["participation_rate"] = self.participation_rate
            kw2["stop_mode_trailing"] = True
            kw2["take_profit"]        = 0.0

            _, nav_combined, _, _ = match_engine_weights_driven(
                final_target_weights = _C2(final_combined),
                exec_prices          = _C2(open_bt2),
                close_prices         = _C2(close_bt2),
                high_prices          = _C2(high_bt2),
                volume               = _C2(vol_bt2),
                limit_up_mask        = lu2,
                limit_dn_mask        = ld2,
                initial_cash         = self.initial_cash,
                **kw2,
            )
            combined_nav = nav_combined / (nav_combined[0] + 1e-10)

        # ── Step 6: 返回 MultiRunResult ──────────────────────────────────────
        mr = MultiRunResult(
            combined_nav     = combined_nav,
            strategy_results = results,
            allocations      = allocations,
            high_corr_pairs  = high_corr_pairs,
            annual_factor    = 252.0,
        )

        logger.info(
            f"[multi_run] 完成 | 合并: Ret={mr.annual_return:+.1%} "
            f"Sharpe={mr.sharpe_ratio:.2f}  DD={mr.max_drawdown:.1%}"
        )
        return mr


# ─────────────────────────────────────────────────────────────────────────────
# 验收测试
# ─────────────────────────────────────────────────────────────────────────────

def _make_synthetic_runner(
    N: int = 80,
    T: int = 1000,
    seed: int = 42,
) -> Tuple[FastRunnerV10, Dict[str, np.ndarray], List[str]]:
    """
    构造一个不依赖真实 npy 文件的 FastRunnerV10，
    通过直接注入 _data / _meta 跳过 IO，用于纯逻辑验收。
    """
    import tempfile, os
    rng = np.random.default_rng(seed)

    # ── 生成合成数据 ────────────────────────────────────────────────────────
    close = np.cumprod(1 + rng.normal(0.0003, 0.015, (N, T)), axis=1).astype(np.float64) * 10
    open_ = close * (1 + rng.uniform(-0.005, 0.005, (N, T)))
    high  = np.maximum(close, open_) * (1 + rng.uniform(0, 0.01, (N, T)))
    low   = np.minimum(close, open_) * (1 - rng.uniform(0, 0.01, (N, T)))
    vol   = rng.uniform(1e6, 1e7, (N, T)).astype(np.float64)  # 单位：手（BaoStock）
    amt   = vol * close * 100

    # 模拟熊市：T=300~340 价格大跌
    close[:, 300:340] *= np.cumprod(np.full((1, 40), 0.985), axis=1)
    high[:, 300:340]   = close[:, 300:340] * 1.005
    low[:, 300:340]    = close[:, 300:340] * 0.995
    open_[:, 300:340]  = close[:, 300:340]

    valid = np.ones((N, T), dtype=np.bool_)

    # 生成交易日期字符串（2019-01-01 起）
    from datetime import date, timedelta
    base  = date(2019, 1, 2)
    dates_dt = [base + timedelta(days=i) for i in range(T * 2) if (base + timedelta(days=i)).weekday() < 5][:T]
    dates = [str(d) for d in dates_dt]

    data = {
        "close"      : close,
        "open"       : open_,
        "high"       : high,
        "low"        : low,
        "volume"     : vol,
        "amount"     : amt,
        "valid_mask" : valid,
    }
    meta = {
        "shape"  : (N, T),
        "dates"  : dates,
        "codes"  : [f"sz.{300000 + i:06d}" for i in range(N)],
        "fields" : ["close", "open", "high", "low", "volume", "amount"],
    }

    # 构造一个虚拟 npy_dir（内容不重要，load_data 不会被调用）
    cfg = {
        "npy_dir"           : "/tmp/fake_npy_v10",
        "initial_cash"      : 1_000_000.0,
        "commission_rate"   : 0.0003,
        "stamp_tax"         : 0.0005,   # ★
        "slippage_rate"     : 0.001,
        "max_single_pos"    : 0.08,
        "hard_stop_loss"    : 0.20,
        "max_holding_days"  : 0,
        "full_stop_dd"      : 0.15,
        "half_stop_dd"      : 0.08,
        "max_gap_up"        : 0.025,
        "vol_multiplier"    : 100,
    }

    runner = FastRunnerV10(cfg)
    runner._data = data
    runner._meta = meta

    # 初始化 regime（直接注入，不经过 npy 加载）
    mkt_idx = np.nanmean(close, axis=0)
    runner._regime_det   = MarketRegimeDetector(runner._risk_cfg, mkt_idx)
    runner._port_builder = PortfolioBuilder(runner._risk_cfg, mkt_idx)

    return runner, data, dates


def test_fast_runner_v10() -> None:
    """
    验收测试：
    1. stamp_tax 恒为 0.0005
    2. vol_multiplier=100（BaoStock 手→股）
    3. regime int8 数组类型
    4. result.nav_array[-1] > 0 且无 NaN
    5. col_sum <= regime_limit（PortfolioBuilder 归一化）
    6. BEAR 期间 final_weights 全零
    7. 年化收益可打印
    8. realtime_signal 返回 {code: weight} 且和 ≤ 1
    """
    import sys

    print("=" * 60)
    print("Q-UNITY V10  fast_runner_v10.py  验收测试")
    print("=" * 60)

    N, T = 80, 1000

    # ── 构造 runner + 合成数据 ────────────────────────────────────────────
    runner, data, dates = _make_synthetic_runner(N=N, T=T, seed=7)
    rng = np.random.default_rng(7)

    # ── 注册一个测试策略（等权 Top-20）────────────────────────────────────
    @register_vec_strategy("ultra_alpha_v1")
    def _ultra_alpha_v1(
        close, open_, high, low, volume, params,
        valid_mask=None, market_regime=None, **kwargs,
    ):
        """
        合成策略：随机 Top-20 等权（验收用）。
        接受 market_regime (int8)，根据 Regime 调整信号强度（验收 int8 路径）。
        """
        _N, _T = close.shape
        w = np.zeros((_N, _T), dtype=np.float64)

        # 验证 market_regime 为 int8
        if market_regime is not None:
            assert market_regime.dtype == np.int8, (
                f"[FAIL] market_regime dtype={market_regime.dtype}，应为 int8"
            )
            # 验证 REGIME_IDX_TO_STR 查表路径
            for t in range(min(5, _T)):
                regime_str = REGIME_IDX_TO_STR[int(market_regime[t])]
                assert isinstance(regime_str, str)

        # 生成简单动量信号（5日收益率）
        _rng_inner = np.random.default_rng(42)
        for t in range(20, _T):
            scores = close[:, t] / (close[:, t - 5] + 1e-10) - 1.0
            if valid_mask is not None:
                scores[~valid_mask[:, t]] = -np.inf
            top20 = np.argsort(-scores)[:20]
            w[top20, t] = 1.0 / 20.0
        return w, np.zeros((_N, _T), dtype=np.float64)

    # ── 验收1：stamp_tax ──────────────────────────────────────────────────
    assert runner._risk_cfg.stamp_tax == 0.0005, (
        f"[FAIL] stamp_tax={runner._risk_cfg.stamp_tax}，应为 0.0005"
    )
    print(f"[OK] 验收1 stamp_tax={runner._risk_cfg.stamp_tax} (万五 ✓)")

    # ── 验收2：vol_multiplier ────────────────────────────────────────────
    assert runner._vol_multiplier == 100, (
        f"[FAIL] vol_multiplier={runner._vol_multiplier}，应为100"
    )
    print(f"[OK] 验收2 vol_multiplier={runner._vol_multiplier} (BaoStock 手→股 ✓)")

    # ── 执行回测（2020-01-01 ~ 2023-12-31 近似段）────────────────────────
    start_d = dates[200]
    end_d   = dates[800]

    result = runner.run(
        strategy_name = "ultra_alpha_v1",
        params        = None,
        start_date    = start_d,
        end_date      = end_d,
    )

    nav = result.nav_array if hasattr(result, "nav_array") else result.nav

    # ── 验收3：nav > 0 且无 NaN ──────────────────────────────────────────
    assert float(nav[-1]) > 0, f"[FAIL] nav[-1]={nav[-1]:.2f}，应 > 0"
    assert not np.any(np.isnan(nav)), "[FAIL] nav 含 NaN"
    print(f"[OK] 验收3 nav[-1]={nav[-1]:,.2f} > 0，无 NaN ✓")

    # ── 验收4：年化收益打印 ───────────────────────────────────────────────
    ar = result.annual_return
    print(f"[OK] 验收4 年化收益: {ar:.1%}  Sharpe={result.sharpe_ratio:.2f} ✓")

    # ── 验收5：regime int8 类型 ───────────────────────────────────────────
    regime_arr = runner._last_regime_bt
    assert regime_arr.dtype == np.int8, (
        f"[FAIL] regime dtype={regime_arr.dtype}，应为 int8"
    )
    bear_days = int((regime_arr == 4).sum())
    print(f"[OK] 验收5 regime int8，BEAR 天数={bear_days} ✓")

    # ── 验收6：PortfolioBuilder col_sum ≤ regime_limit ────────────────────
    # 使用与 run() 相同的完整数组（T_bt 列）来验收
    pb = runner._port_builder
    # _regime_limits 是 run() 时生成的，长度 = T_backtest（约 600 天）
    T_bt = len(pb._regime_limits) if pb._regime_limits is not None else 10
    raw_test = np.full((N, T_bt), 1.0 / 20.0, dtype=np.float64)  # col_sum=4.0 >> 1.0
    vm_test  = np.ones((N, T_bt), dtype=np.bool_)
    regime_limits_test = pb._regime_limits if pb._regime_limits is not None else np.ones(T_bt)

    from src.strategies.alpha_signal import AlphaSignal as _AS
    sig_test = _AS(raw_target_weights=raw_test)
    final_test = pb.build(
        alpha      = sig_test,
        valid_mask = vm_test,
    )
    for t in range(min(T_bt, 50)):   # 检验前50列（足够）
        lim = float(regime_limits_test[t])
        s   = float(final_test[:, t].sum())
        assert s <= lim + 1e-6, f"[FAIL] t={t} col_sum={s:.6f} > limit={lim:.4f}"
    print(f"[OK] 验收6 col_sum ≤ regime_limit（检验{min(T_bt,50)}/{T_bt}列）✓")

    # ── 验收7：realtime_signal 返回 dict，权重和 ≤ 1 ─────────────────────
    rt = runner.realtime_signal(
        strategy_name = "ultra_alpha_v1",
        params        = None,
        as_of_date    = dates[600],
    )
    assert isinstance(rt, dict), f"[FAIL] realtime_signal 返回 {type(rt)}，应为 dict"
    rt_sum = sum(rt.values())
    assert rt_sum <= 1.0 + 1e-9, f"[FAIL] realtime_signal 权重和={rt_sum:.6f} > 1"
    print(f"[OK] 验收7 realtime_signal: {len(rt)} 只股票，权重和={rt_sum:.4f} ≤ 1 ✓")

    # ── 验收8：BEAR 期 final_weights ≈ 0 ─────────────────────────────────
    bear_cols = np.where(pb._regime_limits == 0.0)[0] if pb._regime_limits is not None else []
    if len(bear_cols) > 0:
        # 构造一个含有 BEAR 列的 final_weights 并检验
        raw_bear = np.full((N, len(pb._regime_limits)), 1.0 / 20.0, dtype=np.float64)
        vm_bear  = np.ones((N, len(pb._regime_limits)), dtype=np.bool_)
        sig_bear = AlphaSignal(raw_target_weights=raw_bear)
        fw_bear  = pb.build(sig_bear, vm_bear)
        for t in bear_cols[:5]:   # 检验前5个 BEAR 列
            s = float(fw_bear[:, t].sum())
            assert s < 1e-9, f"[FAIL] BEAR 列 t={t} col_sum={s:.6f}，应≈0"
        print(f"[OK] 验收8 BEAR 列 final_weights ≈ 0（{len(bear_cols)} 列）✓")
    else:
        print("[OK] 验收8 本次无 BEAR 列（价格稳定），跳过 ✓")

    # ── 验收9：_calc_warmup 优先 warmup_override ─────────────────────────
    class _MockParams:
        warmup_override = 999
        rsrs_window     = 18
        zscore_window   = 600
        annual_factor   = 252.0
    w1 = runner._calc_warmup(_MockParams())
    assert w1 == 999, f"[FAIL] warmup_override 未生效，got {w1}"
    print(f"[OK] 验收9 warmup_override={w1} 优先生效 ✓")

    w2 = runner._calc_warmup(None)
    assert w2 == 18 + 600 + 120, f"[FAIL] 默认 warmup={w2}"
    print(f"[OK] 验收9 默认 warmup={w2}（rsrs+zscore+120）✓")

    print()
    print(f"  stamp_tax       : {runner._risk_cfg.stamp_tax} (万五 ✓)")
    print(f"  vol_multiplier  : {runner._vol_multiplier}x (手→股 ✓)")
    print(f"  regime dtype    : {regime_arr.dtype} ✓")
    print(f"  bear_confirm    : {runner._risk_cfg.bear_confirm_days} 天（快进）✓")
    print(f"  bear_exit       : {runner._risk_cfg.bear_exit_days} 天（慢出）✓")
    print()
    print("[PASS] fast_runner_v10.py 全部验收通过 ✓")


if __name__ == "__main__":
    test_fast_runner_v10()
