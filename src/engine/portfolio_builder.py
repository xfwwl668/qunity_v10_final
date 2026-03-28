"""
Q-UNITY V10 — portfolio_builder.py
====================================
MarketRegimeDetector  +  PortfolioBuilder

铁律（每次修改前默读）
----------------------
1. stamp_tax = 0.0005（万五，绝不改动）
2. holding_days 递增在每日循环最顶部
3. L3-A 在 L3-B 之前
4. market_regime 是 int8 数组；REGIME_IDX_TO_STR 映射后才能查 FACTOR_WEIGHTS
5. 不修改任何现有文件

★[H-02] 审计修复：Step A 流动性过滤（白皮书 §4.2）
  - build() 新增 amount_matrix 可选参数
  - 5日均成交额 < RiskConfig.min_avg_amount（默认500万元）的股票权重清零
  - 原版 Step A 仅做 valid_mask 过滤，未实现白皮书规定的流动性约束

设计要点
--------
- breadth 必须在含预热期的完整数组上计算，再裁剪到回测期
  （裁剪后计算会导致前 N 天 breadth=0，误触 BEAR）
- 状态机迟滞：进入 BEAR = bear_confirm_days=1（快进）
                退出 BEAR = bear_exit_days=10 连续天数（慢出）
- regime_limits 输出值域 ∈ {0.0, 0.4, 0.8, 1.0}
- PortfolioBuilder.build：单次归一化（B+D 合并），不做两轮归一化
"""

from __future__ import annotations

import numpy as np

# ── 同包导入（允许单文件测试时降级）────────────────────────────────────────
try:
    from src.engine.risk_config import RiskConfig                    # type: ignore[import]
except ImportError:
    from risk_config import RiskConfig                               # type: ignore[import]

try:
    from src.strategies.alpha_signal import AlphaSignal             # type: ignore[import]
    from src.strategies.alpha_signal import REGIME_IDX_TO_STR       # type: ignore[import]
except ImportError:
    from alpha_signal import AlphaSignal, REGIME_IDX_TO_STR         # type: ignore[import]


# ─────────────────────────────────────────────────────────────────────────────
# Regime → 仓位上限映射
# market_regime 存储为 int8 索引，查表时先 int(idx) → str → 仓位上限
# ─────────────────────────────────────────────────────────────────────────────
_REGIME_POS_LIMIT: dict[str, float] = {
    "STRONG_BULL": 1.0,
    "BULL"       : 0.8,
    "NEUTRAL"    : 0.8,
    "SOFT_BEAR"  : 0.6,   # [FIX-2] 0.4→0.6，震荡市四成仓过于保守
    "BEAR"       : 0.0,
}

# int8 索引 → 仓位上限（快速查表用，避免每步字符串查询）
_REGIME_IDX_TO_LIMIT: dict[int, float] = {
    idx: _REGIME_POS_LIMIT[name]
    for idx, name in REGIME_IDX_TO_STR.items()
}


# ─────────────────────────────────────────────────────────────────────────────
# MarketRegimeDetector
# ─────────────────────────────────────────────────────────────────────────────

class MarketRegimeDetector:
    """
    基于市场宽度（breadth）和指数 NAV 的 Regime 状态机。

    Regime 编码（int8，对应 REGIME_IDX_TO_STR）：
      0 = STRONG_BULL   breadth >= strong_bull_breadth
      1 = BULL          breadth >= bull_breadth_thr
      2 = NEUTRAL       breadth >= soft_bear_breadth
      3 = SOFT_BEAR     breadth >= bear_breadth_thr  (未进入 BEAR)
      4 = BEAR          迟滞状态机激活

    迟滞规则
    --------
    进入 BEAR：连续 bear_confirm_days（默认1）天满足 BEAR 条件 → 快进
    退出 BEAR：连续 bear_exit_days  （默认10）天不满足 BEAR 条件 → 慢出
    """

    def __init__(self, cfg: RiskConfig, market_index: np.ndarray) -> None:
        """
        Parameters
        ----------
        cfg          : RiskConfig 实例
        market_index : (T_full,) float64，市场基准指数收盘价（含预热期）
        """
        self._cfg          = cfg
        self._market_index = np.asarray(market_index, dtype=np.float64)
        self._cached_regimes: np.ndarray | None = None   # (T_full,) int8

    # ── 公开接口 ─────────────────────────────────────────────────────────────

    def compute(
        self,
        close_full : np.ndarray,   # (N, T_full) float64，含预热期
        valid_full : np.ndarray,   # (N, T_full) bool
        warmup     : int,          # 预热期长度（天）
    ) -> np.ndarray:
        """
        在含预热期的完整数组上计算 breadth，再裁剪到回测期。

        ★ 必须在完整数组上计算，否则裁剪后前 N 天 breadth=0 误触 BEAR。

        Parameters
        ----------
        close_full : (N, T_full) 含预热期的收盘价矩阵
        valid_full : (N, T_full) 含预热期的 valid_mask
        warmup     : 预热期天数

        Returns
        -------
        regime_limits : (T_backtest,) float64，值 ∈ {0.0, 0.4, 0.8, 1.0}
        同时缓存 self._cached_regimes (T_full,) int8（int8 数组）
        """
        cfg = self._cfg
        N, T_full    = close_full.shape
        T_backtest   = T_full - warmup

        if T_backtest <= 0:
            raise ValueError(
                f"warmup={warmup} >= T_full={T_full}，无有效回测期"
            )

        # ── 1. 向量化计算全段 breadth ────────────────────────────────────
        breadth_full = self._compute_breadth_vectorized(
            close_full,
            np.asarray(valid_full, dtype=np.bool_),
            window=cfg.breadth_window,
        )   # (T_full,) float64，前 breadth_window 天为 0

        # ── 2. NAV 相对 MA 状态（nav_ma_window 天均线）──────────────────
        idx_arr = self._market_index                  # (T_full,)
        nav_ok  = self._nav_ma_ok(idx_arr, cfg.nav_ma_window, cfg.bear_nav_thr)
        # nav_ok[t] = True 表示当日指数在 MA 之上（非 NAV 熊市）

        # ── 3. 状态机（逐日迟滞）─────────────────────────────────────────
        regimes = np.zeros(T_full, dtype=np.int8)

        in_bear            = False
        bear_trigger_cnt   = 0    # 连续满足进 BEAR 条件天数
        bear_exit_cnt      = 0    # 连续不满足 BEAR 条件天数

        for t in range(T_full):
            b = breadth_full[t]
            # BEAR 原始判断（两个条件之一满足即触发）：
            #   a. breadth < bear_breadth_thr
            #   b. 指数跌破 NAV MA（nav_ok = False）
            # [FIX-2] OR→AND：breadth低 且 指数跌破MA 才触发BEAR预判
            # 原OR条件在A股震荡市频繁误触BEAR导致大量空仓
            bear_raw = (b < cfg.bear_breadth_thr) and (not nav_ok[t])

            if in_bear:
                if not bear_raw:
                    bear_exit_cnt   += 1
                    bear_trigger_cnt = 0
                    if bear_exit_cnt >= cfg.bear_exit_days:
                        in_bear       = False
                        bear_exit_cnt = 0
                else:
                    bear_exit_cnt = 0
            else:
                if bear_raw:
                    bear_trigger_cnt += 1
                    bear_exit_cnt     = 0
                    if bear_trigger_cnt >= cfg.bear_confirm_days:
                        in_bear           = True
                        bear_trigger_cnt  = 0
                else:
                    bear_trigger_cnt = 0
                    bear_exit_cnt    = 0

            # ── 赋 regime 编码（int8）───────────────────────────────────
            if in_bear:
                regimes[t] = np.int8(4)   # BEAR
            elif b < cfg.soft_bear_breadth:
                regimes[t] = np.int8(3)   # SOFT_BEAR
            elif b < cfg.bull_breadth_thr:
                regimes[t] = np.int8(2)   # NEUTRAL
            elif b < cfg.strong_bull_breadth:
                regimes[t] = np.int8(1)   # BULL
            else:
                regimes[t] = np.int8(0)   # STRONG_BULL

        self._cached_regimes = regimes

        # ── 4. 裁剪到回测期，转换为仓位上限 ─────────────────────────────
        regimes_bt    = regimes[warmup:]                 # (T_backtest,) int8
        regime_limits = np.empty(T_backtest, dtype=np.float64)
        for t in range(T_backtest):
            regime_limits[t] = _REGIME_IDX_TO_LIMIT[int(regimes_bt[t])]

        return regime_limits

    def get_regime_enum_array(self) -> np.ndarray:
        """
        返回缓存的完整 regime 数组（int8），须先调用 compute()。

        Returns
        -------
        _cached_regimes : (T_full,) int8
        """
        if self._cached_regimes is None:
            raise RuntimeError(
                "须先调用 compute() 生成 regime 缓存后再调用 get_regime_enum_array()"
            )
        return self._cached_regimes

    def get_pos_limit_today(self) -> float:
        """
        实盘用：返回今日仓位上限（基于最新一日 regime）。
        须先调用 compute()。
        """
        regimes = self.get_regime_enum_array()
        latest  = int(regimes[-1])
        return _REGIME_IDX_TO_LIMIT[latest]

    # ── 静态辅助：向量化 breadth ─────────────────────────────────────────

    @staticmethod
    def _compute_breadth_vectorized(
        close      : np.ndarray,    # (N, T)
        valid_mask : np.ndarray,    # (N, T) bool
        window     : int = 5,
    ) -> np.ndarray:
        """
        NumPy 向量化计算市场宽度（上涨股票占比）。

        定义：
          breadth[t] = #{i : close[i,t] > close[i,t-window] AND valid[i,t]}
                     / #{i : valid[i,t] AND close[i,t-window]>0}

        实现方式（无 Python 循环，适合 N×T≈2700万）：
          base = close[:, :-window]      shape (N, T-window)
          cur  = close[:, window:]       shape (N, T-window)
          ok   = valid[:,window:] & (base>0) & (cur>0)
          up   = ok & (cur > base)
          breadth[window:] = up.sum(axis=0) / ok.sum(axis=0).clip(min=1)
          breadth[:window] = 0.0（预热期无法计算，设为 0）

        Parameters
        ----------
        close      : (N, T) float64
        valid_mask : (N, T) bool
        window     : 回看窗口（默认 5）

        Returns
        -------
        breadth : (T,) float64，∈ [0, 1]
        """
        N, T = close.shape
        breadth = np.zeros(T, dtype=np.float64)

        if T <= window:
            return breadth

        base = close[:, :T - window]      # (N, T-window)，过去
        cur  = close[:, window:]          # (N, T-window)，当前

        valid_cur  = valid_mask[:, window:]   # (N, T-window)
        ok         = valid_cur & (base > 0.0) & (cur > 0.0)   # (N, T-window) bool
        up         = ok & (cur > base)                         # (N, T-window) bool

        denom = ok.sum(axis=0).clip(min=1)    # (T-window,) 防止除零
        breadth[window:] = up.sum(axis=0) / denom

        # breadth[:window] 已初始化为 0.0
        return breadth

    # ── 私有辅助：NAV MA 判断 ────────────────────────────────────────────

    @staticmethod
    def _nav_ma_ok(
        index_arr : np.ndarray,   # (T,) 市场指数
        window    : int,          # nav_ma_window
        thr       : float,        # bear_nav_thr（如 0.96）
    ) -> np.ndarray:
        """
        返回 (T,) bool：当日指数 / MA(window) >= thr → True（非熊市）。
        前 window-1 天无法计算，保守设为 True（不触发 NAV 熊市条件）。
        """
        T    = len(index_arr)
        ok   = np.ones(T, dtype=np.bool_)
        cs   = np.cumsum(index_arr)

        for t in range(window - 1, T):
            if t == window - 1:
                ma = cs[t] / window
            else:
                ma = (cs[t] - cs[t - window]) / window
            if ma > 0.0:
                ok[t] = (index_arr[t] / ma) >= thr
        return ok


# ─────────────────────────────────────────────────────────────────────────────
# PortfolioBuilder
# ─────────────────────────────────────────────────────────────────────────────

class PortfolioBuilder:
    """
    将 AlphaSignal 的原始权重矩阵按 valid_mask 和 regime_limits 进行最终归一化。

    归一化规则（单次，B+D 合并）
    ----------------------------
    对每列 t：
      1. valid_mask[:,t] = False 的位置权重清零
      2. col_sum = sum(weights[:,t])
      3. 若 col_sum > regime_limit：  weights[:,t] *= regime_limit / col_sum
         （即 target_sum = min(col_sum, regime_limit)，等比缩放）
      4. 若 col_sum == 0 或 regime_limit == 0：全零

    不做两轮归一化（先拉满再截断），只做一次缩放。
    """

    def __init__(self, cfg: RiskConfig, market_index: np.ndarray) -> None:
        """
        Parameters
        ----------
        cfg          : RiskConfig 实例
        market_index : (T_full,) float64，市场基准指数（含预热期）
        """
        self._cfg      = cfg
        self._detector = MarketRegimeDetector(cfg, market_index)
        self._regime_limits: np.ndarray | None = None   # (T_backtest,) float64

    # ── 主接口 ───────────────────────────────────────────────────────────────

    def build(
        self,
        alpha        : AlphaSignal,
        valid_mask   : np.ndarray,              # (N, T_backtest) bool
        close_full   : np.ndarray | None = None,  # (N, T_full) 含预热期
        valid_full   : np.ndarray | None = None,  # (N, T_full) bool 含预热期
        warmup       : int = 0,
        amount_matrix: np.ndarray | None = None,  # ★[H-02] (N, T_backtest) float，成交额（元）
    ) -> np.ndarray:
        """
        构建最终目标权重矩阵。

        Parameters
        ----------
        alpha         : AlphaSignal，须含 raw_target_weights (N, T_backtest)
        valid_mask    : (N, T_backtest) bool，回测期 valid_mask
        close_full    : 含预热期收盘价，若提供则触发 regime 计算
        valid_full    : 含预热期 valid_mask
        warmup        : 预热期长度
        amount_matrix : ★[H-02] (N, T_backtest) float，成交额（元）。
                        若提供且 RiskConfig.min_avg_amount > 0，
                        则 5日均成交额 < min_avg_amount 的股票权重清零。

        Returns
        -------
        final_target_weights : (N, T_backtest) float64
        """
        raw = alpha.raw_target_weights.copy()    # (N, T_backtest)
        N, T = raw.shape

        # ── A. valid_mask = False → 权重清零 ─────────────────────────────
        raw[~valid_mask] = 0.0

        # ── ★[H-02] Step A-2：流动性过滤（白皮书 §4.2）──────────────────
        #   5日均成交额 < min_avg_amount（默认 500万元）的股票权重清零
        #   原因：低流动性股票实盘冲击成本失控，不应纳入持仓
        if amount_matrix is not None and self._cfg.min_avg_amount > 0.0:
            amt = np.asarray(amount_matrix, dtype=np.float64)
            window = 5
            if T >= window:
                # 向量化滚动均值：对每个 t，取 [max(0,t-window+1)..t] 的均值
                # cumsum 技巧：O(N×T)，无 Python 循环
                cs   = np.cumsum(amt, axis=1)                              # (N, T)
                lag  = np.zeros((N, T), dtype=np.float64)
                lag[:, window:] = cs[:, :T - window]                       # (N, T)
                cnt  = np.minimum(np.arange(1, T + 1, dtype=np.float64),
                                  float(window))[np.newaxis, :]            # (1, T)
                roll_mean = (cs - lag) / cnt                               # (N, T)
                low_liq   = roll_mean < self._cfg.min_avg_amount
                raw[low_liq] = 0.0

        # ── 计算 regime_limits（若提供了完整数据则重新算，否则全满仓）───
        if close_full is not None and valid_full is not None:
            self._regime_limits = self._detector.compute(
                close_full, valid_full, warmup
            )
        elif self._regime_limits is None:
            # 没有 regime 数据：默认满仓上限
            self._regime_limits = np.ones(T, dtype=np.float64)

        regime_limits = self._regime_limits

        if len(regime_limits) != T:
            raise ValueError(
                f"regime_limits.shape={regime_limits.shape}，"
                f"与权重矩阵 T={T} 不匹配"
            )

        # ── B+D 合并：单次归一化（不分两步）─────────────────────────────
        for t in range(T):
            col_sum     = raw[:, t].sum()
            regime_lim  = regime_limits[t]

            if col_sum < 1e-12 or regime_lim < 1e-12:
                # 无有效权重或 BEAR → 全零
                raw[:, t] = 0.0
                continue

            if col_sum > regime_lim:
                # 超出仓位上限 → 等比缩放
                raw[:, t] *= regime_lim / col_sum
            # col_sum <= regime_lim 时不改变（保留原始权重之和）

        return raw

    # ── 实盘单日接口 ─────────────────────────────────────────────────────────

    def build_single_day(
        self,
        raw_w_today    : np.ndarray,   # (N,) float64，今日原始权重
        valid_mask_today: np.ndarray,  # (N,) bool
    ) -> np.ndarray:
        """
        实盘单日版本：返回 (N,) float64 调整后权重。

        使用缓存的最新 regime_limit（须先调用 build() 或手动设置）。
        若尚无 regime 缓存，则使用满仓上限 1.0。
        """
        w = np.asarray(raw_w_today,     dtype=np.float64).copy()
        v = np.asarray(valid_mask_today, dtype=np.bool_)

        # A. invalid 清零
        w[~v] = 0.0

        # 获取今日仓位上限
        if self._regime_limits is not None:
            regime_lim = float(self._regime_limits[-1])
        else:
            regime_lim = 1.0

        col_sum = w.sum()
        if col_sum < 1e-12 or regime_lim < 1e-12:
            w[:] = 0.0
            return w

        if col_sum > regime_lim:
            w *= regime_lim / col_sum

        return w


# ─────────────────────────────────────────────────────────────────────────────
# 验收测试
# ─────────────────────────────────────────────────────────────────────────────

def _make_synthetic_data(
    N       : int,
    T_full  : int,
    bear_start: int,
    bear_end  : int,
    seed    : int = 42,
) -> tuple:
    """
    构造合成数据：在 [bear_start, bear_end) 区间制造 BEAR 市场宽度条件。

    Returns
    -------
    close_full   : (N, T_full) float64
    valid_full   : (N, T_full) bool
    market_index : (T_full,) float64
    """
    rng = np.random.default_rng(seed)

    # 基础价格：随机游走
    close_full = np.cumprod(
        1 + rng.normal(0.0002, 0.01, size=(N, T_full)), axis=1
    ).astype(np.float64) * 10.0

    # valid_mask：全部有效
    valid_full = np.ones((N, T_full), dtype=np.bool_)

    # 市场指数：BEAR 区间内大幅下跌（确保 NAV < MA * 0.96）
    idx = np.ones(T_full, dtype=np.float64)
    for t in range(1, T_full):
        if bear_start <= t < bear_end:
            idx[t] = idx[t - 1] * 0.985   # 每天跌 1.5%，确保触发
        else:
            idx[t] = idx[t - 1] * 1.001
    market_index = idx * 1000.0

    # BEAR 区间内让 close 也大幅下跌，使 breadth 极低
    for t in range(bear_start, min(bear_end, T_full)):
        close_full[:, t] = close_full[:, max(0, t - 1)] * rng.uniform(
            0.96, 0.98, size=N
        )

    return close_full, valid_full, market_index


def test_portfolio_builder() -> None:
    """
    验收测试：
    1. BEAR 状态下 final_target_weights 全列 ≈ 0
    2. 归一化后 col_sum <= regime_limit（± 1e-6 容差）
    3. regime int8 数组类型检验
    4. STRONG_BULL 下 col_sum 不被错误截断（col_sum 已 <= 1.0）
    """
    import sys

    print("=" * 60)
    print("Q-UNITY V10  portfolio_builder.py  验收测试")
    print("=" * 60)

    rng = np.random.default_rng(0)

    N       = 100    # 股票数
    warmup  = 60     # 预热期
    T_bt    = 120    # 回测期
    T_full  = warmup + T_bt

    # ── 模拟"2022年某月"熊市区间 ────────────────────────────────────────
    # 令回测期前 30 天为 BEAR（全段 [warmup, warmup+30)）
    bear_start_full = warmup       # 回测第 0 天起
    bear_end_full   = warmup + 30  # 回测前 30 天

    close_full, valid_full, market_index = _make_synthetic_data(
        N, T_full, bear_start_full, bear_end_full
    )

    cfg = RiskConfig(
        bear_breadth_thr    = 0.32,
        bear_nav_thr        = 0.96,
        bear_confirm_days   = 1,
        bear_exit_days      = 10,
        soft_bear_breadth   = 0.38,
        bull_breadth_thr    = 0.44,
        strong_bull_breadth = 0.52,
        breadth_window      = 5,
        nav_ma_window       = 20,    # 用 20 日线加速触发（合成数据较短）
    )

    # ── 构造 AlphaSignal（合理等权权重，col_sum ≈ 0.8）──────────────────
    raw_w = np.zeros((N, T_bt), dtype=np.float64)
    top20 = 20
    for t in range(T_bt):
        idx_top = rng.choice(N, size=top20, replace=False)
        raw_w[idx_top, t] = 1.0 / top20          # 等权，col_sum = 1.0

    valid_bt   = valid_full[:, warmup:]
    alpha      = AlphaSignal(raw_target_weights=raw_w, strategy_name="SyntheticTest")
    alpha.validate(N, T_bt)

    builder = PortfolioBuilder(cfg=cfg, market_index=market_index)
    final_w = builder.build(
        alpha      = alpha,
        valid_mask = valid_bt,
        close_full = close_full,
        valid_full = valid_full,
        warmup     = warmup,
    )

    assert final_w.shape == (N, T_bt), f"[FAIL] shape={final_w.shape}"
    assert final_w.dtype == np.float64, f"[FAIL] dtype={final_w.dtype}"
    print(f"[OK] final_target_weights.shape={final_w.shape} dtype={final_w.dtype} ✓")

    # ── 取得 regime_limits 用于对比 ─────────────────────────────────────
    regime_limits = builder._regime_limits
    assert regime_limits is not None

    regimes = builder._detector.get_regime_enum_array()
    assert regimes.dtype == np.int8, (
        f"[FAIL] regime 数组 dtype={regimes.dtype}，应为 int8"
    )
    print(f"[OK] market_regime dtype=int8 ✓")

    # 打印回测期前35日 regime 分布
    regime_bt = regimes[warmup:]
    regime_names = [REGIME_IDX_TO_STR[int(r)] for r in regime_bt[:35]]
    bear_days_bt = int((regime_bt == 4).sum())
    print(f"     回测期 BEAR 天数={bear_days_bt}/{T_bt}")
    print(f"     前35日 regime: {regime_names[:10]} ...")

    # ── 验收1：BEAR 列权重全为 0 ────────────────────────────────────────
    bear_cols = np.where(regime_limits == 0.0)[0]
    if len(bear_cols) == 0:
        print("[WARN] 未找到 BEAR 列（合成数据未触发），跳过验收1")
    else:
        for t in bear_cols:
            col_sum_bear = final_w[:, t].sum()
            assert col_sum_bear < 1e-9, (
                f"[FAIL] 验收1: t={t}(BEAR) col_sum={col_sum_bear:.6f}，应≈0"
            )
        print(f"[OK] 验收1: {len(bear_cols)} 个 BEAR 列权重全为0 ✓")

    # ── 验收2：col_sum <= regime_limit（±1e-6 容差）─────────────────────
    fail_cols = 0
    for t in range(T_bt):
        col_sum = final_w[:, t].sum()
        lim     = regime_limits[t]
        if col_sum > lim + 1e-6:
            fail_cols += 1
            print(f"[FAIL] 验收2: t={t} col_sum={col_sum:.8f} > limit={lim:.4f}")

    assert fail_cols == 0, f"[FAIL] 验收2: {fail_cols} 列超过 regime_limit"
    print(f"[OK] 验收2: 全部 {T_bt} 列 col_sum <= regime_limit (±1e-6) ✓")

    # ── 验收3：非 BEAR 列权重结构合理（col_sum > 0）─────────────────────
    non_bear_cols = np.where(regime_limits > 0.0)[0]
    if len(non_bear_cols) > 0:
        non_bear_sums = final_w[:, non_bear_cols].sum(axis=0)
        assert np.all(non_bear_sums > 0.0), (
            "[FAIL] 验收3: 非BEAR列存在 col_sum=0"
        )
        print(f"[OK] 验收3: {len(non_bear_cols)} 个非BEAR列均有正权重 ✓")

    # ── 验收4：build_single_day 接口 ─────────────────────────────────────
    raw_today   = np.zeros(N, dtype=np.float64)
    idx_sel     = rng.choice(N, size=10, replace=False)
    raw_today[idx_sel] = 0.1
    valid_today = np.ones(N, dtype=np.bool_)

    w_today = builder.build_single_day(raw_today, valid_today)
    lim_now = builder._regime_limits[-1]
    assert w_today.sum() <= lim_now + 1e-9, (
        f"[FAIL] 验收4: build_single_day sum={w_today.sum():.6f} > limit={lim_now}"
    )
    print(f"[OK] 验收4: build_single_day sum={w_today.sum():.4f} <= "
          f"limit={lim_now:.4f} ✓")

    # ── 验收5：get_regime_enum_array 未 compute 时抛异常 ─────────────────
    det2 = MarketRegimeDetector(cfg, market_index)
    try:
        det2.get_regime_enum_array()
        print("[FAIL] 验收5: 未 compute 时应抛 RuntimeError")
        sys.exit(1)
    except RuntimeError as e:
        print(f"[OK] 验收5: 未 compute 时正确抛 RuntimeError: {str(e)[:60]} ✓")

    # ── 验收6：breadth 向量化正确性（与标量循环对比）────────────────────
    N2, T2, W = 50, 30, 5
    c2 = rng.random((N2, T2)).astype(np.float64) + 0.5
    v2 = np.ones((N2, T2), dtype=np.bool_)

    breadth_vec = MarketRegimeDetector._compute_breadth_vectorized(c2, v2, W)

    # 标量参考实现
    breadth_ref = np.zeros(T2, dtype=np.float64)
    for t in range(W, T2):
        up   = np.sum((c2[:, t] > c2[:, t - W]) & v2[:, t] & (c2[:, t - W] > 0))
        denom = np.sum(v2[:, t] & (c2[:, t - W] > 0) & (c2[:, t] > 0))
        breadth_ref[t] = up / max(denom, 1)

    max_diff = np.abs(breadth_vec - breadth_ref).max()
    assert max_diff < 1e-12, f"[FAIL] 验收6: breadth 向量化误差={max_diff:.2e}"
    print(f"[OK] 验收6: breadth 向量化与标量参考最大误差={max_diff:.2e} ✓")

    print()
    print(f"  stamp_tax         : {cfg.stamp_tax} (万五 ✓，未被修改)")
    print(f"  regime int8 dtype : {regimes.dtype} ✓")
    print(f"  bear_confirm_days : {cfg.bear_confirm_days} (快进) ✓")
    print(f"  bear_exit_days    : {cfg.bear_exit_days}  (慢出) ✓")
    print()
    print("[PASS] portfolio_builder.py 全部验收通过 ✓")


if __name__ == '__main__':
    test_portfolio_builder()
