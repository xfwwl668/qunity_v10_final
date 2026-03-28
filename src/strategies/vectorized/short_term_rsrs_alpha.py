"""
Q-UNITY V10 — short_term_rsrs_alpha.py
========================================
ShortTermRSRS 纯因子版（AlphaSignal 输出）

迁移自 short_term_rsrs_vec.py，删除风控层，保留完整因子流水线：

✘ 删除：熊市保护（breadth + NAV/MA 双重门控）/ ever_bought / dropout
✘ 删除：ATR 追踪止损循环（entry_price 状态机）
✘ 删除：冷却计数器（cool_down）/ sell_signals 双轨
✓ 保留：RSRS OLS Beta Z-Score（来自 alpha_hunter_v2 内核）
✓ 保留：ATR（Wilder EMA, window=14）→ 风险预算仓位
✓ 保留：5日动量门控（> mom_threshold）
✓ 保留：量比门控（> volume_confirm_ratio）
✓ 保留：Top-N 截面内 ATR 归一化权重（源码 N11-FIX）
✓ valid_mask=False → score=-inf → weight=0

仓位计算（与源码 N11-FIX 对齐）：
  raw_risk_weights = risk_budget / (atr_pct × atr_multiplier)
  Top-N 截面内归一化（相对差异保留：高波动→低权重，低波动→高权重）
"""
from __future__ import annotations

import numpy as np
from typing import Optional

try:
    from src.strategies.alpha_signal import AlphaSignal, _score_to_weights, _ema_smooth_factor
except ImportError:
    from alpha_signal import AlphaSignal, _score_to_weights          # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# 因子内核（忠实移植）
# ─────────────────────────────────────────────────────────────────────────────


# ── [FIX-S-01] 策略注册导入 ───────────────────────────────────────────────────
try:
    from src.strategies.registry import register_vec_strategy
except ImportError:
    try:
        from strategies.registry import register_vec_strategy  # type: ignore
    except ImportError:
        def register_vec_strategy(name):  # type: ignore
            """Fallback stub when registry not importable."""
            def _decorator(fn): return fn
            return _decorator


def _rsrs_beta_r2(high: np.ndarray, low: np.ndarray, window: int) -> tuple:
    """
    RSRS OLS Beta + R²（与 alpha_hunter_v2_vec._compute_rsrs_matrix 对齐）。
    窗口含当日：[t-window+1:t+1]。
    """
    N, T = high.shape
    beta = np.full((N, T), np.nan, dtype=np.float64)
    r2   = np.full((N, T), np.nan, dtype=np.float64)
    for t in range(window - 1, T):
        s  = max(0, t - window + 1)
        x  = low[:,  s:t + 1]
        y  = high[:, s:t + 1]
        xm = x.mean(1, keepdims=True)
        ym = y.mean(1, keepdims=True)
        dx, dy = x - xm, y - ym
        sxy = (dx * dy).sum(1)
        sxx = (dx * dx).sum(1)
        syy = (dy * dy).sum(1)
        ok  = sxx > 1e-12
        beta[ok, t] = sxy[ok] / sxx[ok]
        ok2 = ok & (syy > 1e-12)
        ss_res = syy[ok2] - sxy[ok2] ** 2 / sxx[ok2]
        r2[ok2, t] = np.clip(1.0 - ss_res / syy[ok2], 0.0, 1.0)
    return beta, r2


def _rsrs_zscore(beta: np.ndarray, window: int) -> np.ndarray:
    """时序滚动 Z-Score（样本标准差，与 _compute_rsrs_zscore 对齐）。"""
    N, T = beta.shape
    z = np.full((N, T), np.nan, dtype=np.float64)
    for t in range(window, T):
        w   = beta[:, t - window:t]
        cnt = (~np.isnan(w)).sum(1)
        ok  = cnt >= 30
        mu  = np.nanmean(w, axis=1)
        std = np.nanstd(w, axis=1, ddof=1)
        cur = beta[:, t]
        v   = ok & (std > 1e-10) & ~np.isnan(cur)
        z[v, t] = (cur[v] - mu[v]) / std[v]
    return z


def _compute_atr(
    high : np.ndarray,
    low  : np.ndarray,
    close: np.ndarray,
    window: int = 14,
) -> np.ndarray:
    """
    ATR（Wilder EMA）：忠实移植自 _compute_atr_batch（含 BUG-J Fix）。
    第 0 日用 H-L 近似，t<window 暂存 TR，t==window 取 SMA 初始化，
    t>window 用 Wilder EMA = (prev × (w-1) + TR) / w。
    """
    N, T = high.shape
    atr = np.full((N, T), np.nan, dtype=np.float64)

    for i in range(N):
        hl0 = high[i, 0] - low[i, 0]
        atr[i, 0] = hl0 if hl0 > 0.0 else 1e-6

        for t in range(1, T):
            hl = high[i, t] - low[i, t]
            hc = abs(high[i, t] - close[i, t - 1])
            lc = abs(low[i,  t] - close[i, t - 1])
            tr = max(hl, hc, lc)

            if t < window:
                atr[i, t] = tr
            elif t == window:
                tr_sum = sum(atr[i, t - k] for k in range(1, window + 1))
                atr[i, t] = tr_sum / window
            else:
                prev = atr[i, t - 1]
                atr[i, t] = (prev * (window - 1) + tr) / window \
                             if not np.isnan(prev) else tr

    return atr


def _vol_ma(volume: np.ndarray, window: int = 20) -> np.ndarray:
    """
    批量成交量均值，(N,T)。
    [FIX-S-03] 窗口[t-window:t]不含当日，与V91对齐。
    """
    N, T = volume.shape
    vm = np.full((N, T), np.nan, dtype=np.float64)
    if T <= window:
        return vm
    cs = np.cumsum(volume, axis=1)
    vm[:, window] = cs[:, window - 1] / window
    if T > window + 1:
        vm[:, window + 1:] = (cs[:, window:-1] - cs[:, :-window - 1]) / window
    return vm


# ─────────────────────────────────────────────────────────────────────────────
# 主 Alpha 函数
# ─────────────────────────────────────────────────────────────────────────────

@register_vec_strategy("short_term_rsrs")
def short_term_rsrs_alpha(
    close        : np.ndarray,
    open_        : np.ndarray,
    high         : np.ndarray,
    low          : np.ndarray,
    volume       : np.ndarray,
    params,
    valid_mask   : Optional[np.ndarray] = None,
    market_regime: Optional[np.ndarray] = None,
    **kw,
) -> AlphaSignal:
    """
    ShortTermRSRS 纯因子版。

    入场门控（源码完全对齐）：
      G1: RSRS Z-Score > rsrs_threshold
      G2: 5日动量      > mom_threshold
      G3: 量比         > volume_confirm_ratio
      G4: valid_mask   = True

    score = RSRS Z-Score（门控通过后），门控不通过则 -inf。
    仓位 = risk_budget / (atr_pct × atr_multiplier)，Top-N 内归一化（N11-FIX）。
    """
    N, T = close.shape

    rsrs_w       = int(getattr(params, "rsrs_window",          18))  if params else 18
    zscore_w     = int(getattr(params, "zscore_window",        600)) if params else 600
    rsrs_thresh  = float(getattr(params, "rsrs_threshold",      0.7)) if params else 0.7
    mom_thresh   = float(getattr(params, "mom_threshold",       0.03))if params else 0.03
    vol_confirm  = float(getattr(params, "volume_confirm_ratio",1.5)) if params else 1.5
    risk_budget  = float(getattr(params, "risk_budget",         0.02))if params else 0.02
    atr_mult     = float(getattr(params, "atr_multiplier",      2.0)) if params else 2.0
    top_n        = int(getattr(params, "top_n",                 20))  if params else 20
    max_s_pos    = float(getattr(params, "max_single_pos",      0.20))if params else 0.20

    # ── F1: RSRS ─────────────────────────────────────────────────────────────
    beta_mat, _r2 = _rsrs_beta_r2(high, low, rsrs_w)
    rsrs_z        = _rsrs_zscore(beta_mat, zscore_w)

    # ── F2: ATR（风险预算仓位基础）───────────────────────────────────────────
    atr = _compute_atr(high, low, close, window=14)

    # ── F3: 5日动量 ──────────────────────────────────────────────────────────
    mom_w    = 5
    momentum = np.full((N, T), np.nan, dtype=np.float64)
    if T > mom_w:
        with np.errstate(divide="ignore", invalid="ignore"):
            momentum[:, mom_w:] = (
                close[:, mom_w:] / (close[:, :T - mom_w] + 1e-10) - 1.0
            )

    # ── F4: 量比（当日 / 20日均量）───────────────────────────────────────────
    vol_ma_arr  = _vol_ma(volume, window=20)
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_ratio = volume / (vol_ma_arr + 1e-10)

    # ── 门控 ─────────────────────────────────────────────────────────────────
    gate_rsrs = rsrs_z > rsrs_thresh
    gate_mom  = momentum > mom_thresh
    gate_vol  = vol_ratio > vol_confirm

    buy_candidate = gate_rsrs & gate_mom & gate_vol
    if valid_mask is not None:
        vm = np.asarray(valid_mask, dtype=bool)
        buy_candidate &= vm
    else:
        vm = np.ones((N, T), dtype=bool)

    # ── 风险预算原始权重（源码 N11-FIX）──────────────────────────────────────
    with np.errstate(divide="ignore", invalid="ignore"):
        atr_pct       = atr / (close + 1e-10)
        atr_pct_safe  = np.where(atr_pct > 1e-8, atr_pct, np.nan)
        raw_risk_w    = risk_budget / (atr_pct_safe * atr_mult + 1e-10)
        equal_w       = 1.0 / max(top_n, 1)
        raw_risk_w    = np.where(np.isfinite(raw_risk_w), raw_risk_w, equal_w)

    # ── score = RSRS Z-Score（门控过滤）──────────────────────────────────────
    # [FIX-EMA-02] 对 RSRS Z-Score 做因子级 EMA 平滑（降低截面排名波动）
    _ema = int(getattr(params, "factor_ema_span", 5)) if params else 5
    rsrs_z_s = _ema_smooth_factor(rsrs_z, _ema) if _ema > 1 else rsrs_z
    score = np.where(buy_candidate, np.nan_to_num(rsrs_z_s, nan=-np.inf), -np.inf)

    # [L4-FIX] 涨跌停硬封锁：T日封死涨/跌停点位在T+1无法成交
    limit_mask = kw.get("limit_reach_mask")
    if limit_mask is not None:
        score[limit_mask] = -np.inf

    # [L4-FIX] 数值稳态保护
    score = np.nan_to_num(score, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)

    # ── Top-N 选股 + 截面 ATR 归一化权重 ─────────────────────────────────────
    # [FIX-BUG1] 原循环直接写 raw_weights[idx,t]=w，绕过 _score_to_weights 防抖，
    # 导致年换手率 8000%+。修复方案：
    #   Step-1: _score_to_weights 负责防抖选股（得到布尔持仓掩码）。
    #   Step-2: 对防抖选中的股票，在截面内用 ATR 归一化权重覆盖（保留 N11-FIX 精神）。
    #   Step-3: 列归一化确保 col_sum ≤ 1。
    # 注：max_single_pos=1.0 让防抖层不截断权重，截断在 Step-2 的 min(w, max_s_pos) 完成。
    # [FIX-DB-01] 策略级防抖参数：可通过 params 或 config.json.strategy_params 配置
    dropout_d = 2   # ★[FIX-DEBOUNCE] 策略专属，来自 exit_config 规格
    exit_buf  = 3   # ★[FIX-DEBOUNCE] 策略专属，来自 exit_config 规格
    debounced_mask = _score_to_weights(
        score, top_n=top_n, max_single_pos=1.0,
        dropout_days=dropout_d, exit_buffer=exit_buf,
        hard_invalid=None if valid_mask is None else ~np.asarray(valid_mask, dtype=bool),
    ) > 0

    raw_weights = np.zeros((N, T), dtype=np.float64)
    for t in range(T):
        top_idx = np.where(debounced_mask[:, t])[0]
        if len(top_idx) == 0:
            continue

        # 截面 ATR 归一化（N11-FIX）
        raw_top = raw_risk_w[top_idx, t]
        raw_sum = raw_top.sum()
        if raw_sum > 1e-9:
            norm_top = raw_top / raw_sum
        else:
            norm_top = np.full(len(top_idx), 1.0 / len(top_idx))

        for ri, idx in enumerate(top_idx):
            w = float(norm_top[ri])
            w = max(w, 1.0 / len(top_idx)) if w < 1e-9 else w
            w = min(w, max_s_pos)
            raw_weights[idx, t] = w

        # 列归一化确保 col_sum ≤ 1
        cs = raw_weights[:, t].sum()
        if cs > 1.0 + 1e-9:
            raw_weights[:, t] /= cs

    return AlphaSignal(
        raw_target_weights=raw_weights,
        score=score,
        strategy_name="short_term_rsrs",
        exit_config={
            "stop_mode"       : "entry_price",
            "hard_stop_loss"  : 0.10,
            "take_profit"     : 0.18,
            "max_holding_days": 10,
            "dropout_days"    : 2,
            "exit_buffer"     : 3,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# 验收测试
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    rng = np.random.default_rng(5)
    N, T = 50, 300
    c = np.cumprod(1 + rng.normal(0.0003, 0.018, (N, T)), axis=1).astype(np.float64) * 10
    h = c * (1 + rng.uniform(0.002, 0.028, (N, T)))
    l = c * (1 - rng.uniform(0.002, 0.028, (N, T)))
    v = rng.uniform(1e6, 5e7, (N, T))

    alpha = short_term_rsrs_alpha(c, c, h, l, v, None)

    assert alpha.raw_target_weights.shape == (N, T), \
        f"shape FAIL: {alpha.raw_target_weights.shape}"
    assert not np.any(alpha.raw_target_weights < 0), "negative weight FAIL"
    assert alpha.raw_target_weights.sum(axis=0).max() <= 1.0 + 1e-6, \
        f"col_sum FAIL: {alpha.raw_target_weights.sum(axis=0).max():.6f}"

    nz = (alpha.raw_target_weights.sum(axis=0) > 0).sum()
    print(f"[PASS] short_term_rsrs_alpha: shape={alpha.raw_target_weights.shape} "
          f"max_col_sum={alpha.raw_target_weights.sum(axis=0).max():.4f} "
          f"nonzero_cols={nz}/{T} ✓")
    sys.exit(0)
