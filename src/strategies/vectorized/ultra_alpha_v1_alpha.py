"""
Q-UNITY V10 — ultra_alpha_v1_alpha.py
======================================
UltraAlpha V1 纯因子版（AlphaSignal 输出）

设计变更（相对 ultra_alpha_v1_vec.py）：
  ✘ 删除：熊市全仓卖出 / ever_bought / dropout / 持仓天数控制
  ✘ 删除：sell_signals / buy_signals 双轨信号
  ✓ 保留：F1 RSRS + F2 动量 + F3 反转 + F4 量价聪明钱
  ✓ 保留：动态因子权重（由 breadth 决定，不再清仓）
  ✓ 返回：AlphaSignal（raw_target_weights + score）

有效性规则：
  - valid_mask=False → score = -inf → weight = 0
  - 减仓用 raw_weights[mask] *= 0.5（不使用 exit_ratio 字段）
  - col_sum ≤ 1.0（由 _score_to_weights 保证）
"""
from __future__ import annotations

import numpy as np
from typing import Optional

# ── AlphaSignal & 工具函数 ────────────────────────────────────────────────────
try:
    from src.strategies.alpha_signal import AlphaSignal, _score_to_weights, _ema_smooth_factor
except ImportError:
    from alpha_signal import AlphaSignal, _score_to_weights          # type: ignore

# ── Numba 可选（无则纯 NumPy）────────────────────────────────────────────────
try:
    from numba import njit, prange
    _NUMBA = True
except ImportError:
    _NUMBA = False
    def njit(*a, **kw):
        return (lambda fn: fn) if not a else (a[0] if callable(a[0]) else lambda fn: fn)
    prange = range


# ─────────────────────────────────────────────────────────────────────────────
# 纯 NumPy 因子内核（供无 Numba 环境使用）
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


def _rsrs_beta_numpy(high: np.ndarray, low: np.ndarray, window: int = 18) -> np.ndarray:
    """滑动窗口 OLS Beta：以 low 为 x，high 为 y。(N,T) → (N,T)"""
    N, T = high.shape
    beta = np.full((N, T), np.nan, dtype=np.float64)
    for t in range(window, T):
        x = low[:, t - window:t]           # (N, window)
        y = high[:, t - window:t]
        xm = x.mean(axis=1, keepdims=True)
        ym = y.mean(axis=1, keepdims=True)
        dx = x - xm
        dy = y - ym
        sxy = (dx * dy).sum(axis=1)
        sxx = (dx * dx).sum(axis=1)
        mask = sxx > 1e-12
        beta[mask, t] = sxy[mask] / sxx[mask]
    return beta


def _rolling_zscore_numpy(
    mat: np.ndarray,
    window: int = 300,
    min_count: int = 30,
) -> np.ndarray:
    """时序滚动 Z-Score，(N,T) → (N,T)，使用样本标准差。"""
    N, T = mat.shape
    z = np.full((N, T), np.nan, dtype=np.float64)
    for t in range(window, T):
        w = mat[:, t - window:t]            # (N, window)
        cnt = (~np.isnan(w)).sum(axis=1)    # (N,)
        ok  = cnt >= min_count
        mu  = np.nanmean(w, axis=1)         # (N,)
        std = np.nanstd(w, axis=1, ddof=1)  # 样本标准差
        cur = mat[:, t]
        valid = ok & (std > 1e-10) & ~np.isnan(cur)
        z[valid, t] = (cur[valid] - mu[valid]) / std[valid]
    return z


def _cross_zscore_numpy(mat: np.ndarray) -> np.ndarray:
    """截面 Z-Score，(N,T) → (N,T)，使用样本标准差。"""
    N, T = mat.shape
    z = np.full((N, T), np.nan, dtype=np.float64)
    for t in range(T):
        col = mat[:, t]
        valid = ~np.isnan(col) & (col > -90.0)
        if valid.sum() < 5:
            continue
        mu  = col[valid].mean()
        std = col[valid].std(ddof=1)
        if std < 1e-10:
            continue
        z[valid, t] = (col[valid] - mu) / std
    return z


def _upvol_ratio_numpy(
    close: np.ndarray,
    volume: np.ndarray,
    window: int = 10,
) -> np.ndarray:
    """上涨日成交量占比，(N,T) → (N,T)。"""
    N, T = close.shape
    ratio = np.full((N, T), np.nan, dtype=np.float64)
    for t in range(window + 1, T):
        c_cur  = close[:, t - window:t]     # (N, window)
        c_prev = close[:, t - window - 1:t - 1]
        v_win  = volume[:, t - window:t]
        total  = v_win.sum(axis=1)
        up_vol = np.where(c_cur > c_prev, v_win, 0.0).sum(axis=1)
        ok = total > 1e-8
        ratio[ok, t] = up_vol[ok] / total[ok]
    return ratio


def _volatility_numpy(close: np.ndarray, window: int = 20) -> np.ndarray:
    """20日年化波动率，(N,T) → (N,T)。"""
    N, T = close.shape
    vol = np.full((N, T), np.nan, dtype=np.float64)
    for t in range(window + 1, T):
        p0 = close[:, t - window:t - 1]
        p1 = close[:, t - window + 1:t]
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = np.where((p0 > 1e-8) & (p1 > 1e-8), np.log(p1 / (p0 + 1e-10)), np.nan)
        std = np.nanstd(ret, axis=1, ddof=1)
        vol[:, t] = std * np.sqrt(252.0)
    return vol


def _market_breadth_numpy(close: np.ndarray, window: int = 5,
                           valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """市场宽度（上涨股占比），(N,T) → (T,)。"""
    N, T = close.shape
    breadth = np.full(T, np.nan, dtype=np.float64)
    for t in range(window, T):
        base = close[:, t - window]
        cur  = close[:, t]
        ok   = (~np.isnan(base)) & (~np.isnan(cur)) & (base > 1e-8)
        if valid_mask is not None:
            ok = ok & valid_mask[:, t]
        if ok.sum() < 5:
            continue
        breadth[t] = (cur[ok] > base[ok]).mean()
    return breadth


# ─────────────────────────────────────────────────────────────────────────────
# 主 Alpha 函数
# ─────────────────────────────────────────────────────────────────────────────

@register_vec_strategy("ultra_alpha_v1")
def ultra_alpha_v1_alpha(
    close      : np.ndarray,                     # (N, T) float64
    open_      : np.ndarray,                     # (N, T) float64
    high       : np.ndarray,                     # (N, T) float64
    low        : np.ndarray,                     # (N, T) float64
    volume     : np.ndarray,                     # (N, T) float64
    params,
    valid_mask : Optional[np.ndarray] = None,    # (N, T) bool
    market_regime: Optional[np.ndarray] = None,  # (T,)  int8（V10 铁律，不做熊市清仓）
    **kw,
) -> AlphaSignal:
    """
    UltraAlpha V1 纯因子版（AlphaSignal 输出）。

    因子体系（与 V8 完全相同）：
      F1 RSRS Beta → 时序 Z-Score → 截面 Z-Score
      F2 中期动量（20日，跳过最近1天）→ 截面 Z-Score
      F3 短期反转（5日取负）→ 截面 Z-Score
      F4 量价聪明钱（上涨日成交量占比）→ 时序 Z-Score → 截面 Z-Score

    市场宽度用于动态权重（不再触发清仓）：
      brd >= 0.52 → BULL 权重  [0.38, 0.30, 0.12, 0.20]
      brd <  0.42 → 偏熊权重   [0.32, 0.18, 0.30, 0.20]
      其他        → NEUTRAL权重 [0.35, 0.25, 0.20, 0.20]

    返回 AlphaSignal，raw_target_weights 等权，col_sum ≤ 1.0。
    """
    N, T = close.shape

    rsrs_w   = int(getattr(params, "rsrs_window",      18)) if params is not None else 18
    zscore_w = int(getattr(params, "ultra_zscore_window",
                           getattr(params, "zscore_window", 200) if params else 200) if params else 200)  # [FIX-U-01] 250→200 加快信号响应
    top_n    = int(getattr(params, "top_n",            25)) if params is not None else 25
    min_price = float(getattr(params, "min_price",     5.0)) if params is not None else 5.0
    r2_thresh = float(getattr(params, "rsrs_r2_threshold", 0.50)) if params is not None else 0.50  # [FIX-U-01] 0.65→0.50 扩大候选池
    max_s_pos = float(getattr(params, "max_single_pos", 0.08)) if params is not None else 0.08

    # ── F1: RSRS ─────────────────────────────────────────────────────────────
    beta_mat = _rsrs_beta_numpy(high, low, window=rsrs_w)
    # R² 近似（用 beta 的符号代替，完整实现需两遍 OLS）
    r2_mat   = np.full((N, T), np.nan, dtype=np.float64)
    for t in range(rsrs_w, T):
        x  = low[:, t - rsrs_w:t]
        y  = high[:, t - rsrs_w:t]
        xm = x.mean(axis=1, keepdims=True)
        ym = y.mean(axis=1, keepdims=True)
        dx, dy = x - xm, y - ym
        sxy = (dx * dy).sum(axis=1)
        sxx = (dx * dx).sum(axis=1)
        syy = (dy * dy).sum(axis=1)
        ok  = (sxx > 1e-12) & (syy > 1e-12)
        r2_mat[ok, t] = np.clip(1.0 - (syy[ok] - sxy[ok]**2 / sxx[ok]) / syy[ok], 0.0, 1.0)

    rsrs_z_ts = _rolling_zscore_numpy(beta_mat, zscore_w, min_count=30)
    # [FIX-EMA-02] 对时序 Z-Score 做因子级 EMA 后再截面归一化（稳定排名）
    _ema = int(getattr(params, "factor_ema_span", 5)) if params is not None else 5
    rsrs_z_ts_s = _ema_smooth_factor(rsrs_z_ts, _ema) if _ema > 1 else rsrs_z_ts
    rsrs_z    = _cross_zscore_numpy(rsrs_z_ts_s)

    # ── F2: 中期动量（20日，跳最近1天）──────────────────────────────────────
    mom_w  = 20
    mom_raw = np.full((N, T), np.nan, dtype=np.float64)
    if T > mom_w + 1:
        denom = close[:, :T - mom_w - 1]
        numer = close[:, mom_w:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            mom_raw[:, mom_w + 1:] = numer / (denom + 1e-10) - 1.0
    mom_z = _cross_zscore_numpy(mom_raw)

    # ── F3: 短期反转（5日取负）───────────────────────────────────────────────
    rev_w   = 5
    rev_raw = np.full((N, T), np.nan, dtype=np.float64)
    if T > rev_w:
        with np.errstate(divide="ignore", invalid="ignore"):
            rev_raw[:, rev_w:] = -(close[:, rev_w:] / (close[:, :T - rev_w] + 1e-10) - 1.0)
    rev_z = _cross_zscore_numpy(rev_raw)

    # ── F4: 量价聪明钱 ────────────────────────────────────────────────────────
    upvol_raw = _upvol_ratio_numpy(close, volume, window=10)
    upvol_z_ts = _rolling_zscore_numpy(upvol_raw, window=60, min_count=15)
    upvol_z    = _cross_zscore_numpy(upvol_z_ts)

    # ── 市场宽度（动态权重，不清仓）──────────────────────────────────────────
    breadth = _market_breadth_numpy(close, window=5, valid_mask=valid_mask)
    vol_annual = _volatility_numpy(close, window=20)

    # ── 综合评分矩阵（无 Python per-t 循环，全量向量化）─────────────────────
    # 权重矩阵：根据 breadth 分档（(T,) → 广播）
    brd = np.where(np.isnan(breadth), 0.5, breadth)       # (T,)
    w1 = np.where(brd >= 0.52, 0.38, np.where(brd < 0.42, 0.32, 0.35))
    w2 = np.where(brd >= 0.52, 0.30, np.where(brd < 0.42, 0.18, 0.25))
    w3 = np.where(brd >= 0.52, 0.12, np.where(brd < 0.42, 0.30, 0.20))
    w4 = 0.20

    # 因子 NaN → 0（有效权重归一化）
    factors  = [rsrs_z, mom_z, rev_z, upvol_z]        # 每个 (N, T)
    weights_ = [w1[np.newaxis, :], w2[np.newaxis, :],
                w3[np.newaxis, :], np.full((1, T), w4)]  # 广播形状 (1,T)

    valid_f = np.stack([~np.isnan(f) for f in factors], axis=0)   # (4, N, T)
    w_arr   = np.stack(weights_, axis=0)                            # (4, 1, T)
    wsum    = (valid_f * w_arr).sum(axis=0)                         # (N, T)
    wsum    = np.where(wsum < 1e-6, 1.0, wsum)

    score_num = sum(
        w * np.where(np.isnan(f), 0.0, f)
        for w, f in zip(weights_, factors)
    )
    score_mat = score_num / wsum                                    # (N, T)

    # ── 门控 ─────────────────────────────────────────────────────────────────
    price_gate = close >= min_price
    rsrs_gate  = rsrs_z_ts > 0.0
    r2_gate    = r2_mat > r2_thresh
    vol_vec    = vol_annual
    vol_gate   = np.where(np.isnan(vol_vec), True, vol_vec <= 0.80)

    if valid_mask is not None:
        vm = np.asarray(valid_mask, dtype=bool)
    else:
        vm = np.ones((N, T), dtype=bool)

    buy_candidate = rsrs_gate & price_gate & r2_gate & vol_gate & vm
    score_mat_filtered = np.where(buy_candidate, score_mat, -np.inf)
    # invalid_mask → -inf
    score_mat_filtered[~vm] = -np.inf

    # [L4-FIX] 涨跌停硬封锁
    limit_mask = kw.get("limit_reach_mask")
    if limit_mask is not None:
        score_mat_filtered[limit_mask] = -np.inf

    # [L4-FIX] 数值稳态保护
    score_mat_filtered = np.nan_to_num(score_mat_filtered, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)

    # ── Top-N 等权权重（列级）────────────────────────────────────────────────
    # [FIX-DB-01] 策略级防抖参数：可通过 params 或 config.json.strategy_params 配置
    dropout_d   = 5   # ★[FIX-DEBOUNCE] 策略专属，来自 exit_config 规格
    exit_buf    = 7   # ★[FIX-DEBOUNCE] 策略专属，来自 exit_config 规格
    raw_weights = _score_to_weights(score_mat_filtered, top_n=top_n, max_single_pos=max_s_pos,
                                      dropout_days=dropout_d, exit_buffer=exit_buf,
                                      hard_invalid=None if valid_mask is None else ~vm)

    return AlphaSignal(
        raw_target_weights = raw_weights,
        score              = score_mat_filtered,
        strategy_name      = "ultra_alpha_v1",
        exit_config        = {
            "stop_mode"       : "trailing",
            "hard_stop_loss"  : 0.15,
            "take_profit"     : 0.25,
            "max_holding_days": 25,
            "dropout_days"    : 5,
            "exit_buffer"     : 7,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# 验收测试
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    rng = np.random.default_rng(0)
    N, T = 50, 300
    c = np.cumprod(1 + rng.normal(0, 0.01, (N, T)), axis=1) * 10
    h = c * (1 + rng.uniform(0, 0.005, (N, T)))
    l = c * (1 - rng.uniform(0, 0.005, (N, T)))
    v = rng.uniform(1e6, 1e7, (N, T))

    alpha = ultra_alpha_v1_alpha(c, c, h, l, v, None)
    assert alpha.raw_target_weights.shape == (N, T), "shape FAIL"
    assert not np.any(alpha.raw_target_weights < 0), "negative weight FAIL"
    assert alpha.raw_target_weights.sum(axis=0).max() <= 1.0 + 1e-6, "col_sum FAIL"
    print(f"[PASS] ultra_alpha_v1_alpha: shape={alpha.raw_target_weights.shape} "
          f"max_col_sum={alpha.raw_target_weights.sum(axis=0).max():.4f} ✓")
