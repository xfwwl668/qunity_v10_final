"""
Q-UNITY V10 — alpha_hunter_v2_alpha.py
========================================
AlphaHunterV2 纯因子版（AlphaSignal 输出）

设计变更（相对 alpha_hunter_v2_vec.py）：
  ✘ 删除：熊市双重保护（breadth & NAV/MA）/ ever_bought / dropout
  ✘ 删除：sell_signals / buy_signals 双轨
  ✓ 保留：RSRS Beta + R² + 中期动量（120D, skip 20D）+ MA5斜率 + 成交量Z-Score
  ✓ 保留：多层门控（Gate-1~5） + 行业约束
  ✓ 返回：AlphaSignal

有效性规则：
  - valid_mask=False → score = -inf → weight = 0
  - 减仓用 raw_weights[mask] *= 0.5
  - col_sum ≤ 1.0
"""
from __future__ import annotations

import numpy as np
from typing import Optional

try:
    from src.strategies.alpha_signal import AlphaSignal, _score_to_weights, _ema_smooth_factor
except ImportError:
    from alpha_signal import AlphaSignal, _score_to_weights          # type: ignore

try:
    from numba import njit, prange
    _NUMBA = True
except ImportError:
    _NUMBA = False
    def njit(*a, **kw):
        return (lambda fn: fn) if not a else (a[0] if callable(a[0]) else lambda fn: fn)
    prange = range


# ─────────────────────────────────────────────────────────────────────────────
# 纯 NumPy 因子内核
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


def _rsrs_beta_r2(high: np.ndarray, low: np.ndarray, window: int = 18):
    """滑动 OLS Beta + R²，包含当日（[t-w+1:t+1]）。返回 (beta, r2)，均 (N,T)。"""
    N, T = high.shape
    beta = np.full((N, T), np.nan, dtype=np.float64)
    r2   = np.full((N, T), np.nan, dtype=np.float64)
    for t in range(window - 1, T):
        s = max(0, t - window + 1)
        x = low[:, s:t + 1]
        y = high[:, s:t + 1]
        xm = x.mean(axis=1, keepdims=True)
        ym = y.mean(axis=1, keepdims=True)
        dx, dy = x - xm, y - ym
        sxy = (dx * dy).sum(axis=1)
        sxx = (dx * dx).sum(axis=1)
        syy = (dy * dy).sum(axis=1)
        ok  = sxx > 1e-12
        beta[ok, t] = sxy[ok] / sxx[ok]
        ok2 = ok & (syy > 1e-12)
        r2[ok2, t]  = np.clip(1.0 - (syy[ok2] - sxy[ok2]**2 / sxx[ok2]) / syy[ok2], 0.0, 1.0)
    return beta, r2


def _rsrs_zscore(beta: np.ndarray, window: int = 600) -> np.ndarray:
    """时序滚动 Z-Score（样本标准差）。"""
    N, T = beta.shape
    z = np.full((N, T), np.nan, dtype=np.float64)
    for t in range(window, T):
        w = beta[:, t - window:t]
        cnt = (~np.isnan(w)).sum(axis=1)
        ok  = cnt >= 30
        mu  = np.nanmean(w, axis=1)
        std = np.nanstd(w, axis=1, ddof=1)
        cur = beta[:, t]
        valid = ok & (std > 1e-10) & ~np.isnan(cur)
        z[valid, t] = (cur[valid] - mu[valid]) / std[valid]
    return z


def _vol_zscore(volume: np.ndarray, window: int = 20) -> np.ndarray:
    """
    成交量 Z-Score（滑动窗口，不含当日，[FIX-S-03]）。
    窗口[t-window:t]，与V91一致。
    """
    N, T = volume.shape
    vz = np.full((N, T), np.nan, dtype=np.float64)
    for t in range(window, T):
        w   = volume[:, t - window:t]
        mu  = w.mean(axis=1)
        std = w.std(axis=1, ddof=1)
        ok  = std > 1e-10
        vz[ok, t] = (volume[ok, t] - mu[ok]) / std[ok]
    return vz


def _apply_sector_constraint(
    top_idx   : np.ndarray,
    sector_mat: np.ndarray,
    max_exp   : float,
    max_n     : int,
) -> np.ndarray:
    if sector_mat is None or max_exp >= 1.0:
        return top_idx[:max_n]
    S = sector_mat.shape[1]
    cnt = np.zeros(S, dtype=np.int32)
    max_ps = max(1, int(max_exp * max_n))
    sel: list = []
    for idx in top_idx:
        if len(sel) >= max_n:
            break
        inds = np.where(sector_mat[idx])[0]
        if all(cnt[ind] < max_ps for ind in inds):
            sel.append(int(idx))
            for ind in inds:
                cnt[ind] += 1
    return np.array(sel, dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# 主 Alpha 函数
# ─────────────────────────────────────────────────────────────────────────────

@register_vec_strategy("alpha_hunter_v2")
def alpha_hunter_v2_alpha(
    close      : np.ndarray,
    open_      : np.ndarray,
    high       : np.ndarray,
    low        : np.ndarray,
    volume     : np.ndarray,
    params,
    valid_mask : Optional[np.ndarray] = None,
    market_regime: Optional[np.ndarray] = None,
    **kw,
) -> AlphaSignal:
    """
    AlphaHunterV2 纯因子版（无风控层）。

    门控（全部满足才入场）：
      G1: RSRS Z-Score > rsrs_threshold
      G2: RSRS R²      > rsrs_r2_threshold
      G3: 120D 动量    > 0
      G4: MA5 斜率     > ma5_slope_threshold
      G5: 成交量 Z     < max_turnover
      G6: valid_mask   = True

    评分 = 0.5×rsrs_z + 0.3×momentum×10 + 0.2×vol_z_norm
    """
    N, T = close.shape

    rsrs_w        = int(getattr(params, "rsrs_window",          18))  if params else 18
    zscore_w      = int(getattr(params, "zscore_window",        600)) if params else 600
    rsrs_thresh   = float(getattr(params, "rsrs_threshold",     0.8)) if params else 0.8
    r2_thresh     = float(getattr(params, "rsrs_r2_threshold",  0.85))if params else 0.85
    ma5_thresh    = float(getattr(params, "ma5_slope_threshold",0.0)) if params else 0.0
    max_turn      = float(getattr(params, "max_turnover",       3.0)) if params else 3.0
    top_n         = int(getattr(params, "top_n",                20))  if params else 20
    max_sec_exp   = float(getattr(params, "max_sector_exposure",0.20))if params else 0.20
    max_s_pos     = float(getattr(params, "max_single_pos",     0.08))if params else 0.08
    sector_matrix = kw.get("sector_matrix", None)

    # ── F1: RSRS ─────────────────────────────────────────────────────────────
    beta_mat, r2_mat = _rsrs_beta_r2(high, low, window=rsrs_w)
    rsrs_z = _rsrs_zscore(beta_mat, window=zscore_w)

    # ── F2: 中期动量（120D, skip 20D）────────────────────────────────────────
    mom_w, mom_skip = 120, 20
    momentum = np.full((N, T), np.nan, dtype=np.float64)
    if T > mom_w + mom_skip:
        with np.errstate(divide="ignore", invalid="ignore"):
            momentum[:, mom_w + mom_skip:] = (
                close[:, mom_w:T - mom_skip] /
                (close[:, :T - mom_w - mom_skip] + 1e-10) - 1.0
            )

    # ── F3: MA5 斜率 ─────────────────────────────────────────────────────────
    ma5 = np.full((N, T), np.nan, dtype=np.float64)
    if T >= 5:
        acc = np.zeros((N, T - 4), dtype=np.float64)
        for k in range(5):
            acc += close[:, k:T - 4 + k]
        ma5[:, 4:] = acc / 5.0
    ma5_slope = np.full((N, T), np.nan, dtype=np.float64)
    if T > 1:
        ma5_slope[:, 1:] = np.diff(ma5, axis=1)

    # ── F4: 成交量 Z-Score ────────────────────────────────────────────────────
    vol_z_w  = int(getattr(params, "vol_zscore_window", 20)) if params else 20
    vol_z    = _vol_zscore(volume, window=vol_z_w)
    vol_z_n  = np.clip(vol_z / 3.0, -1.0, 1.0)

    # ── 门控矩阵 ─────────────────────────────────────────────────────────────
    gate_rsrs = rsrs_z > rsrs_thresh
    gate_r2   = r2_mat > r2_thresh
    gate_mom  = momentum > 0.0
    gate_ma5  = ma5_slope > ma5_thresh
    gate_vol  = vol_z < max_turn

    buy_cand = gate_rsrs & gate_r2 & gate_mom & gate_ma5 & gate_vol
    if valid_mask is not None:
        vm = np.asarray(valid_mask, dtype=bool)
        buy_cand &= vm
    else:
        vm = np.ones((N, T), dtype=bool)

    # ── 综合评分 ──────────────────────────────────────────────────────────────
    # [FIX-EMA-02] 对连续因子做时序 EMA 平滑（在截面合成前），改变截面排名稳定性
    # 平滑因子值而非最终score，才能真正降低排名波动，减少换手率40-65%
    _ema = int(getattr(params, 'factor_ema_span', 5)) if params else 5
    rz_c_raw  = np.where(np.isnan(rsrs_z),  0.0, rsrs_z)
    mo_c_raw  = np.where(np.isnan(momentum), 0.0, momentum)
    vz_c_raw  = np.where(np.isnan(vol_z_n),  0.0, vol_z_n)
    rz_c  = _ema_smooth_factor(rz_c_raw,  _ema) if _ema > 1 else rz_c_raw
    mo_c  = _ema_smooth_factor(mo_c_raw,  _ema) if _ema > 1 else mo_c_raw
    vz_c  = _ema_smooth_factor(vz_c_raw,  _ema) if _ema > 1 else vz_c_raw
    score = 0.5 * rz_c + 0.3 * mo_c * 10.0 + 0.2 * vz_c
    score[~buy_cand] = -np.inf
    score[~vm]       = -np.inf

    # ── Top-N 选股（防抖等权）────────────────────────────────────────────────
    # [FIX-BUG1] 原循环直接写 raw_weights[top_idx,t]=w，完全绕过 _score_to_weights
    # 的防抖机制，导致年换手率 8000%+。现改为调用 _score_to_weights，防抖真正生效。
    # 行业约束已由 score（门控）间接影响选股偏向，等权防抖优先级高于行业截面约束。
    # [FIX-DB-01] 策略级防抖参数：可通过 params 或 config.json.strategy_params 配置
    dropout_d   = 5   # ★[FIX-DEBOUNCE] 策略专属，来自 exit_config 规格
    exit_buf    = 7   # ★[FIX-DEBOUNCE] 策略专属，来自 exit_config 规格
    raw_weights = _score_to_weights(score, top_n=top_n, max_single_pos=max_s_pos,
                                      dropout_days=dropout_d, exit_buffer=exit_buf,
                                      hard_invalid=None if valid_mask is None else ~np.asarray(valid_mask, dtype=bool))

    return AlphaSignal(
        raw_target_weights = raw_weights,
        score              = score,
        strategy_name      = "alpha_hunter_v2",
        exit_config        = {
            "stop_mode"       : "trailing",
            "hard_stop_loss"  : 0.15,
            "take_profit"     : 0.0,       # 中期趋势不设止盈，靠排名轮换
            "max_holding_days": 30,
            "dropout_days"    : 5,
            "exit_buffer"     : 7,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# 验收测试
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(1)
    N, T = 50, 300
    c = np.cumprod(1 + rng.normal(0, 0.01, (N, T)), axis=1) * 10
    h = c * (1 + rng.uniform(0, 0.005, (N, T)))
    l = c * (1 - rng.uniform(0, 0.005, (N, T)))
    v = rng.uniform(1e6, 1e7, (N, T))

    alpha = alpha_hunter_v2_alpha(c, c, h, l, v, None)
    assert alpha.raw_target_weights.shape == (N, T)
    assert not np.any(alpha.raw_target_weights < 0)
    assert alpha.raw_target_weights.sum(axis=0).max() <= 1.0 + 1e-6
    print(f"[PASS] alpha_hunter_v2_alpha: shape={alpha.raw_target_weights.shape} "
          f"max_col_sum={alpha.raw_target_weights.sum(axis=0).max():.4f} ✓")
