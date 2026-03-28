"""
Q-UNITY V10 — sentiment_reversal_alpha.py
===========================================
SentimentReversal 纯因子版（AlphaSignal 输出）

迁移自 sentiment_reversal_vec.py，删除风控层，保留完整因子流水线：

✘ 删除：hold_days 最小持仓保护 / ever_bought 状态追踪
✘ 删除：execution_mask 幽灵卖出修正 / sell_signals 双轨
✘ 删除：买入优先逻辑（清除卖出信号）
✓ 保留：panic_score（跌幅 Z-Score + 量 Z-Score + 振幅 Z-Score 三维度）
✓ 保留：fomo_score（连涨天数 + 量 Z-Score clip）
✓ 保留：_compute_percentile_rank（252日历史百分位排名）
✓ 保留：panic_pct > panic_threshold 作为入场 score
✓ valid_mask=False → score=-inf → weight=0

情绪因子说明（源码原文）：
  恐慌分 = 0.4×(-ret_z) + 0.4×vol_z + 0.2×amp_z
  FOMO 分 = 0.5×(连涨天数/window) + 0.5×clip(vol_z/3, 0, 1)
  买入信号：panic_pct > panic_threshold（恐慌百分位极值 → 反转潜力）
"""
from __future__ import annotations

import numpy as np
from typing import Optional

try:
    from src.strategies.alpha_signal import AlphaSignal, _score_to_weights
except ImportError:
    from alpha_signal import AlphaSignal, _score_to_weights          # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# 因子计算内核（忠实移植自 Numba 版，纯 NumPy）
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


def _zscore_of_series(arr: np.ndarray, x: float) -> float:
    """对数组 arr 计算 x 的 Z-Score（样本标准差，与源码 S-03-FIX 一致）。"""
    n = len(arr)
    if n < 2:
        return 0.0
    mu  = arr.mean()
    std = arr.std(ddof=1)
    if std > 1e-10:
        return (x - mu) / std
    return 0.0


def _compute_panic_score(
    close : np.ndarray,
    volume: np.ndarray,
    high  : np.ndarray,
    low   : np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """
    恐慌评分：0.4×(-ret_z) + 0.4×vol_z + 0.2×amp_z
    [FIX-S-04] N维向量化（消除O(N×T)双重Python循环，约15min→5s）。
    """
    N, T = close.shape
    panic = np.full((N, T), np.nan, dtype=np.float64)
    if T <= window + 1:
        return panic
    p0 = close[:, :-1]; p1 = close[:, 1:]
    ok_r = p0 > 1e-8
    ret_all = np.where(ok_r, p1 / (p0 + 1e-10) - 1.0, 0.0)
    amp_all = np.where(ok_r, (high[:, 1:] - low[:, 1:]) / (p0 + 1e-10), 0.0)
    for t in range(window + 1, T):
        js = t - window - 1
        if js < 0:
            continue
        hist_r = ret_all[:, js:t - 1]
        hist_v = volume[:, t - window:t]
        hist_a = amp_all[:, js:t - 1]
        mu_r = hist_r.mean(1); std_r = hist_r.std(1, ddof=1)
        mu_v = hist_v.mean(1); std_v = hist_v.std(1, ddof=1)
        mu_a = hist_a.mean(1); std_a = hist_a.std(1, ddof=1)
        z_ret = np.where(std_r > 1e-10, (ret_all[:, t-1] - mu_r) / (std_r + 1e-10), 0.0)
        z_vol = np.where(std_v > 1e-10, (volume[:, t]    - mu_v) / (std_v + 1e-10), 0.0)
        z_amp = np.where(std_a > 1e-10, (amp_all[:, t-1] - mu_a) / (std_a + 1e-10), 0.0)
        panic[:, t] = 0.4 * (-z_ret) + 0.4 * z_vol + 0.2 * z_amp
    return panic

def _compute_fomo_score(
    close : np.ndarray,   # (N, T)
    volume: np.ndarray,
    window: int = 10,
) -> np.ndarray:
    """
    FOMO 评分：0.5×(up_days/window) + 0.5×clip(vol_z/3, 0, 1)
    前 window+1 列为 NaN（与源码一致）。
    """
    N, T = close.shape
    fomo = np.full((N, T), np.nan, dtype=np.float64)

    for t in range(window + 1, T):
        # 连涨天数
        up_days = np.zeros(N, dtype=np.float64)
        for k in range(1, window + 1):
            up_days += (close[:, t - k] >= close[:, t - k - 1]).astype(float)

        # 成交量 Z-Score（过去 min(20, t) 天，样本标准差）
        vol_hist_len = min(20, t)
        if vol_hist_len < 5:
            continue
        v_win = volume[:, t - vol_hist_len:t]    # (N, vol_hist_len)
        mu_v  = v_win.mean(axis=1)
        std_v = v_win.std(axis=1, ddof=1)
        ok_v  = std_v > 1e-10
        vol_z = np.where(ok_v, (volume[:, t] - mu_v) / (std_v + 1e-10), 0.0)
        vol_z_clip = np.clip(vol_z / 3.0, 0.0, 1.0)

        fomo[:, t] = 0.5 * (up_days / window) + 0.5 * vol_z_clip

    return fomo


def _compute_percentile_rank(
    scores    : np.ndarray,    # (N, T)
    hist_window: int = 252,
) -> np.ndarray:
    """
    基于历史分布的百分位排名（源码 BUG-8 Fix 向量化版 NumPy 实现）。
    值域 [0, 1]，前 hist_window 列为 NaN；有效历史 < 30 时为 NaN。
    平滑百分位：(rank + 0.5) / n（与源码一致）。
    """
    N, T = scores.shape
    pct = np.full((N, T), np.nan, dtype=np.float64)

    for t in range(hist_window, T):
        cur = scores[:, t]
        hist = scores[:, t - hist_window:t]   # (N, hist_window)
        for i in range(N):
            cv = cur[i]
            if np.isnan(cv):
                continue
            hw = hist[i]
            valid_h = hw[~np.isnan(hw)]
            if len(valid_h) < 30:
                continue
            # searchsorted 等效于源码插入排序后二分查找
            sorted_h = np.sort(valid_h)
            lo = int(np.searchsorted(sorted_h, cv, side="left"))
            pct[i, t] = (float(lo) + 0.5) / float(len(valid_h))

    return pct


# ─────────────────────────────────────────────────────────────────────────────
# 主 Alpha 函数
# ─────────────────────────────────────────────────────────────────────────────

@register_vec_strategy("sentiment_reversal")
def sentiment_reversal_alpha(
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
    SentimentReversal 纯因子版。

    买入 score = panic_pct（恐慌百分位）。
    panic_pct > panic_threshold 的股票按百分位降序选 Top-N。
    fomo_score 不用于 weights（不做减仓信号）。

    valid_mask=False → score=-inf → weight=0
    """
    N, T = close.shape

    panic_thr  = float(getattr(params, "panic_threshold",   0.65)) if params else 0.65  # [FIX-SR-01] 0.80→0.65 配合EMA平滑
    ov_w       = int(getattr(params,   "sentiment_window",   20))   if params else 20
    top_n      = int(getattr(params,   "top_n",              20))   if params else 20
    max_s_pos  = float(getattr(params, "max_single_pos",     0.10)) if params else 0.10
    ema_span   = int(getattr(params,   "panic_ema_span",     10))   if params else 10    # [FIX-SR-01] 新增: EMA平滑窗口
    dropout_d  = 2   # ★[FIX-DEBOUNCE] 策略专属，来自 exit_config 规格     # [FIX-SR-01] 新增: 可配置防抖
    exit_buf   = 3   # ★[FIX-DEBOUNCE] 策略专属，来自 exit_config 规格     # [FIX-SR-01] 新增: 可配置缓冲

    # ── 因子计算 ─────────────────────────────────────────────────────────────
    panic      = _compute_panic_score(close, volume, high, low, ov_w)
    panic_pct  = _compute_percentile_rank(panic, hist_window=252)

    # [FIX-SR-01] EMA平滑 panic_pct：减少日间信号噪声，使防抖真正有效
    # 原始 panic_pct 脉冲性极强（80%持仓仅1天），EMA平滑后信号连续性提升
    # 年换手率从 7641% 降至约 3500-5000%（配合下方 dropout_days=5）
    if ema_span > 1:
        alpha_ema = 2.0 / (ema_span + 1)
        smoothed_pct = np.full_like(panic_pct, np.nan)
        for t in range(1, T):
            prev = smoothed_pct[:, t - 1]
            cur  = panic_pct[:, t]
            ok   = ~np.isnan(cur) & ~np.isnan(prev)
            smoothed_pct[:, t] = np.where(
                ok, alpha_ema * cur + (1 - alpha_ema) * prev,
                np.where(np.isnan(prev), cur, prev),
            )
        score_base = smoothed_pct
    else:
        score_base = panic_pct

    # ── 评分：平滑后 panic_pct，低于阈值的设为 -inf ──────────────────────────
    score = score_base.copy()

    # panic_pct_smoothed < threshold 或 NaN → 不入场
    score[np.isnan(score_base)]        = -np.inf
    score[np.nan_to_num(score_base, nan=0.0) <= panic_thr] = -np.inf

    # valid_mask=False → -inf
    if valid_mask is not None:
        score[~np.asarray(valid_mask, dtype=bool)] = -np.inf

    # [L4-FIX] 涨跌停硬封锁：T日已封死涨停则T+1不可买，已封死跌停则T+1不可卖
    limit_mask = kw.get("limit_reach_mask")
    if limit_mask is not None:
        score[limit_mask] = -np.inf

    # [L4-FIX] 数值稳态保护
    score = np.nan_to_num(score, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)

    # ── Top-N 等权（使用策略级 dropout_days/exit_buffer）─────────────────────
    raw_weights = _score_to_weights(
        score, top_n=top_n, max_single_pos=max_s_pos,
        dropout_days=dropout_d, exit_buffer=exit_buf,
        hard_invalid=None if valid_mask is None else ~np.asarray(valid_mask, dtype=bool),
    )

    return AlphaSignal(
        raw_target_weights=raw_weights,
        score=score,
        strategy_name="sentiment_reversal",
        exit_config={
            "stop_mode"       : "entry_price",
            "hard_stop_loss"  : 0.08,
            "take_profit"     : 0.12,
            "max_holding_days": 7,
            "dropout_days"    : 2,
            "exit_buffer"     : 3,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# 验收测试
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    rng = np.random.default_rng(4)
    N, T = 50, 300
    c = np.cumprod(1 + rng.normal(0.0003, 0.015, (N, T)), axis=1).astype(np.float64) * 10
    h = c * (1 + rng.uniform(0.001, 0.025, (N, T)))
    l = c * (1 - rng.uniform(0.001, 0.025, (N, T)))
    v = rng.uniform(1e6, 5e7, (N, T))

    alpha = sentiment_reversal_alpha(c, c, h, l, v, None)

    assert alpha.raw_target_weights.shape == (N, T), \
        f"shape FAIL: {alpha.raw_target_weights.shape}"
    assert not np.any(alpha.raw_target_weights < 0), "negative weight FAIL"
    assert alpha.raw_target_weights.sum(axis=0).max() <= 1.0 + 1e-6, \
        f"col_sum FAIL: {alpha.raw_target_weights.sum(axis=0).max():.6f}"

    nz = (alpha.raw_target_weights.sum(axis=0) > 0).sum()
    print(f"[PASS] sentiment_reversal_alpha: shape={alpha.raw_target_weights.shape} "
          f"max_col_sum={alpha.raw_target_weights.sum(axis=0).max():.4f} "
          f"nonzero_cols={nz}/{T} ✓")
    sys.exit(0)
