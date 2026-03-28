"""
Q-UNITY V10 — momentum_reversal_alpha.py
==========================================
MomentumReversal 纯因子版（AlphaSignal 输出）

迁移自 momentum_reversal_vec.py，删除风控层，保留完整因子流水线：

✘ 删除：熊市保护（breadth + mkt_mom 双重门控）/ ever_bought / hold_days
✘ 删除：Bottom-N 卖出信号 / sell_signals 双轨
✓ 保留：3因子（动量 momentum_window + 反转 reversal_window + 质量 IR 60D）
✓ 保留：横截面百分位排名（_cross_section_rank）
✓ 保留：市场方向自适应权重（mkt_mom > market_thresh → 加大动量权重）
✓ 保留：3权重归一化（mw+rw+qw = 1）
✓ 有效性：valid_mask=False → score=NaN → -inf → weight=0

因子说明：
  动量因子  : momentum_window 日收益率（默认20D）
  反转因子  : reversal_window 日收益率取负（默认5D，超跌反转）
  质量因子  : 60日信息比率 IR = mean(daily_ret) / std(daily_ret) × √252
"""
from __future__ import annotations

import numpy as np
from typing import Optional

try:
    from src.strategies.alpha_signal import AlphaSignal, _score_to_weights, _ema_smooth_factor
except ImportError:
    from alpha_signal import AlphaSignal, _score_to_weights          # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# 因子计算内核（忠实移植自 Numba 版本，纯 NumPy）
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


def _momentum_batch(close: np.ndarray, window: int = 20) -> np.ndarray:
    """N日动量：close[t] / close[t-window] - 1，前 window 列为 NaN。"""
    N, T = close.shape
    mom = np.full((N, T), np.nan, dtype=np.float64)
    if T > window:
        with np.errstate(divide="ignore", invalid="ignore"):
            base = close[:, :T - window]
            ok   = base > 1e-8
            mom[:, window:] = np.where(ok, close[:, window:] / (base + 1e-10) - 1.0, np.nan)
    return mom


def _reversal_batch(close: np.ndarray, window: int = 5) -> np.ndarray:
    """N日反转（负动量）：-(close[t]/close[t-window]-1)，前 window 列为 NaN。"""
    N, T = close.shape
    rev = np.full((N, T), np.nan, dtype=np.float64)
    if T > window:
        with np.errstate(divide="ignore", invalid="ignore"):
            base = close[:, :T - window]
            ok   = base > 1e-8
            rev[:, window:] = np.where(ok, -(close[:, window:] / (base + 1e-10) - 1.0), np.nan)
    return rev


def _quality_batch(close: np.ndarray, window: int = 60) -> np.ndarray:
    """
    质量因子（信息比率代理）：mean(daily_ret) / std(daily_ret) × √252。
    前 window+1 列为 NaN；有效观测数 < 20 时为 NaN。
    样本标准差（ddof=1），与源码一致。
    """
    N, T = close.shape
    quality = np.full((N, T), np.nan, dtype=np.float64)
    for t in range(window + 1, T):
        p0 = close[:, t - window:t - 1]     # (N, window-1)
        p1 = close[:, t - window + 1:t]     # (N, window-1)
        ok  = p0 > 1e-8
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(ok, p1 / (p0 + 1e-10) - 1.0, np.nan)
        cnt = ok.sum(axis=1)
        ok2 = cnt >= 20
        mu  = np.nanmean(r, axis=1)
        std = np.nanstd(r, axis=1, ddof=1)
        valid = ok2 & (std > 1e-10)
        quality[valid, t] = mu[valid] / std[valid] * np.sqrt(252.0)
    return quality


def _cross_section_rank(arr: np.ndarray) -> np.ndarray:
    """
    横截面百分位排名（0→1，升序），NaN 处保持 NaN。
    有效股票数 < 10 时全部返回 NaN（与源码 Bug49 Fix 一致）。
    分母为 n_valid（与源码 M-03-FIX 一致）。
    """
    N = len(arr)
    ranks = np.full(N, np.nan, dtype=np.float64)
    ok    = ~np.isnan(arr)
    valid_idx = np.where(ok)[0]
    if len(valid_idx) < 10:
        return ranks
    order = np.argsort(arr[valid_idx])
    n_valid = len(order)
    for rank_i, orig_i in enumerate(order):
        ranks[valid_idx[orig_i]] = rank_i / float(n_valid)   # M-03-FIX: n_valid
    return ranks


# ─────────────────────────────────────────────────────────────────────────────
# 主 Alpha 函数
# ─────────────────────────────────────────────────────────────────────────────

@register_vec_strategy("momentum_reversal")
def momentum_reversal_alpha(
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
    MomentumReversal 纯因子版。

    综合评分（市场方向自适应，来自源码逻辑）：
      牛市（mkt_mom > market_thresh）：eff_mw = mw+0.1, eff_rw = rw-0.1
      熊市/震荡：                       eff_mw = mw-0.1, eff_rw = rw+0.1
      NaN 时：使用原始 mw/rw
      三权重强制归一化：eff_mw + eff_rw + eff_qw = 1.0

    得分 = eff_mw×Rank(mom) + eff_rw×Rank(rev) + eff_qw×Rank(quality)
    NaN 因子填 0（源码 S-01-FIX：不填 0.5，避免无数据新股误入 Top-N）
    """
    N, T = close.shape

    mom_w    = int(getattr(params, "momentum_window",  20)) if params else 20
    rev_w    = int(getattr(params, "reversal_window",   5)) if params else 5
    top_n    = int(getattr(params, "top_n",            20)) if params else 20
    mw       = float(getattr(params, "momentum_weight", 0.5)) if params else 0.5
    rw       = float(getattr(params, "reversal_weight", 0.3)) if params else 0.3
    qw       = float(getattr(params, "quality_weight",  0.2)) if params else 0.2
    mkt_thr  = float(getattr(params, "market_thresh",   0.0)) if params else 0.0
    max_s_pos = float(getattr(params, "max_single_pos", 0.08)) if params else 0.08

    # ── 三因子计算 ───────────────────────────────────────────────────────────
    mom     = _momentum_batch(close, mom_w)
    rev     = _reversal_batch(close, rev_w)
    quality = _quality_batch(close, 60)

    # ── 市场整体动量（方向判断，[FIX-S-07]过滤invalid股票）──────────────
    with np.errstate(all="ignore"):
        mom_valid = np.where(valid_mask, mom, np.nan) if valid_mask is not None else mom
        mkt_mom = np.nanmean(mom_valid, axis=0)

    # ── 逐日合成评分 ─────────────────────────────────────────────────────────
    score_mat = np.full((N, T), np.nan, dtype=np.float64)

    for t in range(T):
        # 市场方向自适应权重（与源码 BUG-N Fix 完全对齐）
        mm_t = mkt_mom[t]
        if np.isnan(mm_t):
            eff_mw, eff_rw = mw, rw
        elif mm_t > mkt_thr:
            eff_mw = min(mw + 0.1, 0.9)
            eff_rw = max(rw - 0.1, 0.0)
        else:
            eff_mw = max(mw - 0.1, 0.0)
            eff_rw = min(rw + 0.1, 0.9)

        eff_qw = max(1.0 - eff_mw - eff_rw, 0.0)
        total_w = eff_mw + eff_rw + eff_qw
        if total_w > 1e-6:
            eff_mw /= total_w
            eff_rw /= total_w
            eff_qw /= total_w

        # 横截面排名
        rank_m = _cross_section_rank(mom[:, t])
        rank_r = _cross_section_rank(rev[:, t])
        rank_q = _cross_section_rank(quality[:, t])

        # 综合得分（NaN→0，不填0.5）
        score = (
            eff_mw * np.nan_to_num(rank_m, nan=0.0)
            + eff_rw * np.nan_to_num(rank_r, nan=0.0)
            + eff_qw * np.nan_to_num(rank_q, nan=0.0)
        )

        # valid_mask=False → NaN（与源码一致）
        if valid_mask is not None:
            score[~valid_mask[:, t]] = np.nan

        score_mat[:, t] = score

    # NaN → -inf（_score_to_weights 跳过 -inf 位置）
    # [FIX-MR-01] 若某列所有三因子排名均为 NaN（预热期不足），score 全为 0.0
    # _score_to_weights 会将这些 0 值股票纳入 Top-N，产生无意义的幽灵持仓。
    # 修复：检测 "全零列"（三因子均为 NaN 的列），将其强制置 -inf（空仓）。
    all_nan_col = (
        np.all(np.isnan(mom),     axis=0) &
        np.all(np.isnan(rev),     axis=0) &
        np.all(np.isnan(quality), axis=0)
    )
    # [FIX-EMA-02] 对合成评分做因子级 EMA 平滑（在截面选股前稳定排名）
    _ema = int(getattr(params, "factor_ema_span", 5)) if params else 5
    score_mat_s = _ema_smooth_factor(score_mat, _ema) if _ema > 1 else score_mat
    score_inf = np.where(np.isnan(score_mat_s), -np.inf, score_mat_s)
    score_inf[:, all_nan_col] = -np.inf   # 预热期幽灵持仓完全清零

    # [FIX-DB-01] 策略级防抖参数：可通过 params 或 config.json.strategy_params 配置
    dropout_d   = 3   # ★[FIX-DEBOUNCE] 策略专属，来自 exit_config 规格
    exit_buf    = 5   # ★[FIX-DEBOUNCE] 策略专属，来自 exit_config 规格
    raw_weights = _score_to_weights(score_inf, top_n=top_n, max_single_pos=max_s_pos,
                                      dropout_days=dropout_d, exit_buffer=exit_buf,
                                      hard_invalid=None if valid_mask is None else ~np.asarray(valid_mask, dtype=bool))

    return AlphaSignal(
        raw_target_weights=raw_weights,
        score=score_inf,
        strategy_name="momentum_reversal",
        exit_config={
            "stop_mode"       : "trailing",
            "hard_stop_loss"  : 0.12,
            "take_profit"     : 0.18,
            "max_holding_days": 20,
            "dropout_days"    : 3,
            "exit_buffer"     : 5,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# 验收测试
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    rng = np.random.default_rng(3)
    N, T = 50, 300
    c = np.cumprod(1 + rng.normal(0.0003, 0.015, (N, T)), axis=1).astype(np.float64) * 10

    alpha = momentum_reversal_alpha(c, c, c, c, np.ones((N, T)), None)

    assert alpha.raw_target_weights.shape == (N, T), \
        f"shape FAIL: {alpha.raw_target_weights.shape}"
    assert not np.any(alpha.raw_target_weights < 0), "negative weight FAIL"
    assert alpha.raw_target_weights.sum(axis=0).max() <= 1.0 + 1e-6, \
        f"col_sum FAIL: {alpha.raw_target_weights.sum(axis=0).max():.6f}"

    nz = (alpha.raw_target_weights.sum(axis=0) > 0).sum()
    print(f"[PASS] momentum_reversal_alpha: shape={alpha.raw_target_weights.shape} "
          f"max_col_sum={alpha.raw_target_weights.sum(axis=0).max():.4f} "
          f"nonzero_cols={nz}/{T} ✓")
    sys.exit(0)
