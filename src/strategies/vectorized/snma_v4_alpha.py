"""
Q-UNITY V10 — snma_v4_alpha.py
================================
SNMA-V4 + THS 板块共振纯因子版（AlphaSignal 输出）

迁移自 snma_v4_ths_vec.py，删除风控层，保留完整因子流水线：

✘ 删除：absent_days / max_holding 强制卖出 / ever_in_top_n
✘ 删除：Whale_Pulse 排名 < 20% 的弱势卖出信号
✘ 删除：sell_signals 双轨
✓ 保留：Whale_Pulse_raw 因子（body_ratio × money_density）
✓ 保留：Winsorize → 截面 Rank（_winsorize_cs + _cross_section_rank_batch）
✓ 保留：Momentum_20d 截面 Rank（skip 2D）
✓ 保留：DQC_Gap = Rank(WP) - weight_mom × Rank(MOM)
✓ 保留：DQC_Gap 截面 Z-Score → Alpha 原始分
✓ 保留：板块共振加权（concept_ids 非 None 时 × boost_factor）
✓ 保留：价格门控（min_price）/ valid_mask 门控
✓ valid_mask=False → score=-inf → weight=0

因子说明（源码原文）：
  body_ratio    = (close - open) / (high - low + ε)   K线方向性强度
  money_density = log1p(amount / (volume × price + ε)) 资金密度
  Whale_Pulse   = body_ratio × money_density
  DQC_Gap       = Rank(WP) - 0.5 × Rank(MOM_20d)      动量质量缺口
"""
from __future__ import annotations

import numpy as np
from typing import Optional

try:
    from src.strategies.alpha_signal import AlphaSignal, _score_to_weights, _ema_smooth_factor
except ImportError:
    from alpha_signal import AlphaSignal, _score_to_weights          # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# 因子内核（忠实移植自 Numba 版，纯 NumPy）
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


def _compute_whale_pulse(
    close : np.ndarray,
    open_ : np.ndarray,
    high  : np.ndarray,
    low   : np.ndarray,
    volume: np.ndarray,
    amount: np.ndarray,
    eps   : float = 1e-6,
) -> np.ndarray:
    """
    Whale_Pulse_raw = body_ratio × money_density（与源码 _compute_whale_pulse_batch 一致）。
    NaN 输入位置输出 NaN；volume<=0 或 close<=0 位置输出 NaN。
    """
    N, T   = close.shape
    out    = np.full((N, T), np.nan, dtype=np.float64)
    ok     = (~np.isnan(close)) & (~np.isnan(open_)) & (~np.isnan(high)) & \
             (~np.isnan(low))   & (~np.isnan(volume)) & (~np.isnan(amount)) & \
             (volume > 0) & (close > 0)

    hl_range     = high - low + eps
    body_ratio   = np.where(ok, (close - open_) / hl_range, np.nan)
    price_approx = (high + low) / 2.0 + eps
    money_density = np.where(ok, np.log1p(amount / (volume * price_approx + eps)), np.nan)

    out = np.where(ok, body_ratio * money_density, np.nan)
    return out


def _cross_section_rank_batch(mat: np.ndarray) -> np.ndarray:
    """
    批量截面百分位排名（升序，含 Ties 均排，与源码 _cross_section_rank_batch 一致）。
    NaN 保持 NaN；有效值 < 2 的列全为 NaN。
    """
    N, T   = mat.shape
    rank_m = np.full((N, T), np.nan, dtype=np.float64)
    for t in range(T):
        col = mat[:, t]
        ok  = ~np.isnan(col)
        idx = np.where(ok)[0]
        n   = len(idx)
        if n < 2:
            continue
        order = np.argsort(col[idx])          # 升序
        # 平均排名 → 分位数（含 ties）
        pos = 0
        sorted_vals = col[idx[order]]
        while pos < n:
            cur_v = sorted_vals[pos]
            end   = pos + 1
            while end < n and sorted_vals[end] == cur_v:
                end += 1
            pct = ((pos + 1) + end) / 2.0 / float(n)   # avg_rank / n
            for j in range(pos, end):
                rank_m[idx[order[j]], t] = pct
            pos = end
    return rank_m


def _compute_momentum_20d(close: np.ndarray) -> np.ndarray:
    """
    20日动量（skip 2D）：close[t-2] / close[t-22] - 1
    与源码 _compute_momentum_20d_batch 一致。
    """
    N, T = close.shape
    out  = np.full((N, T), np.nan, dtype=np.float64)
    skip, lookback = 2, 20
    if T > lookback + skip:
        p_now  = close[:, skip:T - lookback]       # close[t-skip]
        p_past = close[:, :T - lookback - skip]    # close[t-lookback-skip]
        ok = (p_past > 1e-8) & ~np.isnan(p_past) & ~np.isnan(p_now)
        out[:, lookback + skip:] = np.where(ok, p_now / (p_past + 1e-10) - 1.0, np.nan)
    return out


def _zscore_cs(mat: np.ndarray) -> np.ndarray:
    """截面 Z-Score（样本标准差），零方差填 0（与源码一致）。"""
    N, T = mat.shape
    out  = np.full((N, T), np.nan, dtype=np.float64)
    for t in range(T):
        col   = mat[:, t]
        ok    = ~np.isnan(col)
        valid = col[ok]
        if len(valid) < 5:
            continue
        mu  = valid.mean()
        std = valid.std(ddof=1)
        if std < 1e-10:
            out[ok, t] = 0.0
        else:
            out[:, t] = (col - mu) / std
    return out


def _winsorize_cs(mat: np.ndarray, n_sigma: float = 3.0) -> np.ndarray:
    """截面 3σ-Winsorize（零方差填均值，与源码一致）。"""
    N, T = mat.shape
    out  = mat.copy()
    for t in range(T):
        col   = mat[:, t]
        ok    = ~np.isnan(col)
        valid = col[ok]
        if len(valid) < 3:
            continue
        mu  = valid.mean()
        std = valid.std(ddof=1)
        if std < 1e-10:
            out[ok, t] = mu
        else:
            lo, hi     = mu - n_sigma * std, mu + n_sigma * std
            out[ok, t] = np.clip(valid, lo, hi)
    return out


def _apply_concept_resonance(
    alpha_score: np.ndarray,    # (N, T)
    whale_pulse: np.ndarray,    # (N, T)
    concept_ids: np.ndarray,    # (N, T) uint16
    top_pct    : float = 0.10,
    boost      : float = 1.5,
) -> np.ndarray:
    """
    板块共振加权（与源码 _apply_concept_resonance 完全对齐）。
    强势概念（Whale_Pulse 均值前 top_pct）内个股 Alpha × boost。
    """
    N, T    = alpha_score.shape
    boosted = alpha_score.copy()

    for t in range(T):
        wp_t    = whale_pulse[:, t]
        cid_t   = concept_ids[:, t].astype(np.int32)
        alpha_t = alpha_score[:, t]

        unique_cids = np.unique(cid_t)
        unique_cids = unique_cids[unique_cids > 0]
        if len(unique_cids) == 0:
            continue

        concept_avg: dict = {}
        for cid in unique_cids:
            mask  = (cid_t == cid)
            valid = wp_t[mask]
            valid = valid[~np.isnan(valid)]
            if len(valid) >= 2:
                concept_avg[int(cid)] = float(np.mean(valid))

        if len(concept_avg) < 2:
            continue

        sorted_cids = sorted(concept_avg, key=lambda c: concept_avg[c], reverse=True)
        top_k       = max(1, int(np.ceil(len(sorted_cids) * top_pct)))
        resonance   = set(sorted_cids[:top_k])

        for i in range(N):
            if int(cid_t[i]) in resonance and not np.isnan(alpha_t[i]):
                boosted[i, t] = alpha_t[i] * boost

    return boosted


# ─────────────────────────────────────────────────────────────────────────────
# 主 Alpha 函数
# ─────────────────────────────────────────────────────────────────────────────

@register_vec_strategy("snma_v4")
def snma_v4_alpha(
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
    SNMA-V4 + THS 板块共振纯因子版。

    因子流水线（与源码完全对齐）：
      Step 1: Whale_Pulse_raw
      Step 2: Winsorize → 截面 Rank → wp_rank
      Step 3: Momentum_20d 截面 Rank → mom_rank
      Step 4: DQC_Gap = wp_rank - weight_mom × mom_rank
      Step 5: DQC_Gap 截面 Z-Score → alpha_raw
      Step 6: 板块共振加权（若提供 concept_ids）
      Step 7: 价格门控 + valid_mask → score；Top-N 等权

    amount=None 时用 close×volume×100 近似（与源码一致）。
    SOFT_BEAR/BEAR 时 raw_weights × 0.5（轻度减仓，不清仓）。
    """
    N, T = close.shape

    extra        = getattr(params, "extra", {}) or {} if params else {}
    top_n        = int(extra.get("top_n",          getattr(params, "top_n",        30)  if params else 30))
    min_price    = float(extra.get("min_price",    getattr(params, "min_price",    3.0) if params else 3.0))
    top_pct      = float(extra.get("concept_top_pct", 0.10))
    boost        = float(extra.get("boost_factor",    1.5))
    weight_mom   = float(extra.get("weight_mom",      0.5))
    score_thr    = float(extra.get("score_thr",       0.0))
    max_s_pos    = float(getattr(params, "max_single_pos", 0.10)) if params else 0.10

    amount    = kw.get("amount", None)
    concept_ids = kw.get("concept_ids", None)

    # ── 成交额准备 ───────────────────────────────────────────────────────────
    if amount is None:
        amount_arr = (close * volume * 100.0).astype(np.float64)
    else:
        amount_arr = np.asarray(amount, dtype=np.float64)
    amount_arr = np.where(
        np.isnan(amount_arr) | np.isinf(amount_arr) | (amount_arr < 0),
        0.0, amount_arr
    )

    # ── Step 1: Whale_Pulse_raw ──────────────────────────────────────────────
    wp_raw = _compute_whale_pulse(
        close.astype(np.float64), open_.astype(np.float64),
        high.astype(np.float64),  low.astype(np.float64),
        volume.astype(np.float64), amount_arr,
    )

    # ── Step 2: WhalePulse 3日滚动均值平滑 + Winsorize + 截面 Rank ───────────
    # [FIX-SN-01] wp_raw 是日线量价比，日间波动极大导致排名每天大幅变动
    # 添加3日滚动均值平滑后，截面排名更稳定，换手率可降低约40-60%
    wp_smooth = np.full_like(wp_raw, np.nan)
    for t in range(2, T):
        w3 = wp_raw[:, t-2:t+1]          # (N, 3)
        cnt = (~np.isnan(w3)).sum(axis=1)
        wp_smooth[:, t] = np.where(cnt >= 2, np.nanmean(w3, axis=1), np.nan)
    # 前2天无3日窗口，用原始值填充（不丢失早期信号）
    wp_smooth[:, :2] = wp_raw[:, :2]

    wp_wins = _winsorize_cs(wp_smooth)
    wp_rank = _cross_section_rank_batch(wp_wins)

    # ── Step 3: Momentum_20d 截面 Rank ───────────────────────────────────────
    mom_raw  = _compute_momentum_20d(close.astype(np.float64))
    mom_rank = _cross_section_rank_batch(mom_raw)

    # ── Step 4: DQC_Gap ──────────────────────────────────────────────────────
    valid_both = ~(np.isnan(wp_rank) | np.isnan(mom_rank))
    dqc_gap    = np.full((N, T), np.nan, dtype=np.float64)
    dqc_gap[valid_both] = wp_rank[valid_both] - weight_mom * mom_rank[valid_both]

    # ── Step 5: Alpha 原始分（截面 Z-Score）──────────────────────────────────
    # [FIX-EMA-02] 对 DQC_Gap 做因子级 EMA 平滑（稳定截面排名，降低换手率）
    _ema = int(getattr(params, "factor_ema_span", 5)) if params else 5
    dqc_gap_s = _ema_smooth_factor(dqc_gap, _ema) if _ema > 1 else dqc_gap
    alpha_raw = _zscore_cs(dqc_gap_s)

    # ── Step 6: 板块共振加权 ─────────────────────────────────────────────────
    if concept_ids is not None:
        cid_arr = np.asarray(concept_ids).astype(np.uint16)
        # 扩展为 (N, T)（若传入 (N,) 静态 ID）
        if cid_arr.ndim == 1:
            cid_arr = np.broadcast_to(cid_arr[:, None], (N, T)).copy()
        alpha_boosted = _apply_concept_resonance(alpha_raw, wp_raw, cid_arr, top_pct, boost)
    else:
        alpha_boosted = alpha_raw

    # ── Step 7: 门控 + score ─────────────────────────────────────────────────
    price_ok  = (close >= min_price) & ~np.isnan(close) & (close > 0)
    alpha_ok  = ~np.isnan(alpha_boosted)
    if valid_mask is not None:
        vm = np.asarray(valid_mask, dtype=bool)
    else:
        vm = np.ones((N, T), dtype=bool)

    buy_ok      = price_ok & vm & alpha_ok
    score_masked = alpha_boosted.copy()
    score_masked[~buy_ok]             = -np.inf
    score_masked[score_masked < score_thr] = -np.inf

    # [L4-FIX] 涨跌停硬封锁：防止利用涨跌停期间的虚假Alpha信号
    limit_mask = kw.get("limit_reach_mask")
    if limit_mask is not None:
        score_masked[limit_mask] = -np.inf

    # [L4-FIX] 数值稳态保护
    score_masked = np.nan_to_num(score_masked, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)

    # ── Top-N 等权 ────────────────────────────────────────────────────────────
    # [FIX-DB-01] 策略级防抖参数：可通过 params 或 config.json.strategy_params 配置
    dropout_d   = 3   # ★[FIX-DEBOUNCE] 策略专属，来自 exit_config 规格
    exit_buf    = 5   # ★[FIX-DEBOUNCE] 策略专属，来自 exit_config 规格
    raw_weights = _score_to_weights(score_masked, top_n=top_n, max_single_pos=max_s_pos,
                                      dropout_days=dropout_d, exit_buffer=exit_buf,
                                      hard_invalid=None if valid_mask is None else ~np.asarray(valid_mask, dtype=bool))

    # [FIX-B12] 删除策略层手动 ×0.5 减仓。
    # PortfolioBuilder 已通过 regime_limits（SOFT_BEAR=0.4, BEAR=0.0）统一缩放，
    # 若策略层再 ×0.5，会造成双重减仓（实际仓位只有预期的20%），严重欠配。
    # 市场环境判断统一交由 PortfolioBuilder 处理，策略层只输出纯 Alpha 权重。

    return AlphaSignal(
        raw_target_weights=raw_weights,
        score=score_masked,
        strategy_name="snma_v4",
        exit_config={
            "stop_mode"       : "trailing",
            "hard_stop_loss"  : 0.15,
            "take_profit"     : 0.25,
            "max_holding_days": 25,
            "dropout_days"    : 3,
            "exit_buffer"     : 5,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# 验收测试
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    rng = np.random.default_rng(6)
    N, T = 50, 300
    c  = np.cumprod(1 + rng.normal(0.0003, 0.016, (N, T)), axis=1).astype(np.float64) * 10
    h  = c * (1 + rng.uniform(0, 0.02, (N, T)))
    l  = c * (1 - rng.uniform(0, 0.02, (N, T)))
    o  = c * (1 + rng.normal(0, 0.005, (N, T)))
    v  = rng.uniform(1e6, 5e7, (N, T))
    amt = c * v * rng.uniform(0.8, 1.2, (N, T))

    # 模拟概念 ID（5个概念）
    cid_1d      = rng.integers(0, 6, N).astype(np.uint16)
    concept_ids = np.broadcast_to(cid_1d[:, None], (N, T)).copy()
    reg         = np.array([2] * 250 + [3] * 30 + [4] * 20, dtype=np.int8)

    alpha = snma_v4_alpha(c, o, h, l, v, None,
                          market_regime=reg, amount=amt, concept_ids=concept_ids)

    assert alpha.raw_target_weights.shape == (N, T), \
        f"shape FAIL: {alpha.raw_target_weights.shape}"
    assert not np.any(alpha.raw_target_weights < 0), "negative weight FAIL"
    assert alpha.raw_target_weights.sum(axis=0).max() <= 1.0 + 1e-6, \
        f"col_sum FAIL: {alpha.raw_target_weights.sum(axis=0).max():.6f}"

    nz = (alpha.raw_target_weights.sum(axis=0) > 0).sum()
    print(f"[PASS] snma_v4_alpha: shape={alpha.raw_target_weights.shape} "
          f"max_col_sum={alpha.raw_target_weights.sum(axis=0).max():.4f} "
          f"nonzero_cols={nz}/{T} ✓")
    sys.exit(0)
