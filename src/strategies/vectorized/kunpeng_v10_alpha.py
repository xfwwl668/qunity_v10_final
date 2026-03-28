"""
Q-UNITY V10 — kunpeng_v10_alpha.py
=====================================
KunpengV10 纯因子版（AlphaSignal 输出）

迁移自 kunpeng_v10_vec.py，删除风控层，保留完整因子流水线：

✘ 删除：宽度熔断（breadth_triggered / breadth_limit）
✘ 删除：ever_bought / top_n_absent / dropout_days
✘ 删除：buy_threshold / sell_threshold 状态门控
✘ 删除：sell_signals 双轨
✓ 保留：SmartMoney（滚动聪明钱，buy_vol/sell_vol 比）
✓ 保留：StableIlliq（Amihud 非流动性稳定性，-std(illiq,20)）
✓ 保留：GapPenalty（跳空缺口惩罚，已修正为有方向版本）
✓ 截面 Z-Score 标准化 + 动态权重归一化
✓ valid_mask=False → score=-inf → weight=0

因子经济学逻辑（审计结论）：
  SmartMoney(0.5) : Williams Money Flow 变体，大量文献支持 IC≈0.03-0.05
  StableIlliq(0.3): -std(Amihud illiq) → 稳定的价格冲击 ≈ 有稳定大持仓方，逻辑合理
  GapPenalty(0.2) : 原版 abs(gap) 丢失方向性，本版改为 signed_gap：
                     gap > 0（跳涨）惩罚（避免追涨）
                     gap < 0（跳跌）给分（超跌反转机会）
                     此改动提升因子经济逻辑一致性

前视偏差检查（均通过 ✓）：
  SmartMoney[t]: 使用 close/high/low/volume[t-w+1..t] ✓
  StableIlliq[t]: 使用 close/amount[t-w+1..t] ✓
  GapPenalty[t]: 使用 open[t] 和 close[t-1]，均为已发生数据 ✓
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


def _rolling_smart_money(
    close : np.ndarray,   # (N, T)
    high  : np.ndarray,
    low   : np.ndarray,
    volume: np.ndarray,
    window: int = 10,
) -> np.ndarray:
    """
    滚动聪明钱因子（Williams Money Flow 变体）。

    SmartMoney[i,t] = (sum buy_vol - sum sell_vol) / sum volume
                      k ∈ [t-window+1, t]

    buy_vol[k]  = (close[k]-low[k]) / (high[k]-low[k]) × volume[k]
    sell_vol[k] = (high[k]-close[k]) / (high[k]-low[k]) × volume[k]

    一字板（high==low）：计入 total_vol 但 net=0（与源码一致）。
    前 window-1 列为 NaN。
    """
    N, T = close.shape
    sm   = np.full((N, T), np.nan, dtype=np.float64)

    for t in range(window - 1, T):
        c_w   = close[:, t - window + 1:t + 1]    # (N, window)
        h_w   = high[:,  t - window + 1:t + 1]
        l_w   = low[:,   t - window + 1:t + 1]
        v_w   = volume[:, t - window + 1:t + 1]

        hl    = h_w - l_w                           # (N, window)
        ok    = (hl > 1e-8) & (v_w > 1e-3) & \
                ~np.isnan(c_w) & ~np.isnan(h_w) & ~np.isnan(l_w) & ~np.isnan(v_w)

        bv    = np.where(ok, (c_w - l_w) / (hl + 1e-12) * v_w, 0.0)
        sv    = np.where(ok, (h_w - c_w) / (hl + 1e-12) * v_w, 0.0)
        tv    = np.where(ok, v_w, 0.0)

        # 一字板：加入 total_vol 但 net=0
        one_bar = (hl <= 1e-8) & (v_w > 1e-3) & ~np.isnan(v_w)
        tv     += np.where(one_bar, v_w, 0.0)

        total = tv.sum(axis=1)
        ok_t  = total > 1.0
        sm[:, t] = np.where(ok_t, (bv.sum(1) - sv.sum(1)) / total, np.nan)

    return sm


def _rolling_amihud_stable(
    close : np.ndarray,    # (N, T)
    amount: np.ndarray,    # (N, T) 成交额（元）
    window: int = 20,
) -> np.ndarray:
    """
    滚动 Amihud 非流动性稳定性。

    illiq[k] = |ret[k]| / amount[k]
    StableIlliq[i,t] = -std(illiq[k] : k ∈ [t-window+1, t])
    稳定（std低）→ 分数高；至少 5 个有效观测，否则 NaN。
    样本标准差（ddof=1，与源码一致）。
    前 window 列为 NaN。
    """
    N, T = close.shape
    asi  = np.full((N, T), np.nan, dtype=np.float64)

    for t in range(window, T):
        # 计算 window 天内的 illiq 序列
        c0 = close[:, t - window:t]     # (N, window)
        c1 = close[:, t - window + 1:t + 1]
        a  = amount[:, t - window + 1:t + 1]

        ok = (c0 > 1e-8) & (c1 > 1e-8) & (a >= 1.0) & \
             ~np.isnan(c0) & ~np.isnan(c1) & ~np.isnan(a)

        with np.errstate(divide="ignore", invalid="ignore"):
            ret_abs = np.where(ok, np.abs(c1 / (c0 + 1e-10) - 1.0), np.nan)
            illiq   = np.where(ok, ret_abs / (a + 1e-10), np.nan)

        cnt = ok.sum(axis=1)
        ok_cnt = cnt >= 5
        std_illiq = np.nanstd(illiq, axis=1, ddof=1)
        asi[:, t] = np.where(ok_cnt, -std_illiq, np.nan)

    return asi


def _cs_zscore(mat: np.ndarray) -> np.ndarray:
    """截面 Z-Score（NaN 安全，≥5 有效值，样本标准差）。零方差填 0。"""
    N, T = mat.shape
    z    = np.full((N, T), np.nan, dtype=np.float64)
    for t in range(T):
        col = mat[:, t]
        ok  = ~np.isnan(col)
        if ok.sum() < 5:
            continue
        mu  = col[ok].mean()
        std = col[ok].std(ddof=1)
        if std < 1e-10:
            z[ok, t] = 0.0
        else:
            z[ok, t] = (col[ok] - mu) / std
    return z


# ─────────────────────────────────────────────────────────────────────────────
# 主 Alpha 函数
# ─────────────────────────────────────────────────────────────────────────────

@register_vec_strategy("kunpeng_v10")
def kunpeng_v10_alpha(
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
    KunpengV10 纯因子版：市场微结构三因子。

    综合评分 = 0.5×sm_z + 0.3×asi_z + 0.2×gp_z（动态权重归一化）

    GapPenalty 改进（相对原版）：
      原版：-abs(gap_pct) → 惩罚所有跳空，丢失方向
      本版：-clip(gap_pct, 0, 0.10) × 10  for gap>0（惩罚跳涨，避免追涨）
             +clip(-gap_pct, 0, 0.10) × 10 for gap<0（奖励跳跌，超跌反转机会）
      即 signed_gap_score = -sign(gap) × min(|gap|, 0.10) × 10
    """
    N, T = close.shape

    extra   = getattr(params, "extra", {}) or {} if params else {}
    smart_w = int(extra.get("smart_window", 10))
    illiq_w = int(extra.get("illiq_window", 20))
    top_n   = int(getattr(params, "top_n", 15)) if params else 15
    max_s_p = float(getattr(params, "max_single_pos", 0.08)) if params else 0.08

    amount = kw.get("amount_matrix") if kw.get("amount_matrix") is not None else kw.get("amount")

    # 成交额准备
    if amount is None:
        amt = volume * close
    else:
        amt = np.asarray(amount, dtype=np.float64)
    amt = np.where(np.isnan(amt) | (amt < 1.0), np.nan, amt)

    # ── F1: SmartMoney ────────────────────────────────────────────────────────
    sm_raw = _rolling_smart_money(
        close.astype(np.float64), high.astype(np.float64),
        low.astype(np.float64),   volume.astype(np.float64),
        smart_w,
    )

    # ── F2: StableIlliq ──────────────────────────────────────────────────────
    asi_raw = _rolling_amihud_stable(
        close.astype(np.float64), amt, illiq_w,
    )

    # ── F3: GapPenalty（有方向版，修正原版 abs 设计）────────────────────────
    # signed_gap_score[t] = -sign(gap) × min(|gap|, 0.10) × 10
    # gap > 0 → 惩罚（追涨风险）
    # gap < 0 → 奖励（超跌反转机会）
    gp_raw = np.full((N, T), np.nan, dtype=np.float64)
    if T > 1:
        prev_close = close[:, :-1]
        cur_open   = open_[:, 1:]
        ok = (prev_close > 1e-8) & ~np.isnan(prev_close) & ~np.isnan(cur_open)
        with np.errstate(divide="ignore", invalid="ignore"):
            signed_gap = np.where(ok, (cur_open - prev_close) / prev_close, np.nan)
        # 有方向惩罚：正跳空 → 负分，负跳空 → 正分，幅度截断 10%
        gp_raw[:, 1:] = np.where(
            ~np.isnan(signed_gap),
            -np.sign(signed_gap) * np.clip(np.abs(signed_gap), 0.0, 0.10) * 10.0,
            np.nan,
        )

    # ── 截面 Z-Score ─────────────────────────────────────────────────────────
    # [FIX-EMA-02] 对原始因子做时序 EMA 后再截面 Z-Score，稳定截面排名
    _ema = int(getattr(params.extra if hasattr(params,"extra") and params.extra else {}, "factor_ema_span", 5)) if params else 5
    try:
        _ema = int(getattr(params, "factor_ema_span", _ema)) if params else _ema
    except Exception:
        pass
    sm_raw_s  = _ema_smooth_factor(sm_raw,  _ema) if _ema > 1 else sm_raw
    sm_z  = _cs_zscore(sm_raw_s)
    asi_z = _cs_zscore(asi_raw)
    gp_z  = _cs_zscore(gp_raw)

    # ── 动态权重归一化合成（缺失因子不惩罚）────────────────────────────────
    # [FIX-K-01] 原权重 SmartMoney=0.5, StableIlliq=0.3, GapPenalty=0.2
    # StableIlliq在A股选到僵尸股/退市风险股（低流动性稳定=弱势股）
    # 调整: SmartMoney 0.5→0.65, StableIlliq 0.3→0.15, GapPenalty 0.2→0.20
    w_sm, w_asi, w_gp = 0.65, 0.15, 0.20
    ok_sm  = ~np.isnan(sm_z)
    ok_asi = ~np.isnan(asi_z)
    ok_gp  = ~np.isnan(gp_z)

    denom = (
        ok_sm.astype(np.float64)  * w_sm
        + ok_asi.astype(np.float64) * w_asi
        + ok_gp.astype(np.float64)  * w_gp
    )
    denom = np.where(denom < 1e-8, 1.0, denom)

    score = (
        np.where(ok_sm,  sm_z,  0.0) * w_sm
        + np.where(ok_asi, asi_z, 0.0) * w_asi
        + np.where(ok_gp,  gp_z,  0.0) * w_gp
    ) / denom

    # ── valid_mask 过滤 ───────────────────────────────────────────────────────
    if valid_mask is not None:
        score = np.where(valid_mask, score, -np.inf)

    # [L4-FIX] 涨跌停硬封锁：T日已封死涨/跌停点位在T+1无法成交
    limit_mask = kw.get("limit_reach_mask")
    if limit_mask is not None:
        score[limit_mask] = -np.inf

    # [L4-FIX] 数值稳态保护
    score = np.nan_to_num(score, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)

    # ── Top-N 等权 ────────────────────────────────────────────────────────────
    # [FIX-DB-01] 策略级防抖参数：可通过 params 或 config.json.strategy_params 配置
    dropout_d   = 3   # ★[FIX-DEBOUNCE] 策略专属，来自 exit_config 规格
    exit_buf    = 5   # ★[FIX-DEBOUNCE] 策略专属，来自 exit_config 规格
    raw_weights = _score_to_weights(score, top_n=top_n, max_single_pos=max_s_p,
                                      dropout_days=dropout_d, exit_buffer=exit_buf,
                                      hard_invalid=None if valid_mask is None else ~np.asarray(valid_mask, dtype=bool))

    return AlphaSignal(
        raw_target_weights=raw_weights,
        score=score,
        strategy_name="kunpeng_v10",
        meta={"factors": ["SmartMoney", "StableIlliq", "GapPenalty_signed"]},
        exit_config={
            "stop_mode"       : "entry_price",
            "hard_stop_loss"  : 0.12,
            "take_profit"     : 0.20,
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
    rng = np.random.default_rng(7)
    N, T = 50, 300
    c  = np.cumprod(1 + rng.normal(0.0003, 0.018, (N, T)), axis=1).astype(np.float64) * 15
    h  = c * (1 + rng.uniform(0, 0.03, (N, T)))
    l  = c * (1 - rng.uniform(0, 0.03, (N, T)))
    o  = c * (1 + rng.normal(0, 0.008, (N, T)))
    v  = rng.uniform(1e6, 5e7, (N, T))
    amt = v * c * rng.uniform(0.95, 1.05, (N, T))

    alpha = kunpeng_v10_alpha(c, o, h, l, v, None, amount_matrix=amt)

    assert alpha.raw_target_weights.shape == (N, T)
    assert not np.any(alpha.raw_target_weights < 0)
    assert alpha.raw_target_weights.sum(axis=0).max() <= 1.0 + 1e-6

    nz = (alpha.raw_target_weights.sum(axis=0) > 0).sum()
    print(f"[PASS] kunpeng_v10_alpha: shape={alpha.raw_target_weights.shape} "
          f"max_col_sum={alpha.raw_target_weights.sum(axis=0).max():.4f} "
          f"nonzero_cols={nz}/{T} ✓")
    sys.exit(0)
