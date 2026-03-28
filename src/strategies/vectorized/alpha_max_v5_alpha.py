"""
Q-UNITY V10 — alpha_max_v5_alpha.py
=====================================
AlphaMaxV5 纯因子版（AlphaSignal 输出）

迁移自 alpha_max_v5_vec.py，删除风控层，保留完整七因子流水线：

✘ 删除：buy_threshold / sell_threshold 门控 / ever_bought / dropout
✘ 删除：sell_signals 双轨
✓ 保留：F1 EP(1/PE) + F2 Growth + F3 Momentum + F4 Quality(ROE)
         + F5 Reversal + F6 Liquidity(-Amihud) + F7 ResidualVol
✓ 保留：截面 Z-Score + 动态权重归一化（缺失因子不惩罚有效股票）
✓ 保留：复合分数再 Z-Score（提升跨期可比性）
✓ 保留：PE/ROE/Growth 外部矩阵优雅降级
✓ valid_mask=False → score=-inf → weight=0

因子经济学逻辑（审计结论，均通过）：
  F1 EP(0.20)      : Fama-French 价值因子，文献最强支持之一
  F2 Growth(0.15)  : 成长质量，A股盈利增速与未来收益正相关
  F3 Momentum(0.15): 20D动量；A股短期动量弱于成熟市场，建议权重偏低✓
  F4 ROE(0.20)     : 盈利质量因子，低波动高收益来源
  F5 Reversal(0.10): 5D超跌反转，A股显著存在
  F6 -Amihud(0.10) : 流动性溢价，低流动性股票长期超额收益
  F7 -ResVol(0.10) : 低特质波动率异象，全球普遍存在

前视偏差检查（均通过 ✓）：
  所有 OHLCV 因子在 close[t] 计算，信号用于 t+1 开盘 ✓
  PE/ROE/Growth 使用公告日前向填充，无未来数据 ✓
  ResidualVol 中 mkt_ret[t] = 当日等权市场收益，已发生 ✓

潜在改进建议（未影响迁移决定）：
  1. 将 F3 Momentum 窗口从 20D 延长至 60D（规避 A 股短期反转区间）
  2. F7 mkt_ret 可改用预置指数收益而非等权代理
"""
from __future__ import annotations

import numpy as np
from typing import Optional

try:
    from src.strategies.alpha_signal import AlphaSignal, _score_to_weights, _ema_smooth_factor
except ImportError:
    from alpha_signal import AlphaSignal, _score_to_weights          # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# 因子内核（忠实移植自源码 Python/Numba 版）
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


def _compute_market_ret(
    close     : np.ndarray,
    valid_mask: Optional[np.ndarray],
) -> np.ndarray:
    """每日等权市场收益率 (T,)，用于残差波动率计算。"""
    N, T   = close.shape
    mkt    = np.zeros(T, dtype=np.float64)
    c_cur  = close[:, 1:]
    c_prev = close[:, :-1]
    ok     = (c_prev > 1e-8) & (c_cur > 1e-8) & ~np.isnan(c_prev) & ~np.isnan(c_cur)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.where(ok, c_cur / c_prev - 1.0, np.nan)
    if valid_mask is not None:
        ret = np.where(valid_mask[:, 1:], ret, np.nan)
    n_v     = (~np.isnan(ret)).sum(axis=0)
    mkt[1:] = np.where(n_v >= 5, np.nanmean(ret, axis=0), 0.0)
    return mkt


def _rolling_momentum(close: np.ndarray, window: int = 20) -> np.ndarray:
    """N日价格动量：close[t]/close[t-window] - 1。"""
    N, T = close.shape
    mom  = np.full((N, T), np.nan, dtype=np.float64)
    if T > window:
        with np.errstate(divide="ignore", invalid="ignore"):
            d = close[:, :T - window]
            n = close[:, window:]
            mom[:, window:] = np.where((d > 1e-8) & ~np.isnan(d) & ~np.isnan(n),
                                       n / d - 1.0, np.nan)
    return mom


def _rolling_reversal(close: np.ndarray, window: int = 5) -> np.ndarray:
    """N日短期反转（取负）：-(close[t]/close[t-window]-1)。"""
    N, T = close.shape
    rev  = np.full((N, T), np.nan, dtype=np.float64)
    if T > window:
        with np.errstate(divide="ignore", invalid="ignore"):
            d = close[:, :T - window]
            n = close[:, window:]
            rev[:, window:] = np.where((d > 1e-8) & ~np.isnan(d) & ~np.isnan(n),
                                       -(n / d - 1.0), np.nan)
    return rev


def _rolling_amihud(
    close : np.ndarray,
    amount: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """滚动 Amihud 流动性（取负值）：-mean(|ret|/amount, window)。"""
    N, T = close.shape
    liq  = np.full((N, T), np.nan, dtype=np.float64)
    for t in range(window, T):
        c0  = close[:, t - window:t]
        c1  = close[:, t - window + 1:t + 1]
        a   = amount[:, t - window + 1:t + 1]
        ok  = (c0 > 1e-8) & (c1 > 1e-8) & (a >= 1.0) & \
              ~np.isnan(c0) & ~np.isnan(c1) & ~np.isnan(a)
        with np.errstate(divide="ignore", invalid="ignore"):
            ret_abs = np.where(ok, np.abs(c1 / (c0 + 1e-10) - 1.0), np.nan)
            illiq   = np.where(ok, ret_abs / (a + 1e-10), np.nan)
        cnt = ok.sum(axis=1)
        ok_c = cnt >= 5
        liq[:, t] = np.where(ok_c, -np.nanmean(illiq, axis=1), np.nan)
    return liq


def _compute_resvol(
    close  : np.ndarray,
    mkt_ret: np.ndarray,
    window : int = 60,
) -> np.ndarray:
    """
    残差波动率（-std(ret_i - beta×mkt_ret, window)）。
    OLS: beta = Cov(ret_i, mkt) / Var(mkt)。
    至少 10 个有效观测；样本标准差（ddof=1）。
    """
    N, T = close.shape
    rv   = np.full((N, T), np.nan, dtype=np.float64)

    for t in range(window + 1, T):
        # 滚动日收益率
        c0  = close[:, t - window:t]         # (N, window)
        c1  = close[:, t - window + 1:t + 1]
        mr  = mkt_ret[t - window + 1:t + 1]  # (window,)
        ok  = (c0 > 1e-8) & (c1 > 1e-8) & ~np.isnan(c0) & ~np.isnan(c1) & \
              ~np.isnan(mr[np.newaxis, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            ret_i = np.where(ok, c1 / (c0 + 1e-10) - 1.0, np.nan)
        mr_b = np.where(ok, mr[np.newaxis, :], np.nan)

        cnt  = ok.sum(axis=1)
        ok_c = cnt >= 10

        mu_r = np.nanmean(ret_i, axis=1)
        mu_m = np.nanmean(mr_b,  axis=1)

        cov_rm = np.nanmean((ret_i - mu_r[:, None]) * (mr_b - mu_m[:, None]), axis=1) * cnt / np.maximum(cnt - 1, 1)
        var_m  = np.nanmean((mr_b  - mu_m[:, None]) ** 2, axis=1)                      * cnt / np.maximum(cnt - 1, 1)

        beta   = np.where(var_m > 1e-10, cov_rm / var_m, 1.0)
        resid  = ret_i - beta[:, None] * mr_b
        rv[:, t] = np.where(ok_c, -np.nanstd(resid, axis=1, ddof=1), np.nan)

    return rv


def _cs_zscore(mat: np.ndarray) -> np.ndarray:
    """截面 Z-Score（NaN 安全，≥5 有效值，样本标准差，零方差填 0）。"""
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

@register_vec_strategy("alpha_max_v5")
def alpha_max_v5_alpha(
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
    AlphaMaxV5 纯因子版：机构级七因子多因子模型。

    因子：F1 EP + F2 Growth + F3 Mom + F4 ROE + F5 Rev + F6 Liq + F7 ResVol
    流水线：各因子截面 Z-Score → 动态权重合成 → 复合分数截面 Z-Score → Top-N

    pe_matrix / roe_matrix / growth_matrix 来自 kw，缺失时对应因子权重自动归零重分配。
    """
    N, T = close.shape
    extra = getattr(params, "extra", {}) or {} if params else {}

    ep_w       = float(extra.get("ep_weight",   0.20))
    growth_w   = float(extra.get("growth_w",    0.15))
    mom_w_cfg  = float(extra.get("mom_w",       0.15))
    quality_w  = float(extra.get("quality_w",   0.20))
    rev_w      = float(extra.get("rev_w",       0.10))
    liq_w      = float(extra.get("liq_w",       0.10))
    resvol_w   = float(extra.get("resvol_w",    0.10))
    mom_win    = int(extra.get("mom_window",    20))
    rev_win    = int(extra.get("rev_window",    10))   # [FIX-AM-01] 5→10: 超跌反转窗口延长，减少日间噪声，降低换手率
    liq_win    = int(extra.get("liq_window",    20))
    rv_win     = int(extra.get("resvol_window", 60))
    min_price  = float(extra.get("min_price",   0.0))
    top_n      = int(getattr(params, "top_n",   20)) if params else 20
    max_s_p    = float(getattr(params, "max_single_pos", 0.08)) if params else 0.08

    pe_matrix     = kw.get("pe_matrix",     None)
    roe_matrix    = kw.get("roe_matrix",    None)
    growth_matrix = kw.get("growth_matrix", None)
    amount_matrix = kw.get("amount_matrix") if kw.get("amount_matrix") is not None else kw.get("amount")

    # 成交额准备
    if amount_matrix is None:
        amt = volume * close
    else:
        amt = np.asarray(amount_matrix, dtype=np.float64)
    amt = np.where(np.isnan(amt) | (amt < 1.0), np.nan, amt)

    # 市场收益率（F7 用）
    mkt_ret = _compute_market_ret(close, valid_mask)

    # ── 因子计算 ─────────────────────────────────────────────────────────────
    # F1: EP
    has_ep = pe_matrix is not None
    if has_ep:
        pe     = np.asarray(pe_matrix, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            ep_raw = np.where((pe > 0.5) & (pe < 200.0) & ~np.isnan(pe), 1.0 / pe, np.nan)
    else:
        ep_raw = np.full((N, T), np.nan, dtype=np.float64)

    # F2: Growth
    has_growth = growth_matrix is not None
    growth_raw = np.asarray(growth_matrix, dtype=np.float64) if has_growth else \
                 np.full((N, T), np.nan, dtype=np.float64)

    # F3: Momentum
    mom_raw = _rolling_momentum(close.astype(np.float64), mom_win)

    # F4: Quality (ROE)
    has_roe = roe_matrix is not None
    quality_raw = np.asarray(roe_matrix, dtype=np.float64) if has_roe else \
                  np.full((N, T), np.nan, dtype=np.float64)

    # F5: Reversal
    rev_raw = _rolling_reversal(close.astype(np.float64), rev_win)

    # F6: Liquidity (-Amihud mean)
    liq_raw = _rolling_amihud(close.astype(np.float64), amt, liq_win)

    # F7: ResidualVol
    resvol_raw = _compute_resvol(close.astype(np.float64), mkt_ret, rv_win)

    # ── 截面 Z-Score ─────────────────────────────────────────────────────────
    ep_z      = _cs_zscore(ep_raw)
    growth_z  = _cs_zscore(growth_raw)
    mom_z     = _cs_zscore(mom_raw)
    quality_z = _cs_zscore(quality_raw)
    rev_z     = _cs_zscore(rev_raw)
    liq_z     = _cs_zscore(liq_raw)
    resvol_z  = _cs_zscore(resvol_raw)

    # ── 动态权重合成（缺失因子权重重新归一化）─────────────────────────────────
    base_w = np.array([ep_w, growth_w, mom_w_cfg, quality_w, rev_w, liq_w, resvol_w],
                      dtype=np.float64)
    factors = np.stack([ep_z, growth_z, mom_z, quality_z, rev_z, liq_z, resvol_z], axis=0)
    # (7, N, T)

    valid_f = ~np.isnan(factors)
    w_mat   = base_w[:, np.newaxis, np.newaxis]
    denom   = (valid_f.astype(np.float64) * w_mat).sum(axis=0)   # (N, T)
    denom   = np.where(denom < 1e-8, 1.0, denom)

    filled      = np.where(valid_f, factors, 0.0)
    composite_r = (filled * w_mat).sum(axis=0) / denom
    composite_r = np.where(valid_f.any(axis=0), composite_r, np.nan)

    # 再做截面 Z-Score（提升跨期可比性）
    composite_z = _cs_zscore(composite_r)
    # [FIX-EMA-02] 对合成评分做因子级 EMA 平滑（稳定截面排名，降低换手率）
    _ema = int(getattr(params, "factor_ema_span", 5)) if params else 5
    composite = _ema_smooth_factor(composite_z, _ema) if _ema > 1 else composite_z

    # ── 过滤 ─────────────────────────────────────────────────────────────────
    if valid_mask is not None:
        composite = np.where(valid_mask, composite, -np.inf)
    if min_price > 0.0:
        composite = np.where(close >= min_price, composite, -np.inf)
    composite = np.where(np.isnan(composite), -np.inf, composite)

    # ── Top-N 等权 ────────────────────────────────────────────────────────────
    # [FIX-DB-01] 策略级防抖参数：可通过 params 或 config.json.strategy_params 配置
    dropout_d   = 7   # ★[FIX-DEBOUNCE] 策略专属，来自 exit_config 规格
    exit_buf    = 8   # ★[FIX-DEBOUNCE] 策略专属，来自 exit_config 规格
    raw_weights = _score_to_weights(composite, top_n=top_n, max_single_pos=max_s_p,
                                      dropout_days=dropout_d, exit_buffer=exit_buf,
                                      hard_invalid=None if valid_mask is None else ~np.asarray(valid_mask, dtype=bool))

    active = []
    if has_ep:     active.append("F1-EP")
    if has_growth: active.append("F2-Growth")
    active += ["F3-Mom", "F5-Rev", "F6-Liq", "F7-ResVol"]
    if has_roe:    active.append("F4-Quality")

    return AlphaSignal(
        raw_target_weights=raw_weights,
        score=composite,
        strategy_name="alpha_max_v5",
        meta={"active_factors": active, "n_factors": len(active)},
        exit_config={
            "stop_mode"       : "trailing",
            "hard_stop_loss"  : 0.18,
            "take_profit"     : 0.0,
            "max_holding_days": 60,
            "dropout_days"    : 7,
            "exit_buffer"     : 8,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# 验收测试
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    rng = np.random.default_rng(8)
    N, T = 50, 300

    c  = np.cumprod(1 + rng.normal(0.0002, 0.016, (N, T)), axis=1).astype(np.float64) * 20
    h  = c * (1 + rng.uniform(0, 0.025, (N, T)))
    l  = c * (1 - rng.uniform(0, 0.025, (N, T)))
    o  = c * (1 + rng.normal(0, 0.007, (N, T)))
    v  = rng.uniform(1e6, 5e7, (N, T))
    amt = v * c
    pe  = rng.uniform(5, 100, (N, T)).astype(np.float64)
    roe = rng.uniform(0.03, 0.35, (N, T)).astype(np.float64)
    gr  = rng.uniform(-0.3, 0.8, (N, T)).astype(np.float64)

    # Test 1: 完整七因子
    alpha = alpha_max_v5_alpha(
        c, o, h, l, v, None,
        pe_matrix=pe, roe_matrix=roe, growth_matrix=gr, amount_matrix=amt,
    )
    assert alpha.raw_target_weights.shape == (N, T)
    assert not np.any(alpha.raw_target_weights < 0)
    assert alpha.raw_target_weights.sum(axis=0).max() <= 1.0 + 1e-6
    nz = (alpha.raw_target_weights.sum(axis=0) > 0).sum()
    print(f"[PASS] alpha_max_v5_alpha (7因子): shape={alpha.raw_target_weights.shape} "
          f"max_col_sum={alpha.raw_target_weights.sum(axis=0).max():.4f} "
          f"nonzero_cols={nz}/{T} factors={alpha.meta['active_factors']} ✓")

    # Test 2: 降级（无基本面数据，4因子）
    alpha2 = alpha_max_v5_alpha(c, o, h, l, v, None, amount_matrix=amt)
    assert alpha2.raw_target_weights.shape == (N, T)
    assert not np.any(alpha2.raw_target_weights < 0)
    assert alpha2.raw_target_weights.sum(axis=0).max() <= 1.0 + 1e-6
    nz2 = (alpha2.raw_target_weights.sum(axis=0) > 0).sum()
    print(f"[PASS] alpha_max_v5_alpha (降级4因子): "
          f"max_col_sum={alpha2.raw_target_weights.sum(axis=0).max():.4f} "
          f"nonzero_cols={nz2}/{T} factors={alpha2.meta['active_factors']} ✓")
    sys.exit(0)
