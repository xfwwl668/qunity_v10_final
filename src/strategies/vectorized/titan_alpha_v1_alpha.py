"""
Q-UNITY V10 — titan_alpha_v1_alpha.py
=======================================
TitanAlpha V1 纯因子版（AlphaSignal 输出）

迁移自 titan_alpha_v1_vec.py，删除风控层，保留完整因子流水线：

✘ 删除：四态市场状态机（is_bear/is_bull）/ ever_bought / dropout
✘ 删除：sell_signals / buy_signals 双轨
✓ 保留：6因子全流程（F1 RSRS-R² / F2 EP / F3 ROE动量 / F4 SUE衰减 / F5 LVOL / F6 MOM）
✓ 保留：三步中性化（Winsorize + 行业Z-Score + OLS市值残差）
✓ 保留：门控（RSRS方向 / R² / 最低价格 / 波动率上限 / valid_mask）
✓ Regime → FACTOR_WEIGHTS 动态权重（★铁律：int8→int→str→查表）
✓ BEAR 状态所有权重为 0.0 → score = -inf → 自然空仓
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

try:
    from src.strategies.alpha_signal import AlphaSignal, _score_to_weights, _ema_smooth_factor
except ImportError:
    from alpha_signal import AlphaSignal, _score_to_weights          # type: ignore

# ─────────────────────────────────────────────────────────────────────────────
# ★ 铁律：REGIME_IDX_TO_STR + FACTOR_WEIGHTS 必须在文件顶部定义
# market_regime 是 int8 数组；访问前先 int(regime[t]) → str → 查 FACTOR_WEIGHTS
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


REGIME_IDX_TO_STR: dict[int, str] = {
    0: 'STRONG_BULL',
    1: 'BULL',
    2: 'NEUTRAL',
    3: 'SOFT_BEAR',
    4: 'BEAR',
}

# 因子维度：rsrs=F1, momentum=F6, financial=F2+F3+F4合并, vol=F5, concept=回退动量
FACTOR_WEIGHTS: dict[str, dict[str, float]] = {
    'STRONG_BULL': {'rsrs': 0.15, 'momentum': 0.50, 'financial': 0.15, 'vol': 0.10, 'concept': 0.10},
    'BULL'       : {'rsrs': 0.25, 'momentum': 0.35, 'financial': 0.20, 'vol': 0.10, 'concept': 0.10},
    'NEUTRAL'    : {'rsrs': 0.35, 'momentum': 0.20, 'financial': 0.25, 'vol': 0.10, 'concept': 0.10},
    'SOFT_BEAR'  : {'rsrs': 0.45, 'momentum': 0.05, 'financial': 0.30, 'vol': 0.15, 'concept': 0.05},
    'BEAR'       : {'rsrs': 0.00, 'momentum': 0.00, 'financial': 0.00, 'vol': 0.00, 'concept': 0.00},
}

_DEFAULT_REGIME_STR = 'NEUTRAL'


# ─────────────────────────────────────────────────────────────────────────────
# 纯 NumPy 基础内核（忠实移植自 Numba 版本）
# ─────────────────────────────────────────────────────────────────────────────

def _ols_beta_r2(
    high: np.ndarray, low: np.ndarray, window: int
) -> Tuple[np.ndarray, np.ndarray]:
    """批量滑动 OLS Beta + R²，窗口 [t-window:t]，不含当日（与源码一致）。"""
    N, T = high.shape
    beta = np.full((N, T), np.nan, dtype=np.float64)
    r2   = np.full((N, T), np.nan, dtype=np.float64)
    for t in range(window, T):
        x  = low[:,  t - window:t]
        y  = high[:, t - window:t]
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


def _ts_zscore(mat: np.ndarray, window: int, min_count: int = 30) -> np.ndarray:
    """时序滚动 Z-Score（样本标准差），(N,T)→(N,T)。"""
    N, T = mat.shape
    z = np.full((N, T), np.nan, dtype=np.float64)
    for t in range(window, T):
        w   = mat[:, t - window:t]
        cnt = (~np.isnan(w)).sum(1)
        ok  = cnt >= min_count
        mu  = np.nanmean(w, axis=1)
        std = np.nanstd(w, axis=1, ddof=1)
        cur = mat[:, t]
        v   = ok & (std > 1e-10) & ~np.isnan(cur)
        z[v, t] = (cur[v] - mu[v]) / std[v]
    return z


def _cs_zscore(mat: np.ndarray) -> np.ndarray:
    """截面 Z-Score（样本标准差），零方差列填 0（与 _ta_zscore_cs 一致）。"""
    N, T = mat.shape
    z = np.full((N, T), np.nan, dtype=np.float64)
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


def _winsorize_cs(mat: np.ndarray, n_sigma: float = 3.0) -> np.ndarray:
    """截面 3σ-Winsorize（与 _ta_winsorize_cs 一致）。"""
    N, T = mat.shape
    out = mat.copy()
    for t in range(T):
        col = mat[:, t]
        ok  = ~np.isnan(col)
        if ok.sum() < 3:
            continue
        mu  = col[ok].mean()
        std = col[ok].std(ddof=1)
        if std < 1e-10:
            out[ok, t] = mu
            continue
        lo, hi = mu - n_sigma * std, mu + n_sigma * std
        out[ok, t] = np.clip(col[ok], lo, hi)
    return out


def _sector_zscore(
    mat: np.ndarray, sector_ids: np.ndarray, n_sectors: int
) -> np.ndarray:
    """行业内截面 Z-Score（与 _ta_sector_zscore 一致）。"""
    N, T = mat.shape
    z = np.full((N, T), np.nan, dtype=np.float64)
    for t in range(T):
        col = mat[:, t]
        for sec in range(n_sectors):
            idx = np.where((sector_ids == sec) & ~np.isnan(col))[0]
            if len(idx) < 3:
                continue
            mu  = col[idx].mean()
            std = col[idx].std(ddof=1)
            z[idx, t] = (col[idx] - mu) / std if std > 1e-10 else 0.0
        idx_unc = np.where((sector_ids < 0) & ~np.isnan(col))[0]
        if len(idx_unc) >= 2:
            mu  = col[idx_unc].mean()
            std = col[idx_unc].std(ddof=1)
            z[idx_unc, t] = (col[idx_unc] - mu) / std if std > 1e-10 else 0.0
    return z


def _ols_mktcap_residual(fz: np.ndarray, log_mc: np.ndarray) -> np.ndarray:
    """OLS 残差法消除市值效应（与 _ta_mktcap_ols_residual 一致）。"""
    N, T = fz.shape
    res = np.full((N, T), np.nan, dtype=np.float64)
    for t in range(T):
        fc = fz[:, t]; mc = log_mc[:, t]
        ok = ~np.isnan(fc) & ~np.isnan(mc) & (mc > 0)
        if ok.sum() < 10:
            res[~np.isnan(fc), t] = fc[~np.isnan(fc)]; continue
        mf = fc[ok].mean(); mm = mc[ok].mean()
        cov = ((fc[ok] - mf) * (mc[ok] - mm)).sum()
        var = ((mc[ok] - mm) ** 2).sum()
        if var < 1e-10:
            res[~np.isnan(fc), t] = fc[~np.isnan(fc)]; continue
        b = cov / var; a = mf - b * mm
        ok2 = ~np.isnan(fc) & ~np.isnan(mc) & (mc > 0)
        res[ok2, t] = fc[ok2] - a - b * mc[ok2]
        ok3 = ~np.isnan(fc) & ~ok2
        res[ok3, t] = fc[ok3]
    return res


def _neutralize_3step(
    raw: np.ndarray,
    sector_ids: np.ndarray,
    n_sectors: int,
    log_mktcap: Optional[np.ndarray],
) -> np.ndarray:
    """三步中性化：Winsorize → 行业Z-Score → OLS市值残差 → 截面Z-Score。"""
    f = _winsorize_cs(raw)
    f = _sector_zscore(f, sector_ids, n_sectors) if n_sectors > 0 else _cs_zscore(f)
    if log_mktcap is not None:
        f = _ols_mktcap_residual(f, log_mktcap)
    f = _cs_zscore(f)
    if np.all(np.isnan(f)):
        f = _winsorize_cs(raw)
    return f


def _build_sector_ids(
    sector_matrix: Optional[np.ndarray], N: int
) -> Tuple[np.ndarray, int]:
    if sector_matrix is None:
        return np.full(N, -1, dtype=np.int32), 0
    S = sector_matrix.shape[1]
    ids = np.full(N, -1, dtype=np.int32)
    for i in range(N):
        for s in range(S):
            if sector_matrix[i, s]:
                ids[i] = s; break
    return ids, S


def _realized_vol(close: np.ndarray, window: int = 20) -> np.ndarray:
    """20日年化波动率（与 _ta_realized_vol 一致）。"""
    N, T = close.shape
    vol = np.full((N, T), np.nan, dtype=np.float64)
    for t in range(window + 1, T):
        p0 = close[:, t - window - 1:t - 1]
        p1 = close[:, t - window:t]
        ok = (p0 > 1e-8) & (p1 > 1e-8)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(ok, np.log(np.where(ok, p1 / (p0 + 1e-10), 1.0)), np.nan)
        cnt = ok.sum(1)
        mu  = np.nanmean(r, axis=1)
        dev = np.where(ok, (r - mu[:, None]) ** 2, np.nan)
        var_arr = np.where(
            cnt >= window // 2,
            np.nansum(dev, axis=1) / np.maximum(cnt - 1, 1), np.nan
        )
        var_arr = np.where(cnt >= window // 2, np.maximum(var_arr, 0.0), np.nan)
        vol[:, t] = np.sqrt(np.where(var_arr > 0, var_arr * 252.0, np.nan))
    return vol


# ─────────────────────────────────────────────────────────────────────────────
# 主 Alpha 函数
# ─────────────────────────────────────────────────────────────────────────────

@register_vec_strategy("titan_alpha_v1")
def titan_alpha_v1_alpha(
    close        : np.ndarray,
    open_        : np.ndarray,
    high         : np.ndarray,
    low          : np.ndarray,
    volume       : np.ndarray,
    params,
    valid_mask   : Optional[np.ndarray] = None,
    market_regime: Optional[np.ndarray] = None,   # (T,) int8 ← V10 铁律
    **kw,
) -> AlphaSignal:
    """
    TitanAlpha V1 纯因子版：6因子 + 三步中性化 + Regime 动态权重。

    ★ market_regime (int8) → REGIME_IDX_TO_STR[int(r)] → FACTOR_WEIGHTS[str]
    ★ BEAR 权重全 0.0 → score = -inf → weights = 0（自然空仓）
    ★ 门控保留（RSRS方向 / R² / 价格 / 波动率 / valid_mask）
    ★ 减仓：raw_weights[mask] *= 0.5
    """
    N, T = close.shape

    rsrs_w    = int(getattr(params, "rsrs_window", 18)) if params else 18
    zscore_w  = int(getattr(params, "titan_zscore_window",
                            getattr(params, "zscore_window", 300) if params else 300
                            ) if params else 300)
    top_n     = int(getattr(params,  "top_n",                25))  if params else 25
    min_price = float(getattr(params,"min_price",            5.0)) if params else 5.0
    r2_thresh = float(getattr(params,"rsrs_r2_threshold",   0.45)) if params else 0.45
    max_s_pos = float(getattr(params,"max_single_pos",       0.20))if params else 0.20
    max_sec_exp = float(getattr(params, "max_sector_exposure", 0.20)) if params else 0.20
    vol_cap   = 1.20

    pe_matrix    = kw.get("pe_matrix",     None)
    roe_matrix   = kw.get("roe_matrix",    None)
    sue_matrix   = kw.get("sue_matrix",    None)
    mktcap_matrix= kw.get("mktcap_matrix", None)
    days_ann     = kw.get("days_since_ann",None)
    sector_matrix= kw.get("sector_matrix", None)

    sector_ids, n_sec = _build_sector_ids(sector_matrix, N)
    log_mktcap: Optional[np.ndarray] = None
    if mktcap_matrix is not None:
        mc = mktcap_matrix.copy()
        mc[mc <= 0] = np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            log_mktcap = np.log(mc)

    # ── F1: RSRS-R² ──────────────────────────────────────────────────────────
    beta_mat, r2_mat = _ols_beta_r2(high, low, rsrs_w)
    rsrs_r2_raw = np.where(np.isnan(beta_mat) | np.isnan(r2_mat),
                            np.nan, beta_mat * r2_mat)
    rsrs_ts_raw = _ts_zscore(beta_mat, zscore_w, 30)   # 方向门控用
    rsrs_r2_ts  = _ts_zscore(rsrs_r2_raw, zscore_w, 30)
    f1_rsrs     = _neutralize_3step(rsrs_r2_ts, sector_ids, n_sec, log_mktcap)

    # ── F2: EP = 1/PE ─────────────────────────────────────────────────────────
    has_f2 = pe_matrix is not None
    if has_f2:
        ep_raw = np.where(
            (pe_matrix > 0.5) & (pe_matrix < 200.0), 1.0 / pe_matrix, np.nan
        )
        f2_ep = _neutralize_3step(ep_raw, sector_ids, n_sec, log_mktcap)
    else:
        f2_ep = np.zeros((N, T), dtype=np.float64)

    # ── F3: ROE 改善（delta vs EWMA span=60）─────────────────────────────────
    has_f3 = roe_matrix is not None
    if has_f3:
        alpha_ew = 2.0 / 61.0
        roe_ewma = np.full((N, T), np.nan, dtype=np.float64)
        roe_ewma[:, 0] = roe_matrix[:, 0]
        for tt in range(1, T):
            prev = roe_ewma[:, tt - 1]
            cur  = roe_matrix[:, tt]
            valid_ew = ~np.isnan(cur) & ~np.isnan(prev)
            roe_ewma[:, tt] = np.where(
                valid_ew, alpha_ew * cur + (1 - alpha_ew) * prev,
                np.where(np.isnan(prev), cur, prev)
            )
        f3_roe = _neutralize_3step(roe_matrix - roe_ewma, sector_ids, n_sec, log_mktcap)
    else:
        f3_roe = np.zeros((N, T), dtype=np.float64)

    # ── F4: SUE 时间衰减 ──────────────────────────────────────────────────────
    has_f4 = sue_matrix is not None
    if has_f4:
        if days_ann is not None:
            d_c = np.maximum(np.nan_to_num(days_ann, nan=0.0), 0.0)
            sue_decayed = sue_matrix * np.exp(-0.03 * d_c)
        else:
            sue_decayed = sue_matrix.copy()
        f4_sue = _neutralize_3step(sue_decayed, sector_ids, n_sec, log_mktcap)
    else:
        f4_sue = np.zeros((N, T), dtype=np.float64)

    # ── F5: LVOL ─────────────────────────────────────────────────────────────
    vol_annual = _realized_vol(close, window=20)
    f5_lvol    = _neutralize_3step(
        np.where(np.isnan(vol_annual), np.nan, -vol_annual),
        sector_ids, n_sec, log_mktcap
    )

    # ── F6: 中期动量（21D, skip 2D）──────────────────────────────────────────
    _mw, _sk = 21, 2
    mom_raw = np.full((N, T), np.nan, dtype=np.float64)
    if T > _mw + _sk:
        with np.errstate(divide="ignore", invalid="ignore"):
            mom_raw[:, _mw + _sk:] = (
                close[:, _mw:-_sk] / (close[:, :T - _mw - _sk] + 1e-10) - 1.0
            )
    f6_mom = _neutralize_3step(mom_raw, sector_ids, n_sec, log_mktcap)

    # ── 门控（保留，删除熊市清仓逻辑）───────────────────────────────────────
    _rsrs_ts = np.nan_to_num(rsrs_ts_raw, nan=0.0)
    gate_trend = _rsrs_ts > 0.0
    gate_r2    = r2_mat > r2_thresh
    gate_price = close >= min_price
    gate_vol_g = np.where(np.isnan(vol_annual), True, vol_annual <= vol_cap)
    gate_vm    = np.asarray(valid_mask, dtype=bool) if valid_mask is not None \
                 else np.ones((N, T), dtype=bool)
    buy_ok = gate_trend & gate_r2 & gate_price & gate_vol_g & gate_vm

    # ── 逐日合成评分（★铁律：int8→int→REGIME_IDX_TO_STR→FACTOR_WEIGHTS）─────
    n_fin_avail = int(has_f2) + int(has_f3) + int(has_f4)
    score_mat = np.full((N, T), np.nan, dtype=np.float64)

    for t in range(T):
        # ★ 铁律
        if market_regime is not None:
            regime_str = REGIME_IDX_TO_STR[int(market_regime[t])]
        else:
            regime_str = _DEFAULT_REGIME_STR

        fw = FACTOR_WEIGHTS[regime_str]
        if sum(fw.values()) < 1e-10:           # BEAR → -inf
            score_mat[:, t] = -np.inf
            continue

        # 财务因子合并（等权可用的 F2/F3/F4）
        if n_fin_avail > 0:
            fin_cols = ([f2_ep[:, t]] if has_f2 else []) + \
                       ([f3_roe[:, t]] if has_f3 else []) + \
                       ([f4_sue[:, t]] if has_f4 else [])
            fin_col = np.nanmean(np.stack(fin_cols, axis=0), axis=0)
        else:
            fin_col = np.zeros(N, dtype=np.float64)

        factor_cols = {
            'rsrs'     : f1_rsrs[:, t],
            'momentum' : f6_mom[:, t],
            'financial': fin_col,
            'vol'      : f5_lvol[:, t],
            'concept'  : np.full(N, np.nan, dtype=np.float64),  # [FIX-S-02] NaN→由w_sum_eff归一化分配
        }

        score_num = np.zeros(N, dtype=np.float64)
        w_sum_eff = np.zeros(N, dtype=np.float64)
        for fname, col_f in factor_cols.items():
            w = fw[fname]
            if w < 1e-10:
                continue
            ok_f = ~np.isnan(col_f)
            score_num += w * np.where(ok_f, col_f, 0.0)
            w_sum_eff += w * ok_f.astype(np.float64)

        w_sum_eff = np.where(w_sum_eff < 1e-6, 1.0, w_sum_eff)
        score_t   = np.where(buy_ok[:, t], score_num / w_sum_eff, -np.inf)
        score_mat[:, t] = score_t

    score_mat[~gate_vm] = -np.inf
    # [FIX-EMA-02] 对合成评分矩阵做因子级 EMA 平滑（稳定多因子截面排名）
    _ema = int(getattr(params, "factor_ema_span", 5)) if params else 5
    score_mat_ema = _ema_smooth_factor(score_mat, _ema) if _ema > 1 else score_mat
    # BEAR 列（score=-inf）EMA 不应传播，需恢复
    bear_mask = np.all(np.isneginf(score_mat), axis=0)
    score_mat_ema[:, bear_mask] = -np.inf
    score_mat_ema[~gate_vm] = -np.inf

    # [FIX-SECTOR] 行业约束前置屏蔽（v9版本删除了此逻辑，此处补回）
    # v9 注释声称 "sector_matrix 通过 score_mat -inf 屏蔽" 但从未执行。
    # 修复：每个时间步对每个行业只保留得分最高的 max_ps 只，其余设为 -inf。
    # 当 sector_matrix 为 None 或 max_sec_exp>=1.0 时跳过。
    if sector_matrix is not None and max_sec_exp < 1.0 and n_sec > 0:
        max_ps = max(1, int(round(max_sec_exp * top_n)))
        S_dim  = sector_matrix.shape[1]
        for t in range(T):
            col = score_mat_ema[:, t]
            if np.all(np.isneginf(col)):
                continue
            for s in range(S_dim):
                members = np.where(sector_matrix[:, s])[0]
                if len(members) == 0:
                    continue
                mem_scores = col[members]
                if np.all(np.isneginf(mem_scores)):
                    continue
                order   = np.argsort(mem_scores)[::-1]
                to_mask = members[order[max_ps:]]
                col[to_mask] = -np.inf
            score_mat_ema[:, t] = col

    # ── Top-N 等权（防抖）────────────────────────────────────────────────────
    # [FIX-BUG1] 原循环（含行业约束）直接写 raw_weights[top_idx,t]=w，完全绕过
    # _score_to_weights 防抖，导致年换手率 8000%+。现改为调用 _score_to_weights，
    # 防抖真正生效。BEAR 列 score_mat=-inf → _score_to_weights 自然输出 0，空仓保持。
    # [FIX-DB-01] 策略级防抖参数：可通过 params 或 config.json.strategy_params 配置
    dropout_d   = 7   # ★[FIX-DEBOUNCE] 策略专属，来自 exit_config 规格
    exit_buf    = 10   # ★[FIX-DEBOUNCE] 策略专属，来自 exit_config 规格
    raw_weights = _score_to_weights(score_mat_ema, top_n=top_n, max_single_pos=max_s_pos,
                                      dropout_days=dropout_d, exit_buffer=exit_buf,
                                      hard_invalid=None if valid_mask is None else ~np.asarray(valid_mask, dtype=bool))

    return AlphaSignal(
        raw_target_weights = raw_weights,
        score              = score_mat,
        strategy_name      = "titan_alpha_v1",
        meta               = {"regime_type": "int8_6factor"},
        exit_config        = {
            "stop_mode"       : "trailing",
            "hard_stop_loss"  : 0.22,   # Regime已处理系统风险，仅防黑天鹅
            "take_profit"     : 0.0,
            "max_holding_days": 0,      # Regime机制决定退出时机
            "dropout_days"    : 7,
            "exit_buffer"     : 10,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# 验收测试
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    rng = np.random.default_rng(2)
    N, T = 50, 300
    c = np.cumprod(1 + rng.normal(0, 0.01, (N, T)), axis=1).astype(np.float64) * 10
    h = c * (1 + rng.uniform(0, 0.005, (N, T)))
    l = c * (1 - rng.uniform(0, 0.005, (N, T)))
    v = rng.uniform(1e6, 1e7, (N, T))

    reg = np.array([0]*50 + [1]*50 + [2]*100 + [3]*60 + [4]*40, dtype=np.int8)
    assert reg.dtype == np.int8, "regime dtype FAIL"

    pe  = rng.uniform(8, 60, (N, T)).astype(np.float64)
    roe = rng.uniform(0.04, 0.25, (N, T)).astype(np.float64)

    alpha = titan_alpha_v1_alpha(c, c, h, l, v, None,
                                  market_regime=reg, pe_matrix=pe, roe_matrix=roe)

    assert alpha.raw_target_weights.shape == (N, T)
    assert not np.any(alpha.raw_target_weights < 0)
    assert alpha.raw_target_weights.sum(axis=0).max() <= 1.0 + 1e-6
    bear_sum = alpha.raw_target_weights[:, 260:].sum()
    assert bear_sum == 0.0, f"BEAR 列非零: {bear_sum}"
    assert REGIME_IDX_TO_STR[int(reg[0])]  == 'STRONG_BULL'
    assert REGIME_IDX_TO_STR[int(reg[-1])] == 'BEAR'
    assert FACTOR_WEIGHTS['BEAR']['rsrs']  == 0.0

    print(f"[PASS] titan_alpha_v1_alpha: shape={alpha.raw_target_weights.shape} "
          f"max_col_sum={alpha.raw_target_weights.sum(axis=0).max():.4f} "
          f"BEAR列权重和={bear_sum:.0f} regime_dtype={reg.dtype} ✓")
    sys.exit(0)
