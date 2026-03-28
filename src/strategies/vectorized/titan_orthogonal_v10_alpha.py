"""
Q-UNITY V10 — titan_orthogonal_v10_alpha.py
============================================
泰坦正交 V10：严格中性化 + 残差量价防御型 Alpha
"""
from __future__ import annotations
import numpy as np
from typing import Optional

try:
    from src.strategies.alpha_signal import AlphaSignal, _score_to_weights
    from src.strategies.registry import register_vec_strategy
except ImportError:
    pass
try:
    from numba import njit, prange
except ImportError:
    def njit(*a, **kw): return lambda f: f
    prange = range

def _rsrs_beta_r2(high: np.ndarray, low: np.ndarray, window: int = 18):
    N, T = high.shape
    beta = np.full((N, T), np.nan, dtype=np.float64)
    r2   = np.full((N, T), np.nan, dtype=np.float64)
    for t in range(window - 1, T):
        h_w = high[:, t-window+1:t+1]
        l_w = low[:, t-window+1:t+1]
        l_mean = np.nanmean(l_w, axis=1)
        h_mean = np.nanmean(h_w, axis=1)
        
        # cov and var
        cov = np.nanmean((l_w - l_mean[:, None]) * (h_w - h_mean[:, None]), axis=1)
        var_l = np.nanvar(l_w, axis=1)
        var_h = np.nanvar(h_w, axis=1)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            b = np.where(var_l > 1e-8, cov / var_l, np.nan)
            rho2 = np.where((var_l > 1e-8) & (var_h > 1e-8), 
                            (cov**2) / (var_l * var_h), np.nan)
        beta[:, t] = b
        r2[:, t] = rho2
    return beta, r2

def _ortho_rank_zscore(
    raw_factor: np.ndarray, 
    mktcap: np.ndarray, 
    sector_matrix: np.ndarray, 
    valid_mask: np.ndarray
) -> np.ndarray:
    N, T = raw_factor.shape
    ortho_score = np.full((N, T), np.nan, dtype=np.float64)
    
    for t in range(T):
        valid = valid_mask[:, t] & ~np.isnan(raw_factor[:, t]) & ~np.isnan(mktcap[:, t]) & (mktcap[:, t] > 0)
        valid_idx = np.where(valid)[0]
        if valid_idx.size < 100:
            continue
            
        cap_log = np.log(mktcap[valid_idx, t])
        cap_bins = np.quantile(cap_log, np.linspace(0, 1, 11))
        cap_bins[-1] += 1.0 
        cap_groups = np.digitize(cap_log, cap_bins) - 1
        
        sec = sector_matrix[valid_idx, t] if sector_matrix is not None else np.zeros_like(cap_groups)
        group_keys = sec * 100 + cap_groups
        unique_groups = np.unique(group_keys)
        
        for g in unique_groups:
            g_mask = (group_keys == g)
            g_idx = valid_idx[g_mask]
            if g_idx.size > 2:
                g_vals = raw_factor[g_idx, t]
                ranks = np.argsort(np.argsort(g_vals))
                z = (ranks - np.mean(ranks)) / (np.std(ranks) + 1e-8)
                ortho_score[g_idx, t] = z
                
    return ortho_score

@register_vec_strategy("titan_orthogonal_v10")
def titan_orthogonal_v10_alpha(
    close: np.ndarray, open_: np.ndarray, high: np.ndarray, low: np.ndarray,
    volume: np.ndarray, amount: np.ndarray, valid_mask: np.ndarray,
    sector_matrix: Optional[np.ndarray] = None,
    mktcap_matrix: Optional[np.ndarray] = None,
    **kwargs
) -> AlphaSignal:
    
    N, T = close.shape
    
    beta, r2 = _rsrs_beta_r2(high, low, window=18)
    rsrs_f = beta * r2
    
    with np.errstate(divide='ignore', invalid='ignore'):
        hl_range = high - low
        sm_flow = np.where(hl_range > 1e-8, (close - low) / hl_range * amount, 0.0)
    
    ret_5d = np.full((N, T), np.nan)
    ret_5d[:, 5:] = (close[:, 5:] - close[:, :-5]) / (close[:, :-5] + 1e-8)
    mom_rev = -ret_5d
    
    mkt = mktcap_matrix if mktcap_matrix is not None else (close * volume * 100) 
    
    f1_ortho = _ortho_rank_zscore(rsrs_f, mkt, sector_matrix, valid_mask)
    f2_ortho = _ortho_rank_zscore(sm_flow, mkt, sector_matrix, valid_mask)
    f3_ortho = _ortho_rank_zscore(mom_rev, mkt, sector_matrix, valid_mask)
    
    score = 0.45 * f1_ortho + 0.35 * f3_ortho + 0.20 * f2_ortho

    # [L4-FIX] 涨跌停硬封锁：T日已封死涨/跌停点位在T+1无法成交
    limit_mask = kwargs.get("limit_reach_mask")
    if limit_mask is not None:
        score[limit_mask] = -np.inf

    # [L4-FIX] 数值稳态保护
    score = np.nan_to_num(score, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)

    if valid_mask is not None:
        score[~valid_mask] = -np.inf
    
    raw_weights = _score_to_weights(
        score, 
        top_n=30,               
        max_single_pos=0.04,    
        exit_buffer=10,         
        dropout_days=4,         
        hard_invalid=~np.asarray(valid_mask, dtype=bool) if valid_mask is not None else None
    )
    
    return AlphaSignal(
        raw_target_weights = raw_weights,
        score              = score,
        strategy_name      = "titan_orthogonal_v10",
        exit_config        = {
            "stop_mode"       : "trailing",   
            "hard_stop_loss"  : 0.08,
            "take_profit"     : 0.18,
            "max_holding_days": 18,
        },
    )
