from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

# Core registration imports
from src.strategies.registry import register_vec_strategy
from src.strategies.alpha_signal import AlphaSignal, _score_to_weights

@register_vec_strategy("sniper_v13_hfq")
def sniper_v13_hfq_alpha(
    close: np.ndarray, open_: np.ndarray, high: np.ndarray, low: np.ndarray,
    volume: np.ndarray, amount: np.ndarray, valid_mask: np.ndarray,
    **kwargs
) -> AlphaSignal:
    """
    Q-UNITY V10 Sniper Mode [HFQ Edition] - Vectorized Industrial Version
    - Optimized for 50,000 CNY Retail Portfolios
    - High-integrity registration fix (No hidden dependencies)
    """
    N, T = close.shape
    rsrs_window = 18
    zscore_window = 300

    # 1. Micro-cap filter (Bottom 20% by Market Cap)
    mcap = kwargs.get("market_cap_total")
    mcap_rank_mask = np.zeros_like(close, dtype=bool)
    if mcap is not None:
        for t in range(T):
            day_mcap = mcap[:, t]
            active_today = valid_mask[:, t]
            if np.any(active_today):
                valid_day_mcap = day_mcap[active_today]
                thresh = np.percentile(valid_day_mcap, 20)
                mcap_rank_mask[:, t] = (day_mcap <= thresh)
    else:
        mcap_rank_mask = np.ones_like(close, dtype=bool)

    # 2. RSRS Standardized Slope
    rsrs_slope = np.zeros_like(close, dtype=np.float32)
    for t in range(rsrs_window, T):
        h_win = high[:, t-rsrs_window:t]
        l_win = low[:, t-rsrs_window:t]
        l_mean = np.mean(l_win, axis=1, keepdims=True)
        h_mean = np.mean(h_win, axis=1, keepdims=True)
        cov_lh = np.mean((l_win - l_mean) * (h_win - h_mean), axis=1)
        var_l  = np.var(l_win, axis=1)
        rsrs_slope[:, t] = cov_lh / (var_l + 1e-9)

    rsrs_z = np.zeros_like(rsrs_slope)
    for t in range(zscore_window, T):
        win = rsrs_slope[:, t-zscore_window:t]
        std_ = np.std(win, axis=1)
        mean_ = np.mean(win, axis=1)
        rsrs_z[:, t] = (rsrs_slope[:, t] - mean_) / (std_ + 1e-9)

    # 2b. [FIX-SN-01] Factor Smoothing (EMA)
    # Apply EMA to the RSRS-z factor to reduce ranking jitter
    rsrs_z = _ema_smooth_factor(rsrs_z.astype(np.float64), span=5).astype(np.float32)

    # 3. Market Regime Gate
    regime = kwargs.get("market_regime")
    regime_gate = (regime >= 2) if regime is not None else np.ones(T, dtype=bool)

    # 4. Strength Confirmation (Close Location)
    # Reject "Upper Wick" traps; ensure price is in top 60% of daily range
    close_loc = (close - low) / (high - low + 1e-9)
    strength_gate = (close_loc > 0.40)

    # 5. Combined Signal Generation
    final_signal = (
        (rsrs_z > 0.8) &
        strength_gate &
        mcap_rank_mask &
        valid_mask
    )
    # Apply regime gate across all stocks
    final_signal &= regime_gate[None, :]

    scores = np.zeros_like(close, dtype=np.float32)
    scores[final_signal] = rsrs_z[final_signal]
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)

    # 6. Optimized Portfolio Weights (Top 10 concentrated for 50k capital)
    raw_weights = _score_to_weights(
        scores,
        top_n=10,
        max_single_pos=0.08,
        exit_buffer=2,
        hard_invalid=~valid_mask
    )

    return AlphaSignal(
        raw_target_weights = raw_weights,
        score              = scores,
        strategy_name      = "sniper_v13_hfq",
        exit_config        = {
            "stop_mode"       : "entry_price",
            "hard_stop_loss"  : 0.12,
            "take_profit"     : 0.25,
            "max_holding_days": 20,
        }
    )
