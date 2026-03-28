"""
Q-UNITY V10 — retail_sniper_v10_alpha.py
============================================
散户狙击 V10：游资缩量首阴战法（专为5万小资金定制）
设计目标：高吞吐、极速复利、截断亏损、规避量化猎杀。

核心逻辑（缩量回踩均线+情绪反核）：
1. 趋势过滤：标的必须在 MA20 之上（右侧强势股）。
2. 缩量洗盘：今日成交量骤降（< 昨日 60%），且今日为阴线或小十字星。
3. 波动要求：振幅足够大（>4%），洗出了恐慌盘。
4. 资金利用极化：只买 Top 2（各 50% 仓位），不拉大锅饭。
5. T+1/T+2 闪电战：6% 止盈，4% 硬止损，最多持仓 3 天。
"""
from __future__ import annotations
import numpy as np
from typing import Optional

try:
    from src.strategies.alpha_signal import AlphaSignal, _score_to_weights
    from src.strategies.registry import register_vec_strategy
except ImportError:
    pass

@register_vec_strategy("retail_sniper_v10")
def retail_sniper_v10_alpha(
    close: np.ndarray, open_: np.ndarray, high: np.ndarray, low: np.ndarray,
    volume: np.ndarray, amount: np.ndarray, valid_mask: np.ndarray,
    **kwargs
) -> AlphaSignal:
    
    N, T = close.shape
    
    # ── 1. 均线与基础特征 ──
    ma20 = np.full((N, T), np.nan, dtype=np.float64)
    for t in range(20, T):
        ma20[:, t] = np.nanmean(close[:, t-20:t], axis=1)
        
    ret = np.full((N, T), np.nan, dtype=np.float64)
    ret[:, 1:] = (close[:, 1:] - close[:, :-1]) / (close[:, :-1] + 1e-8)
    
    vol_ratio = np.full((N, T), np.nan, dtype=np.float64)
    vol_ratio[:, 1:] = volume[:, 1:] / (volume[:, :-1] + 1e-8)
    
    amplitude = np.full((N, T), np.nan, dtype=np.float64)
    amplitude[:, 1:] = (high[:, 1:] - low[:, 1:]) / (close[:, :-1] + 1e-8)

    # ── 2. 游资狙击条件（严苛门控） ──
    # 条件1: 股价在生命线 MA20 之上（强势多头）
    cond_trend = close > ma20
    
    # 条件2: 缩量洗盘（成交量萎缩至昨日 60% 以下，避开量化高频互砍的放量区）
    cond_shrink = vol_ratio < 0.60
    
    # 条件3: 绿盘回踩，但没有跌停（-9% ~ -1% 之间）
    cond_pullback = (ret < -0.01) & (ret > -0.09)
    
    # 条件4: 振幅必须大于 4%（日内有清洗恐慌盘的动作）
    cond_volatility = amplitude > 0.04
    
    # 条件5: 剔除流动性枯竭的死水票（今日成交额 > 3000万，保证 5万 资金能瞬间买出）
    cond_liquidity = amount > 3e7
    
    # 综合买入信号掩码
    action_mask = cond_trend & cond_shrink & cond_pullback & cond_volatility & cond_liquidity & valid_mask
    
    # ── 3. 打分排序（抓最恐慌的、下影线最长的做核按钮反弹） ──
    # 分数 = (Close - Low) / (High - Low)
    # 下影线越长，说明承接盘拉回的力度越巨大，反弹预期越猛烈
    with np.errstate(divide='ignore', invalid='ignore'):
        score = (close - low) / (high - low + 1e-8)
    
    # 不满足绝对游资狙击条件的，直接一票否决
    score[~action_mask] = -np.inf

    # [L4-FIX] 涨跌停硬封锁：T日已封死涨/跌停点位在T+1无法成交
    limit_mask = kwargs.get("limit_reach_mask")
    if limit_mask is not None:
        score[limit_mask] = -np.inf

    # [L4-FIX] 数值稳态保护
    score = np.nan_to_num(score, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)

    # ── 4. 散户级极化资金配置 ──
    raw_weights = _score_to_weights(
        score, 
        top_n=3,                # 只出击排名前 3 的猎物
        max_single_pos=0.45,    # 极其激进：单只股票占据总资金的 45% (留10%现金防爆仓滑点)
        exit_buffer=0,          # 零宽容：不在前3立马走人
        dropout_days=1,         # 毫不恋战：次日不强势立马砍掉
        hard_invalid=~np.asarray(valid_mask, dtype=bool)
    )
    
    return AlphaSignal(
        raw_target_weights = raw_weights,
        score              = score,
        strategy_name      = "retail_sniper_v10",
        exit_config        = {
            "stop_mode"       : "entry_price",  
            "hard_stop_loss"  : 0.045,       # 绝对生命线：亏损 4.5% 直接断臂求生，保住本金
            "take_profit"     : 0.07,        # 闪电止盈：赚 7% 马上落袋，绝不格局（对抗量化砸盘）
            "max_holding_days": 3,           # T+1 / T+2 战法：最长只拿 3 天，不涨就滚
        },
    )
