"""
Q-UNITY V10 — src/strategies/vectorized/weak_to_strong_alpha.py
================================================================
弱转强策略（统一版）— 回测与实盘共用同一份策略代码

设计原则：
  - 回测模式：close/open/high/low/volume 来自 npy memmap，
              extra_factors={}，所有因子使用日线代理计算
  - 实盘模式：基础数组来自 LiveDataAdapter，
              extra_factors 包含真实竞价/资金数据，自动覆盖代理因子

信号输出：AlphaSignal.raw_target_weights（V10 标准接口）

因子来源与精度：
  回测模式（代理）           实盘模式（真实数据）
  ─────────────────────────────────────────────────
  炸板识别  日线 high/close    炸板池 API（zhaban_codes）
  竞价强度  open/prev_close   实时 spot 昨收+今开
  主力资金  量价组合近似        股票资金流向 API
  板块强度  日线涨幅中位数      （暂用日线，两模式相同）
  市场情绪  market_regime     market_regime
  技术形态  MA 排列            MA 排列（相同）

铁律（每次修改前默读）
----------------------
1. stamp_tax = 0.0005（万五，由 RiskConfig 控制，本文件不涉及）
2. market_regime 是 int8 数组；查 FACTOR_WEIGHTS 前必须 REGIME_IDX_TO_STR 转换
3. holding_days 递增由内核保证，策略层不干预
4. extra_factors 中任何键缺失或为 None → 自动回退到代理计算

[FIX-BUG4] 修复：全部因子计算代码（炸板/竞价/资金/龙头/板块/情绪/技术）
原先错误地缩进在 for t_cur 循环体外部（函数级），导致只在循环结束后执行一次，
使用最后一次迭代留下的 c0/c1/c2/h1/v1/v2 变量，只写入 raw_weights_full[:, n_days-1]。
即：整个回测期只有最后一天有信号，其余 T-1 天权重全零，策略完全失效。
修复：将所有因子计算+综合评分+入场过滤+权重分配移入 for 循环体内部（缩进正确）。
"""
from __future__ import annotations

import numpy as np

try:
    from src.strategies.alpha_signal import AlphaSignal, REGIME_IDX_TO_STR, _score_to_weights
    from src.strategies.registry import register_vec_strategy
except ImportError:
    from alpha_signal import AlphaSignal, REGIME_IDX_TO_STR, _score_to_weights  # type: ignore
    try:
        from registry import register_vec_strategy  # type: ignore
    except ImportError:
        def register_vec_strategy(name):
            def decorator(fn): return fn
            return decorator


# ─────────────────────────────────────────────────────────────────────────────
# 因子权重（按市场状态分档，来自 hunter_v6 SystemConfig.score_weights）
# ─────────────────────────────────────────────────────────────────────────────

# [FIX-W-02] 统一使用大写键，与 REGIME_IDX_TO_STR 返回值完全对齐。
FACTOR_WEIGHTS: dict = {
    "STRONG_BULL": {
        "zhaban":    0.18,
        "auction":   0.15,
        "money":     0.17,
        "leader":    0.18,
        "sector":    0.17,
        "sentiment": 0.10,
        "technical": 0.05,
    },
    "BULL": {
        "zhaban":    0.20,
        "auction":   0.15,
        "money":     0.20,
        "leader":    0.15,
        "sector":    0.15,
        "sentiment": 0.10,
        "technical": 0.05,
    },
    "NEUTRAL": {
        "zhaban":    0.20,
        "auction":   0.15,
        "money":     0.25,
        "leader":    0.15,
        "sector":    0.10,
        "sentiment": 0.10,
        "technical": 0.05,
    },
    "SOFT_BEAR": {
        "zhaban":    0.22,
        "auction":   0.12,
        "money":     0.28,
        "leader":    0.15,
        "sector":    0.08,
        "sentiment": 0.10,
        "technical": 0.05,
    },
    "BEAR": {
        "zhaban":    0.25,
        "auction":   0.10,
        "money":     0.30,
        "leader":    0.15,
        "sector":    0.05,
        "sentiment": 0.10,
        "technical": 0.05,
    },
}

# [FIX-W-02] 与 FACTOR_WEIGHTS 一致，统一大写键
_SENTIMENT_MAP = {
    "STRONG_BULL": 1.00,
    "BULL":        0.75,
    "NEUTRAL":     0.50,
    "SOFT_BEAR":   0.30,
    "BEAR":        0.15,
}

_EPS = 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# 策略主函数
# ─────────────────────────────────────────────────────────────────────────────

@register_vec_strategy("weak_to_strong")
def weak_to_strong_alpha(
    close:         np.ndarray,          # [n_stocks, n_days]  float32
    open_:         np.ndarray,          # [n_stocks, n_days]
    high:          np.ndarray,          # [n_stocks, n_days]
    low:           np.ndarray,          # [n_stocks, n_days]
    volume:        np.ndarray,          # [n_stocks, n_days]
    params,                             # StrategyParams（duck-typed）
    market_regime: np.ndarray = None,   # [n_days] int8
    valid_mask:    np.ndarray = None,   # [n_stocks, n_days] bool
    extra_factors: dict       = None,   # 实盘注入真实 intraday 因子
    **kwargs,
) -> AlphaSignal:
    """
    弱转强策略统一版。

    Parameters
    ----------
    close, open_, high, low, volume : ndarray [n_stocks, n_days]
        日线 OHLCV，来自 npy memmap（回测）或 LiveDataAdapter（实盘）。
    params : StrategyParams
        从 config.json strategy_params.weak_to_strong 读取。
    market_regime : ndarray [n_days] int8
        市场状态，由 MarketRegimeDetector 输出。
    valid_mask : ndarray [n_stocks, n_days] bool
        可交易标记。
    extra_factors : dict or None
        实盘模式下由 LiveDataAdapter 注入：
            "zhaban_codes"     : set[str]   昨日炸板股代码（覆盖日线近似）
            "auction_score"    : ndarray[n] 真实竞价评分
            "money_flow_score" : ndarray[n] 真实主力资金评分

    Returns
    -------
    AlphaSignal
    """
    if extra_factors is None:
        extra_factors = {}

    n_stocks, n_days = close.shape

    # ── 边界防护 ────────────────────────────────────────────────────────────
    if n_days < 5:
        return AlphaSignal(raw_target_weights=np.zeros((n_stocks, n_days), dtype=np.float64))

    # ── 读取入场阈值参数（只读一次，在循环外）──────────────────────────────
    min_price     = float(getattr(params, "min_price",            3.0))
    max_price     = float(getattr(params, "max_price",           80.0))
    score_thresh  = float(getattr(params, "score_threshold",     50.0))
    vol_ratio_min = float(getattr(params, "vol_ratio_min",        1.3))
    # [FIX-W-03] 单股上限：与引擎 max_single_pos 保持一致，默认 0.15（策略参数）
    max_single_pos = float(getattr(params, "max_single_position",
                                   getattr(params, "max_single_pos", 0.15)))
    # [FIX-B2] weak_to_strong 接入防抖引擎，需要 top_n 参数
    top_n = int(getattr(params, "top_n", 10)) if params else 10

    # ── 实盘 extra_factors（逐日不变，只取一次）────────────────────────────
    zhaban_codes_live: set     = extra_factors.get("zhaban_codes",     set())
    auction_live:      np.ndarray = extra_factors.get("auction_score",    None)
    money_live:        np.ndarray = extra_factors.get("money_flow_score", None)
    codes_list: list             = kwargs.get("codes", [])

    # ── 初始化输出权重矩阵 (N, T) ────────────────────────────────────────────
    raw_weights_full = np.zeros((n_stocks, n_days), dtype=np.float64)
    # [FIX-B2] score_mat 存储每日连续评分，供防抖引擎使用
    score_mat = np.zeros((n_stocks, n_days), dtype=np.float64)

    # ── 逐日计算（t >= 3，使用 t-1, t-2 日数据，无前视）──────────────────────
    # [FIX-BUG4] 全部因子计算代码必须在此 for 循环内部，缩进4格。
    # 原代码将因子计算放在循环体外（函数作用域），属于严重缩进 Bug，
    # 导致仅最后一天有信号，其余回测日权重全零。
    for t_cur in range(3, n_days):
        t0 = t_cur          # 当日（信号计算日）
        t1 = t_cur - 1      # 昨日
        t2 = t_cur - 2      # 前日

        c0 = close[:, t0]
        c1 = close[:, t1]
        c2 = close[:, t2]
        h1 = high[:, t1]
        o0 = open_[:, t0]
        v1 = volume[:, t1]
        v2 = volume[:, t2]

        # ── 市场状态（每日独立查询）──────────────────────────────────────────
        if market_regime is not None:
            regime_str = REGIME_IDX_TO_STR.get(int(market_regime[t0]), "NEUTRAL")
        else:
            regime_str = "NEUTRAL"

        w = FACTOR_WEIGHTS.get(regime_str, FACTOR_WEIGHTS["NEUTRAL"])

        # 熊市 / 软熊市空仓
        # [FIX-W-04] 原条件冗余（BEAR 的 sentiment=0.15<0.2 恒成立），改为直接判断 regime_str
        if regime_str in ("BEAR", "SOFT_BEAR"):
            continue

        # ═══════════════════════════════════════════════════════════════════
        # 因子 1: 炸板识别（优先使用实盘炸板池，回退到日线近似）
        # ═══════════════════════════════════════════════════════════════════
        if zhaban_codes_live:
            # ── 实盘路径 ────────────────────────────────────────────────
            touched_limit = np.array(
                [c in zhaban_codes_live for c in codes_list],
                dtype=bool,
            )
            did_zhaban = touched_limit
        else:
            # ── 回测路径：日线近似 ────────────────────────────────────
            limit_line    = c2 * 1.095
            touched_limit = h1 >= limit_line
            did_zhaban    = c1 < h1 * 0.999

        # 炸板深度评分
        zhaban_depth_raw = np.where(h1 > _EPS, (h1 - c1) / (h1 + _EPS) * 100, 10.0)
        depth_score  = np.clip(100 - (zhaban_depth_raw - 2) * 10, 30, 100)

        # 炸板日涨幅评分
        prev_chg  = np.where(c2 > _EPS, (c1 / (c2 + _EPS) - 1) * 100, 0.0)
        chg_score = np.where(prev_chg >= 9.5, 100,
                    np.where(prev_chg >= 9.0,  85,
                    np.where(prev_chg >= 8.0,  70, 45))).astype(np.float64)

        zhaban_factor = depth_score * 0.4 + chg_score * 0.6

        # ═══════════════════════════════════════════════════════════════════
        # 因子 2: 竞价强度（实盘=真实 spot 数据，回测=open/prev_close 近似）
        # ═══════════════════════════════════════════════════════════════════
        if auction_live is not None:
            auction_factor = np.clip(auction_live, 0, 100).astype(np.float64)
        else:
            open_chg = np.where(c1 > _EPS, (o0 / (c1 + _EPS) - 1) * 100, 0.0)
            auction_factor = np.clip(
                np.where((open_chg >= 1) & (open_chg <= 4),
                         100 - np.abs(open_chg - 2.5) * 10,
                np.where((open_chg >= 0) & (open_chg < 1),
                         85 - open_chg * 10,
                np.where((open_chg > 4) & (open_chg <= 6),
                         70 - (open_chg - 4) * 10,
                np.where((open_chg >= -2) & (open_chg < 0),
                         60 + open_chg * 15,
                         np.maximum(20.0, 40 - np.abs(open_chg - 5) * 5))))),
                20, 100,
            ).astype(np.float64)

        # ═══════════════════════════════════════════════════════════════════
        # 因子 3: 主力资金（实盘=资金流向 API，回测=量价组合近似）
        # ═══════════════════════════════════════════════════════════════════
        if money_live is not None:
            money_factor = np.clip(money_live, 0, 100).astype(np.float64)
        else:
            vol_ratio      = np.where(v2 > _EPS, v1 / (v2 + _EPS), 1.0)
            price_position = np.where(
                (h1 - c2) > _EPS,
                (c1 - c2) / (h1 - c2 + _EPS),
                0.5,
            )
            money_raw    = vol_ratio * 0.5 + price_position * 0.5
            money_factor = np.clip(money_raw * 100, 10, 100).astype(np.float64)

        # ═══════════════════════════════════════════════════════════════════
        # 因子 4: 龙头地位（量能 z-score 近似，两模式相同）
        # ═══════════════════════════════════════════════════════════════════
        v1_mean = float(np.nanmean(v1)) if v1.size > 0 else 1.0
        v1_std  = float(np.nanstd(v1))  if v1.size > 0 else 1.0
        vol_z   = (v1 - v1_mean) / (v1_std + _EPS)
        leader_factor = np.clip(50.0 + vol_z * 10, 20, 100).astype(np.float64)

        # ═══════════════════════════════════════════════════════════════════
        # 因子 5: 板块强度（日线行业涨幅中位数对比，两模式相同）
        # ═══════════════════════════════════════════════════════════════════
        ret_t1        = np.where(c2 > _EPS, (c1 / (c2 + _EPS) - 1) * 100, 0.0)
        median_ret    = float(np.nanmedian(ret_t1))
        sector_factor = np.clip(50.0 + (ret_t1 - median_ret) * 5, 10, 100).astype(np.float64)

        # ═══════════════════════════════════════════════════════════════════
        # 因子 6: 市场情绪（market_regime 映射，两模式相同）
        # ═══════════════════════════════════════════════════════════════════
        sentiment_score  = _SENTIMENT_MAP.get(regime_str, 0.5)
        sentiment_factor = np.full(n_stocks, sentiment_score * 100, dtype=np.float64)

        # ═══════════════════════════════════════════════════════════════════
        # 因子 7: 技术形态（MA 多头排列，两模式相同）
        # ═══════════════════════════════════════════════════════════════════
        if t0 >= 19:
            ma5_t  = np.mean(close[:, t0 - 4 : t0 + 1],  axis=1)
            ma10_t = np.mean(close[:, t0 - 9 : t0 + 1],  axis=1)
            ma20_t = np.mean(close[:, t0 - 19: t0 + 1],  axis=1)
            bull    = (c0 > ma5_t) & (ma5_t > ma10_t) & (ma10_t > ma20_t)
            partial = (c0 > ma5_t) & (ma5_t > ma10_t)
            tech_factor = np.where(bull, 85.0, np.where(partial, 65.0, 45.0)).astype(np.float64)
        else:
            tech_factor = np.full(n_stocks, 50.0, dtype=np.float64)

        # ═══════════════════════════════════════════════════════════════════
        # 综合评分
        # ═══════════════════════════════════════════════════════════════════
        score = (
            zhaban_factor    * w["zhaban"]
            + auction_factor * w["auction"]
            + money_factor   * w["money"]
            + leader_factor  * w["leader"]
            + sector_factor  * w["sector"]
            + sentiment_factor * w["sentiment"]
            + tech_factor    * w["technical"]
        )

        # ═══════════════════════════════════════════════════════════════════
        # 入场过滤
        # ═══════════════════════════════════════════════════════════════════
        vol_ratio_t1 = np.where(v2 > _EPS, v1 / (v2 + _EPS), 1.0)

        entry_mask = (
            touched_limit
            & did_zhaban
            & (score >= score_thresh)
            & (c0 >= min_price)
            & (c0 <= max_price)
            & (vol_ratio_t1 >= vol_ratio_min)
        )
        if valid_mask is not None:
            entry_mask = entry_mask & valid_mask[:, t0]

        # ═══════════════════════════════════════════════════════════════════
        # [FIX-B2] 不再直接写 raw_weights_full，改为写 score_mat（连续信号）
        # 原 softmax 直写导致：昨天买入的股票今天若不满足脉冲条件即全部清仓，
        # 无任何防抖，年换手率高达 12000%+，交易成本吃光所有本金。
        # 修复：将满足 entry_mask 的 softmax 强度存入 score_mat，
        # 不满足条件的股票score保持0（非-inf），让防抖状态机决定持仓延续。
        # ═══════════════════════════════════════════════════════════════════
        if entry_mask.any():
            sel   = np.where(entry_mask, score, -1e9)
            exp_s = np.exp((sel - sel.max()) / 10.0)
            exp_s = np.where(entry_mask, exp_s, 0.0)
            total = exp_s.sum()
            if total > _EPS:
                score_mat[:, t_cur] = (exp_s / total).astype(np.float64)
        # 若 entry_mask 全为 False，score_mat[:, t_cur] 保持 0
        # _score_to_weights 会把0分股票排在最末尾，防抖保护已持仓股票

    # [L4-FIX] 涨跌停硬封锁：T日已封死涨/跌停点位在T+1无法成交
    limit_mask = kwargs.get("limit_reach_mask")
    if limit_mask is not None:
        score_mat[limit_mask] = -np.inf

    # [L4-FIX] 数值稳态保护
    score_mat = np.nan_to_num(score_mat, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)

    # [FIX-B2] 用标准防抖权重引擎替代 softmax 直写
    # entry_mask 的效果：新仓只有在评分>0（即满足今日买点）时才能入场；
    # 旧仓只要排名未跌出 top_n+exit_buffer 就由防抖保护，无需今天有买点。
    dropout_d = 1   # ★[FIX-DEBOUNCE] 策略专属，来自 exit_config 规格
    exit_buf  = 2   # ★[FIX-DEBOUNCE] 策略专属，来自 exit_config 规格
    # score_mat>0 的列即为今日有信号，作为新仓入场门控
    buy_mask_mat = score_mat > 0
    raw_weights_full = _score_to_weights(
        score_mat,
        top_n          = top_n,
        max_single_pos = max_single_pos,
        dropout_days   = dropout_d,
        exit_buffer    = exit_buf,
        hard_invalid   = (None if valid_mask is None
                         else ~np.asarray(valid_mask, dtype=bool)),
    )

    # [FIX-W-01] 返回 (N, T) 矩阵，与 fast_runner_v10 期望格式一致
    return AlphaSignal(
        raw_target_weights=raw_weights_full,
        exit_config={
            "stop_mode"       : "entry_price",  # 炸板事件二元结果
            "hard_stop_loss"  : 0.07,
            "take_profit"     : 0.12,
            "max_holding_days": 5,
            "dropout_days"    : 1,
            "exit_buffer"     : 2,
        },
    )
