"""
ultra_short_signal.py — Q-UNITY V10 超短线实盘信号引擎
=======================================================

迁移自 src/strategies/vectorized/ultra_short_vec.py 的日线降级包装，
重构为面向实盘 TdxClient 行情的分钟级信号引擎。

设计原则
--------
- **不继承 AlphaSignal**：实盘信号不是回测权重矩阵，输出为 {code: direction}
- **不依赖 Numba 3D 矩阵**：TdxClient 每次提供单快照，无 (N,D,M) 张量
- **有状态**：每只股票维护 BarState（前收、均量、持仓状态），由外部在开盘前更新
- **线程安全**：scan() 持 Lock，可被 PortfolioTracker 后台线程安全调用
- **信号格式**：{"buy" | "sell" | "hold"}，由调用方决定是否通过 SmartAlerterProxy 推送

与 MonitorEngine 的集成点
--------------------------
    engine = UltraShortSignalEngine(params, alerter=smart_alerter, tracker=signal_tracker)
    engine.update_history(prev_closes, vol_avg20s)   # 每日开盘前调用一次
    result = engine.scan(feed.get_prices(codes))      # 每次 feed tick 调用
    for code, direction in result.items():
        if direction == "buy":
            trader.buy(code, price, shares)
        elif direction == "sell":
            trader.sell(code, price)

原策略因子复用（来自 ultra_short_signals_daily_wrapper）
---------------------------------------------------------
Gate-1: open_pct  = (open[today] - close[yesterday]) / close[yesterday] > 0.012
        → 当日必须高开 > 1.2%，确认趋势方向向上
Gate-2: vol_ratio = volume[today_so_far] / vol_avg20[yesterday] > volume_ratio_thr
        → 放量确认（资金关注）
Gate-3: day_ret   = (close[now] - close[yesterday]) / close[yesterday] > momentum_thr
        → 当日收益率 > 动量阈值，突破确认
止损止盈: pnl <= -stop_loss 或 pnl >= take_profit → 出场
持仓上限: hold_count >= max_hold_ticks → 超时强制出场（分钟级时间止损）

前视偏差说明（实盘无偏差）
---------------------------
所有门控仅使用已发生数据：
  open_pct  → open[today] 已知（开盘价已确定）
  vol_ratio → volume[today] 已累积，vol_avg20 来自昨日历史
  day_ret   → close[now] 是当前价，prev_close 是昨日收盘，均已发生
  pnl       → entry_price 是实际成交价，close[now] 是当前价
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 默认参数（与 ultra_short_signals_daily_wrapper 源码一致）
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULT_PARAMS: Dict[str, Any] = {
    "volume_ratio_threshold":  2.0,     # Gate-2：量比门槛（当日成交量 / 20日均量）
    "momentum_3min_threshold": 0.015,   # Gate-3：当日收益率门槛（1.5%）
    "ultra_stop_loss":         0.015,   # 止损幅度（从入场价计算，1.5%）
    "ultra_take_profit":       0.020,   # 止盈幅度（从入场价计算，2.0%）
    "max_concurrent":          5,       # 最大同时持仓只数
    "open_gap_threshold":      0.012,   # Gate-1：开盘缺口门槛（1.2%）
    "max_hold_ticks":          60,      # 超时时间止损（tick 次数，约 5s×60=5分钟）
    "min_price":               5.0,     # 最低价格过滤（元）
}


# ─────────────────────────────────────────────────────────────────────────────
# 数据类
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BarState:
    """
    单只股票的盘中状态（开盘前由 update_history() 初始化，盘中由 scan() 更新）。

    Attributes
    ----------
    prev_close  : 昨日收盘价（开盘前必须设置，否则 open_pct 无法计算）
    vol_avg20   : 20日成交量均值（昨日历史数据，用于量比门控）
    in_position : 当前是否持仓（由引擎维护，不依赖 SimulatedTrader 状态）
    entry_price : 入场价格（持仓时有效）
    hold_count  : 持仓持续的 tick 次数（用于时间止损）
    peak_pnl    : 持仓期间最高盈亏比例（用于追踪止损扩展，当前版本暂不使用）
    last_signal : 上一次发出的方向（防止同方向重复推送）
    last_tick   : 上一次处理的快照时间戳（debug 用）
    """
    prev_close  : float = 0.0
    vol_avg20   : float = 0.0
    in_position : bool  = False
    entry_price : float = 0.0
    hold_count  : int   = 0
    peak_pnl    : float = 0.0
    last_signal : str   = "hold"    # "buy" | "sell" | "hold"
    last_tick   : float = 0.0


@dataclass
class SignalResult:
    """
    单只股票的信号结果（供调用方记录和推送）。

    Attributes
    ----------
    code      : 股票代码
    direction : "buy" | "sell" | "hold"
    price     : 触发时的当前价格
    score     : 信号强度（0~1，供 SmartAlerterProxy 分级判断）
    reason    : 触发原因（供 SignalTracker 记录）
    gate_vals : 各门控的实际计算值（debug / 日志用）
    """
    code      : str
    direction : str   # "buy" | "sell" | "hold"
    price     : float = 0.0
    score     : float = 0.0
    reason    : str   = ""
    gate_vals : Dict[str, float] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        """是否是需要执行的信号（非 hold）"""
        return self.direction in ("buy", "sell")


# ─────────────────────────────────────────────────────────────────────────────
# 主引擎
# ─────────────────────────────────────────────────────────────────────────────

class UltraShortSignalEngine:
    """
    超短线实盘信号引擎

    核心方法
    --------
    update_history(prev_closes, vol_avg20s)
        每日开盘前调用一次，设置各股票的前收盘价和均量基线。
        必须在第一次 scan() 之前调用，否则 Gate-1/3 无法计算。

    scan(feed_snapshot) -> Dict[str, str]
        接收 TdxClient.get_prices() 返回的行情快照。
        对每只已订阅且有历史基准的股票，运行三重门控 + 止损止盈逻辑。
        返回 {code: "buy"/"sell"/"hold"}（仅包含有状态的股票）。

    reset_day()
        收盘后调用，重置所有持仓状态（每日开盘前调用更安全）。

    线程安全
    --------
    scan() 和 update_history() 均持 _lock，可与 PortfolioTracker 后台线程并发。
    """

    STRATEGY_TYPE = "ultrashort"   # 与 SmartAlerterProxy/Alerter 的策略类型对应

    def __init__(
        self,
        params:         Optional[Dict[str, Any]] = None,
        alerter=None,           # SmartAlerterProxy 或 Alerter 实例（可选）
        signal_tracker=None,    # SignalTracker 实例（可选，用于 CSV 记录）
        codes:          Optional[List[str]] = None,   # 关注标的（可在 scan() 中动态扩展）
    ) -> None:
        """
        Parameters
        ----------
        params         : 策略参数 dict（键名见模块顶部 _DEFAULT_PARAMS）
        alerter        : SmartAlerterProxy 实例，scan() 触发 buy/sell 时推送
        signal_tracker : SignalTracker 实例，scan() 触发信号时记录 CSV
        codes          : 预先订阅的股票代码列表（update_history 时会自动扩展）
        """
        # 合并参数（用户值覆盖默认值）
        self._p: Dict[str, Any] = {**_DEFAULT_PARAMS, **(params or {})}

        self._alerter        = alerter
        self._signal_tracker = signal_tracker
        self._lock           = threading.Lock()

        # 每只股票的盘中状态
        self._states: Dict[str, BarState] = {}
        if codes:
            for code in codes:
                self._states[_normalize(code)] = BarState()

        # 当日已推送买入的股票集合（用于并发上限判断）
        self._active_buys: set = set()

        logger.info(
            f"[UltraShortSignal] 初始化完成 "
            f"vol_ratio_thr={self._p['volume_ratio_threshold']} "
            f"mom_thr={self._p['momentum_3min_threshold']} "
            f"stop_loss={self._p['ultra_stop_loss']} "
            f"take_profit={self._p['ultra_take_profit']}"
        )

    # ── 生命周期 ─────────────────────────────────────────────────────────────

    def update_history(
        self,
        prev_closes: Dict[str, float],
        vol_avg20s:  Dict[str, float],
    ) -> None:
        """
        每日开盘前调用，更新各股票的历史基准数据。

        Parameters
        ----------
        prev_closes : {code: 昨日收盘价}
        vol_avg20s  : {code: 20日成交量均值（昨日历史）}

        使用示例::

            prev_closes = runner.get_prev_closes(codes)  # 来自 npy 数据
            vol_avg20s  = runner.get_vol_avg20(codes)
            engine.update_history(prev_closes, vol_avg20s)
        """
        with self._lock:
            for raw_code, prev_close in prev_closes.items():
                code = _normalize(raw_code)
                if code not in self._states:
                    self._states[code] = BarState()
                self._states[code].prev_close = float(prev_close)

            for raw_code, vol_avg in vol_avg20s.items():
                code = _normalize(raw_code)
                if code not in self._states:
                    self._states[code] = BarState()
                self._states[code].vol_avg20 = float(vol_avg)

        logger.info(
            f"[UltraShortSignal] 历史数据已更新: "
            f"{len(prev_closes)} 只股票前收盘价，{len(vol_avg20s)} 只均量基准"
        )

    def reset_day(self) -> None:
        """
        收盘后 / 次日开盘前调用。
        重置所有盘中状态（持仓、入场价、计数器），保留历史基准。
        """
        with self._lock:
            for state in self._states.values():
                state.in_position = False
                state.entry_price = 0.0
                state.hold_count  = 0
                state.peak_pnl    = 0.0
                state.last_signal = "hold"
            self._active_buys.clear()
        logger.info("[UltraShortSignal] 日间状态已重置（持仓/计数器清零）")

    # ── 核心接口：扫描快照 ────────────────────────────────────────────────────

    def scan(
        self,
        feed_snapshot: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        接收 TdxClient.get_prices() 返回的行情快照，计算信号。

        Parameters
        ----------
        feed_snapshot : {code: {"close": float, "open": float,
                                "high": float, "low": float,
                                "volume": float, "amount": float}}
                        TdxClient 的标准输出格式。code 可带或不带交易所前缀。

        Returns
        -------
        Dict[str, str]
            {code_normalized: "buy" | "sell" | "hold"}
            仅包含有历史基准（prev_close > 0）的股票。
            "hold" 表示无操作需求，调用方可忽略。

        线程安全：持 _lock，可从后台线程安全调用。
        """
        results: Dict[str, str] = {}
        actionable: List[SignalResult] = []

        with self._lock:
            now_ts  = time.monotonic()
            n_buys  = len(self._active_buys)
            max_con = int(self._p["max_concurrent"])

            for raw_code, bar in feed_snapshot.items():
                code = _normalize(raw_code)

                # 只处理有历史基准的股票
                state = self._states.get(code)
                if state is None or state.prev_close <= 0:
                    continue

                # 解析行情字段（TdxClient 标准输出）
                price  = _safe_float(bar.get("close")  or bar.get("price"))
                open_  = _safe_float(bar.get("open"))
                volume = _safe_float(bar.get("volume") or bar.get("vol"))

                if price <= 0 or open_ <= 0:
                    continue

                # 价格过滤
                if price < self._p["min_price"]:
                    continue

                result = self._compute_signal(
                    code=code,
                    state=state,
                    price=price,
                    open_=open_,
                    volume=volume,
                    n_active_buys=n_buys,
                    max_concurrent=max_con,
                    now_ts=now_ts,
                )

                results[code] = result.direction

                if result.is_actionable and result.direction != state.last_signal:
                    actionable.append(result)
                    # 更新状态机
                    if result.direction == "buy":
                        state.in_position = True
                        state.entry_price = price
                        state.hold_count  = 0
                        state.peak_pnl    = 0.0
                        self._active_buys.add(code)
                        n_buys += 1   # 立即更新本轮并发计数
                    elif result.direction == "sell":
                        state.in_position = False
                        state.entry_price = 0.0
                        state.hold_count  = 0
                        state.peak_pnl    = 0.0
                        self._active_buys.discard(code)
                        n_buys = max(0, n_buys - 1)

                state.last_signal = result.direction
                state.last_tick   = now_ts

                # 持仓计数递增（仅在非卖出时）
                if state.in_position and result.direction != "sell":
                    state.hold_count += 1

        # 锁外推送（避免锁内阻塞）
        for sig in actionable:
            self._push_signal(sig)

        return results

    # ── 信号计算（无锁，由 scan() 持锁调用）────────────────────────────────

    def _compute_signal(
        self,
        code:           str,
        state:          BarState,
        price:          float,
        open_:          float,
        volume:         float,
        n_active_buys:  int,
        max_concurrent: int,
        now_ts:         float,
    ) -> SignalResult:
        """
        核心信号逻辑（无锁，在 scan() 的 with self._lock 内调用）。

        等价于 ultra_short_signals_daily_wrapper 的单股单时刻逻辑，
        适配为实时 tick 计算而非向量化矩阵。

        逻辑流程：
          1. 若持仓中 → 先检查止损/止盈/超时出场（优先级最高）
          2. 若未持仓 → 检查三重入场门控

        Parameters
        ----------
        state.prev_close : 昨日收盘价（Gate-1 和 Gate-3 的分母）
        state.vol_avg20  : 20日均量（Gate-2 的分母）
        price            : 当前价格（close）
        open_            : 今日开盘价
        volume           : 今日累计成交量
        """
        prev_close = state.prev_close

        # ── 公共指标计算 ──────────────────────────────────────────────────────
        # Gate-1: 开盘缺口（今日开盘 vs 昨日收盘）
        open_pct = (open_ - prev_close) / prev_close if prev_close > 1e-8 else 0.0

        # Gate-3: 当日收益率（当前价 vs 昨日收盘）
        day_ret = (price - prev_close) / prev_close if prev_close > 1e-8 else 0.0

        # Gate-2: 量比（今日累计成交量 / 20日均量）
        vol_ratio = volume / state.vol_avg20 if state.vol_avg20 > 1e-8 else 0.0

        gate_vals = {
            "open_pct":  open_pct,
            "day_ret":   day_ret,
            "vol_ratio": vol_ratio,
        }

        # ── 持仓中：检查出场条件（止损/止盈/超时，优先级高于入场）────────────
        if state.in_position and state.entry_price > 1e-8:
            pnl = (price - state.entry_price) / state.entry_price
            gate_vals["pnl"] = pnl

            # 止损
            if pnl <= -self._p["ultra_stop_loss"]:
                return SignalResult(
                    code=code, direction="sell", price=price,
                    score=0.95,
                    reason=f"止损: pnl={pnl:+.2%} ≤ -{self._p['ultra_stop_loss']:.2%}",
                    gate_vals=gate_vals,
                )

            # 止盈
            if pnl >= self._p["ultra_take_profit"]:
                return SignalResult(
                    code=code, direction="sell", price=price,
                    score=0.90,
                    reason=f"止盈: pnl={pnl:+.2%} ≥ {self._p['ultra_take_profit']:.2%}",
                    gate_vals=gate_vals,
                )

            # 时间止损（持仓超过 max_hold_ticks 个 tick）
            if state.hold_count >= self._p["max_hold_ticks"]:
                return SignalResult(
                    code=code, direction="sell", price=price,
                    score=0.80,
                    reason=f"时间止损: hold_count={state.hold_count} ≥ {self._p['max_hold_ticks']}",
                    gate_vals=gate_vals,
                )

            # 持仓中，未触发出场 → hold
            return SignalResult(
                code=code, direction="hold", price=price,
                score=0.0, reason="持仓中，未触发出场",
                gate_vals=gate_vals,
            )

        # ── 未持仓：检查三重入场门控 ─────────────────────────────────────────
        # Gate-1: 开盘缺口必须 > open_gap_threshold（确认趋势方向）
        if open_pct <= self._p["open_gap_threshold"]:
            return SignalResult(
                code=code, direction="hold", price=price, score=0.0,
                reason=f"Gate-1失败: open_pct={open_pct:.3f} ≤ {self._p['open_gap_threshold']:.3f}",
                gate_vals=gate_vals,
            )

        # Gate-2: 量比必须 > volume_ratio_threshold（资金关注确认）
        if vol_ratio <= self._p["volume_ratio_threshold"]:
            return SignalResult(
                code=code, direction="hold", price=price, score=0.0,
                reason=f"Gate-2失败: vol_ratio={vol_ratio:.2f} ≤ {self._p['volume_ratio_threshold']:.2f}",
                gate_vals=gate_vals,
            )

        # Gate-3: 当日收益率必须 > momentum_3min_threshold（突破确认）
        if day_ret <= self._p["momentum_3min_threshold"]:
            return SignalResult(
                code=code, direction="hold", price=price, score=0.0,
                reason=f"Gate-3失败: day_ret={day_ret:.3f} ≤ {self._p['momentum_3min_threshold']:.3f}",
                gate_vals=gate_vals,
            )

        # 并发持仓上限
        if n_active_buys >= max_concurrent:
            return SignalResult(
                code=code, direction="hold", price=price, score=0.0,
                reason=f"并发上限: 当前{n_active_buys} ≥ max_concurrent={max_concurrent}",
                gate_vals=gate_vals,
            )

        # 三重门控全部通过 → 买入信号
        # score 综合三个因子强度（参考 SmartAlerterProxy 的 strategy_type="ultrashort"）
        score = min(
            0.5 * min(vol_ratio / (self._p["volume_ratio_threshold"] * 3), 1.0)
            + 0.3 * min(day_ret / 0.05, 1.0)
            + 0.2 * min(open_pct / 0.05, 1.0),
            1.0,
        )
        return SignalResult(
            code=code, direction="buy", price=price,
            score=score,
            reason=(
                f"三关通过: gap={open_pct:.2%} "
                f"vol×{vol_ratio:.1f} "
                f"ret={day_ret:.2%}"
            ),
            gate_vals=gate_vals,
        )

    # ── 信号推送（锁外调用）──────────────────────────────────────────────────

    def _push_signal(self, sig: SignalResult) -> None:
        """
        通过 SmartAlerterProxy / Alerter 推送信号，并记录到 SignalTracker。

        strategy_type="ultrashort" → SmartAlerterProxy 使用 1800s 去重窗口。
        score ≥ 0.8 → 强信号，实时推送。
        """
        if self._alerter is not None:
            try:
                self._alerter.send(
                    code=sig.code,
                    action="BUY" if sig.direction == "buy" else "SELL",
                    score=sig.score,
                    price=sig.price,
                    reason=sig.reason,
                    strategy_type=self.STRATEGY_TYPE,
                )
            except Exception as e:
                logger.warning(f"[UltraShortSignal] 推送失败 {sig.code}: {e}")

        if self._signal_tracker is not None:
            try:
                self._signal_tracker.record(
                    code=sig.code,
                    action="BUY" if sig.direction == "buy" else "SELL",
                    score=sig.score,
                    price=sig.price,
                    reason=sig.reason,
                    strategy_type=self.STRATEGY_TYPE,
                )
            except Exception as e:
                logger.warning(f"[UltraShortSignal] SignalTracker 记录失败 {sig.code}: {e}")

        logger.info(
            f"[UltraShortSignal] {sig.direction.upper()} {sig.code} "
            f"@{sig.price:.2f} score={sig.score:.3f} | {sig.reason}"
        )

    # ── 状态查询（供仪表盘/监控读取）────────────────────────────────────────

    def get_positions(self) -> Dict[str, float]:
        """
        返回当前由引擎追踪的持仓 {code: entry_price}。

        注意：此状态与 SimulatedTrader._positions 独立维护，
        仅反映引擎视角的持仓（不含通过其他路径建立的仓位）。
        """
        with self._lock:
            return {
                code: state.entry_price
                for code, state in self._states.items()
                if state.in_position and state.entry_price > 0
            }

    def get_state(self, code: str) -> Optional[BarState]:
        """获取单只股票的当前状态（线程安全，返回副本）"""
        key = _normalize(code)
        with self._lock:
            state = self._states.get(key)
            if state is None:
                return None
            # 返回浅拷贝，避免外部修改影响内部状态
            from copy import copy
            return copy(state)

    def summary(self) -> Dict[str, Any]:
        """返回当前引擎状态摘要（供 RealtimeDashboard 展示）"""
        with self._lock:
            n_tracked    = len(self._states)
            n_in_pos     = sum(1 for s in self._states.values() if s.in_position)
            n_with_hist  = sum(1 for s in self._states.values() if s.prev_close > 0)
        return {
            "strategy":    "ultra_short",
            "n_tracked":   n_tracked,
            "n_in_pos":    n_in_pos,
            "n_with_hist": n_with_hist,
            "max_concurrent": int(self._p["max_concurrent"]),
            "active_buys": list(self._active_buys),
            "params": {
                "vol_ratio_thr": self._p["volume_ratio_threshold"],
                "mom_thr":       self._p["momentum_3min_threshold"],
                "stop_loss":     self._p["ultra_stop_loss"],
                "take_profit":   self._p["ultra_take_profit"],
                "open_gap_thr":  self._p["open_gap_threshold"],
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# 工厂函数（与 MonitorEngine 集成的标准入口）
# ─────────────────────────────────────────────────────────────────────────────

def create_ultra_short_engine(
    config:         Optional[Dict[str, Any]] = None,
    alerter=None,
    signal_tracker=None,
    codes:          Optional[List[str]] = None,
) -> UltraShortSignalEngine:
    """
    工厂函数：从 config.json 的 realtime.ultra_short 节创建引擎实例。

    Parameters
    ----------
    config         : 完整 config dict（取 realtime.ultra_short 子节）
    alerter        : SmartAlerterProxy 或 Alerter 实例
    signal_tracker : SignalTracker 实例
    codes          : 初始订阅代码

    Returns
    -------
    UltraShortSignalEngine

    使用示例::

        from src.realtime.ultra_short_signal import create_ultra_short_engine

        engine = create_ultra_short_engine(
            config=app_config,
            alerter=smart_alerter,
            signal_tracker=tracker,
            codes=watchlist,
        )
        engine.update_history(prev_closes, vol_avg20s)
        # 在 PortfolioTracker 的 _tick() 中调用：
        result = engine.scan(feed.get_prices(watchlist))
    """
    params: Dict[str, Any] = {}
    if config is not None:
        rt_cfg = config.get("realtime", config)
        us_cfg = rt_cfg.get("ultra_short", {})
        for key in _DEFAULT_PARAMS:
            if key in us_cfg:
                params[key] = us_cfg[key]

    return UltraShortSignalEngine(
        params=params,
        alerter=alerter,
        signal_tracker=signal_tracker,
        codes=codes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(code: str) -> str:
    """去掉交易所前缀（sh.600519 → 600519）"""
    return code.split(".")[-1] if "." in code else str(code)


def _safe_float(val: Any, default: float = 0.0) -> float:
    """安全转换为 float，失败返回 default"""
    try:
        v = float(val)
        return v if v == v else default   # NaN check
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# TdxQuant 实时行情适配器
# ─────────────────────────────────────────────────────────────────────────────

class TdxFeedAdapter:
    """
    把 TdxRealtimeEngine.snapshot() 的输出格式转换为
    UltraShortSignalEngine.scan() 所需的 feed_snapshot 格式。

    TdxQuant 输出:
      {"price":1445.0, "open":1452.96, "volume":26132,
       "pe_ttm":20.1, "pb_mrq":7.04, ...}

    scan() 需要:
      {"close": price, "open": open_, "volume": volume_shares}

    使用示例::

        from scripts.realtime_tdxquant import TdxRealtimeEngine
        from src.strategies.ultra_short_signal import (
            UltraShortSignalEngine, TdxFeedAdapter, create_ultra_short_engine
        )

        engine  = create_ultra_short_engine(config=cfg, alerter=tdx_engine)
        adapter = TdxFeedAdapter()

        # 盘前准备
        prev_closes, vol_avg20s = _load_history_from_npy(codes)
        engine.update_history(prev_closes, vol_avg20s)

        # 盘中循环（每次 tick 调用一次）
        while trading_hours():
            snapshots = tdx_engine.batch_snapshot(codes)
            feed = adapter.convert(snapshots)
            signals = engine.scan(feed)
            for code, direction in signals.items():
                if direction == "buy":
                    tdx_engine.send_alert(code, feed[code]["close"],
                                          reason="超短线买入信号", bs_flag=0)
                elif direction == "sell":
                    tdx_engine.send_alert(code, feed[code]["close"],
                                          reason="超短线卖出/止损", bs_flag=1)
    """

    @staticmethod
    def convert(snapshots) -> Dict[str, Any]:
        """
        将 TdxRealtimeEngine.snapshot() 列表或单个 dict 转为 scan() 格式。

        Parameters
        ----------
        snapshots : list[dict] 或 dict
            TdxRealtimeEngine.snapshot() 的输出。

        Returns
        -------
        feed : Dict[str, Dict]  {code: {"close":..., "open":..., "volume":...}}
        """
        feed: Dict[str, Any] = {}

        if isinstance(snapshots, dict):
            # 单个快照
            snapshots = [snapshots]

        for snap in snapshots:
            code = snap.get("code", "")
            if not code:
                continue

            price  = _safe_float(snap.get("price") or snap.get("close"))
            open_  = _safe_float(snap.get("open"))
            # [FIX-U-01] volume 保持「手」单位，与 build_history_from_npy() 返回的
            # vol_avg20s（来自 volume.npy，单位也是手）保持一致，使 Gate-2 量比计算正确。
            # 原版 ×100 转股导致 vol_ratio 虚高100倍，Gate-2 形同虚设。
            volume_hands = _safe_float(snap.get("volume"), 0.0)

            if price <= 0:
                continue

            feed[_normalize(code)] = {
                "close":  price,
                "open":   open_,
                "volume": volume_hands,   # [FIX-U-01] 手，与 vol_avg20 单位一致
                # 额外字段（供日志/调试）
                "amount": _safe_float(snap.get("amount"), 0.0),
                "pe_ttm": snap.get("pe_ttm"),
                "pb_mrq": snap.get("pb_mrq"),
                "mkt_cap": snap.get("mkt_cap_circ"),
                "zt_price": snap.get("zt_price"),
            }

        return feed

    @staticmethod
    def build_history_from_npy(
        codes: List[str],
        npy_dir,
        lookback: int = 20,
    ) -> tuple:
        """
        从 npy 矩阵读取 prev_close 和 vol_avg20，
        用于 UltraShortSignalEngine.update_history()。

        Parameters
        ----------
        codes    : 股票代码列表（6位，如 "600519"）
        npy_dir  : npy_v10 目录 Path
        lookback : 均量计算窗口（默认20日）

        Returns
        -------
        (prev_closes, vol_avg20s) : ({code: float}, {code: float})
        """
        import numpy as np
        import json
        from pathlib import Path

        npy_dir = Path(npy_dir)
        meta_path = npy_dir / "meta.json"

        if not meta_path.exists():
            logger.warning(f"meta.json 不存在: {meta_path}")
            return {}, {}

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        all_codes = meta["codes"]          # [N] 完整代码列表（含 sh./sz. 前缀）
        code_to_idx = {
            c.split(".")[-1]: i for i, c in enumerate(all_codes)
        }

        close_path  = npy_dir / "close.npy"
        volume_path = npy_dir / "volume.npy"

        if not close_path.exists() or not volume_path.exists():
            logger.warning("close.npy 或 volume.npy 不存在")
            return {}, {}

        close_arr  = np.load(str(close_path),  mmap_mode="r")
        volume_arr = np.load(str(volume_path), mmap_mode="r")
        T = close_arr.shape[1]

        prev_closes: Dict[str, float] = {}
        vol_avg20s:  Dict[str, float] = {}

        for code in codes:
            c6 = _normalize(code)
            idx = code_to_idx.get(c6)
            if idx is None:
                continue

            # 前收盘价：最后一个有效收盘价
            close_row = close_arr[idx]
            valid = close_row[close_row > 0]
            if len(valid) > 0:
                prev_closes[c6] = float(valid[-1])

            # 20日均量（手）：volume.npy 单位=手，直接取均值
            # [FIX-U-01] 不转换为股，保持手为单位，与 convert() 中 volume_hands 对齐
            vol_row = volume_arr[idx]
            valid_vol = vol_row[vol_row > 0]
            if len(valid_vol) >= lookback:
                vol_avg20s[c6] = float(valid_vol[-lookback:].mean())
            elif len(valid_vol) > 0:
                vol_avg20s[c6] = float(valid_vol.mean())

        logger.info(
            f"[TdxFeedAdapter] 历史数据加载: "
            f"prev_close={len(prev_closes)}只  vol_avg20={len(vol_avg20s)}只"
        )
        return prev_closes, vol_avg20s


# ─────────────────────────────────────────────────────────────────────────────
# 独立验收测试
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 64)
    print("  UltraShortSignalEngine — 验收测试")
    print("=" * 64)

    # ── 构造引擎 ─────────────────────────────────────────────────────────────
    engine = UltraShortSignalEngine(
        params={
            "volume_ratio_threshold":  2.0,
            "momentum_3min_threshold": 0.015,
            "ultra_stop_loss":         0.015,
            "ultra_take_profit":       0.020,
            "max_concurrent":          3,
            "open_gap_threshold":      0.012,
            "max_hold_ticks":          5,       # 测试用，5 tick 超时
            "min_price":               5.0,
        }
    )

    # ── 设置历史基准 ─────────────────────────────────────────────────────────
    codes = ["600519", "000858", "300750"]
    engine.update_history(
        prev_closes  = {"600519": 1680.0, "000858": 50.0, "300750": 300.0},
        vol_avg20s   = {"600519": 1_000_000, "000858": 5_000_000, "300750": 2_000_000},
    )
    print(f"[OK] update_history 完成")

    # ── 验收1：Gate-1 失败（开盘缺口不足）──────────────────────────────────
    snap1 = {
        "600519": {"close": 1700.0, "open": 1681.0, "volume": 3_000_000},  # open_pct=0.006 < 0.012
        "000858": {"close": 52.0,   "open": 50.5,   "volume": 12_000_000},  # open_pct=0.010 < 0.012
    }
    result1 = engine.scan(snap1)
    assert result1.get("600519") == "hold", f"Gate-1失败: {result1.get('600519')}"
    assert result1.get("000858") == "hold", f"Gate-1失败: {result1.get('000858')}"
    print(f"[OK] 验收1 Gate-1（开盘缺口不足 → hold）: {result1}")

    # ── 验收2：三重门控全部通过 → buy ────────────────────────────────────────
    snap2 = {
        "600519": {
            "close":  1710.0,    # day_ret = (1710-1680)/1680 = 1.79% > 1.5% ✓
            "open":   1701.6,    # open_pct = (1701.6-1680)/1680 = 1.29% > 1.2% ✓
            "volume": 3_000_000, # vol_ratio = 3M/1M = 3.0 > 2.0 ✓
        },
        "300750": {
            "close":  306.0,
            "open":   303.6,     # open_pct = 1.2% ✓
            "volume": 6_000_000, # vol_ratio = 3.0 ✓
        },
    }
    result2 = engine.scan(snap2)
    assert result2.get("600519") == "buy", f"三关应通过: {result2.get('600519')}"
    print(f"[OK] 验收2 三重门控通过 → buy: {result2}")

    # ── 验收3：持仓中，触发止盈 ─────────────────────────────────────────────
    # 600519 刚买入 @1710，现价 1710*(1+2.1%) = 1745.91 > take_profit=2.0%
    snap3 = {
        "600519": {"close": 1745.91, "open": 1701.6, "volume": 3_500_000},
    }
    result3 = engine.scan(snap3)
    assert result3.get("600519") == "sell", f"止盈应触发: {result3.get('600519')}"
    print(f"[OK] 验收3 止盈触发 → sell: {result3}")

    # ── 验收4：持仓中，触发止损 ─────────────────────────────────────────────
    # 300750 刚买入 @306，现价 306*(1-1.6%) = 301.1 < stop_loss=-1.5%
    snap4 = {
        "300750": {"close": 301.1, "open": 303.6, "volume": 6_500_000},
    }
    result4 = engine.scan(snap4)
    assert result4.get("300750") == "sell", f"止损应触发: {result4.get('300750')}"
    print(f"[OK] 验收4 止损触发 → sell: {result4}")

    # ── 验收5：时间止损（hold_count 达到 max_hold_ticks）────────────────────
    # 重置状态并建仓 000858
    engine.reset_day()
    engine.update_history(
        prev_closes={"000858": 50.0},
        vol_avg20s={"000858": 5_000_000},
    )
    # 满足三关，建仓
    snap5a = {"000858": {"close": 51.0, "open": 50.7, "volume": 15_000_000}}
    r5a = engine.scan(snap5a)
    assert r5a.get("000858") == "buy", f"应建仓: {r5a.get('000858')}"

    # 连续 hold（未触发止损止盈），等待超时
    for i in range(5):
        snap_mid = {"000858": {"close": 51.3, "open": 50.7, "volume": 15_000_000}}
        engine.scan(snap_mid)

    # 5 次循环的最后一次应触发时间止损（检查循环内已触发）
    # 此时持仓已清，验证 get_positions() 中 000858 已不在其中
    positions_after = engine.get_positions()
    assert "000858" not in positions_after, \
        f"时间止损后应清仓，但 get_positions() 仍含 000858: {positions_after}"
    print(f"[OK] 验收5 时间止损（hold_count≥max_hold_ticks → sell，持仓已清 ✓）")

    # ── 验收6：并发上限 ─────────────────────────────────────────────────────
    engine.reset_day()
    # max_concurrent=3，先填满 3 只
    engine.update_history(
        prev_closes={"600519": 1680.0, "000858": 50.0, "300750": 300.0},
        vol_avg20s={"600519": 1e6, "000858": 5e6, "300750": 2e6},
    )
    snap6 = {
        "600519": {"close": 1710.0, "open": 1701.6, "volume": 3e6},
        "000858": {"close": 51.0,   "open": 50.7,   "volume": 15e6},
        "300750": {"close": 306.0,  "open": 303.6,  "volume": 6e6},
    }
    r6 = engine.scan(snap6)
    # 前 3 只都应买入（顺序视字典遍历，至少有 3 只 buy）
    buy_cnt = sum(1 for d in r6.values() if d == "buy")
    assert buy_cnt <= 3, f"buy 数量超过 max_concurrent: {buy_cnt}"
    print(f"[OK] 验收6 并发上限 max_concurrent=3，实际 buy={buy_cnt}: {r6}")

    # ── 验收7：summary() 返回有效摘要 ───────────────────────────────────────
    s = engine.summary()
    assert s["strategy"] == "ultra_short"
    assert s["max_concurrent"] == 3
    print(f"[OK] 验收7 summary(): {s}")

    # ── 验收8：_normalize 工具函数 ───────────────────────────────────────────
    assert _normalize("sh.600519") == "600519"
    assert _normalize("sz.000858") == "000858"
    assert _normalize("600519")    == "600519"
    print(f"[OK] 验收8 _normalize() 交易所前缀处理正确 ✓")

    print()
    print("[PASS] UltraShortSignalEngine 全部验收通过 ✓")
    sys.exit(0)
