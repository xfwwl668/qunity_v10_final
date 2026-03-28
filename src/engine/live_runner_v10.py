"""
Q-UNITY V10 — src/engine/live_runner_v10.py
============================================
实盘/模拟运行器（统一架构版）

复用 V10 的所有基础设施：
  RiskConfig · PortfolioBuilder · MarketRegimeDetector
  VEC_STRATEGY_REGISTRY · AlphaSignal

数据来自 LiveDataAdapter（AkShare），通过 extra_factors 注入
真实竞价/资金数据，策略函数与回测路径完全相同。

运行方式：
    # 实盘模式（9:30 后，使用当日真实数据）
    python -m src.engine.live_runner_v10 --strategy weak_to_strong --mode live

    # 模拟模式（任意时间，使用昨日数据验证流程）
    python -m src.engine.live_runner_v10 --strategy weak_to_strong --mode paper

铁律（每次修改前默读）
----------------------
1. stamp_tax = 0.0005（从 RiskConfig 读取，绝不手改）
2. 信号生成必须使用已收盘数据，不允许当日前视
3. valid_mask 退市/停牌股必须为 False
4. market_regime 是 int8 数组，查表前必须 REGIME_IDX_TO_STR 转换
5. 不修改任何现有文件
"""
from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── 项目根加入 sys.path ──────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# 结果容器
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LiveSignal:
    """单只股票的实盘信号"""
    code:              str
    name:              str
    weight:            float     # PortfolioBuilder 分配的目标仓位
    score:             float
    zhaban_depth:      float
    open_change:       float
    volume_ratio:      float
    price:             float
    regime:            str
    data_mode:         str        # "live"（真实数据）or "proxy"（代理因子）
    signal_time:       str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))

    def to_dict(self) -> dict:
        return {
            "code": self.code, "name": self.name,
            "weight_pct": round(self.weight * 100, 2),
            "score": round(self.score, 1),
            "zhaban_depth": round(self.zhaban_depth, 2),
            "open_change": round(self.open_change, 2),
            "volume_ratio": round(self.volume_ratio, 2),
            "price": self.price,
            "regime": self.regime,
            "data_mode": self.data_mode,
            "signal_time": self.signal_time,
        }


@dataclass
class LiveRunResult:
    """单次运行结果"""
    signals:     List[LiveSignal]
    regime:      str
    trade_date:  str
    n_candidates: int
    n_signals:   int
    elapsed_sec: float
    extra_info:  Dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"{'='*60}",
            f"  Q-UNITY V10 实盘信号  {self.trade_date}",
            f"{'='*60}",
            f"  市场状态   : {self.regime}",
            f"  候选股     : {self.n_candidates} 只",
            f"  有效信号   : {self.n_signals} 只",
            f"  耗时       : {self.elapsed_sec:.1f}s",
        ]
        if self.signals:
            lines.append(f"\n  {'代码':<8} {'仓位':<7} {'评分':<6} {'炸板深度':<9} {'数据来源'}")
            lines.append(f"  {'-'*50}")
            for s in self.signals:
                lines.append(
                    f"  {s.code:<8} {s.weight*100:>5.1f}%  "
                    f"{s.score:>5.1f}  {s.zhaban_depth:>6.2f}%     {s.data_mode}"
                )
        lines.append(f"{'='*60}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 运行器主体
# ─────────────────────────────────────────────────────────────────────────────

class LiveRunnerV10:
    """
    实盘/模拟运行器。

    与 FastRunnerV10 的关系：
      FastRunnerV10  → 批量回测，npy memmap，Numba 撮合
      LiveRunnerV10  → 实盘/模拟，AkShare，真实仓位输出
      两者共享：RiskConfig · PortfolioBuilder · 策略注册表 · AlphaSignal
    """

    def __init__(
        self,
        config_path: str = "config.json",
        strategy_name: str = "weak_to_strong",
        mode: str = "paper",       # "live" | "paper"
        lookback_days: int = 30,
        cache_ttl: int = 60,
    ):
        self.strategy_name = strategy_name
        self.mode          = mode

        # ── 加载配置 ────────────────────────────────────────────────────────
        cfg = self._load_config(config_path)
        self.cfg = cfg

        # ── 构造 RiskConfig ──────────────────────────────────────────────────
        from src.engine.risk_config import RiskConfig
        risk_kw = {k: v for k, v in cfg.get("risk", {}).items() if not k.startswith("_")}
        self.risk_config = RiskConfig(**risk_kw)
        assert self.risk_config.stamp_tax == 0.0005, (
            f"[FAIL] stamp_tax={self.risk_config.stamp_tax}，应为 0.0005"
        )

        # ── 策略参数 ─────────────────────────────────────────────────────────
        from src.engine.optimizer_v10 import StrategyParams
        sp_cfg = cfg.get("strategy_params", {}).get(strategy_name, {})
        self.params = StrategyParams(**sp_cfg)

        # ── 实盘数据适配器 ────────────────────────────────────────────────────
        from src.data.live_data_adapter import LiveDataAdapter
        self.adapter = LiveDataAdapter(
            lookback_days=lookback_days,
            cache_ttl_seconds=cache_ttl,
            max_workers=cfg.get("data", {}).get("n_workers", 8),
        )

        # ── 市场状态检测器 ────────────────────────────────────────────────────
        from src.engine.portfolio_builder import MarketRegimeDetector
        self.regime_detector = MarketRegimeDetector(self.risk_config)

        # ── 策略函数 ─────────────────────────────────────────────────────────
        from src.strategies.registry import get_alpha_fn
        self.alpha_fn = get_alpha_fn(strategy_name)

        logger.info(
            f"LiveRunnerV10 初始化: strategy={strategy_name} "
            f"mode={mode} stamp_tax={self.risk_config.stamp_tax}"
        )

    # ── 主入口 ───────────────────────────────────────────────────────────────

    def run(self) -> LiveRunResult:
        """
        执行单次信号生成。

        流程：
          1. LiveDataAdapter → 基础 numpy 数组 + extra_factors
          2. MarketRegimeDetector → market_regime
          3. alpha_fn() → AlphaSignal（共享策略代码）
          4. PortfolioBuilder → 归一化权重 + regime 缩放
          5. 生成 LiveSignal 列表
        """
        t0 = time.perf_counter()

        # ── 1. 获取数据快照 ──────────────────────────────────────────────────
        logger.info("获取实盘数据快照...")
        snapshot = self.adapter.get_snapshot(force_refresh=(self.mode == "live"))
        if snapshot is None:
            logger.error("快照获取失败，终止本次运行")
            return LiveRunResult([], "unknown", "N/A", 0, 0, time.perf_counter() - t0)

        n, n_days = snapshot.close.shape
        logger.info(f"快照就绪: {n} 只股票 × {n_days} 天，日期 {snapshot.trade_date}")

        # ── 2. 市场状态 ───────────────────────────────────────────────────────
        # 用 close 最后一段（20天窗口）计算 breadth，当 npy 不可用时以全市场近似
        try:
            nav_proxy = snapshot.close[:, -1].mean() / (snapshot.close[:, -60].mean() + 1e-8)
            breadth_proxy = float((snapshot.close[:, -1] > snapshot.close[:, -2]).mean())
            # 构造 int8 market_regime 数组（仅用最后1天）
            # [FIX-L-01] 正确的 idx 映射（与 alpha_signal.REGIME_IDX_TO_STR 一致）：
            # 0=STRONG_BULL, 1=BULL, 2=NEUTRAL, 3=SOFT_BEAR, 4=BEAR
            # 原版 STRONG_BULL→1、BEAR→3 全部错误，导致仓位上限和因子权重偏移。
            if breadth_proxy >= self.risk_config.strong_bull_breadth:
                regime_idx = 0   # STRONG_BULL (仓位上限 1.0)
            elif breadth_proxy >= self.risk_config.bull_breadth_thr:
                regime_idx = 1   # BULL        (仓位上限 0.8)
            elif breadth_proxy >= self.risk_config.soft_bear_breadth:
                regime_idx = 2   # NEUTRAL     (仓位上限 0.8)
            elif breadth_proxy >= self.risk_config.bear_breadth_thr:
                regime_idx = 3   # SOFT_BEAR   (仓位上限 0.4)
            else:
                regime_idx = 4   # BEAR        (仓位上限 0.0 → 空仓)
            market_regime = np.array([regime_idx], dtype=np.int8)
        except Exception as exc:
            logger.warning(f"regime 计算失败，使用 normal: {exc}")
            market_regime = np.array([0], dtype=np.int8)

        from src.strategies.alpha_signal import REGIME_IDX_TO_STR
        regime_str = REGIME_IDX_TO_STR.get(int(market_regime[-1]), "normal")
        logger.info(f"当前市场状态: {regime_str} (idx={int(market_regime[-1])})")

        # ── 3. 调用策略函数（与回测路径完全相同的代码）──────────────────────
        logger.info(f"运行策略: {self.strategy_name}")
        try:
            alpha_signal = self.alpha_fn(
                close         = snapshot.close,
                open_         = snapshot.open_,
                high          = snapshot.high,
                low           = snapshot.low,
                volume        = snapshot.volume,
                params        = self.params,
                market_regime = market_regime,
                valid_mask    = snapshot.valid_mask[:, np.newaxis],  # [n, 1] → broadcast
                extra_factors = snapshot.extra_factors,
                codes         = snapshot.codes,   # 传给策略的 kwargs，用于炸板池匹配
            )
        except Exception as exc:
            logger.error(f"策略运行失败: {exc}", exc_info=True)
            return LiveRunResult([], regime_str, snapshot.trade_date, 0, 0, time.perf_counter() - t0)

        # ── 4. PortfolioBuilder 归一化 ────────────────────────────────────────
        # [FIX-L-02] 原版传 alpha_signal= 但 build() 参数名是 alpha=，会 TypeError 崩溃。
        # 实盘用 build_single_day()：接受 (N,) 原始权重 + (N,) valid_mask，
        # 使用缓存的最新 regime_limit（由 regime_idx 决定）。
        from src.engine.portfolio_builder import PortfolioBuilder, _REGIME_IDX_TO_LIMIT
        builder = PortfolioBuilder(self.risk_config)

        # 手动设置今日 regime_limit（避免调用 compute() 需要全段数据）
        regime_lim = _REGIME_IDX_TO_LIMIT.get(int(market_regime[-1]), 1.0)
        builder._regime_limits = np.array([regime_lim], dtype=np.float64)

        # 取最后一列权重作为今日单日权重
        raw_w_today = alpha_signal.raw_target_weights[:, -1].astype(np.float64)
        target_weights: np.ndarray = builder.build_single_day(
            raw_w_today     = raw_w_today,
            valid_mask_today = snapshot.valid_mask.astype(np.bool_),
        )

        # ── 5. 生成 LiveSignal 列表 ───────────────────────────────────────────
        live_signals: List[LiveSignal] = []
        has_live_data = bool(snapshot.extra_factors.get("zhaban_codes"))

        t1_idx = n_days - 2
        t2_idx = n_days - 3

        for i, code in enumerate(snapshot.codes):
            w = float(target_weights[i])
            if w < 1e-6:
                continue

            price    = float(snapshot.close[i, -1])
            h1       = float(snapshot.high[i, t1_idx])
            c1       = float(snapshot.close[i, t1_idx])
            c2       = float(snapshot.close[i, t2_idx])
            v1       = float(snapshot.volume[i, t1_idx])
            v2       = float(snapshot.volume[i, t2_idx])
            o0       = float(snapshot.open_[i, -1])

            zhaban_depth = (h1 - c1) / (h1 + 1e-8) * 100 if h1 > 0 else 0.0
            open_change  = (o0 / (c1 + 1e-8) - 1) * 100 if c1 > 0 else 0.0
            vol_ratio    = v1 / (v2 + 1e-8) if v2 > 0 else 1.0

            live_signals.append(LiveSignal(
                code         = code,
                name         = code,    # 实盘可通过 stock_info 查询名称
                weight       = w,
                score        = 0.0,     # 已归一化，原始 score 不再暴露
                zhaban_depth = zhaban_depth,
                open_change  = open_change,
                volume_ratio = vol_ratio,
                price        = price,
                regime       = regime_str,
                data_mode    = "live" if has_live_data else "proxy",
            ))

        live_signals.sort(key=lambda x: x.weight, reverse=True)
        elapsed = time.perf_counter() - t0

        result = LiveRunResult(
            signals      = live_signals,
            regime       = regime_str,
            trade_date   = snapshot.trade_date,
            n_candidates = int((alpha_signal.raw_target_weights[:, -1] > 0).sum()),  # [FIX-L-03] 取最后列
            n_signals    = len(live_signals),
            elapsed_sec  = elapsed,
            extra_info   = {
                "has_live_zhaban":   bool(snapshot.extra_factors.get("zhaban_codes")),
                "has_live_auction":  snapshot.extra_factors.get("auction_score") is not None,
                "has_live_money":    snapshot.extra_factors.get("money_flow_score") is not None,
                "stamp_tax":         self.risk_config.stamp_tax,
            },
        )

        logger.info(f"运行完成: {len(live_signals)} 个信号，耗时 {elapsed:.1f}s")
        return result

    # ── 工具方法 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _load_config(config_path: str) -> dict:
        p = Path(config_path)
        if not p.exists():
            logger.warning(f"config.json 不存在: {p}，使用空配置")
            return {}
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def export_signals(self, result: LiveRunResult, output_dir: str = "results") -> Path:
        """导出信号到 CSV"""
        import pandas as pd
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        fp  = out / f"live_signals_{self.strategy_name}_{ts}.csv"
        df  = pd.DataFrame([s.to_dict() for s in result.signals])
        df.to_csv(fp, index=False, encoding="utf-8-sig")
        logger.info(f"信号已导出: {fp}")
        return fp


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        prog="live_runner_v10.py",
        description="Q-UNITY V10 实盘/模拟运行器",
    )
    parser.add_argument("--strategy", "-s", default="weak_to_strong",
                        help="策略名称（默认 weak_to_strong）")
    parser.add_argument("--mode", choices=["live", "paper"], default="paper",
                        help="live=实盘  paper=模拟（默认 paper）")
    parser.add_argument("--config", "-c", default="config.json",
                        help="配置文件路径")
    parser.add_argument("--lookback", type=int, default=30,
                        help="历史窗口天数（默认 30）")
    parser.add_argument("--output", "-o", default="results",
                        help="信号输出目录")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    runner = LiveRunnerV10(
        config_path   = args.config,
        strategy_name = args.strategy,
        mode          = args.mode,
        lookback_days = args.lookback,
    )

    result = runner.run()
    print(result.summary())

    if result.signals:
        fp = runner.export_signals(result, args.output)
        print(f"\n信号已保存: {fp}")

    # 显示数据来源透明度
    info = result.extra_info
    print(f"\n数据来源：")
    print(f"  炸板池     : {'✓ 真实 API' if info.get('has_live_zhaban') else '△ 日线近似'}")
    print(f"  竞价评分   : {'✓ 真实 API' if info.get('has_live_auction') else '△ open/prev_close'}")
    print(f"  主力资金   : {'✓ 真实 API' if info.get('has_live_money') else '△ 量价组合'}")
    print(f"  印花税     : {info.get('stamp_tax')} (万五 ✓)")


if __name__ == "__main__":
    main()
