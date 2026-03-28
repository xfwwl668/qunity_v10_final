from __future__ import annotations
import warnings
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
# NOT_IMPLEMENTED 字段集合 —— 任何此集合内的字段被用户显式设置为非默认值时，
# __post_init__ 会发出 UserWarning，防止用户误以为修改有效。
# ★[FIX-T5] 从注释型标记升级为运行时主动警告
# ─────────────────────────────────────────────────────────────────────────────
_NOT_IMPL_DEFAULTS: dict = {
    "stop_loss_lookback"  : 0,
    "max_sector_pos"      : 0.0,
    "market_impact_eta"   : 0.0,
    "market_impact_alpha" : 0.5,
    "account_id"          : "",
    "max_daily_orders"    : 300,
    "max_cancel_rate"     : 0.60,
    "single_stock_limit"  : 0.045,
}


@dataclass
class RiskConfig:
    # ── Regime状态机 ─────────────────────────────────────────────────
    bear_breadth_thr    : float = 0.25
    bear_nav_thr        : float = 0.96
    bear_confirm_days   : int   = 5
    bear_exit_days      : int   = 8
    soft_bear_breadth   : float = 0.32
    bull_breadth_thr    : float = 0.40
    strong_bull_breadth : float = 0.50
    breadth_window      : int   = 20
    nav_ma_window       : int   = 60

    # ── 组合止损（Numba内核内部执行）────────────────────────────────
    full_stop_dd        : float = 0.25
    half_stop_dd        : float = 0.12
    stop_recovery_days  : int   = 10

    # ── 执行参数 ─────────────────────────────────────────────────────
    allow_fractional    : bool  = True
    rebalance_threshold : float = 0.05
    min_trade_value     : float = 1000.0
    max_single_pos      : float = 0.08
    hard_stop_loss      : float = 0.20
    max_holding_days    : int   = 0
    min_avg_amount      : float = 5e6
    max_gap_up          : float = 0.025

    # ── 执行参与率 ───────────────────────────────────────────────────
    participation_rate  : float = 0.10

    # ── 成本参数 ─────────────────────────────────────────────────────
    commission_rate     : float = 0.0003
    stamp_tax           : float = 0.0005   # 万五（2024-09-24起）★ 不可修改
    slippage_rate       : float = 0.001
    min_commission      : float = 5.0

    # ── [NOT_IMPLEMENTED] 字段：已定义，但尚未接入内核 ────────────────
    # ★[FIX-T5] 修改这些字段不会影响回测结果。
    #   __post_init__ 会在用户传入非默认值时主动发出 UserWarning。
    #   实现路线图：
    #     stop_loss_lookback  → numba_kernels L3-B rolling-max 窗口
    #     max_sector_pos      → portfolio_builder 行业约束层
    #     market_impact_eta/alpha → 内核买入路径冲击成本模型
    #     account_id/max_daily_orders/max_cancel_rate/single_stock_limit → 实盘合规层
    stop_loss_lookback  : int   = 0
    max_sector_pos      : float = 0.0
    market_impact_eta   : float = 0.0
    market_impact_alpha : float = 0.5
    account_id          : str   = ""
    max_daily_orders    : int   = 300
    max_cancel_rate     : float = 0.60
    single_stock_limit  : float = 0.045

    def __post_init__(self) -> None:
        """★[FIX-T5] 对 NOT_IMPLEMENTED 字段的非默认赋值发出明确警告。"""
        for fname, default_val in _NOT_IMPL_DEFAULTS.items():
            actual = getattr(self, fname)
            if actual != default_val:
                warnings.warn(
                    f"[RiskConfig] '{fname}' 被设置为 {actual!r}，"
                    f"但该字段尚未接入回测引擎（[NOT_IMPLEMENTED]），"
                    f"修改不会对回测/实盘结果产生任何影响。"
                    f"默认值为 {default_val!r}。",
                    UserWarning,
                    stacklevel=2,
                )

    def to_kernel_kwargs(self) -> dict:
        """返回传入 Numba 内核的参数字典。NOT_IMPLEMENTED 字段不在此返回。"""
        return dict(
            commission_rate     = self.commission_rate,
            stamp_tax           = self.stamp_tax,
            slippage_rate       = self.slippage_rate,
            participation_rate  = self.participation_rate,
            min_trade_value     = self.min_trade_value,
            rebalance_threshold = self.rebalance_threshold,
            max_single_pos      = self.max_single_pos,
            hard_stop_loss      = self.hard_stop_loss,
            max_holding_days    = self.max_holding_days,
            allow_fractional    = self.allow_fractional,
            min_commission      = self.min_commission,
            full_stop_dd        = self.full_stop_dd,
            half_stop_dd        = self.half_stop_dd,
            max_gap_up          = self.max_gap_up,
            stop_recovery_days  = self.stop_recovery_days,
            # ★[FIX-EXIT] 默认止损模式（策略级 exit_config 会在 fast_runner 中覆盖）
            stop_mode_trailing  = True,
            take_profit         = 0.0,
        )


if __name__ == '__main__':
    import sys

    print("=" * 60)
    print("Q-UNITY V10  risk_config.py  验收测试")
    print("=" * 60)

    cfg = RiskConfig()
    assert cfg.stamp_tax == 0.0005
    print(f"[OK] stamp_tax = {cfg.stamp_tax} (万五 ✓)")

    kw = cfg.to_kernel_kwargs()
    required = {
        "commission_rate", "stamp_tax", "slippage_rate",
        "participation_rate", "min_trade_value", "rebalance_threshold",
        "max_single_pos", "hard_stop_loss", "max_holding_days",
        "allow_fractional", "min_commission", "full_stop_dd",
        "half_stop_dd", "max_gap_up", "stop_recovery_days",
        "stop_mode_trailing", "take_profit",
    }
    missing = required - kw.keys()
    assert not missing, f"[FAIL] to_kernel_kwargs 缺少键: {missing}"
    # NOT_IMPLEMENTED 字段不应出现在 kernel_kwargs
    for f in _NOT_IMPL_DEFAULTS:
        assert f not in kw, f"[FAIL] NOT_IMPLEMENTED 字段 '{f}' 不应出现在 kernel_kwargs"
    print(f"[OK] to_kernel_kwargs 包含 {len(kw)} 个有效键，NOT_IMPL 字段已过滤 ✓")

    # ★[FIX-T5] 修改 NOT_IMPLEMENTED 字段应触发 UserWarning
    import warnings as _w
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        cfg2 = RiskConfig(stop_loss_lookback=20, market_impact_eta=0.6)
    ni_warns = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert len(ni_warns) == 2, f"[FAIL] 期望2个警告，实际: {ni_warns}"
    print(f"[OK] NOT_IMPLEMENTED 字段警告正确触发: {len(ni_warns)} 条 ✓")

    print("\n[PASS] risk_config.py 全部验收通过 ✓")
    sys.exit(0)
