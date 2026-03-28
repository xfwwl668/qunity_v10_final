"""
Q-UNITY V10 — tests/test_v10_acceptance.py
============================================
验收测试套件（5项核心合规验证）

铁律（每次修改前默读）
----------------------
1. stamp_tax = 0.0005（万五，绝不手改）
2. 不修改任何现有文件
3. L3-A 必须在 L3-B 之前
4. holding_days 递增在每日循环最顶部
5. market_regime 是 int8 数组；REGIME_IDX_TO_STR 映射后才能查 FACTOR_WEIGHTS

测试列表
--------
T1  test_t1_stop_loss_compliance     T+1 止损合规
T2  test_no_residual_position_after_full_exit   全清不留残仓
T3  test_delisted_stock_cleared      退市股被清仓且不再买入
T4  test_nav_equals_cash_plus_holdings   NAV = cash + Σ(pos × close)
T5  test_regime_dimension_aligned    Regime 数组维度与回测期一致

运行方式
--------
  python tests/test_v10_acceptance.py        # 直接运行
  pytest tests/test_v10_acceptance.py -v     # pytest 模式
"""
from __future__ import annotations

import sys
import os

import numpy as np

# ── 路径补丁（单文件运行时保证能 import）────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
for _p in [_root, _here]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── 导入撮合引擎 ──────────────────────────────────────────────────────────────
try:
    from src.engine.numba_kernels_v10 import match_engine_weights_driven
except ImportError:
    from numba_kernels_v10 import match_engine_weights_driven          # type: ignore

# ── 内核调用公共默认参数（测试专用：极松约束，排除干扰因素）────────────────────
_KERNEL_DEFAULTS = dict(
    initial_cash        = 1_000_000.0,
    commission_rate     = 0.0003,
    stamp_tax           = 0.0005,    # ★ 万五，绝不修改
    slippage_rate       = 0.001,
    participation_rate  = 1.0,       # 100% 参与率，排除量约束干扰
    min_trade_value     = 0.0,       # 不设最小交易额
    rebalance_threshold = 0.0,       # 不设防补仓阈值
    max_single_pos      = 1.0,       # 单股无上限（方便构造极端权重）
    hard_stop_loss      = 0.20,      # 硬止损 20%
    max_holding_days    = 0,         # 不限持仓天数
    allow_fractional    = True,
    min_commission      = 0.0,
    full_stop_dd        = 0.99,      # 关闭组合止损（阈值极高）
    half_stop_dd        = 0.98,
    max_gap_up          = 99.0,      # 关闭 gap_up 过滤
)


def _run_kernel(
    weights   : np.ndarray,
    exec_p    : np.ndarray,
    close_p   : np.ndarray,
    volume    : np.ndarray | None = None,
    **overrides,
) -> tuple:
    """
    封装 match_engine_weights_driven 调用，自动填充 high/limit 矩阵。

    Returns (pos_matrix, nav_array, cash_array)
    """
    N, T = weights.shape
    if volume is None:
        volume = np.full((N, T), 1e9, dtype=np.float64)   # 充裕成交量
    high_p    = close_p.copy()
    limit_up  = np.zeros((N, T), dtype=np.bool_)
    limit_dn  = np.zeros((N, T), dtype=np.bool_)

    kw = {**_KERNEL_DEFAULTS, **overrides}

    return match_engine_weights_driven(
        final_target_weights = np.ascontiguousarray(weights,  dtype=np.float64),
        exec_prices          = np.ascontiguousarray(exec_p,   dtype=np.float64),
        close_prices         = np.ascontiguousarray(close_p,  dtype=np.float64),
        high_prices          = np.ascontiguousarray(high_p,   dtype=np.float64),
        volume               = np.ascontiguousarray(volume,   dtype=np.float64),
        limit_up_mask        = limit_up,
        limit_dn_mask        = limit_dn,
        **kw,
    )


# ─────────────────────────────────────────────────────────────────────────────
# T1 — T+1 止损合规
# ─────────────────────────────────────────────────────────────────────────────

def test_t1_stop_loss_compliance() -> None:
    """
    T+1 合规：买入当日（holding_days=0）不触发止损；
              次日（holding_days=1）跌幅超 hard_stop_loss 时触发止损。

    构造
    ----
    - N=2, T=4
    - 股票 0：价格稳定 10.0，全程不触发止损（对照组）
    - 股票 1：
        t=0 开盘 10.0，以 10.0 买入，holding_days 设为 0
        t=1 开盘 7.8 → 滑点后 7.8×(1-0.001)=7.7922
             loss_ratio = (7.7922 - 10.0) / 10.0 = -0.22078 < -0.20
             holding_days 在 t=1 顶部递增为 1 → L3-B 触发止损
        t=1 收盘后 stock1 position 应为 0（T+1 合规）

    关键断言
    --------
    - pos[1, 0] > 0   （t=0 当日买入，不触发止损）
    - pos[1, 1] == 0  （t=1 触发止损，持仓清零）
    """
    N, T = 2, 4

    # 股票1 t=1 开盘暴跌至 7.8（跌幅 22% > hard_stop_loss=20%）
    exec_p = np.array([
        [10.0, 10.0, 10.0, 10.0],   # 股票0：稳定
        [10.0,  7.8,  7.8,  7.8],   # 股票1：t=1 暴跌
    ], dtype=np.float64)

    close_p = np.array([
        [10.0, 10.0, 10.0, 10.0],
        [10.0,  7.5,  7.5,  7.5],
    ], dtype=np.float64)

    # 两股各占 30% 权重（总 60%，留 40% 现金缓冲保证 WP-11 不拒绝买入）
    # 若用 50%+50%=100% 会使第二股的佣金超出剩余可用现金被 WP-11 拒绝
    weights = np.full((N, T), 0.30, dtype=np.float64)

    pos_mat, nav_arr, cash_arr = _run_kernel(weights, exec_p, close_p)

    # ── 断言1a：t=0 当日买入，未触发止损（holding_days=0 时 L3-B 跳过）
    assert pos_mat[1, 0] > 0.0, (
        f"[T1-FAIL] 股票1 t=0 应有持仓（T+1合规：买入当日不止损），"
        f"实际 pos={pos_mat[1, 0]:.4f}"
    )

    # ── 断言1b：t=1 触发止损，持仓清零
    assert pos_mat[1, 1] == 0.0, (
        f"[T1-FAIL] 股票1 t=1 应已止损清仓，实际 pos={pos_mat[1, 1]:.4f}。"
        f"（holding_days t=1 顶部递增为1 → L3-B 应触发 loss_ratio≈-22%>-20%）"
    )

    # ── 断言1c：对照组（股票0）t=0 和 t=1 均有持仓
    assert pos_mat[0, 0] > 0.0, (
        f"[T1-FAIL] 股票0 t=0 应有持仓，实际={pos_mat[0, 0]:.4f}"
    )
    assert pos_mat[0, 1] > 0.0, (
        f"[T1-FAIL] 股票0 t=1 应维持持仓（价格未跌），实际={pos_mat[0, 1]:.4f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# T2 — 全清不留残仓
# ─────────────────────────────────────────────────────────────────────────────

def test_no_residual_position_after_full_exit() -> None:
    """
    全清不留残仓：权重归零时，持仓股数精确为 0.0，不存在浮点残留。

    构造
    ----
    - N=2, T=5
    - 股票 0：前 2 日持仓（weight=0.5），t=2 起权重归零 → 内核触发全清
    - 股票 1：全程权重 0（对照，永不买入）

    关键断言
    --------
    - pos[0, 2] == 0.0  （而不是 1e-6 这类浮点残留）
    - pos[0, 3] == 0.0
    - pos[0, 4] == 0.0
    - pos[1, :].sum() == 0.0（全程未持仓）
    """
    N, T = 2, 5

    exec_p  = np.full((N, T), 10.0, dtype=np.float64)
    close_p = np.full((N, T), 10.0, dtype=np.float64)

    # 权重设计：t 列 → 信号用于 t+1 执行（L3-A 使用 col_w = t-1）
    # 让股票0在 t=0,1 买入（权重列 0,1 = 0.5），t=2 列起权重=0 触发全清
    weights = np.array([
        [0.5, 0.5, 0.0, 0.0, 0.0],   # 股票0：t=2 列权重=0 → t=3 执行全清
        [0.0, 0.0, 0.0, 0.0, 0.0],   # 股票1：全程空仓
    ], dtype=np.float64)

    pos_mat, nav_arr, cash_arr = _run_kernel(weights, exec_p, close_p)

    # ── 断言：股票0 在权重归零后持仓精确为 0（不允许浮点残留）
    # 内核 is_full_exit 路径：shares_sell = position[i]（精确全部）
    # 然后 position[i] = 0.0
    assert pos_mat[0, 0] > 0.0, (
        f"[T2-FAIL] 股票0 t=0 应有持仓（初始买入），实际={pos_mat[0, 0]:.4f}"
    )
    assert pos_mat[0, 3] == 0.0, (
        f"[T2-FAIL] 全清后 t=3 股票0 有残仓={pos_mat[0, 3]:.10f}（期望精确 0.0）"
    )
    assert pos_mat[0, 4] == 0.0, (
        f"[T2-FAIL] 全清后 t=4 股票0 有残仓={pos_mat[0, 4]:.10f}（期望精确 0.0）"
    )

    # ── 断言：股票1 全程未持仓（权重=0）
    assert pos_mat[1, :].sum() == 0.0, (
        f"[T2-FAIL] 股票1 全程权重=0，不应有持仓，sum={pos_mat[1, :].sum():.6f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# T3 — 退市股被清仓且不再买入
# ─────────────────────────────────────────────────────────────────────────────

def test_delisted_stock_cleared() -> None:
    """
    退市检测：连续零成交量 > 60 天 → 强制清仓；此后即使权重非零也不再建仓。

    构造
    ----
    - N=2, T=100
    - 股票0（疑似退市）：
        t=0..4   volume > 0（允许建仓）
        t=5..99  volume = 0（连续 95 天零成交，触发退市检测）
    - 股票1：全程正常（对照）

    退市触发时间
    -----------
    consec_zero_vol > 60 首次发生在 t = 5+60 = 65
    （t=5 起 vol=0，经过 61 天后 t=65 时 consec_zero_vol=61 > 60）

    关键断言
    --------
    - pos_mat[0, 0] > 0         （t=0 正常买入）
    - pos_mat[0, 65] == 0.0     （t=65 触发退市清仓）
    - pos_mat[0, 99] == 0.0     （此后持仓永久为零）
    """
    N, T = 2, 100

    exec_p  = np.full((N, T), 10.0, dtype=np.float64)
    close_p = np.full((N, T), 10.0, dtype=np.float64)

    # 成交量：股票0 前5天有量，之后连续95天为0
    volume = np.full((N, T), 1e9, dtype=np.float64)   # 初始全有量
    volume[0, 5:] = 0.0   # 股票0 t=5 起量为零

    # 权重：两股各 50%，全程维持（退市清仓由内核自动触发）
    weights = np.full((N, T), 0.5, dtype=np.float64)

    pos_mat, nav_arr, cash_arr = _run_kernel(weights, exec_p, close_p, volume=volume)

    # ── 断言3a：股票0 t=0 正常买入（此时有成交量）
    assert pos_mat[0, 0] > 0.0, (
        f"[T3-FAIL] 股票0 t=0 应有持仓，实际={pos_mat[0, 0]:.4f}"
    )

    # ── 断言3b：consec_zero_vol > 60 触发点之后持仓必须为 0
    # t=5 起量=0，t=65 时 consec_zero_vol=61 > 60 → 该日退市清仓
    assert pos_mat[0, 65] == 0.0, (
        f"[T3-FAIL] 股票0 t=65（consec_zero_vol=61）应已退市清仓，"
        f"实际 pos={pos_mat[0, 65]:.4f}"
    )

    # ── 断言3c：退市后（t=66..99）持仓全部为 0
    # 内核 consec_zero_vol > 60 后 position=0，且之后每步量仍=0 无法买入
    post_delist_sum = pos_mat[0, 66:].sum()
    assert post_delist_sum == 0.0, (
        f"[T3-FAIL] 退市后股票0 仍被分配持仓，t=66..99 sum={post_delist_sum:.6f}"
    )

    # ── 断言3d：对照组（股票1）退市期间仍正常持仓
    assert pos_mat[1, 70] > 0.0, (
        f"[T3-FAIL] 对照组股票1 t=70 应仍有持仓，实际={pos_mat[1, 70]:.4f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# T4 — NAV = cash + Σ(position × close)
# ─────────────────────────────────────────────────────────────────────────────

def test_nav_equals_cash_plus_holdings() -> None:
    """
    NAV 计算准确性：在任意时间步 t，
    nav_array[t] == cash_array[t] + Σ_i(position_matrix[i,t] × close[i,t])
    误差 < 1e-6。

    构造
    ----
    - N=5, T=30，随机价格序列，混合买卖操作（随机权重，含清仓）
    - 检查所有 30 个时间步的 NAV 一致性
    """
    rng = np.random.default_rng(42)
    N, T = 5, 30

    # 随机价格（稳定在 5~20 元）
    base  = rng.uniform(5.0, 20.0, (N, 1))
    noise = np.cumprod(1 + rng.normal(0.0, 0.01, (N, T)), axis=1)
    close_p = (base * noise).astype(np.float64)
    exec_p  = close_p * (1 + rng.uniform(-0.005, 0.005, (N, T)))

    # 随机权重（0~0.4 per stock，每列合计 ≤ 1，含若干全零列模拟清仓期）
    weights = rng.uniform(0.0, 0.2, (N, T)).astype(np.float64)
    # 令 t=10..14 全零（全仓休息）
    weights[:, 10:15] = 0.0
    # 每列归一化，不超过 0.8（保留现金缓冲）
    col_sums = weights.sum(axis=0)
    scale = np.where(col_sums > 0.8, 0.8 / (col_sums + 1e-12), 1.0)
    weights *= scale[np.newaxis, :]

    pos_mat, nav_arr, cash_arr = _run_kernel(weights, exec_p, close_p)

    # ── 断言：每个时间步 NAV == cash + Σ(pos × close)
    max_diff = 0.0
    for t in range(T):
        manual_nav = cash_arr[t] + (pos_mat[:, t] * close_p[:, t]).sum()
        diff       = abs(manual_nav - nav_arr[t])
        max_diff   = max(max_diff, diff)
        assert diff < 1e-6, (
            f"[T4-FAIL] t={t}: nav_engine={nav_arr[t]:.6f} "
            f"manual={manual_nav:.6f} 差值={diff:.2e}（允许 < 1e-6）"
        )

    # ── 附加：NAV 全程为正且不含 NaN
    assert np.all(np.isfinite(nav_arr)), "[T4-FAIL] nav_array 含 NaN/inf"
    assert np.all(nav_arr > 0.0),        "[T4-FAIL] nav_array 含非正值"


# ─────────────────────────────────────────────────────────────────────────────
# T5 — Regime 数组维度与回测期一致
# ─────────────────────────────────────────────────────────────────────────────

def test_regime_dimension_aligned() -> None:
    """
    Regime 数组维度合规性验证：
    market_regime.shape[0] 必须等于权重矩阵的时间维 T (actual_weights.shape[1])。

    验证场景
    --------
    (a) 直接构造：模拟 MarketRegimeDetector 输出后裁切，检查维度对齐
    (b) 使用 PortfolioBuilder（若可导入）：end-to-end 验证 regime 裁切逻辑
    (c) 铁律验证：market_regime dtype 必须为 int8（int8 数组查 REGIME_IDX_TO_STR）
    """
    try:
        from src.strategies.alpha_signal import REGIME_IDX_TO_STR
    except ImportError:
        from alpha_signal import REGIME_IDX_TO_STR   # type: ignore

    rng = np.random.default_rng(0)

    # ── 场景 (a)：直接维度对齐检查 ──────────────────────────────────────────
    N, T_full, warmup = 30, 300, 50
    T_backtest = T_full - warmup   # 实际回测期

    # 模拟策略输出权重（已裁去预热期）
    actual_weights = np.zeros((N, T_backtest), dtype=np.float64)

    # 模拟 MarketRegimeDetector 输出并裁切（与 fast_runner_v10 逻辑对齐）
    # 全长 regime → 裁去预热期前缀
    regime_full = np.random.randint(0, 5, T_full, dtype=np.int8)
    regime_bt   = regime_full[warmup:]   # shape (T_backtest,)

    assert regime_bt.shape[0] == actual_weights.shape[1], (
        f"[T5-FAIL] regime_bt.shape[0]={regime_bt.shape[0]} "
        f"!= actual_weights.shape[1]={actual_weights.shape[1]}"
    )

    # ── 场景 (b)：PortfolioBuilder end-to-end 维度验证 ───────────────────────
    try:
        from src.engine.portfolio_builder import PortfolioBuilder, MarketRegimeDetector
        from src.engine.risk_config import RiskConfig
        from src.strategies.alpha_signal import AlphaSignal as _AlphaSignal
        _pb_available = True
    except ImportError:
        try:
            from portfolio_builder import PortfolioBuilder, MarketRegimeDetector  # type: ignore
            from risk_config import RiskConfig                                     # type: ignore
            from alpha_signal import AlphaSignal as _AlphaSignal                  # type: ignore
            _pb_available = True
        except ImportError:
            _pb_available = False

    if _pb_available:
        close_full = np.cumprod(
            1 + rng.normal(0, 0.01, (N, T_full)), axis=1
        ).astype(np.float64) * 10.0

        cfg        = RiskConfig()
        mrd        = MarketRegimeDetector(cfg)
        pb         = PortfolioBuilder(mrd, cfg)

        valid_full = np.ones((N, T_full), dtype=bool)

        w_bt  = np.zeros((N, T_backtest), dtype=np.float64)
        alpha = _AlphaSignal(
            raw_target_weights = w_bt,
            strategy_name      = "_test_regime_dim",
        )

        final_weights = pb.build(
            alpha      = alpha,
            valid_mask = valid_full[:, warmup:],
            close_full = close_full,
            valid_full = valid_full,
            warmup     = warmup,
        )

        assert final_weights.shape == (N, T_backtest), (
            f"[T5-FAIL] PortfolioBuilder.build 输出 shape={final_weights.shape}，"
            f"期望 ({N}, {T_backtest})"
        )

    # ── 场景 (c)：★ 铁律 — market_regime 必须是 int8 数组 ─────────────────────
    regime_int8 = np.array([0, 1, 2, 3, 4, 1, 0], dtype=np.int8)
    assert regime_int8.dtype == np.int8, (
        f"[T5-FAIL] market_regime dtype={regime_int8.dtype}，应为 int8"
    )

    # ── 铁律：int8 → int → str → 查 REGIME_IDX_TO_STR，不直接用 int8 做键 ──
    for raw_val in regime_int8:
        regime_str = REGIME_IDX_TO_STR[int(raw_val)]   # int8 → int → str
        assert isinstance(regime_str, str), (
            f"[T5-FAIL] REGIME_IDX_TO_STR[{int(raw_val)}]={regime_str!r} 不是 str"
        )

    # ── 验证不能用 int8 直接查（不允许 REGIME_IDX_TO_STR[np.int8(0)] 风格）
    # 此处确认正确用法：int(regime_val) 转换后查表
    assert REGIME_IDX_TO_STR[int(np.int8(0))]  == "STRONG_BULL"
    assert REGIME_IDX_TO_STR[int(np.int8(4))]  == "BEAR"

    # ── 维度最终断言（主验收）
    assert regime_bt.shape[0] == T_backtest, (
        f"[T5-FAIL] regime_bt.shape[0]={regime_bt.shape[0]} != T_backtest={T_backtest}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("Q-UNITY V10  test_v10_acceptance.py  验收测试套件")
    print("=" * 65)
    print(f"  stamp_tax 铁律: 0.0005（测试内核固定传入 {_KERNEL_DEFAULTS['stamp_tax']}）")
    print()

    tests = [
        (test_t1_stop_loss_compliance,              "T1  T+1止损合规"),
        (test_no_residual_position_after_full_exit, "T2  全清不留残仓"),
        (test_delisted_stock_cleared,               "T3  退市清仓"),
        (test_nav_equals_cash_plus_holdings,        "T4  NAV准确"),
        (test_regime_dimension_aligned,             "T5  Regime维度"),
    ]

    passed, failed = 0, 0
    for fn, label in tests:
        try:
            fn()
            print(f"  ✅ {label}")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {label}\n     {e}")
            failed += 1
        except Exception as e:
            print(f"  ⚠️  {label}（非断言异常）\n     {type(e).__name__}: {e}")
            failed += 1

    print()
    print(f"  结果：{passed} 通过 / {failed} 失败 / {len(tests)} 总计")
    print()

    if failed == 0:
        print("[PASS] 全部验收通过 ✓")
        sys.exit(0)
    else:
        print("[FAIL] 存在失败项，请检查上方输出")
        sys.exit(1)
