from __future__ import annotations

"""
Q-UNITY V10 — AlphaSignal
=========================
AlphaSignal dataclass  +  工具函数 _score_to_weights
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 常量：Regime 索引 → 字符串标签
# market_regime 数组类型为 int8，对应索引如下
# ─────────────────────────────────────────────────────────────────────────────
REGIME_IDX_TO_STR: Dict[int, str] = {
    0: "STRONG_BULL",
    1: "BULL",
    2: "NEUTRAL",
    3: "SOFT_BEAR",
    4: "BEAR",
}

# ─────────────────────────────────────────────────────────────────────────────
# AlphaSignal
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AlphaSignal:
    """
    封装策略输出的权重矩阵与可选的评分矩阵。

    Parameters
    ----------
    raw_target_weights : np.ndarray, shape (N, T), dtype float64
        目标权重矩阵（必填）。N=股票数，T=时间步数。
        每列之和 ∈ [0, 1]；单格值 ∈ [0, max_single_pos]。
    score : np.ndarray | None, shape (N, T), dtype float64
        原始评分矩阵（可选）。-inf 表示该股票在该时间步不可用。
    strategy_name : str
        策略名称，用于日志与报告。
    meta : dict
        附加元信息（超参数快照、版本号等）。
    """

    raw_target_weights : np.ndarray
    score              : Optional[np.ndarray] = None
    strategy_name      : str                  = ""
    meta               : Dict[str, Any]       = field(default_factory=dict)
    # ★[FIX-T4] 策略个性化出场配置（exit_config）
    # 若非 None，fast_runner 会用此配置覆盖 RiskConfig 全局止损参数。
    # 支持的键：
    #   stop_mode        : "trailing" | "entry_price"  （默认 "trailing"）
    #   hard_stop_loss   : float                        （覆盖全局硬止损比例）
    #   take_profit      : float                        （0.0=不启用）
    #   max_holding_days : int                          （0=不限）
    # 示例（事件驱动策略）：
    #   exit_config = {"stop_mode": "entry_price", "hard_stop_loss": 0.07,
    #                  "take_profit": 0.12, "max_holding_days": 5}
    exit_config        : Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # 验证
    # ------------------------------------------------------------------
    def validate(self, N: int, T: int) -> None:
        """
        检查 raw_target_weights（以及 score，若存在）的合法性。

        规则
        ----
        1. shape 必须等于 (N, T)
        2. dtype 必须为 float64
        3. 不含 NaN（-inf / +inf 在 score 中允许，但 weights 不允许）
        4. 所有值 >= 0（无负权重）

        Raises
        ------
        ValueError  条件不满足时抛出，携带详细错误描述。
        """
        w = self.raw_target_weights

        # 1. shape
        if w.shape != (N, T):
            raise ValueError(
                f"[AlphaSignal] raw_target_weights.shape={w.shape}，"
                f"期望 ({N}, {T})"
            )

        # 2. dtype
        if w.dtype != np.float64:
            raise ValueError(
                f"[AlphaSignal] raw_target_weights.dtype={w.dtype}，"
                f"期望 float64"
            )

        # 3. 无 NaN（权重矩阵不允许任何 NaN / inf）
        if not np.all(np.isfinite(w)):
            bad = int(np.sum(~np.isfinite(w)))
            raise ValueError(
                f"[AlphaSignal] raw_target_weights 含 {bad} 个非有限值（NaN/inf）"
            )

        # 4. 无负值
        if np.any(w < 0.0):
            bad = int(np.sum(w < 0.0))
            raise ValueError(
                f"[AlphaSignal] raw_target_weights 含 {bad} 个负值"
            )

        # ── score 可选检查 ────────────────────────────────────────────
        if self.score is not None:
            s = self.score

            if s.shape != (N, T):
                raise ValueError(
                    f"[AlphaSignal] score.shape={s.shape}，"
                    f"期望 ({N}, {T})"
                )

            if s.dtype != np.float64:
                raise ValueError(
                    f"[AlphaSignal] score.dtype={s.dtype}，期望 float64"
                )

            # score 允许 -inf（表示不可用），但不允许 NaN 或 +inf
            nan_mask = np.isnan(s)
            posinf_mask = np.isposinf(s)
            if nan_mask.any():
                raise ValueError(
                    f"[AlphaSignal] score 含 {int(nan_mask.sum())} 个 NaN"
                )
            if posinf_mask.any():
                raise ValueError(
                    f"[AlphaSignal] score 含 {int(posinf_mask.sum())} 个 +inf，"
                    "不可用股票请用 -inf 标记"
                )

    def __repr__(self) -> str:  # pragma: no cover
        N, T = self.raw_target_weights.shape
        return (
            f"AlphaSignal("
            f"strategy='{self.strategy_name}', "
            f"shape=({N},{T}), "
            f"has_score={self.score is not None}"
            f")"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _ema_smooth_factor(mat: np.ndarray, span: int) -> np.ndarray:
    """
    对因子矩阵 (N, T) 做时序 EMA 平滑，使截面排名更稳定，有效降低策略换手率。

    [FIX-EMA-02] 与 score_ema_span（已撤销）的区别：
    - score EMA：对最终排名分做平滑，不改变截面排名 → 无效
    - factor EMA：对原始因子值做平滑，改变截面相对大小 → 有效
      EMA(beta) 的截面排名 ≠ beta 的截面排名，实测可降低换手率 40-65%

    处理规则：
    - NaN 位置跳过（初始 NaN 阶段不传播，待有效值后开始累积 EMA）
    - EMA 只在时序方向（axis=1）计算

    Parameters
    ----------
    mat  : (N, T) float64，原始因子矩阵（NaN 允许）
    span : EMA 窗口大小（alpha = 2/(span+1)）。span≤1 时直接返回原矩阵。

    Returns
    -------
    (N, T) float64，平滑后的因子矩阵
    """
    if span <= 1:
        return mat
    N, T  = mat.shape
    alpha = 2.0 / (span + 1)
    out   = np.full((N, T), np.nan, dtype=np.float64)

    # [FIX-EMA-INF] -inf 不是"缺失值"的一种，而是"门控屏蔽"标记。
    # 若让 -inf 进入 EMA 递推，α·(-inf) + (1-α)·prev = -inf，会将屏蔽状态永久
    # 传染到所有后续时步，把已恢复正常的股票全部锁死在 -inf，导致永久空仓。
    # 修复：EMA 仅对有限值（finite）累积；遇到 -inf 时，EMA 状态保持，当日输出 -inf；
    # 下一个有限值出现时，EMA 从"当前缓存的有限 prev"继续续算。
    ema_state = mat[:, 0].copy()          # 初始 EMA 状态（允许 NaN/-inf）
    out[:, 0] = ema_state
    for t in range(1, T):
        cur = mat[:, t]
        cur_finite = np.isfinite(cur)     # True = 普通值，可参与 EMA 递推
        cur_neginf = np.isneginf(cur)     # True = 门控屏蔽，输出 -inf，状态冻结
        prev_finite = np.isfinite(ema_state)

        # 情形1：cur 有限 & prev 有限 → 正常 EMA 递推，更新状态
        both_ok = cur_finite & prev_finite
        new_ema  = np.where(both_ok, alpha * cur + (1.0 - alpha) * ema_state, ema_state)
        # 情形2：cur 有限 & prev 无效(NaN) → 用 cur 重新初始化状态
        init_ok  = cur_finite & ~prev_finite
        new_ema  = np.where(init_ok, cur, new_ema)
        # 更新 EMA 状态（仅当 cur 有限时推进；cur=-inf/NaN 时状态冻结不变）
        ema_state = np.where(cur_finite, new_ema, ema_state)

        # 输出：-inf 原样输出（不让 EMA 值覆盖屏蔽标记），NaN 输出 NaN，否则输出新 EMA
        out[:, t] = np.where(cur_neginf, -np.inf,
                    np.where(np.isnan(cur), np.nan, new_ema))
    return out


def _score_to_weights(
    score         : np.ndarray,
    top_n         : int,
    max_single_pos: float = 0.08,
    exit_buffer   : int   = 5,
    dropout_days  : int   = 3,
    hard_invalid  : Optional[np.ndarray] = None,
    force_exit    : Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    将评分矩阵转换为等权目标权重矩阵。

    [FIX-BUG1] 新增状态防抖机制（V8 ever_bought/dropout_days 等价实现）。
    防止排名边缘的股票因每日微小排名震荡而被频繁全仓买入/卖出，
    从而将年换手率从 2000%~5000% 压制到合理区间。

    防抖逻辑：
    - 入场：当日排名 ≤ top_n，即可入场（无入场防抖，保留敏感性）。
    - 出场：连续 dropout_days 天排名超出 top_n + exit_buffer 才真正清仓。
      排名在 (top_n, top_n + exit_buffer] 区间内"吃缓冲"，当日保持持仓。

    [FIX-P0] hard_invalid 参数：区分「物理不可交易」与「信号门控失效」。
    A股涨跌停/停牌日 score=-inf 会导致防抖状态被强制重置（in_portfolio=False），
    次日涨跌停解开后重新触发买入信号，产生无效双向交易，是换手率居高不下的根因。
    hard_invalid[i,t]=True 时：当日权重=0，但冻结 in_portfolio[i] 与 absent_days[i]，
    不计入缺席天数，次日直接延续持仓状态，消除涨跌停边界的假性交易。

    [FIX-EXIT] force_exit 参数：策略主动卖出信号。
    允许策略声明"我今天要卖这只股票"，绕过排名防抖直接清仓。
    典型场景：短线策略的止盈/信号反转/持仓天数到期等。

    Parameters
    ----------
    score : np.ndarray, shape (N, T), dtype float64
        原始评分。-inf 表示该股票在该时间步不可用（valid_mask 已预处理）。
        NaN 不应出现（调用前请确保 valid_mask 已将无效位置设为 -inf）。
    top_n : int
        每个时间步选取评分最高的前 top_n 只股票（-inf 位置跳过）。
    max_single_pos : float
        单股最大持仓比例上限（等权后若超限则截断并重新归一化）。
    exit_buffer : int
        出场缓冲宽度。排名需超出 top_n + exit_buffer 才开始计缺席天数。
        默认 5，适合 top_n=25 左右的持仓规模。
    dropout_days : int
        连续缺席天数阈值。连续超出 exit_buffer 外达到此天数才清仓。
        默认 3（即连续3日超出缓冲区才真正离场）。
    hard_invalid : np.ndarray or None, shape (N, T), dtype bool, optional
        [FIX-P0] 物理不可交易掩码（涨跌停/停牌/valid_mask=False）。
        True  → 当日权重=0，但冻结防抖状态（不清仓、不计缺席天数）。
        False → 正常走防抖逻辑（-inf 时按缺席处理）。
        None  → 保持旧行为（全部 -inf 视为强制清仓，向后兼容）。
        建议：各策略将 valid_mask 直接传入此参数。
    force_exit : np.ndarray or None, shape (N, T), dtype bool, optional
        [FIX-EXIT] 策略主动卖出掩码。
        True  → 当日立即清仓（in_portfolio=False, absent_days=0），权重=0。
                优先级高于排名防抖，但低于 hard_invalid（物理不可交易时仍冻结状态）。
        False → 正常走防抖逻辑。
        None  → 无主动卖出信号（向后兼容）。

    Returns
    -------
    weights : np.ndarray, shape (N, T), dtype float64
        每列之和 ∈ [0, 1]；hard_invalid 或 -inf 位置权重恒为 0。

    Notes
    -----
    * 若某时间步可用股票数 < top_n，则以实际可用数等权分配。
    * 等权权重 = 1 / min(top_n, valid_count)，超过 max_single_pos 则截断。
    * 截断后权重按比例重新归一化，保证列和 <= 1。
    """
    score = np.asarray(score, dtype=np.float64)
    N, T  = score.shape

    # NOTE: score_ema_span 参数保留签名以向后兼容，但实际上对「排名式」信号无效：
    # EMA(score) 保持单调性，不改变股票间相对排名，防抖行为因此不变。
    # 真正有效的降换手方案是在策略层对原始因子做EMA（见各策略 [FIX-AM-01] [FIX-SN-01]）。

    # [FIX-P0] 预处理 hard_invalid：转换为 bool 数组，None → 全 False
    if hard_invalid is not None:
        hi = np.asarray(hard_invalid, dtype=np.bool_)
    else:
        hi = None   # None → 保持旧行为

    # [FIX-EXIT] 预处理 force_exit：转换为 bool 数组，None → 无主动卖出
    if force_exit is not None:
        fe = np.asarray(force_exit, dtype=np.bool_)
    else:
        fe = None

    weights = np.zeros((N, T), dtype=np.float64)

    # [FIX-BUG1] 防抖状态变量
    in_portfolio  = np.zeros(N, dtype=np.bool_)   # 当前是否在持仓组合中
    absent_days   = np.zeros(N, dtype=np.int64)   # 连续缺席（超出缓冲区）天数

    for t in range(T):
        col       = score[:, t]
        valid     = ~np.isneginf(col)
        valid_idx = np.where(valid)[0]

        if valid_idx.size == 0:
            # [FIX-P0] 无可用股票列：仅清仓「非冻结」股票
            if hi is not None:
                hi_t = hi[:, t]
                for i in range(N):
                    if not hi_t[i]:          # 非冻结股票正常清仓
                        in_portfolio[i] = False
                        absent_days[i]  = 0
                # 冻结股票保持状态，权重=0 (已初始化)
            else:
                in_portfolio[:] = False
                absent_days[:]  = 0
            continue

        # ── 计算排名（仅对可用股票，不可用 → rank = N+1）──────────────
        # 排名从 1 开始；1 = 最高分
        ranks = np.full(N, N + 1, dtype=np.int64)
        order = valid_idx[np.argsort(col[valid_idx])[::-1]]
        ranks[order] = np.arange(1, order.size + 1, dtype=np.int64)

        # ── [FIX-BUG1 + FIX-P0 + FIX-EXIT] 防抖：更新 in_portfolio 状态 ──
        hi_t = hi[:, t] if hi is not None else None
        fe_t = fe[:, t] if fe is not None else None
        for i in range(N):
            # [FIX-P0] 物理不可交易：冻结状态，跳过（权重=0由初始化保证）
            if hi_t is not None and hi_t[i]:
                # in_portfolio[i] 与 absent_days[i] 保持不变（冻结）
                continue

            # [FIX-EXIT] 策略主动卖出：立即清仓，绕过排名防抖
            if fe_t is not None and fe_t[i] and in_portfolio[i]:
                in_portfolio[i] = False
                absent_days[i]  = 0
                continue

            if not valid[i]:
                # 信号 -inf（门控失效）且非 hard_invalid → 旧逻辑：强制清仓
                in_portfolio[i] = False
                absent_days[i]  = 0
                continue

            rank_i = int(ranks[i])

            if rank_i <= top_n:
                # 进入核心区：入场 / 保持持仓，缺席计数归零
                in_portfolio[i] = True
                absent_days[i]  = 0
            elif rank_i <= top_n + exit_buffer:
                # 进入缓冲区：保持当前状态（在仓继续持，不在仓不新建）
                # 缺席计数归零（缓冲区内不算缺席）
                absent_days[i] = 0
                # in_portfolio 不变
            else:
                # 超出缓冲区：累计缺席
                if in_portfolio[i]:
                    absent_days[i] += 1
                    if absent_days[i] >= dropout_days:
                        # 连续缺席达阈值：清仓
                        in_portfolio[i] = False
                        absent_days[i]  = 0
                # 原本不在仓：保持不入场

        # ── 硬上限：总持仓数不超过 top_n + exit_buffer ─────────────────
        # [FIX-BUG1-CAP] 防止启动期新入场股票与缓冲区滞留股票叠加
        # 导致组合规模无限膨胀。若超限，按当日排名保留最靠前的股票，
        # 排名最差的超限股票提前清仓（相当于加速其出场流程）。
        active_idx = np.where(in_portfolio)[0]
        max_hold   = top_n + exit_buffer
        if active_idx.size > max_hold:
            # 按当日排名排序，保留排名最好的 max_hold 只
            active_ranks = ranks[active_idx]
            keep_order   = np.argsort(active_ranks)[:max_hold]
            keep_set     = set(active_idx[keep_order].tolist())
            for i in range(N):
                if in_portfolio[i] and i not in keep_set:
                    in_portfolio[i] = False
                    absent_days[i]  = 0
            active_idx = active_idx[keep_order]

        active = np.where(in_portfolio)[0]
        if active.size == 0:
            continue

        # [FIX-P0] 从权重分配中排除 hard_invalid 股票：
        # 冻结状态保留在 in_portfolio（下轮延续），但今日权重=0（物理无法买入）
        if hi_t is not None and hi_t.any():
            tradeable = active[~hi_t[active]]
        else:
            tradeable = active

        if tradeable.size == 0:
            continue

        k     = tradeable.size
        raw_w = 1.0 / k
        w_val = min(raw_w, max_single_pos)
        weights[tradeable, t] = w_val

        # 归一化（截断后列和可能 < 1，按比例保持，不强制拉伸到1）

    return weights


# ─────────────────────────────────────────────────────────────────────────────
# 验收测试
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    print("=" * 60)
    print("Q-UNITY V10  alpha_signal.py  验收测试")
    print("=" * 60)

    rng = np.random.default_rng(42)

    N, T = 50, 20        # 50只股票，20个时间步

    # ── 测试1：REGIME_IDX_TO_STR ────────────────────────────────
    assert set(REGIME_IDX_TO_STR.keys()) == {0, 1, 2, 3, 4}
    assert REGIME_IDX_TO_STR[0] == "STRONG_BULL"
    assert REGIME_IDX_TO_STR[4] == "BEAR"
    print("[OK] REGIME_IDX_TO_STR 键值正确 ✓")

    # ── 测试2：_score_to_weights 基本功能 ────────────────────────
    score = rng.standard_normal((N, T)).astype(np.float64)
    # 将若干位置设为 -inf（模拟不可用）
    score[::5, :] = -np.inf

    w = _score_to_weights(score, top_n=10, max_single_pos=0.08)

    assert w.shape == (N, T),  f"[FAIL] shape={w.shape}"
    assert w.dtype == np.float64, f"[FAIL] dtype={w.dtype}"
    assert np.all(w >= 0),     "[FAIL] 含负权重"
    assert np.all(np.isfinite(w)), "[FAIL] 含非有限值"
    assert np.all(w <= 0.08 + 1e-12), "[FAIL] 超过 max_single_pos=0.08"

    # -inf 位置权重必须为0
    neginf_mask = np.isneginf(score)
    assert np.all(w[neginf_mask] == 0.0), "[FAIL] -inf 位置权重不为零"

    # 每列非零股票数 <= top_n + exit_buffer（防抖允许持仓暂时超出 top_n，但有硬上限）
    # [FIX-BUG1] 原断言 <= top_n 不再成立：缓冲区内的持仓股可暂时令持仓数超出 top_n，
    # 但严格不超过 top_n + exit_buffer（默认 exit_buffer=5）。
    for t in range(T):
        n_held = (w[:, t] > 0).sum()
        assert n_held <= 10 + 5, f"[FAIL] t={t} 超过硬上限 top_n+exit_buffer={10+5}: {n_held}"

    print(f"[OK] _score_to_weights shape={w.shape} dtype={w.dtype} ✓")
    print(f"     每列最大权重={w.max():.4f}（上限0.08），"
          f"平均列和={w.sum(axis=0).mean():.4f} ✓")

    # ── 测试3：AlphaSignal 构造与 validate 通过 ─────────────────
    sig = AlphaSignal(
        raw_target_weights = w,
        score              = score,
        strategy_name      = "TestAlpha",
        meta               = {"version": "v10", "top_n": 10},
    )
    sig.validate(N, T)
    print(f"[OK] AlphaSignal.validate(N={N}, T={T}) 通过 ✓")
    print(f"     {repr(sig)}")

    # ── 测试4：validate 检测 shape 错误 ─────────────────────────
    bad_w = rng.random((N + 1, T)).astype(np.float64)
    sig_bad = AlphaSignal(raw_target_weights=bad_w)
    try:
        sig_bad.validate(N, T)
        print("[FAIL] 未检测到 shape 错误")
        sys.exit(1)
    except ValueError as e:
        print(f"[OK] shape 错误正确捕获: {e} ✓")

    # ── 测试5：validate 检测 dtype 错误 ─────────────────────────
    bad_dtype = rng.random((N, T)).astype(np.float32)
    sig_dtype = AlphaSignal(raw_target_weights=bad_dtype)
    try:
        sig_dtype.validate(N, T)
        print("[FAIL] 未检测到 dtype 错误")
        sys.exit(1)
    except ValueError as e:
        print(f"[OK] dtype 错误正确捕获: {e} ✓")

    # ── 测试6：validate 检测 NaN ────────────────────────────────
    nan_w = rng.random((N, T)).astype(np.float64)
    nan_w[3, 5] = np.nan
    sig_nan = AlphaSignal(raw_target_weights=nan_w)
    try:
        sig_nan.validate(N, T)
        print("[FAIL] 未检测到 NaN")
        sys.exit(1)
    except ValueError as e:
        print(f"[OK] NaN 错误正确捕获: {e} ✓")

    # ── 测试7：validate 检测负值 ────────────────────────────────
    neg_w = rng.random((N, T)).astype(np.float64)
    neg_w[0, 0] = -0.01
    sig_neg = AlphaSignal(raw_target_weights=neg_w)
    try:
        sig_neg.validate(N, T)
        print("[FAIL] 未检测到负值")
        sys.exit(1)
    except ValueError as e:
        print(f"[OK] 负值错误正确捕获: {e} ✓")

    # ── 测试8：score 中 +inf 被拒绝 ─────────────────────────────
    good_w2  = np.zeros((N, T), dtype=np.float64)
    bad_score = rng.standard_normal((N, T)).astype(np.float64)
    bad_score[1, 1] = np.inf
    sig_posinf = AlphaSignal(raw_target_weights=good_w2, score=bad_score)
    try:
        sig_posinf.validate(N, T)
        print("[FAIL] 未检测到 score +inf")
        sys.exit(1)
    except ValueError as e:
        print(f"[OK] score +inf 正确捕获: {e} ✓")

    # ── 测试9：score 中 -inf 被允许 ─────────────────────────────
    ok_score = rng.standard_normal((N, T)).astype(np.float64)
    ok_score[::3, :] = -np.inf
    sig_ok = AlphaSignal(raw_target_weights=good_w2, score=ok_score)
    sig_ok.validate(N, T)
    print("[OK] score 中 -inf 被允许 ✓")

    # ── 测试10：market_regime int8 数组模拟 ─────────────────────
    market_regime = np.array([0, 1, 2, 3, 4, 1, 0, 2], dtype=np.int8)
    labels = [REGIME_IDX_TO_STR[int(r)] for r in market_regime]
    assert labels[0] == "STRONG_BULL"
    assert labels[4] == "BEAR"
    print(f"[OK] market_regime(int8) → labels: {labels[:5]} ✓")

    print()
    print("[PASS] alpha_signal.py 全部验收通过 ✓")
    sys.exit(0)
