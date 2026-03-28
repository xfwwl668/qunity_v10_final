"""
debug_backtest.py — Q-UNITY V10 回测诊断工具
=============================================
在回测完成后调用，输出：
  1. Regime 分布（牛/熊/中性各占多少天）
  2. 权重矩阵活跃天 vs 实际持仓天 → 找门控损耗
  3. 按 Regime 分组的收益贡献
  4. 空仓期市场涨跌（确认空仓是否真的保护了资金）
  5. 换手率分解（进场 vs 出场 vs 再平衡各贡献多少）

用法：
    from scripts.debug_backtest import run_debug
    run_debug(runner, strategy_name='alpha_max_v5')
"""
from __future__ import annotations
import sys
import numpy as np
from pathlib import Path

# 兼容直接运行和 import
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.strategies.alpha_signal import REGIME_IDX_TO_STR
except ImportError:
    REGIME_IDX_TO_STR = {0:'STRONG_BULL',1:'BULL',2:'NEUTRAL',3:'SOFT_BEAR',4:'BEAR'}


def run_debug(runner, strategy_name: str = '') -> None:
    """
    在 runner.run_backtest() 之后调用，输出完整诊断。

    Parameters
    ----------
    runner : FastRunnerV10 实例（已完成一次回测）
    strategy_name : 策略名称（用于标题）
    """
    # ── 取出必要数据 ──────────────────────────────────────────────────────
    pos     = getattr(runner, '_last_pos_matrix',   None)  # (N, T)
    nav     = getattr(runner, '_last_nav_array',    None)  # (T,)
    regime  = getattr(runner, '_last_regime_bt',    None)  # (T,) int8
    dates   = getattr(runner, '_last_period_dates', None)  # list[str]
    pb      = getattr(runner, '_port_builder',      None)

    if pos is None or nav is None:
        print('[debug_backtest] 无法获取回测矩阵，请先运行 run_backtest()')
        return

    T = nav.shape[0]
    N = pos.shape[0]
    dates_short = [str(d)[:10] for d in dates] if dates else [str(i) for i in range(T)]

    title = f'【诊断报告】{strategy_name}  N={N}  T={T}天'
    print()
    print('=' * 65)
    print(f'  {title}')
    print('=' * 65)

    # ── 1. Regime 分布 ───────────────────────────────────────────────────
    print()
    print('── 1. Regime 分布 ──────────────────────────────────────────')
    if regime is not None:
        from collections import Counter
        dist = Counter(int(r) for r in regime)
        regime_limits = pb._regime_limits if pb is not None else None
        for idx in range(5):
            cnt   = dist.get(idx, 0)
            name  = REGIME_IDX_TO_STR.get(idx, str(idx))
            bar   = '█' * int(cnt / T * 40)
            limit = {0:1.0, 1:0.8, 2:0.8, 3:0.4, 4:0.0}.get(idx, '?')
            print(f'  {name:12s} {cnt:4d}天 ({cnt/T*100:5.1f}%)  仓位上限={limit}  {bar}')

        bear_days  = dist.get(4, 0)
        sbear_days = dist.get(3, 0)
        hold_days_regime = T - bear_days  # BEAR以外都可入场
        print(f'  {"合计可入场":12s} {hold_days_regime:4d}天 ({hold_days_regime/T*100:5.1f}%)  (非BEAR)')
    else:
        print('  [SKIP] regime 数组不可用')

    # ── 2. 权重活跃天 vs 持仓天 ──────────────────────────────────────────
    print()
    print('── 2. 权重→持仓 漏斗分析 ──────────────────────────────────')
    actual_hold = int((pos.sum(axis=0) > 1e-3).sum())
    print(f'  实际持仓天数   : {actual_hold:4d} / {T} ({actual_hold/T*100:.1f}%)')
    if regime is not None:
        non_bear = int((regime != 4).sum())
        print(f'  Regime可入场天 : {non_bear:4d} / {T} ({non_bear/T*100:.1f}%)')
        if non_bear > 0:
            leakage = non_bear - actual_hold
            print(f'  策略门控损耗   : {leakage:4d}天  ({leakage/T*100:.1f}%)  ← 非BEAR但未持仓的天数')
            print(f'  有效利用率     : {actual_hold/non_bear*100:.1f}%  (持仓/可入场)')

    # ── 3. Regime分组收益 ─────────────────────────────────────────────────
    print()
    print('── 3. 按 Regime 分组收益贡献 ───────────────────────────────')
    if regime is not None and len(nav) > 1:
        daily_ret = np.diff(nav) / (nav[:-1] + 1e-10)
        # regime[t] 对应 nav[t]→nav[t+1]的收益
        for idx in range(5):
            mask = (regime[:len(daily_ret)] == idx)
            if mask.sum() == 0:
                continue
            rets  = daily_ret[mask]
            cum_r = float(np.prod(1 + rets) - 1) * 100
            avg_r = float(np.mean(rets)) * 100
            name  = REGIME_IDX_TO_STR.get(idx, str(idx))
            is_invested = idx in (0, 1, 2, 3)  # SOFT_BEAR 半仓也参与
            flag = '持仓' if is_invested else '空仓'
            print(f'  {name:12s} ({flag}) {mask.sum():4d}天  '
                  f'累计{cum_r:+7.2f}%  日均{avg_r:+.3f}%')

    # ── 4. 空仓期市场涨跌（评估空仓是否有效保护） ─────────────────────────
    print()
    print('── 4. 空仓期市场涨跌（空仓是否保护了资金）──────────────────')
    if regime is not None and len(nav) > 1:
        daily_ret = np.diff(nav) / (nav[:-1] + 1e-10)
        bear_mask = (regime[:len(daily_ret)] == 4)
        if bear_mask.sum() > 0:
            # 用全仓等权买入作为基准（若可以）
            close_bt = getattr(runner, '_data', {}).get('close')
            if close_bt is not None:
                # 简单代理：用所有valid股票的等权收益作为市场基准
                valid_m = getattr(runner, '_data', {}).get('valid_mask')
                if valid_m is not None and close_bt.shape[1] >= T:
                    c  = close_bt[:, -T:].astype(np.float64)
                    vm = valid_m[:, -T:].astype(np.bool_)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        dr_mkt = np.where(
                            vm[:, :-1] & vm[:, 1:] & (c[:, :-1] > 0),
                            (c[:, 1:] - c[:, :-1]) / c[:, :-1], np.nan)
                    mkt_daily = np.nanmean(dr_mkt, axis=0)  # (T-1,)

                    bear_mkt = mkt_daily[bear_mask[:len(mkt_daily)]]
                    cum_bear_mkt = float(np.nanprod(1 + bear_mkt) - 1) * 100
                    avg_bear_mkt = float(np.nanmean(bear_mkt)) * 100
                    print(f'  BEAR期市场等权收益: 累计{cum_bear_mkt:+.1f}%  日均{avg_bear_mkt:+.3f}%')
                    if cum_bear_mkt < -5:
                        print(f'  ✓ 空仓确实规避了亏损（市场在BEAR期下跌{-cum_bear_mkt:.1f}%）')
                    elif cum_bear_mkt > 5:
                        print(f'  ⚠ 空仓错过了上涨（市场在BEAR期上涨{cum_bear_mkt:.1f}%）→ Regime参数可能过于保守')
                    else:
                        print(f'  ≈ 空仓期市场基本横盘，空仓效果中性')

    # ── 5. 换手率分解 ─────────────────────────────────────────────────────
    print()
    print('── 5. 换手率分解 ────────────────────────────────────────────')
    # 优先从 RunResult 直接取（更准确），否则从 pos_matrix 重算
    buy_cnt  = getattr(runner, '_last_result_buy_count',  None)
    sell_cnt = getattr(runner, '_last_result_sell_count', None)
    final_pos= getattr(runner, '_last_result_final_pos',  None)

    if buy_cnt is None:
        _pos_diff = np.diff(pos, axis=1)
        buy_cnt   = int((_pos_diff > 1e-3).sum())
        sell_cnt  = int((_pos_diff < -1e-3).sum())
        final_pos = int((pos[:, -1] > 1e-3).sum())

    total_trades = buy_cnt + sell_cnt
    print(f'  买入操作次数 : {buy_cnt:6,}')
    print(f'  卖出操作次数 : {sell_cnt:6,}')
    print(f'  合计交易次数 : {total_trades:6,}')
    print(f'  期末持仓股数 : {final_pos:6,} 只')
    if total_trades > 0:
        buy_pct = buy_cnt / total_trades * 100
        sell_pct = sell_cnt / total_trades * 100
        print(f'  买入占比     : {buy_pct:.1f}%   卖出占比: {sell_pct:.1f}%')
        if abs(buy_pct - sell_pct) > 20:
            side = '偏买入（仓位持续堆积）' if buy_pct > sell_pct else '偏卖出（单向漏水）'
            print(f'  ⚠ 买卖严重不对称（{side}）→ 检查L3-C对称threshold')
    close_bt = getattr(runner, '_data', {}).get('close')
    if close_bt is not None and len(nav) > 1 and close_bt.shape[1] >= T:
        _pd = np.diff(pos, axis=1)
        c   = close_bt[:, -T:].astype(np.float64)
        years = T / 252.0
        traded_val = float(np.abs(_pd * c[:, -T+1:]).sum())
        ann_to = traded_val / float(np.mean(nav)) / years * 100
        print(f'  年换手率估算 : {ann_to:,.0f}%/年')

    # ── 6. 连续空仓段分析 ──────────────────────────────────────────────────
    print()
    print('── 6. 最长连续空仓段 ────────────────────────────────────────')
    invested_mask = (pos.sum(axis=0) > 1e-3)
    max_gap = 0
    max_gap_start = 0
    cur_gap = 0
    cur_start = 0
    for t in range(T):
        if not invested_mask[t]:
            if cur_gap == 0:
                cur_start = t
            cur_gap += 1
            if cur_gap > max_gap:
                max_gap = cur_gap
                max_gap_start = cur_start
        else:
            cur_gap = 0
    if max_gap > 0:
        s = dates_short[max_gap_start]
        e = dates_short[min(max_gap_start + max_gap - 1, T - 1)]
        print(f'  最长空仓段: {max_gap}天  ({s} → {e})')
        if max_gap > 250:
            print(f'  ⚠ 超过250天连续空仓 → 极可能是 stop_recovery=252天 棘轮触发！')
        elif max_gap > 60:
            print(f'  ⚠ 超过60天连续空仓 → 建议检查 stop_recovery_days 配置')
        else:
            print(f'  ✓ 空仓段长度合理（< 60天）')

    print()
    print('=' * 65)
    print('  诊断完毕。如需修复：')
    print('  · 空仓时间过多 → 调高 bear_breadth_thr (当前0.25建议0.30~0.35)')
    print('  · 连续空仓>250天 → stop_recovery_days 仍为252（检查config.json）')
    print('  · 换手率过高 → 增大 rebalance_threshold 或调低 top_n')
    print('  · 策略门控损耗大 → 检查策略内部的门控条件阈值')
    print('=' * 65)


if __name__ == '__main__':
    print('run_debug(runner, strategy_name=...) 在回测后调用')
