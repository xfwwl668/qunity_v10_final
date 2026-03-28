"""
adj_detector.py — Q-UNITY V9.1 除权日检测与复权验证

修复历史：
  V9.1 — 按板块动态阈值（创业板/科创板 20%，主板 10%，ST 5%，北交所 30%）
  V8.x — 固定 12% 阈值，无法区分创业板真实跌停与除权

板块涨跌幅限制：
  主板 (600/601/603/000/001/002/003) : ±10%（ST: ±5%）
  创业板 (300/301)                   : ±20%（ST: ±20%，上市首5日无限制）
  科创板 (688/689)                   : ±20%（ST: ±20%，上市首5日无限制）
  北交所 (8xxxxx/4xxxxx)             : ±30%
  新股上市前5日                       : 无限制（不做除权检测）
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 板块判断
# ─────────────────────────────────────────────────────────────────────────────

def get_board_limit(code: str, is_st: bool = False) -> float:
    """
    根据股票代码返回单日最大跌幅阈值（用于除权判断）。

    返回值：略高于涨跌幅限制，留出浮点误差空间。
    例：主板 10% → 返回 0.115，创业板 20% → 返回 0.225

    Parameters
    ----------
    code    : 股票代码（支持 "600519"、"sz.300750"、"sh.688981" 等格式）
    is_st   : 是否为ST股（影响主板阈值）
    """
    num = _extract_num(code)

    # 北交所（8xxxxx / 4xxxxx）
    if num.startswith("8") or num.startswith("4"):
        return 0.325  # ±30% + buffer

    # 科创板 688/689
    if num.startswith("688") or num.startswith("689"):
        return 0.225  # ±20% + buffer

    # 创业板 300/301
    if num.startswith("300") or num.startswith("301"):
        return 0.225  # ±20% + buffer

    # 主板：ST ±5%，普通 ±10%
    if is_st:
        return 0.058  # ±5% + buffer
    return 0.115      # ±10% + buffer


def is_gem_or_star(code: str) -> bool:
    """判断是否为创业板/科创板（享有20%涨跌幅）"""
    num = _extract_num(code)
    return (num.startswith("300") or num.startswith("301")
            or num.startswith("688") or num.startswith("689"))


def _extract_num(code: str) -> str:
    """从各种格式中提取纯数字代码"""
    code = code.strip()
    if "." in code:
        parts = code.split(".")
        for p in parts:
            if p.isdigit():
                return p.zfill(6)
    for prefix in ("sh", "sz", "bj", "sh.", "sz.", "bj."):
        if code.lower().startswith(prefix):
            rest = code[len(prefix):]
            if rest.isdigit():
                return rest.zfill(6)
    if code.isdigit():
        return code.zfill(6)
    return code


# ─────────────────────────────────────────────────────────────────────────────
# detect_ex_rights：检测疑似除权日
# ─────────────────────────────────────────────────────────────────────────────

def detect_ex_rights(
    close: np.ndarray,
    dates: List,
    threshold: float = 0.115,          # 默认主板阈值；建议调用方传入 get_board_limit()
    code: str = "",                     # V9.1 新增：用于自动确定板块阈值
    is_st: bool = False,
    volume: Optional[np.ndarray] = None,
    volume_amplify_ratio: float = 1.5,
    skip_first_days: int = 5,          # 新股上市前N日不做检测
    unadj_close: Optional[np.ndarray] = None,  # [BUG-1.5 FIX] 不复权收盘价（首选判据）
) -> List:
    """
    检测价格序列中疑似除权事件。

    V9.1 变更：
      - 若传入 code，自动用 get_board_limit(code, is_st) 覆盖 threshold
      - skip_first_days: 新股上市前5日无涨跌幅限制，跳过检测防误报
      - 创业板/科创板真实跌停 -20% 不再被误判为除权

    [BUG-1.5 FIX] unadj_close 判据（最高优先级）：
      - 真正的除权日特征：unadj_close 大幅跳降，而后复权 close 平稳
      - 真实暴跌特征：unadj_close 和 close 均大幅下跌
      - 若提供 unadj_close，优先用此判据而非成交量过滤（无量跌停是A股常态）
    """
    if code:
        threshold = get_board_limit(code, is_st)

    if len(close) < 2:
        return []

    close = np.asarray(close, dtype=np.float64)
    ex_dates: List = []
    vol_arr = np.asarray(volume, dtype=np.float64) if volume is not None else None
    unadj_arr = np.asarray(unadj_close, dtype=np.float64) if unadj_close is not None else None

    for i in range(max(1, skip_first_days), len(close)):
        prev = close[i - 1]
        curr = close[i]
        if np.isnan(prev) or np.isnan(curr) or prev <= 0:
            continue
        pct_change = (curr - prev) / prev
        if pct_change <= -threshold:
            # ── [BUG-1.5 FIX] unadj_close 优先判据 ──────────────────────
            # 除权日特征：unadj_close 大幅跳降，后复权 close 基本平稳
            # 真实暴跌特征：unadj_close 和后复权 close 均大幅下跌
            # 无量跌停（一字跌停）是 A 股常态，不能用成交量排除真实跌停
            if unadj_arr is not None and i < len(unadj_arr):
                u_prev = unadj_arr[i - 1]
                u_curr = unadj_arr[i]
                if not np.isnan(u_prev) and not np.isnan(u_curr) and u_prev > 0:
                    unadj_pct = (u_curr - u_prev) / u_prev
                    # 若不复权价格也大幅下跌（超过阈值的一半），则为真实行情而非除权
                    if unadj_pct <= -(threshold * 0.5):
                        continue  # 真实价格跌停，不是除权
                    # 否则：不复权价格平稳但复权价格大跌 → 典型除权日
                    ex_dates.append(dates[i])
                    continue
            # 无 unadj_close 时，退化为成交量过滤（兼容旧逻辑，但精度较低）
            if vol_arr is not None and i >= 5:
                recent_vols = vol_arr[max(0, i - 20):i]
                valid_vols = recent_vols[recent_vols > 0]
                if len(valid_vols) > 0:
                    avg_vol = np.mean(valid_vols)
                    if avg_vol > 1e-8:
                        vol_ratio = vol_arr[i] / avg_vol
                        if vol_ratio > volume_amplify_ratio:
                            # 放量大跌 → 真实行情，不是除权
                            continue
            ex_dates.append(dates[i])

    return ex_dates


# ─────────────────────────────────────────────────────────────────────────────
# mark_ex_rights_in_valid_mask
# ─────────────────────────────────────────────────────────────────────────────

def mark_ex_rights_in_valid_mask(
    valid_mask: np.ndarray,
    codes: List[str],
    dates: List[date],
    close_matrix: np.ndarray,
    window_days: int = 5,
    threshold: float = 0.115,          # 仅作默认值，实际按板块动态计算
) -> Tuple[np.ndarray, Dict[str, List[date]]]:
    """
    在 valid_mask 中标记所有股票的除权窗口（除权日前后 window_days 天）。
    V9.1: 每只股票独立使用板块阈值。
    """
    valid_mask = valid_mask.copy()
    date_index: Dict[date, int] = {d: i for i, d in enumerate(dates)}
    ex_rights_info: Dict[str, List[date]] = {}
    n, t = valid_mask.shape

    for i, code in enumerate(codes):
        ex_dates = detect_ex_rights(
            close=close_matrix[i],
            dates=dates,
            code=code,              # 自动按板块选阈值
        )
        if not ex_dates:
            continue

        ex_rights_info[code] = ex_dates
        for ex_date in ex_dates:
            if ex_date not in date_index:
                continue
            idx = date_index[ex_date]
            lo = max(0, idx - window_days)
            hi = min(t, idx + window_days + 1)
            valid_mask[i, lo:hi] = False

    return valid_mask, ex_rights_info


# ─────────────────────────────────────────────────────────────────────────────
# validate_adj_type：验证是否为后复权数据
# ─────────────────────────────────────────────────────────────────────────────

def validate_adj_type(
    close: np.ndarray,
    dates: List,
    code: str = "",                     # V9.1 新增：自动确定板块阈值
    is_st: bool = False,
    ex_rights_dates: Optional[List] = None,
    max_drop_threshold: float = 0.115,  # 默认主板；传入 code 时自动覆盖
) -> Tuple[bool, str]:
    """
    验证价格序列是否符合后复权（hfq）特征。

    V9.1 变更：
      - 若传入 code，自动用板块阈值替换 max_drop_threshold
      - 创业板/科创板最大单日跌幅阈值为 0.225（20% + buffer）
      - ST 主板阈值为 0.058（5% + buffer）
    """
    if code:
        max_drop_threshold = get_board_limit(code, is_st)

    close = np.asarray(close, dtype=np.float64)
    valid = close[~np.isnan(close)]

    if len(valid) < 20:
        return True, "数据点不足，跳过验证"

    returns = np.diff(valid) / valid[:-1]
    big_drops = int(np.sum(returns < -max_drop_threshold))

    board = _board_name(code)
    limit_pct = f"{(max_drop_threshold - 0.015) * 100:.0f}%"  # 还原实际限制

    if big_drops == 0:
        return True, f"[{board}] 价格序列无大幅跳降（阈值{limit_pct}），符合后复权特征"

    if big_drops >= 2:
        reason = (
            f"[{board}] 价格序列出现 {big_drops} 次跌幅 >{limit_pct} 的跳降，"
            "高度疑似前复权或原始未复权数据混入"
        )
        return False, reason

    # 1次大幅跳降：检查是否在已知除权日附近
    if ex_rights_dates:
        import datetime as _dt
        known_ex: set = set()
        for ex_d in ex_rights_dates:
            if isinstance(ex_d, str):
                try:
                    ex_d = _dt.date.fromisoformat(ex_d)
                except ValueError:
                    continue
            for delta in range(-3, 4):
                known_ex.add(ex_d + _dt.timedelta(days=delta))

        close_arr = close[~np.isnan(close)]
        dates_arr = [d for d, c in zip(dates, close) if not np.isnan(c)]
        anomaly_dates = [
            dates_arr[i]
            for i in range(1, len(close_arr))
            if (close_arr[i] - close_arr[i - 1]) / close_arr[i - 1] < -max_drop_threshold
        ]
        unexplained = [d for d in anomaly_dates if d not in known_ex]
        if unexplained:
            return False, (
                f"[{board}] 在非已知除权日出现跳降：{unexplained[:3]}，"
                "疑似数据为前复权或混入原始价格"
            )
        return True, f"[{board}] 跳降发生在已知除权日附近，属正常原始数据特征"

    return False, (
        f"[{board}] 出现 1 次跌幅 >{limit_pct} 的跳降，"
        "无法确认是否为真实行情，建议人工核查"
    )


def _board_name(code: str) -> str:
    num = _extract_num(code)
    if num.startswith("688") or num.startswith("689"):
        return "科创板±20%"
    if num.startswith("300") or num.startswith("301"):
        return "创业板±20%"
    if num.startswith("8") or num.startswith("4"):
        return "北交所±30%"
    return "主板±10%"
