"""
★[FIX-T6] src/engine/alpha_signal.py — 重定向模块

历史上此文件与 src/strategies/alpha_signal.py 存在分叉。
fast_runner_v10.py 实际导入的是 strategies 版本（含 hard_invalid、
force_exit、_ema_smooth_factor 等扩展），engine 版本是功能不完整的孤儿。

修复：本文件改为纯重定向，所有导入统一指向 strategies 版本，
消除两份文件不同步的隐患。

★ 请勿在此文件添加任何业务逻辑 ★
"""
from src.strategies.alpha_signal import (   # noqa: F401  re-export everything
    AlphaSignal,
    REGIME_IDX_TO_STR,
    _score_to_weights,
    _ema_smooth_factor,
)

__all__ = [
    "AlphaSignal",
    "REGIME_IDX_TO_STR",
    "_score_to_weights",
    "_ema_smooth_factor",
]
