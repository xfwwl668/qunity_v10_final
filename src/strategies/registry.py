"""
Q-UNITY V10 — src/strategies/registry.py
==========================================
策略注册表（含自动发现）

VEC_STRATEGY_REGISTRY : Dict[str, Callable]
    全局策略注册表。键为策略名称（str），值为策略函数。

register_vec_strategy(name) : decorator
    注册策略的装饰器工厂。

get_alpha_fn(name) : Callable
    按名称取策略函数，未注册则抛 KeyError（含已注册名单）。

list_vec_strategies() : List[str]
    返回所有已注册策略名称列表。

铁律（每次修改前默读）
----------------------
1. stamp_tax = 0.0005（万五，绝不手改）
2. market_regime 是 int8 数组；REGIME_IDX_TO_STR 映射后才能查 FACTOR_WEIGHTS

修复记录
--------
[FIX-S-01] 新增 _auto_discover()：扫描 vectorized/*.py 并 import，
           触发 @register_vec_strategy 装饰器注册，解决注册表启动时为空的问题。
"""
from __future__ import annotations

import importlib
import logging
import pathlib
import warnings
from typing import Callable, Dict, List

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 全局注册表
# ─────────────────────────────────────────────────────────────────────────────

VEC_STRATEGY_REGISTRY: Dict[str, Callable] = {}


# ─────────────────────────────────────────────────────────────────────────────
# 注册装饰器
# ─────────────────────────────────────────────────────────────────────────────

def register_vec_strategy(name: str) -> Callable:
    """
    策略注册装饰器工厂。

    用法
    ----
    @register_vec_strategy("my_strategy")
    def my_strategy_alpha(...) -> AlphaSignal:
        ...
    """
    def decorator(fn: Callable) -> Callable:
        VEC_STRATEGY_REGISTRY[name] = fn
        return fn
    return decorator


def register(name: str) -> Callable:
    """register 是 register_vec_strategy 的简写别名。"""
    return register_vec_strategy(name)


def get_alpha_fn(name: str) -> Callable:
    """
    按名称取策略函数，未注册则抛 KeyError。
    """
    if name not in VEC_STRATEGY_REGISTRY:
        available = list(VEC_STRATEGY_REGISTRY.keys())
        raise KeyError(
            f"策略 '{name}' 未注册。\n"
            f"已注册策略: {available}\n"
            f"提示: 确认 @register_vec_strategy('{name}') 装饰器已生效。"
        )
    return VEC_STRATEGY_REGISTRY[name]


def list_vec_strategies() -> List[str]:
    """返回所有已注册策略名称列表（字典序）。"""
    return sorted(VEC_STRATEGY_REGISTRY.keys())


# ─────────────────────────────────────────────────────────────────────────────
# [FIX-S-01] 自动发现：扫描 vectorized/*.py 触发装饰器注册
# ─────────────────────────────────────────────────────────────────────────────

_STRATEGIES_DIR = pathlib.Path(__file__).parent / "vectorized"
_AUTO_DISCOVERED = False


def _auto_discover() -> None:
    """
    扫描 src/strategies/vectorized/*.py 并逐一 import，
    触发模块内的 @register_vec_strategy 装饰器完成注册。

    只执行一次（_AUTO_DISCOVERED 标志防止重复扫描）。
    加载失败的模块只发出 warning，不中断其他策略的注册。
    """
    global _AUTO_DISCOVERED
    if _AUTO_DISCOVERED:
        return
    _AUTO_DISCOVERED = True

    if not _STRATEGIES_DIR.exists():
        logger.warning(f"[registry] 策略目录不存在: {_STRATEGIES_DIR}")
        return

    for path in sorted(_STRATEGIES_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue
        # 尝试两种 import 路径（支持 src.strategies.* 和 flat 模式）
        for module_name in [
            f"src.strategies.vectorized.{path.stem}",
            f"strategies.vectorized.{path.stem}",
        ]:
            try:
                importlib.import_module(module_name)
                break
            except ModuleNotFoundError:
                continue
            except Exception as exc:
                warnings.warn(
                    f"[registry] 策略模块加载失败 {path.name}: {exc}",
                    stacklevel=2,
                )
                break


# 模块首次 import 时自动执行发现
_auto_discover()
