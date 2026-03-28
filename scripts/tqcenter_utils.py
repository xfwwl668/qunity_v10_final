"""
scripts/tqcenter_utils.py
==========================
TdxQuant (tqcenter) 路径解析工具 -- 从 config.json 读取，不写死路径。

所有需要使用 TdxQuant 的模块统一从这里获取路径，保证单一配置点。

用法：
    from scripts.tqcenter_utils import find_tqcenter, import_tq, get_tq

    # 方式1：只获取路径
    tq_path = find_tqcenter()   # 返回字符串路径，找不到返回 None

    # 方式2：直接 import tqcenter 模块
    tq_module = import_tq()     # 返回 tqcenter 模块，失败返回 None

    # 方式3：获取初始化好的 tq 对象（已调用 initialize）
    tq = get_tq(__file__)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── 项目根目录（从本文件反推）────────────────────────────────────────────────
_THIS_DIR    = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent if _THIS_DIR.name == "scripts" else _THIS_DIR

_CONFIG_PATH = _PROJECT_ROOT / "config.json"

# ── 默认搜索路径（config.json 不存在时的兜底）────────────────────────────────
_DEFAULT_SEARCH_PATHS = [
    r"D:\SOFT(DONE)\tdx\ncb\PYPlugins\user",
    r"C:\new_tdx\PYPlugins\user",
    r"D:\new_tdx\PYPlugins\user",
    r"E:\new_tdx\PYPlugins\user",
    r"C:\通达信\PYPlugins\user",
    r"D:\通达信\PYPlugins\user",
]

_cached_path: Optional[str] = None   # 缓存，避免重复 IO


def _load_config() -> dict:
    """读取 config.json，失败返回空 dict。"""
    try:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def find_tqcenter(config_path: str = None) -> Optional[str]:
    """
    按以下优先级查找 tqcenter.py 所在目录：

      1. config.json → tdxquant.tq_dir（用户明确指定，最高优先级）
      2. config.json → tdxquant.search_paths（配置的搜索列表）
      3. 内置默认搜索路径（兜底）

    找到则返回目录字符串，找不到返回 None。
    """
    global _cached_path

    if _cached_path is not None:
        return _cached_path

    cfg = _load_config()
    tq_cfg = cfg.get("tdxquant", {})

    # 1. 明确指定的路径
    explicit = tq_cfg.get("tq_dir", "").strip()
    if explicit and os.path.exists(os.path.join(explicit, "tqcenter.py")):
        _cached_path = explicit
        logger.info(f"[tqcenter_utils] 使用配置路径: {explicit}")
        return _cached_path

    # 2+3. 搜索路径列表（合并配置 + 默认）
    search_list = tq_cfg.get("search_paths", []) + _DEFAULT_SEARCH_PATHS
    for p in search_list:
        p = str(p).strip()
        if p and os.path.exists(os.path.join(p, "tqcenter.py")):
            _cached_path = p
            logger.info(f"[tqcenter_utils] 自动发现: {p}")
            return _cached_path

    logger.warning("[tqcenter_utils] 未找到 tqcenter.py，TdxQuant 不可用")
    return None


def import_tq():
    """
    导入 tqcenter 模块，返回模块对象。失败返回 None。

    会自动将 tqcenter 目录加入 sys.path。
    """
    tq_dir = find_tqcenter()
    if tq_dir is None:
        return None
    if tq_dir not in sys.path:
        sys.path.insert(0, tq_dir)
    try:
        import importlib
        return importlib.import_module("tqcenter")
    except Exception as e:
        logger.warning(f"[tqcenter_utils] import tqcenter 失败: {e}")
        return None


def get_tq(caller_file: str = None):
    """
    获取已初始化的 tq 对象（tqcenter.tq）。
    caller_file 传入 __file__ 即可。

    用法：
        from scripts.tqcenter_utils import get_tq
        tq = get_tq(__file__)
        if tq is None:
            print("TdxQuant 不可用")
    """
    mod = import_tq()
    if mod is None:
        return None
    try:
        tq_obj = mod.tq
        init_path = str(caller_file) if caller_file else __file__
        tq_obj.initialize(init_path)
        return tq_obj
    except Exception as e:
        logger.warning(f"[tqcenter_utils] tq.initialize 失败: {e}")
        return None


def is_available() -> bool:
    """快速检查 TdxQuant 是否可用（不初始化）。"""
    return find_tqcenter() is not None


def set_tq_dir(path: str) -> bool:
    """
    运行时临时设置 TdxQuant 目录（覆盖 config.json）。
    同时更新缓存，使后续调用立即生效。
    """
    global _cached_path
    if os.path.exists(os.path.join(path, "tqcenter.py")):
        _cached_path = path
        if path not in sys.path:
            sys.path.insert(0, path)
        logger.info(f"[tqcenter_utils] 手动设置路径: {path}")
        return True
    logger.warning(f"[tqcenter_utils] 路径无效（tqcenter.py 不存在）: {path}")
    return False


def reset_cache() -> None:
    """清除缓存，下次调用重新搜索（用于测试或路径变更后）。"""
    global _cached_path
    _cached_path = None


if __name__ == "__main__":
    # 简单自测
    print(f"config.json: {_CONFIG_PATH}")
    path = find_tqcenter()
    print(f"tqcenter 路径: {path or '未找到'}")
    print(f"TdxQuant 可用: {is_available()}")
