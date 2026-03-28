"""
Q-UNITY V10 — scripts/utils_paths.py
=====================================
统一路径读取工具函数，所有脚本通过此模块获取 npy 目录，
避免各脚本独立硬编码导致 V10 pipeline 断链（C-02修复）。
"""
from __future__ import annotations
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def get_npy_dir(version: str = "v10") -> Path:
    """
    从 config.json 读取 npy 目录路径。

    Parameters
    ----------
    version : "v10"（默认）→ npy_v10_dir / "data/npy_v10"
              "v91" → npy_dir / "data/npy"

    Returns
    -------
    Path 对象（可能不存在，由调用方创建）
    """
    cfg_path = PROJECT_ROOT / "config.json"
    if cfg_path.exists():
        try:
            with open(cfg_path, encoding="utf-8") as f:
                c = json.load(f)
            if version == "v10":
                raw = c.get("npy_v10_dir") or c.get("data", {}).get("npy_dir", "")
            else:
                raw = c.get("npy_dir") or c.get("data", {}).get("npy_dir", "data/npy")
            if raw:
                p = Path(raw)
                return p if p.is_absolute() else PROJECT_ROOT / p
        except Exception:
            pass
    # Fallback
    return PROJECT_ROOT / ("data/npy_v10" if version == "v10" else "data/npy")


def get_parquet_dir(adj: str = "hfq") -> Path:
    """
    从 config.json 读取 parquet 目录路径。
    adj: "hfq"（默认，V10）或 "qfq"（备用）
    """
    cfg_path = PROJECT_ROOT / "config.json"
    if cfg_path.exists():
        try:
            with open(cfg_path, encoding="utf-8") as f:
                c = json.load(f)
            if adj == "hfq":
                raw = c.get("parquet_dir_hfq") or c.get("data", {}).get("parquet_dir", "")
            else:
                raw = c.get("parquet_dir_qfq") or c.get("data", {}).get("parquet_dir", "data/daily_parquet_qfq")
            if raw:
                p = Path(raw)
                return p if p.is_absolute() else PROJECT_ROOT / p
        except Exception:
            pass
    return PROJECT_ROOT / ("data/daily_parquet_hfq" if adj == "hfq" else "data/daily_parquet_qfq")
