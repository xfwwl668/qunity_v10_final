# 根据 Q-UNITY V8 计划书 v2.1 实现
"""
Q-UNITY V8.0 数据层核心数据类

计划书 §5.1 数据适配层接口定义
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# MatrixBundle — 日线 (N, T) 矩阵容器
# ---------------------------------------------------------------------------

@dataclass
class MatrixBundle:
    """
    日线数据的核心容器，以列优先 (N, T) 矩阵存储 OHLCV 字段。

    Attributes
    ----------
    codes       : 股票代码列表，长度 N
    dates       : 交易日列表，长度 T
    open        : 开盘价矩阵 (N, T) float32
    high        : 最高价矩阵 (N, T) float32
    low         : 最低价矩阵 (N, T) float32
    close       : 收盘价矩阵 (N, T) float32
    volume      : 成交量矩阵 (N, T) float32
    valid_mask  : 有效性掩码矩阵 (N, T) bool
                  False = 停牌 / NB-21新股保护窗口 / 复权疑似异常
    adj_type    : 复权类型标记，"qfq"(V10默认) | "hfq" | "raw"
    """

    codes: List[str]
    dates: List[date]
    open: np.ndarray       # (N, T) float32
    high: np.ndarray       # (N, T) float32
    low: np.ndarray        # (N, T) float32
    close: np.ndarray      # (N, T) float32
    volume: np.ndarray     # (N, T) float32
    valid_mask: np.ndarray # (N, T) bool
    adj_type: str = "qfq"  # V10 QFQ
    # Q-UNITY V8 ULTIMATE: 同花顺概念 ID 矩阵
    # concept_ids[i, t] = 个股 i 在 t 日所属的主概念 uint16 ID（0=未知/无概念）
    concept_ids: Optional[np.ndarray] = None  # (N, T) uint16

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        n = len(self.codes)
        t = len(self.dates)
        for name, arr in (
            ("open", self.open),
            ("high", self.high),
            ("low", self.low),
            ("close", self.close),
            ("volume", self.volume),
        ):
            if arr.shape != (n, t):
                raise ValueError(
                    f"MatrixBundle.{name} shape {arr.shape} != ({n}, {t})"
                )
        if self.valid_mask.shape != (n, t):
            raise ValueError(
                f"MatrixBundle.valid_mask shape {self.valid_mask.shape} != ({n}, {t})"
            )
        # 验证 concept_ids（可选字段）
        if self.concept_ids is not None:
            if self.concept_ids.shape != (n, t):
                raise ValueError(
                    f"MatrixBundle.concept_ids shape {self.concept_ids.shape} != ({n}, {t})"
                )
            if self.concept_ids.dtype != np.uint16:
                self.concept_ids = self.concept_ids.astype(np.uint16)

    # ------------------------------------------------------------------
    @property
    def n_stocks(self) -> int:
        """股票数量 N"""
        return len(self.codes)

    @property
    def n_days(self) -> int:
        """交易日数量 T"""
        return len(self.dates)

    def slice_dates(
        self,
        start: date,
        end: date,
    ) -> "MatrixBundle":
        """按日期范围切片，返回新 MatrixBundle"""
        idx = [i for i, d in enumerate(self.dates) if start <= d <= end]
        if not idx:
            raise ValueError(f"No dates in [{start}, {end}]")
        s, e = idx[0], idx[-1] + 1
        return MatrixBundle(
            codes=self.codes,
            dates=self.dates[s:e],
            open=self.open[:, s:e],
            high=self.high[:, s:e],
            low=self.low[:, s:e],
            close=self.close[:, s:e],
            volume=self.volume[:, s:e],
            valid_mask=self.valid_mask[:, s:e],
            adj_type=self.adj_type,
            concept_ids=self.concept_ids[:, s:e] if self.concept_ids is not None else None,
        )


# ---------------------------------------------------------------------------
# MemMapMeta — npy 文件元数据
# ---------------------------------------------------------------------------

@dataclass
class MemMapMeta:
    """
    描述一组 .npy memmap 文件的元数据，对应计划书 §5.1。

    存储于 data/npy/meta.json 或 data/npy_minute/meta.json。

    Attributes
    ----------
    npy_dir     : npy 文件所在目录的绝对路径
    codes       : 股票代码列表（长度 N）
    dates       : 交易日列表（长度 T），对分钟数据为日列表
    shape       : (N, T) 或 (N, D, M)
    fields      : 已写出的字段列表，例如 ["close","open","high","low","volume"]
    dtype       : numpy dtype 字符串，"float32"
    adj_type    : 复权类型
    build_time  : ISO 格式构建时间戳
    sha256      : 各字段文件的 SHA-256 校验（Dict[field, hexstr]）
    """

    npy_dir: str
    codes: List[str]
    dates: List[str]          # ISO 格式日期字符串
    shape: Tuple[int, ...]
    fields: List[str]
    dtype: str = "float32"
    adj_type: str = "qfq"  # V10 QFQ
    build_time: str = ""
    sha256: Dict[str, str] = field(default_factory=dict)
    extra: Dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def save(self, path: Optional[Path] = None) -> Path:
        """序列化为 meta.json"""
        if path is None:
            path = Path(self.npy_dir) / "meta.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        obj = {
            "npy_dir": self.npy_dir,
            "codes": self.codes,
            "dates": self.dates,
            "shape": list(self.shape),
            "fields": self.fields,
            "dtype": self.dtype,
            "adj_type": self.adj_type,
            "build_time": self.build_time,
            "sha256": self.sha256,
            "extra": self.extra,
        }
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> "MemMapMeta":
        """从 meta.json 反序列化"""
        obj = json.loads(path.read_text())
        # [V10-FIX] 兼容旧版 meta.json，如果缺少 npy_dir，则以 meta.json 所在目录为准
        if "npy_dir" not in obj:
            obj["npy_dir"] = str(path.parent.resolve())
        obj["shape"] = tuple(obj["shape"])
        return cls(**obj)

    def get_array_path(self, field_name: str) -> Path:
        """返回指定字段的 .npy 文件路径"""
        return Path(self.npy_dir) / f"{field_name}.npy"



