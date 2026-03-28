# 根据 Q-UNITY V8 计划书 v2.1 实现
"""
复权验证钩子 — 在数据写入 Parquet 之前执行
对应计划书 §11.6.2

调用方式（追加到 pipeline.py 的 save 流程后）：

    from src.data.adj_validator_hook import AdjValidatorHook
    hook = AdjValidatorHook(config)
    df = hook.validate_and_tag(df, code, source="baostock")
    if df is None:
        # 数据被拒绝，跳过写入
        continue

功能：
    - 标注 adj_type / adj_source / adj_factor 字段
    - 检测前复权混入（基于价格跳变检测）
    - 记录除权事件到日志
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AdjValidatorHook:
    """
    数据写入前的复权类型验证与标注钩子。

    在 StockDataPipeline.save_to_parquet() 之前调用，用于：
        1. 为每条写入记录标注 adj_type / adj_source / adj_factor 字段
        2. 检测疑似前复权混入（后复权数据不应出现大幅价格跳降）
        3. 依据 config["data_adj_policy"]["daily"]["reject_qfq"] 决定是否拒绝

    Parameters
    ----------
    config : 与 config.json 对应的配置字典，需含 "data_adj_policy" 节
    """

    VALID_SOURCES = frozenset({"baostock", "akshare", "tdx_converted"})

    def __init__(self, config: dict) -> None:
        policy = config.get("data_adj_policy", {})
        daily_policy = policy.get("daily", {})
        self._reject_qfq: bool = daily_policy.get("reject_qfq", True)
        self._threshold: float = daily_policy.get("ex_rights_threshold", 0.12)
        self._auto_detect: bool = daily_policy.get("auto_detect_ex_rights", True)

    # ------------------------------------------------------------------
    def validate_and_tag(
        self,
        df: pd.DataFrame,
        code: str,
        source: str,
        declared_adj_type: str = "qfq"  # V10 QFQ,
    ) -> Optional[pd.DataFrame]:
        """
        验证并标注复权信息。

        Parameters
        ----------
        df                  : 待写入的 OHLCV DataFrame，需含 "close" 和 "date" 列
        code                : 股票代码（仅用于日志）
        source              : 数据来源，"baostock" | "akshare" | "tdx_converted"
        declared_adj_type   : 来源声明的复权类型，"qfq"(V10) | "hfq" | "raw"

        Returns
        -------
        标注后的 DataFrame（含 adj_type / adj_source / adj_factor 列），
        若验证失败且 reject_qfq=True 则返回 None（调用方应跳过写入）。
        """
        if source not in self.VALID_SOURCES:
            logger.warning(f"[AdjHook] {code} 未知数据源: {source}，允许通过但记录警告")

        # 复制并标注基础字段
        df = df.copy()
        df["adj_type"] = declared_adj_type
        df["adj_source"] = source
        df["adj_factor"] = 1.0  # 默认；后续可由 adj_converter 更新

        if not self._auto_detect:
            return df

        # 仅对 "hfq" 声明的数据进行前复权混入检测
        if declared_adj_type != "qfq":  # V10 QFQ
            logger.debug(
                f"[AdjHook] {code} declared_adj_type={declared_adj_type}，跳过后复权验证"
            )
            return df

        if "close" not in df.columns or len(df) < 20:
            return df

        from src.data.adj_detector import detect_ex_rights, validate_adj_type

        close_arr = pd.to_numeric(df["close"], errors="coerce").values
        dates = df["date"].tolist() if "date" in df.columns else list(range(len(df)))

        # 快速预检：detect_ex_rights 发现跳降
        ex_dates = detect_ex_rights(close_arr, dates, threshold=self._threshold)

        if ex_dates:
            # 精确验证：validate_adj_type 综合判断
            is_hfq, reason = validate_adj_type(
                close=close_arr,
                dates=dates,
                ex_rights_dates=None,  # 无先验除权日，完全自动
                max_drop_threshold=self._threshold,
            )

            if not is_hfq:
                msg = (
                    f"{code} 数据疑似含前复权或原始未复权数据: {reason}，"
                    f"疑似除权日: {ex_dates[:3]}{'...' if len(ex_dates) > 3 else ''}"
                )
                # Bug 21 Fix: When reject_qfq=False, tag as qfq_suspect and STILL return df
                # (original code also returned df, but be explicit about the flow)
                if self._reject_qfq:
                    logger.error(f"[AdjHook] {msg}，数据已拒绝写入 (reject_qfq=True)")
                    return None
                else:
                    logger.warning(f"[AdjHook] {msg}，数据已写入但标记为 adj_type=qfq_suspect")
                    df["adj_type"] = "qfq_suspect"
                    return df  # Bug 21: Explicitly return tagged df instead of falling through
            else:
                logger.debug(f"[AdjHook] {code} 通过后复权验证: {reason}")

        return df



