# 根据 Q-UNITY V8 计划书 v2.1 实现
"""
TDX 前复权数据转换为后复权数据
对应计划书 §11.4 TDX数据后复权转换

背景：
    TDX（通达信）默认提供前复权（qfq）数据，而 Q-UNITY V8 统一使用后复权（hfq）。
    本模块提供：
        1. convert_qfq_to_hfq()          — 通过复权因子进行价格转换
        2. fetch_adj_factor_from_baostock() — 从 BaoStock 获取复权因子序列

复权公式（后复权）：
    hfq_price(t) = raw_price(t) × adj_factor(t)

其中 adj_factor(t) 为累积复权因子（BaoStock 提供，从上市日起始，
每次除权后因子发生跳变），保证最早的历史价格最大、最新价格接近原始价格。

注意：
    - BaoStock 的 adjustflag=1 对应前复权，adjustflag=2 对应后复权（直接使用）
    - 本模块的 convert_qfq_to_hfq() 用于当 BaoStock 不可用时，
      对 TDX 前复权价格进行补救转换（精度略低于 BaoStock 直接采集）
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# fetch_adj_factor_from_baostock
# ---------------------------------------------------------------------------

def fetch_adj_factor_from_baostock(
    code: str,
    start_date: str,
    end_date: str,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    从 BaoStock 获取指定股票的后复权因子序列。

    Parameters
    ----------
    code        : 股票代码，格式 "sh.600000" 或 "sz.000001"
                  若传入 "600000"/"000001" 会自动转换为 BaoStock 格式
    start_date  : 开始日期，"YYYY-MM-DD"
    end_date    : 结束日期，"YYYY-MM-DD"
    max_retries : BaoStock 请求最大重试次数

    Returns
    -------
    DataFrame，列 ["date", "adj_factor"]，按日期升序排列。
    date 为 date 对象，adj_factor 为 float64。

    Raises
    ------
    RuntimeError : BaoStock 登录失败或数据拉取失败（超过重试次数）
    ImportError  : baostock 未安装

    Notes
    -----
    BaoStock 接口：bs.query_adjust_factor(code, start_date, end_date)
    返回字段：adjustDate, foreAdjustFactor, backAdjustFactor
    本函数使用 backAdjustFactor（后复权因子）。
    """
    try:
        import baostock as bs
    except ImportError as e:
        raise ImportError(
            "baostock 未安装，请运行: pip install baostock"
        ) from e

    # 规范化股票代码
    code_bs = _normalize_code_baostock(code)

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            lg = bs.login()
            if lg.error_code != "0":
                raise RuntimeError(
                    f"BaoStock 登录失败: {lg.error_msg}"
                )
            try:
                rs = bs.query_adjust_factor(
                    code=code_bs,
                    start_date=start_date,
                    end_date=end_date,
                )
                if rs.error_code != "0":
                    raise RuntimeError(
                        f"BaoStock 查询失败 [{code_bs}]: {rs.error_msg}"
                    )
                rows: List[List[str]] = []
                while rs.next():
                    rows.append(rs.get_row_data())

                if not rows:
                    logger.warning(
                        f"[AdjConverter] {code_bs} 无复权因子数据 "
                        f"[{start_date}, {end_date}]"
                    )
                    return pd.DataFrame(
                        columns=["date", "adj_factor"]
                    )

                df = pd.DataFrame(
                    rows,
                    columns=rs.fields,
                )
                df = df.rename(
                    columns={
                        "adjustDate": "date",
                        "backAdjustFactor": "adj_factor",
                    }
                )
                df["date"] = pd.to_datetime(df["date"]).dt.date
                df["adj_factor"] = pd.to_numeric(
                    df["adj_factor"], errors="coerce"
                )
                df = df[["date", "adj_factor"]].dropna()
                df = df.sort_values("date").reset_index(drop=True)
                logger.info(
                    f"[AdjConverter] {code_bs} 获取 {len(df)} 条复权因子"
                )
                return df
            finally:
                bs.logout()

        except RuntimeError as e:
            last_err = e
            logger.warning(
                f"[AdjConverter] {code_bs} 第 {attempt + 1} 次请求失败: {e}"
            )

    raise RuntimeError(
        f"BaoStock 复权因子获取失败（{max_retries} 次重试）: {last_err}"
    )


# ---------------------------------------------------------------------------
# convert_qfq_to_hfq
# ---------------------------------------------------------------------------

def convert_qfq_to_hfq(
    df_qfq: pd.DataFrame,
    adj_factors: pd.DataFrame,
    ohlcv_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    将通达信前复权（qfq）DataFrame 转换为后复权（hfq）。

    转换公式：
        hfq_price(t) = qfq_price(t) / qfq_factor(t) × hfq_factor(t)

    其中 qfq_factor(t) 通过前复权因子推导，hfq_factor(t) 来自 BaoStock。
    简化实现：直接对前复权价格乘以校正比例，使最终价格序列与后复权一致。

    Parameters
    ----------
    df_qfq      : 含前复权 OHLCV 的 DataFrame，必须含 "date" 列
    adj_factors : fetch_adj_factor_from_baostock() 返回的 DataFrame
                  列 ["date", "adj_factor"]
    ohlcv_cols  : 需要转换的价格列，默认 ["open","high","low","close"]
                  volume 列不做复权调整

    Returns
    -------
    转换后的 DataFrame（复制），新增 "adj_type"="qfq"(V10)、"adj_factor" 列

    Notes
    -----
    若 adj_factors 为空（BaoStock 无数据），则不做转换并发出警告。
    此方案精度低于直接从 BaoStock 采集后复权价格，仅作为兜底。
    """
    if ohlcv_cols is None:
        ohlcv_cols = ["open", "high", "low", "close"]

    df = df_qfq.copy()

    if adj_factors.empty:
        logger.warning("[AdjConverter] 复权因子为空，不做转换")
        df["adj_type"] = "qfq_unconverted"
        df["adj_factor"] = 1.0
        return df

    # Bug 20 Fix: Align adj_factor to each trading day correctly.
    # adj_factor should use the actual factor on the ex-rights date (no ffill on that day).
    # We build a complete daily series then forward-fill ONLY for non-ex-rights days.
    df_dates = (
        pd.to_datetime(df["date"]).dt.date
        if not isinstance(df["date"].iloc[0], date)
        else df["date"]
    )

    # Build a date-indexed series of known adj_factors
    adj_series = adj_factors.set_index("date")["adj_factor"]
    # Reindex to every trading date; known ex-rights dates get actual values
    factor_series = adj_series.reindex(df_dates)
    # Forward-fill non-ex-rights days using last known adj_factor
    # Use bfill for any leading NaN (dates before first known factor)
    factor_series = factor_series.ffill().bfill()
    factors = factor_series.values.astype(np.float64)

    # 计算前复权因子（TDX 前复权使得最近价格 ≈ 原始价格，历史价格被下调）
    # 通过后复权因子 / 当前最新后复权因子 来还原原始价格，再乘以后复权因子
    # 精确做法：qfq_price × (hfq_factor / hfq_factor_latest) 逆推
    # 此处采用简化方式：直接缩放使序列首日对齐后复权
    last_factor = factors[-1] if len(factors) > 0 else 1.0

    for col in ohlcv_cols:
        if col not in df.columns:
            continue
        prices = pd.to_numeric(df[col], errors="coerce").values.astype(np.float64)
        # [D-01-FIX] 正确的逐日后复权公式：
        # 关系链：qfq_price[t] = raw_price[t] × (latest_factor / factor_t)
        #         hfq_price[t] = raw_price[t] × factor_t
        # 推导：raw_price[t] = qfq_price[t] × factor_t / latest_factor
        #       hfq_price[t] = qfq_price[t] × factor_t² / latest_factor
        #
        # 原实现简化为 qfq × latest_factor，等价于假设 factor_t = latest_factor（所有历史
        # 日期使用同一个复权因子），导致多次除权的早期历史价格被低估，与正确 hfq 序列存在
        # 系统性偏差（除权次数越多、越久远，偏差越大）。
        # 注意：此 TDX 路径为兜底方案；BaoStock 路径直接下载 hfq 数据不受此影响。
        if last_factor > 1e-8:
            df[col] = prices * (factors ** 2) / last_factor
        else:
            df[col] = prices * last_factor  # fallback: last_factor≈0 异常情况

    df["adj_type"] = "qfq"  # V10 QFQ
    df["adj_factor"] = factors
    logger.info(
        f"[AdjConverter] 已转换 {len(df)} 行前复权→后复权"
    )
    return df


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _normalize_code_baostock(code: str) -> str:
    """
    将股票代码规范化为 BaoStock 格式 "sh.XXXXXX" / "sz.XXXXXX"。

    Examples
    --------
    "600000"   → "sh.600000"
    "000001"   → "sz.000001"
    "sh600000" → "sh.600000"
    """
    code = str(code).strip()
    if "." in code:
        return code.lower()
    if code.startswith("sh") or code.startswith("SH"):
        return f"sh.{code[2:]}"
    if code.startswith("sz") or code.startswith("SZ"):
        return f"sz.{code[2:]}"
    # 按首位数字判断交易所
    if code.startswith("6"):
        return f"sh.{code}"
    return f"sz.{code}"



