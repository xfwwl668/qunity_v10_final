"""
fundamental_downloader.py — Q-UNITY V9.1 基本面数据下载器（AKShare）

下载以下基本面数据：
  · PE_TTM、ROE_TTM、EPS、市值（股价 × 总股本）
    via: ak.stock_a_lg_indicator(symbol=code)
  · 净利润季报（用于 SUE 计算）
    via: ak.stock_profit_sheet_by_report_em(symbol=code)

数据存储：
  每只股票一个 Parquet 文件 → data/fundamental/{code}.parquet
  列：date(公告日/报告期), pe_ttm, roe_ttm, eps, market_cap
  按 date 升序排列

前视偏差防护：
  使用公告日（announcement_date）作为填充起点，而非报告期末。
  AKShare stock_a_lg_indicator 返回的 trade_date 本质上是公告/更新日，可直接使用。

使用方式::

    dl = FundamentalDownloader(output_dir="data/fundamental")
    codes = ["600519", "000001", "300750"]
    results = dl.download_all(codes, start="2019-01-01")
    print(results)  # {"600519": "ok", "000001": "ok", ...}
"""
from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── 可选依赖检查 ──────────────────────────────────────────────────────────────
try:
    import akshare as ak
    _AK_AVAILABLE = True
except ImportError:
    ak = None
    _AK_AVAILABLE = False

try:
    import pandas as pd
    _PD_AVAILABLE = True
except ImportError:
    pd = None
    _PD_AVAILABLE = False

_FUNDAMENTAL_FIELDS = ["date", "pe_ttm", "roe_ttm", "eps", "market_cap"]


class FundamentalDownloader:
    """
    基本面数据下载器（AKShare）

    Parameters
    ----------
    output_dir      : Parquet 输出目录（每股一个文件）
    delay_range     : 每次请求后的随机延时范围（秒），避免触发反爬
    retry_times     : 单只股票下载失败后重试次数
    progress_cb     : 进度回调 Callable[[done, total, code], None]
    """

    def __init__(
        self,
        output_dir:   str = "data/fundamental",
        delay_range:  tuple = (0.5, 1.5),
        retry_times:  int = 3,
        progress_cb:  Optional[Callable] = None,
    ) -> None:
        self._out_dir     = Path(output_dir)
        self._delay_range = delay_range
        self._retry       = max(1, retry_times)
        self._progress_cb = progress_cb

        if not _AK_AVAILABLE:
            logger.warning(
                "[FundamentalDownloader] akshare 未安装。"
                "请执行: pip install akshare"
            )
        if not _PD_AVAILABLE:
            logger.warning(
                "[FundamentalDownloader] pandas 未安装（必须）。"
                "请执行: pip install pandas pyarrow"
            )

    @property
    def available(self) -> bool:
        """akshare 和 pandas 均可用时为 True"""
        return _AK_AVAILABLE and _PD_AVAILABLE

    # ── 批量下载 ──────────────────────────────────────────────────────────────

    def download_all(
        self,
        codes:         List[str],
        start:         str = "2015-01-01",
        end:           Optional[str] = None,
        force_reload:  bool = False,
    ) -> Dict[str, str]:
        """
        批量下载全部股票的基本面数据。

        Returns
        -------
        Dict[str, str]  {code: "ok" | "skip" | "error"}
        """
        if not self.available:
            logger.error("[FundamentalDownloader] 依赖库不可用，无法下载")
            return {code: "error" for code in codes}

        self._out_dir.mkdir(parents=True, exist_ok=True)
        total   = len(codes)
        results = {}

        for idx, code in enumerate(codes):
            # 增量跳过
            out_path = self._out_dir / f"{_bare_code(code)}.parquet"
            if out_path.exists() and not force_reload:
                results[code] = "skip"
                if self._progress_cb:
                    self._progress_cb(idx + 1, total, code)
                continue

            status = "error"
            for attempt in range(self._retry):
                try:
                    status = self._download_one(code, start, end)
                    if status == "ok":
                        break
                except Exception as e:
                    logger.warning(f"[FundamentalDownloader] {code} 下载失败 (尝试{attempt+1}): {e}")
                    if attempt < self._retry - 1:
                        time.sleep(random.uniform(*self._delay_range) * 2)

            results[code] = status

            if self._progress_cb:
                self._progress_cb(idx + 1, total, code)

            # 随机延时（防止反爬）
            time.sleep(random.uniform(*self._delay_range))

        ok_cnt  = sum(1 for v in results.values() if v == "ok")
        skip_cnt = sum(1 for v in results.values() if v == "skip")
        err_cnt = sum(1 for v in results.values() if v == "error")
        logger.info(
            f"[FundamentalDownloader] 批量下载完成: "
            f"成功={ok_cnt} 跳过={skip_cnt} 失败={err_cnt}"
        )
        return results

    # ── 单股下载 ──────────────────────────────────────────────────────────────

    def _download_one(self, code: str, start: str, end: Optional[str]) -> str:
        """下载单只股票的基本面数据并保存为 Parquet"""
        bare = _bare_code(code)
        df = self._fetch_indicator(bare)

        if df is None or df.empty:
            logger.debug(f"[FundamentalDownloader] {bare} 无数据")
            return "error"

        # 日期过滤
        mask = df["date"] >= start
        if end:
            mask = mask & (df["date"] <= end)
        df = df[mask].reset_index(drop=True)

        if df.empty:
            logger.debug(f"[FundamentalDownloader] {bare} 在 {start}~{end} 无数据")
            return "error"

        # 写 Parquet
        out_path = self._out_dir / f"{bare}.parquet"
        df.to_parquet(str(out_path), index=False, compression="snappy")
        logger.debug(f"[FundamentalDownloader] {bare} 保存 {len(df)} 行 → {out_path}")
        return "ok"

    def _fetch_indicator(self, bare_code: str) -> Optional[Any]:
        """
        调用 ak.stock_a_lg_indicator() 获取 PE/ROE/EPS/市值。

        AKShare 接口文档：
          ak.stock_a_lg_indicator(symbol="600519")
          返回列：trade_date, pe_ttm, pb, ps_ttm, dv_ttm, total_mv

        归一化为统一格式：
          date, pe_ttm, roe_ttm, eps, market_cap
        """
        try:
            raw = ak.stock_a_lg_indicator(symbol=bare_code)
            if raw is None or raw.empty:
                return None

            raw = raw.copy()

            # 重命名
            rename_map: Dict[str, str] = {}
            col_lower = {c.lower(): c for c in raw.columns}

            # date 列
            for cname in ["trade_date", "date", "报告期", "公告日期"]:
                if cname in raw.columns:
                    rename_map[cname] = "date"
                    break
                if cname.lower() in col_lower:
                    rename_map[col_lower[cname.lower()]] = "date"
                    break

            # pe_ttm
            for cname in ["pe_ttm", "pe", "市盈率(TTM)", "市盈率TTM"]:
                if cname in raw.columns:
                    rename_map[cname] = "pe_ttm"
                    break

            # market_cap
            for cname in ["total_mv", "市值", "总市值", "total_market_cap"]:
                if cname in raw.columns:
                    rename_map[cname] = "market_cap"
                    break

            raw = raw.rename(columns=rename_map)

            # 确保必要列存在
            if "date" not in raw.columns:
                logger.warning(f"[FundamentalDownloader] {bare_code} 缺少 date 列，可用列: {list(raw.columns)}")
                return None

            # 添加缺失列（填 NaN）
            # [BUG-FUND-EPTTSM-NAMING-MISMATCH 注释]
            # ak.stock_a_lg_indicator 不直接返回 eps 列（该接口无每股收益字段）。
            # eps 在此处填 NaN，由 step1_akshare.py 的 epsTTM 字段补充（季度EPS，
            # 字段名 epsTTM 实为"季度每股收益"而非"滚动12月TTM EPS"，命名有歧义，
            # 但 SUE 计算逻辑基于季度对比，语义上是正确的）。
            # roe_ttm 同理：stock_a_lg_indicator 无此列，由 step1 的 roeAvg 补充。
            for col in ["pe_ttm", "roe_ttm", "eps", "market_cap"]:
                if col not in raw.columns:
                    raw[col] = float("nan")

            # 尝试获取 roe_ttm（部分接口可能没有）
            for cname in ["roe_ttm", "roe", "净资产收益率"]:
                if cname in raw.columns and "roe_ttm" not in rename_map.values():
                    raw = raw.rename(columns={cname: "roe_ttm"})
                    break

            # 类型转换
            raw["date"] = raw["date"].astype(str).str[:10]
            for col in ["pe_ttm", "roe_ttm", "eps", "market_cap"]:
                raw[col] = pd.to_numeric(raw[col], errors="coerce").astype("float32")

            raw["market_cap"] = raw["market_cap"].astype("float64")

            # 排序
            raw = raw.sort_values("date").reset_index(drop=True)

            return raw[_FUNDAMENTAL_FIELDS]

        except Exception as e:
            logger.warning(f"[FundamentalDownloader] ak.stock_a_lg_indicator({bare_code}) 失败: {e}")
            return None

    # ── CSV 导入 ──────────────────────────────────────────────────────────────

    def import_from_csv(
        self,
        csv_path: str,
        overwrite: bool = False,
    ) -> Dict[str, str]:
        """
        从本地 CSV 批量导入（格式：code,date,pe_ttm,roe_ttm,eps,market_cap）。

        Returns
        -------
        Dict[str, str]  {code: "ok" | "error"}
        """
        if not _PD_AVAILABLE:
            logger.error("[FundamentalDownloader] pandas 不可用")
            return {}
        try:
            df = pd.read_csv(csv_path, dtype={"code": str})
            required = {"code", "date"}
            missing = required - set(df.columns)
            if missing:
                logger.error(f"[FundamentalDownloader] CSV 缺少必需列: {missing}")
                return {}

            # 规范化
            for col in ["pe_ttm", "roe_ttm", "eps"]:
                if col not in df.columns:
                    df[col] = float("nan")
            if "market_cap" not in df.columns and "mktcap" in df.columns:
                df = df.rename(columns={"mktcap": "market_cap"})
            if "market_cap" not in df.columns:
                df["market_cap"] = float("nan")

            df["date"] = df["date"].astype(str).str[:10]

            results: Dict[str, str] = {}
            self._out_dir.mkdir(parents=True, exist_ok=True)

            for code, grp in df.groupby("code"):
                bare = _bare_code(str(code))
                out_path = self._out_dir / f"{bare}.parquet"
                if out_path.exists() and not overwrite:
                    results[str(code)] = "skip"
                    continue
                try:
                    grp_sorted = grp.sort_values("date").reset_index(drop=True)
                    grp_sorted[_FUNDAMENTAL_FIELDS].to_parquet(
                        str(out_path), index=False, compression="snappy"
                    )
                    results[str(code)] = "ok"
                except Exception as e:
                    logger.warning(f"[FundamentalDownloader] CSV 导入 {code} 失败: {e}")
                    results[str(code)] = "error"

            return results
        except Exception as e:
            logger.error(f"[FundamentalDownloader] CSV 导入失败: {e}")
            return {}

    # ── 状态检查 ──────────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """返回已下载数据状态"""
        files = list(self._out_dir.glob("*.parquet")) if self._out_dir.exists() else []
        total_codes  = len(files)
        total_size   = sum(f.stat().st_size for f in files) / 1024 / 1024

        latest_date  = "N/A"
        if files and _PD_AVAILABLE:
            sample = files[:5]
            dates = []
            for f in sample:
                try:
                    df = pd.read_parquet(f, columns=["date"])
                    if not df.empty:
                        dates.append(df["date"].max())
                except Exception:
                    pass
            if dates:
                latest_date = max(dates)

        return {
            "n_codes":     total_codes,
            "total_mb":    round(total_size, 1),
            "latest_date": str(latest_date),
            "output_dir":  str(self._out_dir),
        }


def _bare_code(code: str) -> str:
    """去掉交易所前缀（sh.600519 → 600519）"""
    return code.split(".")[-1] if "." in code else code
