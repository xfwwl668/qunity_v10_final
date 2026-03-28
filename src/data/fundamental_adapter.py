"""
fundamental_adapter.py — Q-UNITY V9.1 基本面数据 → npy 矩阵构建器

输入：
  · data/fundamental/{code}.parquet（每股一个文件）
  · data/npy/meta.json（日线矩阵的 codes 和 dates）

输出（保存到 data/npy/）：
  · pe_ttm.npy     (N, T) float32   市盈率(TTM)
  · roe_ttm.npy    (N, T) float32   净资产收益率(TTM)
  · eps.npy        (N, T) float32   每股收益
  · mktcap.npy     (N, T) float64   总市值（元）
  · sue.npy        (N, T) float32   标准化未预期盈余（SUE）
  · days_ann.npy   (N, T) int16     距最近公告的交易日数

关键设计：
  1. 无偏前向填充：使用公告日 date（而非报告期）作为填充起点，
     防止在公告发布前使用了该财务数据（前视偏差）。
  2. 仅在 dates 列表中的交易日上填充，不跨非交易日跳步。
  3. SUE 简化计算：(EPS_t - EPS_{t-4Q}) / std(差值, 8Q)
     若历史 EPS 数据不足 5 个季度，该时期 SUE = NaN（填 0.0）。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    _PD_AVAILABLE = True
except ImportError:
    pd = None
    _PD_AVAILABLE = False


class FundamentalAdapter:
    """
    基本面 Parquet → npy 矩阵构建器

    Parameters
    ----------
    fundamental_dir : str  每股 Parquet 所在目录
    npy_dir         : str  日线 npy 矩阵目录（读 meta.json，写输出文件）
    """

    def __init__(
        self,
        fundamental_dir: str = "data/fundamental",
        npy_dir:         str = "data/npy",
    ) -> None:
        self._fund_dir = Path(fundamental_dir)
        self._npy_dir  = Path(npy_dir)

    # ── 主接口 ────────────────────────────────────────────────────────────────

    def build(
        self,
        fields: Optional[List[str]] = None,
        progress_cb=None,
    ) -> Dict[str, Any]:
        """
        构建基本面 npy 矩阵。

        Parameters
        ----------
        fields      : 需要构建的字段列表，默认全部
                      ["pe_ttm", "roe_ttm", "eps", "mktcap", "sue", "days_ann"]
        progress_cb : Callable[[done, total, code], None]

        Returns
        -------
        Dict  包含 shape, fields_built, n_missing, elapsed_s
        """
        if not _PD_AVAILABLE:
            raise RuntimeError("pandas 未安装，无法构建基本面矩阵")

        import time
        t0 = time.perf_counter()

        # 加载日线矩阵元信息
        codes, dates = self._load_meta()
        N, T = len(codes), len(dates)
        logger.info(f"[FundamentalAdapter] 开始构建: N={N}, T={T}")

        if fields is None:
            fields = ["pe_ttm", "roe_ttm", "eps", "mktcap", "sue", "days_ann"]

        # 为每个字段分配输出矩阵
        arrays: Dict[str, np.ndarray] = {}
        dtypes = {
            "pe_ttm":   np.float32,
            "roe_ttm":  np.float32,
            "eps":      np.float32,
            "mktcap":   np.float64,
            "sue":      np.float32,
            "days_ann": np.int16,
        }
        fill_defaults = {
            "pe_ttm": np.nan, "roe_ttm": np.nan, "eps": np.nan,
            "mktcap": np.nan, "sue": 0.0, "days_ann": 0,
        }
        for f in fields:
            arr = np.full((N, T), fill_defaults.get(f, np.nan), dtype=dtypes.get(f, np.float32))
            arrays[f] = arr

        # 构建日期索引（date_str → column_index）
        date_idx: Dict[str, int] = {d: i for i, d in enumerate(dates)}

        # 逐股填充
        n_missing = 0
        total = len(codes)
        for stock_i, code in enumerate(codes):
            bare = _bare(code)
            parquet_path = self._fund_dir / f"{bare}.parquet"

            if not parquet_path.exists():
                n_missing += 1
                if progress_cb:
                    progress_cb(stock_i + 1, total, code)
                continue

            try:
                df = pd.read_parquet(str(parquet_path))
                self._fill_stock(
                    stock_i, df, date_idx, arrays, fields, T
                )
            except Exception as e:
                logger.debug(f"[FundamentalAdapter] {bare} 填充失败: {e}")
                n_missing += 1

            if progress_cb:
                progress_cb(stock_i + 1, total, code)

        # 保存 npy 文件
        self._npy_dir.mkdir(parents=True, exist_ok=True)
        field_map = {"mktcap": "mktcap.npy", "days_ann": "days_ann.npy"}
        for f in fields:
            fname = field_map.get(f, f"{f}.npy")
            np.save(str(self._npy_dir / fname), arrays[f])
            logger.info(f"[FundamentalAdapter] 保存 {fname}: shape={arrays[f].shape}")

        elapsed = time.perf_counter() - t0
        logger.info(
            f"[FundamentalAdapter] 构建完成: fields={fields}, "
            f"n_missing={n_missing}/{N}, elapsed={elapsed:.1f}s"
        )
        return {
            "shape":         (N, T),
            "fields_built":  fields,
            "n_missing":     n_missing,
            "elapsed_s":     round(elapsed, 2),
        }

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _load_meta(self) -> Tuple[List[str], List[str]]:
        """加载日线矩阵的 codes 和 dates"""
        meta_path = self._npy_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"[FundamentalAdapter] meta.json 不存在: {meta_path}\n"
                "请先构建日线 npy 矩阵"
            )
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        codes = meta.get("codes", [])
        dates = meta.get("dates", [])
        if not codes or not dates:
            raise ValueError("[FundamentalAdapter] meta.json 中 codes 或 dates 为空")

        # 统一为字符串格式
        codes = [str(c) for c in codes]
        dates = [str(d)[:10] for d in dates]
        return codes, dates

    def _fill_stock(
        self,
        stock_i:  int,
        df:       Any,          # pd.DataFrame
        date_idx: Dict[str, int],
        arrays:   Dict[str, np.ndarray],
        fields:   List[str],
        T:        int,
    ) -> None:
        """
        将单只股票的基本面数据前向填充到矩阵的相应行。

        核心逻辑：
          1. 遍历 df 中每一行（按公告日升序）
          2. 找到公告日对应的列索引 t_start
          3. 从 t_start 到下一条公告日前，用本期数据前向填充
          4. days_ann[t] = t - t_start（距最近公告的交易日数）
        """
        if df.empty:
            return

        df = df.sort_values("date").reset_index(drop=True)

        # 找到 eps_prev（去年同期 EPS，用于 SUE 计算）
        # [BUG-SUE-SEASONAL-FIX] eps_series改为存 (t_col, eps_val, quarter_key)
        # quarter_key = "YYYY-QN" 从 statDate 提取，用于精确季度对齐
        # 修复前: eps_series[i-4] 假设第4个历史条目是去年同期，缺季报时对应错误季度
        # 修复后: 按 quarter_key 精确匹配同季度（如 2022-Q3 对应 2021-Q3）
        eps_series: List[Tuple[int, float, str]] = []  # (t_col, eps_val, quarter_key)

        n_rows = len(df)

        for row_i in range(n_rows):
            ann_date = str(df.iloc[row_i]["date"])[:10]
            t_start = date_idx.get(ann_date)

            # 若公告日不在交易日历中，找最近的下一个交易日
            if t_start is None:
                t_start = _find_next_trading_day(ann_date, date_idx)
            if t_start is None:
                continue

            # 本条记录填充的结束列（下一条公告日 - 1 或 T）
            if row_i + 1 < n_rows:
                next_ann = str(df.iloc[row_i + 1]["date"])[:10]
                t_end_raw = date_idx.get(next_ann) or _find_next_trading_day(next_ann, date_idx)
                t_end = t_end_raw if t_end_raw is not None else T
            else:
                t_end = T

            t_end = min(t_end, T)

            if t_start >= t_end:
                continue

            # 填充各字段
            row_data = df.iloc[row_i]
            col_fill_slice = slice(t_start, t_end)

            if "pe_ttm" in fields:
                val = _safe_float32(row_data.get("pe_ttm"))
                arrays["pe_ttm"][stock_i, t_start:t_end] = val
            if "roe_ttm" in fields:
                val = _safe_float32(row_data.get("roe_ttm"))
                arrays["roe_ttm"][stock_i, t_start:t_end] = val
            if "eps" in fields or "sue" in fields:
                eps_val = _safe_float32(row_data.get("eps"))
                if "eps" in fields:
                    arrays["eps"][stock_i, t_start:t_end] = eps_val
                if not np.isnan(eps_val):
                    # [BUG-SUE-SEASONAL-FIX] 提取 statDate → quarter_key
                    stat_raw = str(df.iloc[row_i].get("statDate", "") or "")[:10]
                    quarter_key = ""
                    if len(stat_raw) >= 10:
                        try:
                            import datetime as _dt
                            sd = _dt.date.fromisoformat(stat_raw)
                            qn = (sd.month - 1) // 3 + 1  # 1=Q1,2=Q2,3=Q3,4=Q4
                            quarter_key = f"{sd.year}-Q{qn}"
                        except ValueError:
                            pass
                    eps_series.append((t_start, float(eps_val), quarter_key))
            if "mktcap" in fields:
                mkt = _safe_float64(row_data.get("market_cap"))
                arrays["mktcap"][stock_i, t_start:t_end] = mkt
            if "days_ann" in fields:
                for t_col in range(t_start, t_end):
                    days = min(t_col - t_start, 32767)  # int16 上限
                    arrays["days_ann"][stock_i, t_col] = days

        # SUE 计算（需要至少 5 个季度的 EPS）
        if "sue" in fields and len(eps_series) >= 5:
            self._fill_sue(stock_i, eps_series, arrays["sue"], T)

    def _fill_sue(
        self,
        stock_i:    int,
        eps_series: List[Tuple[int, float, str]],
        sue_arr:    np.ndarray,
        T:          int,
    ) -> None:
        """
        简化版 SUE 计算：

          SUE_t = (EPS_t - EPS_{t-4Q}) / std(差值_过去8Q)

        [BUG-SUE-SEASONAL-FIX] 改用 quarter_key 精确季度匹配，而非序号偏移 i-4。
        原实现中，若公司缺失某季报，eps_series[i-4] 指向错误季度（如Q3对比Q1）。
        修复后：优先按 YYYY-QN 匹配同期；无 quarter_key 时回退序号偏移兼容旧数据。
        """
        Q = 4
        diffs: List[Tuple[int, float]] = []  # (t_col, diff)

        # 建立 quarter_key → eps_val 映射，用于精确同期查找
        qkey_map: dict = {}
        for entry in eps_series:
            t_col, eps_val, qkey = entry
            if qkey:
                qkey_map[qkey] = eps_val

        def _prev_year_qkey(qkey: str) -> str:
            try:
                yr, qn = qkey.split("-Q")
                return f"{int(yr) - 1}-Q{qn}"
            except Exception:
                return ""

        for i in range(Q, len(eps_series)):
            t_cur, eps_cur, qkey_cur = eps_series[i]
            # 优先按季度精确匹配去年同期
            if qkey_cur:
                prev_qkey = _prev_year_qkey(qkey_cur)
                if prev_qkey in qkey_map:
                    diff = eps_cur - qkey_map[prev_qkey]
                    diffs.append((t_cur, diff))
                    continue
            # 回退：无 quarter_key 或找不到同期时用序号偏移（兼容旧数据）
            _, eps_prev, _ = eps_series[i - Q]
            diffs.append((t_cur, eps_cur - eps_prev))

        if len(diffs) < 2:
            return

        # 滚动 std（用前 8 个差值估算标准差）
        for j, (t_col, diff) in enumerate(diffs):
            window = [d for _, d in diffs[max(0, j - 8): j + 1]]
            if len(window) < 2:
                continue
            mean_d = sum(window) / len(window)
            std_d  = (sum((x - mean_d) ** 2 for x in window) / (len(window) - 1)) ** 0.5
            if std_d < 1e-6:
                sue = 0.0
            else:
                sue = float(diff / std_d)
            # [BUG-FUNDA-SUE-FILL-BOUNDARY FIX] 用 diffs[j+1][0] 作为前向填充边界。
            # 原代码使用 eps_series[idx_in_orig + Q]（4条记录后 ≈ 1年），导致单季 SUE 值
            # 被前向填充长达1年（覆盖4个季度），严重污染后续季度的 SUE 信号。
            # 正确逻辑：每条 SUE 只填充到下一条可比较季度公告日为止（约1季度）。
            t_end = T
            if j + 1 < len(diffs):
                nxt_t = diffs[j + 1][0]
                if nxt_t > t_col:
                    t_end = min(nxt_t, T)
            sue_arr[stock_i, t_col:t_end] = np.float32(max(-10.0, min(10.0, sue)))

    # ── 验证 ──────────────────────────────────────────────────────────────────

    def validate(self) -> Dict[str, Any]:
        """验证已构建的 npy 矩阵"""
        results: Dict[str, Any] = {}
        expected_fields = ["pe_ttm", "roe_ttm", "eps", "mktcap", "sue", "days_ann"]

        for f in expected_fields:
            fname_map = {"mktcap": "mktcap.npy", "days_ann": "days_ann.npy"}
            fname = fname_map.get(f, f"{f}.npy")
            path = self._npy_dir / fname
            if not path.exists():
                results[f] = {"exists": False}
                continue
            arr = np.load(str(path), mmap_mode="r")
            nan_pct = float(np.isnan(arr.astype(np.float64)).mean()) if arr.dtype.kind in ("f",) else 0.0
            results[f] = {
                "exists":   True,
                "shape":    arr.shape,
                "nan_pct":  round(nan_pct, 3),
                "dtype":    str(arr.dtype),
            }
        return results


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _bare(code: str) -> str:
    return code.split(".")[-1] if "." in code else code


def _safe_float32(val: Any) -> np.float32:
    try:
        v = float(val)
        if np.isnan(v) or np.isinf(v):
            return np.float32(np.nan)
        return np.float32(v)
    except (TypeError, ValueError):
        return np.float32(np.nan)


def _safe_float64(val: Any) -> np.float64:
    try:
        v = float(val)
        if np.isnan(v) or np.isinf(v):
            return np.float64(np.nan)
        return np.float64(v)
    except (TypeError, ValueError):
        return np.float64(np.nan)


def _find_next_trading_day(date_str: str, date_idx: Dict[str, int]) -> Optional[int]:
    """找到 date_str 之后（含）最近的交易日索引
    
    [BUG-FUNDA-FIND-NEXT-LINEAR-SORT FIX] 缓存排序后的日期列表，避免每次调用重排。
    原实现每次调用都执行 sorted(date_idx.items())，O(T log T) per call。
    对于 N=5000股票、每股20条公告、T=3750: 5000×20 次调用 × O(T log T) = 极慢。
    修复：用 _sorted_dates_cache 模块级缓存，id(date_idx) 变化时才重建。
    """
    cache = _find_next_trading_day._cache
    cache_key = id(date_idx)
    if cache.get("key") != cache_key:
        cache["key"] = cache_key
        cache["sorted"] = sorted(date_idx.items())  # 仅在 date_idx 对象变化时重建
    for d, idx in cache["sorted"]:
        if d >= date_str:
            return idx
    return None

_find_next_trading_day._cache: dict = {}
