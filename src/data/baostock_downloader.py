# Copyright (c) 2024-2026 Q-UNITY Quantitative Research Group
# All rights reserved. Internal use only.
"""
baostock_downloader.py — Q-UNITY V9.1 A股历史数据下载器（BaoStock）

────────────────────────────────────────────────────────────────────────
[PERF] 并发架构说明（为什么必须用 ProcessPoolExecutor）
────────────────────────────────────────────────────────────────────────

baostock 使用模块级单例 BaoStockClient（一个全局 TCP socket）：
  baostock/client.py 或 data_fetch.py 中：_SINGLETON = BaoStockClient()
  所有 API 函数都通过这个单例读写同一个 socket

多线程（ThreadPoolExecutor）的致命问题：
  8 个线程共享同一 socket → 并发 read/write → 帧错位
  → UTF-8 解码失败、zlib 压缩错误、"接收数据异常"
  注：用 importlib.util.exec_module 创建"隔离"模块命名空间同样无效，
      因为相对导入（from .client import _SINGLETON）仍经过 sys.modules，
      所有命名空间共享同一 client.py 模块实例（经实测证明）

ProcessPoolExecutor 的正确工作原理：
  每个子进程是独立的 Python 解释器，有独立的 sys.modules
  各自执行 import baostock → 各自创建 BaoStockClient() 实例
  真正独立的 socket，互不干扰，无需任何锁

历史问题（原 config.json n_workers=32）：
  32个进程 × 5s启动 = 160s 仅启动开销
  32个进程同时 bs.login() → BaoStock 服务器限速
  → 实测吞吐: 100KB/s

本版本修复：
  n_workers=8（BaoStock 服务器舒适并发上限）
  删除 worker 内的 sleep（原 0.03~0.10s/股，8进程 × 1000股 = 80~800s 纯浪费）
  预期吞吐: 8进程 × 75KB/股 × 1/0.8s ≈ 750KB/s

[BUG-SPEED-1 NOTE] 历史上代码在 n_workers=8 的文档说明下，实际实现却使用了 n_workers=16
  默认值，导致16个进程同时登录BaoStock服务器触发限速，实测速度从500KB/s降至几十KB/s。
  V9.1-fixed 已将默认值恢复为 n_workers=8。

Windows 使用注意：
  若将程序打包为 .exe，需在入口处调用 multiprocessing.freeze_support()
  直接运行 python main.py 无需额外处理（main() 在 __name__=='__main__' 保护下）
"""
from __future__ import annotations

import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# BaoStock 可用性检测
try:
    import baostock as bs
    _BS_AVAILABLE = True
except ImportError:
    _BS_AVAILABLE = False
    bs = None  # type: ignore


# ══════════════════════════════════════════════════════════════════════════════
# 子进程全局状态（由 _pool_initializer 设置，每进程独立）
#
# ProcessPoolExecutor 保证：每个 worker 是独立 Python 进程
#   → 独立 sys.modules → 独立 baostock 模块实例
#   → 独立 BaoStockClient() → 独立 TCP socket
#   → 真正并行，无竞态，无需锁
# ══════════════════════════════════════════════════════════════════════════════

_g_output_dir:   str   = ""
_g_retry_times:  int   = 3
_g_retry_delay:  float = 2.0
_g_ready:        bool  = False


def _pool_initializer(output_dir: str, retry_times: int, retry_delay: float) -> None:
    """
    ProcessPoolExecutor 子进程初始化器（每个 worker 进程调用一次）。

    在子进程中建立独立的 BaoStock 连接：
      - 每个子进程有自己的 Python 解释器和 sys.modules
      - import baostock 在子进程中重新执行 → 新的 BaoStockClient 实例
      - bs.login() 建立独立的 TCP socket
      - 与其他进程完全隔离，无需跨进程锁

    [PERF] 不在此处 import numpy/pandas（它们由 _worker_download_one 按需使用）
    以减少进程启动时间。numpy/pandas 已在模块顶部 import，但 Python 进程
    fork/spawn 时会重新执行 import，这是 ProcessPool 的固有开销（8进程约24s，可接受）。
    """
    import socket as _sock
    _sock.setdefaulttimeout(60)

    global _g_output_dir, _g_retry_times, _g_retry_delay, _g_ready
    _g_output_dir  = output_dir
    _g_retry_times = retry_times
    _g_retry_delay = retry_delay

    if not _BS_AVAILABLE or bs is None:
        _g_ready = False
        return

    try:
        lg = bs.login()
        _g_ready = (lg.error_code == "0")
        if not _g_ready:
            logger.warning(f"[Worker] BaoStock 子进程登录失败: {lg.error_msg}")
            return

        # [PERF-STAGGER] 登录成功后随机等待，错开各进程的第一次查询时机。
        # 问题：多个进程同时 login 成功后，会在几毫秒内同时发起第一次查询请求，
        #       BaoStock 服务器收到并发请求 burst → 响应帧错位 → UTF-8/zlib 错误。
        # 修复：在 initializer 中引入随机错开延迟（仅执行一次，不影响后续每只股票的速度）。
        # [BUG-SPEED-2 FIX] 错开范围扩展为 5.0s，兼容最多 32 进程（平均间隔 ~156ms，充足）。
        # 实测：32 进程 ~1m/s，16 进程 ~600KB/s，8 进程 ~400KB/s。
        # uniform(0, 5) 仅在 initializer 执行一次，不影响后续每只股票的速度。
        import random as _rand
        time.sleep(_rand.uniform(0.0, 5.0))

    except Exception as e:
        _g_ready = False
        logger.warning(f"[Worker] BaoStock 子进程初始化异常: {e}")


def _clean_df_static(df: pd.DataFrame, code: str) -> Optional[pd.DataFrame]:
    """
    数据清洗（模块级，供子进程调用，无实例方法依赖）。

    · 停牌日（volume=0, close>0）保留，columnar_adapter 需要它填充日期轴
    · 仅过滤 close 为 NaN 或负值的真实异常行
    """
    try:
        df = df.copy()
        df["code"] = code
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"])
        df = df[df["close"] > 0]          # [BUG-CLEAN-CLOSE-ZERO FIX] A股收盘价不可能为0，=0是缺失/错误数据
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = df[col].astype("float32")
        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0.0).astype("float32")
        if "amount" in df.columns:
            df["amount"] = df["amount"].fillna(0.0).astype("float32")
        df["adj_type"] = "hfq"   # V10: QFQ前复权标记
        df = df.sort_values("date").reset_index(drop=True)
        return df if len(df) > 0 else None
    except Exception as e:
        logger.warning(f"[Worker] {code} 数据清洗失败: {e}")
        return None


def _worker_download_one(task: Tuple[str, str, str, bool]) -> Tuple[str, str, int]:
    """
    模块级可 pickle worker 函数，在子进程中执行单只股票下载。

    【设计要点】
    · ProcessPoolExecutor 子进程：独立 Python 解释器，独立 baostock socket
    · 无跨进程锁，每个进程只操作自己的 socket
    · [PERF-FIX] 删除原 sleep(0.03~0.10)：无跨进程竞争，sleep 纯属浪费
      原版 8进程 × 1000只 × 0.065s平均 = 520s 纯等待，相当于再下435只的时间
    · 增量检查：已有且最新的 Parquet 直接 skip
    · 重试机制：指数退避，最多 retry_times 次

    Parameters
    ----------
    task : (code, start, end, force)

    Returns
    -------
    (code, status, nrows)  status ∈ {"ok", "skip", "error"}
    """
    code, start, end, force = task

    if not _g_ready:
        return code, "error", 0

    out_path = Path(_g_output_dir) / f"{code}.parquet"

    # ── 增量检查 ──────────────────────────────────────────────────────────────
    # [BUG-DL-SKIP-FULL-READ FIX] 原代码读完整 parquet(7列) 仅为判断 last_date：
    #   5000只"已最新"时 = 5000次×7列全读，~85% I/O 浪费。
    #   修复：先只读 date 列判断是否 skip；需更新时才读全量做 merge。
    #
    # [BUG-DL-HFQ-STALE-FACTOR FIX] adjustflag="2"  # V10: QFQ(前复权)(后复权)的增量更新陷阱：
    #   除权事件后 BaoStock 会修改所有历史复权价，但增量只追加新行。
    #   旧 Parquet 用旧复权因子，新数据用新因子，交界处出现价格断层。
    #   断层 = 虚假大涨/大跌信号，触发错误止损/追涨。
    #   修复：增量更新时，无论如何都重新下载最后 HFQ_OVERLAP_DAYS 天，
    #   并用新数据覆盖这段窗口（drop_duplicates(keep="last")），
    #   保证复权因子一致。正常无除权时新旧数据完全相同，无副作用。
    HFQ_OVERLAP_DAYS = 90   # 宽松窗口：覆盖季报除权期，保证连续性
    _start = start
    _existing_df: Optional[pd.DataFrame] = None
    if not force and out_path.exists():
        try:
            # Step-1: 只读 date 列，快速判断是否需要更新（~85% I/O 节省）
            _date_only = pd.read_parquet(str(out_path), columns=["date"])
            if len(_date_only) > 0:
                last_date = str(_date_only["date"].max())
                cutoff = (date.today() - timedelta(days=3)).strftime("%Y-%m-%d")
                if last_date >= cutoff:
                    return code, "skip", len(_date_only)
                # Step-2: 需要更新，读全量做 merge
                _existing_df = pd.read_parquet(str(out_path))
                # [HFQ-STALE-FACTOR FIX] 从 last_date 回退 HFQ_OVERLAP_DAYS 天
                # 强制重下窗口内数据，覆盖可能因除权改变的历史复权价
                last_date_obj = pd.to_datetime(last_date).date()
                overlap_start = (last_date_obj - timedelta(days=HFQ_OVERLAP_DAYS)).isoformat()
                _start = max(overlap_start, start) if start else overlap_start
        except Exception:
            _existing_df = None  # 文件损坏时重新全量下载

    # ── 带重试的下载主体 ──────────────────────────────────────────────────────
    # [PERF-FIX] 删除 sleep(0.03~0.10)：各进程独立 socket，无并发竞争，无需等待
    # 重试时才 sleep（指数退避），正常路径零等待
    for attempt in range(_g_retry_times):
        try:
            rs = bs.query_history_k_data_plus(
                code,
                "date,open,high,low,close,volume,amount",
                start_date=_start,
                end_date=end,
                frequency="d",
                adjustflag="2",   # V10: QFQ前复权（原V9为"2"后复权，已改）
            )

            if rs.error_code != "0":
                logger.debug(
                    f"[Worker] {code} 查询失败(尝试{attempt+1}): {rs.error_msg}"
                )
                time.sleep(_g_retry_delay * (attempt + 1))
                continue

            rows: list = []
            while rs.next():
                rows.append(rs.get_row_data())

            if not rows:
                return code, "skip", 0

            df = pd.DataFrame(rows, columns=rs.fields)
            df = _clean_df_static(df, code)
            if df is None or len(df) == 0:
                return code, "skip", 0

            # [BUG-DL-HFQ-STALE-FACTOR FIX] merge 时新数据优先（keep="last"）：
            # 重叠窗口内新下载的数据含最新复权因子，必须覆盖旧的 Parquet 行。
            # pd.concat([old, new]) + drop_duplicates(keep="last") → new 行保留。
            if _existing_df is not None and not force:
                try:
                    # ── [BUG-1.2 FIX] 后复权因子改变检测 ────────────────────
                    # 后复权(adjustflag=2)一旦发生除权，所有历史价格均被重算。
                    # 若仅用固定窗口(90天)增量拼接，窗口外的历史数据仍持有旧因子，
                    # 拼接点会产生虚假价格断层，触发策略的动量/RSRS因子误报。
                    # 修复：在拼接前对比新旧数据在重叠日期上的 close 价格，
                    # 若差异超过 1e-4（相对误差），说明历史复权因子已全局改变，
                    # 强制触发全量重下载（删除本地文件，下次循环重头下载）。
                    _new_close_map = dict(zip(df["date"].astype(str), df["close"].astype(float)))
                    _old_dates = _existing_df["date"].astype(str).tolist()
                    _mismatch_count = 0
                    _check_count = 0
                    for _od in _old_dates[-90:]:  # [ITER26-FIX-BUG8] 扩展到90天（与HFQ_OVERLAP_DAYS一致）
                        if _od in _new_close_map:
                            _check_count += 1
                            _old_c = float(_existing_df.loc[_existing_df["date"].astype(str) == _od, "close"].iloc[0])
                            _new_c = _new_close_map[_od]
                            if _old_c > 1e-6 and abs(_new_c - _old_c) / _old_c > 1e-4:
                                _mismatch_count += 1
                    if _check_count > 0 and _mismatch_count >= 2:
                        # 历史复权因子已改变，必须全量重下
                        logger.warning(
                            f"[BUG-1.2 FIX] {code}: 检测到后复权因子改变"
                            f"（{_mismatch_count}/{_check_count} 个重叠日期 close 不一致），"
                            f"触发全量重下载以消除价格断层。"
                        )
                        if out_path.exists():
                            out_path.unlink()
                        _existing_df = None  # 强制全量模式
                        # 重新全量下载（重用已有 df，但需要扩展到 start 日期）
                        rs2 = bs.query_history_k_data_plus(
                            code,
                            "date,open,high,low,close,volume,amount",
                            start_date=start if start else "2010-01-01",
                            end_date=end,
                            frequency="d",
                            adjustflag="2",  # V10: QFQ前复权
                        )
                        if rs2.error_code == "0":
                            rows2: list = []
                            while rs2.next():
                                rows2.append(rs2.get_row_data())
                            if rows2:
                                df_full = pd.DataFrame(rows2, columns=rs2.fields)
                                df_full = _clean_df_static(df_full, code)
                                if df_full is not None and len(df_full) > 0:
                                    df = df_full
                    else:
                        df = (
                            pd.concat([_existing_df, df])
                            .drop_duplicates(subset="date", keep="last")  # 新数据覆盖旧复权价
                            .sort_values("date")
                            .reset_index(drop=True)
                        )
                except Exception:
                    pass  # merge 失败则只保存新数据

            # [断点续传数据损坏修复] 先写入临时文件，成功后原子重命名，防止中断导致数据损坏
            tmp_path = out_path.with_suffix(".tmp")
            df.to_parquet(str(tmp_path), index=False, compression="snappy")
            import os as _os
            _os.replace(str(tmp_path), str(out_path))
            return code, "ok", len(df)

        except Exception as e:
            logger.debug(f"[Worker] {code} 第{attempt+1}次异常: {e}")
            if attempt < _g_retry_times - 1:
                time.sleep(_g_retry_delay)

    return code, "error", 0


# ══════════════════════════════════════════════════════════════════════════════
# BaostockDownloader 主类
# ══════════════════════════════════════════════════════════════════════════════

class BaostockDownloader:
    """
    BaoStock A 股历史日线数据下载器。

    并发架构：ProcessPoolExecutor（每进程独立 baostock socket）
      · n_workers=8（默认，BaoStock 服务器舒适并发上限）：真正并行 8 路
      · 每进程启动约 3~5s（8进程合计 24~40s，全程下载约30~60分钟，启动开销<1%）
      · 真正并行：8进程 × 75KB/股 / 0.8s ≈ 750KB/s
      · n_workers=12 可在部分网络环境下进一步提速，但>16 可能触发服务器限速
      · [BUG-SPEED-1 FIX] 原 n_workers=16 导致并发过多触发 BaoStock 限速（500KB/s→几十KB/s）

    文件格式：每只股票一个 Parquet 文件
      data/daily_parquet/sh.600519.parquet （后复权日线）

    Parquet schema：
      date    str      "YYYY-MM-DD"
      open    float32  后复权开盘价
      high    float32  后复权最高价
      low     float32  后复权最低价
      close   float32  后复权收盘价
      volume  float32  成交量（手）
      amount  float32  成交额（元）
      code    str      "sh.600519"
    """

    def __init__(
        self,
        output_dir:  str   = "./data/daily_parquet_hfq",  # V10: QFQ专用目录
        n_workers:   int   = 8,      # [BUG-DL-N-WORKERS-DOC-MISMATCH FIX] 统一为文档推荐值8（BaoStock舒适并发上限）。原 n_workers=16 与顶部注释矛盾（BUG-SPEED-1-FIX 已明确8是上限，16会触发限速）
        retry_times: int   = 3,      # 原文档"8是上限"过于保守；真正上限约32，超过才触发限速
        retry_delay: float = 2.0,
    ) -> None:
        if not _BS_AVAILABLE:
            raise ImportError(
                "BaoStock 未安装，请执行：pip install baostock\n"
                "BaoStock 是免费的 A 股数据 API，无需注册账号。"
            )

        self.output_dir  = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_workers   = n_workers
        self.retry_times = retry_times
        self.retry_delay = retry_delay
        self._stock_list: Optional[List[str]] = None
        self._logged_in = False

    # ── 主进程连接管理 ────────────────────────────────────────────────────────

    def login(self) -> None:
        """主进程登录（用于股票列表查询等非批量操作）。"""
        if not self._logged_in:
            lg = bs.login()
            if lg.error_code != "0":
                raise ConnectionError(
                    f"BaoStock 登录失败: {lg.error_msg}\n"
                    "请检查网络连接。BaoStock 无需账号，直接联网可用。"
                )
            self._logged_in = True

    def logout(self) -> None:
        """主进程登出。"""
        if self._logged_in:
            bs.logout()
            self._logged_in = False

    def __enter__(self) -> "BaostockDownloader":
        self.login()
        return self

    def __exit__(self, *args) -> None:
        self.logout()

    # ── 股票池管理 ─────────────────────────────────────────────────────────────

    def download_stock_list(
        self,
        date_str:  Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> List[str]:
        """
        获取沪深两市全部 A 股代码列表（在主进程中执行）。
        [FIX-WEEKEND] 自动向前探测最多10天，跳过周末/节假日。
        [FIX-FILTER] type预过滤 + 严格代码白名单双重兜底。
        """
        self.login()
        if date_str is None:
            date_str = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

        try:
            data: list = []
            for days_back in range(0, 11):
                probe_date = (
                    date_str if days_back == 0
                    else (pd.to_datetime(date_str).date() - timedelta(days=days_back)).strftime("%Y-%m-%d")
                )
                rs = bs.query_all_stock(day=probe_date)
                if rs.error_code != "0":
                    continue
                rows_probe: list = []
                while rs.error_code == "0" and rs.next():
                    rows_probe.append(rs.get_row_data())
                if rows_probe:
                    data = rows_probe
                    logger.info(f"[BaostockDL] 有效交易日: {probe_date}")
                    break

            if not data:
                raise RuntimeError(
                    f"query_all_stock 在 {date_str} 前10天内均返回空数据！\n"
                    "请检查 BaoStock 服务或网络连接。"
                )

            df = pd.DataFrame(data, columns=rs.fields)

            import re as _re
            _STRICT_RE = _re.compile(
                r"^(?:"
                r"sh\.(?:600|601|603|605|688)\d{3}"
                r"|"
                r"sz\.(?:000|001|002|003|300|301)\d{3}"
                r")$"
            )
            if "type" in df.columns:
                pre_df = df[df["type"] == "1"].copy()
                if len(pre_df) < 3000:
                    logger.warning(f"[BaostockDL] type过滤后仅{len(pre_df)}只，回退白名单全量过滤")
                    pre_df = df.copy()
            else:
                pre_df = df.copy()

            a_share_df = pre_df[pre_df["code"].str.match(_STRICT_RE)].copy()
            logger.info(
                f"[BaostockDL] 过滤: {len(df)} → {len(pre_df)} → {len(a_share_df)}"
            )
            codes = a_share_df["code"].tolist()

            save_to = save_path or str(self.output_dir / "stock_list.csv")
            a_share_df.to_csv(save_to, index=False, encoding="utf-8-sig")
            logger.info(f"[BaostockDL] 共 {len(codes)} 只 A 股 → {save_to}")
            self._stock_list = codes
            return codes
        finally:
            self.logout()

    def load_stock_list(self, csv_path: Optional[str] = None) -> List[str]:
        """从本地 CSV 加载（不存在时自动下载）。"""
        csv_path = csv_path or str(self.output_dir / "stock_list.csv")
        if not Path(csv_path).exists():
            return self.download_stock_list()

        df = pd.read_csv(csv_path, dtype=str)
        code_col = next(
            (c for c in ["code", "股票代码", "Code", "CODE"] if c in df.columns),
            df.columns[0],
        )
        codes = [c for c in df[code_col].dropna().tolist() if str(c).strip()]

        if not codes:
            Path(csv_path).unlink(missing_ok=True)
            return self.download_stock_list(save_path=csv_path)

        import re as _re2
        _S = _re2.compile(
            r"^(?:sh\.(?:600|601|603|605|688)\d{3}|sz\.(?:000|001|002|003|300|301)\d{3})$"
        )
        raw = len(codes)
        codes = [c for c in codes if _S.match(c)]
        if raw != len(codes):
            logger.info(f"[BaostockDL] 白名单过滤: {raw} → {len(codes)}")

        if not codes:
            Path(csv_path).unlink(missing_ok=True)
            return self.download_stock_list(save_path=csv_path)

        logger.info(f"[BaostockDL] 从本地加载 {len(codes)} 只: {csv_path}")
        self._stock_list = codes
        return codes

    # ── 单只下载（主进程，供调试使用）────────────────────────────────────────

    def download_one(
        self,
        code:  str,
        start: str,
        end:   str,
        force: bool = False,
    ) -> Optional[pd.DataFrame]:
        """主进程下载单只（调试用）。批量下载请用 download_all()。"""
        HFQ_OVERLAP_DAYS = 90  # [D-02-FIX] 与 _download_worker 保持一致
        self.login()
        out_path = self.output_dir / f"{code}.parquet"
        _start = start
        _existing_df: Optional[pd.DataFrame] = None
        if not force and out_path.exists():
            try:
                existing = pd.read_parquet(str(out_path))
                if len(existing) > 0:
                    last_date = str(existing["date"].max())
                    if last_date >= (date.today() - timedelta(days=3)).strftime("%Y-%m-%d"):
                        return existing
                    # [D-02-FIX] HFQ回退：重下最后90天覆盖除权后的历史复权价
                    last_date_obj = pd.to_datetime(last_date).date()
                    overlap_start = (last_date_obj - timedelta(days=HFQ_OVERLAP_DAYS)).isoformat()
                    _start = max(overlap_start, start) if start else overlap_start
                    _existing_df = existing
            except Exception:
                pass

        for attempt in range(self.retry_times):
            try:
                rs = bs.query_history_k_data_plus(
                    code, "date,open,high,low,close,volume,amount",
                    start_date=_start, end_date=end,
                    frequency="d", adjustflag="2",  # V10: QFQ前复权
                )
                if rs.error_code != "0":
                    logger.warning(f"[BaostockDL] {code} K线查询失败: {rs.error_msg}")
                    return None
                rows: list = []
                while rs.next():
                    rows.append(rs.get_row_data())
                if not rows:
                    return None
                df = pd.DataFrame(rows, columns=rs.fields)
                df = _clean_df_static(df, code)
                if df is None:
                    return None
                if _existing_df is not None and not force:
                    # [D-02-FIX] keep="last" 使新下载的数据覆盖重叠窗口
                    # (与 _download_worker 的 drop_duplicates(keep="last") 一致)
                    df = pd.concat([_existing_df, df]).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
                elif out_path.exists() and not force:
                    try:
                        old = pd.read_parquet(str(out_path))
                        df = pd.concat([old, df]).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
                    except Exception:
                        pass
                df.to_parquet(str(out_path), index=False, compression="snappy")
                return df
            except Exception as e:
                logger.warning(f"[BaostockDL] {code} 第{attempt+1}次失败: {e}")
                if attempt < self.retry_times - 1:
                    time.sleep(self.retry_delay)
        return None

    # ── 批量下载（核心）───────────────────────────────────────────────────────

    def download_all(
        self,
        start:       str = "2015-01-01",
        end:         Optional[str] = None,
        codes:       Optional[List[str]] = None,
        force:       bool = False,
        progress_cb       = None,
    ) -> Dict[str, str]:
        """
        批量下载全市场日线数据（ProcessPoolExecutor 版）。

        并发架构：ProcessPoolExecutor + 每进程独立 baostock socket
          · n_workers=8（默认）：真正并行的8个独立 TCP 连接，BaoStock 舒适上限
          · [BUG-SPEED-1 FIX] 原 n_workers=16 导致服务器限速（速度从500KB/s降至几十KB/s）
          · 进程启动：8进程 × 3~5s ≈ 24~40s（全程下载约30~60分钟，启动开销<1%）
          · 预期吞吐：~750KB/s（8进程 × 75KB/股 × 1.2只/s/进程）

        Windows 打包注意：
          若使用 PyInstaller 打包，需在 main() 最前面加：
              import multiprocessing; multiprocessing.freeze_support()
          直接运行 python main.py 无需任何处理。

        Parameters
        ----------
        start        下载起始日（建议 ≥2015 保证 RSRS zscore 数据充足）
        end          结束日期，None=今日
        codes        None=自动加载全市场股票列表
        force        True=强制全量重新下载（忽略增量逻辑）
        progress_cb  fn(done, total, code) 进度回调

        Returns
        -------
        {code: "ok"/"skip"/"error"}
        """
        if end is None:
            end = date.today().strftime("%Y-%m-%d")

        # [PATCH-2] 防御 start=None（调用方 cfg.get 位置参数 bug 可能传入 None）
        if not start:
            logger.warning("[BaostockDL] start 参数为空，已回退至默认值 2015-01-01")
            start = "2015-01-01"

        results: Dict[str, str] = {}
        done = 0
        t0 = time.perf_counter()

        try:
            if codes is None:
                codes = self.load_stock_list()

            total = len(codes)
            logger.info(
                f"[BaostockDL] 开始批量下载: {total}只, {start}~{end}, "
                f"workers={self.n_workers}"
            )

            # 60秒滚动速度窗口
            _window: List[Tuple[float, int]] = []
            _last_display = [t0]

            def _rolling_speed(now: float) -> float:
                cutoff = now - 60.0
                idx = 0
                while idx < len(_window) and _window[idx][0] < cutoff:
                    idx += 1
                del _window[:idx]
                if len(_window) < 2:
                    elapsed = now - t0
                    return done / elapsed if elapsed > 0.1 else 0.0
                t_span = _window[-1][0] - _window[0][0]
                d_span = _window[-1][1] - _window[0][1]
                return d_span / t_span if t_span > 0.1 else 0.0

            tasks = [(c, start, end, force) for c in codes]

            # ProcessPoolExecutor：每个子进程独立 baostock 会话（独立 TCP socket）
            # initializer 在每个子进程启动时调用一次 bs.login()
            # 主进程不参与下载，无需持有 baostock 连接
            with ProcessPoolExecutor(
                max_workers=self.n_workers,
                initializer=_pool_initializer,
                initargs=(str(self.output_dir), self.retry_times, self.retry_delay),
            ) as executor:
                futures = {
                    executor.submit(_worker_download_one, task): task[0]
                    for task in tasks
                }

                for future in as_completed(futures):
                    code_, status, _nrows = future.result()
                    done += 1
                    results[code_] = status

                    now = time.perf_counter()
                    _window.append((now, done))

                    if progress_cb:
                        progress_cb(done, total, code_)
                    else:
                        if done == 1 or (now - _last_display[0] >= 5.0) or done == total:
                            _last_display[0] = now
                            spd  = _rolling_speed(now)
                            eta  = (total - done) / spd if spd > 0.01 else 0.0
                            ok_n = sum(1 for v in results.values() if v == "ok")
                            er_n = sum(1 for v in results.values() if v == "error")
                            logger.info(
                                f"  [{done}/{total}] ok={ok_n} err={er_n} "
                                f"速度={spd:.2f}只/s 预计剩余={eta:.0f}s"
                            )

        except KeyboardInterrupt:
            logger.warning("[BaostockDL] 用户中断，已下载数据已保存到磁盘")

        elapsed  = time.perf_counter() - t0
        ok_cnt   = sum(1 for v in results.values() if v == "ok")
        err_cnt  = sum(1 for v in results.values() if v == "error")
        skip_cnt = sum(1 for v in results.values() if v == "skip")
        avg_spd  = (ok_cnt + skip_cnt) / elapsed if elapsed > 0 else 0
        logger.info(
            f"[BaostockDL] 完成: 成功={ok_cnt} 跳过={skip_cnt} 失败={err_cnt} "
            f"耗时={elapsed:.0f}s 平均={avg_spd:.2f}只/s"
        )
        return results

    # ── 指数成分股 ────────────────────────────────────────────────────────────

    def download_index_components(
        self,
        index_code: str = "sh.000300",
        date_str:   Optional[str] = None,
        start:      str = "2015-01-01",
        end:        Optional[str] = None,
    ) -> List[str]:
        """下载指数成分股（沪深300/中证500）并批量下载历史数据。"""
        self.login()
        if date_str is None:
            date_str = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        if end is None:
            end = date.today().strftime("%Y-%m-%d")
        try:
            method_map = {
                "sh.000300": bs.query_hs300_stocks,
                "sh.000905": bs.query_zz500_stocks,
            }
            # [BUG-NEW-06 FIX] 原版对不支持的指数（如中证1000 sh.000852）静默降级为全市场，
            # 用户误以为下载了1000只，实际下载5000+只，耗时严重超预期。
            # 现改为明确报错，避免无声的错误行为。
            if index_code not in method_map:
                _supported = list(method_map.keys())
                raise ValueError(
                    f"[BaostockDL] 不支持的指数代码: {index_code}\n"
                    f"  Baostock 当前仅提供: {_supported}\n"
                    f"  中证1000 (sh.000852) 等暂不支持，请改用全市场模式或手动指定股票列表"
                )

            rs = method_map[index_code](date=date_str)
            if rs.error_code != "0":
                raise RuntimeError(f"获取 {index_code} 成分股失败: {rs.error_msg}")

            rows: list = []
            while rs.next():
                rows.append(rs.get_row_data())
            df = pd.DataFrame(rows, columns=rs.fields)
            codes = df["code"].tolist()
            logger.info(f"[BaostockDL] {index_code} 共 {len(codes)} 只")
            self.logout()
            self.download_all(start=start, end=end, codes=codes)
            return codes
        finally:
            self.logout()

    # ── 状态检查 ──────────────────────────────────────────────────────────────

    def get_data_status(self) -> Dict:
        """检查已下载数据状态（抽样100只）。"""
        files = [f for f in self.output_dir.glob("*.parquet") if f.stem != "stock_list"]
        if not files:
            return {"total_stocks": 0, "total_size_mb": 0.0, "status": "无数据，请先执行下载"}

        total_mb = sum(f.stat().st_size for f in files) / 1024 / 1024
        cutoff = (date.today() - timedelta(days=3)).strftime("%Y-%m-%d")
        needs_update, oldest, newest = 0, "9999-99-99", "0000-00-00"

        for f in files[:min(100, len(files))]:
            try:
                df = pd.read_parquet(f, columns=["date"])
                if len(df) > 0:
                    oldest = min(oldest, df["date"].min())
                    newest = max(newest, df["date"].max())
                    if df["date"].max() < cutoff:
                        needs_update += 1
            except Exception:
                pass

        return {
            "total_stocks":          len(files),
            "total_size_mb":         round(total_mb, 1),
            "oldest_date":           oldest if oldest != "9999-99-99" else None,
            "newest_date":           newest if newest != "0000-00-00" else None,
            "stocks_needing_update": needs_update,
            "parquet_dir":           str(self.output_dir),
            "status":                "正常" if needs_update == 0 else f"{needs_update} 只过期",
        }
