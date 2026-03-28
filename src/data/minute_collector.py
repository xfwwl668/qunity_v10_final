# 根据 Q-UNITY V8 计划书 v2.1 实现
"""
分钟数据采集器
对应计划书 §7.4 MinuteDataCollector

功能：
    - 从通达信（TDX）采集 1min / 5min 等周期 K 线
    - 按月分片存储为 Parquet，路径 data/minute/{code}/{YYYYMM}.parquet
    - 支持断点续传：月文件存在则跳过
    - 提供 validate_data() 基于 MAD 的异常检测
    - batch_collect() 支持多股票并发采集

存储格式（Parquet 列）：
    datetime    : pandas Timestamp，精确到分钟
    open        : float32
    high        : float32
    low         : float32
    close       : float32
    volume      : float32  — 单位：手（100股）
    amount      : float32  — 单位：元（可选，TDX并非所有接口都提供）
    adj_type    : str      — 固定为 "raw"（分钟数据不复权，见计划书 §11.5）

注意：
    - 分钟级数据不复权。回测时 UltraShortStrategy 使用原始价格计算动量，
      与实盘 TDX 实时行情保持一致，避免参数迁移偏差。
    - 分钟数据存在停牌、缺数等问题，validate_data() 负责检测。
"""
from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# TDX 分钟K线周期常量
PERIOD_1MIN  = 8   # TDX category: 8 = 1分钟
PERIOD_5MIN  = 0   # TDX category: 0 = 5分钟（pytdx 内部常量）
PERIOD_15MIN = 1
PERIOD_30MIN = 2
PERIOD_60MIN = 3

# 每次 TDX 请求的最大 K 线条数
TDX_MAX_BARS = 800

# MAD 异常检测倍数
MAD_THRESHOLD = 10.0


class MinuteDataCollector:
    """
    TDX 分钟数据采集器。

    Parameters
    ----------
    host         : TDX 服务器地址，默认使用国内标准节点
    port         : TDX 服务器端口，默认 7709
    period       : K线周期，使用模块常量 PERIOD_1MIN / PERIOD_5MIN 等
    data_root    : 数据根目录，Parquet 文件保存至 {data_root}/minute/{code}/{YYYYMM}.parquet
    connect_timeout : 连接超时秒数
    max_workers  : batch_collect() 线程并发数

    Example
    -------
    >>> collector = MinuteDataCollector(period=PERIOD_5MIN, data_root="data")
    >>> df = collector.fetch_minute_bars("000001", n_bars=2400)
    >>> collector.save_minute_bars(df, "000001")
    >>> collector.batch_collect(["000001", "600000"], n_bars=4800)
    """

    DEFAULT_HOSTS: List[Tuple[str, int]] = [
        ("116.205.171.132", 7709),
        ("110.41.147.114", 7709),
        ("124.71.9.153", 7709),
        ("110.41.2.72", 7709),
        ("116.205.163.254", 7709),
        ("116.205.183.150", 7709),
        ("124.71.187.72", 7709),
        ("124.70.133.119", 7709),
        ("111.230.186.52", 7709),
        ("123.60.73.44", 7709),
        ("123.60.70.228", 7709),
        ("123.60.84.66", 7709),
        ("119.97.185.59", 7709),
        ("60.12.136.250", 7709),
        ("111.229.247.189", 7709),
        ("122.51.120.217", 7709),
        ("115.238.90.165", 7709),
        ("124.71.187.122", 7709),
        ("121.36.225.169", 7709),
        ("122.51.232.182", 7709),
        ("118.25.98.114", 7709),
        ("124.70.199.56", 7709),
    ]

    def __init__(
        self,
        host: Optional[str] = None,
        port: int = 7709,
        period: int = PERIOD_5MIN,
        data_root: str = "data",
        connect_timeout: int = 10,
        max_workers: int = 4,
    ) -> None:
        self.host = host
        self.port = port
        self.period = period
        self.data_root = Path(data_root)
        self.connect_timeout = connect_timeout
        self.max_workers = max_workers
        self._api = None  # lazy connect

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------

    def _get_api(self):
        """惰性初始化 TDX API 连接（线程不安全，batch 时每线程独立创建）"""
        if self._api is not None:
            return self._api
        try:
            from pytdx.hq import TdxHq_API
        except ImportError as e:
            raise ImportError(
                "pytdx 未安装，请运行: pip install pytdx"
            ) from e

        api = TdxHq_API(raise_exception=True)
        hosts = [(self.host, self.port)] if self.host else self.DEFAULT_HOSTS
        for h, p in hosts:
            try:
                # Bug 9 Fix: pass time_out parameter for connection timeout
                api.connect(h, p, time_out=self.connect_timeout)
                logger.info(f"[MinuteCollector] TDX 连接成功: {h}:{p}")
                self._api = api
                return api
            except Exception as e:
                logger.warning(f"[MinuteCollector] TDX 连接失败 {h}:{p}: {e}")
        raise ConnectionError("[MinuteCollector] 所有 TDX 节点连接失败")

    def _close_api(self) -> None:
        if self._api is not None:
            try:
                self._api.disconnect()
            except Exception:
                pass
            self._api = None

    # ------------------------------------------------------------------
    # fetch_minute_bars
    # ------------------------------------------------------------------

    def fetch_minute_bars(
        self,
        code: str,
        n_bars: int = 2400,
        start_dt: Optional[datetime] = None,
        end_dt: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        从 TDX 拉取分钟 K 线数据。

        Parameters
        ----------
        code      : 股票代码，"000001"（无前缀，自动判断市场）
        n_bars    : 拉取最近 n_bars 根 K 线（与 start_dt/end_dt 二选一）
        start_dt  : 开始时间（若提供则忽略 n_bars，按时间范围拉取）
        end_dt    : 结束时间（默认当前时间）

        Returns
        -------
        DataFrame，列 ["datetime","open","high","low","close","volume","amount"]，
        按 datetime 升序，dtype float32（除 datetime 外）。
        adj_type 列固定为 "raw"。

        Raises
        ------
        ConnectionError : TDX 连接失败
        ValueError      : 返回数据为空或格式异常
        """
        market = self._market_code(code)
        api = self._get_api()

        if start_dt is not None:
            # 按时间范围拉取（分批请求）
            df = self._fetch_by_range(api, market, code, start_dt, end_dt)
        else:
            # 拉取最近 n_bars 条
            df = self._fetch_n_bars(api, market, code, n_bars)

        if df.empty:
            logger.warning(f"[MinuteCollector] {code} 返回空数据")
            return df

        df["adj_type"] = "raw"
        return df

    def _fetch_n_bars(
        self,
        api,
        market: int,
        code: str,
        n_bars: int,
    ) -> pd.DataFrame:
        """分批请求最近 n_bars 根 K 线"""
        all_rows: List[pd.DataFrame] = []
        remaining = n_bars
        start_pos = 0

        while remaining > 0:
            batch = min(remaining, TDX_MAX_BARS)
            try:
                raw = api.get_security_bars(
                    category=self.period,
                    market=market,
                    code=code,
                    start=start_pos,
                    count=batch,
                )
            except Exception as e:
                logger.error(f"[MinuteCollector] {code} 请求失败: {e}")
                break

            if not raw:
                break

            df_batch = pd.DataFrame(raw)
            all_rows.append(df_batch)
            remaining -= len(df_batch)
            start_pos += len(df_batch)

            if len(df_batch) < batch:
                break  # 已到数据尽头

        if not all_rows:
            return pd.DataFrame()

        df = pd.concat(all_rows, ignore_index=True)
        return self._normalize_df(df)

    def _fetch_by_range(
        self,
        api,
        market: int,
        code: str,
        start_dt: datetime,
        end_dt: Optional[datetime],
    ) -> pd.DataFrame:
        """按时间范围拉取（反复请求直到覆盖 start_dt）"""
        if end_dt is None:
            end_dt = datetime.now()

        all_rows: List[pd.DataFrame] = []
        start_pos = 0

        while True:
            try:
                raw = api.get_security_bars(
                    category=self.period,
                    market=market,
                    code=code,
                    start=start_pos,
                    count=TDX_MAX_BARS,
                )
            except Exception as e:
                logger.error(f"[MinuteCollector] {code} 请求失败: {e}")
                break

            if not raw:
                break

            # [BUG-V6-HIGH-1 FIX] 在过滤日期范围之前，先检查未过滤批次的时间边界。
            # 原实现：过滤后若 df_batch 为空立即 break，注释说"所有数据都早于 start_dt"。
            # 实际上有两种情况：
            #   Case A: 批次最晚时间 < start_dt  → 已超出左侧边界 → 应 break
            #   Case B: 批次最早时间 > end_dt    → 尚未进入右侧区间 → 应 continue
            # TDX 从 start_pos=0 返回最近数据，随 start_pos 递增向历史方向推进。
            # 当请求历史区间时（end_dt 距今较远），前 N 批数据全在 end_dt 之后（Case B），
            # 原代码首批过滤为空就 break，导致整个函数静默返回空 DataFrame。
            df_raw = self._normalize_df(pd.DataFrame(raw))
            if df_raw.empty:
                break

            batch_latest   = df_raw["datetime"].max()
            batch_earliest = df_raw["datetime"].min()

            if batch_latest < start_dt:
                # Case A: 整个批次都在 start_dt 左侧，继续向历史方向已无意义
                break

            if batch_earliest > end_dt:
                # Case B: 整个批次都在 end_dt 右侧，继续向历史方向推进
                start_pos += TDX_MAX_BARS
                continue

            # 过滤到目标时间范围
            mask = (df_raw["datetime"] >= start_dt) & (
                df_raw["datetime"] <= end_dt
            )
            df_batch = df_raw[mask]
            if not df_batch.empty:
                all_rows.append(df_batch)

            # 检查是否已覆盖 start_dt（无需继续往历史方向请求）
            if batch_earliest <= start_dt:
                break
            start_pos += TDX_MAX_BARS

        if not all_rows:
            return pd.DataFrame()

        df = pd.concat(all_rows, ignore_index=True)
        df = df.drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)
        return df

    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """规范化 TDX 返回的原始 DataFrame"""
        col_map = {
            "datetime": "datetime",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "vol": "volume",
            "volume": "volume",
            "amount": "amount",
        }
        rename = {k: v for k, v in col_map.items() if k in df.columns}
        df = df.rename(columns=rename)

        if "datetime" not in df.columns:
            logger.error("[MinuteCollector] 返回数据缺少 datetime 列")
            return pd.DataFrame()

        df["datetime"] = pd.to_datetime(df["datetime"])

        # Bug 12 Fix: Handle missing amount field; estimate if not provided
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
            else:
                df[col] = np.float32(0.0)

        # amount may not always be provided by TDX; estimate from volume * close
        if "amount" in df.columns:
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce").astype("float32")
        else:
            df["amount"] = (df["volume"] * df["close"]).astype("float32")

        df = df.sort_values("datetime").reset_index(drop=True)
        return df[["datetime", "open", "high", "low", "close", "volume", "amount"]]

    @staticmethod
    def _market_code(code: str) -> int:
        """根据股票代码判断市场编号（0=深圳, 1=上海）"""
        code = str(code).strip()
        if code.startswith("6"):
            return 1  # 上海
        return 0       # 深圳

    # ------------------------------------------------------------------
    # save_minute_bars
    # ------------------------------------------------------------------

    def save_minute_bars(
        self,
        df: pd.DataFrame,
        code: str,
        overwrite: bool = False,
    ) -> Dict[str, Path]:
        """
        将分钟 K 线 DataFrame 按月分片写入 Parquet。

        存储路径：{data_root}/minute/{code}/{YYYYMM}.parquet

        Parameters
        ----------
        df        : fetch_minute_bars() 返回的 DataFrame
        code      : 股票代码
        overwrite : 若月文件已存在是否覆盖（默认 False，断点续传用途）

        Returns
        -------
        Dict[month_str, saved_path]，例如 {"202401": Path("data/minute/000001/202401.parquet")}
        """
        if df.empty:
            logger.warning(f"[MinuteCollector] {code} 空 DataFrame，跳过保存")
            return {}

        base_dir = self.data_root / "minute" / code
        base_dir.mkdir(parents=True, exist_ok=True)

        df = df.copy()
        df["month"] = df["datetime"].dt.strftime("%Y%m")

        saved: Dict[str, Path] = {}
        for month, group in df.groupby("month"):
            fpath = base_dir / f"{month}.parquet"
            if fpath.exists() and not overwrite:
                logger.debug(f"[MinuteCollector] {code} {month} 已存在，跳过（断点续传）")
                continue
            group = group.drop(columns=["month"])
            group.to_parquet(
                fpath,
                index=False,
                compression="zstd",
                engine="pyarrow",
            )
            saved[str(month)] = fpath
            logger.debug(f"[MinuteCollector] {code} {month} 已写入 {fpath}")

        logger.info(
            f"[MinuteCollector] {code} 新写入 {len(saved)} 个月文件"
        )
        return saved

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # scan_best_hosts  （预扫描，选出延迟最低的可用节点）
    # ------------------------------------------------------------------

    @classmethod
    def scan_best_hosts(
        cls,
        hosts: Optional[List[Tuple[str, int]]] = None,
        top_n: int = 6,
        timeout: float = 4.0,
    ) -> List[Tuple[str, int]]:
        """
        并发测速所有候选节点，返回延迟最低的 top_n 个。
        timeout 秒内无响应视为不可用。
        """
        import socket
        from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
        candidates = hosts or cls.DEFAULT_HOSTS

        def _probe(hp: Tuple[str, int]) -> Tuple[float, str, int]:
            h, p = hp
            t0 = time.perf_counter()
            try:
                s = socket.create_connection((h, p), timeout=timeout)
                s.close()
                return time.perf_counter() - t0, h, p
            except Exception:
                return float("inf"), h, p

        results = []
        with ThreadPoolExecutor(max_workers=min(len(candidates), 24)) as ex:
            futs = {ex.submit(_probe, hp): hp for hp in candidates}
            for f in _as_completed(futs):
                lat, h, p = f.result()
                if lat < float("inf"):
                    results.append((lat, h, p))

        results.sort()
        best = [(h, p) for _, h, p in results[:top_n]]
        print(f"  [TDX预扫] {len(candidates)}个节点中找到{len(results)}个可用，"
              f"选最快{len(best)}个（最低延迟={results[0][0]*1000:.0f}ms）"
              if results else f"  [TDX预扫] 所有节点均不可达，使用默认列表")
        return best if best else [candidates[0]]

    # ------------------------------------------------------------------
    # batch_collect
    # ------------------------------------------------------------------

    def batch_collect(
        self,
        codes: List[str],
        n_bars: int = 4800,
        start_dt: Optional[datetime] = None,
        end_dt: Optional[datetime] = None,
        max_workers: Optional[int] = None,
        skip_existing: bool = True,
    ) -> Dict[str, int]:
        """
        批量采集多只股票的分钟数据，多线程并发执行。

        Parameters
        ----------
        codes        : 股票代码列表
        n_bars       : 每只股票拉取的最近 K 线数量
        start_dt     : 开始时间（若提供则按时间范围采集）
        end_dt       : 结束时间
        max_workers  : 并发线程数（默认使用初始化时的 self.max_workers）
        skip_existing: 若该股票当前月文件已存在则跳过（断点续传）

        Returns
        -------
        Dict[code, n_bars_saved]，每只股票实际保存的 K 线总数

        Notes
        -----
        每个线程独立创建 TDX API 连接，避免共享连接的线程安全问题。
        """
        workers = max_workers or self.max_workers
        results: Dict[str, int] = {}
        total = len(codes)
        logger.info(f"[MinuteCollector] 开始批量采集 {total} 只股票，并发={workers}")

        # 预扫描：并发测速，选最快的可用节点（避免逐个尝试导致启动卡顿）
        if self.host:
            best_hosts = [(self.host, self.port)]
        else:
            import sys as _sys
            _sys.stdout.write("  [TDX] 正在预扫描节点速度...\n")
            _sys.stdout.flush()
            best_hosts = MinuteDataCollector.scan_best_hosts(
                hosts=self.DEFAULT_HOSTS, top_n=workers + 2, timeout=4.0)
            if not best_hosts:
                raise ConnectionError("[MinuteCollector] 所有 TDX 节点均不可达")

        # 每个线程轮流分配一个最快节点
        import itertools as _itools
        host_cycle = _itools.cycle(best_hosts)
        host_assignments: Dict[int, Tuple[str, int]] = {}  # thread_idx -> (host, port)

        def _collect_one(args: Tuple[int, str]) -> Tuple[str, int]:
            idx, code = args
            assigned_host, assigned_port = host_assignments.get(idx, best_hosts[0])
            # 每线程独立创建采集器（独立 TDX 连接，使用预分配的最快节点）
            collector = MinuteDataCollector(
                host=assigned_host,
                port=assigned_port,
                period=self.period,
                data_root=str(self.data_root),
                connect_timeout=self.connect_timeout,
                max_workers=1,
            )
            try:
                if skip_existing:
                    # [D-07-FIX] 检查所有目标月份文件是否已存在，而非仅检查当前月
                    # 原逻辑只检查当前月文件，若请求历史回填（start_dt指定过去日期）
                    # 则即使目标月份文件已完整存在，仍会重新采集，浪费带宽。
                    _skip = True
                    if start_dt is not None and end_dt is not None:
                        # 计算目标月份列表
                        from datetime import timedelta as _td
                        _d = start_dt.replace(day=1)
                        _target_months: list = []
                        while _d <= end_dt:
                            _target_months.append(_d.strftime("%Y%m"))
                            # 移至下月
                            if _d.month == 12:
                                _d = _d.replace(year=_d.year + 1, month=1)
                            else:
                                _d = _d.replace(month=_d.month + 1)
                        for _m in _target_months:
                            _fp = self.data_root / "minute" / code / f"{_m}.parquet"
                            if not _fp.exists():
                                _skip = False
                                break
                    else:
                        # 无明确起止时，回退到仅检查当前月
                        current_month = datetime.now().strftime("%Y%m")
                        fpath = self.data_root / "minute" / code / f"{current_month}.parquet"
                        _skip = fpath.exists()
                    if _skip:
                        logger.debug(f"[MinuteCollector] {code} 所有目标月份文件已存在，跳过")
                        return code, 0

                df = collector.fetch_minute_bars(
                    code, n_bars=n_bars, start_dt=start_dt, end_dt=end_dt
                )
                saved = collector.save_minute_bars(df, code, overwrite=False)
                n_saved = sum(
                    len(pd.read_parquet(p)) for p in saved.values()
                ) if saved else 0
                return code, n_saved
            except Exception as e:
                logger.error(f"[MinuteCollector] {code} 采集失败: {e}")
                return code, -1
            finally:
                collector._close_api()

        # 预分配节点（轮询分配最快节点给各线程）
        indexed_codes = [(i, c) for i, c in enumerate(codes)]
        for i, _ in indexed_codes:
            host_assignments[i] = best_hosts[i % len(best_hosts)]

        import sys as _sys
        t0_batch = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_collect_one, ic): ic[1] for ic in indexed_codes}
            done = 0
            ok_n = 0
            for future in as_completed(futures):
                code, n = future.result()
                results[code] = n
                done += 1
                if n >= 0:
                    ok_n += 1
                # 每只股票完成后都刷新进度（stdout 实时显示）
                el = time.perf_counter() - t0_batch
                spd = done / el if el > 0 else 1
                eta = int((total - done) / spd) if spd > 0 else 0
                _sys.stdout.write(
                    f"\r  [{done}/{total}] 成功={ok_n} 失败={done-ok_n} "
                    f"速度={spd:.1f}只/s  ETA {eta//60}m{eta%60:02d}s  ")
                _sys.stdout.flush()
                if done % 50 == 0 or done == total:
                    logger.info(
                        f"[MinuteCollector] 进度 {done}/{total}，成功={ok_n}")
        print()  # 换行

        success = sum(1 for v in results.values() if v >= 0)
        logger.info(
            f"[MinuteCollector] 批量采集完成，成功={success}/{total}"
        )
        return results

    # ------------------------------------------------------------------
    # validate_data
    # ------------------------------------------------------------------

    def validate_data(
        self,
        code: str,
        months: Optional[List[str]] = None,
        mad_multiplier: float = 5.0,  # Bug 11 Fix: reduced from 10.0 to 5.0 for better sensitivity
    ) -> Dict[str, object]:
        """
        对已保存的分钟数据进行异常检测（基于 MAD — 中位数绝对偏差）。

        检测项目：
            1. 缺失值比例（NaN rate）
            2. 零成交量比例（trading halt detection）
            3. 收盘价 MAD 异常点（价格突变）
            4. OHLC 逻辑异常（如 high < low）
            5. 时间间隔异常（缺失 K 线或重复 K 线）

        Parameters
        ----------
        code           : 股票代码
        months         : 月份列表 ["202401","202402"]，默认检查所有已有月文件
        mad_multiplier : MAD 异常判断倍数，默认 10

        Returns
        -------
        Dict 汇总报告，含 total_bars, nan_rate, halt_rate, anomaly_count,
        anomaly_indices, logic_error_count, missing_bars 等字段
        """
        base_dir = self.data_root / "minute" / code
        if not base_dir.exists():
            return {"error": f"{code} 数据目录不存在"}

        # 发现所有月文件
        all_parquets = sorted(base_dir.glob("*.parquet"))
        if months is not None:
            all_parquets = [p for p in all_parquets if p.stem in months]

        if not all_parquets:
            return {"error": f"{code} 无 Parquet 文件"}

        dfs = [pd.read_parquet(p) for p in all_parquets]
        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values("datetime").reset_index(drop=True)

        report: Dict[str, object] = {
            "code": code,
            "total_bars": len(df),
            "months_checked": [p.stem for p in all_parquets],
        }

        close = pd.to_numeric(df["close"], errors="coerce").values

        # 1. NaN rate
        nan_rate = float(np.isnan(close).mean())
        report["nan_rate"] = nan_rate

        # 2. 零成交量比例
        vol = pd.to_numeric(df.get("volume", pd.Series()), errors="coerce").fillna(0).values
        halt_rate = float((vol == 0).mean())
        report["halt_rate"] = halt_rate

        # 3. MAD 异常检测
        valid_close = close[~np.isnan(close)]
        anomaly_count = 0
        anomaly_indices: List[int] = []
        if len(valid_close) >= 10:
            returns = np.diff(valid_close) / (valid_close[:-1] + 1e-10)
            median = np.median(returns)
            mad = np.median(np.abs(returns - median))
            if mad > 1e-10:
                outlier_mask = np.abs(returns - median) > mad_multiplier * mad
                anomaly_indices = list(np.where(outlier_mask)[0] + 1)
                anomaly_count = int(outlier_mask.sum())
        report["anomaly_count"] = anomaly_count
        report["anomaly_indices"] = anomaly_indices[:20]  # 最多返回前20个

        # 4. OHLC 逻辑检验
        logic_errors = 0
        for col_h, col_l in [("high", "low"), ("high", "close"), ("high", "open")]:
            if col_h in df.columns and col_l in df.columns:
                h = pd.to_numeric(df[col_h], errors="coerce").values
                l_ = pd.to_numeric(df[col_l], errors="coerce").values
                logic_errors += int(np.nansum(h < l_))
        report["logic_error_count"] = logic_errors

        # 5. 时间间隔异常
        if len(df) >= 2:
            dts = pd.to_datetime(df["datetime"])
            gaps = dts.diff().dropna()
            period_minutes = {
                PERIOD_1MIN: 1,
                PERIOD_5MIN: 5,
                PERIOD_15MIN: 15,
                PERIOD_30MIN: 30,
                PERIOD_60MIN: 60,
            }.get(self.period, 5)
            expected_gap = pd.Timedelta(minutes=period_minutes)
            # [BUG-V6-MED-1 FIX] 原实现将全部 gaps 与 expected_gap*2 比较，
            # 但每个跨日间隙（15:00→9:30，约 18h）都远超任何 K 线周期的 2 倍，
            # 导致 N 个交易日产生 N-1 个误报，missing_bars_estimate 完全失效。
            # 修复：只统计盘中内（单日内 < 60 分钟）的超大间隙，
            # 过滤掉跨交易日的正常隔夜/午休间隙（>= 60 分钟）。
            intraday_gaps = gaps[gaps < pd.Timedelta(minutes=60)]
            large_gaps = int((intraday_gaps > expected_gap * 2).sum())
            report["missing_bars_estimate"] = large_gaps
        else:
            report["missing_bars_estimate"] = 0

        # 综合评级
        ok = (
            nan_rate < 0.05
            and anomaly_count < len(df) * 0.01
            and logic_errors == 0
        )
        report["pass"] = ok
        report["summary"] = (
            "OK" if ok else
            f"WARN: nan={nan_rate:.2%}, anomalies={anomaly_count}, logic_err={logic_errors}"
        )

        return report


# ══════════════════════════════════════════════════════════════════════════════
# 动态选股目标列表生成器
# ══════════════════════════════════════════════════════════════════════════════

class DynamicTargetSelector:
    """
    从 `data/daily_parquet/` 日线 Parquet 文件中动态筛选分钟数据同步目标。

    筛选逻辑（三层漏斗）：
        Layer-1  基础过滤：剔除 ST / *ST、上市不足 60 个交易日的新股
        Layer-2  流动性排名：按近 5 个交易日平均成交额（amount）取全市场前 N 名
        Layer-3  白名单叠加：强制加入核心权重股（如沪深300龙头），去重后输出

    Parameters
    ----------
    parquet_dir : str | Path
        日线 Parquet 目录（每只股票一个 {code}.parquet）
    whitelist   : list[str]
        白名单代码列表，格式 "000001"（6位，无前缀）
    top_n       : int
        流动性排名取前 N 名，默认 300
    min_ipo_days: int
        上市最少交易日数，默认 60
    n_workers   : int
        读取 Parquet 的并发线程数，默认 8

    Example
    -------
    >>> sel = DynamicTargetSelector(
    ...     parquet_dir="data/daily_parquet",
    ...     whitelist=["600519", "000858", "300750"],
    ...     top_n=300,
    ... )
    >>> codes = sel.get_target_list()   # → List[str]，约 300-350 只
    """

    def __init__(
        self,
        parquet_dir: str,
        whitelist: Optional[List[str]] = None,
        top_n: int = 300,
        min_ipo_days: int = 60,
        n_workers: int = 8,
    ) -> None:
        self.parquet_dir  = Path(parquet_dir)
        self.whitelist    = [str(c).strip().zfill(6) for c in (whitelist or [])]
        self.top_n        = top_n
        self.min_ipo_days = min_ipo_days
        self.n_workers    = n_workers

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def get_target_list(self) -> List[str]:
        """
        返回最终同步目标代码列表（6 位数字字符串，无市场前缀）。

        Returns
        -------
        List[str]
            去重、去白名单外排名后合并的最终列表，长度 ≤ top_n + len(whitelist)

        Raises
        ------
        FileNotFoundError : parquet_dir 不存在或无 Parquet 文件
        """
        if not self.parquet_dir.exists():
            raise FileNotFoundError(
                f"[DynamicTargetSelector] 日线目录不存在: {self.parquet_dir}"
            )

        pq_files = sorted(self.parquet_dir.glob("*.parquet"))
        # 排除 stock_list.parquet（若存在）
        pq_files = [f for f in pq_files if f.stem != "stock_list"]

        if not pq_files:
            raise FileNotFoundError(
                f"[DynamicTargetSelector] {self.parquet_dir} 中无 Parquet 文件"
            )

        logger.info(
            f"[DynamicTargetSelector] 扫描 {len(pq_files)} 只股票 Parquet..."
        )

        # 并行读取末尾数据
        records = self._load_tail_parallel(pq_files)

        # Layer-1: 基础过滤
        records = self._filter_basic(records)
        logger.info(
            f"[DynamicTargetSelector] Layer-1 基础过滤后: {len(records)} 只"
        )

        # Layer-2: 流动性排名
        top_codes = self._rank_liquidity(records)
        logger.info(
            f"[DynamicTargetSelector] Layer-2 流动性前{self.top_n}: {len(top_codes)} 只"
        )

        # Layer-3: 合并白名单
        final = self._merge_whitelist(top_codes)
        logger.info(
            f"[DynamicTargetSelector] Layer-3 合并白名单后: {len(final)} 只"
        )

        return final

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _load_tail_parallel(
        self, pq_files: List[Path]
    ) -> List[Dict[str, object]]:
        """
        并行读取每只股票 Parquet 的末尾数据，提取：
            code, row_count, name (若存在), last_date, avg_amount_5d
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        records: List[Dict[str, object]] = []

        def _read_one(fpath: Path) -> Optional[Dict[str, object]]:
            code = fpath.stem  # e.g. "000001" 或 "sz.000001"
            # 统一去掉市场前缀
            if "." in code:
                code = code.split(".", 1)[1]
            code = code.strip().zfill(6)
            try:
                df = pd.read_parquet(fpath)
                if df.empty:
                    return None
                df = df.sort_values("date").reset_index(drop=True)
                row_count = len(df)
                last_date = df["date"].iloc[-1] if "date" in df.columns else None
                # 近5日平均成交额
                amt_col = "amount" if "amount" in df.columns else None
                if amt_col:
                    tail = df[amt_col].tail(5)
                    avg_amount_5d = float(
                        pd.to_numeric(tail, errors="coerce").fillna(0).mean()
                    )
                else:
                    avg_amount_5d = 0.0
                # 股票名称（部分 Parquet 可能有 name 列）
                name = ""
                if "name" in df.columns:
                    name = str(df["name"].iloc[-1])

                return {
                    "code": code,
                    "row_count": row_count,
                    "name": name,
                    "last_date": last_date,
                    "avg_amount_5d": avg_amount_5d,
                }
            except Exception as e:
                logger.debug(
                    f"[DynamicTargetSelector] 读取 {fpath.name} 失败: {e}"
                )
                return None

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(_read_one, f): f for f in pq_files}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    records.append(result)

        return records

    def _filter_basic(
        self, records: List[Dict[str, object]]
    ) -> List[Dict[str, object]]:
        """
        Layer-1 基础过滤：
            1. 剔除股票名含 "ST" 或 "*ST"（大小写不敏感）
            2. 剔除上市不足 min_ipo_days 个交易日的新股
        """
        filtered = []
        for r in records:
            name = str(r.get("name", "")).upper()
            # ST / *ST 过滤
            if "ST" in name:
                logger.debug(
                    f"[DynamicTargetSelector] 剔除ST股: {r['code']} ({r['name']})"
                )
                continue
            # 新股过滤（行数代理上市天数，日线每行=1交易日）
            if r.get("row_count", 0) < self.min_ipo_days:
                logger.debug(
                    f"[DynamicTargetSelector] 剔除新股: {r['code']} "
                    f"({r['row_count']}行 < {self.min_ipo_days}天)"
                )
                continue
            filtered.append(r)
        return filtered

    def _rank_liquidity(
        self, records: List[Dict[str, object]]
    ) -> List[str]:
        """
        Layer-2 流动性排名：按 avg_amount_5d 降序，取前 top_n 只。
        """
        sorted_records = sorted(
            records,
            key=lambda r: r.get("avg_amount_5d", 0),
            reverse=True,
        )
        return [r["code"] for r in sorted_records[: self.top_n]]

    def _merge_whitelist(self, top_codes: List[str]) -> List[str]:
        """
        Layer-3 白名单叠加：将白名单代码强制加入，去重后返回。
        白名单代码排在列表末尾（相对于 top_codes 保持原顺序）。
        """
        seen = set(top_codes)
        result = list(top_codes)
        for wl_code in self.whitelist:
            if wl_code not in seen:
                result.append(wl_code)
                seen.add(wl_code)
        return result


# ══════════════════════════════════════════════════════════════════════════════
# SmartMinuteSyncer —— 一键动态选股 + 分钟数据同步
# ══════════════════════════════════════════════════════════════════════════════

class SmartMinuteSyncer:
    """
    智能分钟数据同步器：动态选股 → 定向 TDX 拉取 → 可选自动清理。

    与原 MinuteDataCollector.batch_collect() 的区别：
        - 调用 DynamicTargetSelector 自动确定同步目标，无需手工输入代码
        - 封装完整的同步流程（选股 → 采集 → 可选清理）
        - build_minute_npy 成功后可自动删除 data/minute/ 原始 Parquet（释放空间）

    Parameters
    ----------
    parquet_dir  : str | Path
        日线 Parquet 目录（用于动态选股）
    data_root    : str | Path
        分钟数据根目录，Parquet 存储在 {data_root}/minute/
    whitelist    : list[str]
        强制纳入的白名单代码
    top_n        : int
        流动性排名取前 N，默认 300
    n_bars       : int
        每只股票拉取的 5 分钟 K 线数量，默认 2400（≈ 50 交易日）
    max_workers  : int
        TDX 并发线程数，默认 4
    cleanup_after_npy : bool
        调用 cleanup_source_parquet() 时是否删除 data/minute/ 原始文件（默认 True）

    Example
    -------
    >>> syncer = SmartMinuteSyncer(
    ...     parquet_dir="data/daily_parquet",
    ...     data_root="data",
    ...     whitelist=["600519", "000858"],
    ...     top_n=300,
    ... )
    >>> result = syncer.run_sync()          # 选股 + 采集
    >>> syncer.cleanup_source_parquet()     # 清理原始 Parquet（可选）
    """

    def __init__(
        self,
        parquet_dir: str,
        data_root: str = "data",
        whitelist: Optional[List[str]] = None,
        top_n: int = 300,
        n_bars: int = 2400,
        max_workers: int = 4,
        tdx_host: Optional[str] = None,
        cleanup_after_npy: bool = True,
    ) -> None:
        self.parquet_dir      = Path(parquet_dir)
        self.data_root        = Path(data_root)
        self.whitelist        = whitelist or []
        self.top_n            = top_n
        self.n_bars           = n_bars
        self.max_workers      = max_workers
        self.tdx_host         = tdx_host
        self.cleanup_after_npy = cleanup_after_npy

        self._last_target_list: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def run_sync(self) -> Dict[str, int]:
        """
        执行完整的智能同步流程：
            1. 动态选股（DynamicTargetSelector）
            2. 定向 TDX 拉取分钟数据（MinuteDataCollector.batch_collect）

        Returns
        -------
        Dict[code, n_bars_saved]  每只股票实际保存的 K 线数
        """
        # Step-1: 动态选股
        logger.info("[SmartMinuteSyncer] 开始动态选股...")
        try:
            selector = DynamicTargetSelector(
                parquet_dir=str(self.parquet_dir),
                whitelist=self.whitelist,
                top_n=self.top_n,
            )
            target_codes = selector.get_target_list()
        except Exception as e:
            logger.error(f"[SmartMinuteSyncer] 动态选股失败: {e}", exc_info=True)
            raise

        self._last_target_list = target_codes
        logger.info(
            f"[SmartMinuteSyncer] 最终目标: {len(target_codes)} 只股票"
        )

        # Step-2: 定向采集
        logger.info(
            f"[SmartMinuteSyncer] 开始采集 {len(target_codes)} 只股票 "
            f"× {self.n_bars} 根 5min K线，并发={self.max_workers}..."
        )
        collector = MinuteDataCollector(
            host=self.tdx_host,
            period=PERIOD_5MIN,
            data_root=str(self.data_root),
            max_workers=self.max_workers,
        )
        try:
            results = collector.batch_collect(
                codes=target_codes,
                n_bars=self.n_bars,
                skip_existing=False,  # 智能同步总是覆盖最新月份
            )
        finally:
            collector._close_api()

        ok_count = sum(1 for v in results.values() if v >= 0)
        fail_count = len(target_codes) - ok_count
        logger.info(
            f"[SmartMinuteSyncer] 采集完成: 成功={ok_count}, 失败={fail_count}"
        )

        return results

    def get_last_target_list(self) -> Optional[List[str]]:
        """返回上次 run_sync() 生成的目标列表（供调试）"""
        return self._last_target_list

    def cleanup_source_parquet(self) -> int:
        """
        删除 {data_root}/minute/ 下所有原始 Parquet 文件（保留目录结构）。

        设计原则：
            - 仅删除 .parquet 文件，不删除子目录本身（防止 npy 目录误删）
            - 必须在 build_minute_npy 成功后才调用
            - 删除前打印文件数量和预计释放空间

        Returns
        -------
        int : 成功删除的文件数
        """
        minute_dir = self.data_root / "minute"
        if not minute_dir.exists():
            logger.info("[SmartMinuteSyncer] data/minute/ 目录不存在，无需清理")
            return 0

        pq_files = list(minute_dir.rglob("*.parquet"))
        if not pq_files:
            logger.info("[SmartMinuteSyncer] data/minute/ 中无 Parquet 文件")
            return 0

        total_bytes = sum(f.stat().st_size for f in pq_files if f.exists())
        total_mb = total_bytes / 1024 / 1024
        logger.info(
            f"[SmartMinuteSyncer] 准备清理 {len(pq_files)} 个 Parquet 文件，"
            f"预计释放 {total_mb:.1f} MB..."
        )

        deleted = 0
        for fpath in pq_files:
            try:
                fpath.unlink()
                deleted += 1
            except Exception as e:
                logger.warning(
                    f"[SmartMinuteSyncer] 删除失败 {fpath}: {e}"
                )

        logger.info(
            f"[SmartMinuteSyncer] 清理完成：删除 {deleted}/{len(pq_files)} 个文件，"
            f"释放 {total_mb:.1f} MB"
        )
        return deleted

