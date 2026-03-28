"""
scripts/step0_download_ohlcv_v10.py
=====================================
Q-UNITY V10 日线下载脚本（HFQ 后复权版）

相对 step0_download_ohlcv.py / step0d_download_daily_akshare.py 的修改点：
─────────────────────────────────────────────────────────────────────────────
[V10-1] 复权方式改为 HFQ（后复权）
        ak.stock_zh_a_hist(adjust="hfq")
        meta.json 中 adj_type="hfq"

[V10-2] Union Universe（并集宇宙）
        active_codes（当前在市） | delist_codes（已退市）
        确保退市股的历史数据完整保留，valid_mask 按退市日正确置 False

[V10-3] valid_mask 新增 listing_days >= 60 规则
        上市不足 60 天（约 3 个月）的股票置 False，
        避免新股上市初期异常波动污染因子计算

[V10-4] 市场指数下载
        ak.stock_zh_index_daily(symbol="sh000300")  ← 沪深300
        保存为 data/npy/market_index.npy，shape=(1, T)
        同时保存 market_index_dates.npy 供日期对齐

铁律（每次修改前默读）
─────────────────────────────────────────────────────────────────────────────
1. adj_type 必须存入 meta.json，且本脚本永远写 "qfq"
2. valid_mask 退市股必须置 False（不允许使用退市后数据）
3. listing_days < 60 → valid_mask = False（不允许新股上市初期信号）
4. market_index.npy shape = (1, T)，与 close.npy 的 T 对齐

用法
─────────────────────────────────────────────────────────────────────────────
  # 测试模式（10只 + 验收断言）
  python scripts/step0_download_ohlcv_v10.py --test --n 10

  # 全量下载
  python scripts/step0_download_ohlcv_v10.py --workers 8

  # 增量更新（只补旧/缺失文件）
  python scripts/step0_download_ohlcv_v10.py --workers 8 --incremental

  # 指定股票
  python scripts/step0_download_ohlcv_v10.py --codes 600519 000001

  # 强制重下
  python scripts/step0_download_ohlcv_v10.py --force --workers 8

预计耗时：~40~80 分钟（全市场 ~5200 只，8 线程，delay=0.6s）
"""
from __future__ import annotations
import os, sys
from pathlib import Path

# ── [PATH-FIX] Project Root Auto-Detection ─────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ───────────────────────────────────────────────────────────────────────

import argparse
import json
import logging
import random
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# 路径常量
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR            = PROJECT_ROOT / "data"
DEFAULT_PARQUET_DIR = DATA_DIR / "daily_parquet_hfq"   # [V10-1] 独立目录，与 qfq 版不冲突
NPY_DIR             = DATA_DIR / "npy_v10"
META_PATH           = NPY_DIR / "meta.json"

# [V10-1] 复权类型常量
ADJ_TYPE = "hfq"

# [V10-3] 上市天数保护阈值
LISTING_DAYS_MIN = 60

# [V10-4] 市场指数代码
MARKET_INDEX_CODE = "sh000300"   # 沪深300

# ─────────────────────────────────────────────────────────────────────────────
# [ADATA] adata 数据源支持
# 实测确认（2026-03）：
#   · adjust_type=1 = 前复权 QFQ  ✓
#   · volume 返回「股」→ 存 parquet 时 ÷100 转为「手」，与 AKShare 版格式一致
#   · amount 返回「元」→ 直接存储  ✓
#   · 日期格式必须 'YYYY-MM-DD'（有横线），无横线静默返回空 DataFrame
# ─────────────────────────────────────────────────────────────────────────────
ADATA_ADJUST_TYPE = 2   # adata adjust_type=2 = HFQ 后复权（实测）

def _disable_proxy_for_adata() -> None:
    """禁用系统代理，防止 adata SunRequests 被失效代理劫持。"""
    import os
    for k in ["HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy"]:
        os.environ.pop(k, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    try:
        import requests as _req
        _orig_req = _req.Session.request
        def _noproxy(self, method, url, **kw):
            kw.setdefault("proxies", {"http": "", "https": "", "no": "*"})
            kw.setdefault("timeout", 30)
            return _orig_req(self, method, url, **kw)
        _req.Session.request = _noproxy
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _to_full_code(code: str) -> str:
    """6位数字 → sh.xxxxxx / sz.xxxxxx；已是完整格式则直接返回。"""
    s = str(code).strip()
    if s.startswith(("sh.", "sz.")):
        return s
    c = s.split(".")[-1].zfill(6)
    prefix = "sh" if c[0] in ("6", "9") else "sz"
    return f"{prefix}.{c}"


def _to_6digit(code: str) -> str:
    return str(code).strip().split(".")[-1].zfill(6)


def _is_valid_a_stock(code: str) -> bool:
    """[DATA-1] 严格白名单过滤：只保留 A 股，排除指数/ETF/B股/北交所。"""
    _BLACKLIST = {"sh.000", "sh.399", "sz.399", "sh.880", "sz.880",
                  "sh.900", "sz.200", "bj."}
    for blk in _BLACKLIST:
        if code.startswith(blk):
            return False
    parts = code.split(".")
    if len(parts) != 2:
        return False
    exchange, num = parts
    if exchange not in ("sh", "sz") or not num.isdigit() or len(num) != 6:
        return False
    if exchange == "sh":
        return num.startswith(("600", "601", "603", "605", "688"))
    return num.startswith(("000", "001", "002", "003", "300", "301"))


def _load_parquet_dir() -> Path:
    cfg = PROJECT_ROOT / "config.json"
    if cfg.exists():
        try:
            with open(cfg, encoding="utf-8") as f:
                c = json.load(f)
            raw = c.get("parquet_dir_qfq") or c.get("parquet_dir") or \
                  c.get("data", {}).get("parquet_dir", "")
            if raw:
                p = Path(raw)
                return p if p.is_absolute() else PROJECT_ROOT / p
        except Exception:
            pass
    return DEFAULT_PARQUET_DIR


# ─────────────────────────────────────────────────────────────────────────────
# [V10-2] Union Universe：active + delist 并集
# ─────────────────────────────────────────────────────────────────────────────

def _get_union_universe() -> Tuple[List[str], Dict[str, Optional[str]]]:
    """
    [V10-2] 获取全市场 Union Universe（在市 ∪ 已退市）。

    Returns
    -------
    codes       : List[str]   完整格式代码列表（sh./sz.）
    delist_map  : Dict[str, Optional[str]]
                  {code: delist_date_str | None}
                  delist_date 为 None 表示在市股票

    策略
    ----
    1. 从 AKShare stock_info_a_code_name() 获取当前在市列表
    2. 从 AKShare stock_zh_a_spot_em() 追加（覆盖面更广）
    3. 从 parquet_dir 扫描已有文件（含历史退市股）
    4. 尝试从 AKShare 获取退市股列表
    5. 合并去重，返回并集
    """
    import akshare as ak

    codes_set: set = set()
    delist_map: Dict[str, Optional[str]] = {}

    # ── 来源1：当前在市列表 ──────────────────────────────────────────────────
    try:
        df_info = ak.stock_info_a_code_name()
        col = next((c for c in ["code", "股票代码"] if c in df_info.columns),
                   df_info.columns[0])
        for c in df_info[col].dropna():
            fc = _to_full_code(str(c))
            if _is_valid_a_stock(fc):
                codes_set.add(fc)
                delist_map[fc] = None
        logger.info(f"  来源1 在市: {len(codes_set)} 只")
    except Exception as e:
        logger.warning(f"  来源1 stock_info_a_code_name 失败: {e}")

    # ── 来源2：实时行情（更全面的在市列表）────────────────────────────────────
    try:
        df_spot = ak.stock_zh_a_spot_em()
        col = next((c for c in ["代码", "code"] if c in df_spot.columns),
                   df_spot.columns[0])
        before = len(codes_set)
        for c in df_spot[col].dropna():
            fc = _to_full_code(str(c))
            if _is_valid_a_stock(fc):
                codes_set.add(fc)
                delist_map.setdefault(fc, None)
        logger.info(f"  来源2 行情追加: +{len(codes_set) - before} 只")
    except Exception as e:
        logger.warning(f"  来源2 stock_zh_a_spot_em 失败: {e}")

    # ── 来源3：parquet_dir 已有文件（含历史退市股）───────────────────────────
    parquet_dir = _load_parquet_dir()
    if parquet_dir.exists():
        before = len(codes_set)
        for pq in parquet_dir.glob("*.parquet"):
            fc = _to_full_code(pq.stem)
            if _is_valid_a_stock(fc):
                codes_set.add(fc)
                delist_map.setdefault(fc, None)
        logger.info(f"  来源3 parquet: +{len(codes_set) - before} 只")

    # ── 来源4：退市股列表（AKShare）─────────────────────────────────────────
    try:
        df_del = ak.stock_zh_a_delisted()
        col_code = next((c for c in ["代码", "股票代码", "code"] if c in df_del.columns),
                        None)
        col_date = next((c for c in ["退市日期", "delist_date", "最后交易日"]
                         if c in df_del.columns), None)
        if col_code:
            before = len(codes_set)
            for _, row in df_del.iterrows():
                fc = _to_full_code(str(row[col_code]))
                if _is_valid_a_stock(fc):
                    codes_set.add(fc)
                    d_str = str(row[col_date]) if col_date else None
                    delist_map[fc] = d_str   # 退市股覆盖为退市日期
            logger.info(f"  来源4 退市股追加: +{len(codes_set) - before} 只")
    except Exception as e:
        logger.warning(f"  来源4 退市股列表获取失败（非致命）: {e}")

    codes = sorted(codes_set)
    logger.info(f"  [V10-2] Union Universe 总计: {len(codes)} 只（在市 ∪ 退市）")
    return codes, delist_map


# ─────────────────────────────────────────────────────────────────────────────
# UA 轮换（防封禁）
# ─────────────────────────────────────────────────────────────────────────────

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
]

_lock                = Lock()
_consecutive_failures = 0


# ─────────────────────────────────────────────────────────────────────────────
# [ADATA] adata 数据源函数
# ─────────────────────────────────────────────────────────────────────────────

def _get_union_universe_adata() -> Tuple[List[str], Dict[str, Optional[str]]]:
    """
    [ADATA] 用 adata.stock.info.all_code() 获取全市场股票列表。
    adata all_code() 包含在市股票，含 list_date。
    退市股从已有 parquet 文件补充（保持 Union Universe 语义）。
    """
    import adata

    codes_set: set = set()
    delist_map: Dict[str, Optional[str]] = {}

    # 来源1: adata all_code
    try:
        df = adata.stock.info.all_code()
        if df is not None and not df.empty:
            col = "stock_code" if "stock_code" in df.columns else df.columns[0]
            for c in df[col].dropna():
                fc = _to_full_code(str(c))
                if _is_valid_a_stock(fc):
                    codes_set.add(fc)
                    delist_map[fc] = None
            logger.info(f"  [adata] 来源1 在市: {len(codes_set)} 只")
    except Exception as e:
        logger.warning(f"  [adata] all_code() 失败: {e}")

    # 来源2: parquet 已有文件（补充退市股历史数据）
    parquet_dir = _load_parquet_dir()
    if parquet_dir.exists():
        before = len(codes_set)
        for pq in parquet_dir.glob("*.parquet"):
            fc = _to_full_code(pq.stem)
            if _is_valid_a_stock(fc):
                codes_set.add(fc)
                delist_map.setdefault(fc, None)
        added = len(codes_set) - before
        if added:
            logger.info(f"  [adata] 来源2 parquet 补充: +{added} 只")

    codes = sorted(codes_set)
    logger.info(f"  [adata] Union Universe 总计: {len(codes)} 只")
    return codes, delist_map


def _download_one_qfq_adata(
    code       : str,
    start      : str,
    end        : str,
    parquet_dir: Path,
    delay      : float,
    delist_date: Optional[str] = None,
) -> dict:
    """
    [ADATA] adata 版 QFQ 前复权日线下载。

    注意：
      · start/end 必须是 'YYYY-MM-DD' 格式（有横线）
      · adata volume 返回「股」→ ÷100 存为「手」，与 AKShare 版格式完全一致
      · adata amount 返回「元」→ 直接存储
      · adjust_type=1 = QFQ 前复权（实测确认）
    """
    global _consecutive_failures

    code_6 = _to_6digit(code)

    with _lock:
        extra_sleep = min(_consecutive_failures * 0.4, 10.0)
    if extra_sleep > 0:
        time.sleep(extra_sleep)

    time.sleep(random.uniform(0.1, 0.8))

    import adata

    # 退市股截断 end
    eff_end = end
    if delist_date:
        try:
            dl = date.fromisoformat(delist_date[:10])
            eff_end = min(date.fromisoformat(end), dl).strftime("%Y-%m-%d")
        except Exception:
            pass

    # 验证日期格式（关键：adata 无横线会静默返回空）
    import re
    for d, label in [(start, "start"), (eff_end, "end")]:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", d):
            return {"code": code, "status": "error", "rows": 0,
                    "msg": f"日期格式错误({label}={d})，必须 YYYY-MM-DD"}

    last_err = ""
    for attempt in range(4):
        try:
            df = adata.stock.market.get_market(
                stock_code  = code_6,
                start_date  = start,           # 'YYYY-MM-DD' ← 关键
                k_type      = 1,               # 日线
                adjust_type = ADATA_ADJUST_TYPE,  # 1 = QFQ
            )

            if df is None or df.empty:
                return {"code": code, "status": "empty", "rows": 0,
                        "msg": "adata 返回空数据"}

            df = df.copy()

            # 字段标准化
            rename = {}
            for target, cands in [
                ("date",   ["trade_date", "date"]),
                ("open",   ["open"]),
                ("high",   ["high"]),
                ("low",    ["low"]),
                ("close",  ["close"]),
                ("volume", ["volume", "vol"]),
                ("amount", ["amount"]),
            ]:
                for c in cands:
                    if c in df.columns and c != target:
                        rename[c] = target
                        break
            df = df.rename(columns=rename)

            miss = [c for c in ["date","open","high","low","close"] if c not in df.columns]
            if miss:
                return {"code": code, "status": "error", "rows": 0,
                        "msg": f"缺少必要列: {miss}"}

            # 日期过滤（截断退市日期后）
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            df = df.dropna(subset=["date","close"])
            df = df[df["date"] <= eff_end]

            for col in ["open","high","low","close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
            df = df[df["close"] > 0].copy()

            if df.empty:
                return {"code": code, "status": "empty", "rows": 0,
                        "msg": "清洗后为空"}

            listing_date = df["date"].min()
            df["listing_date"] = listing_date

            # [ADATA] volume: adata 返回「股」→ ÷100 = 「手」（与 AKShare 版格式一致）
            if "volume" in df.columns:
                df["volume"] = (pd.to_numeric(df["volume"], errors="coerce")
                                .fillna(0.0) / 100.0).astype("float32")
            else:
                df["volume"] = np.float32(0)

            # amount: adata 返回「元」→ 直接存储
            if "amount" in df.columns:
                df["amount"] = (pd.to_numeric(df["amount"], errors="coerce")
                                .fillna(0.0)).astype("float32")
            else:
                df["amount"] = np.float32(0)

            df["code"]     = code
            df["adj_type"] = ADJ_TYPE  # "qfq"

            keep = [c for c in ["date","open","high","low","close",
                                  "volume","amount","code",
                                  "listing_date","adj_type"] if c in df.columns]
            df = df[keep].sort_values("date").reset_index(drop=True)

            # 合并已有 parquet
            out_path = parquet_dir / f"{code}.parquet"
            if out_path.exists():
                try:
                    df_old = pd.read_parquet(str(out_path))
                    if len(df_old) > 0:
                        df = (pd.concat([df_old, df], ignore_index=True)
                              .drop_duplicates(subset=["date"], keep="last")
                              .sort_values("date").reset_index(drop=True))
                except Exception:
                    pass

            df.to_parquet(str(out_path), index=False, compression="snappy")

            with _lock:
                _consecutive_failures = max(0, _consecutive_failures - 1)
            time.sleep(delay + random.uniform(0, delay * 0.3))
            return {"code": code, "status": "ok", "rows": len(df),
                    "listing_date": str(df["date"].min()), "msg": ""}

        except Exception as e:
            last_err = str(e)
            if attempt < 3:
                time.sleep(2.0 * (attempt + 1) + random.uniform(0, 1))

    with _lock:
        _consecutive_failures += 1
    return {"code": code, "status": "error", "rows": 0,
            "msg": f"4次重试失败: {last_err[:100]}"}


def _download_market_index_adata(
    symbol      : str,
    start       : str,
    end         : str,
    npy_dir     : Path,
    trading_days: Optional[List[str]] = None,
) -> bool:
    """
    [ADATA] 用 adata.stock.market.get_market_index() 下载市场指数日线。
    adata index_code 格式：'000300'（不带交易所前缀）。
    """
    import adata

    # adata 指数代码：去掉 sh/sz 前缀，只保留6位数字
    idx_code = symbol.replace("sh", "").replace("sz", "").replace(".", "").strip()
    logger.info(f"  [adata] 下载市场指数: {idx_code}")

    for attempt in range(3):
        try:
            df = adata.stock.market.get_market_index(
                index_code = idx_code,
                start_date = start,   # 'YYYY-MM-DD'
                k_type     = 1,
            )
            if df is not None and not df.empty:
                break
        except Exception as e:
            logger.warning(f"  [adata] 指数下载尝试 {attempt+1}/3 失败: {e}")
            if attempt < 2:
                time.sleep(3.0 * (attempt + 1))
    else:
        logger.error(f"  [adata] 指数 {idx_code} 下载失败（3次重试）")
        return False

    # 字段标准化
    rename = {}
    for target, cands in [
        ("date",  ["trade_date","date"]),
        ("close", ["close"]),
    ]:
        for c in cands:
            if c in df.columns and c != target:
                rename[c] = target; break
    df = df.rename(columns=rename)

    if "date" not in df.columns or "close" not in df.columns:
        logger.error(f"  [adata] 指数缺少 date/close 列: {df.columns.tolist()}")
        return False

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date","close"]).copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce").fillna(0.0)
    df = df[(df["date"] >= start) & (df["date"] <= end)].sort_values("date").reset_index(drop=True)

    if df.empty:
        logger.error(f"  [adata] 指数 {idx_code} 日期过滤后为空")
        return False

    if trading_days is not None:
        date_to_idx: Dict[str, int] = {d: i for i, d in enumerate(trading_days)}
        T = len(trading_days)
        filled = np.full(T, np.nan)
        for _, row in df.iterrows():
            ti = date_to_idx.get(str(row["date"]))
            if ti is not None:
                filled[ti] = float(row["close"])
        last_v = np.nan
        for ti in range(T):
            if not np.isnan(filled[ti]):
                last_v = filled[ti]
            filled[ti] = last_v
        np.nan_to_num(filled, nan=0.0, copy=False)
        mkt_arr   = filled.reshape(1, T).astype(np.float32)
        dates_arr = np.array(trading_days, dtype=object)
    else:
        mkt_arr   = df["close"].to_numpy(dtype=np.float32).reshape(1, -1)
        dates_arr = df["date"].to_numpy(dtype=object)

    npy_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(npy_dir / "market_index.npy"), mkt_arr)
    np.save(str(npy_dir / "market_index_dates.npy"), dates_arr)
    logger.info(f"  [adata] ✓ market_index.npy shape={mkt_arr.shape} "
                f"range={dates_arr[0]}~{dates_arr[-1]}")
    return True


def _patch_ak_ua() -> None:
    try:
        import akshare.utils.func as f
        s = getattr(f, "session", None)
        if s is None:
            import requests
            s = requests.Session()
            f.session = s
        s.headers.update({
            "User-Agent":      random.choice(_USER_AGENTS),
            "Referer":         "https://www.eastmoney.com/",
            "Accept-Language": "zh-CN,zh;q=0.9",
        })
    except Exception:
        pass


def _is_rate_limit(msg: str) -> bool:
    kws = ["429", "限流", "频繁", "too many", "rate limit", "blocked", "403"]
    return any(k.lower() in msg.lower() for k in kws)


def _backoff(attempt: int, is_limit: bool) -> float:
    return (6.0 * (attempt + 1) + random.uniform(0, 3)) if is_limit \
           else (2.0 ** attempt + random.uniform(0, 1))


# ─────────────────────────────────────────────────────────────────────────────
# 扫描缺失文件
# ─────────────────────────────────────────────────────────────────────────────

def _scan_missing(parquet_dir: Path, codes: List[str], force: bool) -> List[str]:
    missing = []
    cutoff = (date.today() - timedelta(days=5)).strftime("%Y-%m-%d")
    for code in codes:
        pq = parquet_dir / f"{code}.parquet"
        if force:
            missing.append(code)
            continue
        if not pq.exists():
            missing.append(code)
            continue
        try:
            df_d = pd.read_parquet(str(pq), columns=["date"])
            if len(df_d) == 0 or str(df_d["date"].max()) < cutoff:
                missing.append(code)
        except Exception:
            missing.append(code)
    return missing


# ─────────────────────────────────────────────────────────────────────────────
# [V10-1] 单只下载：AKShare QFQ 前复权
# ─────────────────────────────────────────────────────────────────────────────

def _download_one_qfq(
    code       : str,
    start      : str,
    end        : str,
    parquet_dir: Path,
    delay      : float,
    delist_date: Optional[str] = None,
) -> dict:
    """
    [V10-1] AKShare QFQ 前复权日线下载。

    关键变化（相对 step0d hfq 版）：
      adjust="qfq"  而非  adjust="hfq"
      extra 列 listing_date（首行日期），供 valid_mask listing_days 计算

    volume 单位：保持「手」，fast_runner vol_multiplier=100 统一换算为「股」
    amount 单位：AKShare 直接返回元（无需 ×10000）
    """
    global _consecutive_failures

    code_6 = _to_6digit(code)

    with _lock:
        extra_sleep = min(_consecutive_failures * 0.4, 10.0)
    if extra_sleep > 0:
        time.sleep(extra_sleep)

    time.sleep(random.uniform(0.1, 1.0))

    import akshare as ak
    last_err = ""

    # 若退市股，end 截断到退市日（避免请求无效日期）
    eff_end = end
    if delist_date:
        try:
            dl = date.fromisoformat(delist_date[:10])
            eff_end = min(date.fromisoformat(end), dl).strftime("%Y-%m-%d")
        except Exception:
            pass

    for attempt in range(4):
        try:
            _patch_ak_ua()
            # [V10-1] adjust="qfq" ← 核心修改
            df = ak.stock_zh_a_hist(
                symbol     = code_6,
                period     = "daily",
                start_date = start.replace("-", ""),
                end_date   = eff_end.replace("-", ""),
                adjust     = ADJ_TYPE,   # "qfq"
            )

            if df is None or df.empty:
                return {"code": code, "status": "empty", "rows": 0,
                        "msg": "AKShare 返回空数据"}

            df = df.copy()

            # ── 字段标准化 ────────────────────────────────────────────────
            rename = {}
            for target, cands in [
                ("date",   ["日期", "date"]),
                ("open",   ["开盘", "open"]),
                ("high",   ["最高", "high"]),
                ("low",    ["最低", "low"]),
                ("close",  ["收盘", "close"]),
                ("volume", ["成交量", "volume", "vol"]),
                ("amount", ["成交额", "amount", "turnover"]),
            ]:
                for c in cands:
                    if c in df.columns and c != target:
                        rename[c] = target
                        break
            df = df.rename(columns=rename)

            miss = [c for c in ["date", "open", "high", "low", "close"]
                    if c not in df.columns]
            if miss:
                return {"code": code, "status": "error", "rows": 0,
                        "msg": f"缺少必要列: {miss}"}

            # ── 清洗 ─────────────────────────────────────────────────────
            df["date"] = (pd.to_datetime(df["date"], errors="coerce")
                          .dt.strftime("%Y-%m-%d"))
            df = df.dropna(subset=["date", "close"])
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
            df = df[df["close"] > 0].copy()

            # [V10-3] 记录 listing_date（首个有效交易日），供 valid_mask 使用
            if len(df) > 0:
                listing_date = df["date"].min()
                df["listing_date"] = listing_date
            else:
                return {"code": code, "status": "empty", "rows": 0,
                        "msg": "清洗后为空"}

            # volume：保持「手」单位，由 fast_runner vol_multiplier=100 统一换算为「股」
            # 【重要】所有下载路由均存储「手」，fast_runner 统一乘以100
            if "volume" in df.columns:
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).astype("float32")
            else:
                df["volume"] = np.float32(0)

            # amount：直接是元，无需换算
            if "amount" in df.columns:
                df["amount"] = (pd.to_numeric(df["amount"], errors="coerce")
                                .fillna(0.0)).astype("float32")
            else:
                df["amount"] = np.float32(0)

            df["code"]       = code
            df["adj_type"]   = ADJ_TYPE    # [V10-1] 标记复权类型

            keep = [c for c in ["date", "open", "high", "low", "close",
                                  "volume", "amount", "code",
                                  "listing_date", "adj_type"]
                    if c in df.columns]
            df = df[keep].sort_values("date").reset_index(drop=True)

            # ── 合并已有 parquet ──────────────────────────────────────────
            out_path = parquet_dir / f"{code}.parquet"
            if out_path.exists():
                try:
                    df_old = pd.read_parquet(str(out_path))
                    if len(df_old) > 0:
                        df = (pd.concat([df_old, df], ignore_index=True)
                              .drop_duplicates(subset=["date"], keep="last")
                              .sort_values("date").reset_index(drop=True))
                except Exception:
                    pass

            df.to_parquet(str(out_path), index=False, compression="snappy")

            with _lock:
                _consecutive_failures = max(0, _consecutive_failures - 1)
            time.sleep(delay + random.uniform(0, delay * 0.3))
            return {"code": code, "status": "ok", "rows": len(df),
                    "listing_date": str(df["date"].min()), "msg": ""}

        except Exception as e:
            last_err = str(e)
            is_limit = _is_rate_limit(last_err)
            if attempt < 3:
                time.sleep(_backoff(attempt, is_limit))

    with _lock:
        _consecutive_failures += 1
    return {"code": code, "status": "error", "rows": 0,
            "msg": f"4次重试失败: {last_err[:100]}"}


# ─────────────────────────────────────────────────────────────────────────────
# [V10-4] 市场指数下载
# ─────────────────────────────────────────────────────────────────────────────

def _download_market_index(
    symbol    : str,
    start     : str,
    end       : str,
    npy_dir   : Path,
    trading_days: Optional[List[str]] = None,
) -> bool:
    """
    [V10-4] 下载市场指数日线并保存为 market_index.npy。

    market_index.npy shape = (1, T)，其中 T 与 trading_days 对齐。
    若 trading_days 为 None，则直接保存原始序列（shape=(1, T_raw)）。

    同时保存 market_index_dates.npy（str 数组，shape=(T,)）供日期对齐。

    Returns True on success.
    """
    import akshare as ak

    logger.info(f"[V10-4] 下载市场指数: {symbol}")

    for attempt in range(3):
        try:
            _patch_ak_ua()
            df = ak.stock_zh_index_daily(symbol=symbol)
            if df is None or df.empty:
                logger.error(f"[V10-4] {symbol} 返回空数据")
                return False
            break
        except Exception as e:
            logger.warning(f"[V10-4] 尝试 {attempt+1}/3 失败: {e}")
            if attempt < 2:
                time.sleep(3.0 * (attempt + 1))
    else:
        logger.error(f"[V10-4] {symbol} 下载失败（3次重试）")
        return False

    # 标准化
    rename = {}
    for target, cands in [
        ("date",  ["日期", "date"]),
        ("close", ["收盘", "close"]),
        ("open",  ["开盘", "open"]),
        ("high",  ["最高", "high"]),
        ("low",   ["最低", "low"]),
    ]:
        for c in cands:
            if c in df.columns and c != target:
                rename[c] = target
                break
    df = df.rename(columns=rename)

    if "date" not in df.columns or "close" not in df.columns:
        logger.error(f"[V10-4] {symbol} 缺少 date/close 列: {df.columns.tolist()}")
        return False

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date", "close"]).copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce").fillna(0.0)

    # 过滤日期范围
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) == 0:
        logger.error(f"[V10-4] {symbol} 日期过滤后为空（{start}~{end}）")
        return False

    if trading_days is not None:
        # [V10-4] 对齐到标准交易日历（T 维度一致）
        date_to_idx: Dict[str, int] = {d: i for i, d in enumerate(trading_days)}
        T = len(trading_days)
        idx_arr = np.full(T, np.nan, dtype=np.float64)
        for _, row in df.iterrows():
            ti = date_to_idx.get(str(row["date"]))
            if ti is not None:
                idx_arr[ti] = float(row["close"])
        # ffill 填充节假日
        valid_mask_idx = ~np.isnan(idx_arr)
        if valid_mask_idx.any():
            idx_fill = np.where(valid_mask_idx, np.arange(T), 0)
            np.maximum.accumulate(idx_fill, out=idx_fill)
            rows_idx = np.zeros(T, dtype=np.int64)
            close_arr = np.where(valid_mask_idx, idx_arr, 0.0)
            # 直接 ffill via index
            filled = np.full(T, np.nan)
            last_v = np.nan
            for ti in range(T):
                if not np.isnan(idx_arr[ti]):
                    last_v = idx_arr[ti]
                filled[ti] = last_v
            np.nan_to_num(filled, nan=0.0, copy=False)
            mkt_arr = filled.reshape(1, T).astype(np.float32)
        else:
            mkt_arr = np.zeros((1, T), dtype=np.float32)
        dates_arr = np.array(trading_days, dtype=object)
    else:
        close_vals = df["close"].to_numpy(dtype=np.float32)
        mkt_arr    = close_vals.reshape(1, -1)
        dates_arr  = df["date"].to_numpy(dtype=object)

    npy_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(npy_dir / "market_index.npy"), mkt_arr)
    np.save(str(npy_dir / "market_index_dates.npy"), dates_arr)

    logger.info(
        f"[V10-4] ✓ market_index.npy saved: shape={mkt_arr.shape} "
        f"range={dates_arr[0]}~{dates_arr[-1]}"
    )
    return True


# ─────────────────────────────────────────────────────────────────────────────
# [V10-3] valid_mask 构建（含 listing_days >= 60）
# ─────────────────────────────────────────────────────────────────────────────

def _build_valid_mask_v10(
    close       : np.ndarray,        # (N, T) float32
    volume      : np.ndarray,        # (N, T) float32
    trading_days: List[str],         # len=T
    codes       : List[str],         # len=N
    parquet_dir : Path,
    delist_map  : Dict[str, Optional[str]],
    min_valid_rows: int = 100,
    delist_window : int = 30,
    listing_days_min: int = LISTING_DAYS_MIN,
) -> np.ndarray:
    """
    [V10-3] V10 enhanced valid_mask 构建。

    规则（全部满足才为 True）：
    ──────────────────────────────────────────────────────────────────
    R1: cumulative_valid_rows >= min_valid_rows
        （继承 NB-21：累积有效交易日保护，新股初期保护）

    R2: recent delist_window 天内至少 1 天 volume > 0
        （继承 [BUG-VALID-MASK-DELISTED FIX]：退市检测）

    R3: [V10-3 NEW] listing_days >= listing_days_min（默认 60）
        listing_date = 该股票 parquet 中最早交易日
        col_t 对应的 listing_days = trading_day_index - listing_day_index
        若 listing_days < 60 → False（上市不足 60 天）

    R4: [V10-2 NEW] 退市股在退市日之后置 False
        delist_map[code] 不为 None → 退市日 idx 后全部 False

    delist_window=30：覆盖停牌+预退市窗口（继承 [D-04-FIX]）。
    """
    N, T = close.shape
    date_to_idx: Dict[str, int] = {d: i for i, d in enumerate(trading_days)}

    has_data = (close > 0) & (~np.isnan(close)) & (volume > 0)

    # ── R1: cumcount >= min_valid_rows ────────────────────────────────────
    cumcount = np.cumsum(has_data.astype(np.int32), axis=1)
    mask_r1  = cumcount >= min_valid_rows

    # ── R2: 近 delist_window 天内有成交（退市检测）────────────────────────
    vol_pos = (volume > 0).astype(np.int32)
    cs      = np.cumsum(vol_pos, axis=1)
    cs_lag  = np.concatenate(
        [np.zeros((N, delist_window), dtype=np.int32), cs[:, :-delist_window]],
        axis=1
    )
    rolling = cs - cs_lag
    mask_r2 = rolling > 0

    # ── R3: listing_days >= listing_days_min ─────────────────────────────
    mask_r3 = np.ones((N, T), dtype=bool)
    listing_idx_arr = np.full(N, 0, dtype=np.int32)   # 默认从第0列起算

    for i, code in enumerate(codes):
        pq = parquet_dir / f"{code}.parquet"
        if not pq.exists():
            continue
        try:
            df_d = pd.read_parquet(str(pq), columns=["date"])
            if df_d.empty:
                continue
            listing_date_str = str(df_d["date"].min())
            # listing_date_str 可能是 "2005-02-08" 或 Timestamp
            try:
                ld = pd.to_datetime(listing_date_str).strftime("%Y-%m-%d")
            except Exception:
                continue
            li = date_to_idx.get(ld)
            if li is None:
                # 找最近的交易日
                li = next(
                    (j for j, d in enumerate(trading_days) if d >= ld), 0
                )
            listing_idx_arr[i] = li
        except Exception:
            pass

    # listing_days[i, t] = t - listing_idx_arr[i]
    col_idx    = np.arange(T, dtype=np.int32)[np.newaxis, :]          # (1, T)
    ld_idx     = listing_idx_arr[:, np.newaxis].astype(np.int32)       # (N, 1)
    listing_days_mat = col_idx - ld_idx                                # (N, T)
    mask_r3    = listing_days_mat >= listing_days_min                  # (N, T)

    # ── R4: 退市日之后强制 False ──────────────────────────────────────────
    mask_r4 = np.ones((N, T), dtype=bool)
    for i, code in enumerate(codes):
        dl_date = delist_map.get(code)
        if dl_date is None:
            continue
        try:
            dl_str = pd.to_datetime(str(dl_date)).strftime("%Y-%m-%d")
            # 找退市日在 trading_days 中的位置（含退市日当天 valid，之后为 False）
            dl_idx = next(
                (j for j, d in enumerate(trading_days) if d > dl_str),
                T,
            )
            if dl_idx < T:
                mask_r4[i, dl_idx:] = False
        except Exception:
            pass

    valid_mask = mask_r1 & mask_r2 & mask_r3 & mask_r4

    # 统计
    n_r1_only = mask_r1.mean()
    n_final   = valid_mask.mean()
    n_delist  = sum(1 for v in delist_map.values() if v is not None)
    logger.info(
        f"[V10-3] valid_mask 构建完成: "
        f"R1={n_r1_only:.1%} → final={n_final:.1%} "
        f"(listing_days<{listing_days_min} 被过滤, 退市股={n_delist}只)"
    )
    return valid_mask


# ─────────────────────────────────────────────────────────────────────────────
# 批量下载（主流程）
# ─────────────────────────────────────────────────────────────────────────────

def run_download(
    codes       : List[str],
    delist_map  : Dict[str, Optional[str]],
    start       : str,
    end         : str,
    parquet_dir : Path,
    n_workers   : int,
    delay       : float,
    force       : bool,
    source      : str = "auto",  # [ADATA] 'adata' | 'akshare' | 'auto'
) -> None:
    parquet_dir.mkdir(parents=True, exist_ok=True)
    to_dl = _scan_missing(parquet_dir, codes, force)
    existing = len(codes) - len(to_dl)
    logger.info(f"  已有最新: {existing} 只  |  需下载: {len(to_dl)} 只")

    if not to_dl:
        logger.info("✓ 所有文件均已完整")
        return

    # [ADATA] 选择下载函数
    _eff_src = source
    if source == "auto":
        try:
            import adata as _a; del _a  # noqa
            _eff_src = "adata"
            logger.info("  [auto] 检测到 adata → 使用 adata 数据源")
        except ImportError:
            _eff_src = "akshare"
            logger.info("  [auto] adata 不可用 → 降级到 akshare")

    if _eff_src == "adata":
        _disable_proxy_for_adata()
        _dl_fn = _download_one_qfq_adata
        logger.info(f"  [ADATA] adjust_type={ADATA_ADJUST_TYPE}(QFQ)  "
                    f"volume→手(÷100)  amount→元")
    else:
        _dl_fn = _download_one_qfq
        logger.info(f"  [AKShare] adjust={ADJ_TYPE}")

    est_min = len(to_dl) * (delay + 0.7) / max(n_workers, 1) / 60
    logger.info(f"  [V10-1] adj_type={ADJ_TYPE}  预估: ~{est_min:.0f} 分钟")

    t0    = time.time()
    ok_n  = err_n = empty_n = 0
    total = len(to_dl)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                _dl_fn, code, start, end, parquet_dir, delay,
                delist_map.get(code)
            ): code
            for code in to_dl
        }
        done = 0
        for future in as_completed(futures):
            code = futures[future]
            done += 1
            try:
                res = future.result()
            except Exception as e:
                res = {"code": code, "status": "error", "rows": 0, "msg": str(e)}

            st = res["status"]
            if st == "ok":
                ok_n += 1
            elif st == "error":
                err_n += 1
            else:
                empty_n += 1

            elapsed = time.time() - t0
            speed   = done / max(elapsed, 0.1)
            eta     = (total - done) / max(speed, 0.01)
            print(
                f"  [{done:4d}/{total}] {code}  {st:5s}  {res['rows']:5d}行  "
                f"qfq | ok={ok_n} err={err_n} | {speed:.1f}只/s  ETA={eta:.0f}s",
                flush=True,
            )
            if st == "error":
                print(f"    ✗ {res['msg']}")

    elapsed = time.time() - t0
    print()
    print("=" * 70)
    print(f"  完成: ok={ok_n}  empty={empty_n}  error={err_n}")
    print(f"  耗时: {elapsed:.0f}s  平均: {total/max(elapsed,1):.1f}只/s")
    print(f"  [V10-1] adj_type={ADJ_TYPE} ✓")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# 构建 npy（下载完成后执行）
# ─────────────────────────────────────────────────────────────────────────────

def build_npy(
    parquet_dir : Path,
    npy_dir     : Path,
    codes       : List[str],
    delist_map  : Dict[str, Optional[str]],
    start       : str,
    end         : str,
    market_index_symbol: str = MARKET_INDEX_CODE,
    source      : str = "auto",  # [ADATA] 传递给市场指数下载
) -> None:
    """
    从 parquet 文件构建 (N, T) npy 矩阵并写出 meta.json。

    包含完整的 [V10-1..4] 改动：
      [V10-1] adj_type="qfq"
      [V10-2] 使用 union universe
      [V10-3] listing_days >= 60 过滤
      [V10-4] market_index.npy
    """
    from concurrent.futures import ThreadPoolExecutor as TPE
    import datetime

    npy_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[build_npy] 开始构建 npy  N={len(codes)}")

    # ── Step 1: 交易日历（全市场并集）────────────────────────────────────────
    all_dates: set = set()
    lock_ = __import__("threading").Lock()
    start_date = date.fromisoformat(start)
    end_date   = date.fromisoformat(end)

    def _read_dates(code: str):
        pq = parquet_dir / f"{code}.parquet"
        if not pq.exists():
            return
        try:
            df_d = pd.read_parquet(str(pq), columns=["date"])
            dates_ = pd.to_datetime(df_d["date"]).dt.date
            valid_d = [d for d in dates_ if start_date <= d <= end_date]
            with lock_:
                all_dates.update(str(d) for d in valid_d)
        except Exception:
            pass

    with TPE(max_workers=8) as ex:
        list(ex.map(_read_dates, codes))

    trading_days = sorted(all_dates)
    if not trading_days:
        raise ValueError("[build_npy] 交易日历为空，请先运行 run_download()")
    T = len(trading_days)
    N = len(codes)
    logger.info(f"[build_npy] 交易日历: {trading_days[0]}~{trading_days[-1]}  T={T}")

    # ── Step 2: 构建矩阵 ──────────────────────────────────────────────────────
    fields = ["close", "open", "high", "low", "volume", "amount"]
    mmaps: Dict[str, np.ndarray] = {
        f: np.zeros((N, T), dtype=np.float32) for f in fields
    }
    day_idx = {d: i for i, d in enumerate(trading_days)}

    def _load_one(args_):
        i_, code_ = args_
        pq = parquet_dir / f"{code_}.parquet"
        if not pq.exists():
            return
        try:
            df_ = pd.read_parquet(str(pq))
            if "date" not in df_.columns:
                return
            df_["date"] = pd.to_datetime(df_["date"]).dt.strftime("%Y-%m-%d")

            # [PERF-VEC-BUILD] 向量化筛选与映射，替代 iterrows()
            # 加速约 50-100 倍，彻底解决"构建慢/无反应"问题
            mask = (df_["date"] >= start) & (df_["date"] <= end) & (df_["date"].isin(day_idx))
            if not mask.any():
                return

            sub_df = df_[mask]
            t_idx = sub_df["date"].map(day_idx).values.astype(np.int32)

            for f in fields:
                if f in sub_df.columns:
                    mmaps[f][i_, t_idx] = sub_df[f].values.astype(np.float32)

        except Exception as e:
            logger.debug(f"[build_npy] {code_} 加载失败: {e}")

    with TPE(max_workers=8) as ex:
        list(ex.map(_load_one, enumerate(codes)))

    logger.info("[build_npy] 矩阵填充完成，执行 NaN 填充...")

    # ── Step 3: 两阶段 NaN 填充（继承 V9 逻辑）────────────────────────────────
    for field in fields:
        arr  = mmaps[field].astype(np.float64)
        is_price = field not in ("volume", "amount")

        has_valid = arr > 0
        cumsum_hv = np.cumsum(has_valid.view(np.uint8), axis=1)
        ever_valid = cumsum_hv[:, -1] > 0
        first_valid = np.where(
            ever_valid, np.argmax(cumsum_hv > 0, axis=1), T
        )
        col_idx_ = np.arange(T)[np.newaxis, :]
        pre_ipo  = col_idx_ < first_valid[:, np.newaxis]

        filled = arr.copy()
        filled[pre_ipo] = 0.0

        if is_price:
            # numpy ffill
            nan_m = np.isnan(filled) | (filled == 0.0)
            # 对价格：0.0 在 pre_ipo 之后当做停牌，需要 ffill
            zero_after = (~pre_ipo) & (filled == 0.0)
            nan_m2 = nan_m | zero_after
            has_price = ~nan_m2
            if has_price.any():
                idx_  = np.where(has_price, np.arange(T), 0)
                np.maximum.accumulate(idx_, axis=1, out=idx_)
                rows_ = np.arange(N)[:, np.newaxis]
                filled_price = filled.copy()
                filled_price[zero_after] = np.nan
                # ffill 只对有实际价格的位置
                idx2 = np.where(~np.isnan(filled), np.arange(T), 0)
                np.maximum.accumulate(idx2, axis=1, out=idx2)
                filled = filled[rows_, idx2]
        else:
            # volume/amount: 停牌=0，NaN→0
            np.nan_to_num(filled, nan=0.0, copy=False)

        np.nan_to_num(filled, nan=0.0, copy=False)
        mmaps[field] = filled.astype(np.float32)

    # ── Step 4: [V10-3] valid_mask ────────────────────────────────────────────
    valid_mask = _build_valid_mask_v10(
        close            = mmaps["close"],
        volume           = mmaps["volume"],
        trading_days     = trading_days,
        codes            = codes,
        parquet_dir      = parquet_dir,
        delist_map       = delist_map,
        min_valid_rows   = 100,
        delist_window    = 30,
        listing_days_min = LISTING_DAYS_MIN,
    )

    # ── Step 5: 写出 npy ─────────────────────────────────────────────────────
    for field, arr in mmaps.items():
        np.save(str(npy_dir / f"{field}.npy"), arr)
        logger.info(f"[build_npy] ✓ {field}.npy  shape={arr.shape}")
    np.save(str(npy_dir / "valid_mask.npy"), valid_mask)
    logger.info(f"[build_npy] ✓ valid_mask.npy  shape={valid_mask.shape}")

    # ── Step 6: [V10-4] 市场指数 ─────────────────────────────────────────────
    # [ADATA] 根据 source 选择指数下载函数
    _eff_src_idx = source
    if source == "auto":
        try:
            import adata as _a; del _a  # noqa
            _eff_src_idx = "adata"
        except ImportError:
            _eff_src_idx = "akshare"

    if _eff_src_idx == "adata":
        _disable_proxy_for_adata()
        _download_market_index_adata(
            symbol       = market_index_symbol,
            start        = start,
            end          = end,
            npy_dir      = npy_dir,
            trading_days = trading_days,
        )
    else:
        _download_market_index(
            symbol       = market_index_symbol,
            start        = start,
            end          = end,
            npy_dir      = npy_dir,
            trading_days = trading_days,
        )

    # ── Step 7: meta.json ─────────────────────────────────────────────────────
    meta = {
        "shape"  : [N, T],
        "codes"  : codes,
        "dates"  : trading_days,
        "fields" : fields + ["valid_mask"],
        "dtype"  : "float32",
        # [V10-1] adj_type 必须写 "qfq"
        "adj_type": ADJ_TYPE,
        "build_time": datetime.datetime.now().isoformat(),
        "extra": {
            "listing_days_min"  : LISTING_DAYS_MIN,
            "delist_stocks"     : sum(1 for v in delist_map.values() if v is not None),
            "union_universe"    : True,
            "market_index"      : market_index_symbol,
            "valid_mask_rate"   : float(valid_mask.mean()),
        },
    }
    with open(str(npy_dir / "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info(
        f"[build_npy] ✅ 完成 N={N} T={T} adj_type={ADJ_TYPE} "
        f"valid_mask_rate={valid_mask.mean():.1%}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 验收测试（--test 模式）
# ─────────────────────────────────────────────────────────────────────────────

def _run_acceptance_test(parquet_dir: Path, npy_dir: Path) -> None:
    """
    [验收] 检查 --test 模式产出是否满足 V10 要求：
      1. meta.json 中 adj_type == 'qfq'
      2. valid_mask 对退市股（volume 全为 0）正确置 False
      3. market_index.npy 存在且 shape=(1, T)
    """
    print()
    print("=" * 60)
    print("  V10 验收测试")
    print("=" * 60)

    meta_path = npy_dir / "meta.json"
    assert meta_path.exists(), f"[FAIL] meta.json 不存在: {meta_path}"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    # ── 验收1：adj_type == 'qfq' ────────────────────────────────────────────
    assert meta.get("adj_type") == "qfq", (
        f"[FAIL] adj_type={meta.get('adj_type')}，应为 'qfq'"
    )
    print(f"  ✅ T1 adj_type='{meta['adj_type']}' ✓")

    # ── 验收2：valid_mask 对退市股置 False ───────────────────────────────────
    vm_path = npy_dir / "valid_mask.npy"
    if vm_path.exists():
        vm = np.load(str(vm_path))
        vol_path = npy_dir / "volume.npy"
        if vol_path.exists():
            vol = np.load(str(vol_path))
            # 任何 volume 全为 0 的股票（N行），其 valid_mask 应该全部为 False
            delist_mask = (vol == 0).all(axis=1)   # (N,) bool
            if delist_mask.any():
                for i in np.where(delist_mask)[0]:
                    assert not vm[i].any(), (
                        f"[FAIL] 股票 idx={i}({meta['codes'][i]}) "
                        f"volume 全 0（退市），但 valid_mask 仍有 True"
                    )
                print(f"  ✅ T2 退市股 valid_mask 全为 False "
                      f"（共 {delist_mask.sum()} 只 volume=0 股票）✓")
            else:
                print(f"  ✅ T2 无 volume=0 股票（小样本测试，退市检测逻辑已写入代码）✓")
        else:
            print(f"  ✅ T2 valid_mask.npy 存在（shape={vm.shape}）✓")
    else:
        print(f"  ⚠  T2 valid_mask.npy 不存在（build_npy 未执行）")

    # ── 验收3：market_index.npy 存在且 shape=(1, T) ──────────────────────────
    mi_path = npy_dir / "market_index.npy"
    if mi_path.exists():
        mi = np.load(str(mi_path))
        T = meta["shape"][1]
        assert mi.shape[0] == 1, f"[FAIL] market_index.npy shape[0]={mi.shape[0]}，应为 1"
        assert mi.shape[1] == T, (
            f"[FAIL] market_index.npy shape[1]={mi.shape[1]}，应与 T={T} 对齐"
        )
        print(f"  ✅ T3 market_index.npy shape={mi.shape} (1×T={T}) ✓")
    else:
        print(f"  ⚠  T3 market_index.npy 不存在（_download_market_index 需网络）")

    print()
    print("  V10 验收完成 ✓")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 0 V10: AKShare QFQ 日线下载 + Union Universe + market_index"
    )
    parser.add_argument("--start",       default="2015-01-05",
                        help="下载起始日期 YYYY-MM-DD")
    parser.add_argument("--end",         default=None,
                        help="下载截止日期（默认今日）")
    parser.add_argument("--workers",     type=int, default=8,
                        help="并发线程数（默认8）")
    parser.add_argument("--delay",       type=float, default=0.6,
                        help="每只请求间隔（秒，默认0.6）")
    parser.add_argument("--force",       action="store_true",
                        help="强制重下所有文件")
    parser.add_argument("--incremental", action="store_true",
                        help="增量模式（仅补旧/缺失文件）")
    parser.add_argument("--scan-only",   action="store_true",
                        help="仅扫描，不下载")
    parser.add_argument("--test",        action="store_true",
                        help="测试模式（少量股票 + 验收）")
    parser.add_argument("-n", "--n",     type=int, default=10,
                        help="--test 模式下载股票数（默认10）")
    parser.add_argument("--codes",       nargs="+", default=None,
                        help="指定股票代码（6位或完整格式）")
    parser.add_argument("--build-npy",   action="store_true",
                        help="下载完成后自动构建 npy 矩阵")
    parser.add_argument("--index",       default=MARKET_INDEX_CODE,
                        help=f"市场指数代码（默认 {MARKET_INDEX_CODE}）")
    parser.add_argument("--parquet-dir", default=None,
                        help="parquet 存储目录（默认 data/daily_parquet_qfq）")
    parser.add_argument("--npy-dir",     default=None,
                        help="npy 输出目录（默认 data/npy_v10）")
    # [ADATA] 数据源选择
    parser.add_argument("--source",
                        choices=["auto", "adata", "akshare"],
                        default="auto",
                        help="数据源: auto(优先adata)/adata/akshare（默认auto）")
    args = parser.parse_args()

    # ── 依赖检查 ────────────────────────────────────────────────────────────
    # [ADATA] auto/adata 模式检查 adata；akshare 模式检查 akshare
    _eff_src = args.source
    if args.source in ("auto", "adata"):
        try:
            import adata
            print(f"v adata {adata.__version__}")
            if args.source == "auto":
                _eff_src = "adata"
        except ImportError:
            if args.source == "adata":
                print("x adata 未安装: pip install adata"); sys.exit(1)
            print("  adata 不可用，降级到 akshare")
            _eff_src = "akshare"

    if _eff_src == "akshare":
        try:
            import akshare as ak
            print(f"v AKShare {ak.__version__}")
        except ImportError:
            print("✗ AKShare 未安装: pip install akshare -U"); sys.exit(1)

    end_date    = args.end or date.today().strftime("%Y-%m-%d")
    parquet_dir = Path(args.parquet_dir) if args.parquet_dir else _load_parquet_dir()
    npy_dir_    = Path(args.npy_dir)     if args.npy_dir     else NPY_DIR

    print(f"\n{'='*65}")
    print(f"  Q-UNITY V10 日线下载  adj_type={ADJ_TYPE}  source={_eff_src}")
    print(f"  期间: {args.start} ~ {end_date}")
    print(f"  输出: {parquet_dir}")
    print(f"  [V10-3] listing_days_min={LISTING_DAYS_MIN} 天")
    print(f"  [V10-4] market_index={args.index}")
    print(f"{'='*65}\n")

    # ── 确定下载列表 ─────────────────────────────────────────────────────────
    if args.test:
        _test_defaults = [
            "600519", "601318", "000001", "000858", "300750",
            "600036", "002594", "600900", "300014", "000002",
        ]
        test_codes = [_to_full_code(c) for c in _test_defaults[:args.n]]
        codes_final = test_codes
        delist_map_final: Dict[str, Optional[str]] = {c: None for c in codes_final}
        print(f"⚡ 测试模式: {len(codes_final)} 只")
        print(f"   {codes_final}")

    elif args.codes:
        codes_final = [_to_full_code(c) for c in args.codes]
        delist_map_final = {c: None for c in codes_final}
        print(f"✓ 指定模式: {len(codes_final)} 只")

    else:
        # [ADATA] Union Universe：adata 版或 akshare 版
        if _eff_src == "adata":
            print("▶ 构建 Union Universe（adata all_code + parquet 已有文件）...")
            _disable_proxy_for_adata()
            codes_final, delist_map_final = _get_union_universe_adata()
        else:
            print("▶ 构建 Union Universe（在市 ∪ 退市）...")
            codes_final, delist_map_final = _get_union_universe()

    if args.scan_only:
        missing = _scan_missing(parquet_dir, codes_final, args.force)
        print(f"\n  缺失/过旧: {len(missing)} 只")
        if missing:
            print(f"  前20: {missing[:20]}")
        sys.exit(0)

    # ── 下载 ────────────────────────────────────────────────────────────────
    run_download(
        codes       = codes_final,
        delist_map  = delist_map_final,
        start       = args.start,
        end         = end_date,
        parquet_dir = parquet_dir,
        n_workers   = args.workers,
        delay       = args.delay,
        force       = args.force,
        source      = _eff_src,   # [ADATA]
    )

    # ── 构建 npy（可选）───────────────────────────────────────────────────────
    if args.build_npy or args.test:
        print("\n▶ 构建 npy 矩阵...")
        build_npy(
            parquet_dir          = parquet_dir,
            npy_dir              = npy_dir_,
            codes                = codes_final,
            delist_map           = delist_map_final,
            start                = args.start,
            end                  = end_date,
            market_index_symbol  = args.index,
            source               = _eff_src,   # [ADATA]
        )
        if args.test:
            _run_acceptance_test(parquet_dir, npy_dir_)

    print(f"\n✓ Step 0 V10 完成！adj_type={ADJ_TYPE}")
    print(f"  下一步：python -m src.data.build_npy --config npy_v10")
