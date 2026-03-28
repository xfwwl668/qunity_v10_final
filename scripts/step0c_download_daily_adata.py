"""
scripts/step0c_download_daily_adata.py

BaoStock 封禁后的完整替代方案：adata（主力）+ AKShare（备用）日线下载器

────────────────────────────────────────────────────────────
并发架构说明（为何用 ThreadPoolExecutor 而非 ProcessPoolExecutor）
────────────────────────────────────────────────────────────
BaoStock 需要 ProcessPool 是因为其单例 socket 设计（模块级全局 TCP 连接，
多线程共享 → 帧错位崩溃），必须靠多进程隔离 socket。

adata 和 AKShare 是标准 HTTP 请求库（requests/httpx），每次调用独立连接：
  - 完全线程安全，无状态共享
  - 本质是 I/O 密集（网络等待占 95%+），GIL 不影响并发
  - ThreadPoolExecutor 开销最小：无进程创建/销毁，无 IPC 序列化
  - 16 线程 ≈ 16 路 HTTP 同时挂起等待，CPU 占用<5%

用 ProcessPoolExecutor 反而更慢：每进程 64MB+ 内存 + 1-2s 启动时间，
对纯 I/O 任务无任何加速。

推荐并发设置：
  adata  : --workers 8~16（巨潮节点，限速宽松）
  AKShare: --workers 4~8 （东财节点，适度限速）

────────────────────────────────────────────────────────────
adata 2.9.5 版本说明
────────────────────────────────────────────────────────────
confirmed 可用接口：
  adata.stock.market.get_market(stock_code)
    字段：stock_code, trade_date, open, close, high, low,
          volume(股!), amount(元), change_pct, turnover_ratio, pre_close
    注意：★ 无 pe_ttm/pb 估值字段（估值由 AKShare 补充）
    注意：★ volume 已是「股」，不需要 ×100

adata 无 pe_ttm/pb 的处理：
  - 本脚本补充调用 AKShare stock_a_lg_indicator() 追加估值字段
  - is_st 通过股票名称含"ST"推断（需探测可用的 info 接口）

────────────────────────────────────────────────────────────
用法
────────────────────────────────────────────────────────────
  # 先测试连通性
  python test_adata_download.py

  # 全量下载（推荐 8 线程，约 40~60 分钟）
  python scripts/step0c_download_daily_adata.py --workers 8

  # 增量更新
  python scripts/step0c_download_daily_adata.py --workers 8 --incremental

  # 测试 10 只
  python scripts/step0c_download_daily_adata.py --test

  # 强制重下
  python scripts/step0c_download_daily_adata.py --force --workers 8
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# 路径常量
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
try:
    from scripts.utils_paths import get_npy_dir as _gnpy, get_parquet_dir as _gpq
except ImportError:
    import sys as _sys; _sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from utils_paths import get_npy_dir as _gnpy, get_parquet_dir as _gpq  # type: ignore

NPY_DIR             = _gnpy("v10")
META_PATH           = NPY_DIR / "meta.json"
DEFAULT_PARQUET_DIR = _gpq("qfq")

# ─────────────────────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────────────────────

def _load_parquet_dir() -> Path:
    cfg = PROJECT_ROOT / "config.json"
    if cfg.exists():
        try:
            with open(cfg, encoding="utf-8") as f:
                c = json.load(f)
            raw = c.get("parquet_dir_qfq") or c.get("parquet_dir") or c.get("data", {}).get("parquet_dir", "")  # V10: qfq优先
            if raw:
                p = Path(raw)
                return p if p.is_absolute() else PROJECT_ROOT / p
        except Exception:
            pass
    return DEFAULT_PARQUET_DIR


def _to_full_code(code: str) -> str:
    """6位或 sh.XXXXXX 格式 → sh.XXXXXX 格式（columnar_adapter 要求）。"""
    s = str(code).strip()
    if s.startswith(("sh.", "sz.")):
        return s
    c = s.split(".")[-1].zfill(6)
    return f"{'sh' if c[0] in ('6','9') else 'sz'}.{c}"


def _to_6digit(code: str) -> str:
    """sh.600519 → 600519（AKShare/adata 调用格式）。"""
    return str(code).strip().split(".")[-1].zfill(6)


def _probe_adata_fn(obj, candidates: list[str]):
    """在 adata 对象上探测第一个可调用的接口名，返回 (name, fn) 或 (None, None)。"""
    for name in candidates:
        fn = getattr(obj, name, None)
        if callable(fn):
            return name, fn
    return None, None


def _is_valid_a_stock_code(code_full: str) -> bool:
    """
    [BUG-A-STOCK-COUNT-FIX] 仅保留沪深两市 A 股，与 columnar_adapter._is_valid_a_stock 保持一致。

    白名单：
      sh: 600/601/603/605（沪主板）, 688（科创板）
      sz: 000/001/002/003（深主板+中小板）, 300/301（创业板）

    排除：
      北交所（北证A股）：8xxxxx / 4xxxxx → _to_full_code 错误赋为 sz/sh，这里直接拦截
      B 股：sh.9xxxxx, sz.2xxxxx
      指数：sh.000xxx, sz.399xxx
      ETF/基金：sh.5xxxxx, sz.1xxxxx 等
    """
    parts = code_full.split(".")
    if len(parts) != 2:
        return False
    exch, num = parts[0], parts[1]
    if exch not in ("sh", "sz") or not num.isdigit() or len(num) != 6:
        return False
    if exch == "sh":
        return num.startswith(("600", "601", "603", "605", "688"))
    # sz
    return num.startswith(("000", "001", "002", "003", "300", "301"))


def _load_all_codes() -> list[str]:
    """
    返回全市场 sh.XXXXXX 格式代码列表（仅沪深 A 股）。

    优先级：
    1. stock_list.csv（BaoStock 下载时生成）
    2. meta.json（build_npy 生成）
    3. parquet_dir 目录扫描（已有任意 parquet 时）
    4. AKShare stock_info（最终兜底，不依赖 adata）

    [BUG-A-STOCK-COUNT-FIX]
    AKShare stock_info_a_code_name() 返回 ~5432 只，包含：
      · 北交所（北证A股）8xxxxx / 4xxxxx  ≈ 250 只（_to_full_code 错误赋为 sz 前缀）
      · 可能含 B 股、指数残留
    这些股票会被 build_npy 的 _is_valid_a_stock() 过滤，但下载本身是纯粹的浪费。
    修复：从任何来源获取列表后，统一用 _is_valid_a_stock_code() 白名单过滤，
    使下载目标与 build_npy 收录范围严格一致（目标 ~5189 只，与 BaoStock 齐平）。
    """
    parquet_dir = _load_parquet_dir()

    def _filter(codes: list[str]) -> list[str]:
        valid = [c for c in codes if _is_valid_a_stock_code(c)]
        filtered = len(codes) - len(valid)
        if filtered > 0:
            print(f"  → A股白名单过滤: 原始={len(codes)} → 保留={len(valid)}"
                  f"（剔除 {filtered} 个非A股，含北交所/B股/指数）")
        return valid

    # 1. stock_list.csv
    sl = parquet_dir / "stock_list.csv"
    if sl.exists():
        try:
            df = pd.read_csv(str(sl), dtype=str)
            col = next((c for c in ["code", "股票代码", "stock_code"] if c in df.columns), None)
            if col:
                codes = [_to_full_code(c) for c in df[col].dropna() if str(c).strip()]
                codes = _filter(codes)
                if codes:
                    print(f"  ✓ 股票列表来自 stock_list.csv: {len(codes)} 只")
                    return codes
        except Exception:
            pass

    # 2. meta.json
    if META_PATH.exists():
        with open(META_PATH, encoding="utf-8") as f:
            meta = json.load(f)
        codes = [_to_full_code(str(c)) for c in meta["codes"]]
        codes = _filter(codes)
        print(f"  ✓ 股票列表来自 meta.json: {len(codes)} 只")
        return codes

    # 3. 扫描已有 parquet（只统计 A 股 parquet 文件）
    if parquet_dir.exists():
        existing = sorted(parquet_dir.glob("*.parquet"))
        codes = [_to_full_code(p.stem) for p in existing if p.stem != "stock_list"]
        codes = _filter(codes)
        if len(codes) > 100:
            print(f"  ✓ 股票列表来自 parquet_dir 扫描: {len(codes)} 只")
            return codes

    # 4. AKShare 兜底（无需 adata）
    try:
        import akshare as ak
        df = ak.stock_info_a_code_name()
        col = next((c for c in ["code", "股票代码"] if c in df.columns), df.columns[0])
        codes = [_to_full_code(str(c)) for c in df[col].dropna()]
        codes = _filter(codes)
        print(f"  ✓ 股票列表来自 AKShare stock_info: {len(codes)} 只")
        return codes
    except Exception as e:
        print(f"  ⚠ AKShare 股票列表失败: {e}")

    print("✗ 无法获取股票列表。请先运行 build_npy 或使用 --codes")
    sys.exit(1)


_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
]

_lock = Lock()
_consecutive_failures = 0


def _patch_ak_ua():
    try:
        import akshare.utils.func as f
        s = getattr(f, "session", None)
        if s is None:
            import requests; s = requests.Session(); f.session = s
        s.headers.update({
            "User-Agent":      random.choice(_USER_AGENTS),
            "Referer":         "https://www.eastmoney.com/",
            "Accept-Language": "zh-CN,zh;q=0.9",
        })
    except Exception:
        pass


def _is_rate_limit(msg: str) -> bool:
    kw = ["429", "限流", "频繁", "too many", "rate limit", "blocked", "访问过于"]
    return any(k in msg.lower() for k in kw)


def _backoff(attempt: int, is_limit: bool) -> float:
    return (5.0 * (attempt + 1) + random.uniform(0, 2)) if is_limit \
           else (2.0 ** attempt + random.uniform(0, 1))


def _clean_ohlcv(df: pd.DataFrame, code: str, vol_multiplier: float = 1.0) -> Optional[pd.DataFrame]:
    """
    统一清洗 OHLCV DataFrame，填充 code 列。

    vol_multiplier 参数说明（目标：parquet 统一存储「手」，fast_runner×100 换算为「股」）：
    - AKShare 返回「手」  → vol_multiplier=1.0   → parquet 存「手」✓
    - adata   返回「股」  → vol_multiplier=0.01  → parquet 存「手」✓（÷100）
    - 其他来源 不详       → vol_multiplier=1.0   → 默认存原始值
    """
    df = df.copy()
    if "date" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date", "close"])
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    df = df[df["close"] > 0]
    if "volume" in df.columns:
        df["volume"] = (pd.to_numeric(df["volume"], errors="coerce")
                        .fillna(0.0) * vol_multiplier).astype("float32")
    else:
        df["volume"] = np.float32(0)
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0).astype("float32")
    else:
        df["amount"] = np.float32(0)
    df["code"]     = code
    df["adj_type"] = ADJ_TYPE
    return df.sort_values("date").reset_index(drop=True) if len(df) > 0 else None


# ─────────────────────────────────────────────────────────────────────────────
# 单只下载：主力 adata，备用 AKShare
# ─────────────────────────────────────────────────────────────────────────────

def _try_adata_daily(code_6: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """
    adata.stock.market.get_market(stock_code) 日线。

    adata get_market 字段（版本差异宽容查找）：
      trade_date / date → date
      open, close, high, low → 直接使用
      volume(股!) → adata 2.9.5 已是「股」，vol_multiplier=1.0（不需要×100）
      amount(元) → 直接使用
      pe_ttm, pb(=pb_mrq), ps_ttm → 估值（adata 2.9.5 无此字段，由 AKShare 补充）

    [重要] adata get_market 返回的价格是「后复权」HFQ，与 BaoStock adjustflag='2' 等价。
    不复权价格可通过 get_market(adjust=False) 或独立字段获取。
    """
    for attempt in range(3):
        try:
            import adata
            df = adata.stock.market.get_market(stock_code=code_6)
            if df is None or (hasattr(df, "empty") and df.empty):
                return None

            df = df.copy()

            # 日期
            date_col = next((c for c in ["trade_date", "date", "trading_date"]
                             if c in df.columns), None)
            if not date_col:
                return None
            df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")

            # OHLCV 标准化
            for target, cands in [
                ("open",   ["open"]),
                ("high",   ["high"]),
                ("low",    ["low"]),
                ("close",  ["close", "hfq_close", "adj_close"]),
                ("volume", ["volume", "vol", "turnover_vol"]),
                ("amount", ["amount", "turnover", "turnover_amount", "trade_amount"]),
            ]:
                for c in cands:
                    if c in df.columns and c != target:
                        df[target] = df[c]
                        break

            # 估值字段
            val_map = {
                "pe_ttm": ["pe_ttm", "pe"],
                "pb_mrq": ["pb", "pb_mrq", "price_to_book"],
                "ps_ttm": ["ps_ttm", "ps"],
            }
            for target, cands in val_map.items():
                for c in cands:
                    if c in df.columns:
                        df[target] = pd.to_numeric(df[c], errors="coerce").astype("float32")
                        break
                else:
                    df[target] = np.nan

            # unadj_close（不复权）：用于 step3 市值计算
            #
            # [BUG-PRE-CLOSE-AS-UNADJ-FIX]
            # adata 2.9.5 confirmed 字段包含 pre_close（前收盘，即上个交易日的 HFQ close）。
            # 原候选列表含 "pre_close"，导致 unadj_close 被设为「昨日复权价」——
            # 不仅非不复权价，还是前一天的价，完全错误。
            # 修复：候选列表去掉 pre_close；若真实不复权列不存在则尝试 adjust=False 接口；
            # 两者均无则保持 NaN（build_npy/step3 会降级并打印警告，诚实反映数据缺失）。
            _unadj_found = False
            for c in ["unadj_close", "original_close", "close_unadj", "close_bfq", "close_no_adj"]:
                if c in df.columns:
                    df["unadj_close"] = pd.to_numeric(df[c], errors="coerce").astype("float32")
                    _unadj_found = True
                    break

            if not _unadj_found:
                # [BUG-PRE-CLOSE-AS-UNADJ-FIX] 尝试调用 get_market(adjust=False) 获取真实不复权价
                # adata 2.9.5 文档说明支持 adjust 参数，但接口名可能随版本变化
                try:
                    df_bfq = adata.stock.market.get_market(stock_code=code_6, adjust=False)
                    if df_bfq is not None and not df_bfq.empty:
                        bfq_date_col = next(
                            (c for c in ["trade_date", "date", "trading_date"] if c in df_bfq.columns),
                            None
                        )
                        bfq_close_col = next(
                            (c for c in ["close", "unadj_close", "price"] if c in df_bfq.columns),
                            None
                        )
                        if bfq_date_col and bfq_close_col:
                            df_bfq = df_bfq[[bfq_date_col, bfq_close_col]].copy()
                            df_bfq[bfq_date_col] = pd.to_datetime(
                                df_bfq[bfq_date_col], errors="coerce"
                            ).dt.strftime("%Y-%m-%d")
                            df_bfq = df_bfq.rename(columns={
                                bfq_date_col: "date", bfq_close_col: "unadj_close"
                            })
                            df_bfq["unadj_close"] = pd.to_numeric(
                                df_bfq["unadj_close"], errors="coerce"
                            ).astype("float32")
                            df = df.merge(df_bfq[["date", "unadj_close"]], on="date", how="left")
                            _unadj_found = True
                except Exception:
                    pass  # adjust=False 不支持时静默跳过

            if not _unadj_found:
                # 两种途径均无，保持 NaN：市值计算将降级使用 close.npy 并打印警告
                df["unadj_close"] = np.nan

            # 日期过滤
            df = df[(df["date"] >= start) & (df["date"] <= end)]
            return df if not df.empty else None

        except ImportError:
            return None
        except Exception:
            if attempt < 2:
                time.sleep(_backoff(attempt, False))
    return None


def _try_akshare_daily(code_6: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """
    AKShare stock_zh_a_hist 日线（备用路径）。
    估值字段由 stock_a_lg_indicator 独立补充。
    """
    for attempt in range(3):
        try:
            import akshare as ak
            _patch_ak_ua()
            df = ak.stock_zh_a_hist(
                symbol=code_6, period="daily",
                start_date=start.replace("-", ""),
                end_date=end.replace("-", ""),
                adjust="qfq",  # V10: 前复权 QFQ
            )
            if df is None or df.empty:
                return None
            df = df.copy()
            # 中文列 → 英文
            for target, cands in [
                ("date",   ["日期", "date"]),
                ("open",   ["开盘", "open"]),
                ("high",   ["最高", "high"]),
                ("low",    ["最低", "low"]),
                ("close",  ["收盘", "close"]),
                ("volume", ["成交量", "volume"]),
                ("amount", ["成交额", "amount"]),
            ]:
                for c in cands:
                    if c in df.columns and c != target:
                        df.rename(columns={c: target}, inplace=True)
                        break

            # 估值（独立请求，失败不阻断 OHLCV）
            try:
                _patch_ak_ua()
                df_v = ak.stock_a_lg_indicator(symbol=code_6)
                if df_v is not None and not df_v.empty:
                    df_v = df_v.copy()
                    v_date = next((c for c in ["trade_date", "date"] if c in df_v.columns), None)
                    if v_date:
                        df_v["date"] = pd.to_datetime(df_v[v_date],
                                                       errors="coerce").dt.strftime("%Y-%m-%d")
                        for target, cands in [
                            ("pe_ttm", ["pe_ttm", "pe"]),
                            ("pb_mrq", ["pb"]),
                            ("ps_ttm", ["ps_ttm", "ps"]),
                        ]:
                            for c in cands:
                                if c in df_v.columns:
                                    df_v[target] = pd.to_numeric(df_v[c], errors="coerce")
                                    break
                        df_v = df_v[["date"] + [c for c in ["pe_ttm","pb_mrq","ps_ttm"]
                                                if c in df_v.columns]]
                        df = df.merge(df_v, on="date", how="left")
            except Exception:
                pass

            # unadj_close（无法从 hfq 接口直接获取，暂设为 NaN，由 step3 处理）
            if "unadj_close" not in df.columns:
                df["unadj_close"] = np.nan

            df["date"] = pd.to_datetime(df.get("date"), errors="coerce").dt.strftime("%Y-%m-%d")
            return df if not df.empty else None

        except Exception:
            if attempt < 2:
                time.sleep(_backoff(attempt, False))
    return None


def _download_one(code: str, start: str, end: str,
                  parquet_dir: Path, delay: float) -> dict:
    """
    下载单只股票日线。
    V10 QFQ版：主力 AKShare adjust=qfq（保证前复权正确），备用 adata（通常返回HFQ但做降级处理）。
    code 为 sh.600519 格式；输出 sh.600519.parquet。
    """
    global _consecutive_failures

    code_6 = _to_6digit(code)

    with _lock:
        extra = min(_consecutive_failures * 0.3, 8.0)
    if extra > 0:
        time.sleep(extra)

    time.sleep(random.uniform(0, 1.5))

    raw_df = None
    source  = "none"
    df      = None

    # ── [V10 QFQ] 主力：AKShare adjust=qfq（前复权，保证复权正确性）──────
    raw_df = _try_akshare_daily(code_6, start, end)
    if raw_df is not None:
        source = "akshare"
        df = _clean_ohlcv(raw_df, code, vol_multiplier=1.0)  # AKShare: 保持「手」，fast_runner统一×100
    else:
        # ── 备用：adata（get_market返回HFQ，仅AKShare失败时降级使用）────
        raw_df = _try_adata_daily(code_6, start, end)
        if raw_df is not None:
            source = "adata"
            df = _clean_ohlcv(raw_df, code, vol_multiplier=0.01)  # adata返回「股」÷100=「手」，fast_runner×100还原
    if raw_df is None or df is None:
        with _lock:
            _consecutive_failures += 1
        return {"code": code, "status": "error", "rows": 0, "source": "none",
                "msg": "adata 和 AKShare 均失败"}

    # 补充估值列（若来源未提供）
    for col in ["pe_ttm", "pb_mrq", "ps_ttm", "unadj_close"]:
        if col not in df.columns:
            df[col] = np.nan

    keep = [c for c in ["date", "open", "high", "low", "close",
                         "volume", "amount", "code", "adj_type",
                         "pe_ttm", "pb_mrq", "ps_ttm", "unadj_close"] if c in df.columns]
    df = df[keep]

    # 合并已有 parquet
    out_path = parquet_dir / f"{code}.parquet"
    if out_path.exists():
        try:
            df_old = pd.read_parquet(str(out_path))
            if len(df_old) > 0:
                df = pd.concat([df_old, df], ignore_index=True)
                df = df.drop_duplicates(subset=["date"], keep="last")
                df = df.sort_values("date").reset_index(drop=True)
        except Exception:
            pass

    df.to_parquet(str(out_path), index=False, compression="snappy")

    with _lock:
        _consecutive_failures = max(0, _consecutive_failures - 1)

    time.sleep(delay + random.uniform(0, delay * 0.3))
    return {"code": code, "status": "ok", "rows": len(df),
            "source": source, "msg": ""}


# ─────────────────────────────────────────────────────────────────────────────
# is_st.npy 生成（替代 step0_patch_daily_fields 的 BaoStock is_st 下载）
# ─────────────────────────────────────────────────────────────────────────────

def _build_is_st_npy() -> bool:
    """
    构建 valuation_isST.npy：从股票名称中的"ST"字样推断。

    数据来源（降级）：
    1. adata.stock.info.*（版本探测，2.9.5 可用接口名未知）
    2. AKShare stock_info_a_code_name()（最稳定兜底）

    ST判断规则：名称含"ST"（覆盖 *ST、ST、退市ST 等所有变体）。
    沿时间轴静态广播（快照，不反映历史摘帽时刻，对回测已足够）。
    """
    if not META_PATH.exists():
        print("  ⚠ meta.json 不存在，跳过 is_st 生成（请先运行 build_npy）")
        return False

    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)

    codes = meta["codes"]
    N, T  = meta["shape"]
    code6_to_idx = {_to_6digit(str(c)): i for i, c in enumerate(codes)}

    st_set: set[str] = set()
    source = "unknown"

    # 1. 尝试 adata（探测可用接口）
    try:
        import adata
        info_candidates = [
            "all_stock", "stock_list", "get_all_code", "all_code",
            "base_info", "get_base_info", "all_stock_info", "stock_info",
        ]
        fn_name, fn = _probe_adata_fn(adata.stock.info, info_candidates)
        if fn:
            df_info = fn()
            code_col = next((c for c in ["stock_code","code"] if c in df_info.columns), None)
            name_col = next((c for c in ["short_name","name","stock_name","sec_name"]
                             if c in df_info.columns), None)
            if code_col and name_col:
                for _, row in df_info.iterrows():
                    if "ST" in str(row[name_col]).upper():
                        st_set.add(_to_6digit(str(row[code_col])))
                source = f"adata.stock.info.{fn_name}"
    except Exception:
        pass

    # 2. 降级到 AKShare
    if not st_set:
        try:
            import akshare as ak
            df_ak = ak.stock_info_a_code_name()
            code_col = next((c for c in ["code","股票代码"] if c in df_ak.columns), df_ak.columns[0])
            name_col = next((c for c in ["name","股票简称"] if c in df_ak.columns), df_ak.columns[1])
            for _, row in df_ak.iterrows():
                if "ST" in str(row[name_col]).upper():
                    st_set.add(_to_6digit(str(row[code_col])))
            source = "AKShare stock_info_a_code_name"
        except Exception as e:
            print(f"  ⚠ is_st 数据源均失败: {e}")
            return False

    # 构建 (N, T) 矩阵
    is_st_arr = np.zeros((N, T), dtype=np.bool_)
    count = 0
    for c6, idx in code6_to_idx.items():
        if c6 in st_set:
            is_st_arr[idx, :] = True
            count += 1

    out_path = NPY_DIR / "valuation_isST.npy"
    np.save(str(out_path), is_st_arr)
    print(f"  ✓ valuation_isST.npy  shape={is_st_arr.shape}  ST股={count}只  来源={source}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 扫描缺失
# ─────────────────────────────────────────────────────────────────────────────

def _scan_missing(parquet_dir: Path, codes: list[str],
                  start: str, force: bool) -> list[str]:
    """扫描缺失/空/过旧的 parquet 文件（sh.600519.parquet 格式）。"""
    missing = []
    cutoff  = (date.today() - timedelta(days=5)).strftime("%Y-%m-%d")
    for code in codes:
        pq = parquet_dir / f"{code}.parquet"
        if force:
            missing.append(code); continue
        if not pq.exists():
            missing.append(code); continue
        try:
            df_d = pd.read_parquet(str(pq), columns=["date"])
            if len(df_d) == 0 or str(df_d["date"].max()) < cutoff:
                missing.append(code)
        except Exception:
            missing.append(code)
    return missing


# ─────────────────────────────────────────────────────────────────────────────
# 批量下载主函数
# ─────────────────────────────────────────────────────────────────────────────

def run(codes: list[str], start: str, end: str, parquet_dir: Path,
        n_workers: int, delay: float, force: bool) -> None:

    parquet_dir.mkdir(parents=True, exist_ok=True)
    total = len(codes)
    t0 = time.time()

    print(f"\n  ▶ adata+AKShare 日线下载（{n_workers} 线程，delay={delay}s，共 {total} 只）")
    print(f"    范围: {start} ~ {end}  |  输出: {parquet_dir}")

    ok_n = err_n = empty_n = adata_n = ak_n = 0

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_download_one, code, start, end, parquet_dir, delay): code
            for code in codes
        }
        done = 0
        for future in as_completed(futures):
            code = futures[future]
            done += 1
            try:
                res = future.result()
            except Exception as e:
                res = {"code": code, "status": "error", "rows": 0,
                       "source": "none", "msg": str(e)}

            st = res["status"]
            src = res.get("source", "")
            if st == "ok":
                ok_n += 1
                if src == "adata":   adata_n += 1
                if src == "akshare": ak_n    += 1
            elif st == "error": err_n   += 1
            else:               empty_n += 1

            elapsed = time.time() - t0
            speed   = done / max(elapsed, 0.1)
            eta     = (total - done) / max(speed, 0.01)

            print(
                f"  [{done:4d}/{total}] {code}  {st:5s}[{src:7s}] {res['rows']:5d}行 | "
                f"ok={ok_n}(adata={adata_n},ak={ak_n}) err={err_n} | "
                f"{speed:.1f}只/s  ETA={eta:.0f}s",
                flush=True,
            )
            if st == "error":
                print(f"    ✗ {res['msg'][:80]}")

    elapsed = time.time() - t0
    print()
    print("=" * 70)
    print(f"  完成: ok={ok_n}(adata={adata_n} / akshare={ak_n})  "
          f"empty={empty_n}  error={err_n}")
    print(f"  耗时: {elapsed:.0f}s  平均: {total/max(elapsed,1):.1f}只/s")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 0c: adata+AKShare 完整日线下载（BaoStock 替代方案）")
    parser.add_argument("--start",       default="2015-01-05",
                        help="下载起始日（默认 2015-01-05）")
    parser.add_argument("--end",         default=None,
                        help="下载截止日（默认今日）")
    parser.add_argument("--workers",     type=int, default=8,
                        help="并发线程数（默认8，HTTP I/O密集用线程池，建议8~16）")
    parser.add_argument("--delay",       type=float, default=0.5,
                        help="每只下载后基础间隔秒（默认 0.5）")
    parser.add_argument("--force",       action="store_true",
                        help="强制重下全部")
    parser.add_argument("--incremental", action="store_true",
                        help="仅扫描过旧文件（距今>5天），不扫描全部")
    parser.add_argument("--scan-only",   action="store_true",
                        help="只扫描缺失情况，不下载")
    parser.add_argument("--test",        action="store_true",
                        help="测试模式（10只代表性股票）")
    parser.add_argument("--codes",       nargs="+", default=None,
                        help="指定股票代码（支持 600519 或 sh.600519 格式）")
    parser.add_argument("--no-is-st",    action="store_true",
                        help="跳过 is_st.npy 生成")
    args = parser.parse_args()

    # 依赖检查
    has_adata = False
    try:
        import adata as _a
        print(f"✓ adata 版本: {getattr(_a, '__version__', 'unknown')}（主力数据源）")
        has_adata = True
    except ImportError:
        print("⚠ adata 未安装：pip install adata")
        print("  将完全依赖 AKShare（备用路径）")

    try:
        import akshare as ak
        print(f"✓ AKShare 版本: {ak.__version__}（备用数据源）")
    except ImportError:
        print("✗ AKShare 也未安装，无法继续：pip install akshare -U")
        sys.exit(1)

    end_date     = args.end or date.today().strftime("%Y-%m-%d")
    parquet_dir  = _load_parquet_dir()

    # 股票列表
    if args.test:
        all_codes = [_to_full_code(c) for c in
                     ["600519", "601318", "000001", "000858", "300750",
                      "600036", "601012", "002594", "600900", "000002"]]
        print(f"⚡ 测试模式: {len(all_codes)} 只")
    elif args.codes:
        all_codes = [_to_full_code(c) for c in args.codes]
    else:
        all_codes = _load_all_codes()

    print(f"✓ 目标股票: {len(all_codes)} 只  |  parquet 目录: {parquet_dir}")

    # 扫描
    print(f"\n  扫描缺失/空/过旧文件...")
    missing = _scan_missing(parquet_dir, all_codes, args.start, args.force)
    existing = len(all_codes) - len(missing)
    print(f"  已有最新: {existing} 只  |  需要下载: {len(missing)} 只")

    if args.scan_only:
        if missing:
            print(f"\n  缺失列表（前30）:")
            for c in missing[:30]:
                print(f"    {c}")
            if len(missing) > 30:
                print(f"    ... 共 {len(missing)} 只")
        sys.exit(0)

    if not missing:
        print("\n✓ 所有文件均已完整，无需下载")
    else:
        est_min = len(missing) * (args.delay + 0.6) / args.workers / 60
        print(f"\n  预估耗时: ~{est_min:.0f} 分钟  ({args.workers}线程×{args.delay}s/只)")
        run(codes=missing, start=args.start, end=end_date,
            parquet_dir=parquet_dir, n_workers=args.workers,
            delay=args.delay, force=args.force)

    # 生成 is_st.npy
    if not args.no_is_st and NPY_DIR.exists():
        print("\n  生成 valuation_isST.npy...")
        _build_is_st_npy()

    print("\n✓ Step 0c 完成！后续步骤：")
    print("  python -m src.data.build_npy")
    print("  python scripts/step1_download_fundamental_akshare.py --workers 4 --delay 1.0")
    print("  python scripts/step3_build_fundamental_npy.py")
    print("  python scripts/step2_download_concepts.py")
    print("  python scripts/step4_build_concept_npy.py")
