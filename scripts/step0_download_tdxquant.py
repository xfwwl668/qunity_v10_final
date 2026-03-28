"""
scripts/step0_download_tdxquant.py
====================================
Q-UNITY V10 日线 OHLCV 下载器 -- TdxQuant 通达信本地版

【核心优势】
  完全本地，零网络依赖，不存在封IP/超时/返回空等问题
  adata get_market 时好时坏的问题彻底消除

【前提条件】
  1. 通达信客户端已打开并登录
  2. 已完成盘后数据下载（系统->盘后数据下载，起始2015-01-01）
  3. tqcenter.py 在通达信 PYPlugins/user/ 目录

【数据规格（实测确认 2026-03-20）】
  Volume  单位: 股(shares)，TdxQuant 直接返回股数，fast_runner vol_multiplier=1
  Amount  单位: 万元，存 parquet 时 x10000 转元
  dividend_type='front' = QFQ 前复权
  返回格式: dict of DataFrame（宽表），转换为项目标准长表 parquet

【输出格式】（与 step0_download_ohlcv.py 完全兼容）
  data/daily_parquet_qfq/{sh.600519}.parquet
  列: date / open / high / low / close / volume / amount / code / listing_date / adj_type

【用法】
  # 确保通达信已打开，然后：
  python scripts/step0_download_tdxquant.py --test
  python scripts/step0_download_tdxquant.py --workers 8
  python scripts/step0_download_tdxquant.py --workers 8 --incremental
  python scripts/step0_download_tdxquant.py --build-npy
"""

import argparse
import json
import sys
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd

# ── tqcenter 路径自动检测 ────────────────────────────────────────────────────
# [CONFIG] TdxQuant 路径从 config.json 统一读取，不写死
# config.json -> tdxquant.tq_dir 或 tdxquant.search_paths
try:
    _THIS_SCRIPT = Path(__file__).resolve()
    _PROJ_ROOT   = _THIS_SCRIPT.parent.parent
    sys.path.insert(0, str(_PROJ_ROOT / "scripts"))
    from tqcenter_utils import find_tqcenter as _find_tqcenter_util
    _TQ_DIR_STR = _find_tqcenter_util()
    if _TQ_DIR_STR:
        sys.path.insert(0, _TQ_DIR_STR)
        _TQ_DIR = Path(_TQ_DIR_STR)
    else:
        _TQ_DIR = None
except Exception:
    _TQ_DIR = None

def _find_tqcenter():
    """从 config.json 读取 TdxQuant 路径（兼容旧调用）"""
    return str(_TQ_DIR) if _TQ_DIR else None

# ── 项目路径 ─────────────────────────────────────────────────────────────────
PROJECT_ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR            = PROJECT_ROOT / "data"
DEFAULT_PARQUET_DIR = DATA_DIR / "daily_parquet_qfq"
NPY_DIR             = DATA_DIR / "npy_v10"
META_PATH           = NPY_DIR / "meta.json"

try:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from utils_paths import get_npy_dir, get_parquet_dir
    DEFAULT_PARQUET_DIR = get_parquet_dir("qfq")
    NPY_DIR   = get_npy_dir("v10")
    META_PATH = NPY_DIR / "meta.json"
except Exception:
    pass

ADJ_TYPE         = "qfq"
LISTING_DAYS_MIN = 60
MARKET_INDEX_CODE = "000300.SH"  # 沪深300

_lock = Lock()


# ─────────────────────────────────────────────────────────────────────────────
# 代码转换工具
# ─────────────────────────────────────────────────────────────────────────────

def _tdx_to_full(code: str) -> str:
    """600519.SH -> sh.600519"""
    if "." not in code:
        return code
    num, mkt = code.split(".", 1)
    return f"{mkt.lower()}.{num}"

def _full_to_tdx(code: str) -> str:
    """sh.600519 -> 600519.SH"""
    if "." not in code:
        return code
    parts = code.split(".", 1)
    if parts[0].lower() in ("sh", "sz", "bj"):
        return f"{parts[1]}.{parts[0].upper()}"
    return code  # 已是 tdx 格式

def _is_a_stock(code: str) -> bool:
    """只保留主板/创业板/科创板（过滤指数/ETF/北交所可选）"""
    # tdx 格式: 600519.SH 或 full 格式: sh.600519
    c = code.split(".")
    num = c[0] if c[0].isdigit() else c[1] if len(c) > 1 else ""
    num = num.zfill(6)
    sh_ok = num.startswith(("600","601","603","605","688","689"))
    sz_ok = num.startswith(("000","001","002","003","300","301"))
    bj_ok = num.startswith(("83","87","43","92"))  # 北交所，可选
    return sh_ok or sz_ok


# ─────────────────────────────────────────────────────────────────────────────
# 股票列表
# ─────────────────────────────────────────────────────────────────────────────

def get_stock_universe(tq) -> list:
    """
    获取全市场 A 股列表，返回 tdx 格式代码列表（如 600519.SH）。

    实测确认（2026-03-21 Part3）：
      tq.get_stock_list('5')         = 5500只 ✅  字符串'5'，不是int
      tq.get_stock_list(list_type=0) = 5500只 ✅  返回字符串列表
      tq.get_stock_list(list_type=1) = 5500只 ✅  返回[{"Code":...,"Name":...}]
      tq.get_stock_list(list_type=5) = 0只   ✗  int 5 无效

    降级策略：
      1. get_stock_list('5')         ← 主力，字符串参数
      2. get_stock_list(list_type=0) ← 备用，list_type=0
      3. 扫描已有 parquet 文件        ← 增量更新时可用
      4. 离线号段枚举                 ← 最后兜底
    """
    codes = []

    # ── 方式1：字符串参数 '5'（实测5500只）────────────────────────────────
    try:
        result = tq.get_stock_list('5')
        if result and len(result) > 1000:
            for item in result:
                if isinstance(item, str):
                    c = item
                elif isinstance(item, dict):
                    c = item.get("Code", item.get("code", ""))
                else:
                    c = str(item)
                if c and _is_a_stock(c):
                    codes.append(c)
            if codes:
                print(f"  [TdxQuant] 股票列表(get_stock_list('5')): {len(codes)} 只")
                return sorted(set(codes))
    except Exception:
        pass

    # ── 方式2：list_type=0（实测5500只，返回字符串列表）──────────────────
    try:
        result = tq.get_stock_list(list_type=0)
        if result and len(result) > 1000:
            for item in result:
                if isinstance(item, str):
                    c = item
                elif isinstance(item, dict):
                    c = item.get("Code", item.get("code", ""))
                else:
                    c = str(item)
                if c and _is_a_stock(c):
                    codes.append(c)
            if codes:
                print(f"  [TdxQuant] 股票列表(list_type=0): {len(codes)} 只")
                return sorted(set(codes))
    except Exception:
        pass

    # ── 方式3：list_type=1（返回含Name的dict列表）────────────────────────
    try:
        result = tq.get_stock_list(list_type=1)
        if result and len(result) > 1000:
            for item in result:
                c = item.get("Code", "") if isinstance(item, dict) else str(item)
                if c and _is_a_stock(c):
                    codes.append(c)
            if codes:
                print(f"  [TdxQuant] 股票列表(list_type=1): {len(codes)} 只")
                return sorted(set(codes))
    except Exception:
        pass

    # ── 方式4: 扫描已有 parquet（增量更新时的快速路径）──────────────────
    parquet_dir = DEFAULT_PARQUET_DIR
    if parquet_dir.exists():
        pq_codes = []
        for pq in parquet_dir.glob("*.parquet"):
            full = pq.stem   # sh.600519
            tdx  = _full_to_tdx(full)  # 600519.SH
            if _is_a_stock(tdx):
                pq_codes.append(tdx)
        if pq_codes:
            print(f"  [TdxQuant] 从已有 parquet 扫描: {len(pq_codes)} 只")
            return sorted(set(pq_codes))

    # ── 方式5: 离线号段枚举（兜底，含不存在号码，下载时自动跳过）──────────
    print("  [TdxQuant] 使用离线号段枚举（约12000个候选，不存在号码自动跳过）")
    for prefix in ["600","601","603","605","688"]:
        lo = int(prefix) * 1000
        hi = lo + 999
        for n in range(lo, hi+1):
            codes.append(f"{n:06d}.SH")
    for prefix_range in [("000",1,999),("001",1000,1999),("002",2000,2999),
                          ("003",3000,3999),("300",300000,300999),("301",301000,301999)]:
        _, lo, hi = prefix_range
        for n in range(lo, hi+1):
            codes.append(f"{n:06d}.SZ")
    print(f"  [TdxQuant] 号段枚举: {len(codes)} 个候选（含不存在号码，空数据自动跳过）")
    return codes


# ─────────────────────────────────────────────────────────────────────────────
# 增量检查
# ─────────────────────────────────────────────────────────────────────────────

def _needs_update(parquet_path: Path, cutoff_days: int = 3) -> bool:
    if not parquet_path.exists():
        return True
    try:
        df = pd.read_parquet(str(parquet_path), columns=["date"])
        if df.empty:
            return True
        latest  = str(df["date"].max())
        cutoff  = (date.today() - timedelta(days=cutoff_days)).strftime("%Y-%m-%d")
        return latest < cutoff
    except Exception:
        return True


# ─────────────────────────────────────────────────────────────────────────────
# 单只股票下载
# ─────────────────────────────────────────────────────────────────────────────

def _download_one(
    tdx_code: str,      # 600519.SH
    tq,
    parquet_dir: Path,
    start_date: str,    # YYYYMMDD
    end_date: str,      # YYYYMMDD
    incremental: bool,
) -> dict:
    """
    下载单只股票日线，保存为 parquet。

    数据单位：
      Volume: 手（保持，fast_runner vol_multiplier=100 统一转股）
      Amount: 万元 -> x10000 -> 元（存为 float64）
    """
    full_code = _tdx_to_full(tdx_code)   # sh.600519
    out_path  = parquet_dir / f"{full_code}.parquet"

    if incremental and not _needs_update(out_path):
        try:
            df_ex = pd.read_parquet(str(out_path), columns=["date"])
            return {"code": full_code, "status": "skip", "rows": len(df_ex), "msg": ""}
        except Exception:
            pass

    for attempt in range(3):
        try:
            # tqcenter 内部对缺失字段/重连会直接 print() 到 stdout，
            # 无法安全静音（redirect_stdout 修改进程级 sys.stdout 会导致多线程下进度条冻结）。
            # 警告是纯视觉噪音，数据正确，接受它。
            data = tq.get_market_data(
                field_list    = ["Open","High","Low","Close","Volume","Amount"],
                stock_list    = [tdx_code],
                start_time    = start_date,
                end_time      = end_date,
                dividend_type = "front",   # QFQ 前复权
                period        = "1d",
                fill_data     = True,
            )

            if not data or "Close" not in data:
                return {"code": full_code, "status": "empty", "rows": 0,
                        "msg": "get_market_data 返回空"}

            close_df = data["Close"]
            if tdx_code not in close_df.columns or close_df[tdx_code].dropna().empty:
                return {"code": full_code, "status": "empty", "rows": 0,
                        "msg": "该股票无数据"}

            # ── 直接从各字段 series 组装 DataFrame ────────────────────────
            df = pd.DataFrame(index=close_df.index)
            for field, col in [("Open","open"),("High","high"),("Low","low"),
                                ("Close","close"),("Volume","volume"),("Amount","amount")]:
                if field in data and tdx_code in data[field].columns:
                    df[col] = data[field][tdx_code]
                else:
                    df[col] = np.nan

            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: "date"})

            # ── 日期处理 ──────────────────────────────────────────────────
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            df = df.dropna(subset=["date","close"])

            # ── 数值类型 ──────────────────────────────────────────────────
            for col in ["open","high","low","close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
            df = df[df["close"] > 0].copy()

            if df.empty:
                return {"code": full_code, "status": "empty", "rows": 0,
                        "msg": "清洗后为空"}

            # Volume: TdxQuant 返回「股」，直接存储，vol_multiplier=1
            # fast_runner_v10.py vol_multiplier=1（TdxQuant 已为股数，无需转换）
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("float32")

            # Amount: 万元 -> 元（实测确认：TdxQuant Amount 单位=万元）
            df["amount"] = (pd.to_numeric(df["amount"], errors="coerce").fillna(0) * 10000).astype("float64")

            # ── 元信息 ────────────────────────────────────────────────────
            df["code"]         = full_code
            df["listing_date"] = df["date"].min()   # 先用数据第一行，--patch-dates 后用 J_start 修正
            df["adj_type"]     = ADJ_TYPE            # "qfq"

            df = df[["date","open","high","low","close","volume","amount",
                      "code","listing_date","adj_type"]]
            df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
            df = df.reset_index(drop=True)

            # ── 合并已有 parquet（增量追加）──────────────────────────────
            if out_path.exists():
                try:
                    df_old = pd.read_parquet(str(out_path))
                    if len(df_old) > 0:
                        existing = set(df["date"].tolist())
                        df_old   = df_old[~df_old["date"].isin(existing)]
                        df       = pd.concat([df_old, df], ignore_index=True)
                        df       = df.sort_values("date").reset_index(drop=True)
                except Exception:
                    pass

            df.to_parquet(str(out_path), index=False, compression="snappy")
            return {"code": full_code, "status": "ok", "rows": len(df), "msg": ""}

        except Exception as e:
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
            else:
                return {"code": full_code, "status": "error", "rows": 0,
                        "msg": str(e)[:120]}

    return {"code": full_code, "status": "error", "rows": 0, "msg": "3次重试失败"}


# ─────────────────────────────────────────────────────────────────────────────
# 批量下载
# ─────────────────────────────────────────────────────────────────────────────

def run_download(
    tdx_codes: list,
    tq,
    parquet_dir: Path,
    start_date: str,
    end_date: str,
    n_workers: int,
    incremental: bool,
    force: bool,
) -> None:
    parquet_dir.mkdir(parents=True, exist_ok=True)

    if force:
        deleted = 0
        for code in tdx_codes:
            p = parquet_dir / f"{_tdx_to_full(code)}.parquet"
            if p.exists():
                p.unlink(); deleted += 1
        if deleted:
            print(f"  已删除 {deleted} 个旧文件")

    total = len(tdx_codes)
    t0    = time.time()
    ok_n = err_n = skip_n = empty_n = 0

    # COM 接口安全并发上限：8。超过后多线程竞争导致 COM 状态不一致进而崩溃。
    safe_workers = min(n_workers, 8)
    if safe_workers < n_workers:
        print(f"  ⚠ 并发数从 {n_workers} 降至 {safe_workers}（TdxQuant COM 接口安全上限）")
    n_workers = safe_workers

    print(f"\n{'='*65}")
    print(f"  Q-UNITY V10 日线下载 -- TdxQuant 通达信本地版")
    print(f"  股票数: {total}  并发: {n_workers}  QFQ前复权")
    print(f"  Volume单位: 手(x100=股)  Amount单位: 万元->x10000->元")
    print(f"  期间: {start_date} ~ {end_date}")
    print(f"  输出: {parquet_dir}")
    print(f"{'='*65}\n")

    # TdxQuant 是本地接口，并发安全，无需限速
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                _download_one, code, tq, parquet_dir,
                start_date, end_date, incremental
            ): code
            for code in tdx_codes
        }
        done = 0
        for future in as_completed(futures):
            code = futures[future]
            done += 1
            try:
                res = future.result()
            except Exception as e:
                res = {"code": _tdx_to_full(code), "status": "error",
                       "rows": 0, "msg": str(e)}

            st = res["status"]
            if   st == "ok":    ok_n    += 1
            elif st == "skip":  skip_n  += 1
            elif st == "empty": empty_n += 1
            else:               err_n   += 1

            elapsed = time.time() - t0
            spd     = done / max(elapsed, 0.1)
            eta     = (total - done) / max(spd, 0.01)
            print(f"\r  [{done:4d}/{total}] {res['code'][:14]:<14} "
                  f"{st:5s} {res['rows']:5d}行 | "
                  f"ok={ok_n} skip={skip_n} empty={empty_n} err={err_n} | "
                  f"{spd:.1f}只/s ETA={eta:.0f}s  ",
                  end="", flush=True)
            if st == "error":
                print(f"\n    ✗ {res['msg']}")

    elapsed = time.time() - t0
    print(f"\n\n{'='*65}")
    print(f"  完成: ok={ok_n}  skip={skip_n}  empty={empty_n}  error={err_n}")
    print(f"  耗时: {elapsed:.0f}s  平均: {total/max(elapsed,1):.1f}只/s")
    print(f"{'='*65}")


# ─────────────────────────────────────────────────────────────────────────────
# 上市日期补充（从 get_stock_info.J_start）
# ─────────────────────────────────────────────────────────────────────────────

def patch_listing_dates(tdx_codes: list, tq, parquet_dir: Path) -> None:
    """
    用 get_stock_info.J_start 修正每个 parquet 的 listing_date。
    get_market_data 返回的 listing_date 只是数据第一行日期，
    J_start 才是真实上市日期（如平安银行=19910403）。
    """
    print("\n  补充上市日期（get_stock_info.J_start）...")
    patched = 0
    for tdx_code in tdx_codes:
        full_code = _tdx_to_full(tdx_code)
        pq_path   = parquet_dir / f"{full_code}.parquet"
        if not pq_path.exists():
            continue
        try:
            info = tq.get_stock_info(stock_code=tdx_code, field_list=["J_start"])
            if not info:
                continue
            j_start = info.get("J_start")
            if not j_start or j_start == 0:
                continue
            # J_start 格式：20010827（整数）
            listing_str = str(int(j_start))
            listing_date = (f"{listing_str[:4]}-{listing_str[4:6]}-{listing_str[6:8]}"
                            if len(listing_str) == 8 else None)
            if not listing_date:
                continue

            df = pd.read_parquet(str(pq_path))
            if df["listing_date"].iloc[0] != listing_date:
                df["listing_date"] = listing_date
                df.to_parquet(str(pq_path), index=False, compression="snappy")
                patched += 1
        except Exception:
            pass
    print(f"  已修正 {patched} 只股票的上市日期")


# ─────────────────────────────────────────────────────────────────────────────
# 市场指数下载（沪深300 -> market_index.npy）
# ─────────────────────────────────────────────────────────────────────────────

def download_market_index(
    tq,
    npy_dir: Path,
    start_date: str,
    end_date: str,
    trading_days: list = None,
) -> bool:
    print(f"\n  下载市场指数: {MARKET_INDEX_CODE}")
    try:
        data = tq.get_market_data(
            field_list    = ["Close"],
            stock_list    = [MARKET_INDEX_CODE],
            start_time    = start_date,
            end_time      = end_date,
            dividend_type = "none",
            period        = "1d",
        )
        if not data or "Close" not in data:
            print("  ✗ 市场指数返回空")
            return False

        df_idx = data["Close"]
        if MARKET_INDEX_CODE not in df_idx.columns:
            print(f"  ✗ {MARKET_INDEX_CODE} 不在返回列中")
            return False

        df_idx = df_idx[[MARKET_INDEX_CODE]].copy()
        df_idx.index = pd.to_datetime(df_idx.index)
        df_idx = df_idx.sort_index()
        df_idx["date"] = df_idx.index.strftime("%Y-%m-%d")
        df_idx["close"] = pd.to_numeric(df_idx[MARKET_INDEX_CODE], errors="coerce").fillna(0)

        if trading_days is not None:
            date_to_idx = {d: i for i, d in enumerate(trading_days)}
            T = len(trading_days)
            arr = np.full(T, np.nan)
            for _, row in df_idx.iterrows():
                ti = date_to_idx.get(row["date"])
                if ti is not None:
                    arr[ti] = float(row["close"])
            # 前向填充
            last = np.nan
            for i in range(T):
                if not np.isnan(arr[i]):
                    last = arr[i]
                arr[i] = last
            np.nan_to_num(arr, nan=0.0, copy=False)
            mkt_arr   = arr.reshape(1, T).astype(np.float32)
            dates_arr = np.array(trading_days, dtype=object)
        else:
            mkt_arr   = df_idx["close"].to_numpy(dtype=np.float32).reshape(1, -1)
            dates_arr = df_idx["date"].to_numpy(dtype=object)

        npy_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(npy_dir / "market_index.npy"), mkt_arr)
        np.save(str(npy_dir / "market_index_dates.npy"), dates_arr)
        print(f"  ✓ market_index.npy  shape={mkt_arr.shape}  "
              f"range={dates_arr[0]}~{dates_arr[-1]}")
        return True

    except Exception as e:
        print(f"  ✗ 市场指数下载失败: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# build_npy（调用原有 step0_download_ohlcv.py 的 build_npy）
# ─────────────────────────────────────────────────────────────────────────────

def trigger_build_npy(parquet_dir: Path, npy_dir: Path, start: str, end: str) -> None:
    """
    调用 step0_download_ohlcv.py 的 build_npy() 构建 npy 矩阵。
    避免重复实现 valid_mask 等复杂逻辑。
    """
    print("\n  触发 build_npy（复用 step0_download_ohlcv.py）...")
    import subprocess
    script = PROJECT_ROOT / "scripts" / "step0_download_ohlcv.py"
    if not script.exists():
        print(f"  ✗ {script} 不存在")
        return
    rc = subprocess.run([
        sys.executable, str(script),
        "--incremental", "--build-npy",
        "--parquet-dir", str(parquet_dir),
        "--npy-dir",     str(npy_dir),
        "--start",       start,
        "--end",         end,
        "--source",      "adata",  # 仅构建 npy，不下载数据
    ]).returncode
    if rc == 0:
        print("  ✓ npy 矩阵构建完成")
    else:
        print("  ✗ npy 构建失败，请手动运行 step0_download_ohlcv.py --build-npy")


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Q-UNITY V10 日线下载 -- TdxQuant 通达信本地版")
    ap.add_argument("--workers",     type=int,   default=8,
                    help="并发线程数（默认8，TdxQuant本地接口可设16~32）")
    ap.add_argument("--start",       default="20150101",
                    help="起始日期 YYYYMMDD（默认20150101）")
    ap.add_argument("--end",         default=None,
                    help="结束日期 YYYYMMDD（默认今日）")
    ap.add_argument("--incremental", action="store_true",
                    help="增量模式：跳过已有最新文件")
    ap.add_argument("--force",       action="store_true",
                    help="强制重下（删除旧文件）")
    ap.add_argument("--test",        action="store_true",
                    help="测试模式（10只）")
    ap.add_argument("--build-npy",   action="store_true",
                    help="下载后自动构建 npy 矩阵")
    ap.add_argument("--patch-dates", action="store_true", default=True,
                    help="[FIX-S-01] 下载后补充真实上市日期（J_start，默认开启）。"
                         " 用 --no-patch-dates 禁用。")
    ap.add_argument("--no-patch-dates", dest="patch_dates", action="store_false",
                    help="禁用上市日期修正（仅用于调试）")
    ap.add_argument("--out-dir",     default=None,
                    help="parquet 输出目录")
    ap.add_argument("--npy-dir",     default=None,
                    help="npy 输出目录")
    ap.add_argument("--tq-dir",      default=None,
                    help="tqcenter.py 所在目录（自动检测失败时手动指定）")
    args = ap.parse_args()

    # ── tqcenter 路径 ─────────────────────────────────────────────────────
    if args.tq_dir:
        sys.path.insert(0, args.tq_dir)

    try:
        from tqcenter import tq as _tq
    except ImportError:
        print("✗ 无法导入 tqcenter")
        print("  请确认：")
        print("  1. 通达信客户端已打开")
        print("  2. tqcenter.py 在以下路径之一：")
        for p in [r"D:\SOFT(DONE)\tdx\ncb\PYPlugins\user",
                  r"C:\new_tdx\PYPlugins\user"]:
            print(f"     {p}")
        print("  3. 或在 config.json -> tdxquant.tq_dir 指定路径")
        sys.exit(1)

    # ── 初始化 ────────────────────────────────────────────────────────────
    _tq.initialize(__file__)
    print("✓ TdxQuant 初始化成功")

    end_date    = args.end or date.today().strftime("%Y%m%d")
    parquet_dir = Path(args.out_dir) if args.out_dir else DEFAULT_PARQUET_DIR
    npy_dir_    = Path(args.npy_dir) if args.npy_dir else NPY_DIR

    try:
        # ── 获取股票列表 ──────────────────────────────────────────────────
        if args.test:
            tdx_codes = ["600519.SH","601318.SH","000001.SZ","000858.SZ",
                         "300750.SZ","600036.SH","002594.SZ","600900.SH",
                         "300014.SZ","688981.SH"]
            print(f"⚡ 测试模式: {len(tdx_codes)} 只")
        else:
            print("▶ 获取全市场股票列表...")
            tdx_codes = get_stock_universe(_tq)
            print(f"  共 {len(tdx_codes)} 只 A 股")

        # ── 下载日线 ──────────────────────────────────────────────────────
        run_download(
            tdx_codes   = tdx_codes,
            tq          = _tq,
            parquet_dir = parquet_dir,
            start_date  = args.start,
            end_date    = end_date,
            n_workers   = args.workers,
            incremental = args.incremental and not args.force,
            force       = args.force,
        )

        # ── 补充上市日期 ──────────────────────────────────────────────────
        if args.patch_dates:
            patch_listing_dates(tdx_codes, _tq, parquet_dir)

        # ── 构建 npy ──────────────────────────────────────────────────────
        if args.build_npy:
            trigger_build_npy(parquet_dir, npy_dir_, args.start, end_date)

        print(f"\n✓ Step 0 完成！（TdxQuant）")
        print(f"  下一步: python scripts/step0_download_ohlcv.py --build-npy "
              f"--parquet-dir {parquet_dir}")

    finally:
        try:
            _tq.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
