"""
scripts/step0b_fill_missing_daily_akshare.py

BaoStock 封禁后的日线数据补全工具（AKShare 版）

使用场景：
  BaoStock 因多进程并发登录触发黑名单，导致部分股票日线 parquet 文件
  缺失或为空。本脚本扫描 data/daily_parquet/ 目录，找出缺失/空文件，
  用 AKShare stock_zh_a_hist 补下并保存为兼容格式。

AKShare 日线接口特点：
  · 来源：东方财富，无 TCP 登录，不会出现 BaoStock 黑名单问题
  · adjust='qfq' = 前复权（V10 已从 hfq 改为 qfq，保持与 V10 主路径一致）
  · 单次请求返回全部历史，无需按年循环
  · 成交量单位为「手」(×100)，需转为「股」与 BaoStock 格式统一
  · 并发限速：建议 3~6 线程，超过容易被东财封 UA

字段映射（AKShare → 项目 parquet）：
  日期       → date
  开盘       → open
  最高       → high
  最低       → low
  收盘       → close
  成交量     → volume  （× 100：手→股）
  成交额     → amount

用法：
  python scripts/step0b_fill_missing_daily_akshare.py             # 扫描补全
  python scripts/step0b_fill_missing_daily_akshare.py --workers 4
  python scripts/step0b_fill_missing_daily_akshare.py --start 2015-01-05
  python scripts/step0b_fill_missing_daily_akshare.py --force     # 强制重下全部
  python scripts/step0b_fill_missing_daily_akshare.py --scan-only # 仅扫描，不下载
  python scripts/step0b_fill_missing_daily_akshare.py --codes 000001 600000  # 指定股票
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np
import pandas as pd

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

# config.json 中 parquet_dir 的默认路径

# 随机 UA 池（防东财封禁）
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
]

_lock = Lock()
_consecutive_failures = 0


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _load_parquet_dir() -> Path:
    """从 config.json 读取 parquet_dir，找不到时用默认路径。"""
    cfg_path = PROJECT_ROOT / "config.json"
    if cfg_path.exists():
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            raw = cfg.get("parquet_dir_qfq") or cfg.get("parquet_dir") or cfg.get("data", {}).get("parquet_dir", "")  # V10: qfq优先
            if raw:
                p = Path(raw)
                return p if p.is_absolute() else PROJECT_ROOT / p
        except Exception:
            pass
    return DEFAULT_PARQUET_DIR


def _to_full_code(code: str) -> str:
    """
    将任意格式股票代码转换为 sh.XXXXXX / sz.XXXXXX 格式。

    [CRITICAL] columnar_adapter._is_valid_a_stock() 要求文件名格式为
    sh.600519.parquet / sz.000001.parquet，不带前缀的文件会被完全忽略！

    规则（与 step0_patch_daily_fields.to_bs_code 一致）：
      首位 6/9 → sh（沪市主板/科创板）
      其余首位 → sz（深市主板/中小板/创业板）
    """
    s = str(code).strip()
    if s.startswith(("sh.", "sz.")):
        return s
    c = s.split(".")[-1].zfill(6)
    prefix = "sh" if c[0] in ("6", "9") else "sz"
    return f"{prefix}.{c}"


def _to_6digit(code: str) -> str:
    """sh.600519 → 600519，供 AKShare 调用（只接受6位纯数字）。"""
    return str(code).strip().split(".")[-1].zfill(6)


def _load_codes() -> list[str]:
    """
    从 meta.json 加载全市场股票代码，返回 sh.600519 / sz.000001 格式。
    若 meta.json 不存在，扫描 parquet_dir 中已有的 parquet 文件。
    """
    if META_PATH.exists():
        with open(META_PATH, encoding="utf-8") as f:
            meta = json.load(f)
        return [_to_full_code(str(c)) for c in meta["codes"]]

    parquet_dir = _load_parquet_dir()
    existing = [p.stem for p in sorted(parquet_dir.glob("*.parquet"))
                if p.stem != "stock_list"]
    if existing:
        return [_to_full_code(c) for c in existing]

    print("✗ 未找到 meta.json 也无 parquet 文件，请先完成 build_npy 或指定 --codes")
    sys.exit(1)


def _scan_missing(parquet_dir: Path, codes: list[str],
                  start: str, force: bool) -> list[str]:
    """
    扫描缺失/空/过旧的 parquet 文件，返回需要补下的股票代码列表。
    codes 参数为 sh.600519 格式，parquet 文件名也是 sh.600519.parquet。

    判断标准：
    1. parquet 文件不存在
    2. 文件存在但行数为 0（BaoStock 被封后写出的空文件）
    3. 文件最新日期距今 > 5 天（增量更新场景）
    4. --force 模式：全部重下
    """
    missing = []
    cutoff  = (date.today() - timedelta(days=5)).strftime("%Y-%m-%d")

    for code in codes:
        # code 是 sh.600519 格式，文件名是 sh.600519.parquet
        pq = parquet_dir / f"{code}.parquet"
        if force:
            missing.append(code)
            continue
        if not pq.exists():
            missing.append(code)
            continue
        try:
            df_date = pd.read_parquet(str(pq), columns=["date"])
            if len(df_date) == 0:
                missing.append(code)
            elif str(df_date["date"].max()) < cutoff:
                missing.append(code)
        except Exception:
            missing.append(code)

    return missing


def _patch_akshare_ua():
    """给 AKShare 内部 session 注入随机 UA + Referer。"""
    try:
        import akshare.utils.func as _ak_func
        session = getattr(_ak_func, "session", None)
        if session is None:
            import requests
            session = requests.Session()
            _ak_func.session = session
        session.headers.update({
            "User-Agent":      random.choice(_USER_AGENTS),
            "Referer":         "https://www.eastmoney.com/",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        })
    except Exception:
        pass


def _is_rate_limit(msg: str) -> bool:
    keywords = ["429", "限流", "频繁", "too many", "rate limit",
                "blocked", "forbidden", "访问过于", "请求过多"]
    low = msg.lower()
    return any(k in low for k in keywords)


def _backoff(attempt: int, is_limit: bool) -> float:
    if is_limit:
        return 5.0 * (attempt + 1) + random.uniform(0, 2)
    return (2.0 ** attempt) + random.uniform(0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 单只股票下载（AKShare）
# ─────────────────────────────────────────────────────────────────────────────

def _download_one(code: str, start: str, end: str,
                  parquet_dir: Path, delay: float) -> dict:
    """
    用 AKShare stock_zh_a_hist 下载单只股票日线数据。

    code 参数为 sh.600519 格式（columnar_adapter 要求）。
    AKShare 只接受6位纯数字，内部自动转换。
    输出：sh.600519.parquet（与 BaoStock 格式完全兼容）。
    """
    global _consecutive_failures

    code_6 = _to_6digit(code)   # AKShare 用：600519
    # parquet 文件名用完整格式：sh.600519.parquet

    with _lock:
        extra = min(_consecutive_failures * 0.3, 8.0)
    if extra > 0:
        time.sleep(extra)

    import akshare as ak
    time.sleep(random.uniform(0, 2.0))

    for attempt in range(3):
        try:
            _patch_akshare_ua()

            df = ak.stock_zh_a_hist(
                symbol=code_6,
                period="daily",
                start_date=start.replace("-", ""),
                end_date=end.replace("-", ""),
                adjust="qfq"  # V10 QFQ,
            )

            if df is None or df.empty:
                return {"code": code, "status": "empty", "rows": 0,
                        "msg": "AKShare 返回空数据"}

            df = df.copy()

            # ── 字段重命名 ─────────────────────────────────────────────────
            rename_map = {}
            _col_candidates = {
                "date":   ["日期", "date", "trade_date"],
                "open":   ["开盘", "open"],
                "high":   ["最高", "high"],
                "low":    ["最低", "low"],
                "close":  ["收盘", "close"],
                "volume": ["成交量", "volume", "vol"],
                "amount": ["成交额", "amount", "turnover", "money"],
            }
            for target, candidates in _col_candidates.items():
                for c in candidates:
                    if c in df.columns and c != target:
                        rename_map[c] = target
                        break
            df = df.rename(columns=rename_map)

            required = ["date", "open", "high", "low", "close"]
            missing_cols = [c for c in required if c not in df.columns]
            if missing_cols:
                return {"code": code, "status": "error", "rows": 0,
                        "msg": f"缺少列: {missing_cols}"}

            # ── 数据清洗 ───────────────────────────────────────────────────
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            df = df.dropna(subset=["date", "close"])

            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
            df = df[df["close"] > 0]

            # [UNIT-FIX] AKShare 成交量单位=手，BaoStock=股，统一为股（×100）
            if "volume" in df.columns:
                df["volume"] = (pd.to_numeric(df["volume"], errors="coerce")
                                .fillna(0.0) * 100).astype("float32")
            else:
                df["volume"] = 0.0

            if "amount" in df.columns:
                df["amount"] = (pd.to_numeric(df["amount"], errors="coerce")
                                .fillna(0.0)).astype("float32")
            else:
                df["amount"] = 0.0

            # [CODE-FIX] code 列用完整格式 sh.600519，与 BaoStock parquet 一致
            df["code"] = code
            keep = [c for c in ["date", "open", "high", "low", "close",
                                 "volume", "amount", "code"] if c in df.columns]
            df = df[keep].sort_values("date").reset_index(drop=True)

            if len(df) == 0:
                return {"code": code, "status": "empty", "rows": 0, "msg": "清洗后为空"}

            # ── 合并已有 parquet ───────────────────────────────────────────
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

            time.sleep(delay + random.uniform(0, delay * 0.5))
            return {"code": code, "status": "ok", "rows": len(df), "msg": ""}

        except Exception as e:
            err = str(e)
            is_limit = _is_rate_limit(err)
            if attempt < 2:
                time.sleep(_backoff(attempt, is_limit))

    with _lock:
        _consecutive_failures += 1

    return {"code": code, "status": "error", "rows": 0,
            "msg": f"3次重试均失败: {err}"}


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def run(codes: list[str], start: str, end: str, parquet_dir: Path,
        n_workers: int, delay: float, force: bool) -> dict[str, str]:

    parquet_dir.mkdir(parents=True, exist_ok=True)
    total = len(codes)
    results: dict[str, str] = {}
    t0 = time.time()
    done = 0

    print(f"\n  ▶ AKShare 补全下载（{n_workers} 线程，delay={delay}s，{total} 只）")
    print(f"    范围: {start} ~ {end}")
    print(f"    输出: {parquet_dir}")
    print()

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_download_one, code, start, end,
                            parquet_dir, delay): code
            for code in codes
        }

        ok_n = err_n = empty_n = 0

        for future in as_completed(futures):
            code = futures[future]
            done += 1
            try:
                res = future.result()
            except Exception as e:
                res = {"code": code, "status": "error", "rows": 0, "msg": str(e)}

            status = res["status"]
            results[code] = status

            if status == "ok":      ok_n    += 1
            elif status == "error": err_n   += 1
            else:                   empty_n += 1

            elapsed = time.time() - t0
            speed   = done / max(elapsed, 0.1)
            eta     = (total - done) / max(speed, 0.01)

            print(
                f"  [{done:4d}/{total}] {code}  {status:5s} {res['rows']:5d}行 | "
                f"ok={ok_n} err={err_n} empty={empty_n} | "
                f"{speed:.1f}只/s  ETA {eta:.0f}s",
                flush=True,
            )
            if status == "error":
                print(f"    ✗ {res['msg'][:80]}")

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"  完成: ok={ok_n}  empty={empty_n}  error={err_n}")
    print(f"  耗时: {elapsed:.0f}s  平均: {len(codes)/max(elapsed,1):.1f}只/s")
    print("=" * 60)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 0b: AKShare 补全 BaoStock 封禁后缺失的日线 parquet")
    parser.add_argument("--start",     default="2015-01-05",
                        help="下载起始日（默认 2015-01-05）")
    parser.add_argument("--end",       default=None,
                        help="下载截止日（默认今日）")
    parser.add_argument("--workers",   type=int, default=4,
                        help="并发线程数（默认 4，建议不超过 6）")
    parser.add_argument("--delay",     type=float, default=0.8,
                        help="每只下载后的基础间隔秒数（默认 0.8）")
    parser.add_argument("--force",     action="store_true",
                        help="强制重下全部股票（忽略增量检查）")
    parser.add_argument("--scan-only", action="store_true",
                        help="只扫描缺失情况，不实际下载")
    parser.add_argument("--codes",     nargs="+", default=None,
                        help="指定股票代码（默认从 meta.json 加载全市场）")
    args = parser.parse_args()

    # 依赖检查
    try:
        import akshare as ak
        print(f"✓ AKShare 版本: {ak.__version__}")
    except ImportError:
        print("✗ 未安装 akshare: pip install akshare -U")
        sys.exit(1)

    end_date = args.end or date.today().strftime("%Y-%m-%d")
    parquet_dir = _load_parquet_dir()

    # 加载股票列表
    all_codes = args.codes if args.codes else _load_codes()
    all_codes = [_to_full_code(str(c)) for c in all_codes]  # 统一为 sh.XXXXXX 格式
    print(f"✓ 目标股票: {len(all_codes)} 只")
    print(f"✓ parquet 目录: {parquet_dir}")

    # 扫描缺失
    print(f"\n  扫描缺失/空/过旧 parquet 文件...")
    missing = _scan_missing(parquet_dir, all_codes, args.start, args.force)

    # 统计情况
    existing  = len(all_codes) - len(missing)
    print(f"  已有且最新: {existing} 只")
    print(f"  需要补全:   {len(missing)} 只")

    if args.scan_only:
        if missing:
            print("\n  缺失股票列表（前50）:")
            for c in missing[:50]:
                print(f"    {c}")
            if len(missing) > 50:
                print(f"    ... 共 {len(missing)} 只")
        sys.exit(0)

    if not missing:
        print("\n✓ 所有股票 parquet 均已完整，无需补全")
        sys.exit(0)

    print(f"\n  ⚠ 发现 {len(missing)} 只需要补全（来源: AKShare 东方财富）")
    print(f"    并发={args.workers}线程，delay={args.delay}s/只")
    print(f"    预估耗时: {len(missing)*(args.delay+0.8)/args.workers/60:.0f}~"
          f"{len(missing)*(args.delay+2)/args.workers/60:.0f} 分钟")

    run(
        codes=missing,
        start=args.start,
        end=end_date,
        parquet_dir=parquet_dir,
        n_workers=args.workers,
        delay=args.delay,
        force=args.force,
    )

    # 补全后重新扫描，报告残余缺失
    still_missing = _scan_missing(parquet_dir, missing, args.start, False)
    if still_missing:
        print(f"\n⚠ 仍有 {len(still_missing)} 只未成功补全:")
        for c in still_missing[:30]:
            print(f"   {c}")
        if len(still_missing) > 30:
            print(f"   ... 共 {len(still_missing)} 只")
        print("  建议等待 30 分钟后重试，或手动检查这些股票")
    else:
        print(f"\n✓ 全部 {len(missing)} 只已成功补全")
