"""
scripts/step0d_download_daily_akshare.py

AKShare 专用日线下载器（东财节点，独立备用）

────────────────────────────────────────────────────────────
定位
────────────────────────────────────────────────────────────
本脚本是一个干净、独立的 AKShare 日线下载器，与 step0b（fill_missing）
分离，专注于全量/增量日线下载。

数据源矩阵：
  step0c  → adata（主力，巨潮节点）+ AKShare（备用）
  step0d  → AKShare 独立（东财节点，step0c 的完全替代）
  step0b  → AKShare 补丁（仅填缺失文件，维护用）

什么时候用 step0d：
  · adata 被封/不可用，只有 AKShare 能下
  · 与 step0c 数据做双源交叉验证
  · 快速补充某些 adata 下载失败的个股

────────────────────────────────────────────────────────────
并发说明
────────────────────────────────────────────────────────────
AKShare 是标准 HTTP 请求（东方财富接口），I/O 密集，ThreadPoolExecutor 最优。
BaoStock 需要 ProcessPool（单例 socket），AKShare 不需要。

建议 --workers 6~8（东财有轻度限速，UA 轮换后可到 10）。

────────────────────────────────────────────────────────────
用法
────────────────────────────────────────────────────────────
  python scripts/step0d_download_daily_akshare.py --test
  python scripts/step0d_download_daily_akshare.py --workers 8
  python scripts/step0d_download_daily_akshare.py --workers 8 --incremental
  python scripts/step0d_download_daily_akshare.py --codes 600519 000001
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
PROJECT_ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR            = PROJECT_ROOT / "data"
try:
    from scripts.utils_paths import get_npy_dir as _gnpy, get_parquet_dir as _gpq
except ImportError:
    import sys as _sys; _sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from utils_paths import get_npy_dir as _gnpy, get_parquet_dir as _gpq  # type: ignore

NPY_DIR             = _gnpy("v10")
META_PATH           = NPY_DIR / "meta.json"
DEFAULT_PARQUET_DIR = _gpq("qfq")

# ─────────────────────────────────────────────────────────────────────────────
# 工具函数（与 step0b/0c 保持一致）
# ─────────────────────────────────────────────────────────────────────────────

def _to_full_code(code: str) -> str:
    s = str(code).strip()
    if s.startswith(("sh.", "sz.")):
        return s
    c = s.split(".")[-1].zfill(6)
    return f"{'sh' if c[0] in ('6','9') else 'sz'}.{c}"


def _to_6digit(code: str) -> str:
    return str(code).strip().split(".")[-1].zfill(6)


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


def _load_codes() -> list[str]:
    if META_PATH.exists():
        with open(META_PATH, encoding="utf-8") as f:
            return [_to_full_code(str(c)) for c in json.load(f)["codes"]]
    parquet_dir = _load_parquet_dir()
    if parquet_dir.exists():
        codes = [_to_full_code(p.stem) for p in sorted(parquet_dir.glob("*.parquet"))
                 if p.stem not in ("stock_list",)]
        if codes:
            return codes
    # AKShare 获取股票列表
    try:
        import akshare as ak
        df = ak.stock_info_a_code_name()
        col = next((c for c in ["code","股票代码"] if c in df.columns), df.columns[0])
        return [_to_full_code(str(c)) for c in df[col].dropna()]
    except Exception as e:
        print(f"✗ 无法获取股票列表: {e}")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# AKShare UA 轮换（防封禁）
# ─────────────────────────────────────────────────────────────────────────────

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
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
    kws = ["429", "限流", "频繁", "too many", "rate limit", "blocked", "访问过于", "403"]
    return any(k in msg.lower() for k in kws)


def _backoff(attempt: int, is_limit: bool) -> float:
    return (6.0 * (attempt + 1) + random.uniform(0, 3)) if is_limit \
           else (2.0 ** attempt + random.uniform(0, 1))


# ─────────────────────────────────────────────────────────────────────────────
# 扫描缺失文件
# ─────────────────────────────────────────────────────────────────────────────

def _scan_missing(parquet_dir: Path, codes: list[str], force: bool) -> list[str]:
    """返回需要（重）下载的股票代码列表（sh.600519 格式）。"""
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
# 单只下载
# ─────────────────────────────────────────────────────────────────────────────

def _download_one(code: str, start: str, end: str,
                  parquet_dir: Path, delay: float) -> dict:
    """
    AKShare stock_zh_a_hist 下载后复权日线。

    volume 单位处理：
    - AKShare 返回「手」→ ×100 = 股（BaoStock 格式）
    - 与 BaoStock / adata 输出统一为「股」
    """
    global _consecutive_failures

    code_6 = _to_6digit(code)

    with _lock:
        extra = min(_consecutive_failures * 0.4, 10.0)
    if extra > 0:
        time.sleep(extra)

    time.sleep(random.uniform(0.1, 1.5))

    import akshare as ak
    last_err = ""

    for attempt in range(4):
        try:
            _patch_ak_ua()
            df = ak.stock_zh_a_hist(
                symbol=code_6, period="daily",
                start_date=start.replace("-", ""),
                end_date=end.replace("-", ""),
                adjust="qfq",  # V10: 前复权 QFQ
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
                        rename[c] = target; break
            df = df.rename(columns=rename)

            miss = [c for c in ["date","open","high","low","close"] if c not in df.columns]
            if miss:
                return {"code": code, "status": "error", "rows": 0,
                        "msg": f"缺少必要列: {miss}  可用: {df.columns.tolist()}"}

            # ── 清洗 ───────────────────────────────────────────────────────
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            df = df.dropna(subset=["date", "close"])
            for col in ["open","high","low","close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
            df = df[df["close"] > 0]

            # volume：保持「手」单位，fast_runner vol_multiplier=100 统一换算
            if "volume" in df.columns:
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).astype("float32")
            else:
                df["volume"] = np.float32(0)

            if "amount" in df.columns:
                df["amount"] = (pd.to_numeric(df["amount"], errors="coerce")
                                .fillna(0.0)).astype("float32")
            else:
                df["amount"] = np.float32(0)

            df["code"]     = code  # sh.600519 完整格式
            df["adj_type"] = "qfq"   # V10 标记，供兼容性检查使用
            keep = [c for c in ["date","open","high","low","close",
                                 "volume","amount","code","adj_type"] if c in df.columns]
            df = df[keep].sort_values("date").reset_index(drop=True)

            if len(df) == 0:
                return {"code": code, "status": "empty", "rows": 0, "msg": "清洗后为空"}

            # ── 合并已有 parquet ───────────────────────────────────────────
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
            time.sleep(delay + random.uniform(0, delay * 0.4))
            return {"code": code, "status": "ok", "rows": len(df), "msg": ""}

        except Exception as e:
            last_err = str(e)
            is_limit = _is_rate_limit(last_err)
            if attempt < 3:
                time.sleep(_backoff(attempt, is_limit))

    with _lock:
        _consecutive_failures += 1
    return {"code": code, "status": "error", "rows": 0,
            "msg": f"4次重试失败: {last_err[:80]}"}


# ─────────────────────────────────────────────────────────────────────────────
# 批量下载
# ─────────────────────────────────────────────────────────────────────────────

def run(codes: list[str], start: str, end: str, parquet_dir: Path,
        n_workers: int, delay: float, force: bool) -> None:

    parquet_dir.mkdir(parents=True, exist_ok=True)
    total = len(codes)
    t0 = time.time()

    print(f"\n  ▶ AKShare 日线下载（{n_workers} 线程池，delay={delay}s，共 {total} 只）")
    print(f"    范围: {start} ~ {end}  |  输出: {parquet_dir}")

    ok_n = err_n = empty_n = 0

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
                res = {"code": code, "status": "error", "rows": 0, "msg": str(e)}

            st = res["status"]
            if st == "ok":       ok_n    += 1
            elif st == "error":  err_n   += 1
            else:                empty_n += 1

            elapsed = time.time() - t0
            speed   = done / max(elapsed, 0.1)
            eta     = (total - done) / max(speed, 0.01)
            print(
                f"  [{done:4d}/{total}] {code}  {st:5s}  {res['rows']:5d}行 | "
                f"ok={ok_n} err={err_n} | "
                f"{speed:.1f}只/s  ETA={eta:.0f}s",
                flush=True,
            )
            if st == "error":
                print(f"    ✗ {res['msg']}")

    elapsed = time.time() - t0
    print()
    print("=" * 68)
    print(f"  完成: ok={ok_n}  empty={empty_n}  error={err_n}")
    print(f"  耗时: {elapsed:.0f}s  平均: {total/max(elapsed,1):.1f}只/s")
    print("=" * 68)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 0d: AKShare 专用日线下载（东财节点，独立备用）")
    parser.add_argument("--start",       default="2015-01-05")
    parser.add_argument("--end",         default=None)
    parser.add_argument("--workers",     type=int,   default=8,
                        help="并发线程数（默认8，建议6~12，HTTP I/O密集用线程池）")
    parser.add_argument("--delay",       type=float, default=0.6,
                        help="每只间隔秒（默认0.6）")
    parser.add_argument("--force",       action="store_true")
    parser.add_argument("--incremental", action="store_true",
                        help="只下过旧（>5天）文件")
    parser.add_argument("--scan-only",   action="store_true",
                        help="只扫描，不下载")
    parser.add_argument("--test",        action="store_true",
                        help="测试模式（10只）")
    parser.add_argument("--codes",       nargs="+", default=None)
    args = parser.parse_args()

    try:
        import akshare as ak
        print(f"✓ AKShare 版本: {ak.__version__}")
    except ImportError:
        print("✗ AKShare 未安装: pip install akshare -U"); sys.exit(1)

    end_date    = args.end or date.today().strftime("%Y-%m-%d")
    parquet_dir = _load_parquet_dir()

    if args.test:
        all_codes = [_to_full_code(c) for c in
                     ["600519","601318","000001","000858","300750",
                      "600036","002594","600900","300014","000002"]]
        print(f"⚡ 测试模式: {len(all_codes)} 只")
    elif args.codes:
        all_codes = [_to_full_code(c) for c in args.codes]
    else:
        all_codes = _load_codes()

    print(f"✓ 目标: {len(all_codes)} 只  |  输出: {parquet_dir}")

    missing = _scan_missing(parquet_dir, all_codes, args.force)
    existing = len(all_codes) - len(missing)
    print(f"  已有最新: {existing} 只  |  需下载: {len(missing)} 只")

    if args.scan_only:
        if missing:
            print(f"\n  缺失前30: {missing[:30]}")
        sys.exit(0)

    if not missing:
        print("\n✓ 所有文件均已完整")
    else:
        est = len(missing) * (args.delay + 0.7) / args.workers / 60
        print(f"  预估耗时: ~{est:.0f} 分钟")
        run(missing, args.start, end_date, parquet_dir, args.workers, args.delay, args.force)

    print("\n✓ Step 0d 完成！后续：")
    print("  python -m src.data.build_npy")
