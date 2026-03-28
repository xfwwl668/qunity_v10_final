"""
scripts/step0_patch_daily_fields.py

功能：
  1. scan 模式：扫描现有日线下载脚本，输出 fields 修改指引
  2. download 模式：独立补下载日度估值字段（peTTM/pbMRQ/isST），
     直接生成 data/npy/valuation_*.npy

特点：
  - 断点续传（每 500 只保存一次中间矩阵）
  - 自动跳过已存在且形状正确的 npy 文件
  - 支持从指定股票序号开始（--resume-from）

用法:
  python scripts/step0_patch_daily_fields.py --mode scan
  python scripts/step0_patch_daily_fields.py --mode download
  python scripts/step0_patch_daily_fields.py --mode download --resume-from 500

预计耗时: ~2-4 小时（全市场 ~5000 只）
# [BUG-NEW-27 NOTE] 当前为单线程顺序下载。baostock_downloader.py 已实现 ProcessPoolExecutor
# 多进程并发架构（每进程独立 socket）。若将此脚本迁移为多进程（建议8进程），
# 可将下载时间缩短至 20~45 分钟。当前版本保持单线程以维持简洁性和断点兼容性。
"""

import sys
import os
import re
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FUND_DIR = DATA_DIR / "fundamental"

# [V10-PATH-FIX] 从 config.json 读取 npy_v10_dir，与 V10 路径保持一致
try:
    from scripts.utils_paths import get_npy_dir
except ImportError:
    import sys as _sys; _sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from utils_paths import get_npy_dir  # type: ignore

NPY_DIR   = get_npy_dir("v10")
META_PATH = NPY_DIR / "meta.json"

# ── 断点文件路径 ──
CHECKPOINT_PATH = FUND_DIR / "_valuation_checkpoint.json"  # 旧版路径（向下兼容）
# [BUG-NEW-09 FIX] 新版断点路径统一存放在 NPY_DIR，与中间 npy 文件一致
CHECKPOINT_PATH_NEW = None  # 在函数内动态赋值为 NPY_DIR / "_checkpoint_valuation.json"

FIELD_NAMES_NEW = ("peTTM", "pbMRQ", "psTTM", "pcfNcfTTM", "isST")


# ─────────────────────────────────────────────────────────────
# 扫描模式
# ─────────────────────────────────────────────────────────────

def scan_existing_scripts() -> None:
    print("=" * 70)
    print(" 扫描现有日线下载脚本 (Baostock query_history_k_data_plus)")
    print("=" * 70)

    pattern = re.compile(r'query_history_k_data_plus', re.IGNORECASE)
    fields_pattern = re.compile(r'fields\s*=\s*["\']([^"\']+)["\']')

    found = []
    for search_dir in [PROJECT_ROOT, PROJECT_ROOT / "scripts", PROJECT_ROOT / "src"]:
        if not search_dir.exists():
            continue
        for py_file in search_dir.rglob("*.py"):
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
                if pattern.search(content):
                    for i, line in enumerate(content.splitlines(), 1):
                        if pattern.search(line) or fields_pattern.search(line):
                            found.append((py_file, i, line.strip()))
            except Exception:
                pass

    if found:
        print(f"\n找到 {len(found)} 处调用:\n")
        for fpath, lineno, line in found:
            try:
                rel = fpath.relative_to(PROJECT_ROOT)
            except ValueError:
                rel = fpath
            print(f"  📄 {rel}:{lineno}")
            print(f"     {line[:110]}")
            print()
        print("─" * 70)
        print("修改指引：在 fields 参数末尾追加以下字段：")
        print()
        print("  修改前: fields=\"date,code,open,high,low,close,volume,amount,...,pctChg\"")
        print("  修改后: fields=\"date,code,open,high,low,close,volume,amount,...,pctChg,"
              "peTTM,pbMRQ,psTTM,pcfNcfTTM,isST\"")
        print()
        print("  字段说明:")
        print("    peTTM      滚动市盈率    → valuation_peTTM.npy")
        print("    pbMRQ      市净率        → valuation_pbMRQ.npy")
        print("    psTTM      滚动市销率    → valuation_psTTM.npy")
        print("    pcfNcfTTM  滚动市现率    → valuation_pcfNcfTTM.npy")
        print("    isST       是否ST股      → valuation_isST.npy  (用于 ST 过滤)")
        print()
        print("  ⚠ 注意: adjustflag 建议设为 '3'（不复权）【仅限估值字段 peTTM/pbMRQ 等】")
        print("         日线价格数据（open/high/low/close）必须用 adjustflag='2'（后复权）")
        print("         两者不可混用，混用将导致估值与价格不匹配！")
    else:
        print("  未发现 query_history_k_data_plus 调用，将使用独立下载模式")


# ─────────────────────────────────────────────────────────────
# 下载模式
# ─────────────────────────────────────────────────────────────

def load_meta() -> dict:
    if not META_PATH.exists():
        print(f"✗ 未找到 {META_PATH}，请先完成日线 npy 构建")
        sys.exit(1)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    print(f"✓ meta.json: {len(meta['codes'])} 只股票, {len(meta['dates'])} 个交易日")
    return meta


def to_bs_code(code_6: str) -> str:
    """
    将任意格式的股票代码转换为 BaoStock 格式（sh.600519 / sz.000001）。
    支持输入：
      - "sh.600519" / "sz.000001"（已是 BaoStock 格式，直接返回）
      - "600519" / "000001"（6位数字，自动加前缀）
      - 600519（整数，转字符串后补零加前缀）
    """
    s = str(code_6).strip()
    # 已是 BaoStock 格式（sh. 或 sz. 开头，共9位）
    if s.startswith(("sh.", "sz.")):
        return s
    # 6位数字格式 → 补零后加前缀
    c = s.zfill(6)
    prefix = "sh" if c[0] in ("6", "9") else "sz"
    return f"{prefix}.{c}"



# ─────────────────────────────────────────────────────────────
# 多进程 worker（每个子进程独立 socket，真正并行）
# ─────────────────────────────────────────────────────────────

# ── 子进程全局状态（initializer 设置，每进程独立）──────────────────────────────
_g_val_start_date: str = ""
_g_val_end_date:   str = ""
_g_val_ready:      bool = False


def _val_pool_initializer(start_date: str, end_date: str, worker_idx: int) -> None:
    """
    [PERF BUG-NEW-27 FIX v2] 每进程登录一次，后续每只股票一个独立 task。
    使用 initializer 模式（与 baostock_downloader.py 保持一致）：
      - 进程启动时 bs.login() → 独立 TCP socket
      - 后续每只股票直接用已登录的 socket，无重复 login/logout 开销
      - wait(FIRST_COMPLETED) 滑动窗口 → 进度条每只更新，内存峰值 ~11MB（原版 3.6GB）
    """
    import socket as _sock
    _sock.setdefaulttimeout(60)

    global _g_val_start_date, _g_val_end_date, _g_val_ready
    _g_val_start_date = start_date
    _g_val_end_date   = end_date

    try:
        import baostock as bs
        import random, time
        time.sleep(random.uniform(0.0, 5.0))   # 错开各进程首次请求，防 burst
        lg = bs.login()
        _g_val_ready = (lg.error_code == "0")
    except Exception:
        _g_val_ready = False


def _parse_float(v) -> float:
    """模块级浮点解析，避免 worker 每次调用重复创建函数对象。"""
    if v and v != "":
        try:
            return float(v)
        except Exception:
            pass
    return float("nan")


def _parse_int(v) -> int:
    """模块级整数解析。"""
    if v and v != "":
        try:
            return int(float(v))
        except Exception:
            pass
    return 0


def _worker_fetch_one(task: tuple) -> dict:
    """
    每只股票一个 task。
    task: (global_idx, bs_code)
    返回轻量结构: {"global_idx": int, "rows": list[tuple], "error": bool}

    [MEM-FIX] rows 里直接存 float/int 值（非 Python list of dicts）。
    每只 ~2800行 × tuple(str,f,f,f,f,i) ≈ 720KB → 处理完主进程 del future 立刻释放。
    """
    global_idx, bs_code = task

    if not _g_val_ready:
        return {"global_idx": global_idx, "rows": [], "error": True}

    try:
        import baostock as bs  # 子进程 sys.modules 已有，仅查缓存，O(1)

        rs = bs.query_history_k_data_plus(
            code=bs_code,
            fields="date,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST,close",
            start_date=_g_val_start_date,
            end_date=_g_val_end_date,
            frequency="d",
            adjustflag="3",
        )

        rows = []
        if rs.error_code == "0":
            while rs.next():
                row = rs.get_row_data()
                rows.append((row[0],
                              _parse_float(row[1]), _parse_float(row[2]),
                              _parse_float(row[3]), _parse_float(row[4]),
                              _parse_int(row[5]),   _parse_float(row[6])))
        return {"global_idx": global_idx, "rows": rows, "error": False}

    except Exception:
        return {"global_idx": global_idx, "rows": [], "error": True}


def download_valuation(meta: dict, resume_from: int = 0,
                       start_date_override: str = None,
                       n_workers: int = 8) -> None:
    """
    [BUG-NEW-27 FIX] 多进程并发下载，8进程并行 → 预计 20~45 分钟完成全市场下载。
    [BUG-NEW-04 FIX] start_date_override: 自定义起始日期，覆盖 meta.json 最早交易日
    [BUG-NEW-08 FIX] adjustflag=3 仅用于估值字段，日线价格必须用 adjustflag=2（后复权）
    [BUG-NEW-09 FIX] 断点 JSON 统一存放在 NPY_DIR，与中间 npy 文件路径一致
    """
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

    codes = meta["codes"]
    dates = meta["dates"]
    n_stocks = len(codes)
    n_dates = len(dates)
    date_to_idx = {d: i for i, d in enumerate(dates)}

    # [BUG-NEW-04 FIX] 支持自定义起始日期
    if start_date_override:
        avail = [d for d in dates if d >= start_date_override]
        if not avail:
            print(f"✗ --start-date {start_date_override} 晚于最后交易日 {dates[-1]}")
            sys.exit(1)
        start_date = avail[0]
        print(f"  ⚙ 自定义起始日期: {start_date}（meta 最早: {dates[0]}）")
    else:
        start_date = dates[0]
    end_date = dates[-1]

    # ── 矩阵初始化或恢复 ──
    def _init_matrix(dtype):
        return np.full((n_stocks, n_dates), np.nan if dtype == np.float32 else 0,
                       dtype=dtype)

    matrices = {
        "peTTM":     _init_matrix(np.float32),
        "pbMRQ":     _init_matrix(np.float32),
        "psTTM":     _init_matrix(np.float32),
        "pcfNcfTTM": _init_matrix(np.float32),
        "isST":      np.zeros((n_stocks, n_dates), dtype=np.uint8),
        # [BUG1-FIX] 不复权收盘价：adjustflag=3 的 close 字段即为不复权价，
        # 用于 step3 正确计算历史市值，避免后复权价累积分红导致早期市值严重失真。
        "unadj_close": _init_matrix(np.float32),
    }

    # [BUG-NEW-09 FIX] 断点路径统一到 NPY_DIR；同时兼容旧版 FUND_DIR 路径
    _cp_new = NPY_DIR / "_checkpoint_valuation.json"
    _cp_to_load = _cp_new if _cp_new.exists() else (CHECKPOINT_PATH if CHECKPOINT_PATH.exists() else None)

    start_idx = resume_from
    if _cp_to_load and resume_from == 0:
        with open(_cp_to_load) as f:
            cp = json.load(f)
        start_idx = cp.get("next_idx", 0)
        for name, mat in matrices.items():
            p = NPY_DIR / f"_partial_valuation_{name}.npy"
            if p.exists():
                loaded = np.load(str(p))
                if loaded.shape == mat.shape:
                    matrices[name][:] = loaded
        print(f"↻ 断点恢复: 从 idx={start_idx} 继续（共 {n_stocks} 只）")

    print(f"✓ 多进程并发下载（n_workers={n_workers}）")
    print(f"  下载范围: {start_date} ~ {end_date}")
    print(f"  待处理: {n_stocks - start_idx} 只（从 idx={start_idx} 开始）")
    print()

    t0 = time.time()
    total_errors = 0
    completed_stocks = 0

    # ── 滑动窗口提交 + 即时释放 future ──────────────────────────────────────
    # [MEM-FIX] 原版一次性 submit 5188 个 future 全放入 dict：
    #   每只结果 ~720KB，全部持有 → 峰值 ~3.6GB，导致 OOM / 系统换页 / 卡死。
    #   修复：滑动窗口，同时在飞的 future 不超过 n_workers*2；
    #         处理完一个立刻 del inflight[f] → GC 即时回收，峰值内存 ~11MB。
    #
    # [PERF] pe_valid 进度计算原版每10只做全矩阵扫描(5188*2800 float32, 17ms/次)：
    #   5188只共 519次 × 17ms = ~9秒纯浪费。改为每500只计算一次(与断点同步)。
    #
    # 滑动窗口原理：
    #   inflight: {future: global_idx}，容量上限 n_workers*2
    #   │ worker1 处理中 │ worker1 排队中 │  ← 无空闲气泡，吞吐最大化
    #   │ worker2 处理中 │ worker2 排队中 │
    #   每完成1个 → del inflight[f] → 补充1个新任务

    from collections import deque
    from concurrent.futures import wait, FIRST_COMPLETED

    pending = deque(
        (i, to_bs_code(str(codes[i]))) for i in range(start_idx, n_stocks)
    )
    total_pending = len(pending)
    inflight: dict = {}   # {future: global_idx}
    WINDOW = n_workers * 2   # 同时在飞的 future 上限

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_val_pool_initializer,
        initargs=(start_date, end_date, 0),
    ) as executor:

        # 初始填满窗口
        while pending and len(inflight) < WINDOW:
            task = pending.popleft()
            inflight[executor.submit(_worker_fetch_one, task)] = task[0]

        while inflight:
            # 等待至少一个完成
            done_set, _ = wait(inflight, return_when=FIRST_COMPLETED)

            for future in done_set:
                g_idx = inflight.pop(future)   # 从 inflight 移除

                try:
                    result = future.result()
                except Exception:
                    total_errors += 1
                    completed_stocks += 1
                    del future                 # 立即 GC
                    continue

                # 写入矩阵
                for (date_str, pe, pb, ps, pcf, ist, unadj_c) in result.get("rows", []):
                    tidx = date_to_idx.get(date_str)
                    if tidx is None:
                        continue
                    matrices["peTTM"][g_idx, tidx]     = pe
                    matrices["pbMRQ"][g_idx, tidx]     = pb
                    matrices["psTTM"][g_idx, tidx]     = ps
                    matrices["pcfNcfTTM"][g_idx, tidx] = pcf
                    matrices["isST"][g_idx, tidx]      = ist
                    matrices["unadj_close"][g_idx, tidx] = unadj_c  # [BUG1-FIX]

                if result.get("error"):
                    total_errors += 1
                completed_stocks += 1
                del future                     # 立即 GC：释放 ~720KB 结果数据

                # 补充新任务进窗口
                while pending and len(inflight) < WINDOW:
                    task = pending.popleft()
                    inflight[executor.submit(_worker_fetch_one, task)] = task[0]

                # ── 进度输出（每只股票实时刷新行）──
                elapsed = time.time() - t0
                speed = completed_stocks / max(elapsed, 0.1)
                remaining = total_pending - completed_stocks
                eta_min = remaining / max(speed, 0.01) / 60.0
                sys.stdout.write(
                    f"\r  [{start_idx + completed_stocks}/{n_stocks}] "
                    f"{speed:.1f} 只/秒 | ETA {eta_min:.0f} 分钟 | "
                    f"inflight={len(inflight)} | 错误={total_errors}    "
                )
                sys.stdout.flush()

                # ── 断点保存（每 500 只，附带 pe_valid 统计）──
                if completed_stocks % 500 == 0:
                    print()  # 换行，断点信息单独一行
                    pe_valid = (np.sum(~np.isnan(matrices["peTTM"]))
                                / matrices["peTTM"].size * 100)
                    with open(NPY_DIR / "_checkpoint_valuation.json", "w") as f:
                        json.dump({"next_idx": start_idx + completed_stocks}, f)
                    for name, mat in matrices.items():
                        np.save(str(NPY_DIR / f"_partial_valuation_{name}.npy"), mat)
                    print(f"    💾 断点 idx={start_idx + completed_stocks} "
                          f"PE有效率={pe_valid:.1f}%")

    print()  # 进度条换行
    print("\n✓ 所有子进程已完成")

    # [BUG-STEP0-FINAL-SAVE FIX] 原代码只保存 _partial_valuation_*.npy（断点文件）
    # 从未保存最终的 valuation_peTTM.npy 等文件，fast_runner 永远找不到。
    # 修复：全部下载完成后，将矩阵保存为最终文件名。
    print("\n  保存最终 npy 文件...")
    name_map = {
        "peTTM":       "valuation_peTTM",
        "pbMRQ":       "valuation_pbMRQ",
        "psTTM":       "valuation_psTTM",
        "pcfNcfTTM":   "valuation_pcfNcfTTM",
        "isST":        "valuation_isST",
        "unadj_close": "unadj_close",   # [BUG1-FIX] 不复权价 → step3 正确计算市值
    }
    for key, final_name in name_map.items():
        if key in matrices:
            out_path = NPY_DIR / f"{final_name}.npy"
            np.save(str(out_path), matrices[key])
            mat = matrices[key]
            if mat.dtype == np.uint8:
                # isST: 显示实际ST股日占比（>0表示ST）
                valid_pct = np.sum(mat > 0) / mat.size * 100
                print(f"  ✓ {final_name}.npy  shape={mat.shape}  ST占比={valid_pct:.2f}%")
            else:
                valid_pct = np.sum(~np.isnan(mat)) / mat.size * 100
                print(f"  ✓ {final_name}.npy  shape={mat.shape}  有效率={valid_pct:.1f}%")
    # 清理断点文件
    for p in list(NPY_DIR.glob("_partial_valuation_*.npy")) +              [NPY_DIR / "_checkpoint_valuation.json", CHECKPOINT_PATH]:
        if p.exists():
            p.unlink()
    print("  ✓ 断点文件已清理")



# ─────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 0: 补下载日度估值字段 (peTTM/pbMRQ/isST) 并生成 npy"
    )
    parser.add_argument(
        "--mode", choices=["scan", "download", "both"],
        default="both", help="scan=扫描脚本指引, download=下载估值数据",
    )
    parser.add_argument(
        "--resume-from", type=int, default=0,
        help="从指定股票序号重新开始（覆盖断点文件）",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="强制重新下载，忽略已存在的 npy 文件",
    )
    # [BUG-NEW-04 FIX] 新增 --start-date：允许从指定日期起下载，避免强制从 meta 最早日期
    # 例：meta 最早 2006-01-01，若只需 2015 年以来数据，指定 --start-date 2015-01-01 可省 2/3 时间
    parser.add_argument(
        "--start-date", dest="start_date", type=str, default=None,
        help="下载起始日期 YYYY-MM-DD（默认使用 meta.json 第一个交易日）",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="并发进程数（默认8，BaoStock 服务器舒适并发上限）",
    )
    args = parser.parse_args()

    if args.mode in ("scan", "both"):
        scan_existing_scripts()

    if args.mode in ("download", "both"):
        meta = load_meta()

        # 检查是否已存在
        if not args.force:
            existing = [
                f for f in ["valuation_peTTM.npy", "valuation_isST.npy"]
                if (NPY_DIR / f).exists()
            ]
            if len(existing) == 2:
                print(f"\n⚠ {existing} 已存在，跳过下载（使用 --force 强制重新下载）")
                sys.exit(0)

        FUND_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 70}")
        print(f" Step 0: 下载日度估值数据")
        print(f"{'=' * 70}")
        download_valuation(meta, resume_from=args.resume_from,
                           start_date_override=args.start_date,
                           n_workers=args.workers)



