"""
audit_adj_types.py — Q-UNITY V9.1 复权类型批量审计

V9.1 修复：
  - 按板块动态阈值：创业板/科创板 20%，主板 10%，ST 5%，北交所 30%
  - 识别 ST 状态（文件名含 ST 或 parquet 中 isST 字段）
  - 审计报告新增 board 列，方便筛查

用法：
  python -m src.data.audit_adj_types --parquet-dir data/daily_parquet
  python -m src.data.audit_adj_types --parquet-dir data/daily_parquet --workers 16
"""
from __future__ import annotations

import argparse
import csv
import datetime
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.adj_detector import detect_ex_rights, validate_adj_type, _board_name


def _is_st_from_df(df: pd.DataFrame) -> bool:
    """从 DataFrame 判断是否为ST（尝试 isST / is_st 列）"""
    for col in ("isST", "is_st", "isSt"):
        if col in df.columns:
            try:
                val = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(val) > 0 and val.iloc[-1] > 0:
                    return True
            except Exception:
                pass
    return False


def audit_one(pq_path: Path) -> dict:
    """审计单只股票的复权类型（V9.1: 按板块动态阈值）"""
    code = pq_path.stem  # e.g. "600519" or "sz.300750"

    try:
        import pyarrow.parquet as _pq
        pq_file = _pq.ParquetFile(str(pq_path))
        available_cols = pq_file.schema_arrow.names

        read_cols = [c for c in ["date", "close", "volume", "isST", "is_st"]
                     if c in available_cols]
        if "close" not in read_cols:
            return {"code": code, "board": _board_name(code),
                    "status": "empty", "reason": "缺少 close 列"}

        df = pd.read_parquet(pq_path, columns=read_cols)
        if df.empty:
            return {"code": code, "board": _board_name(code),
                    "status": "empty", "reason": "无数据行"}

        close = pd.to_numeric(df["close"], errors="coerce").values
        dates = df["date"].tolist() if "date" in df.columns else list(range(len(df)))
        volume = (pd.to_numeric(df["volume"], errors="coerce").values
                  if "volume" in df.columns else None)

        is_st = _is_st_from_df(df)
        board = _board_name(code)

        # V9.1: 传入 code 和 is_st，自动按板块选阈值
        ex_dates = detect_ex_rights(close, dates, code=code, is_st=is_st,
                                    volume=volume)
        is_hfq, reason = validate_adj_type(close, dates, code=code,
                                           is_st=is_st,
                                           ex_rights_dates=ex_dates)

        # 声明的复权类型（若 parquet 有此字段）
        declared = "unknown"
        if "adj_type" in available_cols:
            try:
                df2 = pd.read_parquet(pq_path, columns=["adj_type"])
                declared = str(df2["adj_type"].iloc[0]) if not df2.empty else "unknown"
            except Exception:
                pass

        return {
            "code":               code,
            "board":              board,
            "status":             "hfq" if is_hfq else "qfq_suspect",
            "is_st":              int(is_st),
            "declared_adj_type":  declared,
            "ex_dates_count":     len(ex_dates),
            "ex_dates_sample":    str(ex_dates[:3]),
            "reason":             reason,
            "n_rows":             len(df),
        }

    except Exception as e:
        return {"code": code, "board": _board_name(code),
                "status": "error", "reason": str(e),
                "is_st": 0, "declared_adj_type": "unknown",
                "ex_dates_count": 0, "ex_dates_sample": "",
                "n_rows": 0}


def audit_all(
    parquet_dir: str,
    workers: int = 8,
    output_dir: str = "data/audit",
) -> dict:
    """批量扫描并返回 {csv_path, hfq_count, qfq_suspect_count, error_count}"""
    pq_dir = Path(parquet_dir)
    pq_files = sorted(pq_dir.glob("*.parquet"))
    # 过滤 stock_list.parquet 等非股票文件
    pq_files = [f for f in pq_files if f.stem not in ("stock_list",)]

    if not pq_files:
        print(f"[Audit] 无 .parquet 文件: {pq_dir}")
        return {"csv_path": "", "hfq_count": 0,
                "qfq_suspect_count": 0, "error_count": 0}

    total = len(pq_files)
    print(f"[Audit] 开始扫描 {total} 只股票，并发={workers}")
    print(f"[Audit] V9.1 动态阈值：主板10% / 创业板科创板20% / ST主板5% / 北交所30%")

    results = []
    t0 = datetime.datetime.now()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(audit_one, p): p for p in pq_files}
        done = 0
        for future in as_completed(futures):
            results.append(future.result())
            done += 1
            if done % 200 == 0 or done == total:
                elapsed = (datetime.datetime.now() - t0).total_seconds()
                print(f"[Audit] 进度 {done}/{total} | 耗时 {elapsed:.1f}s")

    elapsed_total = (datetime.datetime.now() - t0).total_seconds()

    # 按板块统计
    from collections import Counter
    board_counts = Counter(r.get("board", "未知") for r in results
                           if r.get("status") == "qfq_suspect")

    n_hfq    = sum(1 for r in results if r["status"] == "hfq")
    n_susp   = sum(1 for r in results if r["status"] == "qfq_suspect")
    n_err    = sum(1 for r in results if r["status"] == "error")

    print(f"\n{'='*55}")
    print(f"扫描完成: {total} 只股票  耗时 {elapsed_total:.1f}s")
    print(f"  ✓ hfq (后复权):     {n_hfq}")
    print(f"  ⚠ qfq_suspect:      {n_susp}")
    if board_counts:
        for board, cnt in board_counts.most_common():
            print(f"      {board}: {cnt} 只")
    print(f"  ✗ error:            {n_err}")
    print(f"{'='*55}\n")

    # 写出 CSV
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"adj_audit_{ts}.csv"

    fieldnames = ["code", "board", "status", "is_st",
                  "declared_adj_type", "ex_dates_count",
                  "ex_dates_sample", "reason", "n_rows"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(results, key=lambda x: (x.get("status", ""), x.get("code", ""))):
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"[Audit] 报告已写出: {csv_path}")
    return {
        "csv_path":          str(csv_path),
        "hfq_count":         n_hfq,
        "qfq_suspect_count": n_susp,
        "error_count":       n_err,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q-UNITY V9.1 复权类型批量审计")
    parser.add_argument("--parquet-dir", default="data/daily_parquet")
    parser.add_argument("--workers",     type=int, default=8)
    parser.add_argument("--output-dir",  default="data/audit")
    args = parser.parse_args()
    audit_all(args.parquet_dir, args.workers, args.output_dir)
