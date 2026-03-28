"""
scripts/step1_download_fundamental.py

通过 Baostock 下载全市场季度基本面数据：
  - 盈利能力: ROE, EPS_TTM, 净利润, 总股本, 流通股本, 净利率
  - 成长能力: 净利润同比, EPS同比

输出:
  data/fundamental/profit_quarterly.csv
  data/fundamental/growth_quarterly.csv
  data/fundamental/fundamental_merged.csv

[BUG-FUND-01/02 FIX] 原版为单线程顺序下载（3-5小时）。
baostock 使用模块级单例 TCP socket，多线程会帧错位崩溃。
改用 ProcessPoolExecutor（每子进程独立 socket），8 进程并发
可将下载时间缩短至 20~40 分钟。

用法:
  python scripts/step1_download_fundamental.py
  python scripts/step1_download_fundamental.py --start-year 2018
  python scripts/step1_download_fundamental.py --test   # 仅10只股票
  python scripts/step1_download_fundamental.py --force  # 强制重新下载
  python scripts/step1_download_fundamental.py --workers 8
"""

import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FUND_DIR = DATA_DIR / "fundamental"
META_PATH = DATA_DIR / "npy_v10" / "meta.json"  # [FIX-H-04]


def to_bs_code(code_6: str) -> str:
    s = str(code_6).strip()
    if s.startswith(("sh.", "sz.")):
        return s
    c = s.zfill(6)
    return f"{'sh' if c[0] in ('6', '9') else 'sz'}.{c}"


def to_6digit(bs_code: str) -> str:
    return bs_code.split(".")[-1].zfill(6)


def load_codes() -> list:
    if not META_PATH.exists():
        print(f"✗ 未找到 {META_PATH}")
        sys.exit(1)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    codes = [str(c).zfill(6) for c in meta["codes"]]
    print(f"✓ 加载 {len(codes)} 只股票")
    return codes


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    skip = {"code", "pubDate", "statDate"}
    df = df.copy()
    for col in df.columns:
        if col not in skip:
            df[col] = df[col].replace("", np.nan)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 多进程 worker 函数（必须在模块顶层定义，spawn 模式下可被子进程 import）
# [BUG-FUND-01/02 FIX] ProcessPoolExecutor：每进程独立 import baostock
# → 独立 BaoStockClient() → 独立 TCP socket，无竞态，无帧错位
# ─────────────────────────────────────────────────────────────────────────────

def _worker_profit_chunk(task: dict) -> dict:
    """子进程：下载一批股票的盈利数据（profit）。"""
    import baostock as bs

    codes_chunk = task["codes_chunk"]
    start_year  = task["start_year"]
    end_year    = task["end_year"]
    now_year    = task["now_year"]
    avail_q     = task["avail_q"]

    lg = bs.login()
    if lg.error_code != "0":
        return {"rows": [], "fields": None, "error_count": 1}

    all_rows = []
    field_names = None
    error_count = 0

    for code_6 in codes_chunk:
        bs_code = to_bs_code(code_6)
        for year in range(start_year, end_year + 1):
            for quarter in range(1, 5):
                if year == now_year and quarter > avail_q:
                    continue
                try:
                    rs = bs.query_profit_data(code=bs_code, year=year, quarter=quarter)
                    if field_names is None and rs.fields:
                        field_names = rs.fields
                    while rs.error_code == "0" and rs.next():
                        all_rows.append(rs.get_row_data())
                except Exception:
                    error_count += 1

    bs.logout()
    return {"rows": all_rows, "fields": field_names, "error_count": error_count}


def _worker_growth_chunk(task: dict) -> dict:
    """子进程：下载一批股票的成长数据（growth）。"""
    import baostock as bs

    codes_chunk = task["codes_chunk"]
    start_year  = task["start_year"]
    end_year    = task["end_year"]
    now_year    = task["now_year"]
    avail_q     = task["avail_q"]

    lg = bs.login()
    if lg.error_code != "0":
        return {"rows": [], "fields": None, "error_count": 1}

    all_rows = []
    field_names = None
    error_count = 0

    for code_6 in codes_chunk:
        bs_code = to_bs_code(code_6)
        for year in range(start_year, end_year + 1):
            for quarter in range(1, 5):
                if year == now_year and quarter > avail_q:
                    continue
                try:
                    rs = bs.query_growth_data(code=bs_code, year=year, quarter=quarter)
                    if field_names is None and rs.fields:
                        field_names = rs.fields
                    while rs.error_code == "0" and rs.next():
                        all_rows.append(rs.get_row_data())
                except Exception:
                    error_count += 1

    bs.logout()
    return {"rows": all_rows, "fields": field_names, "error_count": error_count}


# ─────────────────────────────────────────────────────────────────────────────
# [SPEED-OPT] 合并 worker：一次 login 同时下载 profit + growth
# 比分开下载快 ~40%（省去 growth 的全部 login/logout + 串行等待）
# ─────────────────────────────────────────────────────────────────────────────

def _worker_combined_chunk(task: dict) -> dict:
    """子进程：一次 login 同时下载 profit + growth，减少登录开销。"""
    import baostock as bs

    codes_chunk = task["codes_chunk"]
    start_year  = task["start_year"]
    end_year    = task["end_year"]
    now_year    = task["now_year"]
    avail_q     = task["avail_q"]

    lg = bs.login()
    if lg.error_code != "0":
        return {"profit_rows": [], "growth_rows": [],
                "profit_fields": None, "growth_fields": None, "error_count": 1}

    profit_rows, growth_rows = [], []
    profit_fields = growth_fields = None
    error_count = 0

    for code_6 in codes_chunk:
        bs_code = to_bs_code(code_6)
        for year in range(start_year, end_year + 1):
            for quarter in range(1, 5):
                if year == now_year and quarter > avail_q:
                    continue
                # profit
                try:
                    rs = bs.query_profit_data(code=bs_code, year=year, quarter=quarter)
                    if profit_fields is None and rs.fields:
                        profit_fields = rs.fields
                    while rs.error_code == "0" and rs.next():
                        profit_rows.append(rs.get_row_data())
                except Exception:
                    error_count += 1
                # growth（同一次 login，接着查）
                try:
                    rs = bs.query_growth_data(code=bs_code, year=year, quarter=quarter)
                    if growth_fields is None and rs.fields:
                        growth_fields = rs.fields
                    while rs.error_code == "0" and rs.next():
                        growth_rows.append(rs.get_row_data())
                except Exception:
                    error_count += 1

    bs.logout()
    return {
        "profit_rows":   profit_rows,
        "growth_rows":   growth_rows,
        "profit_fields": profit_fields,
        "growth_fields": growth_fields,
        "error_count":   error_count,
    }


def download_combined(codes: list, start_year: int, end_year: int,
                       n_workers: int = 16) -> tuple:
    """
    一次多进程同时下载 profit + growth，比分开下载快 ~40%。
    返回 (df_profit, df_growth)。
    """
    FUND_DIR.mkdir(parents=True, exist_ok=True)
    cp_path      = FUND_DIR / "_combined_checkpoint.json"
    partial_p_path = FUND_DIR / "_combined_profit_partial.parquet"
    partial_g_path = FUND_DIR / "_combined_growth_partial.parquet"

    now = datetime.now()
    avail_q = max(0, (now.month - 1 - 2) // 3)

    profit_rows, growth_rows = [], []
    profit_fields = growth_fields = None
    resume_from_idx = 0

    if cp_path.exists():
        with open(cp_path) as f:
            cp = json.load(f)
        resume_from_idx = cp.get("next_idx", 0)
        if partial_p_path.exists():
            dfp = pd.read_parquet(str(partial_p_path))
            profit_rows  = dfp.values.tolist()
            profit_fields = list(dfp.columns)
        if partial_g_path.exists():
            dfg = pd.read_parquet(str(partial_g_path))
            growth_rows  = dfg.values.tolist()
            growth_fields = list(dfg.columns)
        print(f"↻ 断点恢复: idx={resume_from_idx}, profit={len(profit_rows)}条, growth={len(growth_rows)}条")

    pending = codes[resume_from_idx:]
    total   = len(codes)

    if not pending:
        print("  combined 已全部完成")
    else:
        print(f"✓ combined 多进程下载（n_workers={n_workers}，profit+growth 合并）")
        print(f"  待处理: {len(pending)} 只（从 idx={resume_from_idx}）")

        chunk_size = 32
        chunks = [pending[s:s+chunk_size] for s in range(0, len(pending), chunk_size)]
        tasks = [{"codes_chunk": c, "start_year": start_year, "end_year": end_year,
                  "now_year": now.year, "avail_q": avail_q} for c in chunks]

        import time as _time
        t0 = _time.time()
        completed = 0
        total_errors = 0

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_worker_combined_chunk, task): task for task in tasks}
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    print(f"  ⚠ worker 异常: {e}")
                    continue

                profit_rows.extend(result.get("profit_rows", []))
                growth_rows.extend(result.get("growth_rows", []))
                if profit_fields is None and result.get("profit_fields"):
                    profit_fields = result["profit_fields"]
                if growth_fields is None and result.get("growth_fields"):
                    growth_fields = result["growth_fields"]
                total_errors += result.get("error_count", 0)
                chunk_len = len(futures[future]["codes_chunk"])
                completed += chunk_len

                elapsed = _time.time() - t0
                speed   = completed / max(elapsed, 0.1)
                eta_min = (len(pending) - completed) / max(speed, 0.01) / 60.0
                print(f"  [{resume_from_idx+completed}/{total}] "
                      f"profit={len(profit_rows)}条 growth={len(growth_rows)}条 | "
                      f"{speed:.1f}只/秒 | ETA {eta_min:.0f}m | 错误 {total_errors}")
                import sys; sys.stdout.flush()

                # 断点保存（每500只）
                if completed % 500 < chunk_len:
                    with open(cp_path, "w") as f:
                        json.dump({"next_idx": resume_from_idx + completed}, f)
                    if profit_fields and profit_rows:
                        pd.DataFrame(profit_rows, columns=profit_fields).to_parquet(
                            str(partial_p_path), index=False)
                    if growth_fields and growth_rows:
                        pd.DataFrame(growth_rows, columns=growth_fields).to_parquet(
                            str(partial_g_path), index=False)
                    print(f"    💾 断点 idx={resume_from_idx+completed}")

    # 清理断点
    for p in [cp_path, partial_p_path, partial_g_path]:
        if p.exists(): p.unlink()

    # 构建 DataFrame
    if profit_fields is None:
        profit_fields = ["code","pubDate","statDate","roeAvg","npMargin","gpMargin",
                         "netProfit","epsTTM","MBRevenue","totalShare","liqaShare"]
    if growth_fields is None:
        growth_fields = ["code","pubDate","statDate","YOYNI","YOYEPSBasic",
                         "YOYPNI","YOYROE","YOYAsset","YOYEquity"]

    df_p = pd.DataFrame(profit_rows, columns=profit_fields)
    df_g = pd.DataFrame(growth_rows, columns=growth_fields)

    for df, label in [(df_p, "profit"), (df_g, "growth")]:
        df = clean_df(df)
        df["code"] = df["code"].apply(to_6digit)
        # 保留最新更正报告
        df["pubDate"] = pd.to_datetime(df["pubDate"], errors="coerce")
        df = df.sort_values(["code","statDate","pubDate"])
        df = df.drop_duplicates(subset=["code","statDate"], keep="last")
        df = df.sort_values(["code","statDate"])
        out = FUND_DIR / f"{label}_quarterly.csv"
        df.to_csv(str(out), index=False)
        print(f"✓ {len(df)} 条{label}数据 → {out.name}")

    return df_p, df_g


# ─────────────────────────────────────────────────────────────────────────────
# 下载盈利数据（多进程版）
# ─────────────────────────────────────────────────────────────────────────────

def download_profit(codes: list, start_year: int, end_year: int,
                    n_workers: int = 8) -> pd.DataFrame:
    """
    [BUG-FUND-01 FIX] 多进程并发下载盈利数据。
    8进程并行：预计 5~12 分钟（原版 1.5~2.5 小时）。
    断点续传：重启时读取已有 partial parquet，补全缺失部分。
    """
    FUND_DIR.mkdir(parents=True, exist_ok=True)
    cp_path      = FUND_DIR / "_profit_checkpoint.json"
    partial_path = FUND_DIR / "_profit_partial.parquet"

    now = datetime.now()
    # [BUG-NEW-07 FIX] 财报通常滞后1~2个月，避免下载未发布季度
    avail_q = max(0, (now.month - 1 - 2) // 3)

    all_rows   = []
    field_names = None
    resume_from_idx = 0

    if cp_path.exists():
        with open(cp_path) as f:
            cp = json.load(f)
        resume_from_idx = cp.get("next_idx", 0)
        if partial_path.exists():
            df_p = pd.read_parquet(str(partial_path))
            all_rows    = df_p.values.tolist()
            field_names = list(df_p.columns)
        print(f"↻ profit 断点恢复 idx={resume_from_idx}, 已有 {len(all_rows)} 条")

    pending_codes = codes[resume_from_idx:]
    total = len(codes)

    if not pending_codes:
        print("  profit 已全部完成（断点文件）")
    else:
        print(f"✓ profit 多进程下载（n_workers={n_workers}）")
        print(f"  待处理: {len(pending_codes)} 只（从 idx={resume_from_idx}）")

        # chunk_size=32：每批32只登录一次（平衡进度刷新频率和登录开销）
        chunk_size = 32
        chunks = [pending_codes[s:s + chunk_size]
                  for s in range(0, len(pending_codes), chunk_size)]

        tasks = [
            {"codes_chunk": chunk, "start_year": start_year,
             "end_year": end_year, "now_year": now.year, "avail_q": avail_q}
            for chunk in chunks
        ]

        t0 = time.time()
        completed = 0
        total_errors = 0

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_worker_profit_chunk, task): task
                       for task in tasks}
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    print(f"  ⚠ worker 异常: {e}")
                    continue

                all_rows.extend(result.get("rows", []))
                if field_names is None and result.get("fields"):
                    field_names = result["fields"]
                total_errors += result.get("error_count", 0)
                chunk_len = len(futures[future]["codes_chunk"])
                completed += chunk_len

                elapsed = time.time() - t0
                speed   = completed / max(elapsed, 0.1)
                eta_min = (len(pending_codes) - completed) / max(speed, 0.01) / 60.0
                print(f"  profit [{resume_from_idx + completed}/{total}] "
                      f"{len(all_rows)} 条 | {speed:.1f} 只/秒 | "
                      f"ETA {eta_min:.0f}m | 错误 {total_errors}")
                sys.stdout.flush()

                # 断点保存（每完成约500只）
                if completed % 500 < chunk_len:
                    with open(cp_path, "w") as f:
                        json.dump({"next_idx": resume_from_idx + completed}, f)
                    if field_names and all_rows:
                        pd.DataFrame(all_rows, columns=field_names).to_parquet(
                            str(partial_path), index=False)
                    print(f"    💾 profit 断点 idx={resume_from_idx + completed}")

    if field_names is None:
        field_names = ["code", "pubDate", "statDate", "roeAvg", "npMargin",
                       "gpMargin", "netProfit", "epsTTM", "MBRevenue",
                       "totalShare", "liqaShare"]

    df = pd.DataFrame(all_rows, columns=field_names)
    df = clean_df(df)
    df["code"] = df["code"].apply(to_6digit)
    # [BUG-DEDUP FIX] 同一季度可能有更正报告（pubDate不同），保留最新公告日的记录
    df["pubDate"] = pd.to_datetime(df["pubDate"], errors="coerce")
    df = df.sort_values(["code", "statDate", "pubDate"])
    df = df.drop_duplicates(subset=["code", "statDate"], keep="last")
    df = df.sort_values(["code", "statDate"])

    out = FUND_DIR / "profit_quarterly.csv"
    df.to_csv(str(out), index=False)
    print(f"✓ {len(df)} 条盈利数据 → {out.name}")

    for p in [cp_path, partial_path]:
        if p.exists():
            p.unlink()

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 下载成长数据（多进程版）
# ─────────────────────────────────────────────────────────────────────────────

def download_growth(codes: list, start_year: int, end_year: int,
                    n_workers: int = 8) -> pd.DataFrame:
    """
    [BUG-FUND-02 FIX] 多进程并发下载成长数据。
    [BUG-NEW-02 FIX] 断点续传（原版无断点，全程中断须全部重来）
    [BUG-NEW-07 FIX] 季度可用性判断增加2个月发布延迟
    """
    FUND_DIR.mkdir(parents=True, exist_ok=True)
    cp_path      = FUND_DIR / "_growth_checkpoint.json"
    partial_path = FUND_DIR / "_growth_partial.parquet"

    now = datetime.now()
    avail_q = max(0, (now.month - 1 - 2) // 3)

    all_rows   = []
    field_names = None
    resume_from_idx = 0

    if cp_path.exists():
        with open(cp_path) as f:
            cp = json.load(f)
        resume_from_idx = cp.get("next_idx", 0)
        if partial_path.exists():
            df_p = pd.read_parquet(str(partial_path))
            all_rows    = df_p.values.tolist()
            field_names = list(df_p.columns)
        print(f"↻ growth 断点恢复 idx={resume_from_idx}, 已有 {len(all_rows)} 条")

    pending_codes = codes[resume_from_idx:]
    total = len(codes)

    if not pending_codes:
        print("  growth 已全部完成（断点文件）")
    else:
        print(f"✓ growth 多进程下载（n_workers={n_workers}）")
        print(f"  待处理: {len(pending_codes)} 只（从 idx={resume_from_idx}）")

        # chunk_size=32：每批32只登录一次（平衡进度刷新频率和登录开销）
        chunk_size = 32
        chunks = [pending_codes[s:s + chunk_size]
                  for s in range(0, len(pending_codes), chunk_size)]

        tasks = [
            {"codes_chunk": chunk, "start_year": start_year,
             "end_year": end_year, "now_year": now.year, "avail_q": avail_q}
            for chunk in chunks
        ]

        t0 = time.time()
        completed = 0
        total_errors = 0

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_worker_growth_chunk, task): task
                       for task in tasks}
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    print(f"  ⚠ worker 异常: {e}")
                    continue

                all_rows.extend(result.get("rows", []))
                if field_names is None and result.get("fields"):
                    field_names = result["fields"]
                total_errors += result.get("error_count", 0)
                chunk_len = len(futures[future]["codes_chunk"])
                completed += chunk_len

                elapsed = time.time() - t0
                speed   = completed / max(elapsed, 0.1)
                eta_min = (len(pending_codes) - completed) / max(speed, 0.01) / 60.0
                print(f"  growth [{resume_from_idx + completed}/{total}] "
                      f"{len(all_rows)} 条 | {speed:.1f} 只/秒 | "
                      f"ETA {eta_min:.0f}m | 错误 {total_errors}")
                sys.stdout.flush()

                if completed % 500 < chunk_len:
                    with open(cp_path, "w") as f:
                        json.dump({"next_idx": resume_from_idx + completed}, f)
                    if field_names and all_rows:
                        pd.DataFrame(all_rows, columns=field_names).to_parquet(
                            str(partial_path), index=False)
                    print(f"    💾 growth 断点 idx={resume_from_idx + completed}")

    if field_names is None:
        field_names = ["code", "pubDate", "statDate",
                       "YOYEquity", "YOYAsset", "YOYNI", "YOYEPSBasic", "YOYPNI"]

    df = pd.DataFrame(all_rows, columns=field_names)
    df = clean_df(df)
    df["code"] = df["code"].apply(to_6digit)
    # [BUG-DEDUP FIX] 同一季度可能有更正报告（pubDate不同），保留最新公告日的记录
    df["pubDate"] = pd.to_datetime(df["pubDate"], errors="coerce")
    df = df.sort_values(["code", "statDate", "pubDate"])
    df = df.drop_duplicates(subset=["code", "statDate"], keep="last")
    df = df.sort_values(["code", "statDate"])

    out = FUND_DIR / "growth_quarterly.csv"
    df.to_csv(str(out), index=False)
    print(f"✓ {len(df)} 条成长数据 → {out.name}")

    for _cp_clean in [cp_path, partial_path]:
        if _cp_clean.exists():
            _cp_clean.unlink()

    return df


def merge_and_export() -> pd.DataFrame:
    profit_path = FUND_DIR / "profit_quarterly.csv"
    growth_path = FUND_DIR / "growth_quarterly.csv"

    df_p = pd.read_csv(str(profit_path), dtype={"code": str})
    df_g = pd.read_csv(str(growth_path), dtype={"code": str})

    # [BUG-MERGE-KEY FIX] 原版按(code,statDate,pubDate)合并：
    # 若profit和growth对同一季报的pubDate因各自存在更正而不同，
    # 三键合并找不到匹配行，导致该季报的growth字段全部NaN，
    # 进而 fundamental_yoy_ni/yoy_eps 在这些季报上永久缺失。
    # 修复：仅按(code,statDate)合并，pubDate取两表的较晚值（保守point-in-time）。
    df_p["pubDate"] = pd.to_datetime(df_p["pubDate"], errors="coerce")
    df_g["pubDate"] = pd.to_datetime(df_g["pubDate"], errors="coerce")

    # growth表重命名pubDate，避免合并后冲突
    df_g = df_g.rename(columns={"pubDate": "pubDate_g"})

    df = pd.merge(df_p, df_g, on=["code", "statDate"], how="left")

    # pubDate取两者较晚值（两份数据都到位后才算point-in-time可用）
    pub_g = pd.to_datetime(df.get("pubDate_g"), errors="coerce")
    df["pubDate"] = df[["pubDate", "pubDate_g"]].apply(
        lambda r: max(r["pubDate"], r["pubDate_g"])
                  if pd.notna(r["pubDate"]) and pd.notna(r["pubDate_g"])
                  else (r["pubDate"] if pd.notna(r["pubDate"]) else r["pubDate_g"]),
        axis=1
    )
    if "pubDate_g" in df.columns:
        df = df.drop(columns=["pubDate_g"])

    df = df.sort_values(["code", "pubDate"])

    out = FUND_DIR / "fundamental_merged.csv"
    df.to_csv(str(out), index=False)
    print(f"✓ 合并后 {len(df)} 条 → {out.name}")
    print(f"  股票数: {df['code'].nunique()} | 日期: {df['statDate'].min()} ~ {df['statDate'].max()}")

    # 校验：growth列不应大量NaN
    for col in ["YOYNI", "YOYEPSBasic"]:
        if col in df.columns:
            na_pct = df[col].isna().mean() * 100
            if na_pct > 30:
                print(f"  ⚠ {col} 缺失率={na_pct:.1f}%，请检查growth数据是否完整")
            else:
                print(f"  ✓ {col} 缺失率={na_pct:.1f}%")
    return df


if __name__ == "__main__":
    # [BUG-FUND-01/02 FIX] Windows multiprocessing(spawn) 需要 __name__ guard，
    # 防止子进程重入时再次执行 main 逻辑。

    parser = argparse.ArgumentParser(description="Step 1: 下载季度基本面数据")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--test",    action="store_true", help="测试模式(10只)")
    parser.add_argument("--force",   action="store_true", help="强制重新下载")
    # [BUG-FUND-01/02 FIX] 新增 --workers
    parser.add_argument("--workers", type=int, default=8,
                        help="并发进程数（默认8，BaoStock 服务器舒适上限）")
    args = parser.parse_args()

    FUND_DIR.mkdir(parents=True, exist_ok=True)
    end_year = datetime.now().year
    codes = load_codes()

    if args.test:
        codes = ["000001", "000002", "000858", "600000", "600036",
                 "600519", "601318", "300750", "002594", "601012"]
        print(f"⚡ 测试模式: {len(codes)} 只股票")

    if args.force:
        for f in FUND_DIR.glob("*.csv"):
            f.unlink()
        print("🗑 旧数据已清除")

    print("=" * 70)
    print(f" Step 1: 季度基本面下载  {args.start_year} ~ {end_year}  "
          f"（{args.workers} 进程并发，预计 20~40 分钟）")
    print("=" * 70)

    profit_exists = (FUND_DIR / "profit_quarterly.csv").exists()
    growth_exists = (FUND_DIR / "growth_quarterly.csv").exists()

    if profit_exists and growth_exists:
        print("⚠ profit_quarterly.csv + growth_quarterly.csv 均已存在，跳过（--force 可重新下载）")
    else:
        # [SPEED-OPT] 合并下载：一次 login 同时拿 profit+growth，比分开快 ~40%
        download_combined(codes, args.start_year, end_year, n_workers=args.workers)

    merge_and_export()
    print("\n✓ Step 1 完成！")



