"""
scripts/step1_download_fundamental_tdxquant.py
================================================
Q-UNITY V10 基本面数据下载 —— TdxQuant 通达信本地版

【数据来源与字段】
TdxQuant 提供三层基本面接口，互为补充：

  层1: get_stock_info()  63字段，每只股票最新财务快照
       → ROE / EPS / 净利润 / 总资产 / 营收 / 净利率 …
       → J_start(上市日) / IsSTGP / IsQuitGP / rs_hyname(行业)
       单位: 利润/资产均为万元，股本为万股

  层2: get_more_info()   实时估值，约80字段
       → PE_TTM / PB_MRQ / DynaPE / DYRatio
       → Zsz(总市值亿元) / Ltsz(流通市值亿元)
       → ZAF(今日涨幅) / ZAFPre5/20 …
       5ms/只，5000只全市场约25秒

  层3: get_gb_info()     历史股本变动序列
       → Date / Zgb(总股本) / Ltgb(流通股本)
       单位: 股（不是万股）

【输出文件】（与 step1_download_fundamental_akshare.py 兼容格式）
  data/fundamental/fundamental_tdxquant_stock_info.parquet
      → code / name / roe / eps / net_profit / total_share / liq_share
        net_profit_margin / yoy_ni / industry / list_date / is_st / is_quit
  data/fundamental/fundamental_tdxquant_valuation.parquet
      → code / pe_ttm / pb_mrq / dyna_pe / dy_ratio / mkt_cap / mkt_cap_circ
  data/fundamental/fundamental_tdxquant_shares.parquet
      → code / date / total_share / liq_share  (历史股本)

【用法】
  python scripts/step1_download_fundamental_tdxquant.py --test
  python scripts/step1_download_fundamental_tdxquant.py --workers 8
  python scripts/step1_download_fundamental_tdxquant.py --workers 8 --valuation
  python scripts/step1_download_fundamental_tdxquant.py --all
"""

import argparse
import io
import contextlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# ── 路径 ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from tqcenter_utils import get_tq, find_tqcenter

DATA_DIR  = PROJECT_ROOT / "data"
FUND_DIR  = DATA_DIR / "fundamental"
FUND_DIR.mkdir(parents=True, exist_ok=True)

# 万元 → 元 的转换因子（get_stock_info 利润/资产单位是万元）
_WAN = 10_000.0


def _silent(fn, *args, **kwargs):
    """调用 TdxQuant 接口时静音其内部 print 输出"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 层1: get_stock_info — 基础财务快照
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_stock_info_one(tdx_code: str, tq) -> dict:
    """获取单只股票的 get_stock_info 快照，返回标准化字典"""
    try:
        info = _silent(tq.get_stock_info, stock_code=tdx_code, field_list=[])
        if not info:
            return {}

        def _safe(k, default=np.nan):
            v = info.get(k)
            if v is None:
                return default
            try:
                return float(str(v).replace(",", ""))
            except (ValueError, TypeError):
                return default

        # J_start 格式: 20010827（整数）→ "2001-08-27"
        j_start = info.get("J_start", 0) or 0
        try:
            js = str(int(j_start))
            list_date = f"{js[:4]}-{js[4:6]}-{js[6:8]}" if len(js) == 8 else ""
        except Exception:
            list_date = ""

        # 股本：J_zgb/ActiveCapital 单位是万股
        total_share = _safe("J_zgb")       # 万股
        liq_share   = _safe("ActiveCapital")  # 万股

        # 财务数据：万元位
        net_profit  = _safe("J_lyze")      # 净利润（万元）
        total_rev   = _safe("J_yysy")      # 营业收入（万元）
        net_margin  = np.nan
        if total_rev > 0 and not np.isnan(net_profit):
            net_margin = net_profit / total_rev * 100.0  # %

        # EPS: J_mgsy（每股收益），ROE 从 J_jzc（净资产）和净利润推算
        eps    = _safe("J_mgsy")
        net_asset = _safe("J_jzc")    # 净资产（万元）
        roe = np.nan
        if net_asset > 0 and not np.isnan(net_profit):
            roe = net_profit / net_asset * 100.0  # %

        # YOY 净利润增速（无直接字段，需两期对比 → 此处留 nan，由 step3 差分计算）
        yoy_ni = np.nan

        return {
            "code":         tdx_code,
            "name":         str(info.get("Name", "") or ""),
            "roe":          round(roe, 4)        if not np.isnan(roe)        else np.nan,
            "eps":          round(eps, 4)        if not np.isnan(eps)        else np.nan,
            "net_profit":   round(net_profit, 2) if not np.isnan(net_profit) else np.nan,
            "total_rev":    round(total_rev, 2)  if not np.isnan(total_rev)  else np.nan,
            "total_share":  round(total_share, 2)if not np.isnan(total_share)else np.nan,
            "liq_share":    round(liq_share, 2)  if not np.isnan(liq_share)  else np.nan,
            "net_profit_margin": round(net_margin, 4) if not np.isnan(net_margin) else np.nan,
            "yoy_ni":       yoy_ni,
            "industry":     str(info.get("rs_hyname", "") or ""),
            "industry_code":str(info.get("rs_hycode_sim", "") or ""),
            "list_date":    list_date,
            "is_st":        1 if info.get("IsSTGP") else 0,
            "is_quit":      1 if info.get("IsQuitGP") else 0,
        }
    except Exception as e:
        return {"code": tdx_code, "_error": str(e)[:80]}


def download_stock_info(codes: list, tq, n_workers: int = 8) -> pd.DataFrame:
    """批量下载 get_stock_info，返回 DataFrame"""
    total = len(codes)
    print(f"\n  [层1] get_stock_info: {total} 只股票...")
    t0 = time.time()
    rows = []
    ok = err = 0

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_fetch_stock_info_one, c, tq): c for c in codes}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r and "_error" not in r:
                rows.append(r)
                ok += 1
            else:
                err += 1
            if i % 500 == 0 or i == total:
                spd = i / max(time.time() - t0, 0.1)
                print(f"\r    [{i:5d}/{total}] ok={ok} err={err}  {spd:.0f}只/s"
                      f"  ETA={(total-i)/max(spd,0.1):.0f}s  ", end="", flush=True)

    elapsed = time.time() - t0
    print(f"\n  ✓ get_stock_info 完成: {ok}只  耗时={elapsed:.0f}s")
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 层2: get_more_info — 实时估值（PE/PB/市值）
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_more_info_one(tdx_code: str, tq) -> dict:
    """获取单只股票的 get_more_info 估值快照"""
    try:
        more = _silent(tq.get_more_info, stock_code=tdx_code, field_list=[])
        if not more:
            return {}

        def _f(k):
            v = more.get(k)
            if v is None:
                return np.nan
            try:
                return float(str(v).replace(",", ""))
            except (ValueError, TypeError):
                return np.nan

        return {
            "code":         tdx_code,
            "pe_ttm":       _f("StaticPE_TTM"),
            "pb_mrq":       _f("PB_MRQ"),
            "dyna_pe":      _f("DynaPE"),
            "dy_ratio":     _f("DYRatio"),       # 股息率(%)
            "mkt_cap":      _f("Zsz"),            # 总市值（亿元）
            "mkt_cap_circ": _f("Ltsz"),           # 流通市值（亿元）
            "zt_price":     _f("ZTPrice"),        # 涨停价
            "dt_price":     _f("DTPrice"),        # 跌停价
            "zaf_1d":       _f("ZAF"),            # 今日涨幅(%)
            "zaf_5d":       _f("ZAFPre5"),
            "zaf_20d":      _f("ZAFPre20"),
            "his_high_52w": _f("HisHigh"),        # 52周最高
            "his_low_52w":  _f("HisLow"),
            "beta":         _f("BetaValue"),
            "con_zaf":      more.get("ConZAFDateNum", 0) or 0,  # 连涨/跌天数
        }
    except Exception as e:
        return {"code": tdx_code, "_error": str(e)[:80]}


def download_valuation(codes: list, tq, n_workers: int = 8) -> pd.DataFrame:
    """批量下载 get_more_info（PE/PB/市值），返回 DataFrame。约25秒/5000只"""
    total = len(codes)
    print(f"\n  [层2] get_more_info: {total} 只股票（≈{total*0.005:.0f}秒）...")
    t0 = time.time()
    rows = []
    ok = err = 0

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_fetch_more_info_one, c, tq): c for c in codes}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r and "_error" not in r:
                rows.append(r)
                ok += 1
            else:
                err += 1
            if i % 500 == 0 or i == total:
                spd = i / max(time.time() - t0, 0.1)
                print(f"\r    [{i:5d}/{total}] ok={ok} err={err}  {spd:.0f}只/s"
                      f"  ETA={(total-i)/max(spd,0.1):.0f}s  ", end="", flush=True)

    elapsed = time.time() - t0
    print(f"\n  ✓ get_more_info 完成: {ok}只  耗时={elapsed:.0f}s")
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# 层3: get_gb_info — 历史股本序列
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_gb_one(tdx_code: str, tq) -> list:
    """获取单只股票的股本历史列表"""
    try:
        rows = _silent(tq.get_gb_info, stock_code=tdx_code)
        if not rows:
            return []
        result = []
        for r in rows:
            date_int = r.get("Date", 0)
            date_str = str(int(date_int))
            if len(date_str) == 8:
                date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            else:
                continue
            # get_gb_info 的 Zgb/Ltgb 单位是股，转换为万股
            total = (r.get("Zgb") or 0) / 10_000.0
            liq   = (r.get("Ltgb") or 0) / 10_000.0
            result.append({
                "code":        tdx_code,
                "date":        date_str,
                "total_share": round(total, 2),
                "liq_share":   round(liq, 2),
            })
        return result
    except Exception:
        return []


def download_shares_history(codes: list, tq, n_workers: int = 8) -> pd.DataFrame:
    """批量下载股本历史，返回 DataFrame"""
    total = len(codes)
    print(f"\n  [层3] get_gb_info: {total} 只股票股本历史...")
    t0 = time.time()
    all_rows = []
    ok = err = 0

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_fetch_gb_one, c, tq): c for c in codes}
        for i, fut in enumerate(as_completed(futures), 1):
            rows = fut.result()
            if rows:
                all_rows.extend(rows)
                ok += 1
            else:
                err += 1
            if i % 500 == 0 or i == total:
                print(f"\r    [{i:5d}/{total}] ok={ok}  ", end="", flush=True)

    elapsed = time.time() - t0
    print(f"\n  ✓ get_gb_info 完成: {ok}只  {len(all_rows)}条记录  耗时={elapsed:.0f}s")
    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def run(codes: list, tq, n_workers: int,
        do_stock_info: bool = True,
        do_valuation: bool = True,
        do_shares: bool = False) -> None:

    print(f"\n{'='*65}")
    print(f"  Step 1: 基本面下载 — TdxQuant 通达信本地版")
    print(f"  股票数: {len(codes)}  并发: {n_workers}")
    print(f"  输出目录: {FUND_DIR}")
    print(f"{'='*65}")

    if do_stock_info:
        df_info = download_stock_info(codes, tq, n_workers)
        if not df_info.empty:
            out = FUND_DIR / "fundamental_tdxquant_stock_info.parquet"
            df_info.to_parquet(str(out), index=False)
            print(f"  → {out.name}  ({len(df_info)}行)")
            # 同时输出 CSV 供人工检查
            out_csv = FUND_DIR / "fundamental_tdxquant_stock_info.csv"
            df_info.to_csv(str(out_csv), index=False, encoding="utf-8-sig")

            # 统计行业覆盖
            ind_cnt = df_info["industry"].value_counts()
            st_cnt  = int(df_info["is_st"].sum())
            quit_cnt= int(df_info["is_quit"].sum())
            print(f"  行业数: {len(ind_cnt)}  ST股: {st_cnt}  退市: {quit_cnt}")
            print(f"  前5大行业: {dict(ind_cnt.head())}")

    if do_valuation:
        df_val = download_valuation(codes, tq, n_workers)
        if not df_val.empty:
            out = FUND_DIR / "fundamental_tdxquant_valuation.parquet"
            df_val.to_parquet(str(out), index=False)
            print(f"  → {out.name}  ({len(df_val)}行)")
            df_val.to_csv(str(FUND_DIR / "fundamental_tdxquant_valuation.csv"),
                          index=False, encoding="utf-8-sig")

    if do_shares:
        df_sh = download_shares_history(codes, tq, n_workers)
        if not df_sh.empty:
            out = FUND_DIR / "fundamental_tdxquant_shares.parquet"
            df_sh.to_parquet(str(out), index=False)
            print(f"  → {out.name}  ({len(df_sh)}行)")

    print(f"\n✓ Step 1 完成！（TdxQuant）")
    print(f"  下一步: python scripts/step3_build_fundamental_npy.py")


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Q-UNITY V10 基本面下载 — TdxQuant 通达信本地版")
    ap.add_argument("--workers",    type=int, default=8,
                    help="并发线程数（默认8）")
    ap.add_argument("--test",       action="store_true",
                    help="测试模式（10只）")
    ap.add_argument("--stock-info", action="store_true", default=True,
                    help="下载 get_stock_info（ROE/EPS/行业等）[默认开启]")
    ap.add_argument("--valuation",  action="store_true",
                    help="下载 get_more_info（PE/PB/市值）")
    ap.add_argument("--shares",     action="store_true",
                    help="下载 get_gb_info（历史股本序列）")
    ap.add_argument("--all",        action="store_true",
                    help="下载全部三层数据")
    args = ap.parse_args()

    tq = get_tq(__file__)
    if tq is None:
        print("✗ 未找到 TdxQuant，请确认通达信客户端已开启")
        sys.exit(1)

    # 获取股票列表
    if args.test:
        codes = ["600519.SH","601318.SH","000001.SZ","000858.SZ",
                 "300750.SZ","600036.SH","002594.SZ","688981.SH",
                 "000002.SZ","600900.SH"]
        print(f"⚡ 测试模式: {len(codes)} 只")
    else:
        print("▶ 获取股票列表...")
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                result = tq.get_stock_list("5")
            codes = []
            for item in (result or []):
                c = item.get("Code","") if isinstance(item, dict) else str(item)
                if c:
                    codes.append(c)
            print(f"  共 {len(codes)} 只")
        except Exception as e:
            print(f"  ✗ 获取列表失败: {e}")
            sys.exit(1)

    if args.all:
        args.stock_info = args.valuation = args.shares = True

    try:
        run(
            codes       = codes,
            tq          = tq,
            n_workers   = args.workers,
            do_stock_info = args.stock_info or args.all,
            do_valuation  = args.valuation or args.all,
            do_shares     = args.shares or args.all,
        )
    finally:
        try:
            tq.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
