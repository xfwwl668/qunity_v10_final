"""
scripts/step2_download_concepts_tdxquant.py
=============================================
Q-UNITY V10 行业/概念板块下载 —— TdxQuant 通达信本地版

【数据层级】

A. 通达信内置行业（来自 get_stock_info.rs_hyname）
   每只股票都有，无需单独接口调用，精度高
   字段: rs_hyname（如"酿酒"/"全国性银行"）
        rs_hycode_sim（如"X2102"/"X5001"）
   输出: data/fundamental/industry_tdxquant.parquet

B. 通达信概念板块（get_stock_list_in_sector）
   约300~400个概念，每个板块查一次
   示例: 白酒(20只) / 银行(42只) / 半导体(199只)
   输出: data/concepts/tdxquant_concepts.csv
         data/concepts/tdxquant_concept_members.parquet

C. 通达信行业板块（get_sector_list + get_stock_list_in_sector）
   list_type=16(一级) / 17(二级) / 18(三级)
   ※需要在通达信客户端下载研究行业数据才可用
   输出: data/concepts/tdxquant_sectors.csv

【用法】
  python scripts/step2_download_concepts_tdxquant.py --test
  python scripts/step2_download_concepts_tdxquant.py --industry   # 仅行业（快，秒级）
  python scripts/step2_download_concepts_tdxquant.py --concepts   # 仅概念
  python scripts/step2_download_concepts_tdxquant.py --all        # 全部
"""

import argparse
import io
import contextlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from tqcenter_utils import get_tq

DATA_DIR    = PROJECT_ROOT / "data"
FUND_DIR    = DATA_DIR / "fundamental"
CONCEPT_DIR = DATA_DIR / "concepts"
FUND_DIR.mkdir(parents=True, exist_ok=True)
CONCEPT_DIR.mkdir(parents=True, exist_ok=True)


def _silent(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# A. 行业分类（from get_stock_info.rs_hyname）
# ─────────────────────────────────────────────────────────────────────────────

def download_industry(codes: list, tq, n_workers: int = 8) -> pd.DataFrame:
    """
    从 get_stock_info 批量获取行业分类。
    比 get_sector_list 更稳定（无需下载研究行业数据）。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print(f"\n  [A] 行业分类: {len(codes)} 只股票...")
    t0 = time.time()
    rows = []
    total = len(codes)

    def _one(tdx_code):
        try:
            info = _silent(tq.get_stock_info, stock_code=tdx_code, field_list=[])
            if not info:
                return None
            name     = str(info.get("Name", "") or "")
            ind_name = str(info.get("rs_hyname", "") or "")
            ind_code = str(info.get("rs_hycode_sim", "") or "")
            is_st    = 1 if info.get("IsSTGP") else 0
            is_quit  = 1 if info.get("IsQuitGP") else 0
            return {"code": tdx_code, "name": name,
                    "industry": ind_name, "industry_code": ind_code,
                    "is_st": is_st, "is_quit": is_quit}
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_one, c): c for c in codes}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r:
                rows.append(r)
            if i % 500 == 0 or i == total:
                print(f"\r    [{i:5d}/{total}]  {i/max(time.time()-t0,0.1):.0f}只/s  ",
                      end="", flush=True)

    elapsed = time.time() - t0
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    print(f"\n  ✓ 行业分类: {len(df)}只  耗时={elapsed:.0f}s")

    if not df.empty:
        ind_cnt = df["industry"].value_counts()
        print(f"  行业数: {len(ind_cnt)}  前5: {list(ind_cnt.head().index)}")
        out = FUND_DIR / "industry_tdxquant.parquet"
        df.to_parquet(str(out), index=False)
        df.to_csv(str(FUND_DIR / "industry_tdxquant.csv"),
                  index=False, encoding="utf-8-sig")
        print(f"  → {out.name}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# B. 概念板块（get_stock_list_in_sector by name）
# ─────────────────────────────────────────────────────────────────────────────

# 通达信内置的核心概念名称列表（实测可用）
# 注：概念名称需要精确匹配，可通过 get_sector_list(list_type=12) 获取完整列表
_CORE_CONCEPTS = [
    # 行业概念
    "白酒", "银行", "半导体", "新能源汽车", "光伏",
    "人工智能", "大数据", "云计算", "芯片", "医药",
    "券商", "保险", "地产", "煤炭", "钢铁",
    "有色金属", "化工", "军工", "传媒", "游戏",
    "消费电子", "汽车零部件", "锂电池", "储能", "氢能",
    "机器人", "数字经济", "元宇宙", "网络安全", "自动驾驶",
    # 指数概念
    "沪深300成分股", "上证50成分股", "中证500成分股",
    "创业板50成分股", "科创50成分股",
    # 特殊标签
    "MSCI中国概念", "北向资金重仓", "深港通", "沪港通",
    "转债概念", "高股息率",
]


def _get_concept_list(tq) -> list:
    """
    尝试从 get_sector_list(list_type=12) 获取完整概念列表。
    如果返回空（需要下载数据），则使用内置列表。
    """
    try:
        result = _silent(tq.get_sector_list, list_type=12)
        if result and len(result) > 10:
            names = []
            for item in result:
                n = item.get("Name", "") if isinstance(item, dict) else str(item)
                if n:
                    names.append(n)
            if names:
                print(f"  从 get_sector_list(12) 获取到 {len(names)} 个概念名称")
                return names
    except Exception:
        pass
    print(f"  get_sector_list 返回空（需下载研究数据），使用内置 {len(_CORE_CONCEPTS)} 个核心概念")
    return _CORE_CONCEPTS


def download_concepts(tq, concept_names: list = None) -> pd.DataFrame:
    """
    下载概念板块成员列表。
    返回 DataFrame: code / concept_name / stock_name
    """
    if concept_names is None:
        concept_names = _get_concept_list(tq)

    total = len(concept_names)
    print(f"\n  [B] 概念板块: {total} 个概念...")
    t0 = time.time()

    all_rows = []
    concept_meta = []  # (concept_name, member_count)
    empty_count = 0

    for i, name in enumerate(concept_names):
        try:
            members = _silent(tq.get_stock_list_in_sector, name)
            if members and len(members) > 0:
                cnt = 0
                for m in members:
                    code = m.get("Code", "") if isinstance(m, dict) else str(m)
                    mname = m.get("Name", "") if isinstance(m, dict) else ""
                    if code:
                        all_rows.append({
                            "concept_name": name,
                            "code":         code,
                            "stock_name":   mname,
                        })
                        cnt += 1
                concept_meta.append({"concept_name": name, "member_count": cnt})
                print(f"\r    [{i+1:3d}/{total}] {name[:10]:<10} {cnt:3d}只  ", end="", flush=True)
            else:
                empty_count += 1
                concept_meta.append({"concept_name": name, "member_count": 0})
        except Exception as e:
            empty_count += 1

        # 每30个保存一次（断点续传）
        if (i + 1) % 30 == 0 and all_rows:
            _save_concepts_partial(all_rows, concept_meta)

    elapsed = time.time() - t0
    valid = total - empty_count
    print(f"\n  ✓ 概念下载: {valid}/{total} 个有效  {len(all_rows)}条成员记录  耗时={elapsed:.0f}s")

    df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
    if not df.empty:
        _save_concepts_final(df, pd.DataFrame(concept_meta))
    return df


def _save_concepts_partial(rows, meta_rows):
    """中间保存（断点续传）"""
    tmp = CONCEPT_DIR / "_partial_concept_members.parquet"
    pd.DataFrame(rows).to_parquet(str(tmp), index=False)


def _save_concepts_final(df_members, df_meta):
    """最终保存"""
    out1 = CONCEPT_DIR / "tdxquant_concept_members.parquet"
    out2 = CONCEPT_DIR / "tdxquant_concepts.csv"
    df_members.to_parquet(str(out1), index=False)
    df_meta.to_csv(str(out2), index=False, encoding="utf-8-sig")
    print(f"  → {out1.name}  ({len(df_members)}条成员)")
    print(f"  → {out2.name}  ({len(df_meta)}个概念)")
    # 清除中间文件
    tmp = CONCEPT_DIR / "_partial_concept_members.parquet"
    if tmp.exists():
        tmp.unlink()


# ─────────────────────────────────────────────────────────────────────────────
# C. 申万行业板块（get_sector_list list_type=16/17/18）
# ─────────────────────────────────────────────────────────────────────────────

def download_sw_sectors(tq) -> dict:
    """
    下载申万行业板块（需通达信已下载研究行业数据）。
    返回 {level: DataFrame}，level = "L1"/"L2"/"L3"
    """
    print(f"\n  [C] 申万行业板块（需下载研究行业数据）...")
    results = {}
    for lt, label in [(16, "L1一级"), (17, "L2二级"), (18, "L3三级")]:
        try:
            sectors = _silent(tq.get_sector_list, list_type=lt)
            if not sectors:
                print(f"    {label}: 0个（请在通达信下载研究行业数据）")
                continue
            sector_names = []
            for s in sectors:
                n = s.get("Name", "") if isinstance(s, dict) else str(s)
                if n:
                    sector_names.append(n)
            print(f"    {label}: {len(sector_names)} 个板块")

            # 获取每个行业的成员
            all_rows = []
            for sname in sector_names:
                members = _silent(tq.get_stock_list_in_sector, sname)
                if members:
                    for m in members:
                        code  = m.get("Code", "") if isinstance(m, dict) else str(m)
                        mname = m.get("Name", "") if isinstance(m, dict) else ""
                        if code:
                            all_rows.append({
                                "level":       label,
                                "sector_name": sname,
                                "code":        code,
                                "stock_name":  mname,
                            })
            if all_rows:
                df = pd.DataFrame(all_rows)
                out = CONCEPT_DIR / f"tdxquant_sw_{label[:2].lower()}.parquet"
                df.to_parquet(str(out), index=False)
                print(f"    → {out.name}  ({len(df)}条)")
                results[label] = df
        except Exception as e:
            print(f"    {label}: 失败 ({e})")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def run(codes: list, tq, n_workers: int,
        do_industry: bool = True,
        do_concepts: bool = True,
        do_sw:       bool = False,
        test_concepts: list = None) -> None:

    print(f"\n{'='*65}")
    print(f"  Step 2: 行业/概念下载 — TdxQuant 通达信本地版")
    print(f"  输出目录: {CONCEPT_DIR}")
    print(f"{'='*65}")

    if do_industry:
        download_industry(codes, tq, n_workers)

    if do_concepts:
        download_concepts(tq, concept_names=test_concepts)

    if do_sw:
        download_sw_sectors(tq)

    print(f"\n✓ Step 2 完成！")
    print(f"  下一步: python scripts/step4_build_concept_npy.py")


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Q-UNITY V10 行业/概念下载 — TdxQuant")
    ap.add_argument("--workers",  type=int, default=8)
    ap.add_argument("--test",     action="store_true",
                    help="测试模式（10只股票 + 5个概念）")
    ap.add_argument("--industry", action="store_true",
                    help="仅下载行业分类（get_stock_info.rs_hyname）")
    ap.add_argument("--concepts", action="store_true",
                    help="仅下载概念板块")
    ap.add_argument("--sw",       action="store_true",
                    help="下载申万行业（需通达信已下载研究行业数据）")
    ap.add_argument("--all",      action="store_true",
                    help="下载全部（行业+概念+申万行业）")
    args = ap.parse_args()

    tq = get_tq(__file__)
    if tq is None:
        print("✗ 未找到 TdxQuant")
        sys.exit(1)

    if args.test:
        test_codes = ["600519.SH","601318.SH","000001.SZ","000858.SZ",
                      "300750.SZ","600036.SH","002594.SZ","688981.SH",
                      "000002.SZ","600900.SH"]
        test_concepts = ["白酒", "银行", "半导体", "新能源汽车", "光伏"]
        print(f"⚡ 测试模式: {len(test_codes)}只 / {len(test_concepts)}概念")
        run(codes=test_codes, tq=tq, n_workers=args.workers,
            do_industry=True, do_concepts=True, do_sw=False,
            test_concepts=test_concepts)
    else:
        print("▶ 获取股票列表...")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = tq.get_stock_list("5")
        codes = [
            (item.get("Code","") if isinstance(item, dict) else str(item))
            for item in (result or [])
            if (item.get("Code","") if isinstance(item, dict) else str(item))
        ]
        print(f"  共 {len(codes)} 只")

        run(
            codes      = codes,
            tq         = tq,
            n_workers  = args.workers,
            do_industry= args.industry or args.all or (not args.concepts and not args.sw),
            do_concepts= args.concepts or args.all,
            do_sw      = args.sw or args.all,
        )

    try:
        tq.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
