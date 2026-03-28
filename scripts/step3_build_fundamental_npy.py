"""
scripts/step3_build_fundamental_npy.py

将季度基本面 CSV 转为 Q-UNITY 的 (N, T) npy 矩阵。

关键设计：
  1. 使用 pubDate（公告日）做 point-in-time 对齐，完全杜绝未来数据泄露
  2. 每份季报从 pubDate 起 forward-fill 到下次公告日（或矩阵末尾）
  3. 矩阵形状与 meta.json 中的 codes/dates 严格对齐

生成文件（均为 float32，形状 [n_stocks, n_dates]）:
  data/npy/fundamental_roe.npy
  data/npy/fundamental_eps_ttm.npy
  data/npy/fundamental_net_profit.npy
  data/npy/fundamental_total_share.npy
  data/npy/fundamental_liq_share.npy
  data/npy/fundamental_net_profit_margin.npy
  data/npy/fundamental_yoy_ni.npy
  data/npy/fundamental_yoy_eps.npy
  data/npy/market_cap_total.npy  （亿元，需 close.npy + total_share.npy）
  data/npy/market_cap_circ.npy   （亿元，需 close.npy + liq_share.npy）

用法:
  python scripts/step3_build_fundamental_npy.py
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FUND_DIR = DATA_DIR / "fundamental"

# [FIX-C-01] 统一从 config.json 读取 npy_v10_dir，不再硬编码
try:
    from scripts.utils_paths import get_npy_dir
except ImportError:
    import sys; sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from utils_paths import get_npy_dir  # type: ignore

NPY_DIR = get_npy_dir("v10")
META_PATH = NPY_DIR / "meta.json"


# CSV列名 → (npy后缀, 描述)
FACTOR_MAP = {
    "roe":               ("roeAvg",      "ROE(平均)"),
    "eps_ttm":           ("epsTTM",      "EPS-TTM"),
    "net_profit":        ("netProfit",   "净利润(万元)"),
    "total_share":       ("totalShare",  "总股本(万股)"),
    "liq_share":         ("liqaShare",   "流通股本(万股)"),
    "net_profit_margin": ("npMargin",    "净利率(%)"),
    "yoy_ni":            ("YOYNI",       "净利润同比(%)"),
    "yoy_eps":           ("YOYEPSBasic", "EPS同比(%)"),
}


def build_fundamental_npy() -> None:
    print("=" * 70)
    print(" Step 3: 构建基本面 npy 矩阵")
    print("=" * 70)

    # ── 1. 加载 meta ──
    if not META_PATH.exists():
        print(f"✗ 未找到 {META_PATH}")
        sys.exit(1)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # meta["codes"] 格式为 "sh.600519" 或 "sz.000001"（BaoStock 9位格式）
    # 提取 6 位纯数字部分用于与 fundamental_merged.csv 的 code 列匹配
    def _to_6digit(c: str) -> str:
        s = str(c).strip()
        return s.split(".")[-1].zfill(6)  # "sh.600519" → "600519"; "600519" → "600519"

    codes = [_to_6digit(c) for c in meta["codes"]]
    dates = meta["dates"]
    n_stocks = len(codes)
    n_dates = len(dates)
    code_to_idx = {c: i for i, c in enumerate(codes)}
    dates_dt = pd.to_datetime(dates)
    dates_arr = dates_dt.values  # numpy datetime64[ns]

    print(f"  矩阵维度: {n_stocks} × {n_dates}")
    print(f"  日期范围: {dates[0]} ~ {dates[-1]}")

    # ── 2. 加载基本面数据 ──
    merged_path = FUND_DIR / "fundamental_merged.csv"
    if not merged_path.exists():
        print(f"✗ 未找到 {merged_path}，请先运行 step1")
        sys.exit(1)

    df = pd.read_csv(str(merged_path), dtype={"code": str})
    df["code"] = df["code"].str.zfill(6)
    df["pubDate"] = pd.to_datetime(df["pubDate"], errors="coerce")
    df = df.dropna(subset=["pubDate"])
    df = df.sort_values(["code", "pubDate"]).reset_index(drop=True)
    print(f"  基本面数据: {len(df)} 条, {df['code'].nunique()} 只股票")

    # ── 3. 逐因子构建矩阵 ──
    # [BUG-SKIP-ZERO-VALID-FIX] 原逻辑：npy 存在即跳过。
    # 问题：totalShare/liqaShare 等字段若因 Baostock 返回空字符串导致 CSV 全为 NaN，
    # 第一次构建的 npy 有效率 = 0%。之后每次运行都因"已存在"跳过，永久留存坏数据。
    # 修复：检查已存在 npy 的有效率，< 5% 时强制重建（打印警告）。
    _rebuild_valid_threshold = 5.0  # % 有效率低于此值时强制重建

    for factor_name, (col_name, desc) in FACTOR_MAP.items():
        if col_name not in df.columns:
            print(f"  ⚠ 跳过 {factor_name}: 列 '{col_name}' 不存在")
            continue

        out_path = NPY_DIR / f"fundamental_{factor_name}.npy"
        if out_path.exists():
            _existing = np.load(str(out_path), mmap_mode="r")
            _valid_pct = float(np.sum(~np.isnan(_existing)) / _existing.size * 100)
            del _existing
            if _valid_pct >= _rebuild_valid_threshold:
                print(f"  ⚠ {out_path.name} 已存在（有效率={_valid_pct:.1f}%），跳过（删除文件可重建）")
                continue
            else:
                print(f"  ⚠ {out_path.name} 有效率={_valid_pct:.1f}% < {_rebuild_valid_threshold}%，强制重建！")

        print(f"\n  构建 fundamental_{factor_name} ({desc})...")
        matrix = np.full((n_stocks, n_dates), np.nan, dtype=np.float32)

        df_f = df[["code", "pubDate", col_name]].dropna(subset=[col_name])

        for code, grp in df_f.groupby("code"):
            if code not in code_to_idx:
                continue
            si = code_to_idx[code]
            grp = grp.sort_values("pubDate")

            pub_dates = grp["pubDate"].values  # datetime64[ns]
            values = grp[col_name].values.astype(np.float32)

            for j in range(len(pub_dates)):
                pub = pub_dates[j]
                val = values[j]
                if np.isnan(val):
                    continue

                # 公告日在 dates_arr 中的插入位置
                t_start = int(np.searchsorted(dates_arr, pub, side="left"))
                if t_start >= n_dates:
                    continue

                # 结束位置：下次公告日或末尾
                if j + 1 < len(pub_dates):
                    t_end = int(np.searchsorted(
                        dates_arr, pub_dates[j + 1], side="left"
                    ))
                else:
                    t_end = n_dates

                matrix[si, t_start:t_end] = val

        np.save(str(out_path), matrix)
        valid_pct = np.sum(~np.isnan(matrix)) / matrix.size * 100
        print(f"    ✓ {out_path.name}  有效率={valid_pct:.1f}%")

    # ── 4. [BUG-FUND-PE-NEVER-DOWNLOADED FIX] 日频估值 → pe_ttm.npy / pb_mrq.npy / ps_ttm.npy ──
    _build_valuation_npy(n_stocks, n_dates, codes, code_to_idx, dates_arr)

    # ── 5. [BUG-FUND-SUE-NEVER-BUILT FIX] 从 epsTTM+statDate 计算 SUE ──
    _build_sue_npy(n_stocks, n_dates, df, codes, code_to_idx, dates_arr, dates_dt)

    # ── 6. [BUG-FUND-DAYS_ANN-NEVER-BUILT FIX] 从 pubDate 计算距公告交易日数 ──
    _build_days_ann_npy(n_stocks, n_dates, df, codes, code_to_idx, dates_arr)

    # ── 7. 市值矩阵 ──
    print("\n  构建市值矩阵...")
    _build_market_cap(n_stocks, n_dates)

    print(f"\n✓ Step 3 完成！所有基本面 npy 已保存到 {NPY_DIR}")


def _build_valuation_npy(n_stocks, n_dates, codes, code_to_idx, dates_arr):
    """
    [BUG-FUND-PE-NEVER-DOWNLOADED FIX]
    从 valuation_daily.csv（step1 AKShare/adata 生成）构建日频估值矩阵：
      pe_ttm.npy  ← peTTM（市盈率TTM）
      pb_mrq.npy  ← pbMRQ（市净率MRQ）
      ps_ttm.npy  ← psTTM（市销率TTM）

    [FIX-STEP3-TDX-VALUATION] 新增：若 valuation_daily.csv 不存在，
    自动 fallback 到 4a TdxQuant 生成的 fundamental_tdxquant_valuation.parquet。
    原逻辑只读 AKShare 路径，导致菜单提示"4a → 5"但实际 pe_ttm.npy 从未生成。

    日频数据逐日赋值（无需前向填充，每个交易日都有值）。
    """
    val_path = FUND_DIR / "valuation_daily.csv"
    tdx_path = FUND_DIR / "fundamental_tdxquant_valuation.parquet"
    tdx_csv  = FUND_DIR / "fundamental_tdxquant_valuation.csv"

    df_v = None

    if val_path.exists():
        df_v = pd.read_csv(str(val_path), dtype={"code": str})
        df_v["code"] = df_v["code"].str.zfill(6)
        print("  读取 valuation_daily.csv (AKShare/adata路径)")

    elif tdx_path.exists() or tdx_csv.exists():
        # [FIX-STEP3-TDX-VALUATION] fallback: TdxQuant 4a 路径
        src = tdx_path if tdx_path.exists() else tdx_csv
        print(f"  valuation_daily.csv 不存在，使用 TdxQuant fallback: {src.name}")
        if src.suffix == ".parquet":
            df_v = pd.read_parquet(str(src))
        else:
            df_v = pd.read_csv(str(src), dtype={"code": str})
        df_v["code"] = df_v["code"].astype(str).str.zfill(6)

        # TdxQuant 列名映射：pe_ttm → peTTM, pb_mrq → pbMRQ
        col_rename = {"pe_ttm": "peTTM", "pb_mrq": "pbMRQ", "ps_ttm": "psTTM",
                      "PE_TTM": "peTTM", "PB_MRQ": "pbMRQ"}
        df_v.rename(columns=col_rename, inplace=True)

        # TdxQuant 估值是当日快照（无 date 列时用 trade_date）
        for dcol in ["date", "trade_date", "tradeDate"]:
            if dcol in df_v.columns:
                df_v["date"] = df_v[dcol]
                break
        if "date" not in df_v.columns:
            print("  ⚠ TdxQuant 估值文件缺少日期列，跳过 pe_ttm/pb_mrq/ps_ttm 构建")
            return

    else:
        print("  ⚠ 估值数据不存在，跳过 pe_ttm/pb_mrq/ps_ttm 构建")
        print("     解决方法（三选一）：")
        print("     · TdxQuant路径(推荐): 数据管理 → 4a(TdxQuant估值) → 5(重新构建npy)")
        print("     · AKShare路径:        数据管理 → 4c 或 4d → 5")
        print("     · BaoStock路径:       数据管理 → 4e (直接生成 valuation_peTTM.npy，无需5)")
        return

    df_v["date"] = pd.to_datetime(df_v["date"], errors="coerce")
    df_v = df_v.dropna(subset=["date"]).sort_values(["code", "date"])

    col_npy_map = [
        ("peTTM", "pe_ttm.npy",  "PE-TTM"),
        ("pbMRQ", "pb_mrq.npy",  "PB-MRQ"),
        ("psTTM", "ps_ttm.npy",  "PS-TTM"),
    ]
    for col, npy_name, desc in col_npy_map:
        out_path = NPY_DIR / npy_name
        if out_path.exists():
            print(f"  ⚠ {npy_name} 已存在，跳过")
            continue
        if col not in df_v.columns:
            print(f"  ⚠ valuation_daily.csv 缺少 {col} 列，跳过 {npy_name}")
            continue

        print(f"\n  构建 {npy_name} ({desc}) [日频估值]...")
        matrix = np.full((n_stocks, n_dates), np.nan, dtype=np.float32)

        # [ITER11-PERF-VAL-VEC FIX] 向量化赋值：用 searchsorted 批量计算 t 索引，
        # 替代原 O(N×T) 双重 Python 循环（groupby+iterrows），速度提升约 100x。
        df_col = df_v[["code", "date", col]].dropna(subset=[col]).copy()
        df_col = df_col[df_col["code"].isin(code_to_idx)]
        if not df_col.empty:
            # 批量计算交易日索引
            t_idx = np.searchsorted(dates_arr, df_col["date"].values.astype("datetime64[ns]"), side="left")
            s_idx = df_col["code"].map(code_to_idx).values
            vals  = df_col[col].values.astype(np.float32)
            # 只写入范围内的数据
            valid = (t_idx >= 0) & (t_idx < n_dates)
            np.add.at(matrix, (s_idx[valid], t_idx[valid]), 0)  # 触发广播不赋值，仅确认形状
            matrix[s_idx[valid], t_idx[valid]] = vals[valid]

        # [ITER11-PERF-FFILL-VEC FIX] 向量化前向填充：替代原双重 Python 循环，速度提升约 50x。
        df_mat = pd.DataFrame(matrix.T)        # shape (T, N)
        df_mat.ffill(inplace=True)
        matrix = df_mat.values.T.astype(np.float32)

        np.save(str(out_path), matrix)
        valid_pct = np.sum(~np.isnan(matrix)) / matrix.size * 100
        print(f"    ✓ {npy_name}  有效率={valid_pct:.1f}%")


def _build_sue_npy(n_stocks, n_dates, df_merged, codes, code_to_idx, dates_arr, dates_dt):
    """
    [BUG-FUND-SUE-NEVER-BUILT FIX]
    从 fundamental_merged.csv 的 epsTTM + statDate + pubDate 计算 SUE：
      SUE_t = (EPS_t - EPS_{t-4Q}) / std(差值_最近8Q)
    前向填充到下次公告日。使用 quarter_key（YYYY-QN）精确匹配同季度。
    """
    out_path = NPY_DIR / "sue.npy"
    if out_path.exists():
        print("  ⚠ sue.npy 已存在，跳过")
        return
    if "epsTTM" not in df_merged.columns or "statDate" not in df_merged.columns:
        print("  ⚠ fundamental_merged.csv 缺少 epsTTM/statDate 列，跳过 sue.npy")
        return

    print("\n  构建 sue.npy [标准化未预期盈余]...")
    sue_arr = np.zeros((n_stocks, n_dates), dtype=np.float32)

    df_eps = df_merged[["code", "pubDate", "statDate", "epsTTM"]].dropna(
        subset=["epsTTM", "statDate"]
    ).copy()
    df_eps["pubDate"] = pd.to_datetime(df_eps["pubDate"], errors="coerce")
    df_eps = df_eps.dropna(subset=["pubDate"])

    def _quarter_key(stat_str: str) -> str:
        try:
            d = pd.to_datetime(stat_str[:10])
            qn = (d.month - 1) // 3 + 1
            return f"{d.year}-Q{qn}"
        except Exception:
            return ""

    for code, grp in df_eps.groupby("code"):
        if code not in code_to_idx:
            continue
        si = code_to_idx[code]
        grp = grp.sort_values("pubDate").reset_index(drop=True)

        eps_by_qkey = {}
        pub_list = grp["pubDate"].values
        eps_list = grp["epsTTM"].values.astype(float)
        stat_list = grp["statDate"].values

        for idx in range(len(grp)):
            qkey = _quarter_key(str(stat_list[idx]))
            eps_by_qkey[qkey] = float(eps_list[idx])

        qkeys = sorted(eps_by_qkey.keys())
        # 计算差值序列
        # [FIX-S-02] point-in-time SUE 标准化：每次公告时只用截至当时的历史波动率，
        # 消除原版"用全量 diffs[-8:] 标准化"引入的 look-forward normalization 偏差。
        # 原版：先收集所有差值 → 再统一标准化（未来信息渗入早期公告的 std）
        # 修复：按公告时间顺序，每次用已知的 diffs[:n] 计算 std，再标准化当次 diff

        # 预先收集所有差值（按 pubDate 排序），但计算时按序累积
        all_diffs_ordered = []
        for i, qk in enumerate(qkeys):
            y, q = qk.split("-Q")
            prev_qk = f"{str(int(y)-1)}-Q{q}"
            if prev_qk in eps_by_qkey:
                all_diffs_ordered.append(eps_by_qkey[qk] - eps_by_qkey[prev_qk])
            else:
                all_diffs_ordered.append(None)

        if sum(d is not None for d in all_diffs_ordered) < 2:
            continue

        # 将 SUE 前向填充到矩阵（point-in-time 标准化）
        seen_diffs = []
        for idx in range(len(grp)):
            qkey = _quarter_key(str(stat_list[idx]))
            y, q = qkey.split("-Q") if qkey else ("", "")
            if not y:
                continue
            prev_qk = f"{str(int(y)-1)}-Q{q}"
            if prev_qk not in eps_by_qkey:
                continue
            diff = eps_by_qkey[qkey] - eps_by_qkey[prev_qk]
            seen_diffs.append(diff)

            # [FIX-S-02] 只用本次公告「之前」已知的差值计算 std
            hist = seen_diffs[:-1]  # 不含当次（避免自参考）
            if len(hist) < 2:
                std_val = 1.0  # 历史不足时用 1.0，不标准化
            else:
                std_val = float(np.std(hist[-8:])) or 1.0
            sue_val = float(np.clip(diff / std_val, -10.0, 10.0))

            t_start = int(np.searchsorted(dates_arr, pub_list[idx], side="left"))
            if t_start >= n_dates:
                continue
            t_end = n_dates if idx + 1 >= len(grp) else int(
                np.searchsorted(dates_arr, pub_list[idx + 1], side="left")
            )
            sue_arr[si, t_start:min(t_end, n_dates)] = sue_val

    np.save(str(out_path), sue_arr)
    nonzero = np.sum(sue_arr != 0) / sue_arr.size * 100
    print(f"    ✓ sue.npy  非零率={nonzero:.1f}%")


def _build_days_ann_npy(n_stocks, n_dates, df_merged, codes, code_to_idx, dates_arr):
    """
    [BUG-FUND-DAYS_ANN-NEVER-BUILT FIX]
    从 pubDate 计算每个交易日距最近公告日的天数（前向填充）。
    days_ann[i, t] = t - t_last_pub，单位为交易日数，上限 32767（int16 范围）。
    """
    out_path = NPY_DIR / "days_ann.npy"
    if out_path.exists():
        print("  ⚠ days_ann.npy 已存在，跳过")
        return

    print("\n  构建 days_ann.npy [距最近公告交易日数]...")
    days_arr = np.zeros((n_stocks, n_dates), dtype=np.int16)

    df_pub = df_merged[["code", "pubDate"]].copy()
    df_pub["pubDate"] = pd.to_datetime(df_pub["pubDate"], errors="coerce")
    df_pub = df_pub.dropna(subset=["pubDate"])

    # [ITER11-PERF-DAYS-VEC FIX] 向量化构建 days_ann：
    # 原实现用 O(N×T) Python 双重循环；改为 searchsorted 批量计算，速度提升约 50x。
    for code, grp in df_pub.groupby("code"):
        if code not in code_to_idx:
            continue
        si = code_to_idx[code]
        pub_dates_sorted = np.sort(grp["pubDate"].values.astype("datetime64[ns]"))
        # 批量计算每个公告日对应的交易日索引
        ann_t_arr = np.searchsorted(dates_arr, pub_dates_sorted, side="left")
        ann_t_arr = ann_t_arr[ann_t_arr < n_dates]
        if len(ann_t_arr) == 0:
            continue
        # 对每个交易日 t，找到最后一个 ann_t <= t 的公告索引
        # np.searchsorted 找到"插入位置-1"即为最后已过的公告
        t_all = np.arange(n_dates)
        pos = np.searchsorted(ann_t_arr, t_all, side="right") - 1  # 最后一个 <= t 的索引
        has_ann = pos >= 0
        last_ann_t = np.where(has_ann, ann_t_arr[np.maximum(pos, 0)], -1)
        days_since = np.where(has_ann, np.minimum(t_all - last_ann_t, 32767), 0)
        days_arr[si, :] = days_since.astype(np.int16)

    np.save(str(out_path), days_arr)
    valid_pct = np.sum(days_arr > 0) / days_arr.size * 100
    print(f"    ✓ days_ann.npy  有公告后天数={valid_pct:.1f}%")


def _build_market_cap(n_stocks: int, n_dates: int) -> None:
    close_path = NPY_DIR / "close.npy"
    total_share_path = NPY_DIR / "fundamental_total_share.npy"
    liq_share_path = NPY_DIR / "fundamental_liq_share.npy"

    # [BUG-FUND-MKTCAP-ADJ-PRICE-BIAS FIX]
    # 优先使用不复权收盘价（unadj_close.npy）计算市值，避免后复权累积分红导致
    # 早期历史市值系统性偏低的问题。若不复权文件不存在则降级使用 close.npy 并警告。
    unadj_close_path = NPY_DIR / "unadj_close.npy"
    if unadj_close_path.exists():
        close_path_to_use = unadj_close_path
        close_label = "不复权收盘价(unadj_close)"
    elif close_path.exists():
        close_path_to_use = close_path
        close_label = "后复权收盘价(close) ⚠ 早期市值偏低"
        print("    ✗ [CRITICAL ERROR][BUG-FUND-MKTCAP-ADJ-PRICE-BIAS] 未找到 unadj_close.npy！")
        print("      ╔══════════════════════════════════════════════════════════════════════╗")
        print("      ║  【严重警告】使用后复权价(close.npy)计算市值会导致历史市值严重失真！  ║")
        print("      ║  后复权价通过累积分红向前调整，早期历史价格被人为压低 30%~70%；       ║")
        print("      ║  这将使 2010-2018 年的市值被系统性低估，彻底摧毁 OLS 中性化效果，   ║")
        print("      ║  因子暴露在时间序列上产生系统性漂移，Fama-French 回归结果完全失效。  ║")
        print("      ║  ─────────────────────────────────────────────────────────────────  ║")
        print("      ║  解决方法：运行 step0_patch_daily_fields.py 生成 unadj_close.npy，  ║")
        print("      ║  然后重新执行本步骤。在修复之前，市值中性化因子不可信！               ║")
        print("      ╚══════════════════════════════════════════════════════════════════════╝")
    else:
        print("    ⚠ 未找到 close.npy / unadj_close.npy，跳过市值计算")
        return

    if not total_share_path.exists():
        print("    ⚠ 未找到 fundamental_total_share.npy，先运行总股本构建")
        return

    close = np.load(str(close_path_to_use), mmap_mode="r").astype(np.float32)
    total_share = np.load(str(total_share_path), mmap_mode="r").astype(np.float32)

    # [BUG-MKTCAP-ZERO-VALID-FIX] 检查 total_share 是否有有效数据。
    # 若 fundamental_total_share.npy 本身有效率=0%（Baostock 返回空值 / 单位映射失败），
    # close * NaN * 10000 / 1e8 = NaN → market_cap 同样有效率=0%，但不报任何错误。
    # 修复：提前检测，有效率<1% 时打印明确诊断，引导用户重新下载。
    _ts_valid_pct = float(np.sum(~np.isnan(total_share) & (total_share > 0)) / total_share.size * 100)
    if _ts_valid_pct < 1.0:
        print(f"    ✗ [MKTCAP-SOURCE-EMPTY] fundamental_total_share.npy 有效率={_ts_valid_pct:.1f}%")
        print(f"      → 市值无法计算。根因：Baostock query_profit_data 对大多数股票不返回")
        print(f"        totalShare/liqaShare 字段（返回空字符串 → NaN）。")
        print(f"      → 解决方法：重新用 AKShare 版下载（菜单5a），AKShare 有更完整的股本数据。")
        print(f"      → 或手动删除 fundamental_total_share.npy 后重运行步骤6，")
        print(f"        若 fundamental_merged.csv 中 totalShare 列仍全为NaN则需重下。")
        # 仍然写出全 NaN 的 npy（保持文件存在，避免后续步骤报"文件不存在"）
        out_total = NPY_DIR / "market_cap_total.npy"
        np.save(str(out_total), np.full((n_stocks, n_dates), np.nan, dtype=np.float32))
        out_circ  = NPY_DIR / "market_cap_circ.npy"
        np.save(str(out_circ),  np.full((n_stocks, n_dates), np.nan, dtype=np.float32))
        print(f"    ⚠ 已写出全NaN占位 market_cap_total/circ.npy，待股本数据修复后重建")
        return

    # 总市值(亿元) = close(元) × total_share(万股) × 10000 / 1e8
    print(f"    价格来源: {close_label}")
    with np.errstate(invalid="ignore"):
        total_mv = close * total_share * 10000.0 / 1e8

    out_total = NPY_DIR / "market_cap_total.npy"
    np.save(str(out_total), total_mv.astype(np.float32))
    v = np.sum(~np.isnan(total_mv) & (total_mv > 0)) / total_mv.size * 100
    print(f"    ✓ market_cap_total.npy (亿元)  有效率={v:.1f}%")

    if liq_share_path.exists():
        liq_share = np.load(str(liq_share_path), mmap_mode="r").astype(np.float32)
        with np.errstate(invalid="ignore"):
            circ_mv = close * liq_share * 10000.0 / 1e8
        out_circ = NPY_DIR / "market_cap_circ.npy"
        np.save(str(out_circ), circ_mv.astype(np.float32))
        v = np.sum(~np.isnan(circ_mv) & (circ_mv > 0)) / circ_mv.size * 100
        print(f"    ✓ market_cap_circ.npy  (亿元)  有效率={v:.1f}%")


if __name__ == "__main__":
    build_fundamental_npy()



