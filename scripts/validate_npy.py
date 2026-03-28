#!/usr/bin/env python3
"""
validate_npy.py — Q-UNITY V9 NPY 矩阵质量校验脚本

用法：
    python validate_npy.py --npy-dir ./data/npy

校验项目：
  1. NaN 计数 = 0（所有字段）
  2. 标的数量 N 在合理范围（4000 ~ 5500）
  3. amount.npy 存在且单位合理（中位数 > 1e6 元）
  4. OHLCV 逻辑完整性（high >= low，close > 0）
  5. 代码列表纯净性（无指数代码）

成功退出码：0；失败：1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def validate(npy_dir: str, verbose: bool = False) -> bool:
    """
    执行所有校验，返回 True=通过 / False=失败。
    """
    npy_path = Path(npy_dir)
    meta_path = npy_path / "meta.json"

    print(f"\n{'='*60}")
    print(f"  Q-UNITY V10 NPY 矩阵校验报告")
    print(f"  npy_dir: {npy_path.resolve()}")
    print(f"{'='*60}")

    # ── 加载 meta.json ──────────────────────────────────────────────────
    if not meta_path.exists():
        print(f"[FAIL] meta.json 不存在: {meta_path}")
        return False

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    N, T = meta["shape"]
    codes = meta.get("codes", [])
    fields = meta.get("fields", [])
    extra = meta.get("extra", {})

    print(f"\n[INFO] 矩阵规模: N={N}, T={T}")
    print(f"[INFO] 字段列表: {fields}")
    print(f"[INFO] 构建时间: {meta.get('build_time', 'unknown')}")

    all_pass = True

    # ── 校验 1: 标的数量 ───────────────────────────────────────────────
    print(f"\n[CHECK-1] 标的数量")
    if 4000 <= N <= 6500:
        print(f"  ✅ N={N}（在合理范围 4000~6500，V10 Union Universe 含退市股）")
    elif N < 3500:
        print(f"  ⚠️  N={N} < 4000，可能过滤掉了太多标的")
        all_pass = False
    else:
        print(f"  ⚠️  N={N} > 6000，可能包含非 A 股标的")
        all_pass = False

    # ── 校验 2: 代码纯净性（抽查是否有指数代码）──────────────────────
    print(f"\n[CHECK-2] 代码纯净性")
    index_like = [c for c in codes if (
        c.startswith("sh.000") or c.startswith("sz.399") or
        c.startswith("sh.399") or c.startswith("sh.880")
    )]
    if index_like:
        print(f"  ❌ 发现 {len(index_like)} 个疑似指数代码: {index_like[:5]}")
        all_pass = False
    else:
        print(f"  ✅ 未发现指数代码（共 {len(codes)} 只标的）")

    # ── 校验 3: NaN 计数 ───────────────────────────────────────────────
    print(f"\n[CHECK-3] NaN 计数（要求 = 0）")
    nan_from_meta = extra.get("nan_validation", {})
    nan_total     = extra.get("nan_total", -1)

    if nan_total == 0:
        print(f"  ✅ meta.json 记录 NaN_total={nan_total}")
    elif nan_total > 0:
        print(f"  ❌ meta.json 记录 NaN_total={nan_total}！{nan_from_meta}")
        all_pass = False
    else:
        print(f"  [WARN] meta.json 无 NaN 记录，实测验证中...")

    # 实测抽验（每个字段取 100 行采样）
    for field in ["close", "open", "high", "low", "volume", "amount"]:
        npy_file = npy_path / f"{field}.npy"
        if not npy_file.exists():
            if field == "amount":
                print(f"  ❌ amount.npy 不存在！（V9 必须生成 amount）")
                all_pass = False
            else:
                print(f"  [WARN] {field}.npy 不存在，跳过")
            continue

        arr = np.load(str(npy_file))
        nan_cnt = int(np.isnan(arr).sum())
        inf_cnt = int(np.isinf(arr).sum())

        if nan_cnt == 0 and inf_cnt == 0:
            print(f"  ✅ {field}.npy: NaN={nan_cnt}, Inf={inf_cnt}, shape={arr.shape}")
        else:
            print(f"  ❌ {field}.npy: NaN={nan_cnt}, Inf={inf_cnt} ← 数据污染！")
            all_pass = False

        if verbose:
            # 非零统计
            nonzero = arr[arr > 0]
            if len(nonzero) > 0:
                print(f"     非零统计: min={nonzero.min():.4g}, "
                      f"median={np.median(nonzero):.4g}, "
                      f"max={nonzero.max():.4g}")

    # ── 校验 4: amount 单位合理性 ─────────────────────────────────────
    print(f"\n[CHECK-4] amount 单位合理性")
    amt_file = npy_path / "amount.npy"
    if amt_file.exists():
        amt_arr = np.load(str(amt_file))
        amt_pos = amt_arr[amt_arr > 0]
        if len(amt_pos) == 0:
            print(f"  ❌ amount.npy 全为 0，无有效成交额数据！")
            all_pass = False
        else:
            median_val = float(np.median(amt_pos))
            if median_val >= 1e6:
                print(f"  ✅ amount 非零中位数 = {median_val:.2e} 元（≥1e6，单位正确）")
            elif median_val >= 1e2:
                print(
                    f"  ⚠️  amount 非零中位数 = {median_val:.2e}（疑似万元，"
                    "请检查是否已乘以 10000！）"
                )
                all_pass = False
            else:
                print(f"  ❌ amount 中位数 = {median_val:.2e}，数值异常！")
                all_pass = False
        meta_median = extra.get("amount_median_yuan", 0)
        if meta_median > 0:
            print(f"     meta 记录: amount_median_yuan={meta_median:.2e}")
    else:
        print(f"  ❌ amount.npy 不存在！")
        all_pass = False

    # ── 校验 5: OHLCV 逻辑完整性（抽样）─────────────────────────────
    print(f"\n[CHECK-5] OHLCV 逻辑完整性（抽样）")
    high_file  = npy_path / "high.npy"
    low_file   = npy_path / "low.npy"
    close_file = npy_path / "close.npy"
    if high_file.exists() and low_file.exists() and close_file.exists():
        high_arr  = np.load(str(high_file))
        low_arr   = np.load(str(low_file))
        close_arr = np.load(str(close_file))

        # 抽取有效交易日（volume > 0 的格子）
        has_trade = high_arr > 0
        hl_ok = np.all((high_arr[has_trade] >= low_arr[has_trade]))
        cl_ok = np.all((close_arr[has_trade] > 0))

        if hl_ok:
            print(f"  ✅ high >= low（有效格子中）")
        else:
            bad_cnt = int((high_arr[has_trade] < low_arr[has_trade]).sum())
            print(f"  ⚠️  {bad_cnt} 个格子出现 high < low（数据异常）")

        if cl_ok:
            print(f"  ✅ close > 0（有效格子中）")
        else:
            print(f"  ❌ 存在 close <= 0 的有效格子！")
            all_pass = False
    else:
        print(f"  [SKIP] OHLC 文件不全，跳过完整性检查")

    # ── 校验 6: 基本面 npy 检查 ────────────────────────────────────
    print(f"\n[CHECK-6] 基本面 npy 文件")
    # [FIX-CHECK6-ALIGN] 原 CHECK-6 只检查 step3 输出的 6 个基本面文件，
    # 但 main.py 数据加载后还会单独检查 pe_ttm.npy / sue.npy / market_index.npy，
    # 导致 validate_npy 报告"所有校验通过"，实际运行回测时仍弹出缺失警告——自相矛盾。
    # 修复：将 main.py 实际依赖的关键文件分为两类：
    #   REQUIRED_FILES: 缺失时回测质量明显下降，validate 输出 ❌ 并给出修复路径
    #   OPTIONAL_FILES: 仅统计质量检查，文件存在才验证内容，缺失不影响 all_pass
    #
    # [FIX-CHECK6-YOYNI] warn_only=True 的因子超范围仅打印 ⚠️，不影响 all_pass。
    # yoy_ni（净利润同比）在 A 股扭亏为盈、微利基期等场景下极易超出 [-1000,5000]，
    # 属正常数据分布特性，应在因子使用层 clip/winsorize，不应在数据校验层判定失败。

    # ── 6A: 必要文件（main.py 会明确警告缺失的文件）─────────────────
    # 来源说明：
    #   pe_ttm.npy       → 4c/4d(AKShare)→5，或 4a(TdxQuant)→5，或 4e(BaoStock,别名)
    #   sue.npy          → 数据管理 4(TdxQuant基本面) → 5(构建npy)
    #   market_index.npy → 数据管理 2/3(日线下载+构建npy) 自动生成
    #
    # [FIX-CHECK6-PE-ALIAS] pe_ttm.npy 和 valuation_peTTM.npy 功能等价：
    # fast_runner FUND_MAP 优先读 pe_ttm，找不到则 fallback 到 valuation_peTTM。
    # 4e(BaoStock) 直接生成 valuation_peTTM.npy，之前只检查 pe_ttm.npy 导致
    # 4e 用户永远看到 ❌ 误报。此处改为二选一均视为满足。

    # 格式：key=展示名, value=(候选文件列表, 描述, 用途, 修复提示)
    required_files = {
        "pe_ttm": (
            ["pe_ttm.npy", "valuation_peTTM.npy"],
            "PE估值TTM",
            "titan_alpha/alpha_max F2因子",
            "4a(TdxQuant估值)→5，或 4c/4d(AKShare)→5，或 4e(BaoStock直接生成)"
        ),
        "sue": (
            ["sue.npy"],
            "SUE超预期",
            "titan_alpha F4因子",
            "数据管理→4(TdxQuant基本面)→5"
        ),
        "market_index": (
            ["market_index.npy"],
            "市场指数",
            "Regime牛熊检测（关键）",
            "数据管理→3 重新构建npy"
        ),
    }
    missing_required = []
    for key, (aliases, desc, usage, fix_hint) in required_files.items():
        found_file = next((a for a in aliases if (npy_path / a).exists()), None)
        if found_file is None:
            display = " 或 ".join(aliases)
            print(f"  ❌ {display}({desc})  用途:{usage}")
            print(f"       修复: {fix_hint}")
            missing_required.append(key)
            all_pass = False
        else:
            arr = np.load(str(npy_path / found_file), mmap_mode="r")
            print(f"  ✅ {found_file}({desc}): shape={arr.shape} ✓")

    if missing_required:
        print(f"  ⚠️  {len(missing_required)} 个必要文件缺失"
              f"——validate_npy 若报告通过但此处有 ❌，则回测时仍会出现警告，请按修复提示补全数据。")

    # ── 6B: 可选基本面文件（存在时验证内容质量）───────────────────────
    print(f"  ─── 可选基本面（存在时验证）")
    fund_files = {
        "fundamental_roe.npy":              ("ROE",       -100, 200,   False),
        "fundamental_eps_ttm.npy":          ("EPS_TTM",   -100, 1000,  False),
        "fundamental_net_profit_margin.npy":("净利率",    -500, 100,   False),
        "fundamental_yoy_ni.npy":           ("净利润同比", -1000, 5000, True),   # warn_only
        "fundamental_total_share.npy":      ("总股本",    0,    1e8,   False),
        "market_cap_total.npy":             ("总市值亿",  0,    5e5,   False),
    }
    any_fund = False
    for fname, (desc, vmin, vmax, warn_only) in fund_files.items():
        fp = npy_path / fname
        if not fp.exists():
            continue
        any_fund = True
        arr = np.load(str(fp), mmap_mode="r").astype(np.float32)
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            print(f"  ⚠️  {fname}: 全为 NaN（未构建？）")
            continue
        coverage = len(valid) / arr.size * 100
        out_of_range = int(((valid < vmin) | (valid > vmax)).sum())
        if out_of_range > len(valid) * 0.01:
            note = "（仅告警，A股正常分布）" if warn_only else ""
            print(f"  ⚠️  {fname}({desc}): 覆盖率={coverage:.1f}%  "
                  f"超范围[{vmin},{vmax}]={out_of_range}个({out_of_range/len(valid)*100:.1f}%){note}")
            if not warn_only:
                all_pass = False
        else:
            print(f"  ✅ {fname}({desc}): 覆盖率={coverage:.1f}%  "
                  f"中位={float(np.median(valid)):.3g}  超范围={out_of_range}个")
    if not any_fund:
        print(f"  [SKIP] 未找到可选基本面 npy，跳过（需先运行步骤 4→5）")

    # ── 校验 7: point-in-time 完整性（pubDate对齐检查）──────────────
    # [FIX-CHECK7] 原逻辑检查"矩阵前30列"并假设为NaN，但V10 Union Universe
    # 含大量2015年前已上市股票（季报pubDate早于矩阵起点），forward-fill后
    # 前30列有值完全正确，导致误报。
    # 修复：改为抽样验证——对每只股票，找其第一个非NaN格子t，
    # 若t>0且t-1格也非NaN → 说明基本面在无公告日情况下就有值 → 真正的前视泄露。
    print(f"\n[CHECK-7] 基本面 point-in-time 对齐")
    roe_path = npy_path / "fundamental_roe.npy"
    if roe_path.exists():
        roe = np.load(str(roe_path), mmap_mode="r")
        sample_n = min(300, N)
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(N, sample_n, replace=False)
        violations = 0
        checked = 0
        for si in sample_idx:
            row = np.array(roe[si])
            nz = np.where(~np.isnan(row))[0]
            if len(nz) < 2:
                continue
            checked += 1
            # 若两个相邻的非NaN格之间没有任何NaN间隔，
            # 说明存在"从第0格起就有连续值"的情况（statDate对齐特征）
            # 更精准：检查最早非NaN格的前一格是否也非NaN（不可能是pubDate对齐）
            first_t = nz[0]
            if first_t > 0 and not np.isnan(row[first_t - 1]):
                violations += 1
        if checked == 0:
            print(f"  [SKIP] 抽样股票均无基本面数据，跳过")
        else:
            violation_rate = violations / checked
            if violation_rate > 0.05:
                print(f"  ⚠️  PIT 对齐异常：抽样{checked}只中{violations}只存在前视"
                      f"（违规率={violation_rate:.1%}，疑似 statDate 对齐）")
                all_pass = False
            else:
                print(f"  ✅ PIT 对齐正常（抽样{checked}只，违规率={violation_rate:.1%}，pubDate对齐正确）")
    else:
        print(f"  [SKIP] fundamental_roe.npy 不存在")


    # ── 校验 8: V10 adj_type = "hfq" ────────────────────────────────────
    print(f"\n[CHECK-8] V10 复权类型（要求 adj_type=hfq）")
    adj_type = meta.get("adj_type", "")
    if adj_type == "hfq":
        print(f"  ✅ adj_type='{adj_type}' (符合 V10 工业标准) ✓")
    elif adj_type == "qfq":
        print(f"  ❌ 严重逻辑错误：检测到 adj_type='qfq'！")
        print(f"       V10 引擎必须使用后复权(HFQ)以确保物理真实性，前复权会导致回测失效。")
        all_pass = False
    elif adj_type:
        print(f"  ⚠️  adj_type='{adj_type}'，V10 建议使用 'hfq'")
        all_pass = False
    else:
        print(f"  ⚠️  meta.json 无 adj_type 字段（请确认是否为 V10 数据）")

    # ── 校验 8b: 物理价格尺度校验 (防伪装) ─────────────────────────
    close_file = npy_path / "close.npy"
    if close_file.exists():
        c_arr = np.load(str(close_file))
        c_median = float(np.median(c_arr[c_arr > 0]))
        # 工业逻辑：A股全市场 HFQ 中位数通常在 150-300 元以上（受茅台等老股拉动）
        # 如果中位数在 10 元左右，必然是 QFQ
        if adj_type == "hfq" and c_median < 40.0:
            print(f"  ❌ [PRICE-SCALE-FAILURE] 物理价格尺度异常！")
            print(f"       meta 声明为 hfq，但价格中位数仅为 {c_median:.2f} 元。")
            print(f"       这极有可能是用前复权数据伪造的后复权标记，会导致回测产生暴力未来函数！")
            all_pass = False
        elif adj_type == "hfq":
            print(f"  ✅ 物理价格尺度核验通过 (Median={c_median:.2f} 元) ✓")

    # ── 校验 9: V10 market_index.npy ───────────────────────────────────
    print(f"\n[CHECK-9] V10 市场指数 market_index.npy")
    mi_path = npy_path / "market_index.npy"
    if mi_path.exists():
        mi = np.load(str(mi_path))
        if mi.shape[0] == 1 and mi.shape[1] == T:
            print(f"  ✅ market_index.npy shape={mi.shape} (1×T={T}) ✓")
        else:
            print(f"  ⚠️  market_index.npy shape={mi.shape}，期望 (1, {T})")
    else:
        print(f"  ⚠️  market_index.npy 不存在，请运行 step0 --build-npy")

    # ── 校验 10: valid_mask.npy 覆盖率 ──────────────────────────────────
    # [FIX-CHECK10] 原阈值上限 70% 过保守。V10 Union Universe（含退市股历史数据）
    # 且矩阵起点 2015-01-05 时大多数股票已满上市60天，正常覆盖率可达 70~75%。
    # 修复：上限调整为 80%。
    print(f"\n[CHECK-10] valid_mask.npy 覆盖率")
    vm_path = npy_path / "valid_mask.npy"
    if vm_path.exists():
        vm = np.load(str(vm_path))
        vm_rate = float(vm.mean())
        if 0.10 <= vm_rate <= 0.80:
            print(f"  ✅ valid_mask 覆盖率={vm_rate:.1%}（合理范围10%~80%）")
        else:
            print(f"  ⚠️  valid_mask 覆盖率={vm_rate:.1%}（超出合理范围10%~80%，请检查 valid_mask 构建）")
            all_pass = False
    else:
        print(f"  ⚠️  valid_mask.npy 不存在")

    # ── 汇总 ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if all_pass:
        print(f"  🎉 所有校验通过！NPY 矩阵质量合格。")
        print(f"     N={N} | T={T} | NaN=0 | amount.npy ✓")
        print(f"     pe_ttm.npy ✓ | sue.npy ✓ | market_index.npy ✓")
    else:
        # [FIX-CHECK6-ALIGN] 区分 OHLCV 失败 vs 基本面缺失，给出不同提示
        ohlcv_pass = not any(
            f in str(missing_required) for f in ["close", "open", "high", "low", "volume"]
        ) if 'missing_required' in dir() else True
        if missing_required:
            print(f"  ⚠️  OHLCV 矩阵完整，但以下回测必要文件缺失：")
            for f in missing_required:
                print(f"     · {f}")
            print(f"  → 请按上方 ❌ 项的修复提示补全后重新运行验证。")
            print(f"  → 补全前运行回测会收到缺失警告，相关因子退化为0。")
        else:
            print(f"  ❌ 存在校验失败项，请检查上方错误信息！")
    print(f"{'='*60}\n")

    return all_pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Q-UNITY V9 NPY 矩阵质量校验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--npy-dir",
        type=str,
        default="./data/npy_v10",
        help="NPY 文件目录（默认 ./data/npy）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="显示详细统计信息",
    )
    args = parser.parse_args()

    ok = validate(args.npy_dir, verbose=args.verbose)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())



