"""
scripts/check_fundamental_lookahead.py
========================================
检查基本面数据是否存在前视偏差（look-ahead bias）

A股财报披露规则:
  年报: 1月1日 ~ 4月30日
  一季报: 3月31日 ~ 4月30日
  半年报: 7月1日 ~ 8月31日
  三季报: 7月1日 ~ 10月31日

如果 step3 用 report_date（财报期末）而非 pub_date（发布日）对齐数据,
则策略提前 30~90 天使用了"未来信息"，回测收益虚高。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FUND_DIR = ROOT / "data" / "fundamental"


def check_lookahead():
    print("=" * 60)
    print("基本面数据前视偏差检查")
    print("=" * 60)
    print()

    # ── 1. 检查 fundamental_merged.csv ─────────────────────────────────
    f = FUND_DIR / "fundamental_merged.csv"
    if not f.exists():
        print("✗ fundamental_merged.csv 不存在")
        print("  请先执行: python scripts/step1_download_fundamental_akshare.py")
        return False

    try:
        df = pd.read_csv(f, nrows=2000)
    except Exception as e:
        print(f"✗ 读取失败: {e}"); return False

    print(f"  文件: {f}")
    print(f"  行数: {len(df)}（前2000行样本）")
    print(f"  列名: {df.columns.tolist()}")
    print()

    date_cols = [c for c in df.columns if any(
        k in c.lower() for k in ["date", "time", "period", "report", "pub"]
    )]
    print(f"  日期相关列: {date_cols}")
    print()

    # ── 2. 判断是否有 pub_date ─────────────────────────────────────────
    has_pub = any("pub" in c.lower() for c in df.columns)
    has_report = any(c.lower() in ["report_date", "end_date", "period_end"]
                     for c in df.columns)

    if not has_pub:
        print("⚠ 警告：未找到 pub_date（发布日）列")
        print()
        print("  → 基本面矩阵很可能使用了财报期末日期（report_date）")
        print("  → 这意味着策略在财报发布前就'看到'了财报数据")
        print()
        print("  影响的策略:")
        print("    titan_alpha_v1:  F2(EP/PE) + F3(ROE) + F4(SUE) 均受影响")
        print("    alpha_max_v5:    基本面因子受影响")
        print()
        print("  典型高估程度:")
        print("    年化收益: 可能虚高 3~8 个百分点")
        print("    夏普比率: 可能虚高 0.3~0.8")
        print()
        print("  修复方案:")
        print("    1. 重新下载包含 pub_date 的基本面数据")
        print("    2. 修改 step3_build_fundamental_npy.py 用 pub_date 对齐")
        print("    3. 重新构建 pe_ttm.npy / fundamental_roe.npy / sue.npy")
        return False

    # ── 3. 有 pub_date，分析延迟分布 ──────────────────────────────────
    pub_col = next(c for c in df.columns if "pub" in c.lower())
    rep_col = next((c for c in df.columns if c.lower() in ["report_date","end_date","period_end"]), None)

    if rep_col:
        try:
            df[pub_col] = pd.to_datetime(df[pub_col], errors="coerce")
            df[rep_col] = pd.to_datetime(df[rep_col], errors="coerce")
            df["delay"] = (df[pub_col] - df[rep_col]).dt.days
            delay_clean = df["delay"].dropna()
            delay_clean = delay_clean[(delay_clean >= 0) & (delay_clean < 365)]

            print(f"✓ 发现 pub_date 列: {pub_col}")
            print()
            print("  财报披露延迟统计（report_date → pub_date）:")
            print(f"    中位数:  {delay_clean.median():.0f} 天")
            print(f"    平均值:  {delay_clean.mean():.0f} 天")
            print(f"    最大值:  {delay_clean.max():.0f} 天")
            print(f"    90分位:  {delay_clean.quantile(0.9):.0f} 天")
            print()

            if delay_clean.median() > 60:
                print("  ⚠ 中位延迟 > 60 天，确认 step3 必须使用 pub_date 对齐")
            else:
                print("  ✓ 延迟分布正常")
        except Exception as e:
            print(f"  延迟计算失败: {e}")

    # ── 4. 检查 step3 实际使用的对齐方式 ─────────────────────────────
    step3 = ROOT / "scripts" / "step3_build_fundamental_npy.py"
    if step3.exists():
        src = step3.read_text(encoding="utf-8")
        uses_pub = "pub_date" in src
        print()
        print(f"  step3_build_fundamental_npy.py 使用:")
        if uses_pub:
            print("    ✓ pub_date（发布日）→ 无前视偏差")
        else:
            print("    ⚠ 未发现 pub_date 使用，可能用了 report_date")
            print("      请检查 step3 中的日期对齐逻辑")

    print()
    print("=" * 60)
    if has_pub:
        print("总体评估: 数据包含 pub_date，请确认 step3 正确使用")
        return True
    else:
        print("总体评估: ⚠ 存在前视偏差风险，建议修复后重新回测")
        return False


if __name__ == "__main__":
    ok = check_lookahead()
    sys.exit(0 if ok else 1)
