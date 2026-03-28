from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging

# ─────────────────────────────────────────────────────────────────────────────
# 配置与路径
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARQUET_DIR  = PROJECT_ROOT / "data" / "daily_parquet_hfq"
NPY_DIR      = PROJECT_ROOT / "data" / "npy_v10"

# 审计采样标的 (包含蓝筹和微盘股)
SAMPLE_CODES = ["600519", "000001", "300750", "600000", "002624"]

def _c(t, col="\033[94m"): return f"{col}{t}\033[0m"

def audit_report():
    print(_c("\n" + "═"*70))
    print(_c("   Q-UNITY V10 黄金数据交叉审计报告 (HFQ Data Reconciliation)"))
    print(_c("═"*70))

    # 1. 基础环境校验
    if not PARQUET_DIR.exists():
        print(" [✗] 错误: hfq 数据目录不存在，请先执行 DataMenu 选项 [2/2a]")
        return

    # 2. 抽样价格一致性审计
    results = []
    print(f"\n [STEP 1] 正在审计核心标的价格序列 (采样点: 最新/一年前/两年前)...")
    for code in SAMPLE_CODES:
        # 宽搜文件
        matches = list(PARQUET_DIR.glob(f"*{code}*.parquet"))
        if not matches:
            print(f" [!] 警告: 未在 hfq 目录中找到代码 {code}")
            continue

        df = pd.read_parquet(matches[0])
        if df.empty: continue

        # 提取关键点价格
        latest_px = df['close'].iloc[-1]
        hist_1y_px = df['close'].iloc[-250] if len(df) > 250 else np.nan

        # 修正检测：检查 adj_type
        adj_tag = str(df['adj_type'].iloc[0]) if 'adj_type' in df.columns else "unknown"

        results.append({
            "代码": code,
            "最新HFQ价": f"{latest_px:.2f}",
            "一年前HFQ价": f"{hist_1y_px:.2f}",
            "复权标签": adj_tag,
            "文件": matches[0].name
        })

    if results:
        print(pd.DataFrame(results).to_string(index=False))
    else:
        print(" [✗] 无法进行价格审计：没有可用的 Parquet 数据")

    # 3. 基本面与时间轴对齐审计
    print(f"\n [STEP 2] 正在进行因子时间轴交叉审计...")
    meta_path = NPY_DIR / "meta.json"
    if meta_path.exists():
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        dates = meta.get("dates", [])
        codes_all = meta.get("codes", [])
        if dates:
            print(f"  ✓ 矩阵元数据存在: {len(codes_all)}只 × {len(dates)}天")
            print(f"  ✓ 矩阵起始日期: {dates[0]}  截止日期: {dates[-1]}")

            # 异常值检测：检测 0 或 nan 比例
            print(f"\n [STEP 3] 异常值与空洞审计 (Data Gap Analysis)...")
            close_npy = NPY_DIR / "close.npy"
            if close_npy.exists():
                arr = np.load(str(close_npy), mmap_mode='r')
                nan_cnt = np.isnan(arr).sum()
                zero_cnt = (arr == 0).sum()
                total = arr.size
                print(f"  · 价格矩阵 NaN 比例: {nan_cnt/total:.2%}")
                print(f"  · 价格矩阵 Zero 比例: {zero_cnt/total:.2%}")
                if zero_cnt/total > 0.05:
                    print(_c("  [⚠] 风险提示：价格矩阵 0 值比例过高，请检查除权转换逻辑！", "\033[93m"))
            else:
                print("  · 提示: close.npy 尚未生成，跳过矩阵空洞审计")
    else:
        print("  · 提示: meta.json 不存在，请执行 [3] 构建 npy 矩阵")

    print(_c("\n" + "═"*70))
    print(" 审计建议：如果 [STEP 1] 的复权标签不是 hfq，请立即清理 data/ 目录并重下。")
    print(" 审计建议：如果最新 HFQ 价与实盘软件偏差 >1%，请更换数据源 (推荐 BaoStock)。")
    print(_c("═"*70 + "\n"))

if __name__ == "__main__":
    audit_report()
