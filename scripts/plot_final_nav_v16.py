import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 路径修复
ROOT = Path(__file__).resolve().parent.parent
excel_path = ROOT / "results" / "complete_system_audit_v16.xlsx"
plot_path = ROOT / "results" / "v16_strategy_comparison_final.png"

def plot_v16_comparison():
    print(f"\n--- GENERATING V16 STRATEGY COMPARISON PLOT: {plot_path} ---")

    # 1. 加载 Excel 并获取 Sheet 列表
    xlsx = pd.ExcelFile(excel_path)
    sheets = xlsx.sheet_names

    # 2. 筛选策略审计页 (以 Audit_ 开头)
    strat_sheets = [s for s in sheets if s.startswith("Audit_")]

    plt.figure(figsize=(16, 9))

    # 3. 逐个提取 NAV
    for sname in strat_sheets:
        df = pd.read_excel(xlsx, sheet_name=sname)
        # 获取策略原名（从 summary 中匹配更准确，这里简单处理）
        label_name = sname.replace("Audit_", "")

        # 提取 NAV 数组
        nav = df['Engine_NAV_T'].values

        # 绘制曲线
        plt.plot(nav, label=label_name, alpha=0.8, linewidth=1.5)

    # 4. 美化图表
    plt.title("Q-UNITY V10 [V16-ULTIMATE] ENGINE REPAIRED: EQUITY CURVES", fontsize=16)
    plt.xlabel("Days (1200 days Synthetic Universe)", fontsize=12)
    plt.ylabel("Portfolio NAV (Initial: 1.0M)", fontsize=12)

    # 在 Day 600 画正中竖线 (HFQ Jump Point)
    plt.axvline(x=600, color='r', linestyle='--', alpha=0.4, label="HFQ Jump Point (Day 600)")

    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 5. 保存
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"SUCCESS: Comparison plot generated at {plot_path}")

if __name__ == "__main__":
    plot_v16_comparison()
