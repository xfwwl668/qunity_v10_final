#!/usr/bin/env python3
"""
快速开始指南 - 5 分钟快速上手白盒审计框架

此脚本演示如何快速运行审计框架的核心功能
"""

import sys
from pathlib import Path

def main():
    project_root = Path(__file__).parent.parent
    
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║     Q-UNITY V10 白盒审计框架 - 快速开始指南                        ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝

【本框架的目的】
───────────────────────────────────────────────────────────────────
通过 "逆向验证法" 深入审计每个策略的买卖逻辑和数据一致性。

核心特性：
  ✓ 生成 1500+ 天的合成数据（3+ 正弦波）
  ✓ 逐日追踪买卖信号和持仓变化
  ✓ 手算因子值与代码计算对比
  ✓ 检测隐含的前视偏差
  ✓ 验证复权数据的一致性


【快速开始】
───────────────────────────────────────────────────────────────────

第一步：检查系统配置（2 分钟）
────────────────────────────────

  $ python whitebox_audit/diagnose.py

输出示例：
  ✓ Python 环境检查
  ✓ 审计框架文件检查
  ✓ 策略文件检查
  ✓ 引擎文件检查
  ✓ 数据文件检查
  ✓ 输出目录检查
  
✅ 所有检查通过！系统准备就绪。


第二步：运行演示程序（2 分钟）
────────────────────────────────

  $ python whitebox_audit/demo_quick_audit.py

此脚本演示：
  1. 生成可预测的合成数据（后复权）
  2. 验证复权数据的正确性
  3. 演示因子验证流程
  4. 展示前视偏差检测
  5. 模拟交易追踪

输出：
  - 数据样本（5 行开始，5 行结束）
  - 因子验证表格
  - 交易追踪记录
  - 前视检测结果


第三步：运行完整审计（1-5 分钟）
────────────────────────────────────

  基础命令（运行所有策略）：
  $ python -m whitebox_audit.run_whitebox_audit

  自定义参数：
  $ python -m whitebox_audit.run_whitebox_audit \\
      --days 2000 \\
      --cycles 3 \\
      --strategies kunpeng_v10 snma_v4 \\
      --skip-lookahead  # 如果想加速

参数说明：
  --days              合成数据长度（默认 1500）
  --cycles            正弦波数量（默认 3）
  --strategies        指定要审计的策略（默认全部）
  --skip-lookahead    跳过前视检测以加速


【输出文件】
───────────────────────────────────────────────────────────────────

运行后会在 whitebox_audit/outputs/ 生成：

  📁 whitebox_audit/outputs/
  ├─ synthetic_npy/
  │  ├─ synthetic_market_1500d.npy       ← 合成数据
  │  ├─ synthetic_metadata.json
  │  └─ price_chart.png                  ← 价格图表（3+ 正弦波）
  ├─ strategy_reports/
  │  ├─ kunpeng_v10_audit.xlsx          ← 各策略详细报告
  │  ├─ snma_v4_audit.xlsx
  │  ├─ titan_orthogonal_audit.xlsx
  │  └─ ...
  ├─ audit_summary.xlsx                  ← 全策略汇总对比
  ├─ lookahead_report.txt                ← 前视检测结果
  ├─ factor_audit_log.csv                ← 因子验证详情
  └─ debug/
     ├─ trade_trace_kunpeng.csv          ← 交易追踪明细
     ├─ trade_trace_snma.csv
     └─ ...


【如何查看报告】
───────────────────────────────────────────────────────────────────

1️⃣  汇总报告（最重要）
   📄 whitebox_audit/outputs/audit_summary.xlsx
   
   这个 Excel 包含：
   • Summary: 每个策略的总体评分
   • Factors: 因子验证结果（ PASS/WARN/FAIL）
   • Lookahead: 前视风险等级（绿/黄/红）
   • Comparison: 策略对比表

2️⃣  策略详细报告
   📄 whitebox_audit/outputs/strategy_reports/*.xlsx
   
   每个策略的报告包含：
   • 基本信息: 因子数量、权重方式等
   • 因子验证: 每个因子的手算 vs 代码计算
   • 前视检测: 源码分析 + 因果性检验
   • 交易明细: 前 100 笔交易的完整记录
   • 建议: 改进意见

3️⃣  前视检测详情
   📄 whitebox_audit/outputs/lookahead_report.txt
   
   包含：
   • 源码风险评分
   • 因果性测试结果 (IC_forward vs IC_backward)
   • 具体的可疑代码位置
   • 改进建议


【如何解读报告】
───────────────────────────────────────────────────────────────────

✅ PASS 代表：
   • 买卖逻辑正确
   • 没有前视偏差
   • 数据使用规范
   
⚠️  WARN 代表：
   • 可能有轻微问题
   • 需要进一步检查
   • 例如: EMA 参数不一致
   
❌ FAIL 代表：
   • 存在明显问题
   • 需要立即修复
   • 例如: 使用了未来数据


【常见问题】
───────────────────────────────────────────────────────────────────

Q: 运行变慢？
A: 使用 --skip-lookahead 跳过前视检测
   $ python -m whitebox_audit.run_whitebox_audit --skip-lookahead

Q: 只想审计某个策略？
A: 使用 --strategies 参数
   $ python -m whitebox_audit.run_whitebox_audit \\
       --strategies kunpeng_v10

Q: 合成数据不合理？
A: 检查 synthetic_npy/price_chart.png，应该看到 3+ 个波形

Q: 因子验证失败？
A: 检查 factor_audit_log.csv，找出具体的不匹配项

Q: 前视检测结果可疑？
A: 查看 lookahead_report.txt 中的"可疑代码位置"


【进阶用法】
───────────────────────────────────────────────────────────────────

1. 仅生成合成数据（不运行策略）：
   $ python -m whitebox_audit.synthetic_data_generator \\
       --output synthetic_npy --days 2000

2. 单独验证某个策略的因子：
   $ python -m whitebox_audit.factor_verification \\
       --strategy kunpeng_v10 \\
       --data synthetic_npy/synthetic_market_1500d.npy

3. 快速前视检测（仅静态分析）：
   $ python -m whitebox_audit.lookahead_detector \\
       --quick-mode --strategy snma_v4

4. 生成自定义场景数据（熊市、震荡市等）：
   在 synthetic_data_generator.py 中修改 scenario 参数
   - 'bull': 上升趋势
   - 'bear': 下降趋势
   - 'range': 横盘震荡


【下一步行动】
───────────────────────────────────────────────────────────────────

1. ✓ 运行诊断: diagnose.py
2. ✓ 看演示: demo_quick_audit.py
3. ✓ 运行完整审计: run_whitebox_audit.py
4. ✓ 查看报告: audit_summary.xlsx
5. ✓ 根据建议修改策略
6. ✓ 重新审计验证修改效果

【技术文档】
───────────────────────────────────────────────────────────────────

• 用户指南: README.md
• 架构设计: ARCHITECTURE.md
• 审计模板: audit_template.py
• API 文档: 各模块的 docstring

【支持】
───────────────────────────────────────────────────────────────────

如遇问题，请：
1. 检查 README.md 的故障排查部分
2. 查看 ARCHITECTURE.md 理解设计原理
3. 运行 diagnose.py 检查系统配置
4. 查看具体模块的源代码注释


═══════════════════════════════════════════════════════════════════

现在开始：
  $ python whitebox_audit/diagnose.py
  $ python whitebox_audit/demo_quick_audit.py
  $ python -m whitebox_audit.run_whitebox_audit

祝审计顺利！

═══════════════════════════════════════════════════════════════════
    """)


if __name__ == "__main__":
    main()
