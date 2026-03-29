#!/usr/bin/env python3
"""
Q-UNITY V10 — 白盒审计主入口
============================

使用方法:
    python -m whitebox_audit.run_whitebox_audit
    python whitebox_audit/run_whitebox_audit.py

审计内容:
1. 生成合成数据（后复权，3+正弦波，1500D+）
2. 逐策略运行并验证买卖逻辑
3. 检测前视偏差
4. 生成详细 Excel 报告

输出目录:
    results/whitebox_audit_YYYYMMDD_HHMMSS/
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from datetime import datetime

# 确保项目根目录在 Python 路径中
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.chdir(ROOT)


def main():
    """白盒审计主入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Q-UNITY V10 白盒审计套件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="输出目录（默认: results/whitebox_audit_TIMESTAMP）",
    )
    parser.add_argument(
        "--days", "-T",
        type=int,
        default=1500,
        help="合成数据天数（默认: 1500）",
    )
    parser.add_argument(
        "--stocks", "-N",
        type=int,
        default=100,
        help="股票数量（默认: 100）",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=3,
        help="正弦波周期数（默认: 3）",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=None,
        help="指定审计的策略（默认: 全部）",
    )
    parser.add_argument(
        "--skip-lookahead",
        action="store_true",
        help="跳过前视偏差检测（加速运行）",
    )
    parser.add_argument(
        "--data-mode",
        type=str,
        choices=["sinusoidal", "trending_up", "trending_down", "mean_revert", "mixed"],
        default="sinusoidal",
        help="合成数据模式（默认: sinusoidal）",
    )
    
    args = parser.parse_args()
    
    # 设置输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = ROOT / "results" / f"whitebox_audit_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Q-UNITY V10 白盒审计套件")
    print("=" * 70)
    print(f"  输出目录: {output_dir}")
    print(f"  数据天数: {args.days}")
    print(f"  股票数量: {args.stocks}")
    print(f"  正弦周期: {args.cycles}")
    print(f"  数据模式: {args.data_mode}")
    print("=" * 70)
    
    # =========================================================================
    # Step 1: 生成合成数据
    # =========================================================================
    print("\n[Step 1] 生成合成数据...")
    
    from whitebox_audit.synthetic_data_generator import SyntheticDataGenerator
    
    gen = SyntheticDataGenerator(
        N=args.stocks,
        T=args.days,
        seed=42,
        base_price=10.0,
        volatility=0.02,
    )
    
    if args.data_mode == "sinusoidal":
        data = gen.generate_sinusoidal(cycles=args.cycles)
    elif args.data_mode == "trending_up":
        data = gen.generate_trending(direction="up")
    elif args.data_mode == "trending_down":
        data = gen.generate_trending(direction="down")
    elif args.data_mode == "mean_revert":
        data = gen.generate_mean_reverting()
    elif args.data_mode == "mixed":
        data = gen.generate_mixed_regime()
    else:
        data = gen.generate_sinusoidal(cycles=args.cycles)
    
    # 保存合成数据
    npy_dir = output_dir / "synthetic_npy"
    gen.save_as_npy(data, npy_dir)
    print(f"  - 数据已保存: {npy_dir}")
    print(f"  - 价格范围: [{data['close'].min():.2f}, {data['close'].max():.2f}]")
    print(f"  - 日期范围: {gen.dates[0]} ~ {gen.dates[-1]}")
    
    # =========================================================================
    # Step 2: 加载策略
    # =========================================================================
    print("\n[Step 2] 加载策略...")
    
    from src.strategies.registry import VEC_STRATEGY_REGISTRY, _auto_discover
    _auto_discover()
    
    if args.strategies:
        strategies = [s for s in args.strategies if s in VEC_STRATEGY_REGISTRY]
        if not strategies:
            print(f"  [ERROR] 指定的策略不存在: {args.strategies}")
            print(f"  可用策略: {list(VEC_STRATEGY_REGISTRY.keys())}")
            return 1
    else:
        strategies = list(VEC_STRATEGY_REGISTRY.keys())
    
    print(f"  - 将审计 {len(strategies)} 个策略:")
    for name in strategies:
        print(f"    - {name}")
    
    # =========================================================================
    # Step 3: 运行策略审计
    # =========================================================================
    print("\n[Step 3] 运行策略审计...")
    
    from whitebox_audit.strategy_audit_runner import StrategyAuditRunner
    
    runner = StrategyAuditRunner(
        data={k: v for k, v in data.items() if not k.startswith("_")},
        dates=gen.dates,
        codes=gen.codes,
    )
    
    audit_results = {}
    
    for name in strategies:
        print(f"\n  审计: {name}")
        try:
            fn = VEC_STRATEGY_REGISTRY[name]
            result = runner.audit_strategy(fn, name)
            audit_results[name] = result
            
            status_icon = "[PASS]" if not result.issues else "[WARN]"
            print(f"    {status_icon} 权重归一化: {'OK' if result.weights_normalized else 'FAIL'}")
            print(f"    {status_icon} 信号时序: {'OK' if result.signal_timing_correct else 'FAIL'}")
            
            if result.metrics:
                print(f"    - 年化收益: {result.metrics.get('年化收益', 0):.2%}")
                print(f"    - 夏普比率: {result.metrics.get('夏普比率', 0):.2f}")
                print(f"    - 最大回撤: {result.metrics.get('最大回撤', 0):.2%}")
            
            if result.issues:
                print(f"    - 发现 {len(result.issues)} 个问题:")
                for issue in result.issues[:3]:
                    print(f"      ! {issue[:80]}")
                    
        except Exception as e:
            print(f"    [ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    # 导出报告
    runner.export_report(output_dir / "strategy_reports")
    
    # =========================================================================
    # Step 4: 前视偏差检测
    # =========================================================================
    if not args.skip_lookahead:
        print("\n[Step 4] 前视偏差检测...")
        
        from whitebox_audit.lookahead_detector import LookAheadBiasDetector
        
        detector = LookAheadBiasDetector(verbose=True)
        test_data = {k: v for k, v in data.items() if not k.startswith("_")}
        
        for name in strategies:
            print(f"  检测: {name}")
            try:
                fn = VEC_STRATEGY_REGISTRY[name]
                violations = detector.detect_in_strategy(fn, name, test_data)
                
                if violations:
                    high = sum(1 for v in violations if v.severity == "HIGH")
                    print(f"    [WARN] 发现 {len(violations)} 个潜在前视偏差 (HIGH: {high})")
                else:
                    print(f"    [PASS] 未发现明显前视偏差")
                    
            except Exception as e:
                print(f"    [ERROR] 检测失败: {e}")
        
        # 保存报告
        report = detector.generate_report()
        report_path = output_dir / "lookahead_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n  前视偏差报告: {report_path}")
    else:
        print("\n[Step 4] 跳过前视偏差检测")
    
    # =========================================================================
    # Step 5: 生成汇总报告
    # =========================================================================
    print("\n[Step 5] 生成汇总报告...")
    
    import pandas as pd
    
    summary_data = []
    for name, result in audit_results.items():
        summary_data.append({
            "策略": name,
            "权重归一化": "PASS" if result.weights_normalized else "FAIL",
            "信号时序": "PASS" if result.signal_timing_correct else "FAIL",
            "止损逻辑": "PASS" if result.stop_loss_working else "FAIL",
            "问题数": len(result.issues),
            "总收益": result.metrics.get("总收益", 0),
            "年化收益": result.metrics.get("年化收益", 0),
            "夏普比率": result.metrics.get("夏普比率", 0),
            "最大回撤": result.metrics.get("最大回撤", 0),
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / "audit_summary.xlsx"
    summary_df.to_excel(summary_path, index=False)
    print(f"  汇总报告: {summary_path}")
    
    # =========================================================================
    # 完成
    # =========================================================================
    print("\n" + "=" * 70)
    print("白盒审计完成")
    print("=" * 70)
    print(f"\n输出目录: {output_dir}")
    print(f"  - synthetic_npy/     : 合成数据")
    print(f"  - strategy_reports/  : 策略审计报告")
    print(f"  - audit_summary.xlsx : 汇总报告")
    if not args.skip_lookahead:
        print(f"  - lookahead_report.txt : 前视偏差报告")
    
    # 统计
    pass_count = sum(1 for r in audit_results.values() if not r.issues)
    total_count = len(audit_results)
    print(f"\n审计结果: {pass_count}/{total_count} 策略通过审计")
    
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
