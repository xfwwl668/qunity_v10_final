#!/usr/bin/env python3
import subprocess
import os

os.chdir('/vercel/share/v0-project')

print("=" * 70)
print("Git 状态检查")
print("=" * 70)

result = subprocess.run(['git', 'status'], capture_output=True, text=True)
print(result.stdout)

print("\n" + "=" * 70)
print("检查修改的文件")
print("=" * 70)

result = subprocess.run(['git', 'diff', '--name-only'], capture_output=True, text=True)
if result.stdout:
    print("已修改的文件:")
    print(result.stdout)
else:
    print("没有修改的文件")

print("\n" + "=" * 70)
print("检查新增的文件")
print("=" * 70)

result = subprocess.run(['git', 'status', '--short'], capture_output=True, text=True)
print(result.stdout)

print("\n" + "=" * 70)
print("白盒审计文件清单")
print("=" * 70)

audit_files = [
    'whitebox_audit/__init__.py',
    'whitebox_audit/synthetic_data_generator.py',
    'whitebox_audit/trade_tracer.py',
    'whitebox_audit/lookahead_detector.py',
    'whitebox_audit/strategy_audit_runner.py',
    'whitebox_audit/factor_verification.py',
    'whitebox_audit/adjustment_validator.py',
    'whitebox_audit/run_whitebox_audit.py',
    'whitebox_audit/audit_template.py',
    'whitebox_audit/diagnose.py',
    'whitebox_audit/README.md',
    'whitebox_audit/ARCHITECTURE.md',
    'whitebox_audit/AUDIT_EXECUTIVE_SUMMARY.txt',
    'whitebox_audit/README_AUDIT_RESULTS.md',
    'whitebox_audit/COMPLETE_13_STRATEGIES_AUDIT_REPORT.md',
    'whitebox_audit/DEEP_BUG_AUDIT_REPORT_V2.md',
    'whitebox_audit/FINAL_SUMMARY.py',
]

scripts_files = [
    'scripts/run_whitebox_audit.py',
    'scripts/deep_whitebox_audit.py',
    'scripts/generate_audit_summary.py',
    'scripts/audit_all_13_strategies.py',
    'scripts/audit_13_strategies_final.py',
    'scripts/audit_13_complete_manual.py',
    'scripts/standardize_strategy_params.py',
    'scripts/pyproject.toml',
]

root_files = [
    'AUDIT_DELIVERABLES.txt',
    'FINAL_WHITEBOX_AUDIT_SUMMARY.txt',
    'COMPREHENSIVE_AUDIT_FINAL_REPORT.txt',
    'QUICK_REFERENCE_CARD.txt',
]

modified_files = [
    'src/data/adj_converter.py',
    'src/engine/risk_config.py',
    'src/engine/portfolio_builder.py',
    'src/engine/numba_kernels_v10.py',
]

print("\n✓ 白盒审计模块文件:")
for f in audit_files:
    exists = "✓" if os.path.exists(f) else "✗"
    print(f"  {exists} {f}")

print("\n✓ 审计脚本文件:")
for f in scripts_files:
    exists = "✓" if os.path.exists(f) else "✗"
    print(f"  {exists} {f}")

print("\n✓ 审计报告文件:")
for f in root_files:
    exists = "✓" if os.path.exists(f) else "✗"
    print(f"  {exists} {f}")

print("\n✓ 已修改的核心文件:")
for f in modified_files:
    exists = "✓" if os.path.exists(f) else "✗"
    print(f"  {exists} {f}")

print("\n" + "=" * 70)
print("总计文件数量")
print("=" * 70)
all_files = audit_files + scripts_files + root_files + modified_files
count = sum(1 for f in all_files if os.path.exists(f))
print(f"✓ {count} / {len(all_files)} 文件存在")
