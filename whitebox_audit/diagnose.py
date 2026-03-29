#!/usr/bin/env python3
"""
系统诊断脚本 - 检查整个审计框架的完整性和依赖关系
"""

import sys
import os
from pathlib import Path

def check_environment():
    """检查 Python 环境"""
    print("\n📋 Python 环境检查")
    print(f"  Python 版本: {sys.version.split()[0]}")
    print(f"  Python 执行路径: {sys.executable}")
    
    # 检查必要的包
    required_packages = {
        'numpy': 'NumPy (数值计算)',
        'pandas': 'Pandas (数据处理)',
        'numba': 'Numba (JIT 编译)',
    }
    
    missing = []
    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {description} 已安装")
        except ImportError:
            print(f"  ✗ {description} 未安装")
            missing.append(package)
    
    if missing:
        print(f"\n  ⚠ 缺失包: {', '.join(missing)}")
        print(f"  安装命令: pip install {' '.join(missing)}")
        return False
    
    return True


def check_file_structure():
    """检查文件结构"""
    print("\n📁 文件结构检查")
    
    project_root = Path(__file__).parent.parent
    whitebox_dir = project_root / "whitebox_audit"
    
    print(f"  项目根目录: {project_root}")
    print(f"  审计框架目录: {whitebox_dir}")
    
    required_files = [
        "__init__.py",
        "README.md",
        "synthetic_data_generator.py",
        "trade_tracer.py",
        "lookahead_detector.py",
        "factor_verification.py",
        "adjustment_validator.py",
        "strategy_audit_runner.py",
        "run_whitebox_audit.py",
        "audit_template.py",
        "demo_quick_audit.py",
    ]
    
    all_exist = True
    for filename in required_files:
        filepath = whitebox_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"  ✓ {filename:<35} ({size:>8} 字节)")
        else:
            print(f"  ✗ {filename:<35} (缺失)")
            all_exist = False
    
    return all_exist


def check_strategy_files():
    """检查策略文件"""
    print("\n🎯 策略文件检查")
    
    project_root = Path(__file__).parent.parent
    strategies_dir = project_root / "src" / "strategies" / "vectorized"
    
    strategies = [
        "kunpeng_v10_alpha.py",
        "snma_v4_alpha.py",
        "titan_orthogonal_v10_alpha.py",
        "momentum_reversal_alpha.py",
        "short_term_rsrs_alpha.py",
        "sentiment_reversal_alpha.py",
    ]
    
    all_exist = True
    for strategy in strategies:
        filepath = strategies_dir / strategy
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"  ✓ {strategy:<35} ({size:>8} 字节)")
        else:
            print(f"  ✗ {strategy:<35} (缺失)")
            all_exist = False
    
    return all_exist


def check_engine_files():
    """检查引擎文件"""
    print("\n⚙️  引擎文件检查")
    
    project_root = Path(__file__).parent.parent
    engine_dir = project_root / "src" / "engine"
    
    engine_files = [
        "fast_runner_v10.py",
        "numba_kernels_v10.py",
        "portfolio_builder.py",
    ]
    
    all_exist = True
    for filename in engine_files:
        filepath = engine_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"  ✓ {filename:<35} ({size:>8} 字节)")
        else:
            print(f"  ✗ {filename:<35} (缺失)")
            all_exist = False
    
    return all_exist


def check_data_files():
    """检查数据处理文件"""
    print("\n💾 数据处理文件检查")
    
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "src" / "data"
    
    data_files = [
        "adj_converter.py",
        "build_npy.py",
        "alpha_signal.py",
    ]
    
    all_exist = True
    for filename in data_files:
        filepath = data_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"  ✓ {filename:<35} ({size:>8} 字节)")
        else:
            print(f"  ✗ {filename:<35} (缺失)")
            all_exist = False
    
    return all_exist


def check_output_directories():
    """检查输出目录"""
    print("\n📤 输出目录检查")
    
    project_root = Path(__file__).parent.parent
    whitebox_dir = project_root / "whitebox_audit"
    
    output_dirs = [
        "outputs",
        "outputs/synthetic_npy",
        "outputs/strategy_reports",
        "outputs/debug",
    ]
    
    for dirname in output_dirs:
        dirpath = whitebox_dir / dirname
        if dirpath.exists():
            print(f"  ✓ {dirname:<35} (已存在)")
        else:
            print(f"  ℹ {dirname:<35} (将创建)")
            dirpath.mkdir(parents=True, exist_ok=True)
    
    return True


def print_summary():
    """打印总结"""
    print("\n" + "="*70)
    print("📊 诊断总结")
    print("="*70)
    
    print("""
✓ 白盒审计框架完整
✓ 依赖关系检查完成
✓ 输出目录已准备

接下来:
  1. 运行演示: python whitebox_audit/demo_quick_audit.py
  2. 生成合成数据: python -m whitebox_audit.synthetic_data_generator
  3. 运行完整审计: python -m whitebox_audit.run_whitebox_audit --days 1500 --cycles 3
  4. 查看报告: outputs/audit_summary.xlsx

文档:
  - 详细说明: whitebox_audit/README.md
  - 审计模板: whitebox_audit/audit_template.py
  - 演示代码: whitebox_audit/demo_quick_audit.py
    """)


def main():
    """主函数"""
    print("\n" + "🔍 Q-UNITY V10 白盒审计框架诊断")
    print("="*70)
    
    checks = [
        ("Python 环境", check_environment),
        ("审计框架文件", check_file_structure),
        ("策略文件", check_strategy_files),
        ("引擎文件", check_engine_files),
        ("数据文件", check_data_files),
        ("输出目录", check_output_directories),
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"  ✗ 检查异常: {e}")
            results.append((check_name, False))
    
    print_summary()
    
    # 最终结果
    all_passed = all(result for _, result in results)
    if all_passed:
        print("✅ 所有检查通过！系统准备就绪。\n")
        return 0
    else:
        print("⚠️  部分检查失败。请检查上面的错误信息。\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
