"""
scripts/true_oos_test.py
==========================
生命中只用一次的最终 OOS 测试

运行前确认：你已完成策略在 IS 期间的所有调参工作。
运行后：结果被锁定，同一策略+参数不允许重跑（防止反复测试选优）。

OOS 评估标准（全部通过才考虑实盘）:
  ✓ 年化收益 > 12%
  ✓ 夏普比率 > 0.7
  ✓ 最大回撤 < 35%
  ✓ 换手率   < 400%（摩擦成本可控）
"""
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOCK_FILE = ROOT / "data" / ".oos_test_results.json"
OOS_START = "2023-01-01"
OOS_END   = "2024-12-31"


def _load_locks() -> dict:
    if LOCK_FILE.exists():
        try: return json.loads(LOCK_FILE.read_text(encoding="utf-8"))
        except Exception: return {}
    return {}


def _save_lock(key: str, record: dict):
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    locks = _load_locks()
    locks[key] = record
    LOCK_FILE.write_text(json.dumps(locks, ensure_ascii=False, indent=2), encoding="utf-8")


def list_results():
    """显示所有已完成的 OOS 测试结果"""
    locks = _load_locks()
    if not locks:
        print("  尚无 OOS 测试记录")
        return
    print(f"\n  {'策略':<24} {'年化':>8} {'夏普':>6} {'回撤':>8} {'换手':>7} {'测试时间'}")
    print("  " + "─" * 70)
    for key, rec in locks.items():
        passed = _count_passed(rec)
        mark = "🟢" if passed == 4 else ("🟡" if passed >= 2 else "🔴")
        print(f"  {mark} {rec.get('strategy','?'):<22} "
              f"{rec.get('annual_return', 0):>+8.1%}  "
              f"{rec.get('sharpe_ratio', 0):>6.2f}  "
              f"{rec.get('max_drawdown', 0):>8.1%}  "
              f"{rec.get('turnover', 0):>7.0f}  "
              f"{rec.get('test_time','?')[:16]}")


def _count_passed(rec: dict) -> int:
    checks = [
        rec.get("annual_return", 0) > 0.12,
        rec.get("sharpe_ratio", 0) > 0.7,
        rec.get("max_drawdown", 0) < 0.35,
        rec.get("turnover", 9999) < 400,
    ]
    return sum(checks)


def run(strategy_name: str, params: dict = None, force: bool = False):
    """
    执行最终 OOS 测试。

    Parameters
    ----------
    strategy_name : str
    params        : dict，策略参数（None=使用默认参数）
    force         : bool，True=忽略锁定强制重跑（仅调试用）
    """
    params = params or {}
    lock_key = f"{strategy_name}::{json.dumps(params, sort_keys=True)}"

    # 检查是否已测试
    locks = _load_locks()
    if lock_key in locks and not force:
        print(f"\n⛔ 拒绝重复测试！")
        print(f"   {strategy_name} 的 OOS 测试已于 "
              f"{locks[lock_key].get('test_time','?')[:16]} 运行过")
        print(f"   结果: 年化={locks[lock_key].get('annual_return',0):+.1%}  "
              f"夏普={locks[lock_key].get('sharpe_ratio',0):.2f}")
        print()
        print("   这是生命中只用一次的测试，不允许因结果不好而重测。")
        print("   如需测试新参数，请修改参数后重新运行（参数不同=新的锁定记录）。")
        return None

    print(f"\n{'='*58}")
    print(f"  最终 OOS 测试")
    print(f"  策略: {strategy_name}")
    print(f"  参数: {params}")
    print(f"  测试区间: {OOS_START} ~ {OOS_END}")
    print(f"{'='*58}")
    print()
    print("  ⚠ 警告：此测试结果将被永久锁定")
    print("  ⚠ 请确认已完成 IS 期间（2015-2022）的所有调参工作")
    print()
    confirm = input("  输入 YES 确认运行，其他任意键取消: ").strip()
    if confirm != "YES":
        print("  已取消"); return None

    from src.engine.fast_runner_v10 import FastRunnerV10

    cfg = json.load(open(ROOT / "config.json"))
    runner = FastRunnerV10(cfg)
    runner.load_data()

    # 构建参数对象
    if params:
        class _P:
            pass
        p = _P()
        for k, v in params.items():
            setattr(p, k, v)
        p.annual_factor = 252.0
        p.to_dict = lambda: params
    else:
        p = None

    # 触发策略注册
    vec = ROOT / "src" / "strategies" / "vectorized"
    for py in sorted(vec.glob("*_alpha.py")):
        try: __import__(f"src.strategies.vectorized.{py.stem}")
        except Exception: pass

    print(f"\n  运行中...")
    res = runner.run(strategy_name, p, OOS_START, OOS_END)

    # 构建记录
    record = {
        "strategy":      strategy_name,
        "params":        params,
        "oos_period":    f"{OOS_START}~{OOS_END}",
        "annual_return": res.annual_return,
        "sharpe_ratio":  res.sharpe_ratio,
        "max_drawdown":  res.max_drawdown,
        "turnover":      res.turnover,
        "test_time":     datetime.now().isoformat(),
    }

    # 锁定结果
    _save_lock(lock_key, record)

    # 输出
    print()
    print(f"{'='*58}")
    print(f"  OOS 测试结果（已锁定）")
    print(f"{'='*58}")
    print(f"  年化收益: {res.annual_return:+.1%}")
    print(f"  夏普比率: {res.sharpe_ratio:.2f}")
    print(f"  最大回撤: {res.max_drawdown:.1%}")
    print(f"  换手率:   {res.turnover:.0f}")
    print()

    checks = [
        ("年化 > 12%",     res.annual_return > 0.12,  f"{res.annual_return:.1%}"),
        ("夏普 > 0.7",     res.sharpe_ratio  > 0.7,   f"{res.sharpe_ratio:.2f}"),
        ("最大回撤 < 35%", res.max_drawdown  < 0.35,  f"{res.max_drawdown:.1%}"),
        ("换手率 < 400%",  res.turnover      < 400,   f"{res.turnover:.0f}"),
    ]
    passed = 0
    for name, ok, val in checks:
        mark = "✓" if ok else "✗"
        print(f"  {mark} {name}: {val}")
        if ok: passed += 1

    print()
    if passed == 4:
        print("  🟢 全部通过 → 建议小仓实盘（10~20万验证6个月）")
    elif passed >= 2:
        print("  🟡 部分通过 → 继续优化后再考虑实盘")
    else:
        print("  🔴 未通过  → 策略不具备实盘价值，需重新设计")

    return res


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default=None, help="策略名称")
    p.add_argument("--list",     action="store_true", help="显示所有 OOS 记录")
    p.add_argument("--rsrs-window",    type=int, default=18)
    p.add_argument("--zscore-window",  type=int, default=300)
    p.add_argument("--top-n",          type=int, default=25)
    p.add_argument("--force",  action="store_true", help="忽略锁定（调试用）")
    a = p.parse_args()

    if a.list:
        list_results()
    elif a.strategy:
        params = {}
        if a.rsrs_window != 18: params["rsrs_window"] = a.rsrs_window
        if a.zscore_window != 300: params["zscore_window"] = a.zscore_window
        if a.top_n != 25: params["top_n"] = a.top_n
        run(a.strategy, params or None, force=a.force)
    else:
        print("用法:")
        print("  python scripts/true_oos_test.py --strategy titan_alpha_v1")
        print("  python scripts/true_oos_test.py --list")
