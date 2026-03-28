"""
scripts/factor_attribution.py
================================
单因子归因分析：找出 titan_alpha_v1 中哪个因子真正贡献了 Alpha

同时检测因子衰减：2019-2021（量化牛市）vs 2022-2024（拥挤后）
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_strategies():
    vec = ROOT / "src" / "strategies" / "vectorized"
    for py in sorted(vec.glob("*_alpha.py")):
        try: __import__(f"src.strategies.vectorized.{py.stem}")
        except Exception: pass


def _make_single_factor_weights(factor_name: str) -> dict:
    """构造只启用单一因子的 FACTOR_WEIGHTS"""
    all_factors = ["rsrs", "momentum", "financial", "vol", "concept"]
    return {
        regime: {f: (1.0 if f == factor_name else 0.0) for f in all_factors}
        for regime in ["STRONG_BULL", "BULL", "NEUTRAL", "SOFT_BEAR"]
    } | {"BEAR": {f: 0.0 for f in all_factors}}


def test_factor(factor_name: str, start: str, end: str, cfg: dict) -> dict:
    """测试单个因子的收益"""
    from src.engine.fast_runner_v10 import FastRunnerV10
    from src.strategies.registry import register_vec_strategy
    import src.strategies.vectorized.titan_alpha_v1_alpha as titan_mod

    runner = FastRunnerV10(cfg)
    runner.load_data()

    # Monkey-patch FACTOR_WEIGHTS
    original_fw = {k: v.copy() for k, v in titan_mod.FACTOR_WEIGHTS.items()}
    new_fw = _make_single_factor_weights(factor_name)
    titan_mod.FACTOR_WEIGHTS.update(new_fw)

    strat_name = f"_single_{factor_name}"
    register_vec_strategy(strat_name)(titan_mod.titan_alpha_v1_alpha)

    try:
        res = runner.run(strat_name, None, start, end)
        return {
            "factor": factor_name,
            "annual_return": res.annual_return,
            "sharpe": res.sharpe_ratio,
            "max_dd": res.max_drawdown,
            "turnover": res.turnover,
        }
    except Exception as e:
        return {"factor": factor_name, "error": str(e)}
    finally:
        titan_mod.FACTOR_WEIGHTS.update(original_fw)


def run_attribution(start="2019-01-01", end="2024-12-31"):
    cfg = json.load(open(ROOT / "config.json"))
    _load_strategies()

    factors = ["rsrs", "momentum", "financial", "vol"]

    print(f"\n单因子归因分析  [{start} ~ {end}]")
    print("=" * 62)
    print(f"  {'因子':<12} {'年化收益':>10} {'夏普':>7} {'最大回撤':>10} {'换手率':>8}")
    print("  " + "─" * 52)

    rows = []
    for f in factors:
        sys.stdout.write(f"  测试 {f}..."); sys.stdout.flush()
        r = test_factor(f, start, end, cfg)
        if "error" in r:
            print(f"\r  {f:<12}  失败: {r['error']}")
        else:
            print(f"\r  {r['factor']:<12} {r['annual_return']:>+10.1%} "
                  f"{r['sharpe']:>7.2f} {r['max_dd']:>10.1%} {r['turnover']:>8.1f}")
            rows.append(r)

    # ── 因子衰减检测 ────────────────────────────────────────────────────
    print()
    print("  动量因子衰减检测（量化拥挤问题）:")
    print(f"  {'时段':<28} {'年化':>8} {'夏普':>7}")
    print("  " + "─" * 46)
    for label, (s, e) in [
        ("2019-2021 (量化牛市)",  ("2019-01-01", "2021-12-31")),
        ("2022-2024 (拥挤后)",    ("2022-01-01", "2024-12-31")),
    ]:
        r = test_factor("momentum", s, e, cfg)
        if "error" not in r:
            flag = " ← 拥挤衰减" if (r["sharpe"] < 0.5 and e[:4] == "2024") else ""
            print(f"  {label:<28} {r['annual_return']:>+8.1%} {r['sharpe']:>7.2f}{flag}")

    # ── 结论 ────────────────────────────────────────────────────────────
    if rows:
        print()
        print("  归因结论:")
        best = max(rows, key=lambda x: x["sharpe"])
        worst = min(rows, key=lambda x: x["sharpe"])
        print(f"    最强因子: {best['factor']:<12} 夏普={best['sharpe']:.2f}")
        print(f"    最弱因子: {worst['factor']:<12} 夏普={worst['sharpe']:.2f}")
        if worst["sharpe"] < 0.3:
            print(f"    → 建议在 FACTOR_WEIGHTS 中降低 {worst['factor']} 的权重")
        if best["factor"] == "financial":
            print("    → 基本面因子主导 Alpha，请先执行前视偏差检查")
            print("      python scripts/check_fundamental_lookahead.py")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end",   default="2024-12-31")
    a = p.parse_args()
    run_attribution(a.start, a.end)
