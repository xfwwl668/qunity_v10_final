"""
scripts/liquidity_tier_analysis.py
=====================================
流动性分层测试：识别 Alpha 是否依赖小市值/低流动性股票

A 股小票 Alpha 在实盘中往往是幻觉：
  - 日均成交额 < 1000 万的股票，100 万本金买入 4% 仓位
    就已是当日成交额的 4%，真实冲击成本失控
  - 如果只有加入小票后策略才好，说明 Alpha 不可实盘

判读标准：
  加流动性门槛后夏普下降 > 30% → Alpha 依赖小票，实盘不可用
  夏普基本不变              → Alpha 真实，可以实盘
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


def analyze(
    start: str = "2020-01-01",
    end:   str = "2024-12-31",
    strategy: str = None,
):
    from src.engine.fast_runner_v10 import FastRunnerV10
    from src.strategies.registry import list_vec_strategies

    cfg_base = json.load(open(ROOT / "config.json"))
    _load_strategies()
    strats = list_vec_strategies()
    if not strats:
        print("无可用策略"); return

    strategy = strategy or strats[0]
    print(f"\n流动性分层测试  策略={strategy}  [{start}~{end}]")
    print("=" * 72)
    print(f"  {'流动性门槛':>18}  {'年化收益':>10}  {'夏普':>6}  {'最大回撤':>10}  {'换手率':>8}")
    print("  " + "─" * 60)

    # 基准（无门槛）先跑
    baseline = None
    rows = []
    tiers = [
        ("无门槛（原始）",   0),
        ("日均500万+",   5_000_000),
        ("日均1000万+",  10_000_000),
        ("日均3000万+",  30_000_000),
        ("日均5000万+",  50_000_000),
        ("日均1亿+",    100_000_000),
    ]

    for tier_name, min_amt in tiers:
        cfg = json.loads(json.dumps(cfg_base))
        cfg["min_avg_amount"] = min_amt
        runner = FastRunnerV10(cfg)
        runner.load_data()

        sys.stdout.write(f"\r  测试: {tier_name}..."); sys.stdout.flush()
        try:
            res = runner.run(strategy, None, start, end)
            flag = " ← 基准" if min_amt == 0 else ""
            if baseline is None:
                baseline = res.sharpe_ratio
            decay = ""
            if baseline and min_amt > 0 and baseline > 0:
                pct = (res.sharpe_ratio - baseline) / baseline
                if pct < -0.30:
                    decay = f" ⚠ 夏普↓{abs(pct):.0%}"
                elif pct < -0.15:
                    decay = f" △ 夏普↓{abs(pct):.0%}"
            print(f"\r  {tier_name:>18}  {res.annual_return:>+10.1%}  "
                  f"{res.sharpe_ratio:>6.2f}  {res.max_drawdown:>10.1%}  "
                  f"{res.turnover:>8.1f}{flag}{decay}")
            rows.append({
                "tier": tier_name, "min_amt": min_amt,
                "sharpe": res.sharpe_ratio,
                "annual_return": res.annual_return,
                "max_dd": res.max_drawdown,
            })
        except Exception as e:
            print(f"\r  {tier_name:>18}  失败: {e}")

    print()
    if len(rows) >= 2 and baseline:
        # 找到夏普衰减 < 20% 的最高门槛
        viable = [r for r in rows if r["min_amt"] > 0
                  and abs(r["sharpe"] - baseline) / max(baseline, 0.01) < 0.20]
        if viable:
            best_viable = max(viable, key=lambda r: r["min_amt"])
            print(f"  结论: 在 {best_viable['tier']} 门槛下夏普无显著衰减")
            print(f"  → 建议实盘将 min_avg_amount 设为 "
                  f"{best_viable['min_amt']:,.0f}")
            print(f"    (config.json: \"min_avg_amount\": {best_viable['min_amt']})")
        else:
            last = rows[-1]
            decay_pct = abs(last["sharpe"] - baseline) / max(baseline, 0.01)
            if decay_pct > 0.30:
                print("  ⚠ 警告：加流动性门槛后夏普显著下降")
                print("    策略 Alpha 可能高度依赖小票，实盘可行性存疑")
                print("    建议: 重新设计策略，专注于日均成交额 > 5000万的标的")
            else:
                print("  ✓ Alpha 在各流动性层级均保持稳定，可以实盘")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start",    default="2020-01-01")
    p.add_argument("--end",      default="2024-12-31")
    p.add_argument("--strategy", default=None, help="策略名称（默认第一个注册策略）")
    a = p.parse_args()
    analyze(a.start, a.end, a.strategy)
