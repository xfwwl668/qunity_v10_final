"""
scripts/analyze_regime_cost.py
================================
量化 Regime 切换的真实换手成本

测试不同 bear_confirm_days（进熊快慢）对策略绩效和换手率的影响。
A股"假摔"频繁，bear_confirm_days=1 可能导致不必要的高换手。
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
    start: str = "2019-01-01",
    end:   str = "2024-12-31",
    strategy: str = None,
):
    from src.engine.fast_runner_v10 import FastRunnerV10
    from src.engine.portfolio_builder import MarketRegimeDetector, PortfolioBuilder
    from src.strategies.registry import list_vec_strategies

    cfg = json.load(open(ROOT / "config.json"))

    _load_strategies()
    strats = list_vec_strategies()
    if not strats:
        print("无可用策略，请先确认数据已构建"); return

    strategy = strategy or strats[0]
    print(f"\n Regime 切换成本分析  策略={strategy}  [{start}~{end}]")
    print("=" * 68)
    print(f"  {'bear_confirm':>14}  {'bear_exit':>9}  {'换手率':>8}  {'夏普':>6}  "
          f"{'年化':>8}  {'最大回撤':>10}")
    print("  " + "─" * 62)

    rows = []
    for confirm, exit_d in [
        (1,  10),   # 原始默认（快进慢出）
        (3,  10),
        (5,  10),
        (5,  5),
        (10, 15),   # 最保守
    ]:
        cfg_t = json.loads(json.dumps(cfg))
        runner = FastRunnerV10(cfg_t)
        runner.load_data()

        # 修改 RiskConfig
        runner._risk_cfg.bear_confirm_days = confirm
        runner._risk_cfg.bear_exit_days    = exit_d
        # 强制重建 Regime 检测器
        runner._regime_det   = None
        runner._port_builder = None

        try:
            res = runner.run(strategy, None, start, end)
            label = "← 当前默认" if (confirm == 1 and exit_d == 10) else ""
            print(f"  {'进'+str(confirm)+'天/出'+str(exit_d)+'天':>14}  "
                  f"{str(confirm)+'d进/'+str(exit_d)+'d出':>9}  "
                  f"{res.turnover:>8.1f}  {res.sharpe_ratio:>6.2f}  "
                  f"{res.annual_return:>+8.1%}  {res.max_drawdown:>10.1%}  {label}")
            rows.append({
                "confirm": confirm, "exit": exit_d,
                "turnover": res.turnover, "sharpe": res.sharpe_ratio,
                "annual_return": res.annual_return, "max_dd": res.max_drawdown,
            })
        except Exception as e:
            print(f"  {confirm}d进/{exit_d}d出:  失败 {e}")

    if rows:
        best = max(rows, key=lambda r: r["sharpe"])
        baseline = rows[0]
        print()
        print("  结论建议:")
        if best["confirm"] != baseline["confirm"]:
            print(f"    最优配置: bear_confirm_days={best['confirm']}  "
                  f"bear_exit_days={best['exit']}")
            print(f"    vs 当前: 换手率 {baseline['turnover']:.0f} → "
                  f"{best['turnover']:.0f}  "
                  f"夏普 {baseline['sharpe']:.2f} → {best['sharpe']:.2f}")
            print()
            print(f"  修改方式: 在 config.json 中添加 (需要 RiskConfig 支持直接读取):")
            print(f"    'bear_confirm_days': {best['confirm']},")
            print(f"    'bear_exit_days': {best['exit']}")
        else:
            print("    当前默认配置已是最优，无需修改")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start",    default="2019-01-01")
    p.add_argument("--end",      default="2024-12-31")
    p.add_argument("--strategy", default=None)
    a = p.parse_args()
    analyze(a.start, a.end, a.strategy)
