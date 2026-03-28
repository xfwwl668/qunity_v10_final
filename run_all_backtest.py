#!/usr/bin/env python3
"""
非交互式全策略回测脚本
运行所有已注册策略的单策略回测 + 多策略组合回测

★[FIX-LOG] 日志级别从 WARNING 升为 INFO，确保 multi_run 进度可见。
           fast_runner_v10.py 的关键进度改用 print()，不受日志级别影响。
"""
from __future__ import annotations
import json, sys, time, traceback, logging, warnings
from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ★[FIX-LOG] 级别 WARNING → INFO，multi_run 进度不再被吞掉
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
for _noisy in ("numba", "llvmlite", "urllib3", "baostock"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# ── 加载配置 ──────────────────────────────────────────────────────────────
with open(ROOT / "config.json", encoding="utf-8") as f:
    cfg_data = json.load(f)

# ── 加载策略 ──────────────────────────────────────────────────────────────
vec = ROOT / "src" / "strategies" / "vectorized"
for py in sorted(vec.glob("*_alpha.py")):
    try:
        __import__(f"src.strategies.vectorized.{py.stem}")
    except Exception as e:
        print(f"  [WARN] 策略加载失败: {py.stem}: {e}")

from src.strategies.registry import list_vec_strategies
from src.engine.fast_runner_v10 import FastRunnerV10
from src.engine.optimizer_v10 import StrategyParams

# ── 加载数据 ──────────────────────────────────────────────────────────────
print("=" * 70)
print("  Q-UNITY V10 全策略非交互式回测")
print("=" * 70)

print("\n[1/4] 加载数据矩阵...", flush=True)
t0 = time.perf_counter()
runner = FastRunnerV10(cfg_data)
runner.load_data()
print(f"  数据加载完成 ({time.perf_counter() - t0:.1f}s)", flush=True)

# ★[FIX-WARMUP] 全面 JIT 热身 + Numba 可用性检测
print("[2/4] JIT 热身（引擎 / Regime / Builder）...", flush=True)
t0 = time.perf_counter()
with warnings.catch_warnings(record=True) as caught_warns:
    warnings.simplefilter("always")
    try:
        elapsed_warmup = FastRunnerV10.warmup_jit(runner._risk_cfg)
    except Exception as e:
        print(f"  [WARN] 热身失败: {e}", flush=True)
        elapsed_warmup = 0.0

# 将 Numba 不可用警告直接打印（不依赖日志级别）
for w in caught_warns:
    if issubclass(w.category, UserWarning) and "Numba" in str(w.message):
        print(str(w.message), flush=True)

print(f"  热身完成 ({elapsed_warmup:.1f}s)", flush=True)

# ── 全策略列表 ────────────────────────────────────────────────────────────
strats = list_vec_strategies()
print(f"\n[3/4] 已注册策略 ({len(strats)}): {strats}", flush=True)

START = "2019-01-02"
END   = date.today().isoformat()
sp_cfg = cfg_data.get("strategy_params", {})

# ── 单策略逐一回测 ────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"  单策略回测  [{START} ~ {END}]")
print(f"{'=' * 70}")
print(f"\n  {'策略':<25} {'年化收益':>10} {'夏普':>8} {'最大回撤':>10} {'换手率':>8} {'耗时':>8}")
print(f"  {'─' * 75}", flush=True)

results = {}
for name in strats:
    sp = sp_cfg.get(name, {})
    params_obj = StrategyParams(**sp) if sp else None
    try:
        t0 = time.perf_counter()
        res = runner.run(name, params_obj, START, END)
        elapsed = time.perf_counter() - t0
        ar = getattr(res, "annual_return", 0.0)
        sr = getattr(res, "sharpe_ratio",  0.0)
        dd = getattr(res, "max_drawdown",  0.0)
        to = getattr(res, "turnover",      0.0)
        print(f"  {name:<25} {ar:>+10.1%} {sr:>8.2f} {dd:>10.1%} {to:>8.1f} {elapsed:>7.1f}s",
              flush=True)
        results[name] = res
    except Exception as e:
        print(f"  {name:<25} ** 失败: {e} **", flush=True)
        traceback.print_exc()

# ── 多策略组合回测 ────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"  多策略组合回测 (risk_parity)")
print(f"{'=' * 70}", flush=True)

if len(results) >= 2:
    try:
        from src.engine.portfolio_allocator import PortfolioAllocator
        alloc = PortfolioAllocator(method="risk_parity")
        strategy_configs = [
            (name, StrategyParams(**sp_cfg.get(name, {})) if sp_cfg.get(name) else None)
            for name in results
        ]
        print(f"  开始 multi_run（{len(strategy_configs)} 个策略 × Pass1+Pass2）...", flush=True)
        t0 = time.perf_counter()
        multi_res = runner.multi_run(strategy_configs, alloc, START, END)
        elapsed = time.perf_counter() - t0

        ar = getattr(multi_res, "annual_return", 0.0)
        sr = getattr(multi_res, "sharpe_ratio",  0.0)
        dd = getattr(multi_res, "max_drawdown",  0.0)
        print(f"\n  组合结果 (总耗时 {elapsed:.1f}s):")
        print(f"    年化收益: {ar:+.1%}")
        print(f"    夏普比率: {sr:.2f}")
        print(f"    最大回撤: {dd:.1%}")

        if hasattr(multi_res, "allocations") and multi_res.allocations:
            print(f"\n  资金分配:")
            for k, v in multi_res.allocations.items():
                print(f"    {k:<25} {v:.1%}")

    except Exception as e:
        print(f"  组合回测失败: {e}", flush=True)
        traceback.print_exc()
else:
    print("  成功策略不足2个，跳过组合回测", flush=True)

print(f"\n{'=' * 70}")
print(f"  全部回测完成")
print(f"{'=' * 70}", flush=True)
