import json
import time
import os
import sys

# Add current dir to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.engine.fast_runner_v10 import FastRunnerV10
from src.engine.optimizer_v10 import StrategyParams
# Ensure strategy is registered
import src.strategies.vectorized.kunpeng_v10_alpha

def main():
    with open("config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    print("Loading data...")
    runner = FastRunnerV10(cfg)
    runner.load_data()
    print("Warming up JIT...")
    FastRunnerV10.warmup_jit(runner._risk_cfg)
    
    start_date = "2019-01-02"
    end_date = "2026-03-25"
    
    print(f"Running backtest for kunpeng_v10 from {start_date} to {end_date}...")
    t0 = time.time()
    try:
        res = runner.run("kunpeng_v10", None, start_date, end_date)
        print("\nBacktest Results:")
        print(res.to_summary())
        print(f"\nBreakdown: {res.pipeline_breakdown}")
    except Exception as e:
        import traceback
        traceback.print_exc()
    print(f"Elapsed: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
