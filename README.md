# Q-UNITY V10 — 量化回测框架（独立版）

> **五层架构 · QFQ前复权 · stamp_tax=0.0005 · 追踪止损 · 多策略风险平价**

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动交互菜单（推荐）
python main.py

# 3. 或直接运行代码
python -c "
from src.engine.fast_runner_v10 import FastRunnerV10
runner = FastRunnerV10({'npy_dir':'data/npy','stamp_tax':0.0005})
runner.load_data()
result = runner.run('ultra_alpha_v1', None, '2020-01-01', '2023-12-31')
print(result.to_summary())
"
```

## 首次使用流程

```
Step 1  下载数据    python scripts/step0_download_ohlcv.py --workers 8 --build-npy
Step 2  验证安装    python tests/test_v10_acceptance.py
Step 3  跑回测      python main.py  →  选 3（单策略回测）
```

## 目录结构

```
qunity_v10/
├── main.py                    ← 交互式主控台（入口）
├── config.json                ← 配置文件（stamp_tax=0.0005）
├── requirements.txt
├── src/
│   ├── engine/
│   │   ├── fast_runner_v10.py     ← 回测执行器
│   │   ├── numba_kernels_v10.py   ← Numba 撮合内核（追踪止损）
│   │   ├── portfolio_builder.py   ← Layer 2 风控（Regime+流动性）
│   │   ├── portfolio_allocator.py ← 多策略资金分配
│   │   ├── optimizer_v10.py       ← Walk-Forward 参数优化
│   │   ├── risk_config.py         ← 统一风控参数
│   │   └── alpha_signal.py        ← 信号数据结构
│   ├── strategies/
│   │   ├── registry.py            ← 策略注册表
│   │   └── vectorized/
│   │       └── *_alpha.py         ← 9个日线策略
│   └── data/
│       ├── columnar_adapter_v10.py← QFQ 数据适配器
│       └── dataclasses.py         ← MatrixBundle / MemMapMeta
├── scripts/
│   ├── step0_download_ohlcv.py    ← 日线下载（AKShare QFQ）
│   ├── step1_download_fundamental.py
│   ├── step2_download_concepts.py
│   └── daily_run.py               ← 每日自动运行
└── tests/
    └── test_v10_acceptance.py     ← 验收测试
```

## 铁律（不可违反）

| 铁律 | 值 |
|------|-----|
| stamp_tax | **0.0005**（万五，2024-09-24起） |
| adj_type | **qfq**（前复权） |
| 止损机制 | **追踪止损**（从持仓最高价算回撤） |
| 策略函数 | **纯函数**（无状态，相同输入→相同输出） |

## 本版本不包含（需 V9）

| 功能 | 状态 |
|------|------|
| 实盘监控（盘中预警/TDX行情） | V9 realtime 模块 |
| 手动下单辅助 | V9 trader.py |
| 分钟级信号 | V9 ultra_short_vec.py |
| 详细绩效报表（MetricsCalculator） | 待迁移 |
| 数据质量检查工具 | V9 scripts/ |
