# Q-UNITY V10 项目文件清单 (清理后)

## 文件统计

| 类别 | 数量 | 说明 |
|------|------|------|
| 核心引擎 | 8 | src/engine/ |
| 数据处理 | 14 | src/data/ |
| 13个策略 | 13 | src/strategies/vectorized/ |
| 策略基础 | 3 | src/strategies/ |
| 数据下载 | 14 | scripts/step*.py |
| 运行脚本 | 10 | scripts/ |
| 白盒审计 | 8 | whitebox_audit/ |
| 测试 | 3 | tests/ |
| 配置/文档 | 6 | 根目录 |
| **总计** | **79** | |

---

## 一、核心源码 (src/)

### src/engine/ - 回测引擎 (8文件)
| 文件 | 功能 |
|------|------|
| fast_runner_v10.py | 主回测引擎 (已修复) |
| numba_kernels_v10.py | Numba加速内核 (已修复) |
| portfolio_builder.py | 组合构建器 (已修复) |
| portfolio_allocator.py | 资金分配器 |
| risk_config.py | 风险配置 (已修复) |
| optimizer_v10.py | 参数优化器 |
| live_runner_v10.py | 实盘运行器 |
| alpha_signal.py | Alpha信号基类 |

### src/data/ - 数据处理 (14文件)
| 文件 | 功能 |
|------|------|
| adj_converter.py | 复权转换 (已修复Bug#1) |
| build_npy.py | NPY数据构建 |
| baostock_downloader.py | BaoStock下载 |
| columnar_adapter.py | 列式数据适配 |
| fundamental_adapter.py | 基本面适配 |
| fundamental_downloader.py | 基本面下载 |
| live_data_adapter.py | 实时数据适配 |
| minute_adapter.py | 分钟数据适配 |
| minute_collector.py | 分钟数据收集 |
| adj_detector.py | 复权检测 |
| adj_validator_hook.py | 复权验证钩子 |
| audit_adj_types.py | 复权类型审计 |
| ths_adapter.py | 同花顺适配 |
| models.py | 数据模型 |

### src/strategies/vectorized/ - 13个向量化策略
| 文件 | 策略名 |
|------|--------|
| kunpeng_v10_alpha.py | 鲲鹏V10 |
| titan_orthogonal_v10_alpha.py | 泰坦正交V10 |
| snma_v4_alpha.py | SNMA V4 |
| momentum_reversal_alpha.py | 动量反转 |
| short_term_rsrs_alpha.py | 短期RSRS |
| sentiment_reversal_alpha.py | 情绪反转 |
| alpha_hunter_v2_alpha.py | Alpha猎手V2 |
| alpha_max_v5_alpha.py | Alpha极限V5 |
| retail_sniper_v10_alpha.py | 散户狙击V10 |
| sniper_v6a_alpha.py | 狙击手V6A |
| titan_alpha_v1_alpha.py | 泰坦Alpha V1 |
| ultra_alpha_v1_alpha.py | 超级Alpha V1 |
| weak_to_strong_alpha.py | 弱转强 |

### src/strategies/ - 策略基础 (3文件)
| 文件 | 功能 |
|------|------|
| registry.py | 策略注册表 |
| alpha_signal.py | Alpha信号类 |
| ultra_short_signal.py | 超短线信号 |

---

## 二、脚本 (scripts/)

### 数据下载脚本 (14文件)
| 文件 | 功能 |
|------|------|
| step0_download_ohlcv.py | 下载OHLCV数据 |
| step0_download_tdxquant.py | 通达信数据 |
| step0_patch_daily_fields.py | 补丁日数据 |
| step0b_fill_missing_daily_akshare.py | AKShare补缺 |
| step0c_download_daily_adata.py | AData日数据 |
| step0d_download_daily_akshare.py | AKShare日数据 |
| step1_download_fundamental.py | 基本面数据 |
| step1_download_fundamental_akshare.py | AKShare基本面 |
| step1_download_fundamental_tdxquant.py | 通达信基本面 |
| step1b_download_fundamental_adata.py | AData基本面 |
| step2_download_concepts.py | 概念板块 |
| step2_download_concepts_tdxquant.py | 通达信概念 |
| step3_build_fundamental_npy.py | 构建基本面NPY |
| step4_build_concept_npy.py | 构建概念NPY |

### 运行/工具脚本 (10文件)
| 文件 | 功能 |
|------|------|
| daily_run.py | 每日运行 |
| live_trade_tools.py | 实盘工具 |
| true_oos_test.py | 样本外测试 |
| factor_attribution.py | 因子归因 |
| validate_npy.py | NPY验证 |
| analyze_regime_cost.py | Regime成本分析 |
| liquidity_tier_analysis.py | 流动性分析 |
| realtime_tdxquant.py | 实时通达信 |
| tqcenter_utils.py | TQ工具 |
| utils_paths.py | 路径工具 |

### 审计脚本 (4文件)
| 文件 | 功能 |
|------|------|
| run_whitebox_audit.py | 白盒审计主脚本 |
| deep_whitebox_audit.py | 深度审计 |
| blackbox_backtest_comparison.py | 黑盒回测对比 |
| standardize_strategy_params.py | 参数标准化 |

---

## 三、白盒审计框架 (whitebox_audit/)

| 文件 | 功能 |
|------|------|
| synthetic_data_generator.py | 合成数据生成 (1500D, 4正弦波) |
| trade_tracer.py | 交易追踪器 |
| lookahead_detector.py | 前视偏差检测 |
| strategy_audit_runner.py | 策略审计运行器 |
| factor_verification.py | 因子验证 |
| adjustment_validator.py | 复权验证 |
| run_whitebox_audit.py | 主审计脚本 |
| test_bugfix_verification.py | Bug修复验证 |

---

## 四、测试 (tests/)

| 文件 | 功能 |
|------|------|
| test_v10_acceptance.py | V10验收测试 |
| test_audit_regressions.py | 审计回归测试 |

---

## 五、根目录

| 文件 | 功能 |
|------|------|
| run_all_backtest.py | 主回测入口 |
| main.py | 主入口 (待检查) |
| test_kunpeng.py | 鲲鹏测试 |
| config.json | 配置文件 |
| requirements.txt | 依赖 |
| README.md | 项目说明 |
| README_使用说明.md | 中文说明 |
| FINAL_EXECUTIVE_SUMMARY.md | 审计总结 |
| CHANGELOG_fixes.md | 修复日志 |
| CHANGELOG_v9.md | V9变更日志 |
| PROJECT_FILE_INVENTORY.md | 本文件 |

---

## 六、已修复的Bug清单

| Bug | 文件 | 修复内容 |
|-----|------|---------|
| #1 复权公式 | adj_converter.py | hfq = qfq * latest_factor |
| #2 Regime过敏 | risk_config.py | bear_breadth_thr: 0.25->0.20 |
| #3 BEAR仓为零 | portfolio_builder.py | BEAR: 0.0->0.3 |
| #4 止损冷却锁 | numba_kernels_v10.py | 自动解锁机制 |
| #5 停牌清仓 | numba_kernels_v10.py | 90天+20%残值 |

---

## 七、黑盒测试使用方法

```bash
# 1. 下载A股数据
python scripts/step0_download_ohlcv.py

# 2. 运行黑盒测试
python scripts/blackbox_backtest_comparison.py

# 3. 运行完整回测
python run_all_backtest.py
```

---

## 八、已删除的冗余文件 (39个)

### 之前AI的调试文件 (27个)
- certified_v13_audit.py, complete_audit_v15.py, complete_audit_v16.py
- final_audit_v14.py, final_successful_audit.py, real_v10_final_audit.py
- real_v5~v9_audit.py (5个), ultimate_*.py (4个)
- qfq_*.py (2个), physics_*.py (2个), plot_*.py (2个)
- export_excel_audit.py, inject_excel_formulas.py, debug_backtest.py
- data_reconciliation.py, check_fundamental_lookahead.py

### 我的冗余文件 (12个)
- audit_13_*.py (2个), audit_all_13_*.py (2个)
- generate_audit_summary.py, check_and_commit.py
- blackbox_audit_plan.py, cleanup_project.py
- 根目录报告文件 (4个)
