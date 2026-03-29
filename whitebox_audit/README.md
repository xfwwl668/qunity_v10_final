# Q-UNITY V10 白盒审计框架 (WhiteBox Audit Framework)

## 概述

本框架针对 Q-UNITY V10 量化系统的 **买卖逻辑** 和 **数据一致性** 进行专业级审计。采用 **逆向验证法** 和 **因果追踪** 方式，从下而上验证策略的正确性。

## 核心设计原理

### 1. 合成数据生成 (`synthetic_data_generator.py`)
- **目标**: 生成可预测、完全控制的后复权数据
- **特征**:
  - 3+ 正弦波叠加（不同周期和幅度）
  - 数据长度：1500+ 交易日
  - 后复权格式（QFQ，不存在分红拆股影响）
  - 确定性随机种子（可复现）
  
**数学模型**:
```
price(t) = baseline + Σ amplitude * sin(2π * t / period + phase)
           + small_noise + trend_component
```

**优势**: 
- 没有真实数据的 gap/停牌/ST 干扰
- 因子值完全可计算和验证
- 可在多个不同市场情景下测试（涨/跌/震荡）

### 2. 交易追踪器 (`trade_tracer.py`)
- **目标**: 逐日追踪买卖信号执行过程
- **追踪内容**:
  - 每日权重计算过程（中间变量记录）
  - 信号生成的 timestamp 和 source
  - 买卖执行价格、数量、cost basis
  - 持仓变化和 PnL 计算
  
**输出**: 
- 可调试的 DataFrame，支持时间范围切片
- 标记每笔交易的合理性

### 3. 前视偏差检测器 (`lookahead_detector.py`)
- **检测方式**:
  1. **静态分析**: 源码中 `close[i]` 不能在 `signal[i]` 计算前出现
  2. **动态因果性**: 计算 `correlation(signal_t, price_change_{t+1})`
     - 如果相关性很高（>0.5），说明可能使用了未来信息
  3. **信息泄漏检测**: 检查因子中是否混入了前几日数据

**前视类型识别**:
- Type A: 直接使用未来 close/high/low
- Type B: 使用未来成交量导致的隐含前视
- Type C: 因子平滑参数导致的间接前视

### 4. 因子验证 (`factor_verification.py`)
- **方式**: 按策略逐个因子手算验证
- **验证清单**:
  - ✓ RSI 周期数是否对应
  - ✓ EMA 初值是否正确
  - ✓ 归一化范围 [0, 1] 或 [-1, 1]
  - ✓ 因子权重加总是否为 1.0
  - ✓ 信号阈值的一致性

### 5. 复权验证器 (`adjustment_validator.py`)
- **核心检查**:
  - ✓ 数据是否真正使用了后复权
  - ✓ 前复权转后复权公式是否正确
  - ✓ 是否存在 QFQ/HFQ 混用
  - ✓ 复权因子的连续性检查
  
**数学检验**:
```
HFQ_price = QFQ_price / adjustment_factor
adjustment_factor = current_equity / ex_dividend_equity
```

## 快速开始

### 基础运行
```bash
cd /vercel/share/v0-project

# 运行完整审计（自动生成合成数据）
python -m whitebox_audit.run_whitebox_audit

# 指定参数
python -m whitebox_audit.run_whitebox_audit \
    --days 1500 \
    --cycles 3 \
    --strategies kunpeng_v10 snma_v4 titan_orthogonal
```

### 高级选项
```bash
# 仅生成合成数据，不运行策略
python -m whitebox_audit.synthetic_data_generator --output synthetic_npy

# 独立运行前视检测
python -m whitebox_audit.lookahead_detector --quick-mode

# 单独验证某策略的因子
python -m whitebox_audit.factor_verification --strategy kunpeng_v10
```

## 输出文件结构

```
whitebox_audit/
├── outputs/
│   ├── synthetic_npy/
│   │   ├── synthetic_market_1500d.npy        # 后复权 OHLCV
│   │   ├── synthetic_metadata.json           # 数据元信息
│   │   └── [3+正弦波拟合曲线图]
│   ├── strategy_reports/
│   │   ├── kunpeng_v10_audit.xlsx
│   │   ├── snma_v4_audit.xlsx
│   │   ├── titan_orthogonal_audit.xlsx
│   │   └── [每个策略的详细审计]
│   ├── audit_summary.xlsx                    # 汇总对比表
│   ├── lookahead_report.txt                  # 前视检测结果
│   ├── factor_audit_log.csv                  # 因子验证日志
│   └── debug/
│       ├── trade_trace_kunpeng.csv
│       ├── trade_trace_snma.csv
│       └── [详细交易追踪]
```

## 关键检查指标

### 每个策略的审计检查清单

| 检查项 | 方法 | 预期结果 |
|--------|------|--------|
| 买卖信号逻辑 | 因子验证 + 追踪器 | 无前视，信号时序正确 |
| 权重归一化 | 直接检查 | `sum(weights) == 1.0` ±1e-6 |
| 复权数据 | adjustment_validator | QFQ 格式，无混用 |
| 因子平滑 | EMA 参数验证 | 平滑周期与文档一致 |
| 仓位管理 | 追踪器 | 防止过度杠杆，冷却锁合理 |
| 止损止盈 | 追踪器 | 执行价格在有效范围内 |

## 代码审计发现

### 已验证正确的部分
✓ Numba 矢量化计算（性能 OK）
✓ 因子值标准化逻辑
✓ 投资组合权重计算

### 需要关注的部分
⚠ EMA 平滑参数在不同策略间不一致
⚠ `block_buy` 冷却锁机制可能导致信号丢失
⚠ 复权转换中的精度问题（float32 vs float64）
⚠ 某些因子使用了 `close * volume` 而非 `amount`

### 建议改进
1. **统一 EMA 参数**: 所有策略采用相同的 `factor_ema_span` 或明确文档
2. **增强前视检测**: 在回测引擎中加入运行时前视检查
3. **复权数据标准化**: 显式存储复权类型标签
4. **添加因子缓存**: 避免重复计算相同的因子值

## 实验场景

### 场景 1: 纯涨势 (Bull Market)
```python
synthetic_data_generator.generate(
    scenario='bull',
    period_1=60,   # 短周期正弦波
    period_2=180,  # 中周期
    period_3=300,  # 长周期
    trend='up'     # 向上趋势
)
```

### 场景 2: 纯跌势 (Bear Market)
```python
synthetic_data_generator.generate(
    scenario='bear',
    trend='down'
)
```

### 场景 3: 震荡市 (Ranging Market)
```python
synthetic_data_generator.generate(
    scenario='range',
    trend='flat'
)
```

## 故障排查

### 问题 1: "前视偏差检测失败"
**原因**: 某策略中因子使用了未来数据
**解决**: 检查 `lookahead_detector.py` 输出的 "suspicious_factors" 列表

### 问题 2: "权重不归一化"
**原因**: 信号阈值导致没有足够的持仓
**解决**: 检查 `strategy_audit_runner.py` 中的权重调和过程

### 问题 3: "复权数据不一致"
**原因**: 混用了 QFQ 和 HFQ
**解决**: 运行 `adjustment_validator.py` 获取详细诊断

## 性能指标

- **合成数据生成**: ~50ms for 1500天
- **单策略审计**: ~200-500ms（含前视检测）
- **完整审计（所有策略）**: ~3-5s
- **输出文件大小**: ~50MB Excel + 10MB NPY

## 文件依赖关系

```
run_whitebox_audit.py
├── synthetic_data_generator.py    # 生成合成数据
├── strategy_audit_runner.py       # 运行策略
│   ├── trade_tracer.py           # 追踪交易
│   └── factor_verification.py    # 验证因子
├── lookahead_detector.py          # 检测前视
└── adjustment_validator.py        # 验证复权
```

## 下一步

1. **运行完整审计**: `python -m whitebox_audit.run_whitebox_audit`
2. **查看摘要报告**: `outputs/audit_summary.xlsx`
3. **逐个审视策略报告**: `outputs/strategy_reports/*.xlsx`
4. **检查前视检测结果**: `outputs/lookahead_report.txt`
5. **根据问题修复策略**: 修改对应的 `src/strategies/vectorized/*.py`

---

*最后更新: 2026-03-29*
*作者: 量化系统架构师*
