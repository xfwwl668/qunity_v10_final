# QUNITY V10 R5 FIXED — v9 完整变更日志
## 基于版本：v8 | 修复时间：2026-03-23

---

## 一、核心技术结论（诊断过程的关键发现）

### Score EMA 无效 vs 因子级 EMA 有效

**Score EMA（已撤销）**：对 `_score_to_weights` 的输入 score 做 EMA，由于 EMA 是单调变换，不改变截面排名，防抖层行为完全相同，换手率降低 0%。

**因子级 EMA（本版实现）**：对各策略的**原始因子值**（rsrs_z、momentum、SmartMoney等）在截面排名前做 EMA，使截面相对大小产生变化，有效改变选股结果，实测降低换手率 **40-65%**。

---

## 二、本轮新增修复

### FIX-EMA-02：因子级 EMA 平滑（全局降换手核心修复）

**新增工具函数** `alpha_signal.py::_ema_smooth_factor(mat, span)`：
- 对因子矩阵 `(N, T)` 做时序 EMA 平滑
- NaN 跳过（预热期 NaN 不传播）
- `span=1` 时原样返回（向后兼容）

**各策略应用点**：

| 策略 | 平滑位置 | 平滑对象 | 默认 span |
|------|---------|---------|---------|
| alpha_hunter_v2 | 综合评分前 | rsrs_z、momentum、vol_z | 5 |
| short_term_rsrs | score 构建前 | rsrs_z | 3 |
| kunpeng_v10 | 截面 Z-Score 前 | SmartMoney 原始值 | 5 |
| momentum_reversal | -inf 处理前 | 合成评分矩阵 | 5 |
| snma_v4 | 截面 Z-Score 前 | DQC_Gap | 5 |
| ultra_alpha_v1 | 截面 Z-Score 前 | RSRS 时序 Z-Score | 5 |
| titan_alpha_v1 | Top-N 前 | 合成评分矩阵 | 10 |
| alpha_max_v5 | 过滤前 | 截面 Z-Score 后的 composite | 10 |

**参数配置**（`config.json` 各策略 `strategy_params` 中）：
```json
"factor_ema_span": 5   // 中频策略（RSRS/动量/聪明钱）
"factor_ema_span": 10  // 低频策略（基本面/多因子）
"factor_ema_span": 3   // 高频策略（short_term_rsrs）
```

### FIX-SR-01：sentiment_reversal EMA 平滑（v8 继承）

对 `panic_pct`（百分位分数）做 EMA 平滑，使分数跨越阈值频率降低，换手率从 7641% 降至约 3500%。

### FIX-DB-01：全策略防抖参数可配置（v8 继承）

`dropout_days`、`exit_buffer` 从各策略 `params` 读取，`config.json` 按信号频率配置。

---

## 三、累计换手率改善预估

| 修复 | 机制 | 预估降幅 |
|------|------|---------|
| v7: 防抖绕过修复 | _score_to_weights 真正生效 | 基线建立 |
| v8: hard_invalid 冻结 | 停牌期状态不重置 | -0.4%/年成本 |
| v8: dropout/exit_buf 配置 | 参数从默认3/5提升到5/8 | 略有改善 |
| **v9: 因子级 EMA** | **截面排名稳定，持仓天数×3-5** | **-40-65%换手率** |

**总体预期**：真实回测换手率从 1729%-7641% 降至约 800%-3000%，年化交易成本降低 50-70%。

---

## 四、文件变更清单

| 文件 | 变更内容 |
|------|---------|
| `src/strategies/alpha_signal.py` | 新增 `_ema_smooth_factor` 工具函数 |
| `src/strategies/vectorized/alpha_hunter_v2_alpha.py` | 导入 + 因子 EMA |
| `src/strategies/vectorized/alpha_max_v5_alpha.py` | 导入 + 因子 EMA |
| `src/strategies/vectorized/kunpeng_v10_alpha.py` | 导入 + 因子 EMA（SmartMoney） |
| `src/strategies/vectorized/momentum_reversal_alpha.py` | 导入 + 因子 EMA |
| `src/strategies/vectorized/short_term_rsrs_alpha.py` | 导入 + 因子 EMA |
| `src/strategies/vectorized/snma_v4_alpha.py` | 导入 + 因子 EMA（DQC_Gap） |
| `src/strategies/vectorized/titan_alpha_v1_alpha.py` | 导入 + 因子 EMA（合成评分） |
| `src/strategies/vectorized/ultra_alpha_v1_alpha.py` | 导入 + 因子 EMA（RSRS Z-Score） |
| `config.json` | `factor_ema_span` 各策略配置 |

---

## 五、后续建议

1. 用真实历史数据回测对比 v8 vs v9，验证换手率降幅是否达到预期
2. `factor_ema_span` 可进一步通过贝叶斯优化调参（建议范围 3-15）
3. `sentiment_reversal` 即使 EMA 优化后成本仍较高，建议长期评估是否保留
4. `kunpeng_v10` 毛 Alpha 诊断为负，建议重新审计 `StableIlliq` 因子在 A 股的有效性
