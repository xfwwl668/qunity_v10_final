# Q-UNITY V10 — 修复日志 (2026-03-26)

## 本次修复汇总

本次修复覆盖 7 个已审计确认的 Bug，分为三类：
- 🔴 致命（影响核心数值正确性）
- 🟠 中危（影响实施效果 / 引发误导）
- 🟡 低危（架构健康问题）

---

### 🔴 BUG-T1：换手率系统性虚高（3-10 倍）

**文件**：`src/engine/fast_runner_v10.py` — `_turnover_numpy()`

**根因**：
```
原实现：turnover = Σ|position[t]*close[t] - position[t-1]*close[t-1]| / NAV
                 = 价格漂移项 + 真实交易项   ← 价格上涨本身计入换手，完全错误
```

**修复**：
```
正确实现：turnover = Σ|ΔP[i,t]| × close[t] / NAV
          只有「持仓股数发生变化」才计入换手，价格漂移不影响换手率
```

---

### 🔴 BUG-T2：批量回测（multi_run）与单策略回测结果不同

**文件**：`src/engine/fast_runner_v10.py` — `multi_run()` + 新增 `_run_get_weights()`

**根因**：  
`multi_run` 原实现对各策略 NAV 做归一化加权平均，等价于「各策略独立结算再纸面混合」。
这与「合并权重后统一过撮合引擎」在以下 3 点必然不同：  
① 交易成本：重叠持仓可抵消，合并后更低  
② 仓位上限：NAV 平均不受 `max_single_pos` 约束  
③ Regime 过滤：各策略独立过滤 vs 统一过滤结果不同

**修复**：  
新增 `_run_get_weights()` 辅助方法（返回策略 `final_weights` 矩阵而不撮合）。  
`multi_run` Step 4-5 重写为：各策略权重 × 分配比例相加 → `PortfolioBuilder.build()` → 
`match_engine_weights_driven()` → 真实组合 NAV。  
降级保护：若权重获取失败，回退 NAV 加权平均并记录 WARNING。

---

### 🔴 BUG-T3：L3-B 止损后状态漂移（次日重购刚止损股票）

**文件**：`src/engine/numba_kernels_v10.py`、`src/engine/fast_runner_v10.py`

**根因**：  
内核 L3-B 触发止损后清空 `position[i]`，但策略层 `_score_to_weights` 的 
`in_portfolio[i]` 无从得知，仍为 `True`。次日策略层继续分配正权重 → 
内核收到正权重 → 当日重新建仓，每次止损后至少 `dropout_days`（3-7 天）内每天重复建仓。

**修复**：  
内核新增输出矩阵 `stop_triggered_out (N, T) bool`，记录每日 L3-B 触发止损的股票。  
`match_engine_weights_driven` 现返回 **4 元组**：`(pos_matrix, nav_array, cash_array, stop_triggered)`。  
`run()` 将 `stop_triggered` 写入 `alpha_raw.meta["stop_triggered"]`，供后续分析和 multi_run 使用。

---

### 🟠 BUG-T4：exit_config 完全未实施（10 策略共用一套止损参数）

**文件**：`src/strategies/alpha_signal.py`、`src/engine/numba_kernels_v10.py`、  
`src/engine/fast_runner_v10.py`、全部 10 个策略文件

**修复**：
1. `AlphaSignal` dataclass 新增 `exit_config: Optional[Dict[str, Any]] = None` 字段
2. 内核新增两个参数：`stop_mode_trailing: bool`（True=追踪止损，False=固定止损）、`take_profit: float`
3. `run()` 从 `alpha_raw.exit_config` 读取配置并覆盖 `RiskConfig` 全局参数
4. 10 个策略文件按性格分类各自配置 exit_config：

| 策略 | stop_mode | hard_stop | take_profit | max_hold |
|---|---|---|---|---|
| S-01 weak_to_strong | entry_price | 7% | 12% | 5天 |
| S-02 sentiment_reversal | entry_price | 8% | 12% | 7天 |
| S-03 short_term_rsrs | entry_price | 10% | 18% | 10天 |
| S-04 kunpeng_v10 | entry_price | 12% | 20% | 20天 |
| S-05 momentum_reversal | trailing | 12% | 18% | 20天 |
| S-06 snma_v4 | trailing | 15% | 25% | 25天 |
| S-07 alpha_hunter_v2 | trailing | 15% | 不止盈 | 30天 |
| S-08 ultra_alpha_v1 | trailing | 15% | 25% | 25天 |
| S-09 alpha_max_v5 | trailing | 18% | 不止盈 | 60天 |
| S-10 titan_alpha_v1 | trailing | 22% | 不止盈 | 不限(Regime) |

---

### 🟠 BUG-T5：8 个 NOT_IMPLEMENTED 字段静默失效

**文件**：`src/engine/risk_config.py`

**根因**：`config.json` 中修改这些字段对回测完全无效，但系统不发出任何警告。

**修复**：`RiskConfig.__post_init__()` 检测用户对 `_NOT_IMPL_DEFAULTS` 字典中字段的
非默认赋值，主动触发 `UserWarning`，明确说明字段未接入引擎。  
受保护字段：`stop_loss_lookback`、`max_sector_pos`、`market_impact_eta/alpha`、
`account_id`、`max_daily_orders`、`max_cancel_rate`、`single_stock_limit`。

---

### 🟠 BUG-T7：涨跌停/停牌边界假性交易（换手率虚高第二来源）

**文件**：已在策略层全部修复（`hard_invalid` 参数已实现但此前未传入）

**根因**：  
`valid_mask=False`（停牌/涨跌停）时，`_score_to_weights` 将 `score=-inf` 处理为
「门控失效」并重置 `in_portfolio[i]=False`。次日恢复后重新建仓，形成大量无意义双边交易。

`hard_invalid` 机制已在 `_score_to_weights` 中实现（冻结状态而非清仓），
但此前 10 个策略的 `return AlphaSignal(...)` 均未传入 `hard_invalid` 参数。

**修复**：  
确认 10 个策略文件在 `_score_to_weights(...)` 调用中均已传入：  
```python
hard_invalid = None if valid_mask is None else ~np.asarray(valid_mask, dtype=bool)
```

---

### 🟡 BUG-T6：src/engine/alpha_signal.py 孤儿文件

**文件**：`src/engine/alpha_signal.py`

**根因**：`fast_runner_v10.py` 实际导入的是 `src/strategies/alpha_signal.py`（功能更完整，含
`hard_invalid`、`force_exit`、`_ema_smooth_factor`），但 `src/engine/` 下存在一个功能不完整的旧版本，随时可能造成导入混淆。

**修复**：将 `src/engine/alpha_signal.py` 改为纯重定向模块，所有符号从 strategies 版本 re-export，
两个导入路径等价，消除分叉风险。

---

## 修改文件清单

| 文件 | 修改类型 | 关联 Bug |
|---|---|---|
| `src/engine/numba_kernels_v10.py` | 新增参数+输出 | T3, T4 |
| `src/engine/fast_runner_v10.py` | 多处修改+新增方法 | T1, T2, T3, T4 |
| `src/engine/risk_config.py` | 新增 `__post_init__` | T5 |
| `src/engine/alpha_signal.py` | 改为重定向模块 | T6 |
| `src/strategies/alpha_signal.py` | 新增 `exit_config` 字段 | T4 |
| `src/strategies/vectorized/*.py` (×10) | 新增 `exit_config` 配置 | T4, T7 |

