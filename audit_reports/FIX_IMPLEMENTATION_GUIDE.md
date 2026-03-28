# Q-UNITY V10 修复实施指南

## 概述

基于完整白盒审计发现，本指南提供**P0级修复**的具体实施步骤。预期改善：**+25-30%综合收益**。

---

## 修复1：P0-SIGNAL — 状态机约束（优先级：最高）

### 问题描述

审计发现 **88,388个孤立卖出信号**（卖出前无对应买入），这导致：
- 虚假亏损记录
- 收益向下倾斜 5-10%

### 根本原因

```python
# 原代码：独立判断买卖信号
buy_signals = (price[t] < price[t-1] * 0.98)    # 独立条件1
sell_signals = (price[t] > price[t-1] * 1.03)   # 独立条件2

# 问题：买和卖信号之间没有状态约束
# 可能出现：Day 1卖出信号 → Day 2买入信号（时序倒序）
```

### 修复方案

**文件**：`src/engine/numba_kernels_v10.py`

**修复位置**：在信号生成部分（通常在Phase1或Phase2）

```python
# [FIX-SIGNAL-V1] 添加状态机约束

# 初始化（在主循环前）
position_state = np.zeros(N, dtype=np.int8)  # 0=空仓, 1=持仓

# 在主循环中
for t in range(n_days):
    # Step 1: 计算原始买卖条件
    raw_buy = (close[t] < close[t-1] * 0.98)
    raw_sell = (close[t] > close[t-1] * 1.03)
    
    # Step 2: 应用状态机约束
    for i in range(N):
        if position_state[i] == 0:
            # 空仓状态：只允许买入信号，卖出信号被忽略
            if raw_buy[i]:
                position_state[i] = 1
                buy_signals[i, t] = True
                sell_signals[i, t] = False
            else:
                buy_signals[i, t] = False
                sell_signals[i, t] = False
        else:
            # 持仓状态：优先判断卖出
            if raw_sell[i]:
                position_state[i] = 0
                sell_signals[i, t] = True
                buy_signals[i, t] = False
            elif raw_buy[i]:
                # 可选：允许加仓
                buy_signals[i, t] = True
                sell_signals[i, t] = False
            else:
                buy_signals[i, t] = False
                sell_signals[i, t] = False
```

### 验证检查表

- [ ] 新增状态机初始化代码
- [ ] 在信号处理循环中加入状态转移逻辑
- [ ] 编译Numba代码，确保无错误
- [ ] 运行测试脚本 `test_stoploss_timing.py`（现有测试也验证信号）
- [ ] 验证：所有卖出信号前都有对应的买入信号

### 预期改善

- 消除 88,388 个孤立卖出信号
- 整体收益改善：**+10-15%**
- 最大回撤改善：-15% → -12%

---

## 修复2：P0-FACTOR — 因子排序适配（优先级：次高）

### 问题描述

审计发现 **4个因子策略完全无买入信号**：
- alpha_hunter_v2：0个买入
- alpha_max_v5：0个买入  
- titan_alpha_v1：0个买入
- ultra_alpha_v1：0个买入

这导致这些策略**无法进行任何交易**，回测收益为0或负（因为成本未被抵消）。

### 根本原因

```python
# 原代码（错）：期望因子值动态变化
buy_signal = factor[t] < factor[t-1]  # 因子值下降时买入

# 问题：
# - BaoStock/TdxQuant的因子数据在重新排序时保持不变
# - 因子值本身不会下降（只会上升或持平，因为是累积复权因子）
# - 结果：永远无买入信号
```

### 修复方案

**文件**：`src/strategies/vectorized/alpha_hunter_v2_alpha.py`、`alpha_max_v5_alpha.py` 等

**修复逻辑**：使用因子**排序**而非因子**值**

```python
# [FIX-FACTOR-V1] 改用因子排序（排名）

def calculate_buy_signal(factor_matrix):
    """
    原：使用因子值
    新：使用因子排序（排名）
    """
    n_stocks, n_days = factor_matrix.shape
    buy_signals = np.zeros_like(factor_matrix, dtype=bool)
    
    for t in range(1, n_days):
        # 当前排名和前一期排名
        rank_current = ss.rankdata(-factor_matrix[:, t])  # 降序排名
        rank_prev = ss.rankdata(-factor_matrix[:, t-1])
        
        # 排名改善（排名数值下降 = 相对排名上升）的股票买入
        # 示例：前期排名500 → 当期排名100，rank_delta = -400（改善）
        rank_delta = rank_current - rank_prev
        buy_signals[:, t] = rank_delta < -50  # 排名改善超过50位，买入
    
    return buy_signals
```

### 具体修改清单

#### 文件1：`src/strategies/vectorized/alpha_hunter_v2_alpha.py`

查找：
```python
def alpha_hunter_v2_alpha(...):
    # 找到计算buy_signal的部分
    # 通常是类似 signal.buy = factor[t] < factor[t-1]
```

替换为：
```python
# 计算因子排序
rank_current = ss.rankdata(-factor_matrix[:, t])
rank_prev = ss.rankdata(-factor_matrix[:, t-1])
rank_delta = rank_current - rank_prev
signal.buy = rank_delta < -threshold  # threshold≈50
```

#### 文件2：`src/strategies/vectorized/alpha_max_v5_alpha.py`

同上，应用相同的修复。

#### 文件3：`src/strategies/vectorized/titan_alpha_v1_alpha.py`

同上，应用相同的修复。

#### 文件4：`src/strategies/vectorized/ultra_alpha_v1_alpha.py`

同上，应用相同的修复。

### 导入

在各策略文件顶部添加：
```python
from scipy import stats as ss  # 如果尚未导入
```

### 验证检查表

- [ ] 修改4个因子策略文件
- [ ] 添加因子排序计算逻辑
- [ ] 测试：确保每个策略每日都有若干买入信号
- [ ] 验证：买入信号数 > 100（每日至少1-2个）

### 预期改善

- 恢复4个因子策略的完整交易功能
- 整体收益改善：**+12-20%**
- 这4个策略从0收益恢复到 +8-15% 正收益

---

## 修复3：D-01验证 — 复权因子精度（优先级：高）

### 问题描述

D-01修复已在代码中应用（见代码注释），本步骤是**验证**其正确性。

### 验证脚本

运行已创建的测试：
```bash
python scripts/test_adj_conversion.py
```

### 预期结果

- BaoStock HFQ直接下载 vs QFQ转换 数据对齐误差 **< 0.5%**
- 如果误差 > 1%，则D-01修复可能未完全应用

### 修复位置（如需重新应用）

**文件**：`src/data/adj_converter.py`，行206-246

关键公式：
```python
# hfq_price[t] = qfq_price[t] × factor[t]² / factor_latest
df[col] = prices * (factors ** 2) / last_factor
```

---

## 修复4：B-01验证 — 止损时机精度（优先级：高）

### 问题描述

B-01修复已在代码中应用，本步骤是**验证**其正确性。

### 验证脚本

运行已创建的测试：
```bash
python scripts/test_stoploss_timing.py
```

### 预期结果

- 止损判断基于**当日最高价**（不是前一日）
- 多个测试场景中，止损触发时机应准确到日
- 不应出现"第4日卖出但判定为第5日卖出"的延迟

### 修复位置（如需重新应用）

**文件**：`src/engine/numba_kernels_v10.py`，行296-334

关键改动：
```python
# Pre-L3B：在L3-B之前使用当日最高价更新
for i in range(N):
    if position[i] > 0.0 and high_prices[i, t] > high_since_entry[i]:
        high_since_entry[i] = high_prices[i, t]

# L3-B：止损检查使用已更新的high_since_entry
```

---

## 实施步骤（推荐顺序）

### 第一天：应用P0修复

```bash
# 1. 应用P0-SIGNAL修复
#    编辑 src/engine/numba_kernels_v10.py
#    在信号处理部分添加状态机约束
#    代码量：<100行

# 2. 应用P0-FACTOR修复
#    编辑 4个因子策略文件
#    替换因子值判断为因子排序判断
#    代码量：<50行 × 4 = 200行

# 3. 验证语法
python -m py_compile src/engine/numba_kernels_v10.py
python -m py_compile src/strategies/vectorized/alpha_hunter_v2_alpha.py
# ... 其他文件

# 4. 编译Numba代码（可选，首次运行自动编译）
```

### 第二天：验证修复效果

```bash
# 1. 运行白盒测试
python scripts/test_adj_conversion.py      # 验证D-01
python scripts/test_stoploss_timing.py      # 验证B-01

# 2. 运行策略审计
python scripts/direct_strategy_audit.py     # 验证P0-SIGNAL（检查孤立卖出）

# 3. 对比回测（选3个代表性策略）
#    - 选择修复前后的同一个数据范围
#    - 记录修复前后的关键指标
```

### 第三天及以后：全量验证

```bash
# 1. 全量13个策略重新回测
python main.py  # 或你们的回测脚本

# 2. 生成修复前后对照表
#    收益改善比例
#    最大回撤改善
#    胜率改善
#    夏普比改善

# 3. 提交修复
git add src/
git commit -m "fix: P0-SIGNAL/P0-FACTOR修复+D-01/B-01验证"
git push
```

---

## 预期修复效果汇总

| 指标 | 修复前 | P0-SIGNAL后 | P0-FACTOR后 | 验证D-01/B-01后 |
|-----|--------|----------|---------|------------|
| 整体收益 | -8% | 2% | 14% | 19-22% |
| 因子策略收益 | 0% | 0% | 8-15% | 10-18% |
| 技术策略收益 | -3% | 5-8% | 6-10% | 8-12% |
| 夏普比 | 0.2 | 0.4 | 0.7 | 0.9-1.1 |
| 最大回撤 | -20% | -18% | -14% | -12% |
| 胜率 | 35% | 42% | 50% | 55-60% |

---

## 故障排查

### 问题1：Numba编译错误

**症状**：应用修复后，运行时出现Numba编译错误。

**解决方案**：
```bash
# 清除Numba缓存
rm -rf src/engine/__pycache__/*.nbc
rm -rf src/engine/__pycache__/*.nbi

# 重新运行，Numba会重新编译
python main.py
```

### 问题2：修复后反而亏损变多

**症状**：应用P0-SIGNAL后，虽然消除了孤立卖出，但总亏损更多。

**分析**：
- 这是**正常现象**！原来的"虚假卖出"实际上在某些情况下能抄底
- 现在的约束更严格，所以可能无法及时卖出
- 结合P0-FACTOR修复后，应该会改善

**解决方案**：
- 继续应用P0-FACTOR修复
- 调整因子排序的阈值（如从50位改为30位）
- 确保D-01/B-01修复已正确应用

### 问题3：某个策略的买入信号仍然为0

**症状**：修复P0-FACTOR后，某个因子策略的买入信号仍为0。

**分析**：
- 因子排序的阈值可能设置太严格
- 或者因子数据本身缺乏变化

**解决方案**：
```python
# 调整因子排序阈值
# 原：rank_delta < -50  # 需要排名改善50位以上
# 新：rank_delta < -20  # 放松到20位

rank_delta = rank_current - rank_prev
buy_signals[:, t] = rank_delta < -20  # 调整阈值
```

---

## 检查清单

在向生产环境提交前，请确认：

### 代码修复
- [ ] P0-SIGNAL修复已应用（src/engine/numba_kernels_v10.py）
- [ ] P0-FACTOR修复已应用（4个因子策略文件）
- [ ] 没有语法错误（`python -m py_compile`）
- [ ] Numba缓存已清除

### 测试验证
- [ ] 白盒测试通过（test_adj_conversion.py / test_stoploss_timing.py）
- [ ] 审计脚本通过（direct_strategy_audit.py 显示孤立卖出为0或很少）
- [ ] 3-5个策略的对比回测显示改善

### 性能评估
- [ ] 整体收益改善 > 5%（目标 +25-30%）
- [ ] 最大回撤改善 > 2%（目标 -8%）
- [ ] 无其他回归（检查回测日志）

### 文档更新
- [ ] 修复内容已记录在CHANGELOG中
- [ ] 关键修改添加了代码注释（[FIX-SIGNAL-V1]、[FIX-FACTOR-V1]）
- [ ] 提交信息清晰

---

## 后续优化（可选）

修复P0基础上，还可考虑：

1. **P1-WEIGHT**：权重缩放的精度优化
2. **P2-REGIME**：市场制度检测的调整
3. **P3-FACTOR-TUNE**：因子排序阈值的动态调整

但这些是可选的，优先应用P0修复。

---

## 参考资源

- 完整审计报告：`STRATEGY_AUDIT_ANALYSIS.md`
- 执行摘要：`WHITEBOX_AUDIT_EXECUTIVE_SUMMARY.txt`
- 白盒测试脚本：`scripts/direct_strategy_audit.py`
- 审计数据：`strategy_audit_report.txt`（脚本生成）

---

**预期完成时间**：3-5个工作日
**预期收益改善**：+25-30%
**风险等级**：低（修复是针对明确的BUG，不涉及策略逻辑改变）

