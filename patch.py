import re

with open("src/strategies/alpha_signal.py", "r", encoding="utf-8") as f:
    text = f.read()

# Replace the inner part of _score_to_weights with a smoother allocation logic.
# The original code recalculates weight dynamically using `raw_w = 1.0 / k` where `k = tradeable.size`,
# which causes severe weight oscillation when `tradeable` count fluctuates day by day.
# Also original code excludes `hard_invalid` from weights, assigning them 0 weight, which implies forced selling in the engine!

new_logic = """    # [FIX-BUG1] 防抖状态变量
    in_portfolio  = np.zeros(N, dtype=np.bool_)   # 当前是否在持仓组合中
    absent_days   = np.zeros(N, dtype=np.int64)   # 连续缺席（超出缓冲区）天数
    last_target   = np.zeros(N, dtype=np.float64) # [增补] 上日目标权重

    for t in range(T):
        col       = score[:, t]
        valid     = ~np.isneginf(col)
        valid_idx = np.where(valid)[0]
        
        hi_t = hi[:, t] if hi is not None else None
        fe_t = fe[:, t] if fe is not None else None

        if valid_idx.size == 0:
            if hi_t is not None:
                for i in range(N):
                    if not hi_t[i]:
                        in_portfolio[i] = False
                        absent_days[i]  = 0
                        last_target[i]  = 0.0
                    elif in_portfolio[i]:
                        weights[i, t] = last_target[i]
            else:
                in_portfolio[:] = False
                absent_days[:]  = 0
                last_target[:]  = 0.0
            continue

        ranks = np.full(N, N + 1, dtype=np.int64)
        order = valid_idx[np.argsort(col[valid_idx])[::-1]]
        ranks[order] = np.arange(1, order.size + 1, dtype=np.int64)

        for i in range(N):
            if hi_t is not None and hi_t[i]:
                # 物理不可交易：冻结状态，并在今日保留昨天的目标权重
                if in_portfolio[i]:
                    weights[i, t] = last_target[i]
                continue

            if fe_t is not None and fe_t[i] and in_portfolio[i]:
                in_portfolio[i] = False
                absent_days[i]  = 0
                last_target[i]  = 0.0
                continue

            if not valid[i]:
                in_portfolio[i] = False
                absent_days[i]  = 0
                last_target[i]  = 0.0
                continue

            rank_i = int(ranks[i])

            if rank_i <= top_n:
                in_portfolio[i] = True
                absent_days[i]  = 0
            elif rank_i <= top_n + exit_buffer:
                absent_days[i] = 0
            else:
                if in_portfolio[i]:
                    absent_days[i] += 1
                    if absent_days[i] >= dropout_days:
                        in_portfolio[i] = False
                        absent_days[i]  = 0
                        last_target[i]  = 0.0

        active_idx = np.where(in_portfolio)[0]
        # 【修改点2】取消强制按 max_hold 踢人，这也会打断防抖状态！
        # 若需要保护持仓不崩，可以用基于基准体量的分配机制来应对
        
        active = np.where(in_portfolio)[0]
        if active.size == 0:
            continue

        # 【核心修正】固定等权基准为 min(1.0/top_n, max_single_pos) 
        # 而非随每天存活股票数波动，这彻底消隐了目标仓位的“每日微震荡”。
        # Note: A股中可买的数量即使很多，也不会超限太多因为有退出机制。
        # 如果当前持仓超过 top_n，总权重可能会短暂超出 1.0，但最后的归一化会平滑处理。
        
        base_w = min(1.0 / top_n, max_single_pos)
        
        for i in active:
            if hi_t is None or not hi_t[i]:
                weights[i, t] = base_w
                last_target[i] = base_w
        
        # 归一化兜底：如果总权重大于1.0，按比例缩小
        total_w = np.sum(weights[:, t])
        if total_w > 1.0:
            weights[:, t] /= total_w
            # 同步更新 last_target
            for i in active:
                 last_target[i] = weights[i, t]

    return weights"""

# We look for "# [FIX-BUG1] 防抖状态变量" and replace from there to the end before the tests.
match = re.search(r"    # \[FIX-BUG1\] 防抖状态变量.*?(?=\n\n# ──+)", text, re.DOTALL)
if match:
    new_text = text[:match.start()] + new_logic + text[match.end():]
    with open("src/strategies/alpha_signal.py", "w", encoding="utf-8") as f:
        f.write(new_text)
    print("Patch applied successfully.")
else:
    print("Could not find patch location.")
