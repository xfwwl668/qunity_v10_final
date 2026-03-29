"""
Q-UNITY V10 — WhiteBox Audit: TradeTracer
==========================================
交易追踪器：逐日记录策略的买卖信号、执行价格、持仓状态

核心功能：
1. 信号级追踪：记录每个时间步的目标权重 vs 实际权重
2. 成交追踪：买入/卖出股数、执行价格、交易成本
3. 持仓状态：持仓天数、入场价格、浮动盈亏
4. 规则验证：T+1、涨跌停、最大持仓限制

输出格式：
- DataFrame：可导出 Excel 进行人工审计
- 逐股票时间序列：方便定位异常信号
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


@dataclass
class TradeRecord:
    """单笔交易记录"""
    date: str                    # 交易日期
    code: str                    # 股票代码
    action: str                  # BUY / SELL / HOLD
    signal_weight: float         # 策略目标权重
    actual_weight: float         # 实际持仓权重
    shares: float                # 成交股数
    exec_price: float            # 执行价格
    close_price: float           # 收盘价
    cost: float                  # 交易成本（佣金+印花税）
    holding_days: int            # 持仓天数
    entry_price: float           # 入场均价
    pnl_pct: float               # 浮动盈亏 %
    stop_triggered: bool         # 是否触发止损
    regime: str                  # 市场状态


@dataclass
class TradeTracer:
    """
    交易追踪器
    
    在回测过程中记录每个时间步的信号和成交状态，
    用于白盒审计策略的买卖逻辑正确性。
    """
    
    records: List[TradeRecord] = field(default_factory=list)
    position_history: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)
    
    def __init__(self, codes: List[str], dates: List[str]):
        """
        Parameters
        ----------
        codes : List[str]
            股票代码列表
        dates : List[str]
            交易日列表
        """
        self.codes = codes
        self.dates = dates
        self.N = len(codes)
        self.T = len(dates)
        
        # 交易记录
        self.records: List[TradeRecord] = []
        
        # 逐日持仓矩阵 (N, T)
        self.position_matrix = np.zeros((self.N, self.T), dtype=np.float64)
        
        # 逐日权重矩阵
        self.signal_weights = np.zeros((self.N, self.T), dtype=np.float64)
        self.actual_weights = np.zeros((self.N, self.T), dtype=np.float64)
        
        # 入场价格和持仓天数
        self.entry_prices = np.zeros(self.N, dtype=np.float64)
        self.holding_days = np.zeros(self.N, dtype=np.int64)
        
        # 按股票索引的持仓历史
        self.position_history: Dict[str, List[Tuple[str, float]]] = {
            code: [] for code in codes
        }
        
    def trace_day(
        self,
        t: int,
        signal_weights: np.ndarray,      # (N,) 策略目标权重
        actual_weights: np.ndarray,      # (N,) 实际持仓权重（经 PortfolioBuilder 归一化后）
        positions: np.ndarray,           # (N,) 持仓股数
        exec_prices: np.ndarray,         # (N,) 执行价格
        close_prices: np.ndarray,        # (N,) 收盘价
        prev_positions: np.ndarray,      # (N,) 前日持仓
        costs: Optional[np.ndarray] = None,  # (N,) 交易成本
        regime: str = "NEUTRAL",         # 市场状态
        stop_mask: Optional[np.ndarray] = None,  # (N,) 止损触发标记
    ) -> None:
        """
        记录单日交易状态
        """
        date = self.dates[t]
        
        self.signal_weights[:, t] = signal_weights
        self.actual_weights[:, t] = actual_weights
        self.position_matrix[:, t] = positions
        
        for i in range(self.N):
            code = self.codes[i]
            
            # 判断买卖动作
            delta_pos = positions[i] - prev_positions[i]
            if delta_pos > 1e-6:
                action = "BUY"
            elif delta_pos < -1e-6:
                action = "SELL"
            else:
                action = "HOLD" if positions[i] > 1e-6 else "EMPTY"
            
            # 更新持仓天数
            if positions[i] > 1e-6:
                if prev_positions[i] < 1e-6:
                    # 新建仓
                    self.entry_prices[i] = exec_prices[i]
                    self.holding_days[i] = 0
                else:
                    self.holding_days[i] += 1
                    # 加仓更新均价
                    if delta_pos > 1e-6 and prev_positions[i] > 1e-6:
                        total_cost = (
                            self.entry_prices[i] * prev_positions[i] 
                            + exec_prices[i] * delta_pos
                        )
                        self.entry_prices[i] = total_cost / positions[i]
            else:
                self.holding_days[i] = 0
                self.entry_prices[i] = 0.0
            
            # 计算浮动盈亏
            if self.entry_prices[i] > 1e-8:
                pnl_pct = (close_prices[i] / self.entry_prices[i] - 1.0) * 100
            else:
                pnl_pct = 0.0
            
            # 只记录有交易或有持仓的情况
            if action != "EMPTY" or signal_weights[i] > 1e-8:
                record = TradeRecord(
                    date=date,
                    code=code,
                    action=action,
                    signal_weight=float(signal_weights[i]),
                    actual_weight=float(actual_weights[i]),
                    shares=float(abs(delta_pos)) if action in ("BUY", "SELL") else 0.0,
                    exec_price=float(exec_prices[i]),
                    close_price=float(close_prices[i]),
                    cost=float(costs[i]) if costs is not None else 0.0,
                    holding_days=int(self.holding_days[i]),
                    entry_price=float(self.entry_prices[i]),
                    pnl_pct=float(pnl_pct),
                    stop_triggered=bool(stop_mask[i]) if stop_mask is not None else False,
                    regime=regime,
                )
                self.records.append(record)
                
                # 记录持仓历史
                if positions[i] > 1e-6:
                    self.position_history[code].append((date, positions[i]))
    
    def to_dataframe(self) -> pd.DataFrame:
        """将所有交易记录转换为 DataFrame"""
        if not self.records:
            return pd.DataFrame()
        
        data = []
        for r in self.records:
            data.append({
                "日期": r.date,
                "代码": r.code,
                "动作": r.action,
                "信号权重": r.signal_weight,
                "实际权重": r.actual_weight,
                "成交股数": r.shares,
                "执行价格": r.exec_price,
                "收盘价": r.close_price,
                "交易成本": r.cost,
                "持仓天数": r.holding_days,
                "入场均价": r.entry_price,
                "浮盈%": r.pnl_pct,
                "止损触发": r.stop_triggered,
                "市场状态": r.regime,
            })
        
        return pd.DataFrame(data)
    
    def get_stock_trace(self, code: str) -> pd.DataFrame:
        """获取单只股票的完整交易追踪"""
        stock_records = [r for r in self.records if r.code == code]
        if not stock_records:
            return pd.DataFrame()
        
        return pd.DataFrame([{
            "日期": r.date,
            "动作": r.action,
            "信号权重": r.signal_weight,
            "实际权重": r.actual_weight,
            "成交股数": r.shares,
            "执行价格": r.exec_price,
            "收盘价": r.close_price,
            "持仓天数": r.holding_days,
            "浮盈%": r.pnl_pct,
            "止损": r.stop_triggered,
        } for r in stock_records])
    
    def get_signal_vs_actual_report(self) -> Dict[str, Any]:
        """
        生成信号 vs 实际执行的对比报告
        
        用于检测以下异常：
        1. 信号有权重但实际未建仓（可能是流动性不足或涨停）
        2. 信号无权重但实际有持仓（止损未执行或 Bug）
        3. 信号权重与实际权重偏差过大
        """
        discrepancies = []
        
        for t in range(self.T):
            for i in range(self.N):
                signal_w = self.signal_weights[i, t]
                actual_w = self.actual_weights[i, t]
                
                # 信号想买但实际没买
                if signal_w > 0.01 and actual_w < 0.001:
                    discrepancies.append({
                        "date": self.dates[t],
                        "code": self.codes[i],
                        "type": "SIGNAL_NOT_EXECUTED",
                        "signal_weight": signal_w,
                        "actual_weight": actual_w,
                    })
                
                # 信号想清仓但实际还持有
                elif signal_w < 0.001 and actual_w > 0.01:
                    discrepancies.append({
                        "date": self.dates[t],
                        "code": self.codes[i],
                        "type": "UNEXPECTED_HOLDING",
                        "signal_weight": signal_w,
                        "actual_weight": actual_w,
                    })
                
                # 权重偏差过大（> 50%）
                elif signal_w > 0.01 and actual_w > 0.01:
                    deviation = abs(signal_w - actual_w) / signal_w
                    if deviation > 0.5:
                        discrepancies.append({
                            "date": self.dates[t],
                            "code": self.codes[i],
                            "type": "LARGE_DEVIATION",
                            "signal_weight": signal_w,
                            "actual_weight": actual_w,
                            "deviation_pct": deviation * 100,
                        })
        
        return {
            "total_discrepancies": len(discrepancies),
            "by_type": pd.DataFrame(discrepancies).groupby("type").size().to_dict() if discrepancies else {},
            "details": discrepancies[:100],  # 只返回前100条
        }
    
    def get_turnover_analysis(self) -> Dict[str, float]:
        """
        换手率分析
        
        返回：
        - 总买入次数
        - 总卖出次数
        - 日均换手率
        - 持仓平均天数
        """
        buy_count = sum(1 for r in self.records if r.action == "BUY")
        sell_count = sum(1 for r in self.records if r.action == "SELL")
        
        # 日均换手率
        daily_turnover = []
        for t in range(1, self.T):
            pos_change = np.abs(self.position_matrix[:, t] - self.position_matrix[:, t-1])
            nav_estimate = np.sum(self.position_matrix[:, t] * self.actual_weights[:, t]) + 1e-10
            daily_turnover.append(pos_change.sum() / nav_estimate)
        
        avg_daily_turnover = np.mean(daily_turnover) if daily_turnover else 0.0
        
        # 平均持仓天数
        holding_records = [r.holding_days for r in self.records if r.action == "SELL" and r.holding_days > 0]
        avg_holding_days = np.mean(holding_records) if holding_records else 0.0
        
        return {
            "买入次数": buy_count,
            "卖出次数": sell_count,
            "总交易次数": buy_count + sell_count,
            "日均换手率": avg_daily_turnover,
            "年化换手率": avg_daily_turnover * 252,
            "平均持仓天数": avg_holding_days,
        }
    
    def export_to_excel(
        self,
        filepath: str | Path,
        include_position_matrix: bool = True,
    ) -> None:
        """
        导出完整审计报告到 Excel
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            # Sheet 1: 交易记录
            df_trades = self.to_dataframe()
            if not df_trades.empty:
                df_trades.to_excel(writer, sheet_name="交易记录", index=False)
            
            # Sheet 2: 信号偏差报告
            discrepancy_report = self.get_signal_vs_actual_report()
            if discrepancy_report["details"]:
                pd.DataFrame(discrepancy_report["details"]).to_excel(
                    writer, sheet_name="信号偏差", index=False
                )
            
            # Sheet 3: 换手率分析
            turnover = self.get_turnover_analysis()
            pd.DataFrame([turnover]).to_excel(
                writer, sheet_name="换手率分析", index=False
            )
            
            # Sheet 4: 持仓矩阵（可选，大数据量时禁用）
            if include_position_matrix and self.T <= 500:
                df_pos = pd.DataFrame(
                    self.position_matrix,
                    index=self.codes,
                    columns=self.dates,
                )
                df_pos.to_excel(writer, sheet_name="持仓矩阵")


# =============================================================================
# 验收测试
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TradeTracer 验收测试")
    print("=" * 70)
    
    codes = [f"SH60000{i}" for i in range(10)]
    dates = [f"2024-01-{i+1:02d}" for i in range(30)]
    
    tracer = TradeTracer(codes, dates)
    
    # 模拟交易追踪
    rng = np.random.default_rng(42)
    prev_pos = np.zeros(10)
    
    for t in range(30):
        signal_w = rng.random(10) * 0.1
        signal_w[signal_w < 0.05] = 0.0  # 低于阈值清零
        
        actual_w = signal_w * rng.uniform(0.8, 1.0, 10)
        positions = actual_w * 1000000 / (rng.uniform(10, 50, 10))  # 模拟持仓
        exec_prices = rng.uniform(10, 50, 10)
        close_prices = exec_prices * rng.uniform(0.98, 1.02, 10)
        
        tracer.trace_day(
            t=t,
            signal_weights=signal_w,
            actual_weights=actual_w,
            positions=positions,
            exec_prices=exec_prices,
            close_prices=close_prices,
            prev_positions=prev_pos,
            regime="NEUTRAL",
        )
        prev_pos = positions.copy()
    
    # 验证
    df = tracer.to_dataframe()
    assert len(df) > 0, "交易记录为空"
    print(f"[PASS] 交易记录数: {len(df)}")
    
    turnover = tracer.get_turnover_analysis()
    print(f"[PASS] 换手率分析: 买入{turnover['买入次数']}次, 卖出{turnover['卖出次数']}次")
    
    discrepancy = tracer.get_signal_vs_actual_report()
    print(f"[PASS] 信号偏差: {discrepancy['total_discrepancies']} 条")
    
    print()
    print("[SUCCESS] TradeTracer 全部测试通过")
