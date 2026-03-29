"""
Q-UNITY V10 — WhiteBox Audit Suite
===================================
专业量化策略审计框架

审计目标：
1. 验证每个策略的因子计算与买卖逻辑的正确性
2. 检测前视偏差（Look-Ahead Bias）
3. 验证后复权数据一致性
4. 生成可验证的合成数据（正弦波、趋势、震荡）
5. 提供逐日买卖信号与执行价格的完整追踪

作者：Q-UNITY 审计组
版本：V1.0
"""

__version__ = "1.0.0"
__all__ = [
    "SyntheticDataGenerator",
    "StrategyAuditRunner", 
    "TradeTracer",
    "LookAheadBiasDetector",
    "AdjustmentValidator",
]
