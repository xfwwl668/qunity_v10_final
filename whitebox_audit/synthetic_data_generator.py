"""
Q-UNITY V10 — WhiteBox Audit: SyntheticDataGenerator
=====================================================
生成可预测的合成数据用于策略白盒验证

核心设计原则：
1. 后复权数据：所有价格序列模拟后复权（HFQ），首日基准价为最低
2. 多周期正弦波：至少3个完整正弦周期（牛-熊-牛）用于验证策略在不同市况下的行为
3. 1500D+：数据长度 >= 1500 交易日（约6年），覆盖完整牛熊周期
4. 确定性随机：固定种子，确保可复现

数据类型：
- SINUSOIDAL: 多周期正弦波（可验证趋势跟踪策略）
- TRENDING_UP: 持续上涨趋势（可验证动量策略）
- TRENDING_DOWN: 持续下跌趋势（可验证反转策略）
- MEAN_REVERT: 均值回归震荡（可验证震荡策略）
- RANDOM_WALK: 随机游走（基准噪声数据）
"""

from __future__ import annotations

import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field


@dataclass
class SyntheticMeta:
    """合成数据元信息"""
    shape: Tuple[int, int]           # (N, T)
    dates: List[str]                 # 交易日序列
    codes: List[str]                 # 股票代码
    price_type: str                  # 价格类型 "hfq"
    data_mode: str                   # 数据模式
    sine_cycles: int                 # 正弦周期数
    seed: int                        # 随机种子
    base_price: float                # 基准价格
    

class SyntheticDataGenerator:
    """
    合成数据生成器
    
    用于生成可预测的市场数据，白盒验证策略逻辑。
    所有价格序列模拟后复权（HFQ）格式：首日价格为基准，历史不回调。
    """
    
    DEFAULT_T = 1500      # 默认1500交易日
    DEFAULT_N = 100       # 默认100只股票
    DEFAULT_SEED = 42     # 默认随机种子
    DEFAULT_CYCLES = 3    # 默认3个正弦周期
    
    def __init__(
        self,
        N: int = DEFAULT_N,
        T: int = DEFAULT_T,
        seed: int = DEFAULT_SEED,
        base_price: float = 10.0,
        volatility: float = 0.02,
    ):
        """
        Parameters
        ----------
        N : int
            股票数量（默认 100）
        T : int
            交易日数量（默认 1500，约6年）
        seed : int
            随机种子（确保可复现）
        base_price : float
            基准价格（后复权起点）
        volatility : float
            日波动率标准差
        """
        self.N = N
        self.T = T
        self.seed = seed
        self.base_price = base_price
        self.volatility = volatility
        self.rng = np.random.default_rng(seed)
        
        # 生成元数据
        self.dates = self._generate_dates()
        self.codes = [f"SH{600000 + i:06d}" for i in range(N)]
        
    def _generate_dates(self) -> List[str]:
        """生成交易日序列（模拟A股交易日）"""
        import datetime
        start = datetime.date(2018, 1, 2)  # 起始日期
        dates = []
        d = start
        while len(dates) < self.T:
            # 跳过周末
            if d.weekday() < 5:
                dates.append(d.strftime("%Y-%m-%d"))
            d += datetime.timedelta(days=1)
        return dates
    
    # =========================================================================
    # 核心数据生成方法
    # =========================================================================
    
    def generate_sinusoidal(
        self,
        cycles: int = 3,
        trend: float = 0.0001,
        amplitude_range: Tuple[float, float] = (0.3, 0.5),
        phase_shift_range: Tuple[float, float] = (0.0, 2 * np.pi),
    ) -> Dict[str, np.ndarray]:
        """
        生成多周期正弦波数据（模拟牛熊周期）
        
        公式：
            price[t] = base × (1 + A×sin(2π×cycles×t/T + φ)) × (1 + trend)^t × (1 + ε_t)
        
        其中：
            A ∈ amplitude_range（振幅因子）
            φ ∈ phase_shift_range（相位偏移，制造股票间差异）
            ε_t ~ N(0, volatility²)（日噪声）
        
        Parameters
        ----------
        cycles : int
            正弦波周期数（默认 3）
        trend : float
            长期趋势漂移（默认 0.0001，年化约 2.5%）
        amplitude_range : Tuple[float, float]
            振幅范围
        phase_shift_range : Tuple[float, float]
            相位偏移范围
            
        Returns
        -------
        data : Dict[str, np.ndarray]
            包含 close, open, high, low, volume, amount 的字典
        """
        t_arr = np.arange(self.T, dtype=np.float64)  # (T,)
        
        # 每只股票的振幅和相位
        amplitudes = self.rng.uniform(
            amplitude_range[0], amplitude_range[1], size=self.N
        )
        phases = self.rng.uniform(
            phase_shift_range[0], phase_shift_range[1], size=self.N
        )
        
        # 基础正弦波 (N, T)
        sine_wave = np.sin(2 * np.pi * cycles * t_arr[np.newaxis, :] / self.T + phases[:, np.newaxis])
        
        # 带振幅的正弦波
        price_factor = 1.0 + amplitudes[:, np.newaxis] * sine_wave
        
        # 长期趋势
        trend_factor = (1.0 + trend) ** t_arr[np.newaxis, :]
        
        # 日噪声
        noise = 1.0 + self.rng.normal(0, self.volatility, size=(self.N, self.T))
        
        # 后复权收盘价：首日最低，后续累积
        close = self.base_price * price_factor * trend_factor * noise
        close = np.maximum(close, 0.1)  # 防止负价格
        
        # OHLV 派生
        data = self._derive_ohlcv(close)
        
        # 元信息
        data["_meta"] = SyntheticMeta(
            shape=(self.N, self.T),
            dates=self.dates,
            codes=self.codes,
            price_type="hfq",
            data_mode="sinusoidal",
            sine_cycles=cycles,
            seed=self.seed,
            base_price=self.base_price,
        )
        
        return data
    
    def generate_trending(
        self,
        direction: str = "up",
        daily_return: float = 0.0005,
    ) -> Dict[str, np.ndarray]:
        """
        生成趋势性数据
        
        Parameters
        ----------
        direction : str
            趋势方向 "up" 或 "down"
        daily_return : float
            日均收益率（上涨为正，下跌为负）
            
        Returns
        -------
        data : Dict[str, np.ndarray]
        """
        if direction == "down":
            daily_return = -abs(daily_return)
        else:
            daily_return = abs(daily_return)
            
        t_arr = np.arange(self.T, dtype=np.float64)
        
        # 每只股票的收益率有轻微差异
        returns = daily_return * (1 + self.rng.normal(0, 0.2, size=self.N)[:, np.newaxis])
        
        # 累积收益
        cum_returns = (1.0 + returns) ** t_arr[np.newaxis, :]
        
        # 日噪声
        noise = 1.0 + self.rng.normal(0, self.volatility, size=(self.N, self.T))
        
        close = self.base_price * cum_returns * noise
        close = np.maximum(close, 0.1)
        
        data = self._derive_ohlcv(close)
        data["_meta"] = SyntheticMeta(
            shape=(self.N, self.T),
            dates=self.dates,
            codes=self.codes,
            price_type="hfq",
            data_mode=f"trending_{direction}",
            sine_cycles=0,
            seed=self.seed,
            base_price=self.base_price,
        )
        
        return data
    
    def generate_mean_reverting(
        self,
        half_life: int = 20,
        mean_level: float = 1.0,
    ) -> Dict[str, np.ndarray]:
        """
        生成均值回归数据（Ornstein-Uhlenbeck 过程）
        
        Parameters
        ----------
        half_life : int
            均值回归半衰期（天）
        mean_level : float
            长期均值水平（相对于 base_price）
        """
        theta = np.log(2) / half_life  # 回归速度
        
        log_prices = np.zeros((self.N, self.T), dtype=np.float64)
        log_prices[:, 0] = np.log(self.base_price * mean_level)
        
        for t in range(1, self.T):
            innovation = self.rng.normal(0, self.volatility, size=self.N)
            log_prices[:, t] = (
                log_prices[:, t-1] 
                + theta * (np.log(self.base_price * mean_level) - log_prices[:, t-1])
                + innovation
            )
        
        close = np.exp(log_prices)
        close = np.maximum(close, 0.1)
        
        data = self._derive_ohlcv(close)
        data["_meta"] = SyntheticMeta(
            shape=(self.N, self.T),
            dates=self.dates,
            codes=self.codes,
            price_type="hfq",
            data_mode="mean_reverting",
            sine_cycles=0,
            seed=self.seed,
            base_price=self.base_price,
        )
        
        return data
    
    def generate_random_walk(self) -> Dict[str, np.ndarray]:
        """
        生成随机游走数据（无漂移，纯噪声）
        """
        returns = self.rng.normal(0, self.volatility, size=(self.N, self.T))
        cum_returns = np.cumprod(1.0 + returns, axis=1)
        
        close = self.base_price * cum_returns
        close = np.maximum(close, 0.1)
        
        data = self._derive_ohlcv(close)
        data["_meta"] = SyntheticMeta(
            shape=(self.N, self.T),
            dates=self.dates,
            codes=self.codes,
            price_type="hfq",
            data_mode="random_walk",
            sine_cycles=0,
            seed=self.seed,
            base_price=self.base_price,
        )
        
        return data
    
    def generate_mixed_regime(
        self,
        bull_days: int = 500,
        bear_days: int = 300,
        sideways_days: int = 700,
    ) -> Dict[str, np.ndarray]:
        """
        生成混合市场状态数据：牛市 → 熊市 → 震荡
        
        用于验证 MarketRegimeDetector 和策略的市场适应性
        """
        total = bull_days + bear_days + sideways_days
        if total != self.T:
            # 按比例调整
            ratio = self.T / total
            bull_days = int(bull_days * ratio)
            bear_days = int(bear_days * ratio)
            sideways_days = self.T - bull_days - bear_days
            
        t_arr = np.arange(self.T, dtype=np.float64)
        
        # 分阶段生成
        close = np.zeros((self.N, self.T), dtype=np.float64)
        
        # 牛市阶段：强劲上涨
        bull_returns = self.rng.normal(0.001, self.volatility, size=(self.N, bull_days))
        bull_prices = self.base_price * np.cumprod(1.0 + bull_returns, axis=1)
        close[:, :bull_days] = bull_prices
        
        # 熊市阶段：下跌
        bear_start_price = close[:, bull_days - 1]
        bear_returns = self.rng.normal(-0.0015, self.volatility * 1.2, size=(self.N, bear_days))
        bear_prices = bear_start_price[:, np.newaxis] * np.cumprod(1.0 + bear_returns, axis=1)
        close[:, bull_days:bull_days + bear_days] = bear_prices
        
        # 震荡阶段：均值回归
        sideways_start = bull_days + bear_days
        sideways_start_price = close[:, sideways_start - 1]
        sideways_returns = self.rng.normal(0.0, self.volatility * 0.8, size=(self.N, sideways_days))
        sideways_prices = sideways_start_price[:, np.newaxis] * np.cumprod(1.0 + sideways_returns, axis=1)
        close[:, sideways_start:] = sideways_prices
        
        close = np.maximum(close, 0.1)
        
        data = self._derive_ohlcv(close)
        data["_meta"] = SyntheticMeta(
            shape=(self.N, self.T),
            dates=self.dates,
            codes=self.codes,
            price_type="hfq",
            data_mode="mixed_regime",
            sine_cycles=0,
            seed=self.seed,
            base_price=self.base_price,
        )
        data["_regime_boundaries"] = {
            "bull": (0, bull_days),
            "bear": (bull_days, bull_days + bear_days),
            "sideways": (bull_days + bear_days, self.T),
        }
        
        return data
    
    # =========================================================================
    # 辅助方法
    # =========================================================================
    
    def _derive_ohlcv(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """
        从收盘价派生完整 OHLCV 数据
        
        派生规则（模拟真实市场）：
        - open[t] = close[t-1] × (1 + gap_noise)，gap_noise ~ N(0, 0.005)
        - high[t] = max(open[t], close[t]) × (1 + up_noise)，up_noise ~ U(0, 0.02)
        - low[t]  = min(open[t], close[t]) × (1 - down_noise)，down_noise ~ U(0, 0.02)
        - volume[t] ~ LogNormal(16, 1)（约 1e6 ~ 1e8 股）
        - amount[t] = close[t] × volume[t]
        """
        N, T = close.shape
        
        # 开盘价
        open_ = np.zeros_like(close)
        open_[:, 0] = close[:, 0] * (1 + self.rng.normal(0, 0.002, size=N))
        open_[:, 1:] = close[:, :-1] * (1 + self.rng.normal(0, 0.005, size=(N, T-1)))
        
        # 最高价
        up_noise = self.rng.uniform(0.001, 0.02, size=(N, T))
        high = np.maximum(open_, close) * (1 + up_noise)
        
        # 最低价
        down_noise = self.rng.uniform(0.001, 0.02, size=(N, T))
        low = np.minimum(open_, close) * (1 - down_noise)
        
        # 成交量（股）
        volume = np.exp(self.rng.normal(16, 1, size=(N, T)))
        volume = np.clip(volume, 1e5, 1e9).astype(np.float64)
        
        # 成交额（元）
        amount = close * volume
        
        # valid_mask：全部有效
        valid_mask = np.ones((N, T), dtype=np.bool_)
        
        return {
            "close": close.astype(np.float64),
            "open": open_.astype(np.float64),
            "high": high.astype(np.float64),
            "low": low.astype(np.float64),
            "volume": volume.astype(np.float64),
            "amount": amount.astype(np.float64),
            "valid_mask": valid_mask,
        }
    
    def save_as_npy(
        self,
        data: Dict[str, np.ndarray],
        output_dir: str | Path,
    ) -> Path:
        """
        将合成数据保存为 npy 格式（与 FastRunnerV10 兼容）
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存数据文件
        for field in ["close", "open", "high", "low", "volume", "amount", "valid_mask"]:
            if field in data:
                np.save(output_dir / f"{field}.npy", data[field])
        
        # 保存元信息
        meta = data.get("_meta")
        if meta:
            meta_dict = {
                "shape": list(meta.shape),
                "dates": meta.dates,
                "codes": meta.codes,
                "fields": ["close", "open", "high", "low", "volume", "amount"],
                "price_type": meta.price_type,
                "data_mode": meta.data_mode,
                "sine_cycles": meta.sine_cycles,
                "seed": meta.seed,
                "base_price": meta.base_price,
                "audit_info": {
                    "generator": "SyntheticDataGenerator",
                    "version": "1.0.0",
                    "purpose": "whitebox_audit",
                },
            }
            with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta_dict, f, indent=2, ensure_ascii=False)
        
        return output_dir


# =============================================================================
# 验收测试
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SyntheticDataGenerator 验收测试")
    print("=" * 70)
    
    gen = SyntheticDataGenerator(N=50, T=1500, seed=42)
    
    # 测试 1: 正弦波数据
    data_sin = gen.generate_sinusoidal(cycles=3)
    assert data_sin["close"].shape == (50, 1500), "正弦波 shape 错误"
    assert np.all(data_sin["close"] > 0), "正弦波含负价格"
    assert np.all(data_sin["high"] >= data_sin["close"]), "high < close"
    assert np.all(data_sin["low"] <= data_sin["close"]), "low > close"
    print(f"[PASS] 正弦波数据: shape={data_sin['close'].shape}, cycles=3")
    
    # 测试 2: 趋势数据
    data_up = gen.generate_trending(direction="up")
    assert data_up["close"][:, -1].mean() > data_up["close"][:, 0].mean(), "上涨趋势方向错误"
    print(f"[PASS] 上涨趋势数据: 终值均值 {data_up['close'][:, -1].mean():.2f}")
    
    # 测试 3: 混合市场
    data_mix = gen.generate_mixed_regime()
    assert "_regime_boundaries" in data_mix, "缺少市场边界信息"
    print(f"[PASS] 混合市场数据: 边界={data_mix['_regime_boundaries']}")
    
    # 测试 4: 保存为 npy
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        saved_path = gen.save_as_npy(data_sin, tmpdir)
        assert (saved_path / "close.npy").exists(), "close.npy 未生成"
        assert (saved_path / "meta.json").exists(), "meta.json 未生成"
        print(f"[PASS] npy 保存测试: {saved_path}")
    
    print()
    print("[SUCCESS] SyntheticDataGenerator 全部测试通过")
