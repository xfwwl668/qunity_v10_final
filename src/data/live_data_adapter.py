"""
Q-UNITY V10 — src/data/live_data_adapter.py
============================================
实盘数据适配器（TdxQuant 优先，AKShare 降级）

数据源优先级：
  1. TdxQuant（通达信本地，零网络，推荐）
     - 股票列表：tq.get_stock_list('5')
     - 日线历史：tq.get_market_data(dividend_type='front', count=N)
     - 实时快照：tq.get_market_snapshot(stock_code)
  2. AKShare（网络，作为 TdxQuant 不可用时的降级）

路径配置：config.json -> tdxquant.tq_dir（不写死）

铁律（每次修改前默读）
----------------------
1. stamp_tax = 0.0005（万五，绝不手改）
2. 输出 numpy 数组的 dtype/shape 必须与 npy memmap 完全一致
3. 实盘路径严禁前视：所有数据截止到「当前时刻」
4. extra_factors 中的因子名称必须与 weak_to_strong_alpha 中一致
5. 不修改任何现有文件
"""
from __future__ import annotations

import logging
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# [CONFIG] TdxQuant 路径从 config.json 读取
_tq_initialized = False
_tq_obj = None

def _get_tq():
    """获取 TdxQuant tq 对象（懒初始化，失败返回 None）。"""
    global _tq_initialized, _tq_obj
    if _tq_initialized:
        return _tq_obj
    _tq_initialized = True
    try:
        import sys as _sys, pathlib as _pl
        _scripts = _pl.Path(__file__).resolve().parent.parent.parent / "scripts"
        if str(_scripts) not in _sys.path:
            _sys.path.insert(0, str(_scripts))
        from tqcenter_utils import import_tq as _itq
        mod = _itq()
        if mod:
            mod.tq.initialize(__file__)
            _tq_obj = mod.tq
            logger.info("[LiveDataAdapter] TdxQuant 初始化成功")
    except Exception as e:
        logger.debug(f"[LiveDataAdapter] TdxQuant 不可用: {e}，将降级到 AKShare")
    return _tq_obj


# ─────────────────────────────────────────────────────────────────────────────
# 数据容器
# ─────────────────────────────────────────────────────────────────────────────

class LiveSnapshot:
    """单次实盘快照，包含基础数组与 extra_factors"""

    def __init__(
        self,
        codes: List[str],
        close:  np.ndarray,   # [n_stocks, n_days]
        open_:  np.ndarray,
        high:   np.ndarray,
        low:    np.ndarray,
        volume: np.ndarray,
        valid_mask: np.ndarray,         # [n_stocks] bool，当日可交易
        extra_factors: Dict[str, np.ndarray],  # 真实 intraday 因子
        trade_date: str,
        stock_index: Dict[str, int],    # code → 行索引
    ):
        self.codes        = codes
        self.close        = close
        self.open_        = open_
        self.high         = high
        self.low          = low
        self.volume       = volume
        self.valid_mask   = valid_mask
        self.extra_factors = extra_factors
        self.trade_date   = trade_date
        self.stock_index  = stock_index
        self.n_stocks     = len(codes)


# ─────────────────────────────────────────────────────────────────────────────
# 主适配器
# ─────────────────────────────────────────────────────────────────────────────

class LiveDataAdapter:
    """
    实盘数据适配器。

    用法：
        adapter = LiveDataAdapter(lookback_days=30)
        snapshot = adapter.get_snapshot()
        # snapshot.close, .open_, .high, .low, .volume → 传入策略函数
        # snapshot.extra_factors                       → kwargs 注入策略
    """

    def __init__(
        self,
        lookback_days: int = 30,
        cache_ttl_seconds: int = 60,
        max_workers: int = 8,
        min_listing_days: int = 60,
    ):
        self.lookback_days   = lookback_days
        self.cache_ttl       = cache_ttl_seconds
        self.max_workers     = max_workers
        self.min_listing_days = min_listing_days

        self._snapshot_cache: Optional[LiveSnapshot] = None
        self._cache_ts: float = 0.0

    # ── 公开接口 ─────────────────────────────────────────────────────────────

    def get_snapshot(self, force_refresh: bool = False) -> Optional[LiveSnapshot]:
        """
        获取当前实盘快照（带 TTL 缓存）。

        Returns
        -------
        LiveSnapshot | None
            None 表示数据获取失败。
        """
        now = time.time()
        if (
            not force_refresh
            and self._snapshot_cache is not None
            and now - self._cache_ts < self.cache_ttl
        ):
            logger.debug("使用缓存快照")
            return self._snapshot_cache

        try:
            snapshot = self._build_snapshot()
            self._snapshot_cache = snapshot
            self._cache_ts = now
            return snapshot
        except Exception as exc:
            logger.error(f"构建实盘快照失败: {exc}", exc_info=True)
            return self._snapshot_cache  # 返回上一次的缓存，避免断流

    # ── 内部构建流程 ──────────────────────────────────────────────────────────

    def _build_snapshot(self) -> LiveSnapshot:
        trade_date = datetime.now().strftime("%Y%m%d")
        logger.info(f"构建实盘快照 trade_date={trade_date}")

        # [TDXQUANT] 优先使用 TdxQuant，降级到 AKShare
        tq = _get_tq()
        if tq is not None:
            return self._build_snapshot_tdxquant(tq, trade_date)
        else:
            import akshare as ak
            logger.info("TdxQuant 不可用，使用 AKShare")
            return self._build_snapshot_akshare(ak, trade_date)

    def _build_snapshot_tdxquant(self, tq, trade_date: str) -> LiveSnapshot:
        """TdxQuant 实盘快照构建（优先路径）"""
        logger.info("使用 TdxQuant 构建实盘快照")

        # 股票列表
        result = tq.get_stock_list('5')
        codes = []
        for item in (result or []):
            c = item.get("Code","") if isinstance(item, dict) else str(item)
            if c and "." in c:
                codes.append(c)
        if not codes:
            raise RuntimeError("TdxQuant get_stock_list 返回空")
        n = len(codes)

        # 日线历史（前复权，lookback+1 天）
        lookback = max(self.lookback_days, 21)
        full_codes = codes
        data = tq.get_market_data(
            field_list=["Open","High","Low","Close","Volume","Amount"],
            stock_list=full_codes,
            count=lookback + 2,
            dividend_type="front",
            period="1d",
            fill_data=True,
        )
        if not data or "Close" not in data:
            raise RuntimeError("TdxQuant get_market_data 返回空")

        close_df  = data["Close"].reindex(columns=full_codes, fill_value=np.nan)
        open_df   = data["Open"].reindex(columns=full_codes,  fill_value=np.nan)
        high_df   = data["High"].reindex(columns=full_codes,  fill_value=np.nan)
        low_df    = data["Low"].reindex(columns=full_codes,   fill_value=np.nan)
        vol_df    = data["Volume"].reindex(columns=full_codes, fill_value=0.0)

        T = close_df.shape[0]
        close  = close_df.values.T.astype(np.float32)       # (n, T)
        open_  = open_df.values.T.astype(np.float32)
        high   = high_df.values.T.astype(np.float32)
        low    = low_df.values.T.astype(np.float32)
        volume = vol_df.values.T.astype(np.float32)  # 手

        # valid_mask（今日有数据且价格>0）
        valid_mask = (close[:, -1] > 0) & np.isfinite(close[:, -1])

        # 实时快照（PE/PB/市值用于 extra_factors）
        extra_factors = self._fetch_extra_factors_tdxquant(tq, full_codes)

        stock_index = {c: i for i, c in enumerate(full_codes)}
        return LiveSnapshot(
            codes=full_codes, close=close, open_=open_,
            high=high, low=low, volume=volume,
            valid_mask=valid_mask, extra_factors=extra_factors,
            trade_date=trade_date, stock_index=stock_index,
        )

    def _fetch_extra_factors_tdxquant(self, tq, codes: list) -> dict:
        """从 TdxQuant 获取实盘 extra_factors（炸板/竞价/资金）"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        n = len(codes)
        pe_ttm   = np.full(n, np.nan, dtype=np.float32)
        pb_mrq   = np.full(n, np.nan, dtype=np.float32)
        mkt_cap  = np.full(n, np.nan, dtype=np.float32)
        auction  = np.full(n, 50.0,  dtype=np.float32)  # 竞价评分

        def _one(i_code):
            i, code = i_code
            try:
                more = tq.get_more_info(stock_code=code, field_list=[])
                if more:
                    pe = more.get("StaticPE_TTM")
                    pb = more.get("PB_MRQ")
                    mc = more.get("Ltsz")
                    if pe: pe_ttm[i]  = float(str(pe).replace(",",""))
                    if pb: pb_mrq[i]  = float(str(pb).replace(",",""))
                    if mc: mkt_cap[i] = float(str(mc).replace(",",""))
                    # 竞价评分近似：涨幅归一化到0-100
                    zaf = more.get("ZAF", 0) or 0
                    auction[i] = float(np.clip(50 + float(str(zaf)) * 5, 10, 100))
            except Exception:
                pass

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            list(ex.map(_one, enumerate(codes)))

        return {
            "pe_ttm":         pe_ttm,
            "pb_mrq":         pb_mrq,
            "mkt_cap_circ":   mkt_cap,
            "auction_score":  auction,
            "zhaban_codes":   set(),   # 炸板池暂用 AKShare 回退（如可用）
            "money_flow_score": None,  # 主力资金用日线近似
        }

    def _build_snapshot_akshare(self, ak, trade_date: str) -> LiveSnapshot:
        """AKShare 实盘快照构建（降级路径）"""
        logger.info("使用 AKShare 构建实盘快照")

        # 1. 获取全市场股票列表与当日行情
        df_spot = self._fetch_spot(ak)
        codes   = df_spot["代码"].tolist()
        n       = len(codes)
        stock_index = {c: i for i, c in enumerate(codes)}

        # 2. 获取历史日线（用于策略计算窗口）
        close, open_, high, low, volume = self._fetch_history(codes, trade_date)

        # 3. valid_mask：退市/停牌/次新股过滤
        valid_mask = self._build_valid_mask(df_spot, codes)

        # 4. 获取 extra_factors（真实竞价 + 资金流向 + 炸板池）
        extra_factors = self._fetch_extra_factors(ak, codes, stock_index, trade_date, df_spot)

        return LiveSnapshot(
            codes=codes,
            close=close,
            open_=open_,
            high=high,
            low=low,
            volume=volume,
            valid_mask=valid_mask,
            extra_factors=extra_factors,
            trade_date=trade_date,
            stock_index=stock_index,
        )

    # ── 步骤 1: 实时行情 ─────────────────────────────────────────────────────

    def _fetch_spot(self, ak) -> pd.DataFrame:
        """获取全市场实时行情（AkShare）"""
        t0 = time.time()
        try:
            df = ak.stock_zh_a_spot_em()
            logger.info(f"实时行情 {len(df)} 只，耗时 {time.time()-t0:.1f}s")
            return df
        except Exception as exc:
            logger.error(f"获取实时行情失败: {exc}")
            raise

    # ── 步骤 2: 历史日线 ─────────────────────────────────────────────────────

    def _fetch_history(
        self, codes: List[str], trade_date: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        批量获取历史日线，返回 [n_stocks, n_days] numpy 数组。
        优先从本地 npy 读取（如已下载），否则从 AkShare 拉取。
        """
        import akshare as ak
        from concurrent.futures import ThreadPoolExecutor, as_completed

        end_dt   = datetime.strptime(trade_date, "%Y%m%d")
        start_dt = end_dt - timedelta(days=self.lookback_days * 2)
        start_str = start_dt.strftime("%Y%m%d")
        n_days   = self.lookback_days

        n        = len(codes)
        close    = np.full((n, n_days), np.nan, dtype=np.float32)
        open_    = np.full((n, n_days), np.nan, dtype=np.float32)
        high_    = np.full((n, n_days), np.nan, dtype=np.float32)
        low_     = np.full((n, n_days), np.nan, dtype=np.float32)
        volume   = np.full((n, n_days), np.nan, dtype=np.float32)

        def _fetch_one(idx_code):
            idx, code = idx_code
            try:
                df = ak.stock_zh_a_hist(
                    symbol=code,
                    period="daily",
                    start_date=start_str,
                    end_date=trade_date,
                    adjust="qfq",
                )
                if df is None or df.empty:
                    return
                df = df.rename(columns={
                    "开盘": "open", "最高": "high", "最低": "low",
                    "收盘": "close", "成交量": "volume",
                })
                # 取最近 n_days 行
                n_avail = min(len(df), n_days)
                close [idx, -n_avail:] = df["close" ].values[-n_avail:].astype(np.float32)
                open_ [idx, -n_avail:] = df["open"  ].values[-n_avail:].astype(np.float32)
                high_ [idx, -n_avail:] = df["high"  ].values[-n_avail:].astype(np.float32)
                low_  [idx, -n_avail:] = df["low"   ].values[-n_avail:].astype(np.float32)
                volume[idx, -n_avail:] = df["volume"].values[-n_avail:].astype(np.float32)
            except Exception:
                pass  # 个股失败不中断

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = [ex.submit(_fetch_one, (i, c)) for i, c in enumerate(codes)]
            for f in as_completed(futs):
                pass  # 等待全部完成

        logger.info(
            f"历史日线下载完成: {n} 只 × {n_days} 天，"
            f"耗时 {time.time()-t0:.1f}s"
        )
        # 用 0 填充 nan（与 npy memmap 行为一致）
        np.nan_to_num(close,  copy=False)
        np.nan_to_num(open_,  copy=False)
        np.nan_to_num(high_,  copy=False)
        np.nan_to_num(low_,   copy=False)
        np.nan_to_num(volume, copy=False)
        return close, open_, high_, low_, volume

    # ── 步骤 3: valid_mask ───────────────────────────────────────────────────

    def _build_valid_mask(self, df_spot: pd.DataFrame, codes: List[str]) -> np.ndarray:
        """
        过滤退市/停牌/次新股（上市不足 min_listing_days 天）。
        返回 [n_stocks] bool。
        """
        valid = np.ones(len(codes), dtype=bool)
        # 停牌：成交量为 0
        if "成交量" in df_spot.columns:
            vol_series = df_spot.set_index("代码")["成交量"].reindex(codes).fillna(0)
            valid &= (vol_series.values > 0)
        # ST/退市：名称含 ST、退
        if "名称" in df_spot.columns:
            names = df_spot.set_index("代码")["名称"].reindex(codes).fillna("")
            is_st = names.str.contains("ST|退", na=False).values
            valid &= ~is_st
        return valid

    # ── 步骤 4: extra_factors（实盘专属）────────────────────────────────────

    def _fetch_extra_factors(
        self,
        ak,
        codes: List[str],
        stock_index: Dict[str, int],
        trade_date: str,
        df_spot: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        """
        获取实盘专属因子（竞价强度、主力资金、炸板池）。
        当 API 调用失败时，对应因子回退到 None（策略层使用代理因子）。

        Returns
        -------
        dict，键名与 weak_to_strong_alpha 的 extra_factors 参数对应：
            zhaban_codes      : set[str]  昨日炸板股集合
            auction_score     : ndarray[n] 竞价评分（0~100），None=使用代理
            money_flow_score  : ndarray[n] 主力资金评分（0~100），None=使用代理
        """
        n = len(codes)
        extra: Dict[str, object] = {
            "zhaban_codes":     set(),
            "auction_score":    None,
            "money_flow_score": None,
        }

        # ── 4a. 炸板池（前一交易日）────────────────────────────────────────
        prev_date = self._prev_trading_day(trade_date)
        try:
            df_zb = ak.stock_zt_pool_zbgc_em(date=prev_date)
            if df_zb is not None and not df_zb.empty:
                extra["zhaban_codes"] = set(df_zb["代码"].astype(str).str.zfill(6))
                logger.info(f"炸板池 {prev_date}: {len(extra['zhaban_codes'])} 只")
        except Exception as exc:
            logger.warning(f"炸板池获取失败（将用日线近似）: {exc}")

        # ── 4b. 竞价强度（用当日开盘/昨收计算，数据来自 df_spot）──────────
        try:
            spot_idx = df_spot.set_index("代码")
            prev_close_col = "昨收" if "昨收" in df_spot.columns else None
            open_col       = "今开" if "今开" in df_spot.columns else None

            if prev_close_col and open_col:
                prev_close = spot_idx[prev_close_col].reindex(codes).fillna(0).values.astype(np.float64)
                open_price = spot_idx[open_col].reindex(codes).fillna(0).values.astype(np.float64)
                eps = 1e-8
                open_change = np.where(
                    prev_close > eps,
                    (open_price / (prev_close + eps) - 1) * 100,
                    0.0,
                )
                # 评分：最佳区间 1%~4%（来自 hunter_v6 原始逻辑）
                auction_score = np.clip(
                    np.where((open_change >= 1) & (open_change <= 4),
                             100 - np.abs(open_change - 2.5) * 10,
                    np.where((open_change >= 0) & (open_change < 1),
                             85 - open_change * 10,
                    np.where((open_change > 4) & (open_change <= 6),
                             70 - (open_change - 4) * 10,
                    np.where((open_change >= -2) & (open_change < 0),
                             60 + open_change * 15,
                             np.maximum(20.0, 40 - np.abs(open_change - 5) * 5))))),
                    20, 100,
                ).astype(np.float32)
                extra["auction_score"] = auction_score
                logger.debug("竞价评分计算完成（来自实时 spot 数据）")
        except Exception as exc:
            logger.warning(f"竞价评分计算失败（将用日线近似）: {exc}")

        # ── 4c. 主力资金（AkShare 个股资金流向）─────────────────────────────
        # 只对炸板候选股请求，避免全市场 4000+ 次 API 调用
        try:
            zhaban_list = [c for c in codes if c in extra["zhaban_codes"]]
            if zhaban_list:
                money_score = np.full(n, 50.0, dtype=np.float32)  # 默认中性
                money_score = self._fetch_money_flow_scores(
                    ak, zhaban_list, stock_index, money_score, trade_date
                )
                extra["money_flow_score"] = money_score
                logger.debug(f"主力资金评分完成: {len(zhaban_list)} 只炸板候选")
        except Exception as exc:
            logger.warning(f"主力资金评分失败（将用量价近似）: {exc}")

        return extra

    def _fetch_money_flow_scores(
        self, ak, codes: List[str],
        stock_index: Dict[str, int],
        money_score: np.ndarray,
        trade_date: str,
    ) -> np.ndarray:
        """对指定股票列表获取主力资金评分"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _one(code):
            try:
                df = ak.stock_individual_fund_flow(stock=code, market="沪深A股")
                if df is None or df.empty:
                    return code, 50.0
                date_fmt = pd.to_datetime(trade_date).strftime("%Y-%m-%d")
                df["日期"] = pd.to_datetime(df["日期"]).dt.strftime("%Y-%m-%d")
                row = df[df["日期"] == date_fmt]
                if row.empty:
                    return code, 50.0
                main_ratio = float(row.iloc[0].get("主力净流入-净占比", 0)) / 100
                # 映射为 0-100 评分（来自 hunter_v6 MoneyFlowAnalyzer）
                if   main_ratio > 0.50: s = 100
                elif main_ratio > 0.30: s = 85
                elif main_ratio > 0.15: s = 70
                elif main_ratio > 0.00: s = 55
                elif main_ratio > -0.10: s = 40
                elif main_ratio > -0.20: s = 25
                else:                   s = 10
                return code, float(s)
            except Exception:
                return code, 50.0

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            for fut in as_completed([ex.submit(_one, c) for c in codes]):
                code, score = fut.result()
                if code in stock_index:
                    money_score[stock_index[code]] = score

        return money_score

    # ── 工具方法 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _prev_trading_day(date_str: str) -> str:
        """简单往前推，跳过周末（不含节假日，精度足够用于炸板池查询）"""
        dt = datetime.strptime(date_str, "%Y%m%d") - timedelta(days=1)
        while dt.weekday() >= 5:
            dt -= timedelta(days=1)
        return dt.strftime("%Y%m%d")
