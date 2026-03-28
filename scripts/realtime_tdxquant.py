"""
scripts/realtime_tdxquant.py
==============================
Q-UNITY V10 实时行情模块 -- TdxQuant 通达信本地版

【功能】
  1. 实时行情快照（get_market_snapshot）
     每只股票: 价格/涨跌/委买委卖5档/成交量/市值/PE/PB
  2. 行情推送订阅（subscribe_hq）
     有更新自动触发回调，延迟约100ms级别
  3. 超短线信号预警（send_warn）
     计算结果回推到通达信客户端显示预警
  4. 选股结果推送（send_user_block）
     把选股结果写入通达信自定义板块，客户端实时显示

【对接 ultra_short_signal.py】
  实时获取最新一根日线 -> 计算因子 -> 触发信号 -> send_warn 推送

【用法】
  # 单次快照（盘中调用）
  python scripts/realtime_tdxquant.py --snapshot --codes 600519 000001

  # 持续监控模式（订阅推送）
  python scripts/realtime_tdxquant.py --monitor --codes 600519 000001 300750

  # 全市场扫描（收盘后选股）
  python scripts/realtime_tdxquant.py --scan

  # 在代码里 import 使用
  from scripts.realtime_tdxquant import TdxRealtimeEngine
"""

import sys
import json
import time
import argparse
import threading
from datetime import date, datetime
from pathlib import Path
from typing import List, Dict, Optional, Callable

import numpy as np
import pandas as pd

# ── tqcenter 路径 ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# [CONFIG] TdxQuant 路径从 config.json 统一读取
def _add_tq_path():
    try:
        _proj = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(_proj / "scripts"))
        from tqcenter_utils import find_tqcenter as _ftu
        p = _ftu()
        if p:
            sys.path.insert(0, p)
            return True
    except Exception:
        pass
    return False

_add_tq_path()


# ─────────────────────────────────────────────────────────────────────────────
# 实时行情引擎
# ─────────────────────────────────────────────────────────────────────────────

class TdxRealtimeEngine:
    """
    TdxQuant 实时行情引擎。

    用法：
      engine = TdxRealtimeEngine()
      engine.initialize()

      # 单次快照
      snap = engine.snapshot('600519.SH')

      # 订阅推送
      engine.subscribe(['600519.SH','000001.SZ'], callback=my_func)

      # 发送预警到通达信
      engine.send_alert('600519.SH', price=1450.0, reason='RSRS突破信号')

      engine.close()
    """

    # get_more_info 字段映射（实时估值）
    _MORE_INFO_MAP = {
        "PE_TTM":   "StaticPE_TTM",
        "PB_MRQ":   "PB_MRQ",
        "DynaPE":   "DynaPE",
        "DYRatio":  "DYRatio",     # 股息率
        "Zsz":      "Zsz",         # 总市值（亿元）
        "Ltsz":     "Ltsz",        # 流通市值（亿元）
        "ZAF":      "ZAF",         # 今日涨幅%
        "ZAFPre5":  "ZAFPre5",     # 5日涨幅
        "ZAFPre20": "ZAFPre20",    # 20日涨幅
        "MA5":      "MA5Value",    # 5日均价
        "HisHigh":  "HisHigh",     # 52周最高
        "HisLow":   "HisLow",      # 52周最低
        "ZTDate":   "ZTDate_Recent",  # 最近涨停日
        "DTDate":   "DTDate_Recent",  # 最近跌停日
        "ConZAF":   "ConZAFDateNum",  # 连续上涨天数（负=下跌）
        "ZTPrice":  "ZTPrice",     # 涨停价
        "DTPrice":  "DTPrice",     # 跌停价
    }

    # get_market_snapshot 字段映射（盘口实时）
    _SNAPSHOT_MAP = {
        "price":      "Now",
        "open":       "Open",
        "high":       "Max",
        "low":        "Min",
        "prev_close": "LastClose",
        "volume":     "Volume",    # 手
        "amount":     "Amount",    # 万元
        "buy1_p":     "Buyp",
        "buy1_v":     "Buyv",
        "sell1_p":    "Sellp",
        "sell1_v":    "Sellv",
        "change_pct": "ZAFPre3",   # 近似涨跌幅（从more_info）
    }

    def __init__(self, tq_dir: str = None):
        self._tq = None
        self._initialized = False
        self._subscriptions = {}
        self._lock = threading.Lock()
        if tq_dir:
            sys.path.insert(0, tq_dir)

    def initialize(self) -> bool:
        try:
            from tqcenter import tq
            tq.initialize(__file__)
            self._tq = tq
            self._initialized = True
            print("✓ TdxRealtimeEngine 初始化成功")
            return True
        except Exception as e:
            print(f"✗ 初始化失败: {e}")
            print("  请确认通达信客户端已打开")
            return False

    def close(self):
        if self._tq and self._initialized:
            try:
                self._tq.close()
                self._initialized = False
            except Exception:
                pass

    # ── 格式转换 ──────────────────────────────────────────────────────────────

    @staticmethod
    def to_tdx(code: str) -> str:
        """sh.600519 -> 600519.SH"""
        if "." not in code:
            return code
        p = code.split(".", 1)
        if p[0].lower() in ("sh","sz","bj"):
            return f"{p[1]}.{p[0].upper()}"
        return code

    @staticmethod
    def to_full(code: str) -> str:
        """600519.SH -> sh.600519"""
        if "." not in code:
            return code
        num, mkt = code.split(".", 1)
        return f"{mkt.lower()}.{num}"

    # ── 单只快照 ──────────────────────────────────────────────────────────────

    def snapshot(self, code: str) -> Dict:
        """
        获取单只股票完整实时快照。
        合并 get_market_snapshot（盘口）+ get_more_info（估值/市值）。

        返回标准化字典，含：
          price, open, high, low, prev_close, volume(手), amount(万元)
          pe_ttm, pb_mrq, mkt_cap_total(亿元), mkt_cap_circ(亿元)
          change_pct(%), zt_price, dt_price, con_zaf(连涨天数)
        """
        if not self._initialized:
            return {}

        tdx_code = self.to_tdx(code)
        result   = {"code": code, "tdx_code": tdx_code}

        # 盘口快照
        try:
            snap = self._tq.get_market_snapshot(
                stock_code=tdx_code, field_list=[])
            if snap:
                result["price"]      = snap.get("Now", 0)
                result["open"]       = snap.get("Open", 0)
                result["high"]       = snap.get("Max", 0)
                result["low"]        = snap.get("Min", 0)
                result["prev_close"] = snap.get("LastClose", 0)
                result["volume"]     = snap.get("Volume", 0)   # 手
                result["amount"]     = snap.get("Amount", 0)   # 万元
                result["amount_yuan"]= snap.get("Amount", 0) * 10000  # 元
                # 涨跌幅
                prev = snap.get("LastClose", 0)
                now  = snap.get("Now", 0)
                if prev and prev > 0:
                    result["change_pct"] = (now - prev) / prev * 100
                # 委买委卖
                result["buy1_p"]  = snap.get("Buyp", 0)
                result["buy1_v"]  = snap.get("Buyv", 0)
                result["sell1_p"] = snap.get("Sellp", 0)
                result["sell1_v"] = snap.get("Sellv", 0)
        except Exception as e:
            result["snap_error"] = str(e)

        # 估值/市值（get_more_info）
        try:
            more = self._tq.get_more_info(
                stock_code=tdx_code, field_list=[])
            if more:
                result["pe_ttm"]       = _safe_float(more.get("StaticPE_TTM"))
                result["pb_mrq"]       = _safe_float(more.get("PB_MRQ"))
                result["dyna_pe"]      = _safe_float(more.get("DynaPE"))
                result["dy_ratio"]     = _safe_float(more.get("DYRatio"))
                result["mkt_cap_total"]= _safe_float(more.get("Zsz"))   # 亿元
                result["mkt_cap_circ"] = _safe_float(more.get("Ltsz"))  # 亿元
                result["zt_price"]     = _safe_float(more.get("ZTPrice"))
                result["dt_price"]     = _safe_float(more.get("DTPrice"))
                result["con_zaf"]      = more.get("ConZAFDateNum", 0)   # 连涨天数
                result["zaf_1d"]       = _safe_float(more.get("ZAF"))
                result["zaf_5d"]       = _safe_float(more.get("ZAFPre5"))
                result["zaf_20d"]      = _safe_float(more.get("ZAFPre20"))
                result["ma5"]          = _safe_float(more.get("MA5Value"))
                result["his_high"]     = _safe_float(more.get("HisHigh"))
                result["his_low"]      = _safe_float(more.get("HisLow"))
        except Exception as e:
            result["more_error"] = str(e)

        return result

    def batch_snapshot(self, codes: List[str]) -> List[Dict]:
        """批量快照，返回列表"""
        return [self.snapshot(c) for c in codes]

    def snapshot_df(self, codes: List[str]) -> pd.DataFrame:
        """批量快照，返回 DataFrame"""
        rows = self.batch_snapshot(codes)
        return pd.DataFrame(rows)

    # ── 最新日线（接入策略计算）──────────────────────────────────────────────

    def get_latest_bar(self, code: str, count: int = 60) -> pd.DataFrame:
        """
        获取最近 N 根日线（前复权）。
        用于超短线因子计算（RSRS / 量价 / 动量等）。

        返回 DataFrame，列: date/open/high/low/close/volume/amount
        volume 单位: 手  amount 单位: 万元
        """
        if not self._initialized:
            return pd.DataFrame()
        tdx_code = self.to_tdx(code)
        try:
            data = self._tq.get_market_data(
                field_list    = ["Open","High","Low","Close","Volume","Amount"],
                stock_list    = [tdx_code],
                count         = count,
                dividend_type = "front",
                period        = "1d",
                fill_data     = True,
            )
            if not data or "Close" not in data:
                return pd.DataFrame()

            df = pd.DataFrame({
                "open":   data["Open"][tdx_code]   if "Open"   in data else np.nan,
                "high":   data["High"][tdx_code]   if "High"   in data else np.nan,
                "low":    data["Low"][tdx_code]    if "Low"    in data else np.nan,
                "close":  data["Close"][tdx_code],
                "volume": data["Volume"][tdx_code] if "Volume" in data else np.nan,
                "amount": data["Amount"][tdx_code] if "Amount" in data else np.nan,
            })
            df.index.name = "date"
            df = df.reset_index()
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            df["code"] = self.to_full(code)
            return df.dropna(subset=["close"])
        except Exception as e:
            print(f"  get_latest_bar({code}): {e}")
            return pd.DataFrame()

    # ── 行情推送订阅 ─────────────────────────────────────────────────────────

    def subscribe(
        self,
        codes: List[str],
        callback: Callable,
        on_update: Callable = None,
    ) -> bool:
        """
        订阅行情推送（最多100只）。
        有更新时自动调用 callback(code, snapshot_dict)。

        on_update: 可选，接收原始 tqcenter 回调数据格式
        """
        if not self._initialized:
            return False
        tdx_codes = [self.to_tdx(c) for c in codes]

        def _raw_callback(data_str: str):
            try:
                info = json.loads(data_str)
                tdx_code = info.get("Code", "")
                if not tdx_code:
                    return
                full_code = self.to_full(tdx_code)
                snap = self.snapshot(full_code)
                callback(full_code, snap)
                if on_update:
                    on_update(data_str)
            except Exception as e:
                print(f"  [subscribe] 回调异常: {e}")

        try:
            self._tq.subscribe_hq(
                stock_list=tdx_codes,
                callback=_raw_callback,
            )
            print(f"✓ 已订阅 {len(tdx_codes)} 只股票的行情推送")
            return True
        except Exception as e:
            print(f"✗ 订阅失败: {e}")
            return False

    def unsubscribe(self, codes: List[str]) -> None:
        tdx_codes = [self.to_tdx(c) for c in codes]
        try:
            self._tq.unsubscribe_hq(stock_list=tdx_codes)
        except Exception:
            pass

    # ── 预警推送 ─────────────────────────────────────────────────────────────

    def send_alert(
        self,
        code: str,
        price: float,
        reason: str,
        bs_flag: int = 0,   # 0买 1卖 2未知
        close: float = None,
        volume: float = None,
    ) -> bool:
        """
        发送预警信号到通达信客户端。

        bs_flag: 0=买入信号 1=卖出信号 2=观察信号
        """
        if not self._initialized:
            return False
        tdx_code = self.to_tdx(code)
        now_str  = datetime.now().strftime("%Y%m%d%H%M%S")
        try:
            self._tq.send_warn(
                stock_list    = [tdx_code],
                time_list     = [now_str],
                price_list    = [str(price)],
                close_list    = [str(close or price)],
                volum_list    = [str(int(volume or 0))],
                bs_flag_list  = [str(bs_flag)],
                warn_type_list= ["0"],
                reason_list   = [reason[:25]],   # 最多25个汉字
                count         = 1,
            )
            return True
        except Exception as e:
            print(f"  send_alert({code}): {e}")
            return False

    def send_alerts_batch(self, signals: List[Dict]) -> bool:
        """
        批量发送预警。
        signals: [{"code":"600519.SH","price":1450,"reason":"RSRS信号","bs":0}, ...]
        """
        if not signals or not self._initialized:
            return False
        try:
            self._tq.send_warn(
                stock_list    = [self.to_tdx(s["code"]) for s in signals],
                time_list     = [datetime.now().strftime("%Y%m%d%H%M%S")] * len(signals),
                price_list    = [str(s.get("price",0)) for s in signals],
                close_list    = [str(s.get("close", s.get("price",0))) for s in signals],
                volum_list    = [str(int(s.get("volume",0))) for s in signals],
                bs_flag_list  = [str(s.get("bs",0)) for s in signals],
                warn_type_list= ["0"],
                reason_list   = [s.get("reason","信号")[:25] for s in signals],
                count         = len(signals),
            )
            print(f"✓ 已发送 {len(signals)} 条预警到通达信客户端")
            return True
        except Exception as e:
            print(f"✗ 批量预警失败: {e}")
            return False

    # ── 选股结果推送 ─────────────────────────────────────────────────────────

    def push_to_block(
        self,
        codes: List[str],
        block_code: str = "QUNITY",
        show: bool = True,
    ) -> bool:
        """
        把选股结果推送到通达信自定义板块（客户端实时显示）。

        block_code: 通达信板块简称（需在客户端预先创建）
        """
        if not self._initialized:
            return False
        tdx_codes = [self.to_tdx(c) for c in codes]
        try:
            self._tq.send_user_block(
                block_code=block_code,
                stocks=tdx_codes,
                show=show,
            )
            print(f"✓ {len(tdx_codes)} 只股票 -> 板块[{block_code}]")
            return True
        except Exception as e:
            print(f"✗ 推送板块失败: {e}")
            return False

    # ── 全市场扫描 ───────────────────────────────────────────────────────────

    def scan_all(
        self,
        filter_fn: Callable[[Dict], bool],
        codes: List[str] = None,
        max_workers: int = 1,
    ) -> List[Dict]:
        """
        全市场扫描，对每只股票调用 snapshot() 后用 filter_fn 过滤。
        filter_fn(snap_dict) -> bool: 返回 True 表示选中该股票。

        由于 get_more_info 约 5ms/只，5000只约 25 秒，单线程即可。
        """
        if codes is None:
            # 从 parquet 目录读取股票列表
            pq_dir = Path(PROJECT_ROOT / "data" / "daily_parquet_qfq")
            if pq_dir.exists():
                codes = [p.stem for p in pq_dir.glob("*.parquet")]
            else:
                print("✗ 未提供股票列表")
                return []

        selected = []
        total    = len(codes)
        t0       = time.time()

        print(f"  全市场扫描: {total} 只...")
        for i, code in enumerate(codes):
            snap = self.snapshot(code)
            if snap and filter_fn(snap):
                selected.append(snap)
            if (i+1) % 500 == 0:
                elapsed = time.time() - t0
                print(f"\r  [{i+1}/{total}] 已选 {len(selected)} 只  "
                      f"{(i+1)/elapsed:.0f}只/s  "
                      f"ETA={((total-i-1)/((i+1)/elapsed)):.0f}s  ",
                      end="", flush=True)

        elapsed = time.time() - t0
        print(f"\n  扫描完成: {total} 只 -> 选出 {len(selected)} 只  "
              f"耗时 {elapsed:.1f}s")
        return selected


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(str(val).replace(",",""))
    except (ValueError, TypeError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CLI 模式
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Q-UNITY V10 实时行情 -- TdxQuant 通达信本地版")
    ap.add_argument("--snapshot", action="store_true",
                    help="单次快照（盘中使用）")
    ap.add_argument("--monitor",  action="store_true",
                    help="持续监控模式（订阅推送）")
    ap.add_argument("--scan",     action="store_true",
                    help="全市场扫描（收盘后选股）")
    ap.add_argument("--codes",    nargs="+", default=["600519","000001","300750"],
                    help="股票代码（6位）")
    ap.add_argument("--block",    default="QUNITY",
                    help="推送到通达信板块的简称（默认QUNITY）")
    args = ap.parse_args()

    engine = TdxRealtimeEngine()
    if not engine.initialize():
        sys.exit(1)

    # 转换代码格式
    def _to_tdx(c):
        c = c.strip().zfill(6)
        if c.startswith(("6","9")):
            return f"{c}.SH"
        return f"{c}.SZ"

    codes_tdx = [_to_tdx(c) for c in args.codes]

    try:
        if args.snapshot:
            print(f"\n快照模式: {codes_tdx}")
            df = engine.snapshot_df(codes_tdx)
            cols = ["code","price","change_pct","volume","amount",
                    "pe_ttm","pb_mrq","mkt_cap_circ","zt_price","con_zaf"]
            show_cols = [c for c in cols if c in df.columns]
            print(df[show_cols].to_string())

        elif args.monitor:
            print(f"\n监控模式: {codes_tdx}  (Ctrl+C 退出)")

            def on_update(code, snap):
                price  = snap.get("price", 0)
                change = snap.get("change_pct", 0)
                vol    = snap.get("volume", 0)
                ts     = datetime.now().strftime("%H:%M:%S")
                print(f"  [{ts}] {code}  {price:.2f}  "
                      f"{'↑' if change >= 0 else '↓'}{abs(change):.2f}%  "
                      f"量{vol:.0f}手")

            engine.subscribe(codes_tdx, callback=on_update)
            print("  订阅成功，等待行情推送...")
            while True:
                time.sleep(1)

        elif args.scan:
            print(f"\n全市场扫描模式...")

            # 示例过滤条件：涨幅 > 5% 且连涨 >= 2天
            def filter_fn(snap):
                zaf = snap.get("zaf_1d", 0) or 0
                con = snap.get("con_zaf", 0) or 0
                pe  = snap.get("pe_ttm") or 999
                return zaf > 5.0 and con >= 2 and 0 < pe < 100

            selected = engine.scan_all(filter_fn)

            if selected:
                print(f"\n选出 {len(selected)} 只：")
                for s in selected[:20]:
                    print(f"  {s['code']}  涨幅={s.get('zaf_1d',0):.1f}%  "
                          f"连涨={s.get('con_zaf',0)}天  PE={s.get('pe_ttm','N/A')}")

                # 推送到通达信板块
                engine.push_to_block(
                    [s["code"] for s in selected],
                    block_code=args.block,
                )
            else:
                print("  无符合条件的股票")

        else:
            # 默认：显示快照
            print(f"\n快照: {codes_tdx}")
            for code in codes_tdx:
                snap = engine.snapshot(code)
                print(f"\n  {code}:")
                for k, v in snap.items():
                    if v is not None and k not in ("code","tdx_code"):
                        print(f"    {k}: {v}")

    except KeyboardInterrupt:
        print("\n  已中断")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
