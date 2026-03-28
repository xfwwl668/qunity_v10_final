"""
scripts/live_trade_tools.py
==============================
实盘辅助工具集合（统一入口）

功能:
  1. record-signal  : 记录当日信号价格（下单前调用）
  2. record-exec    : 记录实际成交（下单后调用）
  3. analyze        : 分析累计滑点，评估是否需要调整 slippage_rate
  4. position-size  : 计算每只股票的目标股数

用法:
  python scripts/live_trade_tools.py analyze
  python scripts/live_trade_tools.py position-size --capital 200000
  python scripts/live_trade_tools.py record-signal --date 2024-03-01 --strategy titan_alpha_v1
  python scripts/live_trade_tools.py record-exec   --code sh.600519 --price 1680.00 --date 2024-03-04
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TRACK_FILE = ROOT / "data" / "slippage_track.csv"


# ─────────────────────────────────────────────────────────────────────────────
# 滑点追踪
# ─────────────────────────────────────────────────────────────────────────────

def record_signal(signal_date: str, strategy: str, signals: dict):
    """
    记录信号及当日收盘价（作为基准价格）。
    在生成信号后、下单前调用。
    """
    import pandas as pd

    # 用 _fetch_prices 获取信号日价格（TdxQuant 优先，AKShare 降级）
    codes_list = [c for c, w in signals.items() if w >= 1e-4]
    price_map = _fetch_prices(codes_list)

    records = []
    for code, weight in signals.items():
        if weight < 1e-4: continue
        close_px = price_map.get(code)
        if close_px is None:
            print(f"  ⚠ 获取 {code} 价格失败，跳过")
            continue
        records.append({
            "date": signal_date, "strategy": strategy,
            "code": code, "weight": weight,
            "signal_close": close_px,
            "exec_price": None, "exec_date": None, "slippage_pct": None,
        })

    if not records:
        print("  无有效信号记录"); return

    df_new = pd.DataFrame(records)
    if TRACK_FILE.exists():
        df_old = pd.read_csv(TRACK_FILE)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        TRACK_FILE.parent.mkdir(parents=True, exist_ok=True)
        df_all = df_new

    df_all.to_csv(TRACK_FILE, index=False)
    print(f"  ✓ 记录 {len(records)} 条信号 ({signal_date}) → {TRACK_FILE}")


def record_exec(code: str, exec_price: float, exec_date: str):
    """记录实际成交价格。"""
    import pandas as pd

    if not TRACK_FILE.exists():
        print("  ✗ 尚无信号记录，请先执行 record-signal"); return

    df = pd.read_csv(TRACK_FILE)
    # 找最近未成交的同代码记录
    mask = (df["code"] == code) & (df["exec_price"].isna())
    if mask.sum() == 0:
        print(f"  ✗ 未找到 {code} 的待成交记录"); return

    idx = df[mask].index[-1]
    signal_close = float(df.loc[idx, "signal_close"])
    slippage = (exec_price - signal_close) / signal_close

    df.loc[idx, "exec_price"]   = exec_price
    df.loc[idx, "exec_date"]    = exec_date
    df.loc[idx, "slippage_pct"] = slippage
    df.to_csv(TRACK_FILE, index=False)

    direction = "↑追涨" if slippage > 0 else "↓低于"
    print(f"  ✓ {code}  信号价={signal_close:.2f}  实际={exec_price:.2f}  "
          f"滑点={slippage:+.2%} {direction}")


def analyze_slippage():
    """分析累计真实滑点，判断是否需要调整回测参数。"""
    import pandas as pd
    import numpy as np

    if not TRACK_FILE.exists():
        print("  尚无数据，请先记录信号和成交"); return

    df = pd.read_csv(TRACK_FILE)
    df_exec = df[df["slippage_pct"].notna()].copy()
    df_exec["slippage_pct"] = df_exec["slippage_pct"].astype(float)

    total = len(df)
    executed = len(df_exec)
    pending = total - executed

    print(f"\n实盘滑点分析")
    print("=" * 55)
    print(f"  总记录: {total}  已成交: {executed}  待成交: {pending}")

    if executed == 0:
        print("  暂无已执行记录，请先记录成交价格"); return

    print()
    print(f"  平均滑点:    {df_exec['slippage_pct'].mean():+.3%}")
    print(f"  中位滑点:    {df_exec['slippage_pct'].median():+.3%}")
    print(f"  滑点标准差:  {df_exec['slippage_pct'].std():.3%}")
    print(f"  最大正滑点: {df_exec['slippage_pct'].max():+.3%}")
    print(f"  最大负滑点: {df_exec['slippage_pct'].min():+.3%}")

    # 按策略分组
    if "strategy" in df_exec.columns:
        print()
        print("  按策略分组:")
        for strat, grp in df_exec.groupby("strategy"):
            print(f"    {strat:<24} 均值={grp['slippage_pct'].mean():+.3%}  "
                  f"笔数={len(grp)}")

    # 与回测假设对比
    cfg = {}
    cfg_path = ROOT / "config.json"
    if cfg_path.exists():
        cfg = json.load(open(cfg_path))

    bt_slip = float(cfg.get("slippage_rate", 0.001))
    real_slip = df_exec["slippage_pct"].mean()
    abs_real = abs(real_slip)

    print()
    print(f"  回测假设滑点: {bt_slip:.3%}")
    print(f"  实际平均滑点: {real_slip:+.3%}")

    if abs_real > bt_slip * 2:
        print()
        print(f"  ⚠ 实际滑点是回测假设的 {abs_real/bt_slip:.1f} 倍！")
        print(f"    建议调整 config.json: \"slippage_rate\": {abs_real:.4f}")
        print(f"    并重新评估策略可行性")
    elif abs_real > bt_slip * 1.3:
        print(f"  △ 实际滑点略高于假设，轻微影响收益")
    else:
        print(f"  ✓ 实际滑点与回测假设吻合，参数合理")


# ─────────────────────────────────────────────────────────────────────────────
# 仓位计算器
# ─────────────────────────────────────────────────────────────────────────────

def calc_positions(
    signals: dict,
    total_capital: float,
    min_order: float = 5000.0,
    lot_size: int = 100,
    prices: dict = None,
):
    """
    根据信号权重和账户资金计算每只股票的目标股数。

    Parameters
    ----------
    signals       : {code: weight}
    total_capital : 账户总资产（元）
    min_order     : 最小下单金额（元）
    lot_size      : 最小交易单位（A股100股/手）
    prices        : {code: price}，None则自动获取当前价格
    """
    if prices is None:
        prices = _fetch_prices(list(signals.keys()))

    valid = {c: w for c, w in signals.items() if c in prices and prices[c] > 0}
    if not valid:
        print("  ✗ 无有效持仓（未获取到价格）"); return {}

    # 重新归一化
    total_w = sum(valid.values())
    norm = {c: w / total_w for c, w in valid.items()}

    result = {}
    allocated = 0.0
    skipped = []

    print(f"\n仓位计算（总资产={total_capital:,.0f} 元）")
    print("=" * 62)
    print(f"  {'代码':>10}  {'目标权重':>9}  {'股数':>7}  {'金额':>12}  {'实际权重':>9}")
    print("  " + "─" * 54)

    for code, weight in sorted(norm.items(), key=lambda x: -x[1]):
        target_amt = total_capital * weight
        if target_amt < min_order:
            skipped.append((code, target_amt))
            continue

        price = prices[code]
        raw_shares = target_amt / price
        shares = int(raw_shares / lot_size) * lot_size

        if shares == 0:
            skipped.append((code, target_amt))
            continue

        actual_amt = shares * price
        allocated += actual_amt
        actual_w = actual_amt / total_capital

        result[code] = {
            "shares": shares, "amount": actual_amt,
            "price": price, "weight_actual": actual_w,
        }
        print(f"  {code:>10}  {weight:>9.2%}  {shares:>7,}  "
              f"{actual_amt:>12,.0f}  {actual_w:>9.2%}")

    print("  " + "─" * 54)
    cash_reserve = total_capital - allocated
    print(f"  {'合计':>10}  {'':>9}  {'':>7}  "
          f"{allocated:>12,.0f}  {allocated/total_capital:>9.2%}")
    print(f"  现金留存: {cash_reserve:,.0f} 元 ({cash_reserve/total_capital:.1%})")

    if skipped:
        print(f"\n  跳过 {len(skipped)} 只（目标金额<{min_order:.0f} 或价格过高）：")
        for c, amt in skipped:
            print(f"    {c}: 目标={amt:.0f}元")

    return result


def _fetch_prices(codes: list) -> dict:
    """
    获取当前收盘价。
    优先使用 TdxQuant（本地通达信，无网络依赖），
    失败时降级到 AKShare。
    """
    prices = {}

    # 方式1：TdxQuant（优先，本地数据）
    try:
        _tq_dir = _find_tqcenter()
        if _tq_dir:
            import sys as _sys
            _sys.path.insert(0, _tq_dir)
        from tqcenter import tq as _tq
        _tq.initialize(__file__)
        try:
            today = date.today().strftime("%Y%m%d")
            tdx_codes = [_to_tdx_fmt(c) for c in codes]
            data = _tq.get_market_data(
                field_list=["Close"],
                stock_list=tdx_codes,
                start_time=today,
                end_time=today,
                dividend_type="none",
                period="1d",
            )
            if data and "Close" in data:
                close_df = data["Close"]
                for code, tdx_code in zip(codes, tdx_codes):
                    if tdx_code in close_df.columns:
                        val = close_df[tdx_code].iloc[-1]
                        if val and val > 0:
                            prices[code] = float(val)
            print(f"  ✓ TdxQuant 价格获取: {len(prices)}/{len(codes)} 只")
        finally:
            try: _tq.close()
            except: pass
        if prices:
            return prices
    except Exception as e:
        print(f"  · TdxQuant 不可用 ({type(e).__name__})，降级到 AKShare")

    # 方式2：AKShare 降级
    try:
        import akshare as ak
        today = date.today().strftime("%Y%m%d")
        for code in codes:
            c6 = code.replace("sh.", "").replace("sz.", "")
            try:
                df = ak.stock_zh_a_hist(
                    symbol=c6, period="daily",
                    start_date=today, end_date=today, adjust="qfq")
                if len(df) > 0:
                    prices[code] = float(df.iloc[-1]["收盘"])
            except Exception:
                pass
        if prices:
            print(f"  ✓ AKShare 价格获取: {len(prices)}/{len(codes)} 只")
    except ImportError:
        print("  ⚠ AKShare 未安装，无法获取价格")
    return prices


def _find_tqcenter() -> str:
    """自动找 tqcenter.py 目录"""
    import os
    candidates = [
        r"D:\SOFT(DONE)\tdx\ncb\PYPlugins\user",
        r"C:\new_tdx\PYPlugins\user",
        r"D:\new_tdx\PYPlugins\user",
    ]
    for p in candidates:
        if os.path.exists(os.path.join(p, "tqcenter.py")):
            return p
    return None


def _to_tdx_fmt(code: str) -> str:
    """sh.600519 -> 600519.SH"""
    if "." not in code:
        return code
    p = code.split(".", 1)
    if p[0].lower() in ("sh", "sz", "bj"):
        return f"{p[1]}.{p[0].upper()}"
    return code


def _load_today_signals(strategy: str, sig_date: str = None) -> dict:
    """从 signals/ 目录读取最近的信号"""
    sig_date = sig_date or date.today().strftime("%Y-%m-%d")
    sig_dir = ROOT / "signals" / sig_date
    if not sig_dir.exists():
        # 查找最近的信号目录
        base = ROOT / "signals"
        if not base.exists(): return {}
        dirs = sorted([d for d in base.iterdir() if d.is_dir()], reverse=True)
        if not dirs: return {}
        sig_dir = dirs[0]
        print(f"  ⚠ 未找到 {sig_date} 的信号，使用最近的 {sig_dir.name}")

    fn = sig_dir / f"signal_{strategy}.json"
    if not fn.exists():
        fn = sig_dir / "signal_merged.json"
    if not fn.exists():
        print(f"  ✗ 未找到信号文件: {sig_dir}"); return {}

    with open(fn, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("signal", {})


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="实盘辅助工具集合")
    sub = parser.add_subparsers(dest="cmd")

    # record-signal
    ps = sub.add_parser("record-signal", help="记录当日信号价格")
    ps.add_argument("--date",     default=date.today().strftime("%Y-%m-%d"))
    ps.add_argument("--strategy", required=True)

    # record-exec
    pe = sub.add_parser("record-exec", help="记录实际成交")
    pe.add_argument("--code",  required=True)
    pe.add_argument("--price", type=float, required=True)
    pe.add_argument("--date",  default=date.today().strftime("%Y-%m-%d"))

    # analyze
    sub.add_parser("analyze", help="分析累计滑点")

    # position-size
    pp = sub.add_parser("position-size", help="计算目标仓位")
    pp.add_argument("--capital",  type=float, default=200_000)
    pp.add_argument("--strategy", default=None)
    pp.add_argument("--date",     default=date.today().strftime("%Y-%m-%d"))

    args = parser.parse_args()

    if args.cmd == "analyze":
        analyze_slippage()

    elif args.cmd == "record-signal":
        signals = _load_today_signals(args.strategy, args.date)
        if signals:
            record_signal(args.date, args.strategy, signals)
        else:
            print(f"  ✗ 未找到 {args.strategy} 的信号，请先运行 main.py → 实盘信号")

    elif args.cmd == "record-exec":
        record_exec(args.code, args.price, args.date)

    elif args.cmd == "position-size":
        strategy = args.strategy
        if strategy is None:
            # 自动用最近的合并信号
            signals = _load_today_signals("merged", args.date)
            if not signals:
                # 尝试找第一个可用策略
                sig_dir = ROOT / "signals"
                if sig_dir.exists():
                    dirs = sorted([d for d in sig_dir.iterdir() if d.is_dir()], reverse=True)
                    if dirs:
                        for fn in sorted(dirs[0].glob("*.json")):
                            with open(fn) as f:
                                d = json.load(f)
                            signals = d.get("signal", {})
                            if signals:
                                strategy = d.get("strategy", fn.stem)
                                break
        else:
            signals = _load_today_signals(strategy, args.date)

        if not signals:
            print("  ✗ 未找到信号，请先在 main.py → 实盘信号 中生成信号")
        else:
            print(f"  策略: {strategy}  日期: {args.date}")
            calc_positions(signals, args.capital)

    else:
        parser.print_help()
