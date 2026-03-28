#!/usr/bin/env python3
"""
Q-UNITY V10 — main.py  (多级菜单版 · adata 双数据源版)
stamp_tax=0.0005 | adj_type=hfq | V10引擎

══ 数据源架构（2026-03 更新）══════════════════════════════════════════════
  adata（推荐）: 无封禁，日线/基本面/概念均可用，auto 模式自动优先
  AKShare（备用）: 有封IP风险，作为降级选项
  auto 模式: 自动检测 adata 可用性，不可用时降级 AKShare

══ 脚本接口完整审计 ═══════════════════════════════════════════════════════
  step0_download_ohlcv.py:
    --source [auto|adata|akshare]  ← 新增，默认 auto
    --start --end --workers --delay --force --incremental --test -n
    --codes --build-npy --parquet-dir --npy-dir

  step1_download_fundamental_akshare.py:
    --source [auto|adata|akshare]  ← 新增，默认 auto
    --start-year --workers --delay --test --force --verify

  step1_download_fundamental.py (BaoStock):
    --start-year --workers --test --force

  step3_build_fundamental_npy.py:
    无参数 (路径从 config.json 自动读取)

  step2_download_concepts.py:
    --mode [auto|manual|adata|pywencai|akshare|skip]  ← 新增 adata 模式
    --csv --source --delay

  step4_build_concept_npy.py:
    --min-stocks --max-concepts

  validate_npy.py:
    --npy-dir --verbose

══ Bug 修复（相对 QUNITY_V10_fixed.zip）══════════════════════════════════
  B-1 [P0] fast_runner_v10.run(): market_index 按回测窗口切片重建 Regime
  B-2 [P1] fast_runner_v10.run(): amount_bt 传入 build() 激活流动性过滤
  B-3 [P1] fast_runner_v10.run(): 注入缓存 _regime_limits 避免冗余 compute
  B-4 [P2] fast_runner_v10: _compute_rolling_liquidity_mask 补充实现
  B-5 [P2] alpha_signal._score_to_weights: 归一化后重新 clip max_single_pos
"""
from __future__ import annotations
import os, sys
from pathlib import Path

# ── [PATH-FIX] Project Root Auto-Detection ─────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ───────────────────────────────────────────────────────────────────────

import importlib.util, json, logging, time, traceback
from datetime import date, datetime
from math import isnan as import_math_isnan
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# ── matplotlib (可选) ─────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    _MPL = True

    # [FIX-B2] 设置中文字体，消除 CJK glyph missing 警告
    # 优先级：Windows SimHei > macOS PingFang > Linux Noto CJK > 内置 DejaVu（无中文）
    import matplotlib.font_manager as _fm
    _CJK_CANDIDATES = [
        "SimHei", "Microsoft YaHei", "PingFang SC", "STHeiti",
        "Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]
    _available_fonts = {f.name for f in _fm.fontManager.ttflist}
    _chosen_font = next((f for f in _CJK_CANDIDATES if f in _available_fonts), None)
    if _chosen_font:
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = [_chosen_font, "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False  # 正确显示负号
    # 无中文字体时抑制字形警告，避免日志污染
    import warnings as _mpl_warn
    _mpl_warn.filterwarnings("ignore", category=UserWarning, message=".*Glyph.*missing.*")

    # 交互后端（仅用于弹窗显示，失败则仅保存文件）
    _MPL_INTERACTIVE = False
    try:
        for _b in ("TkAgg", "Qt5Agg", "QtAgg", "MacOSX", "WXAgg"):
            try:
                matplotlib.use(_b)
                _MPL_INTERACTIVE = True; break
            except Exception:
                continue
    except Exception:
        pass
    if not _MPL_INTERACTIVE:
        matplotlib.use("Agg")
except ImportError:
    _MPL = False; _MPL_INTERACTIVE = False

# ── colorama (可选) ───────────────────────────────────────────────────────
try:
    from colorama import Fore, Style, init as _ci
    _ci(autoreset=True)
    def _c(t, col=""): return f"{col}{t}{Style.RESET_ALL}"
    RED = Fore.RED; GRN = Fore.GREEN; YLW = Fore.YELLOW
    BLU = Fore.BLUE; CYN = Fore.CYAN; BLD = Style.BRIGHT
except ImportError:
    def _c(t, col=""): return t
    RED = GRN = YLW = BLU = CYN = BLD = ""

# ── 终端工具 ──────────────────────────────────────────────────────────────
def _sep(w=68): return "─" * w

def _get_nav(res) -> "np.ndarray | None":
    """[BUG-NAV-OR] 安全获取 nav_array，避免 numpy 数组参与 or 运算抛 ValueError"""
    nav = getattr(res, "nav_array", None)
    if nav is not None:
        return nav
    return getattr(res, "nav", None)
def _section(t): print(f"\n{_sep()}\n  {t}\n{_sep()}")
def _ok(m):   print(_c(f"  [OK]  {m}", GRN))
def _err(m):  print(_c(f"  [ERR] {m}", RED))
def _info(m): print(_c(f"  [..]  {m}", CYN))
def _warn(m): print(_c(f"  [WARN]{m}", YLW))
def _head(t): print(_c(f"\n{'─'*60}\n  {t}\n{'─'*60}", BLU + BLD))

def _print_result_table(res, start: str = "", end: str = "") -> None:
    """
    以专业中文表格形式输出单策略回测结果。
    分三区块：① 收益概览  ② 风险指标  ③ 运行信息
    """
    W = 62  # 表格总宽度

    def _border(c_l="╔", c_r="╗", fill="═"):
        print(_c(c_l + fill * (W - 2) + c_r, BLU))

    def _mid_sep():
        print(_c("╠" + "═" * (W - 2) + "╣", BLU))

    def _row_title(title: str):
        inner = f"  【 {title} 】"
        print(_c("║" + inner.ljust(W - 2) + "║", BLU + BLD))

    def _row2(label1, val1, label2, val2, col1="", col2=""):
        """双列数据行"""
        cell_w = (W - 2) // 2
        left  = f"  {label1:<10s} {_c(str(val1), col1)}"
        right = f"  {label2:<10s} {_c(str(val2), col2)}"
        # 对齐（去除 ANSI 颜色转义码计算可见长度）
        import re
        vis_left  = len(re.sub(r'\x1b\[[0-9;]*m', '', left))
        vis_right = len(re.sub(r'\x1b\[[0-9;]*m', '', right))
        left_padded  = left  + " " * max(0, cell_w - vis_left)
        right_padded = right + " " * max(0, cell_w - vis_right - 1)
        print(_c("║", BLU) + left_padded + _c("│", BLU) + right_padded + _c("║", BLU))

    def _row1(label, val, col=""):
        """单列全宽行"""
        inner = f"  {label:<12s} {_c(str(val), col)}"
        import re
        vis = len(re.sub(r'\x1b\[[0-9;]*m', '', inner))
        print(_c("║", BLU) + inner + " " * max(0, W - 2 - vis) + _c("║", BLU))

    # ── 评级函数 ─────────────────────────────────────────────
    def _grade_sharpe(v):
        if v >= 1.5:  return _c("★★★ 优秀", GRN + BLD)
        if v >= 1.0:  return _c("★★☆ 良好", GRN)
        if v >= 0.5:  return _c("★☆☆ 一般", YLW)
        return _c("☆☆☆ 较差", RED)

    def _grade_ret(v):
        if v >= 0.20: return _c(f"{v:+.2%}", GRN + BLD)
        if v >= 0.08: return _c(f"{v:+.2%}", GRN)
        if v >= 0.0:  return _c(f"{v:+.2%}", YLW)
        return _c(f"{v:+.2%}", RED)

    def _grade_dd(v):
        if v <= 0.10: return _c(f"-{v:.2%}", GRN)
        if v <= 0.20: return _c(f"-{v:.2%}", YLW)
        return _c(f"-{v:.2%}", RED)

    def _grade_calmar(v):
        if v >= 1.0:  return _c(f"{v:.2f}", GRN + BLD)
        if v >= 0.5:  return _c(f"{v:.2f}", GRN)
        if v >= 0.0:  return _c(f"{v:.2f}", YLW)
        return _c(f"{v:.2f}", RED)

    # ── 取值 ──────────────────────────────────────────────────
    nm   = getattr(res, "strategy_name",  "N/A")
    tot  = getattr(res, "total_return",   float("nan"))
    ann  = getattr(res, "annual_return",  float("nan"))
    shr  = getattr(res, "sharpe_ratio",   float("nan"))
    dd   = getattr(res, "max_drawdown",   float("nan"))
    srt  = getattr(res, "sortino_ratio",  float("nan"))
    cal  = getattr(res, "calmar_ratio",   float("nan"))
    win  = getattr(res, "win_rate",       float("nan"))
    pf   = getattr(res, "profit_factor",  float("nan"))
    vol  = getattr(res, "volatility",     float("nan"))
    trn  = getattr(res, "turnover",       float("nan"))
    inv  = getattr(res, "invested_ratio", float("nan"))   # [FIX-REGIME-STATS] 持仓天比例
    buy_cnt  = getattr(res, "buy_count",      0)          # [NEW] 买入次数
    sell_cnt = getattr(res, "sell_count",     0)          # [NEW] 卖出次数
    final_pos= getattr(res, "final_positions",0)          # [NEW] 期末持仓股数
    ms   = getattr(res, "elapsed_ms",     0.0)
    pd_  = getattr(res, "params_dict",    {})
    nav  = _get_nav(res)
    nav_start = nav[0]  if nav is not None and len(nav) > 0 else 1.0
    nav_end   = (nav[-1] / nav[0]) if nav is not None and len(nav) > 0 and nav[0] > 0 else 1.0

    date_range = f"{start} → {end}" if start and end else "N/A"
    params_str = "  ".join(f"{k}={v}" for k, v in pd_.items()) if pd_ else "默认"

    # 胜率说明：只统计有持仓的交易日（排除Regime强制空仓、全止止损等空仓日）
    win_note = ""
    if not import_math_isnan(inv) and inv < 0.70:
        win_note = _c(f" ⚠持仓仅{inv:.0%}时间", RED)

    _border("╔", "╗", "═")
    _row_title(f"回测报告 · {nm}")
    _mid_sep()

    # ── 区块1：收益概览 ──────────────────────────────────────
    _row_title("收益概览")
    _row2("年化收益率", _grade_ret(ann),
          "累计收益率", _grade_ret(tot))
    _row2("期末净值",   _c(f"{nav_end:.4f}", GRN if nav_end >= 1 else RED),
          "持仓时间占比", _c(f"{inv:.1%}" if not import_math_isnan(inv) else "N/A",
                             GRN if inv >= 0.70 else (YLW if inv >= 0.40 else RED)))
    _row2("Sharpe",    f"{shr:.3f}  {_grade_sharpe(shr)}",
          "Calmar",    _grade_calmar(cal))
    _row1("策略胜率（持仓日）", _c(f"{win:.2%}", GRN if win >= 0.5 else YLW) + win_note)
    _mid_sep()

    # ── 区块2：风险指标 ──────────────────────────────────────
    _row_title("风险指标")
    _row2("最大回撤",   _grade_dd(dd),
          "年化波动率", _c(f"{vol:.2%}", YLW if vol > 0.20 else GRN))
    _row2("Sortino",   _c(f"{srt:.3f}", GRN if srt >= 1.0 else YLW),
          "盈亏比",     _c(f"{pf:.3f}", GRN if pf >= 1.5 else YLW))
    _row1("年均换手率", _c(f"{trn:.1%}/年", YLW if trn > 3.0 else CYN))
    _mid_sep()

    # ── 区块2b：交易活动（新增）─────────────────────────────
    _row_title("交易活动")
    total_trades = buy_cnt + sell_cnt
    _row2("买入操作次数", _c(f"{buy_cnt:,}次", CYN),
          "卖出操作次数", _c(f"{sell_cnt:,}次", CYN))
    _row2("合计交易次数", _c(f"{total_trades:,}次", CYN),
          "期末持仓股数", _c(f"{final_pos}只",
                             GRN if final_pos > 0 else YLW))
    if total_trades > 0:
        buy_ratio = buy_cnt / total_trades
        asymmetry = abs(buy_ratio - 0.5)
        asym_note = _c("（买卖对称）", GRN) if asymmetry < 0.1 else \
                    _c(f"（买卖不对称 {'偏买入' if buy_ratio > 0.5 else '偏卖出'}）", YLW)
        _row1("买卖比例",
              f"买{buy_ratio:.0%} / 卖{1-buy_ratio:.0%}  {asym_note}")
    _mid_sep()

    # ── 区块3：运行信息 ──────────────────────────────────────
    _row_title("运行信息")
    _row1("回测区间",   date_range)
    _row1("策略参数",   params_str)
    _row1("耗时",       f"{ms/1000:.1f}s")
    _border("╚", "╝", "═")
    print()

def _ask(prompt, default=""):
    hint = f" [{default}]" if default else ""
    try:
        r = input(f"\n  {prompt}{hint}: ").strip()
        return r if r else default
    except EOFError:
        print("\n  (EOF)"); return default
    except KeyboardInterrupt:
        print("\n  (中断)"); raise

def _ask_int(prompt, default, lo=0, hi=9999):
    while True:
        r = _ask(prompt, str(default))
        if r.lower() in ("q", "b", "back"): return default
        try:
            v = int(r)
            if lo <= v <= hi: return v
        except ValueError:
            pass
        print(f"  ！请输入 {lo}~{hi} 之间的整数")

def _confirm(prompt):
    return _ask(f"{prompt} (y/n)", "y").lower() in ("y", "yes", "是")

def _ask_float(prompt, default, min_val=0.0, max_val=1.0):
    while True:
        r = _ask(prompt, str(default))
        try:
            v = float(r)
            if min_val <= v <= max_val: return v
            _err(f"请输入 {min_val} ~ {max_val} 的个数值")
        except ValueError:
            _err("请输入有效的小数")

def _run(args):
    import subprocess
    return subprocess.run(args, cwd=str(ROOT)).returncode


# ── NAV 图表 ──────────────────────────────────────────────────────────────
def _ascii_chart(nav_norm, drawdown, dates, W=60, H=12):
    import numpy as np
    n = len(nav_norm)
    if n < 2: return
    idxs = [int(i * (n - 1) / (W - 1)) for i in range(W)]
    vals = [nav_norm[i] for i in idxs]
    vmin, vmax = min(vals), max(vals)
    span = max(vmax - vmin, 1e-6)
    trend = "↑盈利" if vals[-1] >= vals[0] else "↓亏损"
    print()
    print(f"  ┌─ 净值走势（{trend}）{'─' * (W - 12)}┐")
    for row in range(H, -1, -1):
        thr = vmin + span * row / H
        line = ""
        for col, v in enumerate(vals):
            if abs(v - thr) < span / H:
                line += "●" if v >= 1.0 else "○"
            elif v >= thr and (row == 0 or vals[col] < vmin + span * (row + 1) / H):
                line += "│"
            else:
                line += " "
        lbl = f"{thr:.3f}" if row % (H // 3 + 1) == 0 else "      "
        print(f"  │{line}│ {lbl}")
    def fd(i):
        try:
            d = str(dates[i]) if dates else ""
            return d[2:7] if len(d) >= 7 else d[:5]
        except Exception:
            return ""
    l, m, r = fd(0), fd(idxs[W // 2]), fd(idxs[-1])
    pm = (W - len(l) - len(r) - len(m)) // 2
    print(f"  └{'─' * W}┘")
    print(f"    {l}{' ' * max(0, pm)}{m}{' ' * max(0, pm)}{r}")
    print(f"    最终净值:{nav_norm[-1]:.4f}  最大回撤:{float(np.min(drawdown)) * 100:.2f}%")
    print()


def _plot_nav(nav, dates, name, save_path=None):
    import numpy as np
    arr = np.asarray(nav, dtype=np.float64)
    if len(arr) < 2: return
    nrm = arr / max(arr[0], 1e-10)
    pk = np.maximum.accumulate(nrm)
    dd = (nrm - pk) / pk
    print(f"\n  ── 净值曲线 [{name}] ──────────────────────────────────")
    _ascii_chart(nrm, dd, dates)
    if not _MPL: return
    try:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(13, 7), gridspec_kw={"height_ratios": [3, 1]})
        fig.suptitle(f"策略净值 — {name}", fontsize=14, fontweight="bold")
        x = list(range(len(nrm)))
        if dates and len(dates) == len(nrm):
            try:
                x = [datetime.fromisoformat(str(d))
                     if not isinstance(d, datetime) else d for d in dates]
                ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                plt.gcf().autofmt_xdate()
            except Exception:
                pass
        ax1.plot(x, nrm, color="#1F77B4", lw=1.6, label="策略净值", zorder=3)
        ax1.axhline(1.0, color="#AAAAAA", lw=0.9, linestyle="--")
        ax1.fill_between(x, nrm, 1.0, where=(nrm >= 1.0), alpha=0.13, color="#2CA02C")
        ax1.fill_between(x, nrm, 1.0, where=(nrm < 1.0), alpha=0.13, color="#D62728")
        ax1.annotate(
            f"终值{nrm[-1]:.3f}", xy=(x[-1], nrm[-1]),
            xytext=(-60, 8), textcoords="offset points",
            fontsize=9, color="#1F77B4",
            arrowprops=dict(arrowstyle="->", color="#1F77B4", lw=0.8))
        ax1.set_ylabel("净值", fontsize=10)
        ax1.legend(loc="upper left", fontsize=9)
        ax1.grid(True, alpha=0.25)
        ax2.fill_between(x, dd * 100, 0, color="#D62728", alpha=0.65)
        ax2.axhline(0, color="#888888", lw=0.6)
        ax2.set_ylabel("回撤(%)", fontsize=10)
        ax2.set_xlabel("日期", fontsize=10)
        ax2.grid(True, alpha=0.25)
        max_dd_val = float(np.min(dd)) * 100
        ax2.annotate(
            f"最大回撤{max_dd_val:.1f}%",
            xy=(x[int(np.argmin(dd))], max_dd_val),
            xytext=(20, -18), textcoords="offset points",
            fontsize=8, color="#D62728",
            arrowprops=dict(arrowstyle="->", color="#D62728", lw=0.8))
        plt.tight_layout()
        sp = save_path or f"data/nav_{name}_{date.today()}.png"
        Path("data").mkdir(exist_ok=True)
        plt.savefig(sp, dpi=130, bbox_inches="tight")
        _ok(f"净值曲线已保存 → {sp}")
        if _MPL_INTERACTIVE:
            plt.show()
        else:
            plt.close(fig)
            _info("（无图形界面，已保存 PNG）")
    except Exception as e:
        print(f"  ·  matplotlib 失败: {e}")


def _plot_nav_multi(nav_dict, save_path=None):
    if not _MPL or not nav_dict: return
    try:
        import numpy as np
        fig, ax = plt.subplots(figsize=(13, 6))
        fig.suptitle("多策略净值对比", fontsize=14, fontweight="bold")
        colors = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728",
                  "#9467BD", "#8C564B", "#E377C2", "#7F7F7F"]
        for ci, (sname, nav) in enumerate(nav_dict.items()):
            if nav is None or len(nav) < 2: continue
            a = np.asarray(nav, dtype=np.float64)
            nrm = a / max(a[0], 1e-10)
            ax.plot(list(range(len(nrm))), nrm,
                    label=sname, lw=1.4, color=colors[ci % len(colors)])
        ax.axhline(1.0, color="#AAAAAA", lw=0.8, linestyle="--")
        ax.set_ylabel("净值"); ax.set_xlabel("日期")
        ax.legend(loc="upper left", fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        sp = save_path or f"data/nav_compare_{date.today()}.png"
        Path("data").mkdir(exist_ok=True)
        plt.savefig(sp, dpi=120, bbox_inches="tight")
        _ok(f"多策略对比图已保存 → {sp}")
        plt.close(fig)
    except Exception as e:
        print(f"  ·  多策略对比图失败: {e}")


# ── 配置 ──────────────────────────────────────────────────────────────────
DEFAULT_CFG: Dict[str, Any] = {
    "npy_dir":           "data/npy_v10_hfq",
    "npy_v10_dir":       "data/npy_v10_hfq",
    "parquet_dir":       "data/daily_parquet_hfq",
    "parquet_dir_hfq":   "data/daily_parquet_hfq",
    "initial_cash":      1_000_000.0,
    "commission_rate":   0.0003,
    "stamp_tax":         0.0005,
    "slippage_rate":     0.001,
    "max_single_pos":    0.08,
    "hard_stop_loss":    0.20,
    "full_stop_dd":      0.15,
    "half_stop_dd":      0.08,
    "max_gap_up":        0.025,
    "vol_multiplier":    100,
    "min_avg_amount":    5_000_000.0,
    "max_holding_days":  0,
    "allow_fractional":  True,
    "participation_rate": 0.10,
    "data": {
        "parquet_dir":     "data/daily_parquet_hfq",
        "npy_dir":         "data/npy_v10_hfq",
        "fundamental_dir": "data/fundamental",
        "default_start":   "2015-01-01",
        "n_workers":       8,
    },
}


class Config:
    CONFIG_PATH = ROOT / "config.json"

    def __init__(self):
        self._data = self._load()

    def _load(self):
        if self.CONFIG_PATH.exists():
            try:
                with open(self.CONFIG_PATH, encoding="utf-8") as f:
                    user = json.load(f)
                merged = json.loads(json.dumps(DEFAULT_CFG))
                self._deep_merge(merged, user)
                return merged
            except Exception as e:
                print(f"  ! config.json 读取失败({e})，使用默认配置")
        return json.loads(json.dumps(DEFAULT_CFG))

    @staticmethod
    def _deep_merge(base, override):
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                Config._deep_merge(base[k], v)
            else:
                base[k] = v

    def save(self):
        with open(self.CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        _ok(f"配置已保存 → {self.CONFIG_PATH}")

    def get(self, *keys, default=None):
        d = self._data
        for k in keys:
            if not isinstance(d, dict) or k not in d: return default
            d = d[k]
        return d

    def set(self, *kv):
        *keys, value = kv
        d = self._data
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    @property
    def npy_dir(self) -> Path:
        raw = self.get("npy_v10_dir") or self.get("npy_dir") or "data/npy_v10"
        p = Path(raw)
        return p if p.is_absolute() else ROOT / p

    @property
    def parquet_dir(self) -> Path:
        raw = self.get("parquet_dir_qfq") or self.get("parquet_dir") or "data/daily_parquet_qfq"
        p = Path(raw)
        return p if p.is_absolute() else ROOT / p


# ── Runner ────────────────────────────────────────────────────────────────
_runner = None


def _get_runner(cfg: Config, force: bool = False):
    global _runner
    if _runner is not None and not force:
        return _runner
    from src.engine.fast_runner_v10 import FastRunnerV10
    print("  正在加载数据矩阵（首次约 5~30 秒）...")
    t0 = time.perf_counter()
    _runner = FastRunnerV10(cfg._data)
    _runner.load_data()
    _ok(f"数据加载完成（{time.perf_counter() - t0:.1f}s）")
    # [DATA-CHECK] 检查关键基本面文件是否存在，给出明确提示
    # [FIX-PE-ALIAS] pe_ttm.npy 和 valuation_peTTM.npy 等价，任一存在即通过
    _npy = _runner.npy_dir
    _missing = []
    for _aliases, _desc in [
        (["pe_ttm.npy", "valuation_peTTM.npy"], "PE估值(titan_alpha/alpha_max F2因子)"),
        (["fundamental_roe.npy"],               "ROE(titan_alpha/alpha_max质量因子)"),
        (["sue.npy"],                            "SUE超预期(titan_alpha F4因子)"),
        (["market_index.npy"],                   "市场指数(Regime检测)"),
    ]:
        if not any((_npy / _f).exists() for _f in _aliases):
            _missing.append(f"{_aliases[0]}  [{_desc}]")
    if _missing:
        _warn("以下基本面文件缺失，相关因子将退化为0（不影响运行，但会降低策略质量）：")
        for _m in _missing:
            print(f"    ⚠  {_m}")
        _warn("建议：数据管理 → 4a(TdxQuant估值) 或 4e(BaoStock估值) → 5(构建基本面npy)")
    # [FIX-JIT-WARMUP] 数据加载后立即触发 Numba JIT，让编译在回测前完成
    # 若缓存有效 < 0.5s，若需重新编译给用户进度提示
    import threading
    _jit_done = [False]
    def _do_warmup():
        try:
            e = FastRunnerV10.warmup_jit(_runner._risk_cfg)
            _jit_done[0] = True
            if e > 2.0:
                _ok(f"Numba JIT 编译完成（{e:.0f}s），引擎就绪 ✓")
        except Exception:
            pass
    t_jit = time.perf_counter()
    _do_warmup()   # 同步执行，确保首次回测时已编译完毕
    return _runner


def _load_strategies():
    vec = ROOT / "src" / "strategies" / "vectorized"
    for py in sorted(vec.glob("*_alpha.py")):
        try:
            __import__(f"src.strategies.vectorized.{py.stem}")
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# 1. 数据管理菜单 — 所有子进程调用已完整审计
# ═══════════════════════════════════════════════════════════════════════════
class DataMenu:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._checked = False
        self._bs = self._ak = self._ad = self._tq = False

    def _check_conn(self):
        if self._checked: return
        import io, contextlib, socket as sk
        _info("检测数据源连通性...")
        try:
            import baostock as bs
            buf = io.StringIO()
            prev = sk.getdefaulttimeout()
            sk.setdefaulttimeout(2.0)
            try:
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    r = bs.login(); bs.logout()
                self._bs = r.error_code == "0"
            finally:
                sk.setdefaulttimeout(prev)
        except Exception:
            pass
        for name, attr in [("akshare", "_ak"), ("adata", "_ad")]:
            try:
                __import__(name); setattr(self, attr, True)
            except ImportError:
                pass
        # [CONFIG] TdxQuant 路径从 config.json 读取，不写死
        try:
            _scripts_dir = str(ROOT / "scripts")
            if _scripts_dir not in sys.path:
                sys.path.insert(0, _scripts_dir)
            from tqcenter_utils import find_tqcenter as _ftu, import_tq as _itq
            _itq()   # 加入 sys.path，不初始化
            import importlib as _il_
            _il_.import_module("tqcenter")
            self._tq = True
        except Exception:
            pass
        self._checked = True
        _info(f"TdxQuant={'✓ 通达信可用' if self._tq else '✗未检测到'}  "
              f"adata={'✓' if self._ad else '✗'}  "
              f"AKShare={'✓' if self._ak else '✗'}  "
              f"BaoStock={'✓' if self._bs else '✗封禁/离线'}")

    def show(self):
        self._check_conn()
        while True:
            _section("【 数据管理 】")
            meta_ok = (self.cfg.npy_dir / "meta.json").exists()
            pq_cnt = (len(list(self.cfg.parquet_dir.glob("*.parquet")))
                      if self.cfg.parquet_dir.exists() else 0)

            print(_c("  [💡 提示] 狙击手策略(Sniper) 必须完成 [A] 或 [2+4+5+6] 以确保因子对齐", YLW))

            def s(bs=False, ak=False, ad=False):
                p = []
                if bs: p.append(f"BS:{'✓' if self._bs else '✗'}")
                if ak: p.append(f"AK:{'✓' if self._ak else '✗'}")
                if ad: p.append(f"adata:{'✓' if self._ad else '✗'}")
                return f"[{' '.join(p)}]" if p else ""

            print("  ─── 🚀 核心推荐 ────────────────────────────────────────")
            print(_c("  A.   一键全量下载与构建 (Sniper 模式专用, 推荐新机执行)", GRN + BLD))
            print("  1.   查看数据状态")
            print()
            tq_s = "✓通达信" if self._tq else "✗需开通达信"
            print(f"  ─── 日线数据 (HFQ 后复权) ──────────────────────────")
            print(f"  2.   TdxQuant 本地下载（推荐，零网络）  [{tq_s}]  [{pq_cnt}只]")
            print(f"  2a.  adata/AKShare QFQ（auto 自动选源）{s(ak=True,ad=True)}")
            print(f"  2b.  强制 adata HFQ                    {s(ad=True)}")
            print(f"  2c.  强制 AKShare HFQ                  {s(ak=True)}")
            print(f"  2d.  强制 BaoStock HFQ                 {s(bs=True)}")
            print(f"  2t.  测试模式（TdxQuant 10只验证）       [{tq_s}]")
            print()
            print(f"  ─── npy 矩阵构建 ─────────────────────────────────────")
            print(f"  3.   下载日线 + 立即构建 npy         [{'✓' if meta_ok else '未构建'}]")
            print(f"  3a.  仅构建 npy（Parquet 已存在时）")
            print(f"  3b.  对齐辅助矩阵（日线增量更新后执行）")
            print()
            print(f"  ─── 估值 / 基本面（建议完成）────────────────────────")
            print(f"  4.   TdxQuant 基本面（推荐，快速）       [{tq_s}]")
            print(f"  4a.  TdxQuant 行情估值 PE/PB/市值        [{tq_s}]")
            print(f"  4b.  TdxQuant 历史股本序列               [{tq_s}]")
            print(f"  4c.  下载季度基本面 (adata/AKShare auto) {s(ak=True,ad=True)}")
            print(f"  4d.  下载季度基本面 (强制 adata)         {s(ad=True)}")
            print(f"  4e.  下载PE/PB/isST估值 (BaoStock)       {s(bs=True)}")
            print(f"  5.   构建基本面 npy（step3）")
            print()
            print(f"  ─── 概念板块 / 行业 ─────────────────────────────────")
            print(f"  6.   TdxQuant 概念+行业（推荐）          [{tq_s}]")
            print(f"  6a.  TdxQuant 仅行业分类（rs_hyname）    [{tq_s}]")
            print(f"  6b.  TdxQuant 仅概念板块                 [{tq_s}]")
            print(f"  6c.  adata THS 概念（备用）              {s(ad=True)}")
            print()
            print(f"  ─── 实时行情 / 预警（需通达信已开）─────────────────")
            print(f"  7.   实时行情快照（盘中）               [{tq_s}]")
            print(f"  8.   全市场扫描 + 超短线预警推送         [{tq_s}]")
            print()
            print(f"  ─── 验证工具 ────────────────────────────────────────")
            print(f"  9.   验证数据完整性（validate_npy）")
            print(f"  9a.  复权兼容性检查")
            print(f"  r.   刷新连通性检测")
            print(f"  0.   返回主菜单")

            c = _ask("请选择", "0")
            if   c == "0":  break
            elif c.upper() == "A": self._one_click_setup()
            elif c == "1":  self._status()
            elif c == "2":  self._dl_daily_tdxquant()
            elif c == "2a": self._dl_daily_auto(test=False)
            elif c == "2b": self._dl_daily_adata_force()
            elif c == "2c": self._dl_daily_ak_force()
            elif c == "2d": self._dl_daily_bs()
            elif c == "2t": self._dl_daily_tdxquant(test=True)
            elif c == "3":  self._dl_and_build()
            elif c == "3a": self._build_npy_only()
            elif c == "3b": self._align_aux()
            elif c == "4":  self._dl_fund_tdxquant()
            elif c == "4a": self._dl_fund_tdxquant_val()
            elif c == "4b": self._dl_fund_tdxquant_shares()
            elif c == "4c": self._dl_fund_auto()
            elif c == "4d": self._dl_fund_adata_force()
            elif c == "4e": self._dl_valuation_bs()
            elif c == "5":  self._build_fund_npy()
            elif c == "6":  self._concept_tdxquant()
            elif c == "6a": self._concept_tdxquant_industry()
            elif c == "6b": self._concept_tdxquant_concepts()
            elif c == "6c": self._concept()
            elif c == "7":  self._realtime_snapshot()
            elif c == "8":  self._realtime_scan()
            elif c == "9":  self._validate()
            elif c == "9a": self._check_adj_compat()
            elif c.lower() == "r":
                self._checked = False; self._check_conn()
            else:
                print("  ！无效选项")

    def _one_click_setup(self):
        """[Sniper Mode] 一键全量初始化自动化流水线"""
        _head("一键全量初始化 (Sniper 模式全对齐)")
        _info("本操作将依次下载：HFQ日线 -> 实时估值 -> 板块分类 -> 构建全部NPY矩阵")
        _warn("预计总耗时：30-60 分钟。请确保网络畅通，不要中途关闭程序。")

        if not _confirm("确认开始全自动初始化？"): return

        t0 = time.perf_counter()

        # 1. 尝试使用 TdxQuant 下载日线，失败则降级 auto
        _section("STEP 1: 下载日线数据 (HFQ) + 市场指数")
        if self._tq:
            rc = _run([sys.executable, str(ROOT/"scripts"/"step0_download_tdxquant.py"), "--workers", "16"])
        else:
            # [FIX-A-01] 强制下载 sh000300 指数，确保 Regime 逻辑可用
            rc = _run([sys.executable, str(ROOT/"scripts"/"step0_download_ohlcv.py"), "--source", "auto", "--workers", "8"])
        if rc != 0: _err("日线下载失败，中止计划"); return

        # 2. 下载基本面与市值 (Sniper 核心依赖)
        _section("STEP 2: 同步基本面、实时市值与估值 (PE/ST)")
        if self._tq:
            _run([sys.executable, str(ROOT/"scripts"/"step1_download_fundamental_tdxquant.py"), "--valuation", "--workers", "16"])
        else:
            # [FIX-A-02] AKShare 模式下补全估值同步
            _run([sys.executable, str(ROOT/"scripts"/"step1_download_fundamental_akshare.py"), "--source", "auto", "--workers", "8"])

        # [NEW-A-03] 补丁：单独下载 PE/PB/isST 估值 (针对非 Tdx 用户的多源采集)
        if not self._tq and self._bs:
            _info("正在通过 BaoStock 补全 PE/PB 与 ST 标记数据...")
            _run([sys.executable, str(ROOT/"scripts"/"step0_patch_daily_fields.py"), "--mode", "download", "--workers", "8"])

        # 3. 下载概念板块
        _section("STEP 3: 拉取行业概念板块 (Relative Strength 依赖)")
        if self._tq:
            _run([sys.executable, str(ROOT/"scripts"/"step2_download_concepts_tdxquant.py"), "--all"])
        else:
            _run([sys.executable, str(ROOT/"scripts"/"step2_download_concepts.py"), "--mode", "auto"])

        # 4. 构建全量矩阵
        _section("STEP 4: 序列重构与 NPY 矩阵合成 (含 SUE 构建)")
        _run([sys.executable, str(ROOT/"scripts"/"step0_download_ohlcv.py"), "--build-npy"]) # 生成主序列
        _run([sys.executable, str(ROOT/"scripts"/"step3_build_fundamental_npy.py")])         # 生成基本面序列
        _run([sys.executable, str(ROOT/"scripts"/"step4_build_concept_npy.py")])             # 生成概念序列

        # [NEW-A-04] 验证数据完整性
        _section("STEP 5: 数据真实性物理指纹校验")
        _run([sys.executable, str(ROOT/"scripts"/"validate_npy.py")])

        total_min = (time.perf_counter() - t0) / 60
        _ok(f"✅ 全初始化完成！共耗时 {total_min:.1f} 分钟。")
        _ok("狙击手系统就绪：您现在可以进入回测菜单，开始单策略回测 (sniper_hfq_v1)。")

    def _status(self):
        _section("数据状态总览")
        import numpy as np
        npy = self.cfg.npy_dir
        mp = npy / "meta.json"
        if mp.exists():
            meta = json.loads(mp.read_text(encoding="utf-8"))
            sh = meta.get("shape", [0, 0])
            adj = meta.get("adj_type", "未知")
            _ok(f"npy 矩阵: {sh[0]}只 × {sh[1]}天  adj={adj}  → {npy}")
            if adj != "hfq":
                _warn("adj_type 不是 hfq！V10 引擎要求使用后复权(HFQ)以确保价格连续性")
            dates = meta.get("dates", [])
            if dates:
                _ok(f"日期范围: {dates[0]} ~ {dates[-1]}")
        else:
            _err("meta.json 不存在 → 请执行 [3] 构建 npy 矩阵")
        print()
        for stem, lbl, aliases in [
            ("valid_mask",       "有效掩码",        ["valid_mask.npy"]),
            ("pe_ttm",           "PE 估值",          ["pe_ttm.npy", "valuation_peTTM.npy"]),
            ("fundamental_roe",  "ROE 基本面",       ["fundamental_roe.npy"]),
            ("market_cap_total", "总市值",           ["market_cap_total.npy"]),
            ("is_st",            "ST 标记",          ["is_st.npy", "valuation_isST.npy"]),
            ("concept_ids",      "概念板块",         ["concept_ids.npy"]),
            ("sue",              "SUE 超预期",       ["sue.npy"]),
            ("market_index",     "市场指数 (Regime)", ["market_index.npy"]),
        ]:
            exists = any((npy / a).exists() for a in aliases)
            found  = next((a for a in aliases if (npy / a).exists()), f"{stem}.npy")
            _info(f"  {'✓' if exists else '✗'} {lbl:16s} ({found})")
        print()
        pq = self.cfg.parquet_dir
        if pq.exists():
            files = [f for f in pq.glob("*.parquet") if f.stem != "stock_list"]
            mb = sum(f.stat().st_size for f in files) / 1e6
            _ok(f"Parquet 日线: {len(files)}只  {mb:.0f}MB  → {pq}")
        else:
            _err("Parquet 目录不存在 → 请先执行 [2] 下载日线")

    def _check_adj_compat(self):
        """检查 Parquet 目录中各文件的复权类型是否一致（防止混用 QFQ/HFQ）"""
        _section("复权类型兼容性检查")
        import pandas as pd, numpy as np
        pq = self.cfg.parquet_dir
        if not pq.exists():
            _err("Parquet 目录不存在"); return
        files = [f for f in pq.glob("*.parquet") if f.stem != "stock_list"]
        if not files:
            _err("Parquet 目录为空"); return
        # 抽样检查
        sample = files[:min(50, len(files))]
        adj_types = {}
        for f in sample:
            try:
                df = pd.read_parquet(str(f), columns=["adj_type"])
                if len(df) > 0:
                    at = str(df.iloc[0]["adj_type"])
                    adj_types[at] = adj_types.get(at, 0) + 1
            except Exception:
                pass
        print()
        for at, cnt in sorted(adj_types.items(), key=lambda x: -x[1]):
            pct = cnt / sum(adj_types.values()) * 100
            mark = "✓" if at == "qfq" else "⚠"
            print(f"  {mark}  adj_type='{at}':  {cnt} 个文件  ({pct:.0f}%)")
        print()
        if len(adj_types) > 1:
            _warn("发现混合复权类型！不同来源数据无法合并构建 npy 矩阵")
            _warn("解决方案：删除 Parquet 目录，用同一数据源重新全量下载")
        elif "qfq" in adj_types:
            _ok(f"全部 {sum(adj_types.values())} 个文件均为 qfq 前复权 ✓")
        else:
            _warn(f"所有文件为 adj_type='{list(adj_types.keys())[0]}'，非 qfq")


    # ── [TDXQUANT] 基本面下载（step1） ──────────────────────────────────────
    def _dl_fund_tdxquant(self):
        """[4] TdxQuant 基础财务（ROE/EPS/行业/上市日等，来自 get_stock_info）"""
        _section("基本面下载 -- TdxQuant get_stock_info [层1]")
        if not self._tq:
            _err("需要通达信 TdxQuant（客户端已开且已登录）"); return
        script = ROOT / "scripts" / "step1_download_fundamental_tdxquant.py"
        if not script.exists():
            _err(f"脚本不存在: {script}"); return
        workers = _ask_int("并发线程数", 8, 1, 32)
        if not _confirm("确认开始下载（约30~60秒/全市场）"): return
        rc = _run([sys.executable, str(script), "--workers", str(workers)])
        if rc == 0: _ok("基本面下载完成 → data/fundamental/")
        else: _err("下载失败")

    def _dl_fund_tdxquant_val(self):
        """[4a] TdxQuant 实时估值（PE/PB/市值，来自 get_more_info）"""
        _section("估值下载 -- TdxQuant get_more_info [层2]")
        if not self._tq:
            _err("需要通达信 TdxQuant"); return
        script = ROOT / "scripts" / "step1_download_fundamental_tdxquant.py"
        if not script.exists():
            _err(f"脚本不存在: {script}"); return
        workers = _ask_int("并发线程数（建议8，5ms/只约25秒/全市场）", 8, 1, 32)
        if not _confirm("确认下载 PE/PB/市值估值"): return
        rc = _run([sys.executable, str(script),
                   "--workers", str(workers), "--valuation"])
        if rc == 0: _ok("估值下载完成 → fundamental_tdxquant_valuation.parquet")
        else: _err("下载失败")

    def _dl_fund_tdxquant_shares(self):
        """[4b] TdxQuant 历史股本序列（来自 get_gb_info）"""
        _section("股本历史 -- TdxQuant get_gb_info [层3]")
        if not self._tq:
            _err("需要通达信 TdxQuant"); return
        script = ROOT / "scripts" / "step1_download_fundamental_tdxquant.py"
        if not script.exists():
            _err(f"脚本不存在: {script}"); return
        if not _confirm("确认下载历史股本序列"): return
        rc = _run([sys.executable, str(script), "--shares"])
        if rc == 0: _ok("股本历史下载完成")
        else: _err("下载失败")

    # ── [TDXQUANT] 概念+行业下载（step2） ─────────────────────────────────
    def _concept_tdxquant(self):
        """[6] TdxQuant 概念+行业（行业分类+核心概念板块）"""
        _section("行业/概念下载 -- TdxQuant 通达信本地版")
        if not self._tq:
            _err("需要通达信 TdxQuant"); return
        script = ROOT / "scripts" / "step2_download_concepts_tdxquant.py"
        if not script.exists():
            _err(f"脚本不存在: {script}"); return
        _info("行业分类: get_stock_info.rs_hyname（约30秒，无需额外下载）")
        _info("概念板块: get_stock_list_in_sector（约2~5分钟，~30+个核心概念）")
        if not _confirm("确认开始下载行业+概念"): return
        rc = _run([sys.executable, str(script), "--all"])
        if rc == 0: _ok("行业/概念下载完成 → data/concepts/")
        else: _err("下载失败")

    def _concept_tdxquant_industry(self):
        """[6a] TdxQuant 仅行业分类（秒级）"""
        _section("行业分类 -- TdxQuant rs_hyname")
        if not self._tq:
            _err("需要通达信 TdxQuant"); return
        script = ROOT / "scripts" / "step2_download_concepts_tdxquant.py"
        if not script.exists():
            _err(f"脚本不存在: {script}"); return
        if not _confirm("确认下载行业分类"): return
        rc = _run([sys.executable, str(script), "--industry"])
        if rc == 0: _ok("行业分类完成 → industry_tdxquant.parquet")
        else: _err("下载失败")

    def _concept_tdxquant_concepts(self):
        """[6b] TdxQuant 仅概念板块"""
        _section("概念板块 -- TdxQuant get_stock_list_in_sector")
        if not self._tq:
            _err("需要通达信 TdxQuant"); return
        script = ROOT / "scripts" / "step2_download_concepts_tdxquant.py"
        if not script.exists():
            _err(f"脚本不存在: {script}"); return
        _info("将下载约30个核心概念（白酒/银行/半导体/新能源等）")
        _info("若 get_sector_list(12) 可用，将自动获取全量概念列表")
        if not _confirm("确认下载概念板块"): return
        rc = _run([sys.executable, str(script), "--concepts"])
        if rc == 0: _ok("概念下载完成 → tdxquant_concept_members.parquet")
        else: _err("下载失败")

        # ── [TDXQUANT] step0_download_tdxquant.py ────────────────────────────
    def _dl_daily_tdxquant(self, test=False):
        """[2] TdxQuant 通达信本地下载（最推荐，零网络依赖）"""
        mode = "测试模式（10只）" if test else "全量/增量"
        _section(f"下载日线 -- TdxQuant 通达信本地版 [{mode}]")
        if not self._tq:
            _err("未检测到通达信 TdxQuant 接口")
            _info("请确认：1.通达信客户端已打开  2.已完成盘后数据下载")
            _info("tqcenter.py 路径示例：D:\\SOFT(DONE)\\tdx\\ncb\\PYPlugins\\user")
            return
        script = ROOT / "scripts" / "step0_download_tdxquant.py"
        if not script.exists():
            _err(f"脚本不存在: {script}"); return
        if test:
            n = _ask_int("测试只数", 10, 1, 50)
            if not _confirm("确认开始测试下载"): return
            rc = _run([sys.executable, str(script), "--test", "--workers", str(n)])
        else:
            workers = _ask_int("并发线程数（建议16，TdxQuant本地无限速）", 16, 1, 64)
            start   = _ask("起始日期 YYYYMMDD", "20150101")
            incr    = _confirm("增量模式（跳过最新文件，否=全量）")
            patch   = _confirm("补充真实上市日期（首次全量后建议执行）")
            if not _confirm("确认开始下载"): return
            args = [sys.executable, str(script),
                    "--workers", str(workers), "--start", start]
            if incr:  args.append("--incremental")
            if patch: args.append("--patch-dates")
            rc = _run(args)
        if rc == 0:
            _ok("TdxQuant 日线下载完成 → data/daily_parquet_qfq/")
            _info("下一步：[3] 构建 npy 矩阵")
        else:
            _err("下载失败，请查看上方输出")
            _info("备选方案：[2a] adata/AKShare 模式")

    # ── [TDXQUANT] 实时行情快照 ───────────────────────────────────────────
    def _realtime_snapshot(self):
        """[7] 盘中实时快照（含PE/PB/市值）"""
        _section("实时行情快照（TdxQuant 通达信）")
        if not self._tq:
            _err("需要通达信 TdxQuant（客户端已开且已登录）"); return
        script = ROOT / "scripts" / "realtime_tdxquant.py"
        if not script.exists():
            _err(f"脚本不存在: {script}"); return
        raw = _ask("股票代码（空格分隔6位，回车=茅台+平安+宁德）", "600519 000001 300750")
        codes = raw.strip().split() if raw.strip() else ["600519", "000001", "300750"]
        rc = _run([sys.executable, str(script), "--snapshot"] + ["--codes"] + codes)
        if rc != 0: _err("快照失败，请确认通达信客户端已打开")

    # ── [TDXQUANT] 全市场扫描 + 超短线预警 ──────────────────────────────
    def _realtime_scan(self):
        """[8] 全市场扫描 + 信号推送到通达信自定义板块"""
        _section("全市场扫描 + 超短线预警推送（TdxQuant）")
        if not self._tq:
            _err("需要通达信 TdxQuant（客户端已开且已登录）"); return
        script = ROOT / "scripts" / "realtime_tdxquant.py"
        if not script.exists():
            _err(f"脚本不存在: {script}"); return
        _info("默认扫描条件：当日涨幅>5% 且 连涨>=2天")
        _info("选股结果推送到通达信自定义板块，客户端实时查看")
        block = _ask("通达信板块简称（需在客户端预先创建）", "QUNITY")
        if not _confirm("确认开始全市场扫描（约25秒/全市场5000只）"): return
        rc = _run([sys.executable, str(script), "--scan", "--block", block])
        if rc == 0: _ok(f"扫描完成，结果已推送到板块 [{block}]")
        else: _err("扫描失败，请确认通达信已打开")

    # ── step0_download_ohlcv.py: 日线下载（adata/AKShare 双数据源）────
    def _dl_daily_auto(self, test=False):
        """[2] auto 模式：优先 adata，不可用时降级 AKShare"""
        mode = "测试模式（10只）" if test else "全量/增量"
        _section(f"下载日线 — auto 模式（优先 adata） [{mode}]")
        script = ROOT / "scripts" / "step0_download_ohlcv.py"
        if not script.exists(): _err(f"脚本不存在: {script}"); return
        if test:
            n = _ask_int("下载只数", 10, 1, 50)
            if not _confirm("确认开始测试下载"): return
            rc = _run([sys.executable, str(script), "--test", "-n", str(n),
                       "--source", "auto"])
        else:
            workers = _ask_int("并发线程数（建议8）", 8, 1, 32)
            start   = _ask("起始日期", self.cfg.get("data", "default_start") or "2015-01-01")
            end     = _ask("结束日期（留空=今日）", date.today().isoformat())
            force   = _confirm("强制重下全部（否=仅增量更新）")
            if not _confirm("确认开始下载"): return
            args = [sys.executable, str(script),
                    "--start", start, "--end", end, "--workers", str(workers),
                    "--source", "auto"]
            args.append("--force" if force else "--incremental")
            rc = _run(args)
        if rc == 0: _ok("日线下载完成（auto 模式）")
        else: _err("下载失败，请查看上方输出")

    def _dl_daily_adata_force(self):
        """[2a] 强制 adata 模式"""
        _section("下载日线 — adata QFQ（强制，无封禁风险）")
        if not self._ad:
            _err("adata 未安装: pip install adata"); return
        script = ROOT / "scripts" / "step0_download_ohlcv.py"
        if not script.exists(): _err(f"脚本不存在: {script}"); return
        workers = _ask_int("并发线程数（建议8）", 8, 1, 32)
        start   = _ask("起始日期", self.cfg.get("data", "default_start") or "2015-01-01")
        end     = _ask("结束日期（留空=今日）", date.today().isoformat())
        force   = _confirm("强制重下全部（否=仅增量）")
        if not _confirm("确认开始下载"): return
        args = [sys.executable, str(script),
                "--start", start, "--end", end, "--workers", str(workers),
                "--source", "adata"]
        args.append("--force" if force else "--incremental")
        rc = _run(args)
        if rc == 0: _ok("日线下载完成（adata QFQ）")
        else: _err("下载失败，请查看上方输出")

    def _dl_daily_ak_force(self):
        """[2b] 强制 AKShare 模式"""
        _section("下载日线 — AKShare QFQ（强制）")
        if not self._ak:
            _err("AKShare 未安装: pip install akshare -U"); return
        script = ROOT / "scripts" / "step0_download_ohlcv.py"
        if not script.exists(): _err(f"脚本不存在: {script}"); return
        workers = _ask_int("并发线程数（建议8~16）", 8, 1, 32)
        start   = _ask("起始日期", self.cfg.get("data", "default_start") or "2015-01-01")
        end     = _ask("结束日期（留空=今日）", date.today().isoformat())
        force   = _confirm("强制重下全部（否=仅增量）")
        if not _confirm("确认开始下载"): return
        args = [sys.executable, str(script),
                "--start", start, "--end", end, "--workers", str(workers),
                "--source", "akshare"]
        args.append("--force" if force else "--incremental")
        rc = _run(args)
        if rc == 0: _ok("日线下载完成（AKShare QFQ）")
        else: _err("AKShare 失败，请尝试 [2a] adata 模式")

    # ── src/data/baostock_downloader.py: BaoStock QFQ（V9验证，AK封IP备用）
    def _dl_daily_bs(self):
        _section("下载日线 — BaoStock QFQ（AKShare 封IP时备用）")
        _info("使用 V9.1 iter40 验证过的 baostock_downloader.py（已改为 QFQ adjustflag=1）")
        _info("BaoStock 免费账号，与 AKShare 数据节点完全独立")
        if not self._bs:
            _err("BaoStock 连接失败（IP被封或网络异常）")
            _info("检查: python -c \"import baostock as bs; print(bs.login().error_code)\"")
            return
        workers   = _ask_int("并发进程数（BaoStock 建议8）", 8, 1, 16)
        start     = _ask("起始日期", self.cfg.get("data","default_start") or "2015-01-01")
        end       = _ask("结束日期（留空=今日）", date.today().isoformat())
        force     = _confirm("强制重下全部（否=仅增量）")
        test_mode = _confirm("测试模式（10只，快速验证）")
        if not _confirm("确认开始下载（约 60~90 分钟）"): return
        try:
            sys.path.insert(0, str(ROOT))
            from src.data.baostock_downloader import BaostockDownloader
            from scripts.utils_paths import get_parquet_dir
            pq_dir = get_parquet_dir("qfq")
            pq_dir.mkdir(parents=True, exist_ok=True)
            dl = BaostockDownloader(output_dir=str(pq_dir), n_workers=workers)
            _info("获取股票列表...")
            codes = dl.load_stock_list()
            _ok(f"共 {len(codes)} 只A股")
            if test_mode:
                codes = codes[:10]; _info("测试模式: 仅下载 10 只")
            results = dl.download_all(start=start, end=end, codes=codes, force=force)
            ok_n  = sum(1 for v in results.values() if v == "ok")
            err_n = sum(1 for v in results.values() if v == "error")
            _ok(f"BaoStock 下载完成（QFQ）: 成功={ok_n}  失败={err_n}")
        except ImportError as e:
            _err(f"导入失败: {e}")
        except Exception as e:
            _err(f"BaoStock 下载失败: {e}")
            if _confirm("显示详细错误"):
                import traceback; traceback.print_exc()

    # [已移除旧 _dl_daily_adata，功能合并到 _dl_daily_auto/_dl_daily_adata_force]

    # ── step0 --build-npy: 下载完成后自动调用 build_npy()
    def _dl_and_build(self):
        _section("下载日线 + 构建 npy 矩阵（auto 数据源）")
        script = ROOT / "scripts" / "step0_download_ohlcv.py"
        if not script.exists():
            _err(f"脚本不存在: {script}"); return
        workers = _ask_int("并发线程数（建议8）", 8, 1, 32)
        start   = _ask("起始日期", self.cfg.get("data", "default_start") or "2015-01-01")
        npy_out = _ask("npy 输出目录", str(self.cfg.npy_dir))
        force   = _confirm("强制重下全部（否=仅增量）")
        if not _confirm("确认开始（下载+构建约 40~90 分钟）"): return
        args = [sys.executable, str(script),
                "--start", start, "--workers", str(workers),
                "--build-npy", "--npy-dir", npy_out,
                "--source", "auto"]
        args.append("--force" if force else "--incremental")
        rc = _run(args)
        if rc == 0:
            _ok(f"完成！npy 矩阵已保存 → {npy_out}")
            self.cfg.set("npy_v10_dir", npy_out)
            self.cfg.set("data", "npy_dir", npy_out)
            self.cfg.save()
        else:
            _err("执行失败，请查看上方输出")

    # ── step0 --incremental --build-npy: 跳过已有文件，只构建 npy
    def _build_npy_only(self):
        """[FIX-3A] 仅构建 npy，完全不发起网络请求。
        直接调用 src/data/build_npy.py，绕过 step0_download_ohlcv.py 的下载逻辑。
        """
        _section("仅构建 npy 矩阵（Parquet 已存在，零网络）")
        # 优先用纯构建脚本，无下载、无网络
        build_script = ROOT / "src" / "data" / "build_npy.py"
        step0_script = ROOT / "scripts" / "step0_download_ohlcv.py"

        npy_out = _ask("npy 输出目录", str(self.cfg.npy_dir))
        pq_dir  = _ask("Parquet 目录", str(self.cfg.parquet_dir))
        if not Path(pq_dir).exists():
            _err(f"Parquet 目录不存在: {pq_dir}"); return
        if not _confirm("开始构建（约 5~15 分钟，零网络）"): return

        if build_script.exists():
            # 首选：src/data/build_npy.py，纯本地，无任何网络调用
            rc = _run([sys.executable, str(build_script),
                       "--parquet-dir", pq_dir,
                       "--npy-dir",     npy_out,
                       "--workers",     "8"])
        else:
            # 降级：step0 --build-npy（不传 --incremental，跳过下载尝试）
            _warn("build_npy.py 不存在，使用 step0 降级路径（可能产生网络请求噪音）")
            if not step0_script.exists():
                _err(f"脚本不存在: {step0_script}"); return
            rc = _run([sys.executable, str(step0_script),
                       "--build-npy",
                       "--parquet-dir", pq_dir,
                       "--npy-dir",     npy_out])

        if rc == 0:
            _ok(f"npy 矩阵构建完成 → {npy_out}")
            self.cfg.set("npy_v10_dir", npy_out)
            self.cfg.set("data", "npy_dir", npy_out)
            self.cfg.save()
        else:
            _err("构建失败，请查看上方输出")

    def _align_aux(self):
        _section("对齐辅助矩阵（前向填充至基准 T）")
        import gc, uuid, shutil, numpy as np
        npy = self.cfg.npy_dir
        mp = npy / "meta.json"
        if not mp.exists():
            _err("meta.json 不存在，请先执行 [3]"); return
        meta = json.loads(mp.read_text(encoding="utf-8"))
        sh = meta.get("shape", [0, 0])
        NB, TB = int(sh[0]), int(sh[1])
        if NB == 0 or TB == 0:
            _err(f"shape 异常: {sh}"); return
        _info(f"基准维度: {NB} × {TB}")
        SKIP = {"close","open","high","low","volume","amount","valid_mask","limit_pct"}
        n_ok = n_fix = n_skip = n_warn = 0
        for fp in sorted(npy.glob("*.npy")):
            if fp.stem in SKIP: n_skip += 1; continue
            try:
                mm = np.load(str(fp), mmap_mode="r", allow_pickle=False)
                nd, sh_, dt = mm.ndim, mm.shape, mm.dtype
                del mm; gc.collect()
            except Exception as e:
                print(f"  ⚠ {fp.stem}: {e}"); n_warn += 1; continue
            if nd != 2: n_skip += 1; continue
            NC, TC = sh_
            if NC == NB and TC == TB:
                print(f"  ✓ {fp.stem}  ({NC},{TC})  已对齐"); n_ok += 1; continue
            if TC == 0 or TC > TB: n_skip += 1; continue
            if NC != NB:
                print(f"  ✗ {fp.stem}  N不匹配，需重建"); n_warn += 1; continue
            try:
                gap = TB - TC
                arr = np.load(str(fp), allow_pickle=False)
                new = np.concatenate([arr, np.repeat(arr[:, -1:], gap, axis=1)], axis=1).astype(dt)
                del arr; gc.collect()
                tmp = str(npy / f"_tmp_{uuid.uuid4().hex}.npy")
                try:
                    np.save(tmp, new); del new; gc.collect()
                    shutil.move(tmp, str(fp))
                except Exception:
                    try: os.remove(tmp)
                    except: pass
                    raise
                print(f"  ✔ {fp.stem}  ({NC},{TC})→({NB},{TB})  +{gap}列 ffill")
                n_fix += 1
            except Exception as e:
                print(f"  ✗ {fp.stem}  ffill 失败: {e}"); n_warn += 1
        print(f"\n  汇总: 已对齐={n_ok}  修复={n_fix}  跳过={n_skip}  警告={n_warn}")
        if n_fix > 0: _ok(f"共修复 {n_fix} 个辅助矩阵")
        elif n_warn == 0: _ok("所有矩阵维度已一致")

    # ── step1_download_fundamental_akshare.py: adata/AKShare 双数据源 ──
    def _dl_fund_auto(self):
        """[4] auto 模式：优先 adata，不可用时降级 AKShare"""
        _section("下载季度基本面（auto：优先 adata）")
        _info("自动选源：adata 可用则用 adata，否则降级到 AKShare")
        script = ROOT / "scripts" / "step1_download_fundamental_akshare.py"
        if not script.exists():
            _err(f"脚本不存在: {script}"); return
        workers = _ask_int("并发线程数（建议8）", 8, 1, 32)
        test    = _confirm("测试模式（10只）")
        force   = _confirm("强制重新下载（否=已有数据跳过）")
        if not _confirm("确认开始下载"): return
        args = [sys.executable, str(script),
                "--source", "auto", "--workers", str(workers)]
        if test:  args.append("--test")
        if force: args.append("--force")
        rc = _run(args)
        if rc == 0: _ok("基本面数据下载完成 → data/fundamental/")
        else: _err("执行失败，请查看上方输出")

    def _dl_fund_adata_force(self):
        """[4a] 强制 adata 模式"""
        _section("下载季度基本面（强制 adata，43字段）")
        if not self._ad:
            _err("adata 未安装: pip install adata"); return
        script = ROOT / "scripts" / "step1_download_fundamental_akshare.py"
        if not script.exists():
            _err(f"脚本不存在: {script}"); return
        workers = _ask_int("并发线程数（建议8）", 8, 1, 32)
        test    = _confirm("测试模式（10只）")
        force   = _confirm("强制重新下载")
        if not _confirm("确认开始下载"): return
        args = [sys.executable, str(script),
                "--source", "adata", "--workers", str(workers)]
        if test:  args.append("--test")
        if force: args.append("--force")
        rc = _run(args)
        if rc == 0: _ok("基本面数据下载完成（adata 43字段）")
        else: _err("执行失败")

    def _dl_fund_ak_force(self):
        """[4b] 强制 AKShare 模式"""
        _section("下载季度基本面（强制 AKShare）")
        if not self._ak:
            _err("AKShare 未安装: pip install akshare -U"); return
        script = ROOT / "scripts" / "step1_download_fundamental_akshare.py"
        if not script.exists():
            _err(f"脚本不存在: {script}"); return
        workers    = _ask_int("并发线程数（建议8~16）", 8, 1, 64)
        start_year = _ask_int("起始年份（建议2018）", 2018, 2010, 2030)
        test       = _confirm("测试模式（10只）")
        force      = _confirm("强制重新下载")
        if not _confirm("确认开始下载"): return
        args = [sys.executable, str(script),
                "--source", "akshare",
                "--start-year", str(start_year),
                "--workers", str(workers)]
        if test:  args.append("--test")
        if force: args.append("--force")
        rc = _run(args)
        if rc == 0: _ok("基本面数据下载完成（AKShare）")
        else: _err("执行失败，请尝试 [4a] adata 模式")

    # ── step1_download_fundamental.py 接口:
    #   --start-year --workers --test --force  (无 --mode)
    def _dl_fund_bs(self):
        _section("下载季度基本面（BaoStock）")
        if not self._bs:
            _err("BaoStock 连接失败，建议改用 [4] AKShare 版本"); return
        script = ROOT / "scripts" / "step1_download_fundamental.py"
        if not script.exists():
            _err(f"脚本不存在: {script}"); return
        workers    = _ask_int("并发进程数", 8, 1, 32)
        start_year = _ask_int("起始年份（建议2018）", 2018, 2010, 2030)
        test       = _confirm("测试模式（10只）")
        if not _confirm("确认开始下载"): return
        args = [sys.executable, str(script),
                "--start-year", str(start_year),
                "--workers", str(workers)]
        if test: args.append("--test")
        rc = _run(args)
        if rc == 0: _ok("基本面数据下载完成")
        else: _err("执行失败")

    # ── step0_patch_daily_fields.py: BaoStock PE/PB/isST（V9验证）
    def _dl_valuation_bs(self):
        _section("下载PE/PB/isST估值数据（BaoStock，V9验证版）")
        _info("使用 V9.1 验证过的 step0_patch_daily_fields.py")
        _info("字段: peTTM / pbMRQ / psTTM / isST（已改为读 V10 npy 路径）")
        if not self._bs:
            _err("BaoStock 连接失败，建议等解封后再执行"); return
        script = ROOT / "scripts" / "step0_patch_daily_fields.py"
        if not script.exists(): _err(f"脚本不存在: {script}"); return
        workers = _ask_int("并发进程数", 8, 1, 32)
        if not _confirm("开始下载（约 20~45 分钟）"): return
        rc = _run([sys.executable, str(script), "--mode", "download",
                   "--workers", str(workers)])
        if rc == 0: _ok("PE/PB/isST估值下载完成")
        else: _err("执行失败")

    # [已移除旧 _dl_fund_adata V9版，功能合并到 _dl_fund_adata_force（使用新版 step1）]

    # ── step3_build_fundamental_npy.py: 无参数，路径从 config.json 读取
    def _build_fund_npy(self):
        _section("构建基本面 npy 矩阵（step3）")
        _info("路径自动从 config.json 读取（scripts/utils_paths.py）")
        if not (self.cfg.npy_dir / "meta.json").exists():
            _err("请先执行 [3] 构建日线 npy 矩阵"); return
        script = ROOT / "scripts" / "step3_build_fundamental_npy.py"
        if not script.exists():
            _err(f"脚本不存在: {script}"); return
        if not _confirm("开始构建（约 2~5 分钟）"): return
        rc = _run([sys.executable, str(script)])
        if rc == 0: _ok("基本面 npy 矩阵构建完成")
        else: _err("step3 执行失败，请检查 data/fundamental/ 是否有数据")

    # ── step2 接口: --mode [auto|manual|adata|pywencai|akshare|skip]
    # ── step4 接口: --min-stocks --max-concepts
    def _concept(self):
        _section("【 概念板块数据下载 】")
        s2 = ROOT / "scripts" / "step2_download_concepts.py"
        s4 = ROOT / "scripts" / "step4_build_concept_npy.py"
        print(_c("  ─── 云端抓取模式 (网络请求) ──────────────────────────", CYN))
        print("  1. 智能自动模式 (auto)         - 系统自动选源")
        print("  2. 同花顺极速模式 (adata)      - 推荐，无封禁")
        print("  3. 问财脚本模式 (pywencai)     - Node-like 极速版")
        print("  4. AKShare 备用模式            - 有封 IP 风险")
        print(_c("  ─── 工具与合成 ──────────────────────────────────", CYN))
        print("  5. 构建 concept_ids.npy (Step 4) - 下载后必须执行")
        print("  6. 手动导入 ths_map.csv")
        print("  0. 返回")
        c = _ask("请选择", "0")
        if c == "1":
            if not s2.exists(): _err(f"脚本不存在: {s2}"); return
            rc = _run([sys.executable, str(s2), "--mode", "auto"])
            if rc == 0: _ok("概念板块下载完成（auto）")
        elif c == "2":
            if not self._ad:
                _err("adata 未安装: pip install adata"); return
            if not s2.exists(): _err(f"脚本不存在: {s2}"); return
            rc = _run([sys.executable, str(s2), "--mode", "adata"])
            if rc == 0: _ok("概念板块下载完成（adata THS）")
        elif c == "3":
            if not s2.exists(): _err(f"脚本不存在: {s2}"); return
            rc = _run([sys.executable, str(s2), "--mode", "pywencai"])
            if rc == 0: _ok("概念板块下载完成（pywencai）")
        elif c == "4":
            if not s2.exists(): _err(f"脚本不存在: {s2}"); return
            rc = _run([sys.executable, str(s2), "--mode", "akshare", "--source", "auto"])
            if rc == 0: _ok("概念板块下载完成（akshare）")
        elif c == "5":
            if not s4.exists(): _err(f"脚本不存在: {s4}"); return
            rc = _run([sys.executable, str(s4)])
            if rc == 0: _ok("concept_ids.npy 构建完成")
        elif c == "6":
            csv_path = _ask("ths_map.csv 路径（留空自动查找）", "")
            if not s2.exists(): _err(f"脚本不存在: {s2}"); return
            args = [sys.executable, str(s2), "--mode", "manual"]
            if csv_path: args += ["--csv", csv_path]
            rc = _run(args)
            if rc == 0: _ok("概念板块导入完成（manual）")

    # ── validate_npy 接口: --npy-dir --verbose
    def _validate(self):
        _section("验证数据完整性")
        script = ROOT / "scripts" / "validate_npy.py"
        if script.exists():
            verbose = _confirm("显示详细统计信息")
            args = [sys.executable, str(script), "--npy-dir", str(self.cfg.npy_dir)]
            if verbose: args.append("--verbose")
            _run(args)
        else:
            import numpy as np
            _info("validate_npy.py 不存在，手动检查：")
            # pe_ttm 和 valuation_peTTM 等价，任一存在即可
            _pe_found = any((self.cfg.npy_dir / f).exists()
                            for f in ["pe_ttm.npy", "valuation_peTTM.npy"])
            _pe_file  = next((f for f in ["pe_ttm.npy", "valuation_peTTM.npy"]
                              if (self.cfg.npy_dir / f).exists()), "pe_ttm.npy")
            if _pe_found:
                a = np.load(str(self.cfg.npy_dir / _pe_file), mmap_mode="r")
                _ok(f"{_pe_file}  shape={a.shape}  dtype={a.dtype}")
            else:
                _warn("pe_ttm.npy / valuation_peTTM.npy  不存在")
            for fn in ["close","open","high","low","volume","amount",
                       "valid_mask","fundamental_roe",
                       "market_cap_total","is_st","market_index"]:
                p = self.cfg.npy_dir / f"{fn}.npy"
                if p.exists():
                    a = np.load(str(p), mmap_mode="r")
                    _ok(f"{fn}.npy  shape={a.shape}  dtype={a.dtype}")
                else:
                    _warn(f"{fn}.npy  不存在")


# ═══════════════════════════════════════════════════════════════════════════
# 2. 回测菜单
# ═══════════════════════════════════════════════════════════════════════════
class BacktestMenu:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def show(self):
        while True:
            _section("【 回测 】")
            print("  1.  单策略回测              选择策略 + 时间段 + NAV 曲线")
            print("  2.  多策略组合回测          风险平价 / 等权 / 动量倾斜")
            print("  3.  参数扫描                top_n × rsrs_window 矩阵对比")
            print("  4.  Walk-Forward 优化       Optuna OOS 参数优化")
            print("  0.  返回主菜单")
            c = _ask("请选择", "0")
            if   c == "0": break
            elif c == "1": self._single()
            elif c == "2": self._multi()
            elif c == "3": self._scan()
            elif c == "4": self._optimize()
            else: print("  ！无效选项")

    def _pick(self):
        _load_strategies()
        from src.strategies.registry import list_vec_strategies
        strats = list_vec_strategies()
        if not strats:
            _err("无已注册策略，请检查 src/strategies/vectorized/"); return None, None
        print("  可用策略：")
        for i, s in enumerate(strats, 1): print(f"    {i:2d}. {s}")
        c = _ask("选择策略编号（或直接输入名称）", "1")
        try:
            st = strats[int(c) - 1]
        except (ValueError, IndexError):
            st = c
        if st not in strats:
            _err(f"策略 '{st}' 未注册"); return None, None
        return st, strats

    @staticmethod
    def _params(rsrs_w=18, zscore_w=300, top_n=25):
        class _P: pass
        p = _P()
        p.rsrs_window = rsrs_w; p.zscore_window = zscore_w
        p.top_n = top_n; p.annual_factor = 252.0
        p.to_dict = lambda: {"rsrs_window": rsrs_w, "top_n": top_n}
        return p

    def _single(self):
        _head("单策略回测")
        strategy, _ = self._pick()
        if not strategy: return
        start  = _ask("回测起始日期", "2019-01-02")
        end    = _ask("回测截止日期", date.today().isoformat())
        print("\n  参数配置（回车=使用默认值）：")
        rsrs_w   = _ask_int("rsrs_window（RSRS回归窗口）", 18, 10, 50)
        zscore_w = _ask_int("zscore_window（Z-score标准化窗口，影响预热期）", 300, 60, 600)
        top_n    = _ask_int("top_n（选股数）", 25, 5, 60)
        try:
            runner = _get_runner(self.cfg)
            p = self._params(rsrs_w, zscore_w, top_n)
            _info(f"运行 {strategy}  [{start} ~ {end}] ...")
            t0 = time.perf_counter()
            res = runner.run(strategy, p, start, end)
            _ok(f"回测完成（{time.perf_counter() - t0:.1f}s）")
            _print_result_table(res, start, end)
            nav = _get_nav(res)   # [BUG-NAV-OR]
            dates = getattr(res, "dates", None)
            if nav is not None and _confirm("显示净值曲线"):
                _plot_nav(nav, dates, strategy)
            if _confirm("显示诊断报告（Regime/持仓/换手率分解）"):
                try:
                    from scripts.debug_backtest import run_debug
                    run_debug(runner, strategy_name=strategy)
                except Exception as de:
                    _warn(f"诊断报告失败: {de}")
                    import traceback; traceback.print_exc()
            if nav is not None and _confirm("保存 NAV 到 CSV"):
                import pandas as pd
                out = ROOT / "results"; out.mkdir(exist_ok=True)
                fn = out / f"{strategy}_{start[:4]}_{end[:4]}_nav.csv"
                pd.DataFrame({"date": dates or list(range(len(nav))), "nav": nav}).to_csv(fn, index=False)
                _ok(f"已保存 → {fn}")
        except Exception as e:
            _err(f"回测失败: {e}")
            if _confirm("显示详细错误"): traceback.print_exc()

    def _multi(self):
        _head("多策略组合回测")
        _load_strategies()
        from src.strategies.registry import list_vec_strategies
        strats = list_vec_strategies()
        if not strats: _err("无策略"); return
        print("  可用策略：")
        for i, s in enumerate(strats, 1): print(f"    {i:2d}. {s}")
        sel = _ask("选择策略编号（逗号分隔，如 1,2,3）", "")
        selected = []
        for x in sel.split(","):
            try:
                idx = int(x.strip()) - 1
                if 0 <= idx < len(strats): selected.append(strats[idx])
            except ValueError:
                if x.strip() in strats: selected.append(x.strip())
        if len(selected) < 2: _err("至少需要 2 个策略"); return
        _info(f"已选: {selected}")
        print("\n  资金分配方法：")
        for k, v in [("1","equal（等权基准）"),("2","risk_parity（风险平价，推荐）"),("3","momentum_tilt（近期夏普加权）")]:
            print(f"    {k}. {v}")
        method = {"1":"equal","2":"risk_parity","3":"momentum_tilt"}.get(_ask("选择方法","2"), "risk_parity")
        start = _ask("回测起始", "2020-01-01"); end = _ask("回测截止", date.today().isoformat())
        try:
            from src.engine.portfolio_allocator import PortfolioAllocator
            from src.engine.optimizer_v10 import StrategyParams
            runner = _get_runner(self.cfg); alloc = PortfolioAllocator(method=method)
            _info(f"运行 {len(selected)} 策略组合（{method}）...")
            # [FIX-B1] 从 config.json strategy_params 注入各策略参数，不再硬传 None
            # [FIX-MULTI] 修复: {} 作为位置参数被吸入 *keys 导致 unhashable type: 'dict'
            # Config.get(*keys, default=None) 的 default 必须以关键字形式传入
            _sp_cfg = self.cfg.get("strategy_params", default={}) or {}
            strategy_configs = []
            for s in selected:
                sp = _sp_cfg.get(s, {})
                params_obj = StrategyParams(**sp) if sp else None
                strategy_configs.append((s, params_obj))
            res = runner.multi_run(strategy_configs, alloc, start, end)
            _ok("多策略组合完成")
            # 组合总体结果
            _print_result_table(res, start, end)
            # 各子策略明细
            if hasattr(res, "strategy_results") and res.strategy_results:
                print(_c("  ── 各子策略明细 ──────────────────────────────────", BLU))
                for name, r2 in res.strategy_results.items():
                    _print_result_table(r2, start, end)
            if _confirm("保存多策略净值对比图"):
                nd = {}
                for name, r2 in res.strategy_results.items():
                    nav = _get_nav(r2)   # [BUG-NAV-OR]
                    if nav is not None: nd[name] = nav
                if nd: _plot_nav_multi(nd)
        except Exception as e:
            _err(f"组合回测失败: {e}"); traceback.print_exc()

    def _scan(self):
        _head("参数扫描（top_n × rsrs_window）")
        strategy, _ = self._pick()
        if not strategy: return
        start   = _ask("回测起始", "2020-01-01"); end = _ask("回测截止", date.today().isoformat())
        top_ns  = [int(x) for x in _ask("top_n 列表（空格分隔）", "15 25 35").split() if x.isdigit()]
        rsrs_ws = [int(x) for x in _ask("rsrs_window 列表（空格分隔）", "18 22").split() if x.isdigit()]
        if not top_ns:  top_ns  = [25]
        if not rsrs_ws: rsrs_ws = [18]
        try:
            runner = _get_runner(self.cfg)
            rows, nd = [], {}
            total = len(top_ns) * len(rsrs_ws); idx = 0
            for tw in top_ns:
                for rw in rsrs_ws:
                    idx += 1; tag = f"top{tw}_rsrs{rw}"
                    _info(f"[{idx}/{total}] {tag} ...")
                    try:
                        res = runner.run(strategy, self._params(rw, 300, tw), start, end)
                        rows.append({"tag":tag,"ar":res.annual_return,"sr":res.sharpe_ratio,
                                     "dd":res.max_drawdown,"to":res.turnover})
                        nav = _get_nav(res)   # [BUG-NAV-OR]
                        if nav is not None: nd[tag] = nav
                    except Exception as e: _err(f"  {tag} 失败: {e}")
            print(f"\n  {'参数':<20} {'年化收益':>10} {'夏普':>8} {'最大回撤':>10} {'换手率':>8}")
            print(f"  {'─'*60}")
            for r in sorted(rows, key=lambda x: -x["sr"]):
                print(f"  {r['tag']:<20} {r['ar']:>+10.1%} {r['sr']:>8.2f} {r['dd']:>10.1%} {r['to']:>8.1f}")
            if nd and _confirm("\n保存多参数净值对比图"):
                _plot_nav_multi(nd, save_path=f"data/scan_{strategy}_{date.today()}.png")
        except Exception as e:
            _err(f"参数扫描失败: {e}"); traceback.print_exc()

    def _optimize(self):
        _head("Walk-Forward 参数优化（Optuna OOS）")
        strategy, _ = self._pick()
        if not strategy: return
        n_trials   = _ask_int("Optuna 试验次数（建议30~100）", 50, 5, 500)
        objective  = _ask("优化目标 sharpe/calmar/adj_sharpe", "sharpe")
        train_y    = _ask_int("训练窗口年数", 3, 1, 10)
        test_m     = _ask_int("测试窗口月数", 6, 1, 24)
        param_space: Dict[str, Any] = {"rsrs_window":(10,30),"zscore_window":(200,600),"top_n":(15,35)}
        _info(f"默认搜索空间: {param_space}")
        if not _confirm("使用默认搜索空间（否=自定义）"):
            _info("格式: param_name low high（回车结束）")
            param_space = {}
            while True:
                line = _ask(" > ", "").strip()
                if not line: break
                parts = line.split()
                if len(parts) == 3:
                    try: param_space[parts[0]]=(int(parts[1]),int(parts[2])); _ok(f"已添加: {parts[0]}")
                    except ValueError: _err("格式错误，示例: top_n 15 35")
        try:
            from src.engine.optimizer_v10 import OptimizerV10
            from src.engine.risk_config import RiskConfig
            runner = _get_runner(self.cfg); rc = RiskConfig()
            opt = OptimizerV10(runner, rc)
            _info(f"开始优化 {strategy}（{n_trials} trials）...")
            result = opt.optimize(strategy_name=strategy, param_space=param_space,
                                  objective=objective, n_trials=n_trials,
                                  wf_train_years=train_y, wf_test_months=test_m)
            _ok("优化完成")
            print(_c(f"\n  最优参数: {result['best_params']}", GRN + BLD))
            print(_c(f"  OOS {objective}: {result['best_value']:.4f}", GRN))
            if _confirm("保存优化报告到 JSON"):
                out = ROOT / "results"; out.mkdir(exist_ok=True)
                fn = out / f"optim_{strategy}_{date.today()}.json"
                with open(fn, "w", encoding="utf-8") as f:
                    json.dump({"strategy":strategy,"best_params":result["best_params"],
                               f"best_{objective}":result["best_value"]}, f, ensure_ascii=False, indent=2)
                _ok(f"已保存 → {fn}")
        except Exception as e:
            _err(f"优化失败: {e}"); traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════════
# 3. 实盘信号菜单
# ═══════════════════════════════════════════════════════════════════════════
class SignalMenu:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def show(self):
        while True:
            _section("【 实盘信号 】")
            print("  1.  生成今日目标持仓权重      单策略 / 全部策略")
            print("  2.  多策略信号等权合并        Regime 截断后输出")
            print("  3.  查看历史信号文件          signals/ 目录")
            print("  0.  返回主菜单")
            c = _ask("请选择", "0")
            if   c == "0": break
            elif c == "1": self._single()
            elif c == "2": self._merged()
            elif c == "3": self._history()
            else: print("  ！无效选项")

    def _single(self):
        _head("今日目标持仓权重")
        _load_strategies()
        from src.strategies.registry import list_vec_strategies
        strats = list_vec_strategies()
        if not strats: _err("无已注册策略"); return
        print("  可用策略：")
        for i, s in enumerate(strats, 1): print(f"    {i:2d}. {s}")
        c = _ask("选择策略编号（回车=全部策略）", "")
        if c:
            try: selected = [strats[int(c) - 1]]
            except (ValueError, IndexError): selected = [c] if c in strats else strats
        else:
            selected = strats
        today = date.today().strftime("%Y-%m-%d"); as_of = _ask("信号日期", today)
        top_n = _ask_int("显示 Top N 持仓", 10, 1, 50)
        try:
            runner = _get_runner(self.cfg)
            out_dir = ROOT / "signals" / as_of; out_dir.mkdir(parents=True, exist_ok=True)
            for strategy in selected:
                try:
                    sig = runner.realtime_signal(strategy, None, as_of)
                    if not sig: _warn(f"{strategy}: 无信号（BEAR 或数据不足）"); continue
                    ws = sum(sig.values()); _ok(f"{strategy}  →  {len(sig)} 只  权重和={ws:.1%}")
                    for code, w in sorted(sig.items(), key=lambda x: -x[1])[:top_n]:
                        print(f"      {code}  {w:.2%}")
                    fn = out_dir / f"signal_{strategy}.json"
                    with open(fn, "w", encoding="utf-8") as f:
                        json.dump({"strategy":strategy,"date":as_of,"weight_sum":ws,"signal":sig},
                                  f, ensure_ascii=False, indent=2)
                except Exception as e: _err(f"{strategy}: {e}")
            _ok(f"信号文件已保存 → {out_dir}")
        except Exception as e:
            _err(f"信号生成失败: {e}"); traceback.print_exc()

    def _merged(self):
        _head("多策略实盘信号合并与 OMS 差分调仓换流")
        _load_strategies()
        from src.strategies.registry import list_vec_strategies
        strats = list_vec_strategies()
        if not strats: _err("无策略"); return
        print("  可用策略：")
        for i, s in enumerate(strats, 1): print(f"    {i:2d}. {s}")
        sel = _ask("选择策略编号（逗号分隔，回车=全部）", "")
        selected = []
        if sel:
            for x in sel.split(","):
                try:
                    idx = int(x.strip()) - 1
                    if 0 <= idx < len(strats): selected.append(strats[idx])
                except ValueError:
                    if x.strip() in strats: selected.append(x.strip())
        if not selected: selected = strats
        today = date.today().strftime("%Y-%m-%d"); as_of = _ask("信号日期", today)
        
        # ── OMS 实盘配置 ──
        print("\n  [OMS 订单过滤配置]")
        deadband = _ask_float("仓位微调过滤阈值 (Deadband，如 0.005 代表 0.5%)", 0.005)
        
        try:
            runner = _get_runner(self.cfg); merged: Dict[str, float] = {}
            for strategy in selected:
                try:
                    sig = runner.realtime_signal(strategy, None, as_of)
                    for code, w in sig.items():
                        merged[code] = merged.get(code, 0.0) + w / len(selected)
                except Exception as e: _err(f"{strategy}: {e}")
            
            if not merged: _warn("合并信号为空"); return
            
            total = sum(merged.values())
            if total > 1.0:
                for c_ in merged: merged[c_] /= total
                
            _ok(f"最终合并信号: {len(merged)} 只  目标总仓位={sum(merged.values()):.1%}")
            
            # ── 模拟读取昨日持仓 ──
            current_pos_file = ROOT / "signals" / "current_positions.json"
            current_pos = {}
            if current_pos_file.exists():
                if _confirm(f"检测到昨日实盘持仓文件 ({current_pos_file.name})，是否加载以生成交易 Delta 指令？"):
                    try:
                        with open(current_pos_file, "r", encoding="utf-8") as f:
                            current_pos = json.load(f).get("positions", {})
                    except Exception as e:
                        _warn(f"加载昨日持仓失败: {e}，将假定当前为空仓")
            else:
                _info("未检测到 current_positions.json，假定当前为空仓。")
                
            # ── OMS Delta 差分计算与过滤 ──
            # 合并当前持仓标的和目标标的
            all_codes = set(merged.keys()).union(set(current_pos.keys()))
            orders = []
            final_targets = {}
            
            for code in all_codes:
                target_w = merged.get(code, 0.0)
                current_w = current_pos.get(code, 0.0)
                delta_w = target_w - current_w
                
                # OMS 核心过滤：死区 deadband 防抖，低于千分之五的微调直接放弃，继承旧仓位
                if abs(delta_w) < deadband:
                    # 放弃调仓，目标仓位沿用当前仓位
                    if current_w > 0:
                        final_targets[code] = current_w
                    continue
                    
                final_targets[code] = target_w
                
                if delta_w > 0:
                    orders.append((code, "买入", target_w, delta_w))
                elif delta_w < 0:
                    orders.append((code, "卖出(减仓)" if target_w > 0 else "清仓", target_w, delta_w))
                    
            # 重新归一化 final_targets 处理容差
            ft_sum = sum(final_targets.values())
            if ft_sum > 1.0:
                for c_ in final_targets: final_targets[c_] /= ft_sum
            
            # 打印交易指令书
            print(_c("  ── 🚀 实盘执行指令书 (OMS) ──────────────────────────", BLU))
            print(f"  过滤死区: {deadband:.2%}")
            
            # 按买卖动作与绝对变化量排序
            orders.sort(key=lambda x: x[3], reverse=True)
            
            buy_orders = [o for o in orders if o[3] > 0]
            sell_orders = [o for o in orders if o[3] < 0]
            
            print(_c("\n  [买入 / 增仓队列]", GRN))
            if not buy_orders: print("      (无)")
            for code, action, tgt, delta in buy_orders:
                print(f"      {code:10s} | {action:4s} | 变化: +{delta:5.2%}  ->  目标总仓位: {tgt:.2%}")
                
            print(_c("\n  [卖出 / 清仓队列]", RED))
            if not sell_orders: print("      (无)")
            for code, action, tgt, delta in sell_orders:
                print(f"      {code:10s} | {action:4s} | 变化: {delta:6.2%}  ->  目标总仓位: {tgt:.2%}")
                
            print(_c("  ─────────────────────────────────────────────────────", BLU))
            
            if _confirm("保存合并信号与交易指令到 JSON"):
                out = ROOT / "signals" / as_of; out.mkdir(parents=True, exist_ok=True)
                
                # 保存带有差分的完整信号
                output_data = {
                    "date": as_of,
                    "strategies_merged": selected,
                    "deadband_filter": deadband,
                    "target_signal": final_targets,
                    "orders": [{"code": o[0], "action": o[1], "target_weight": o[2], "delta_weight": o[3]} for o in orders]
                }
                
                with open(out / "signal_merged.json", "w", encoding="utf-8") as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                
                # 【极其重要】为了让明天能计算差分，把今天的目标覆盖为"昨日持仓"
                with open(ROOT / "signals" / "current_positions.json", "w", encoding="utf-8") as f:
                    json.dump({"date": as_of, "positions": final_targets}, f, ensure_ascii=False, indent=2)
                    
                _ok(f"信号与指令已保存 → {out / 'signal_merged.json'}")
                _ok(f"系统持仓快照已推进更新 → signals/current_positions.json")
                
        except Exception as e:
            _err(f"合并与指令生成失败: {e}"); traceback.print_exc()

    def _history(self):
        _section("历史信号文件")
        sr = ROOT / "signals"
        if not sr.exists(): _info("signals/ 目录不存在"); return
        days = sorted([d for d in sr.iterdir() if d.is_dir()], reverse=True)
        if not days: _info("signals/ 为空"); return
        for i, d in enumerate(days[:10], 1):
            print(f"    {i:2d}. {d.name}  （{len(list(d.glob('*.json')))} 个文件）")
        c = _ask("选择日期编号查看（回车跳过）", "")
        if not c: return
        try:
            dd = days[int(c) - 1]
            for fn in sorted(dd.glob("*.json")):
                with open(fn, encoding="utf-8") as f: data = json.load(f)
                sig = data.get("signal", {}); ws = sum(sig.values())
                _ok(f"{data.get('strategy', fn.stem)}: {len(sig)} 只  {ws:.1%}")
                for code, w in sorted(sig.items(), key=lambda x: -x[1])[:5]:
                    print(f"      {code}  {w:.2%}")
        except (ValueError, IndexError): pass


# ═══════════════════════════════════════════════════════════════════════════
# 4. 系统工具菜单
# ═══════════════════════════════════════════════════════════════════════════
class SystemMenu:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def show(self):
        while True:
            _section("【 系统工具 】")
            print("  1.  运行验收测试（test_v10_acceptance.py）")
            print("  2.  查看已注册策略列表")
            print("  3.  Numba JIT 预热（加速后续回测 10~50x）")
            print("  4.  验证铁律（stamp_tax / adj_type）")
            print("  5.  重新加载数据矩阵")
            print("  6.  查看 Bug 修复记录（B-1 ~ B-5）")
            print("  0.  返回主菜单")
            c = _ask("请选择", "0")
            if   c == "0": break
            elif c == "1": self._tests()
            elif c == "2": self._list_strats()
            elif c == "3": self._warmup()
            elif c == "4": self._iron_rules()
            elif c == "5": self._reload()
            elif c == "6": self._bugfix()
            else: print("  ！无效选项")

    def _tests(self):
        tf = ROOT / "tests" / "test_v10_acceptance.py"
        if not tf.exists(): _err(f"测试文件不存在: {tf}"); return
        rc = _run([sys.executable, str(tf)])
        if rc == 0: _ok("全部验收通过 ✓")
        else: _err("验收测试失败，请查看上方输出")

    def _list_strats(self):
        _load_strategies()
        from src.strategies.registry import list_vec_strategies
        strats = list_vec_strategies()
        _ok(f"已注册策略（{len(strats)} 个）：")
        for s in strats: print(f"    • {s}")

    def _warmup(self):
        _info("触发 Numba JIT 编译（首次约 30~90 秒，之后常驻缓存）...")
        try:
            from src.engine.fast_runner_v10 import FastRunnerV10
            from src.engine.risk_config import RiskConfig
            rc = RiskConfig()
            t0 = time.perf_counter()
            elapsed = FastRunnerV10.warmup_jit(rc)
            total = time.perf_counter() - t0
            if elapsed < 2.0:
                _ok(f"Numba 缓存命中（{elapsed:.2f}s），引擎已就绪 ✓")
            else:
                _ok(f"Numba JIT 编译完成（{total:.1f}s），后续回测将快 10~50 倍 ✓")
        except Exception as e:
            _warn(f"预热警告（通常可忽略）: {e}")

    def _iron_rules(self):
        _section("铁律验证")
        from src.engine.risk_config import RiskConfig
        rc = RiskConfig()
        if rc.stamp_tax == 0.0005: _ok(f"stamp_tax = {rc.stamp_tax}（万五）✓")
        else: _err(f"stamp_tax = {rc.stamp_tax}  !! 必须为 0.0005 !!")
        mp = self.cfg.npy_dir / "meta.json"
        if mp.exists():
            meta = json.loads(mp.read_text(encoding="utf-8"))
            adj = meta.get("adj_type", "未知")
            if adj == "qfq": _ok(f"adj_type = {adj} ✓")
            else: _warn(f"adj_type = {adj}  建议使用 qfq（前复权）")
        else:
            _warn("meta.json 不存在，无法验证 adj_type")

    def _reload(self):
        global _runner; _runner = None
        _ok("数据缓存已清除，下次回测时将重新加载")

    def _bugfix(self):
        _section("V10 Bug 修复记录（相对 QUNITY_V10_fixed.zip）")
        fixes = [
            ("P0","B-1","fast_runner_v10.run() Step 2",
             "MarketRegimeDetector 用全数据集 market_index 初始化；跨时段 Regime 时间轴错位",
             "每次 run() 按 [t_data_start:t_e] 切片重建 MarketRegimeDetector"),
            ("P1","B-2","fast_runner_v10.run() Step 5",
             "amount_bt 从未传入 PortfolioBuilder.build()；[H-02] 流动性过滤永不生效",
             "传入 amount_matrix=amount_bt"),
            ("P1","B-3","fast_runner_v10.run() Step 5",
             "build() 内部重复调用 compute()；Step 2 算好的 _regime_limits 被丢弃重算",
             "注入缓存 _port_builder._regime_limits = _regime_limits"),
            ("P2","B-4","fast_runner_v10 模块级",
             "_compute_rolling_liquidity_mask = None stub；调用必崩溃",
             "补充完整 NumPy 滚动均值实现"),
            ("P2","B-5","alpha_signal._score_to_weights()",
             "归一化缩放后单格权重可突破 max_single_pos",
             "缩放后重新 np.clip() 到 max_single_pos"),
        ]
        for lvl, bid, loc, symptom, fix in fixes:
            color = RED if lvl == "P0" else (YLW if lvl == "P1" else "")
            print(_c(f"\n  [{lvl}] {bid}", color))
            print(f"         位置: {loc}")
            print(f"         问题: {symptom}")
            print(f"         修复: {fix}")




# ═══════════════════════════════════════════════════════════════════════════
# 5. 策略优化工具菜单（专业建议实施）
# ═══════════════════════════════════════════════════════════════════════════
class OptimizeMenu:
    """
    将专业建议转化为可执行的分析工具:
    1. 前视偏差检查  → 确认基本面数据可信度（最高优先级）
    2. 单因子归因    → 找到真正有效的 Alpha 来源
    3. 流动性分层    → 识别小票幻觉
    4. Regime 切换成本 → 优化熊市判断参数
    5. 最终 OOS 测试 → 生命中只用一次的验证
    6. 实盘辅助工具  → 仓位计算 + 滑点追踪
    """
    def __init__(self, cfg): self.cfg = cfg

    def show(self):
        while True:
            _section("【 策略优化工具 】")
            print("  ─── 第一步：数据可信度（务必先做）──────────────────")
            print("  1.  前视偏差检查        财报用发布日 vs 期末日，可能影响所有基本面策略")
            print()
            print("  ─── 第二步：找到真实 Alpha ──────────────────────────")
            print("  2.  单因子归因分析      识别哪个因子真正贡献 Alpha + 检测衰减")
            print("  3.  流动性分层测试      判断 Alpha 是否依赖无法实盘的小票")
            print("  4.  Regime 切换成本     量化熊市判断参数对换手率的影响")
            print()
            print("  ─── 第三步：最终验证 ────────────────────────────────")
            print("  5.  最终 OOS 测试       生命中只用一次（2023-2024 锁定测试）")
            print("  5l. 查看 OOS 历史记录")
            print()
            print("  ─── 第四步：实盘辅助 ────────────────────────────────")
            print("  6.  仓位计算器          信号权重 → 每只股票买入股数")
            print("  7.  滑点分析            对比回测假设 vs 实际成交价偏差")
            print("  7r. 记录实际成交价      维护滑点追踪数据库")
            print()
            print("  0.  返回主菜单")

            c = _ask("请选择", "0")
            if   c == "0":  break
            elif c == "1":  self._lookahead()
            elif c == "2":  self._factor_attr()
            elif c == "3":  self._liquidity()
            elif c == "4":  self._regime_cost()
            elif c == "5":  self._oos_test()
            elif c == "5l": self._oos_list()
            elif c == "6":  self._position_size()
            elif c == "7":  self._slippage_analyze()
            elif c == "7r": self._slippage_record()
            else: print("  ！无效选项")

    def _lookahead(self):
        _head("前视偏差检查")
        script = ROOT / "scripts" / "check_fundamental_lookahead.py"
        if not script.exists(): _err(f"脚本不存在: {script}"); return
        rc = _run([sys.executable, str(script)])
        if rc == 0: _ok("数据可信度检查通过")
        else: _warn("发现潜在前视偏差，请查看上方详情")

    def _factor_attr(self):
        _head("单因子归因分析")
        _info("测试 titan_alpha_v1 各因子独立收益，约需 5~10 分钟...")
        script = ROOT / "scripts" / "factor_attribution.py"
        if not script.exists(): _err(f"脚本不存在: {script}"); return
        start = _ask("分析起始日期", "2019-01-01")
        end   = _ask("分析截止日期", date.today().isoformat())
        rc = _run([sys.executable, str(script), "--start", start, "--end", end])
        if rc != 0: _err("分析失败，请检查数据是否已构建")

    def _liquidity(self):
        _head("流动性分层测试")
        _info("逐步提高流动性门槛，观察 Alpha 是否消失...")
        script = ROOT / "scripts" / "liquidity_tier_analysis.py"
        if not script.exists(): _err(f"脚本不存在: {script}"); return
        start    = _ask("分析起始日期", "2020-01-01")
        end      = _ask("分析截止日期", date.today().isoformat())
        strategy = _ask("策略名称（回车=自动选第一个）", "")
        args = [sys.executable, str(script), "--start", start, "--end", end]
        if strategy: args += ["--strategy", strategy]
        rc = _run(args)
        if rc != 0: _err("分析失败")

    def _regime_cost(self):
        _head("Regime 切换成本分析")
        _info("对比不同 bear_confirm_days 对换手率和收益的影响...")
        script = ROOT / "scripts" / "analyze_regime_cost.py"
        if not script.exists(): _err(f"脚本不存在: {script}"); return
        start    = _ask("分析起始日期", "2019-01-01")
        end      = _ask("分析截止日期", date.today().isoformat())
        strategy = _ask("策略名称（回车=自动选第一个）", "")
        args = [sys.executable, str(script), "--start", start, "--end", end]
        if strategy: args += ["--strategy", strategy]
        _run(args)

    def _oos_test(self):
        _head("最终 OOS 测试（生命中只用一次）")
        _warn("此测试结果将被永久锁定，不允许因结果不好而重复运行")
        _info("请确认已完成 IS 期间（2015-2022）的所有参数优化")
        print()
        script = ROOT / "scripts" / "true_oos_test.py"
        if not script.exists(): _err(f"脚本不存在: {script}"); return

        # 触发策略注册
        _load_strategies()
        from src.strategies.registry import list_vec_strategies
        strats = list_vec_strategies()
        if not strats: _err("无已注册策略"); return
        print("  可用策略：")
        for i, s in enumerate(strats, 1): print(f"    {i:2d}. {s}")
        c = _ask("选择策略", "1")
        try: strategy = strats[int(c) - 1]
        except: strategy = c
        if strategy not in strats: _err("策略未注册"); return

        rsrs_w = _ask_int("rsrs_window", 18, 10, 50)
        top_n  = _ask_int("top_n", 25, 5, 60)
        args = [sys.executable, str(script),
                "--strategy", strategy,
                "--rsrs-window", str(rsrs_w),
                "--top-n", str(top_n)]
        _run(args)

    def _oos_list(self):
        _head("OOS 测试历史记录")
        script = ROOT / "scripts" / "true_oos_test.py"
        if not script.exists(): _err(f"脚本不存在: {script}"); return
        _run([sys.executable, str(script), "--list"])

    def _position_size(self):
        _head("仓位计算器")
        _info("根据信号权重和账户资金计算每只股票买入股数")
        script = ROOT / "scripts" / "live_trade_tools.py"
        if not script.exists(): _err(f"脚本不存在: {script}"); return
        capital = float(_ask("账户总资产（元）", "200000"))
        today   = date.today().strftime("%Y-%m-%d")
        sig_date = _ask("信号日期", today)
        args = [sys.executable, str(script), "position-size",
                "--capital", str(capital), "--date", sig_date]
        _run(args)

    def _slippage_analyze(self):
        _head("滑点分析报告")
        script = ROOT / "scripts" / "live_trade_tools.py"
        if not script.exists(): _err(f"脚本不存在: {script}"); return
        _run([sys.executable, str(script), "analyze"])

    def _slippage_record(self):
        _head("记录实际成交价")
        _info("每次实际成交后，记录价格以追踪真实滑点")
        script = ROOT / "scripts" / "live_trade_tools.py"
        if not script.exists(): _err(f"脚本不存在: {script}"); return
        code  = _ask("股票代码（如 sh.600519）", "")
        if not code: return
        price = _ask("实际成交价（元）", "")
        try: price = float(price)
        except: _err("价格格式错误"); return
        exec_date = _ask("成交日期", date.today().strftime("%Y-%m-%d"))
        _run([sys.executable, str(script), "record-exec",
              "--code", code, "--price", str(price), "--date", exec_date])

# ═══════════════════════════════════════════════════════════════════════════
# 风险提示 + 主程序
# ═══════════════════════════════════════════════════════════════════════════
def _disclaimer():
    print()
    print("  " + "═" * 66)
    print("  ⚠  风险提示与免责声明（请仔细阅读）")
    print("  " + "─" * 66)
    for line in [
        "本程序为量化研究工具，不构成任何形式的投资建议。",
        "策略目标绩效基于历史回测的内部估计，未经独立OOS验证；",
        "实际收益可能与目标存在重大偏差，包括大幅亏损。",
        "本程序不具备自动下单功能，所有交易须用户在券商平台手动执行。",
        "止损/持仓上限在实盘中须用户自律执行，程序无法替代风控纪律。",
        "量化策略可能因市场结构变化而失效，勿重仓依赖单一策略。",
    ]:
        print(f"  · {line}")
    print("  " + "═" * 66)
    _ask("我已阅读并理解上述风险提示，按 Enter 继续", "")


def main():
    print(_c("\n" + "═" * 68, BLU + BLD))
    print(_c("   Q-UNITY V10  量化回测框架  独立版  （多级菜单版）", BLU + BLD))
    print(_c("   stamp_tax=0.0005(万五) | adj_type=hfq | V10 引擎", CYN))
    print(_c("   Bug 修复: B-1(Regime切片) B-2(流动性) B-3(冗余) B-4 B-5", YLW))
    print(_c("═" * 68, BLU + BLD))

    if importlib.util.find_spec("numpy") is None:
        print(_c("  ✗  缺少依赖: numpy  请运行: pip install numpy", RED))
        sys.exit(1)

    cfg = Config()
    st = float(cfg.get("stamp_tax") or 0)
    if st != 0.0005: _warn(f"config.json stamp_tax={st}，应为 0.0005（万五）")
    else: _ok(f"铁律验证: stamp_tax={st} [OK]")

    df = ROOT / ".v10_disclaimer"
    if not df.exists():
        _disclaimer()
        try: df.touch()
        except Exception: pass

    dm = DataMenu(cfg); bm = BacktestMenu(cfg)
    sm = SignalMenu(cfg); sys_m = SystemMenu(cfg)
    # om instantiated inside MENU block below

    om = OptimizeMenu(cfg)

    MENU = [
        ("1", "【 数据管理 】     下载日线/基本面，三路冗余数据源", dm.show),
        ("2", "【 回测 】         单策略/多策略/参数扫描/WF优化",  bm.show),
        ("3", "【 实盘信号 】     生成今日目标持仓权重，信号合并", sm.show),
        ("4", "【 系统工具 】     验收/策略列表/Numba预热/Bug记录", sys_m.show),
        ("5", "【 策略优化工具 】 前视检查/因子归因/OOS测试/仓位计算", om.show),
        ("0", "退出", None),
    ]

    while True:
        print(_c("\n  主菜单", BLD))
        print(_c("  " + "─" * 60, BLU))
        for key, label, _ in MENU:
            print(f"    {_c(key, YLW + BLD)}  {label}")
        print(_c("  " + "─" * 60, BLU))
        choice = input(_c("\n  请选择: ", CYN)).strip()
        for key, _, fn in MENU:
            if choice == key:
                if fn is None:
                    print(_c("\n  再见！\n", GRN)); sys.exit(0)
                try:
                    fn()
                except KeyboardInterrupt:
                    print(_c("\n  已取消", YLW))
                except Exception as e:
                    _err(f"执行出错: {e}")
                    if _confirm("显示详细错误"): traceback.print_exc()
                break
        else:
            _warn("无效选项，请重新输入")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(_c("\n\n  用户中断，退出。\n", YLW))
        sys.exit(0)
