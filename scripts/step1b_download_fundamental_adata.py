"""
scripts/step1b_download_fundamental_adata.py

adata 专用季度基本面下载器（独立节点，BaoStock/AKShare 均封禁时的终极兜底）

────────────────────────────────────────────────────────────
版本兼容性（adata API 自适应探测）
────────────────────────────────────────────────────────────
adata 不同版本接口名称变化较大，本脚本用探测方式自动发现可用接口：

  已知版本映射（持续更新）：
  ≤2.8.x: adata.stock.finance.get_income_statement(stock_code)
  2.9.x:  接口名已变，运行时自动探测以下候选：
    stock_profit_history / get_profit_history / profit_history /
    income_statement / get_income / get_finance_indicator /
    stock_finance_indicator / get_stock_profit

启动时会打印探测到的真实接口名，并在首次调用成功后缓存，全程使用同一接口。

────────────────────────────────────────────────────────────
输出
────────────────────────────────────────────────────────────
  data/fundamental/fundamental_adata_{code}.csv   → 每只股票一个文件
  data/fundamental/fundamental_merged.csv          → 合并文件（供 step3 使用）

  字段（尽力填充，与 AKShare step1 输出兼容）：
    statDate, pubDate, roe, epsTTM, netProfit, totalShare,
    liqShare, netProfitYOY, roeYOY

────────────────────────────────────────────────────────────
并发说明
────────────────────────────────────────────────────────────
与 step0c 一样使用 ThreadPoolExecutor（HTTP I/O 密集，线程池最优）。
建议 --workers 4~8（基本面接口数据量大，适当放慢）。

────────────────────────────────────────────────────────────
用法
────────────────────────────────────────────────────────────
  # 先测试（自动探测 API + 下 5 只）
  python scripts/step1b_download_fundamental_adata.py --test

  # 全量
  python scripts/step1b_download_fundamental_adata.py --workers 6

  # 合并已有 CSV（不重下，仅重新合并）
  python scripts/step1b_download_fundamental_adata.py --merge-only
"""
from __future__ import annotations

# ── SSL 全局补丁（必须在 import adata 之前执行）─────────────────────────────
# [BUG-SSL-FIX] Windows 上部分网络环境（VPN/防火墙/ISP 中间件）会截断 TLS 握手，
# 导致 SSLEOFError: EOF occurred in violation of protocol。
# 关闭证书验证可绕过此问题（仅用于内网/受信任数据源，量化回测场景可接受）。
import ssl as _ssl
_ssl._create_default_https_context = _ssl._create_unverified_context

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# 路径常量
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR      = PROJECT_ROOT / "data"
FUND_DIR      = DATA_DIR / "fundamental"
# [V10-PATH-FIX] 从 config.json 读取 npy_v10_dir
try:
    from scripts.utils_paths import get_npy_dir as _get_npy_dir
except ImportError:
    import sys as _sys; _sys.path.insert(0, str(DATA_DIR.parent / "scripts"))
    from utils_paths import get_npy_dir as _get_npy_dir  # type: ignore
META_PATH = _get_npy_dir("v10") / "meta.json"

# ─────────────────────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────────────────────

def _to_full_code(code: str) -> str:
    s = str(code).strip()
    if s.startswith(("sh.", "sz.")):
        return s
    c = s.split(".")[-1].zfill(6)
    return f"{'sh' if c[0] in ('6','9') else 'sz'}.{c}"


def _to_6digit(code: str) -> str:
    return str(code).strip().split(".")[-1].zfill(6)


def _load_codes() -> list[str]:
    if META_PATH.exists():
        with open(META_PATH, encoding="utf-8") as f:
            return [_to_full_code(str(c)) for c in json.load(f)["codes"]]
    parquet_dir = _gpq("qfq") if "_gpq" in dir() else (PROJECT_ROOT / "data" / "daily_parquet_qfq")
    if parquet_dir.exists():
        codes = [_to_full_code(p.stem) for p in sorted(parquet_dir.glob("*.parquet"))
                 if p.stem != "stock_list"]
        if codes:
            return codes
    print("✗ 无法获取股票列表，请先运行 build_npy 或使用 --codes")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# adata 基本面接口探测（版本自适应）
# ─────────────────────────────────────────────────────────────────────────────

_FINANCE_CANDIDATES = [
    # ── 实测可用（已 confirmed） ─────────────────────────────────────────────
    # [BUG-SSL-FIX confirmed] adata.stock.finance 只暴露一个公开接口：
    #   get_core_index(stock_code) → adata.stock.finance.core.Core
    # 字段（已实测 2.9.x）：
    #   stock_code, short_name, report_date, report_type, notice_date,
    #   basic_eps, diluted_eps, net_profit_attr_sh, total_rev, gross_profit,
    #   net_margin, roe_wtd, roa_wtd, net_profit_yoy_gr, total_rev_yoy_gr,
    #   gross_margin, curr_ratio, quick_ratio, asset_liab_ratio, ...
    # 注意：★ 无 totalShare/liqaShare（股本数据在 adata.stock.finance 的其他接口）
    "get_core_index",           # adata 2.9.x confirmed ← 最高优先级
    # ── 其他版本历史候选 ─────────────────────────────────────────────────────
    "stock_profit_history",
    "get_profit_history",
    "profit_history",
    "get_income_statement",     # ≤2.8.x
    "income_statement",
    "get_finance_indicator",
    "stock_finance_indicator",
    "get_stock_profit",
    "get_income",
    "profit",
    "stock_income",
    "get_financial_data",
    "financial_data",
    "stock_finance_report",
    "cashflow_history",
    "balance_history",
]

# 全局缓存：一旦探测成功就固定使用同一接口
_g_finance_fn_name: Optional[str] = None
_g_finance_fn      = None
_g_finance_probe_done: bool = False   # [BUG-D-FIX] 区分「未探测」和「已探测但失败」
_probe_lock = Lock()


def _get_finance_fn():
    """探测并缓存 adata.stock.finance 的可用基本面接口。"""
    global _g_finance_fn_name, _g_finance_fn, _g_finance_probe_done

    # 快速路径：探测成功，直接返回缓存
    if _g_finance_fn is not None:
        return _g_finance_fn_name, _g_finance_fn

    # [BUG-D-FIX] 快速失败路径：已探测过但失败，不再重试。
    # 原代码 _g_finance_fn=None 无法区分「未探测」和「已探测失败」，
    # 导致探测失败后每个线程调用都会重新进入 _probe_lock 做完整探测（含网络请求），
    # 16 线程排队重复探测，造成卡顿和大量无效请求。
    if _g_finance_probe_done:
        return None, None

    with _probe_lock:
        # double-check（加锁后再次检查，防止多线程重复进入）
        if _g_finance_fn is not None:
            return _g_finance_fn_name, _g_finance_fn
        if _g_finance_probe_done:
            return None, None

        try:
            import adata
            finance_obj = adata.stock.finance
            # 打印所有可用接口（首次探测时）
            available = [x for x in dir(finance_obj) if not x.startswith("_")]
            print(f"\n  [adata.stock.finance 可用接口]: {available}")
            for name in _FINANCE_CANDIDATES:
                fn = getattr(finance_obj, name, None)
                if callable(fn):
                    # 用测试请求验证（只下 2 行数据）
                    try:
                        test_df = fn(stock_code="600519")
                        if test_df is not None and len(test_df) > 0:
                            _g_finance_fn_name = name
                            _g_finance_fn = fn
                            _g_finance_probe_done = True
                            print(f"  ✓ 基本面接口确认: adata.stock.finance.{name}()")
                            print(f"    字段: {test_df.columns.tolist()}")
                            return _g_finance_fn_name, _g_finance_fn
                    except Exception:
                        continue
            print(f"  ✗ 未找到可用基本面接口（已探测: {_FINANCE_CANDIDATES}）")
            print(f"    adata 可用: {available}")
        except ImportError:
            print("  ✗ adata 未安装")

        # 无论何种原因失败，标记为已探测，后续调用直接走快速失败路径
        _g_finance_probe_done = True
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# 字段标准化（输出与 step1_akshare 兼容）
# ─────────────────────────────────────────────────────────────────────────────

# ── get_core_index 实测字段精确映射表 ──────────────────────────────────────
# 来源：adata.stock.finance.get_core_index("600519") 实测（2.9.x confirmed）
#
#   adata 字段名               → 项目标准名    说明
#   ─────────────────────────────────────────────────────────────
#   report_date                → statDate     报告期末
#   notice_date                → pubDate      ★ 真实公告日（无需偏移估算）
#   roe_wtd                    → roeAvg       加权净资产收益率(%)
#   basic_eps                  → _eps_single  单季 EPS，需 rolling(4).sum()→epsTTM
#   net_profit_attr_sh         → netProfit    归母净利润（元）
#   total_rev                  → MBRevenue    营业总收入（元）
#   net_margin                 → npMargin     ★ 直接可用，净利润率(%)
#   gross_margin               → gpMargin     毛利率(%)
#   net_profit_yoy_gr          → YOYNI        ★ 净利润同比增长率(%)，直接可用
#   total_rev_yoy_gr           → (营收同比)    暂不入 step3 FACTOR_MAP
#   roa_wtd                    → (暂不使用)
#   asset_liab_ratio           → (资产负债率)  暂不入 FACTOR_MAP
#   ─────────────────────────────────────────────────────────────
#   ★ 无 totalShare / liqaShare（股本数据不在此接口）
#     → step3 market_cap 计算需依赖其他来源（AKShare / BaoStock）
#
_CORE_INDEX_MAP = {
    # 报告期 / 公告日
    "statDate":       ["report_date"],
    "pubDate":        ["notice_date"],            # 真实公告日
    # 盈利指标
    "roeAvg":         ["roe_wtd", "roe_non_gaap_wtd", "roe"],
    "_eps_single":    ["basic_eps", "diluted_eps"],  # 单季 EPS → rolling → epsTTM
    "netProfit":      ["net_profit_attr_sh", "net_profit", "net_income"],
    "MBRevenue":      ["total_rev", "total_revenue", "revenue", "operating_revenue"],
    "npMargin":       ["net_margin", "np_margin", "net_profit_margin"],
    "gpMargin":       ["gross_margin"],
    # YOY 增长率（get_core_index 直接提供，无需自算）
    "YOYNI":          ["net_profit_yoy_gr", "net_profit_yoy", "profit_yoy"],
    "_rev_yoy":       ["total_rev_yoy_gr"],       # 暂存，用于 YOY 校验
    # 其他版本接口的字段兜底
    "roeAvg_fallback":["roe_ttm", "return_on_equity", "weighted_roe", "roe_avg",
                       "净资产收益率"],
    "netProfit_fb":   ["profit_to_parent", "net_profit_attr", "net_profit_parent",
                       "归母净利润"],
    "MBRevenue_fb":   ["total_operating_revenue", "营业总收入", "营业收入"],
    "YOYNI_fb":       ["net_income_yoy", "yoy_net_profit", "profit_yoy"],
}


def _normalize_fundamental(df: pd.DataFrame, code: str) -> Optional[pd.DataFrame]:
    """
    将 adata 基本面 DataFrame 标准化为与 step1_akshare 输出一致的格式。

    精确适配 get_core_index 的实测字段（见上方 _CORE_INDEX_MAP）。
    同时对其他接口字段提供通用兜底。

    [BUG-ADATA-01-FIX] code 参数必须是 6 位裸码（"600519"），不能是 "sh.600519"。
    """
    df = df.copy()
    df["code"] = code  # 6 位裸码，如 "600519"

    def _pick(targets: list[str]) -> Optional[str]:
        """返回第一个在 df 中存在的列名。"""
        return next((c for c in targets if c in df.columns), None)

    def _num(col: str) -> pd.Series:
        return pd.to_numeric(df[col], errors="coerce")

    # ── 报告期（statDate）────────────────────────────────────────────────────
    stat_col = _pick(["report_date", "stat_date", "period", "report_period",
                      "end_date", "date", "quarter"])
    if stat_col is None:
        return None
    df["statDate"] = pd.to_datetime(df[stat_col], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["statDate"])
    if df.empty:
        return None

    # ── 排序（后续 rolling / shift 均依赖此顺序）────────────────────────────
    df = df.sort_values("statDate").reset_index(drop=True)

    # ── 公告日（pubDate）——前视偏差防护 ──────────────────────────────────────
    # [BUG-ADATA-03-FIX] get_core_index 有真实 notice_date，直接用。
    # 但若 notice_date == report_date（极少数情况），仍加 +60 天保护。
    pub_col = _pick(["notice_date", "ann_date", "pub_date", "announce_date",
                     "publish_date"])
    if pub_col:
        df["pubDate"] = pd.to_datetime(df[pub_col], errors="coerce").dt.strftime("%Y-%m-%d")
        # 检查：公告日必须晚于报告期，否则用 statDate+60 兜底
        pub_dt  = pd.to_datetime(df["pubDate"],  errors="coerce")
        stat_dt = pd.to_datetime(df["statDate"], errors="coerce")
        bad = pub_dt.isna() | (pub_dt <= stat_dt)
        if bad.any():
            fallback = (stat_dt + pd.Timedelta(days=60)).dt.strftime("%Y-%m-%d")
            df.loc[bad, "pubDate"] = fallback[bad]
    else:
        # 无任何公告日列 → 保守 +60 天
        df["pubDate"] = (
            pd.to_datetime(df["statDate"], errors="coerce") + pd.Timedelta(days=60)
        ).dt.strftime("%Y-%m-%d")

    # ── ROE（roeAvg）─────────────────────────────────────────────────────────
    # get_core_index: roe_wtd（加权 ROE，单位 %）
    roe_col = _pick(["roe_wtd", "roe_non_gaap_wtd", "roe",
                     "roe_ttm", "return_on_equity", "weighted_roe", "roe_avg", "净资产收益率"])
    if roe_col:
        df["roeAvg"] = _num(roe_col).astype("float32")
    else:
        df["roeAvg"] = np.nan

    # ── EPS_TTM（rolling 4 季度累加）──────────────────────────────────────
    # get_core_index: basic_eps（单季 EPS）→ rolling(4, min_periods=1).sum() = TTM EPS
    # [BUG-ADATA-04-FIX] min_periods=1，与 step1_akshare 一致
    eps_col = _pick(["basic_eps", "eps", "diluted_eps", "earnings_per_share",
                     "eps_basic", "每股收益"])
    if eps_col:
        df["epsTTM"] = _num(eps_col).rolling(4, min_periods=1).sum()
    else:
        df["epsTTM"] = np.nan

    # ── 净利润（netProfit，万元）──────────────────────────────────────────────
    # get_core_index: net_profit_attr_sh（归母净利润，单位：元）
    # [BUG-ADATA-08-FIX] step3 FACTOR_MAP 注释"净利润(万元)"，AKShare/BaoStock 版均以
    # 万元存储；get_core_index 返回的是元，需 ÷10000 统一单位，否则差 10000 倍。
    # 自动判断：中位绝对值 > 1e6（元量级）→ ÷10000 转为万元
    np_col = _pick(["net_profit_attr_sh", "net_profit_attr", "net_profit",
                    "net_income", "profit_to_parent", "net_profit_parent", "归母净利润"])
    if np_col:
        raw_np = _num(np_col)
        med_abs = raw_np.abs().dropna().median()
        if pd.notna(med_abs) and med_abs > 1e6:
            raw_np = raw_np / 10000.0
        df["netProfit"] = raw_np.astype("float64")
    else:
        df["netProfit"] = np.nan

    # ── 营业收入（MBRevenue，万元）───────────────────────────────────────────
    # get_core_index: total_rev（元）→ ÷10000 → 万元（同上）
    rev_col = _pick(["total_rev", "total_revenue", "revenue", "operating_revenue",
                     "total_operating_revenue", "MBRevenue", "营业总收入", "营业收入"])
    if rev_col:
        raw_rev = _num(rev_col)
        med_abs_rev = raw_rev.abs().dropna().median()
        if pd.notna(med_abs_rev) and med_abs_rev > 1e6:
            raw_rev = raw_rev / 10000.0
        _rev = raw_rev
        df["MBRevenue"] = _rev.astype("float64")
    else:
        _rev = None
        df["MBRevenue"] = np.nan

    # ── 净利率（npMargin，%）─────────────────────────────────────────────────
    # get_core_index: net_margin（直接可用，%）
    # [BUG-ADATA-06-FIX] 原来完全缺此字段
    npm_col = _pick(["net_margin", "net_profit_margin", "np_margin",
                     "npMargin", "净利率"])
    if npm_col:
        df["npMargin"] = _num(npm_col).astype("float32")
    else:
        # 自算兜底
        if "netProfit" in df.columns and _rev is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                df["npMargin"] = np.where(
                    _rev.abs() > 1e-9,
                    (_num("netProfit") / _rev * 100).clip(-500, 500),
                    np.nan,
                ).astype("float32")
        else:
            df["npMargin"] = np.nan

    # ── 毛利率（gpMargin，%）────────────────────────────────────────────────
    gpm_col = _pick(["gross_margin", "gross_profit_margin", "gpMargin", "毛利率"])
    if gpm_col:
        df["gpMargin"] = _num(gpm_col).astype("float32")

    # ── 总股本 / 流通股本（get_core_index 无此字段，填 NaN）─────────────────
    # ★ 重要说明：get_core_index 不含股本数据。
    #   totalShare/liqaShare = NaN 意味着 step3 的 market_cap 矩阵无法从 adata 单独构建，
    #   需要 AKShare stock_financial_abstract / BaoStock 补充，或接受市值因子缺失。
    for target, cands in [
        ("totalShare", ["total_share", "total_shares", "share_total",
                        "total_equity", "total_capital", "总股本"]),
        ("liqaShare",  ["float_share", "float_shares", "circulating_share",
                        "liq_share", "float_equity", "流通股本", "流通A股"]),
    ]:
        share_col = _pick(cands)
        if share_col:
            v = _num(share_col)
            med = v.dropna().median()
            # [BUG-ADATA-05-FIX] 阈值 10000（原 1000），覆盖超大市值股（工行 ~3520 亿股）
            if pd.notna(med) and 0 < med < 10000:
                v = v * 10000.0   # 亿股 → 万股
            df[target] = v.astype("float64")
        else:
            df[target] = np.nan

    # ── YOY 净利润同比（YOYNI，%）────────────────────────────────────────────
    # get_core_index: net_profit_yoy_gr（直接可用，已是 %）
    yoyni_col = _pick(["net_profit_yoy_gr", "net_profit_yoy", "profit_yoy",
                       "net_income_yoy", "yoy_net_profit", "YOYNI"])
    if yoyni_col:
        df["YOYNI"] = _num(yoyni_col).astype("float32")
    else:
        # 自算兜底
        if "netProfit" in df.columns:
            _np_s = _num("netProfit")
            _prev = _np_s.shift(4)
            df["YOYNI"] = (
                (_np_s - _prev) / _prev.abs().replace(0, float("nan")) * 100
            ).replace([float("inf"), float("-inf")], float("nan")).astype("float32")
        else:
            df["YOYNI"] = np.nan

    # ── YOY EPS（YOYEPSBasic，%）─────────────────────────────────────────────
    _eps_s  = pd.to_numeric(df.get("epsTTM"), errors="coerce")
    _eps_p  = _eps_s.shift(4)
    df["YOYEPSBasic"] = (
        (_eps_s - _eps_p) / _eps_p.abs().replace(0, float("nan")) * 100
    ).replace([float("inf"), float("-inf")], float("nan")).astype("float32")

    # ── YOY ROE（YOYROE，%）──────────────────────────────────────────────────
    _roe_s = pd.to_numeric(df.get("roeAvg"), errors="coerce")
    _roe_p = _roe_s.shift(4)
    df["YOYROE"] = (
        (_roe_s - _roe_p) / _roe_p.abs().replace(0, float("nan")) * 100
    ).replace([float("inf"), float("-inf")], float("nan")).astype("float32")

    # ── 最终输出字段（与 step1_akshare / step3 FACTOR_MAP 完全对齐）──────────
    keep = ["code", "statDate", "pubDate",
            "roeAvg", "npMargin", "gpMargin", "epsTTM", "netProfit", "MBRevenue",
            "totalShare", "liqaShare",
            "YOYNI", "YOYEPSBasic", "YOYROE"]
    out = df[[c for c in keep if c in df.columns]].copy()
    out = out.dropna(subset=["statDate"]).sort_values("statDate")
    return out if len(out) > 0 else None


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# [BUG-SHARE-FIX iter37] 股本补充：三级路径 + 不再静默吞错
# ─────────────────────────────────────────────────────────────────────────────
#
# 根因说明：
#   adata.stock.finance.get_core_index() 是财务绩效接口，返回字段为：
#   report_date, roe_wtd, basic_eps, net_profit_attr_sh, total_rev, net_margin 等。
#   它在设计上不包含 total_share（总股本）/ float_share（流通股本），
#   _normalize_fundamental 中对这两列尝试的所有候选字段名均不存在，因此全部为 NaN。
#
# 影响链：
#   totalShare/liqaShare 全 NaN
#   → step3 构建的 fundamental_total_share.npy / fundamental_liq_share.npy 全 NaN
#   → market_cap_total.npy / market_cap_circ.npy 全 NaN（close × NaN = NaN）
#   → fast_runner 加载的 mktcap 矩阵全 NaN
#   → titan_alpha_v1 / ultra_alpha_v1 的市值中性化步骤被跳过
#   → 组合系统性暴露小市值风格，Alpha 被小市值溢价污染，回测绩效虚高
#
# iter37 修复：
#   原版只依赖 AKShare stock_financial_abstract_ths 单一接口，且 except Exception: pass
#   静默吞掉所有错误（接口变更/限速/列名改动均无任何输出），导致有效率0%但毫无提示。
#
#   修复为三级路径 + 明确打印失败原因：
#   路径1：adata.stock.capital（2.9.x 新增，与 adata 同节点，优先）
#   路径2：AKShare stock_financial_abstract_ths（多参数组合适配各版本）
#   路径3：AKShare stock_individual_info_em（东财个股快照，无历史但有当前值，最后兜底）
#
# 单位说明：
#   AKShare/adata 返回单位因版本而异（万股 or 亿股），通过中位数阈值自动判断后统一转为万股。
#   项目标准：totalShare / liqaShare 单位 = 万股（与 step3 FACTOR_MAP 注释「总股本(万股)」一致）。


def _unify_share_unit(series: "pd.Series") -> "pd.Series":
    """亿股 → 万股：中位数 < 1000 时判断为亿股单位，×10000 转换。"""
    med = series.dropna().median()
    if pd.notna(med) and 0 < med < 1000:
        return series * 10000.0
    return series


def _supplement_shares_from_akshare(df: pd.DataFrame, code_6: str,
                                     verbose: bool = False) -> pd.DataFrame:
    """
    [iter37] 当 totalShare / liqaShare 全为 NaN 时，按三级路径补充历史股本数据。

    路径1：adata.stock.capital（2.9.x 新接口，与 adata 同节点）
    路径2：AKShare stock_financial_abstract_ths（多参数组合，适配各版本）
    路径3：AKShare stock_individual_info_em（仅当前值静态广播，最后兜底）

    [BUG-SILENT-FAIL-FIX] 原版 except Exception: pass 已改为打印错误，
    失败时明确告知用户哪个接口出了什么问题，便于诊断。
    """
    ts_valid = df["totalShare"].notna().any() if "totalShare" in df.columns else False
    ls_valid = df["liqaShare"].notna().any() if "liqaShare" in df.columns else False
    if ts_valid and ls_valid:
        return df  # 已有完整数据，无需补充

    # ─── 内部工具：将补充列左连接合并到 df ────────────────────────────────
    def _merge_sub(ak_sub: pd.DataFrame) -> pd.DataFrame:
        """ak_sub 必须含 statDate / totalShare_ak / liqaShare_ak 三列。"""
        nonlocal df
        df = df.merge(ak_sub, on="statDate", how="left")
        if not ts_valid and "totalShare_ak" in df.columns:
            df["totalShare"] = df.pop("totalShare_ak")
        else:
            df.drop(columns=["totalShare_ak"], inplace=True, errors="ignore")
        if not ls_valid and "liqaShare_ak" in df.columns:
            df["liqaShare"] = df.pop("liqaShare_ak")
        else:
            df.drop(columns=["liqaShare_ak"], inplace=True, errors="ignore")
        return df

    # ═══════════════════════════════════════════════════════════════════════
    # 路径1：adata.stock.capital（优先，与 adata 同节点，速度最快）
    # ═══════════════════════════════════════════════════════════════════════
    try:
        import adata
        capital_ns = getattr(adata.stock, "capital", None)
        if capital_ns is not None:
            for fn_name in ["get_capital", "capital_change", "get_share",
                            "share_change", "get_history_capital", "stock_capital"]:
                fn = getattr(capital_ns, fn_name, None)
                if not callable(fn):
                    continue
                try:
                    df_cap = fn(stock_code=code_6)
                    if df_cap is None or df_cap.empty:
                        continue
                    dcol   = next((c for c in ["trade_date", "date", "report_date",
                                               "end_date", "change_date"]
                                   if c in df_cap.columns), None)
                    ts_col = next((c for c in ["total_share", "total_capital",
                                               "total_equity", "总股本"]
                                   if c in df_cap.columns), None)
                    ls_col = next((c for c in ["float_share", "circulating_share",
                                               "流通股本", "流通A股", "liq_share"]
                                   if c in df_cap.columns), None)
                    if dcol and ts_col:
                        dc = df_cap.copy()
                        # 将日期归到季度末以便与 statDate 对齐
                        dc["statDate"] = (
                            pd.to_datetime(dc[dcol], errors="coerce")
                            .dt.to_period("Q").dt.end_time
                            .dt.strftime("%Y-%m-%d")
                        )
                        dc["totalShare_ak"] = _unify_share_unit(
                            pd.to_numeric(dc[ts_col], errors="coerce"))
                        dc["liqaShare_ak"] = (
                            _unify_share_unit(pd.to_numeric(dc[ls_col], errors="coerce"))
                            if ls_col else pd.Series(np.nan, index=dc.index)
                        )
                        ak_sub = (dc[["statDate", "totalShare_ak", "liqaShare_ak"]]
                                  .drop_duplicates("statDate"))
                        df = _merge_sub(ak_sub)
                        if df["totalShare"].notna().any():
                            if verbose:
                                print(f"  ✓ [{code_6}] 股本来自 adata.stock.capital.{fn_name}")
                            return df
                except Exception:
                    continue
    except Exception:
        pass

    # ═══════════════════════════════════════════════════════════════════════
    # 路径2：AKShare stock_financial_abstract_ths（多参数组合适配各版本）
    # ═══════════════════════════════════════════════════════════════════════
    ak_df      = None
    last_err   = "AKShare 未安装（pip install akshare -U）"
    param_note = ""
    try:
        import akshare as ak
        # 多参数组合：AKShare 各版本参数名不同
        param_tries = [
            {"symbol": code_6, "indicator": "按报告期"},
            {"symbol": code_6},                          # 某些版本无 indicator
            {"symbol": code_6, "indicator": "按年度"},
            {"symbol": code_6, "indicator": "报告期"},
            {"symbol": code_6, "period":    "按报告期"}, # 旧版参数名
        ]
        for kwargs in param_tries:
            try:
                tmp = ak.stock_financial_abstract_ths(**kwargs)
                if tmp is not None and not tmp.empty:
                    ak_df      = tmp
                    param_note = str(kwargs)
                    break
            except TypeError:
                continue  # 参数名不匹配，尝试下一组
            except Exception as e:
                last_err = str(e)
                if any(k in str(e) for k in ["429", "限流", "频繁", "blocked"]):
                    time.sleep(5)  # 限速时暂停再试一次
                    try:
                        tmp = ak.stock_financial_abstract_ths(**kwargs)
                        if tmp is not None and not tmp.empty:
                            ak_df = tmp; param_note = str(kwargs); break
                    except Exception:
                        pass
                continue

        if ak_df is not None and not ak_df.empty:
            date_col = next((c for c in ["报告期", "report_date", "date", "statDate"]
                             if c in ak_df.columns), None)
            if date_col:
                ak_df = ak_df.copy()
                ak_df["statDate"] = (pd.to_datetime(ak_df[date_col], errors="coerce")
                                     .dt.strftime("%Y-%m-%d"))
                ak_df = ak_df.dropna(subset=["statDate"])

                share_map = {
                    "totalShare_ak": ["总股本", "总股本(万股)", "总股本(亿股)",
                                      "total_share", "totalShare"],
                    "liqaShare_ak":  ["流通股本", "流通A股", "流通股本(万股)",
                                      "流通A股(万股)", "流通股本(亿股)", "流通A股(亿股)",
                                      "float_share", "liqaShare"],
                }
                found_any = False
                for target, cands in share_map.items():
                    col = next((c for c in cands if c in ak_df.columns), None)
                    if col:
                        ak_df[target] = _unify_share_unit(
                            pd.to_numeric(ak_df[col], errors="coerce"))
                        found_any = True
                    else:
                        ak_df[target] = np.nan

                if found_any:
                    ak_sub = (ak_df[["statDate", "totalShare_ak", "liqaShare_ak"]]
                              .drop_duplicates("statDate"))
                    df = _merge_sub(ak_sub)
                    if df["totalShare"].notna().any():
                        if verbose:
                            print(f"  ✓ [{code_6}] 股本来自 AKShare ths {param_note}")
                        return df
                else:
                    last_err = f"列名未识别，可用列: {ak_df.columns.tolist()[:8]}"
            else:
                last_err = f"无日期列，可用列: {ak_df.columns.tolist()[:8]}"

        # ─── 路径3：东方财富个股信息（静态快照，无历史，仅最后兜底）──────────
        try:
            info = ak.stock_individual_info_em(symbol=code_6)
            if info is not None and not info.empty and info.shape[1] >= 2:
                info_dict = dict(zip(
                    info.iloc[:, 0].astype(str),
                    info.iloc[:, 1].astype(str)
                ))
                ts_val = ls_val = None
                for k, v in info_dict.items():
                    v_s = str(v).replace(",", "").strip()
                    if "总股本" in k:
                        try:
                            ts_val = float(v_s.replace("亿", "")) * (
                                10000 if "亿" in v_s else 1)
                        except Exception:
                            pass
                    if ("流通" in k and "股" in k) or "流通A" in k:
                        try:
                            ls_val = float(v_s.replace("亿", "")) * (
                                10000 if "亿" in v_s else 1)
                        except Exception:
                            pass
                if ts_val is not None:
                    if not ts_valid:
                        df["totalShare"] = ts_val   # 静态广播：全部季度用同一值
                    if ls_val is not None and not ls_valid:
                        df["liqaShare"] = ls_val
                    if verbose:
                        print(f"  ✓ [{code_6}] 股本来自 AKShare em 个股快照（静态）")
                    return df
        except Exception:
            pass

    except ImportError:
        last_err = "AKShare 未安装（pip install akshare -U）"

    # ─── 全部路径均失败：打印诊断（[BUG-SILENT-FAIL-FIX] 不再静默！）──────
    print(f"  ⚠ [{code_6}] 股本补充三路均失败，market_cap 因子将缺失")
    print(f"    最后错误: {last_err[:100]}")
    print(f"    → 运行 --diagnose-shares 可探测当前环境可用的股本接口")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# [iter37 新增] 股本接口诊断函数（--diagnose-shares 模式调用）
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_share_sources(test_code: str = "600519") -> None:
    """
    快速探测当前环境下所有可用的股本数据接口，帮助用户确认修复路径。

    用法：
        python scripts/step1b_download_fundamental_adata.py --diagnose-shares
        python scripts/step1b_download_fundamental_adata.py --diagnose-shares --codes 600519
    """
    print("=" * 70)
    print("  [iter37] 股本数据接口诊断（探测所有已知路径）")
    print(f"  测试代码：{test_code}")
    print("=" * 70)

    # ── 路径1：adata.stock.capital ──────────────────────────────────────────
    print("\n[路径1] adata.stock.capital 命名空间：")
    try:
        import adata
        print(f"  adata 版本: {getattr(adata, '__version__', 'unknown')}")
        capital_ns = getattr(adata.stock, "capital", None)
        if capital_ns is None:
            print("  ✗ adata.stock.capital 不存在（版本 < 2.9 或接口已移除）")
            print("    升级：pip install adata -U")
        else:
            methods = [x for x in dir(capital_ns)
                       if not x.startswith("_") and callable(getattr(capital_ns, x, None))]
            print(f"  可用方法: {methods}")
            found = False
            for fn_name in methods:
                fn = getattr(capital_ns, fn_name, None)
                if not callable(fn):
                    continue
                try:
                    t0 = time.time()
                    df_c = fn(stock_code=test_code)
                    elapsed = time.time() - t0
                    if df_c is not None and not df_c.empty:
                        print(f"  ✓ adata.stock.capital.{fn_name}() → {len(df_c)}行 ({elapsed:.1f}s)")
                        print(f"    列名: {df_c.columns.tolist()}")
                        scols = [c for c in df_c.columns
                                 if any(k in str(c) for k in ["share","capital","股本","流通"])]
                        print(f"    股本相关列: {scols}")
                        if scols:
                            print(df_c[scols[:3]].tail(3).to_string(index=False))
                        found = True
                        break
                    else:
                        print(f"  · adata.stock.capital.{fn_name}() → 空")
                except Exception as e:
                    print(f"  · adata.stock.capital.{fn_name}() 失败: {e}")
            if not found:
                print("  ✗ capital 所有方法均失败或返回空")
    except ImportError:
        print("  ✗ adata 未安装（pip install adata）")

    # ── 路径2：AKShare stock_financial_abstract_ths ─────────────────────────
    print("\n[路径2] AKShare stock_financial_abstract_ths：")
    try:
        import akshare as ak
        print(f"  AKShare 版本: {ak.__version__}")
        fn = getattr(ak, "stock_financial_abstract_ths", None)
        if fn is None:
            print("  ✗ stock_financial_abstract_ths 不存在（AKShare 版本太旧或已改名）")
            print("    升级：pip install akshare -U")
        else:
            succeeded = False
            for kwargs in [
                {"symbol": test_code, "indicator": "按报告期"},
                {"symbol": test_code},
                {"symbol": test_code, "indicator": "按年度"},
            ]:
                try:
                    t0 = time.time()
                    df_s = ak.stock_financial_abstract_ths(**kwargs)
                    elapsed = time.time() - t0
                    if df_s is not None and not df_s.empty:
                        print(f"  ✓ stock_financial_abstract_ths({kwargs}) → {len(df_s)}行 ({elapsed:.1f}s)")
                        print(f"    列名: {df_s.columns.tolist()}")
                        scols = [c for c in df_s.columns
                                 if any(k in str(c) for k in ["股本","share","capital","流通"])]
                        print(f"    股本相关列: {scols}")
                        if scols:
                            print(df_s[scols[:3]].head(2).to_string(index=False))
                        succeeded = True
                        break
                    else:
                        print(f"  · stock_financial_abstract_ths({kwargs}) → 空")
                except TypeError as e:
                    print(f"  · stock_financial_abstract_ths({kwargs}) → 参数错误: {e}")
                except Exception as e:
                    print(f"  · stock_financial_abstract_ths({kwargs}) → {str(e)[:80]}")
            if not succeeded:
                print("  ✗ 所有参数组合均失败")

        # ── 路径3：东方财富个股信息 ─────────────────────────────────────────
        print("\n[路径3] AKShare stock_individual_info_em（静态快照）：")
        fn3 = getattr(ak, "stock_individual_info_em", None)
        if fn3 is None:
            print("  ✗ stock_individual_info_em 不存在")
        else:
            try:
                t0 = time.time()
                df_i = ak.stock_individual_info_em(symbol=test_code)
                elapsed = time.time() - t0
                if df_i is not None and not df_i.empty:
                    print(f"  ✓ stock_individual_info_em() → {len(df_i)}行 ({elapsed:.1f}s)")
                    if df_i.shape[1] >= 2:
                        info_dict = dict(zip(df_i.iloc[:,0].astype(str),
                                             df_i.iloc[:,1].astype(str)))
                        for k, v in info_dict.items():
                            if any(x in k for x in ["股本","流通"]):
                                print(f"    {k}: {v}")
                else:
                    print("  ✗ stock_individual_info_em → 空")
            except Exception as e:
                print(f"  ✗ stock_individual_info_em → {str(e)[:80]}")
    except ImportError:
        print("  ✗ AKShare 未安装（pip install akshare -U）")

    # ── 汇总建议 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  根据上面结果选择修复路径：")
    print("  · 有✓路径 → python scripts/step1b_download_fundamental_adata.py --supplement-shares-only")
    print("  · 全部✗   → python scripts/step1_download_fundamental_akshare.py --workers 8")
    print("              （AKShare 基本面包含完整股本数据，一步解决）")
    print("=" * 70)

_lock = Lock()
_consecutive_failures = 0


def _fetch_raw(code: str, delay: float) -> dict:
    """
    [BUG-B-FIX] 线程只负责 HTTP 请求，不做任何 pandas/CPU 操作。
    [SPEED-FIX iter37] 股本 HTTP 补充也移到线程内执行。

    原版 [BUG-B-FIX] 把 _supplement_shares_from_akshare 的 HTTP 调用放进主线程
    _process_raw，导致：
      · 16 个工作线程快速填满窗口（0.5s/只）
      · 主线程每处理一只就阻塞 1~3s（AKShare HTTP）
      · 实际吞吐 = 1 / AKShare调用时间 ≈ 0.3只/s，与线程数无关

    修复：_fetch_raw 在线程内依次做两个 HTTP 调用：
      1. adata.finance（基本面核心，~0.5s）
      2. AKShare 股本（~0.5~1.5s，三级路径）
    两者都是纯网络 I/O，GIL 在等待期间自动释放，16 线程真正并行。
    主线程 _process_raw 只做 pandas normalize + 写 CSV（无 HTTP），始终极快。

    返回:
      df_raw      — adata 原始 DataFrame（主线程 _normalize_fundamental 处理）
      shares_raw  — 股本查找结果 dict {totalShare: float|None, liqaShare: float|None,
                                       ts_series: list[(statDate,val)], ls_series: list[...]}
                    主线程直接用此结果合并到 df，无需再做 HTTP
    """
    global _consecutive_failures
    code_6 = _to_6digit(code)
    out_path = FUND_DIR / f"fundamental_adata_{code_6}.csv"

    if out_path.exists() and out_path.stat().st_size > 200:
        return {"code": code, "code_6": code_6, "status": "skip",
                "df_raw": None, "shares_raw": None, "msg": "已存在"}

    with _lock:
        extra = min(_consecutive_failures * 0.5, 10.0)
    if extra > 0:
        time.sleep(extra)

    time.sleep(random.uniform(0.1, 0.8))

    fn_name, fn = _get_finance_fn()
    if fn is None:
        return {"code": code, "code_6": code_6, "status": "error",
                "df_raw": None, "shares_raw": None,
                "msg": "adata 基本面接口未找到（参见启动时输出）"}

    # ── 第一步：adata 基本面 HTTP ────────────────────────────────────────────
    err = ""
    df_raw = None
    for attempt in range(3):
        try:
            df_raw = fn(stock_code=code_6)
            if df_raw is None or (hasattr(df_raw, "empty") and df_raw.empty):
                return {"code": code, "code_6": code_6, "status": "empty",
                        "df_raw": None, "shares_raw": None, "msg": "返回空"}
            break
        except Exception as e:
            err = str(e)
            if attempt < 2:
                time.sleep(2.0 ** attempt + random.uniform(0, 1))
    else:
        with _lock:
            _consecutive_failures += 1
        return {"code": code, "code_6": code_6, "status": "error",
                "df_raw": None, "shares_raw": None, "msg": f"3次重试失败: {err}"}

    time.sleep(delay + random.uniform(0, delay * 0.3))

    # ── 第二步：AKShare 股本 HTTP（三级路径，也在线程内）────────────────────
    # [SPEED-FIX iter37] 股本 HTTP 移到线程，不阻塞主线程处理循环
    shares_raw = _fetch_shares_raw(code_6)

    return {"code": code, "code_6": code_6, "status": "fetched",
            "df_raw": df_raw, "shares_raw": shares_raw, "msg": ""}


def _fetch_shares_raw(code_6: str) -> dict:
    """
    [SPEED-FIX iter37] 在工作线程内执行所有股本 HTTP 查询，返回原始数据供主线程合并。

    返回 dict：
      ts_series  — list[(statDate_str, float)] 总股本时序（空列表=失败）
      ls_series  — list[(statDate_str, float)] 流通股本时序
      ts_static  — float|None  静态总股本（路径3兜底）
      ls_static  — float|None  静态流通股本
      source     — str  数据来源描述
    """
    result = {"ts_series": [], "ls_series": [],
              "ts_static": None, "ls_static": None, "source": "none"}

    # ─── 路径1：adata.stock.capital（优先，与 adata 同节点）───────────────
    try:
        import adata
        capital_ns = getattr(adata.stock, "capital", None)
        if capital_ns is not None:
            for fn_name in ["get_capital", "capital_change", "get_share",
                            "share_change", "get_history_capital", "stock_capital"]:
                fn = getattr(capital_ns, fn_name, None)
                if not callable(fn):
                    continue
                try:
                    df_cap = fn(stock_code=code_6)
                    if df_cap is None or df_cap.empty:
                        continue
                    dcol   = next((c for c in ["trade_date", "date", "report_date",
                                               "end_date", "change_date"]
                                   if c in df_cap.columns), None)
                    ts_col = next((c for c in ["total_share", "total_capital",
                                               "total_equity", "总股本"]
                                   if c in df_cap.columns), None)
                    ls_col = next((c for c in ["float_share", "circulating_share",
                                               "流通股本", "流通A股", "liq_share"]
                                   if c in df_cap.columns), None)
                    if dcol and ts_col:
                        import pandas as _pd
                        dc = df_cap.copy()
                        dc["_sd"] = (
                            _pd.to_datetime(dc[dcol], errors="coerce")
                            .dt.to_period("Q").dt.end_time.dt.strftime("%Y-%m-%d")
                        )
                        dc = dc.dropna(subset=["_sd"])
                        ts_vals = _pd.to_numeric(dc[ts_col], errors="coerce")
                        med = ts_vals.dropna().median()
                        if _pd.notna(med) and 0 < med < 1000:  # 亿股→万股
                            ts_vals = ts_vals * 10000.0
                        result["ts_series"] = list(zip(dc["_sd"], ts_vals))
                        if ls_col:
                            ls_vals = _pd.to_numeric(dc[ls_col], errors="coerce")
                            med2 = ls_vals.dropna().median()
                            if _pd.notna(med2) and 0 < med2 < 1000:
                                ls_vals = ls_vals * 10000.0
                            result["ls_series"] = list(zip(dc["_sd"], ls_vals))
                        result["source"] = f"adata.capital.{fn_name}"
                        return result
                except Exception:
                    continue
    except Exception:
        pass

    # ─── 路径2：AKShare stock_financial_abstract_ths ───────────────────────
    try:
        import akshare as ak
        import pandas as _pd
        ak_df = None
        for kwargs in [
            {"symbol": code_6, "indicator": "按报告期"},
            {"symbol": code_6},
            {"symbol": code_6, "indicator": "按年度"},
        ]:
            try:
                tmp = ak.stock_financial_abstract_ths(**kwargs)
                if tmp is not None and not tmp.empty:
                    ak_df = tmp; break
            except Exception:
                continue

        if ak_df is not None and not ak_df.empty:
            date_col = next((c for c in ["报告期", "report_date", "date", "statDate"]
                             if c in ak_df.columns), None)
            if date_col:
                ak_df = ak_df.copy()
                ak_df["_sd"] = (_pd.to_datetime(ak_df[date_col], errors="coerce")
                                .dt.strftime("%Y-%m-%d"))
                ak_df = ak_df.dropna(subset=["_sd"])

                ts_col = next((c for c in ["总股本", "总股本(万股)", "总股本(亿股)",
                                           "total_share", "totalShare"]
                               if c in ak_df.columns), None)
                ls_col = next((c for c in ["流通股本", "流通A股", "流通股本(万股)",
                                           "流通A股(万股)", "流通股本(亿股)",
                                           "float_share", "liqaShare"]
                               if c in ak_df.columns), None)
                if ts_col:
                    ts_vals = _pd.to_numeric(ak_df[ts_col], errors="coerce")
                    med = ts_vals.dropna().median()
                    if _pd.notna(med) and 0 < med < 1000:
                        ts_vals = ts_vals * 10000.0
                    result["ts_series"] = list(zip(ak_df["_sd"], ts_vals))
                if ls_col:
                    ls_vals = _pd.to_numeric(ak_df[ls_col], errors="coerce")
                    med2 = ls_vals.dropna().median()
                    if _pd.notna(med2) and 0 < med2 < 1000:
                        ls_vals = ls_vals * 10000.0
                    result["ls_series"] = list(zip(ak_df["_sd"], ls_vals))
                if result["ts_series"]:
                    result["source"] = "akshare.ths"
                    return result

        # ─── 路径3：东方财富个股快照 ───────────────────────────────────────
        try:
            info = ak.stock_individual_info_em(symbol=code_6)
            if info is not None and not info.empty and info.shape[1] >= 2:
                info_dict = dict(zip(info.iloc[:, 0].astype(str),
                                     info.iloc[:, 1].astype(str)))
                for k, v in info_dict.items():
                    v_s = str(v).replace(",", "").strip()
                    if "总股本" in k:
                        try:
                            val = float(v_s.replace("亿", ""))
                            if "亿" in v_s:
                                val *= 10000.0
                            result["ts_static"] = val
                        except Exception:
                            pass
                    if ("流通" in k and "股" in k) or "流通A" in k:
                        try:
                            val = float(v_s.replace("亿", ""))
                            if "亿" in v_s:
                                val *= 10000.0
                            result["ls_static"] = val
                        except Exception:
                            pass
                if result["ts_static"] is not None:
                    result["source"] = "akshare.em_snapshot"
        except Exception:
            pass

    except ImportError:
        pass

    return result


def _process_raw(res: dict, start_year: int) -> dict:
    """
    [BUG-B-FIX] 在主线程串行执行 pandas 标准化 + 写 CSV。
    [SPEED-FIX iter37] 股本合并改为纯 pandas 操作（HTTP 已在线程完成）。

    pandas 的 copy/to_datetime/sort_values/rolling/shift 等操作均在 Python 层
    持有 GIL，多线程并发执行只会导致 GIL 竞争，无法真正并行。
    主线程串行处理：零 GIL 竞争，CPU 占用降至正常水平（20-40%）。

    [BUG-SHARE-FIX] 使用 _fetch_raw 已取回的 shares_raw 合并股本数据，
    不再在此处发起任何 HTTP 请求（HTTP 全部在工作线程完成）。
    """
    global _consecutive_failures
    code    = res["code"]
    code_6  = res["code_6"]
    df_raw  = res["df_raw"]
    shares_raw = res.get("shares_raw")
    out_path = FUND_DIR / f"fundamental_adata_{code_6}.csv"

    try:
        df = _normalize_fundamental(df_raw, code_6)
        if df is None:
            return {"code": code, "status": "empty", "rows": 0, "msg": "标准化后为空"}

        # [SPEED-FIX iter37] 纯 pandas 合并（无 HTTP）
        df = _merge_shares_raw(df, shares_raw)

        if "statDate" in df.columns:
            df = df[df["statDate"] >= f"{start_year}-01-01"]

        if len(df) == 0:
            return {"code": code, "status": "empty", "rows": 0,
                    "msg": f"无 {start_year} 年后数据"}

        df.to_csv(str(out_path), index=False, encoding="utf-8-sig")
        with _lock:
            _consecutive_failures = max(0, _consecutive_failures - 1)
        return {"code": code, "status": "ok", "rows": len(df), "msg": ""}

    except Exception as e:
        return {"code": code, "status": "error", "rows": 0, "msg": str(e)}


def _merge_shares_raw(df: pd.DataFrame, shares_raw: dict | None) -> pd.DataFrame:
    """
    [SPEED-FIX iter37] 将 _fetch_shares_raw 返回的股本数据合并进 df（纯 pandas，无 HTTP）。
    """
    if shares_raw is None:
        return df

    ts_valid = df["totalShare"].notna().any() if "totalShare" in df.columns else False
    ls_valid = df["liqaShare"].notna().any() if "liqaShare" in df.columns else False
    if ts_valid and ls_valid:
        return df

    ts_series = shares_raw.get("ts_series", [])
    ls_series = shares_raw.get("ls_series", [])
    ts_static = shares_raw.get("ts_static")
    ls_static = shares_raw.get("ls_static")

    if "statDate" not in df.columns:
        return df

    # ── 时序合并（左连接到 statDate）────────────────────────────────────────
    if not ts_valid and ts_series:
        ts_df = pd.DataFrame(ts_series, columns=["statDate", "totalShare_raw"])
        ts_df = ts_df.dropna(subset=["statDate"]).drop_duplicates("statDate")
        ts_df["totalShare_raw"] = pd.to_numeric(ts_df["totalShare_raw"], errors="coerce")
        df = df.merge(ts_df, on="statDate", how="left")
        df["totalShare"] = df.pop("totalShare_raw")

    if not ls_valid and ls_series:
        ls_df = pd.DataFrame(ls_series, columns=["statDate", "liqaShare_raw"])
        ls_df = ls_df.dropna(subset=["statDate"]).drop_duplicates("statDate")
        ls_df["liqaShare_raw"] = pd.to_numeric(ls_df["liqaShare_raw"], errors="coerce")
        df = df.merge(ls_df, on="statDate", how="left")
        df["liqaShare"] = df.pop("liqaShare_raw")

    # ── 静态兜底（广播到所有行）───────────────────────────────────────────────
    if not ts_valid and ts_static is not None and (
            "totalShare" not in df.columns or df["totalShare"].isna().all()):
        df["totalShare"] = float(ts_static)
    if not ls_valid and ls_static is not None and (
            "liqaShare" not in df.columns or df["liqaShare"].isna().all()):
        df["liqaShare"] = float(ls_static)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 合并 CSV
# ─────────────────────────────────────────────────────────────────────────────

def merge_csvs() -> bool:
    """合并所有 fundamental_adata_*.csv 为 fundamental_merged.csv。"""
    files = sorted(FUND_DIR.glob("fundamental_adata_*.csv"))
    if not files:
        print("  ✗ 无 adata 基本面 CSV 文件")
        return False

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(str(f), dtype=str)
            if len(df) > 0:
                dfs.append(df)
        except Exception:
            pass

    if not dfs:
        return False

    merged = pd.concat(dfs, ignore_index=True)
    out = FUND_DIR / "fundamental_merged.csv"
    merged.to_csv(str(out), index=False, encoding="utf-8-sig")
    print(f"  ✓ 合并完成: {len(merged)} 行  来自 {len(dfs)} 只股票 → {out}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 批量下载
# ─────────────────────────────────────────────────────────────────────────────

def run(codes: list[str], n_workers: int, delay: float, start_year: int) -> None:
    """
    [BUG-B-FIX] 架构重构：滑动窗口提交 + 线程/主线程职责分离

    原版问题：
      1. 一次性提交全部 5189 个任务（字典推导式），16 线程立刻全部满载
      2. 线程内执行 _normalize_fundamental（~14 组 pandas 操作），
         16 线程同时持有 GIL → CPU 100%，真实吞吐率反而更低

    修复架构：
      · 线程（_fetch_raw）：只做 HTTP I/O，返回原始 df_raw，不持有 GIL
      · 主线程（_process_raw）：串行做 pandas 标准化 + 写 CSV，零 GIL 竞争
      · 滑动窗口（WINDOW = n_workers×2）：同时在飞的 future 上限，
        处理完一个立即补充一个，避免 5189 个 future 全驻留内存
    """
    from collections import deque
    from concurrent.futures import wait, FIRST_COMPLETED

    FUND_DIR.mkdir(parents=True, exist_ok=True)
    total = len(codes)
    t0    = time.time()

    print(f"\n  ▶ adata 基本面下载（{n_workers} 线程，delay={delay}s，共 {total} 只）")
    print(f"    [BUG-B-FIX] 滑动窗口模式：线程只做 HTTP，pandas 处理在主线程串行执行")

    ok_n = err_n = empty_n = skip_n = 0
    done = 0

    # 滑动窗口：最多同时在飞 n_workers×2 个 future，处理完一个补充一个
    WINDOW  = n_workers * 2
    pending = deque(codes)
    inflight: dict = {}   # {future: code}

    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        # 初始填满窗口
        while pending and len(inflight) < WINDOW:
            c = pending.popleft()
            inflight[executor.submit(_fetch_raw, c, delay)] = c

        while inflight:
            done_set, _ = wait(inflight, return_when=FIRST_COMPLETED)

            for future in done_set:
                c = inflight.pop(future)
                done += 1

                try:
                    fetch_res = future.result()
                except Exception as e:
                    fetch_res = {"code": c, "code_6": _to_6digit(c),
                                 "status": "error", "df_raw": None, "msg": str(e)}
                del future   # 立即释放，避免 df_raw 长期驻留

                st = fetch_res["status"]

                if st == "skip":
                    res = {"code": c, "status": "skip", "rows": 0, "msg": "已存在"}
                elif st == "fetched":
                    # [BUG-B-FIX] pandas 标准化在主线程串行执行，不占用线程的 GIL
                    res = _process_raw(fetch_res, start_year)
                    del fetch_res   # 释放 df_raw 内存
                else:
                    res = {"code": c, "status": st, "rows": 0, "msg": fetch_res["msg"]}

                final_st = res["status"]
                if final_st == "ok":     ok_n    += 1
                elif final_st == "skip": skip_n  += 1
                elif final_st == "error":err_n   += 1
                else:                    empty_n += 1

                elapsed = time.time() - t0
                speed   = done / max(elapsed, 0.1)
                eta     = (total - done) / max(speed, 0.01)
                print(
                    f"\r  [{done:4d}/{total}] {c}  {final_st:5s}  {res['rows']:4d}行 | "
                    f"ok={ok_n} skip={skip_n} err={err_n} | "
                    f"{speed:.1f}只/s  ETA={eta:.0f}s",
                    end="", flush=True,
                )
                if final_st == "error":
                    print(f"\n    ✗ {res['msg'][:80]}")

                # 补充新任务进窗口
                while pending and len(inflight) < WINDOW:
                    nc = pending.popleft()
                    inflight[executor.submit(_fetch_raw, nc, delay)] = nc

    print()
    print("=" * 68)
    print(f"  完成: ok={ok_n}  skip={skip_n}  empty={empty_n}  error={err_n}")
    print(f"  耗时: {time.time()-t0:.0f}s")
    print("=" * 68)


# ─────────────────────────────────────────────────────────────────────────────
# [BUG-ADATA-02-FIX] 从 daily_parquet 提取估值 → valuation_daily.csv
# ─────────────────────────────────────────────────────────────────────────────

def extract_valuation_from_parquets(
    parquet_dir: Optional[Path] = None,
    out_path: Optional[Path] = None,
) -> bool:
    """
    [BUG-ADATA-02-FIX] step1b 从不生成 valuation_daily.csv，
    导致 step3._build_valuation_npy() 跳过 pe_ttm/pb_mrq/ps_ttm 构建。

    本函数从 step0c 已下载的 daily_parquet/*.parquet 中提取估值字段，
    生成 valuation_daily.csv，格式与 step1_akshare 完全兼容：
      列：code, date, peTTM, pbMRQ, psTTM

    adata.stock.market.get_market() 返回的 pe_ttm / pb 字段由 step0c 存入
    parquet，字段名为 pe_ttm / pb_mrq / ps_ttm（见 step0c `_download_one` 第 534 行）。

    Parameters
    ----------
    parquet_dir : daily_parquet 目录，默认 data/daily_parquet
    out_path    : 输出路径，默认 data/fundamental/valuation_daily.csv
    """
    if parquet_dir is None:
        # 尝试从 config.json 读取
        cfg = PROJECT_ROOT / "config.json"
        if cfg.exists():
            try:
                import json as _json
                with open(cfg, encoding="utf-8") as f:
                    c = _json.load(f)
                raw = c.get("parquet_dir") or c.get("data", {}).get("parquet_dir", "")
                if raw:
                    p = Path(raw)
                    parquet_dir = p if p.is_absolute() else PROJECT_ROOT / p
            except Exception:
                pass
    if parquet_dir is None:
        parquet_dir = _gpq("qfq") if "_gpq" in dir() else (DATA_DIR / "daily_parquet_qfq")

    if out_path is None:
        out_path = FUND_DIR / "valuation_daily.csv"

    if not parquet_dir.exists():
        print(f"  ✗ parquet_dir 不存在: {parquet_dir}")
        print("    请先运行 step0c_download_daily_adata.py 下载日线数据")
        return False

    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"  ✗ {parquet_dir} 中无 parquet 文件")
        return False

    # pe/pb/ps 列的可能名称（step0c 写入时用的标准名）
    PE_CANDS  = ["pe_ttm",  "pe"]
    PB_CANDS  = ["pb_mrq",  "pb"]
    PS_CANDS  = ["ps_ttm",  "ps"]

    print(f"\n  [BUG-ADATA-02-FIX] 从 {len(parquet_files)} 只 parquet 提取估值字段...")
    FUND_DIR.mkdir(parents=True, exist_ok=True)

    chunks = []
    skipped = 0
    for pf in parquet_files:
        try:
            df = pd.read_parquet(str(pf))
            if df.empty:
                skipped += 1
                continue

            # 提取 code（去掉交易所前缀）
            code_6 = pf.stem.split(".")[-1].zfill(6)

            # 日期列
            date_col = next((c for c in ["date", "trade_date"] if c in df.columns), None)
            if date_col is None:
                skipped += 1
                continue

            result = pd.DataFrame()
            result["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
            result["code"] = code_6

            has_val = False
            for target, cands in [("peTTM", PE_CANDS), ("pbMRQ", PB_CANDS), ("psTTM", PS_CANDS)]:
                for c in cands:
                    if c in df.columns:
                        result[target] = pd.to_numeric(df[c], errors="coerce")
                        has_val = True
                        break
                else:
                    result[target] = np.nan

            if not has_val:
                # 该 parquet 无任何估值字段（正常：adata 2.9.5 无此字段）
                skipped += 1
                continue

            # 只保留至少有 peTTM 的行
            result = result.dropna(subset=["peTTM"])
            if len(result) > 0:
                chunks.append(result)

        except Exception:
            skipped += 1
            continue

    if not chunks:
        print(f"  ✗ 所有 parquet 均无 pe_ttm/pb/ps_ttm 字段")
        print(f"    → adata 2.9.5 get_market() 不返回估值字段（confirmed）")
        print(f"    → 估值数据需通过 AKShare stock_a_lg_indicator 单独下载")
        print(f"    → 建议运行: python scripts/step1_download_fundamental_akshare.py --valuation-only")
        return False

    merged = pd.concat(chunks, ignore_index=True)
    merged = merged.sort_values(["code", "date"]).reset_index(drop=True)
    merged.to_csv(str(out_path), index=False, encoding="utf-8-sig")

    n_stocks = merged["code"].nunique()
    print(f"  ✓ valuation_daily.csv: {n_stocks} 只股票, {len(merged)} 条记录 → {out_path}")
    if skipped > 0:
        print(f"    （跳过 {skipped} 只 parquet，无估值字段或文件异常）")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 1b: adata 季度基本面下载（独立节点，终极兜底）")
    parser.add_argument("--workers",    type=int,   default=6,
                        help="并发线程数（默认6）")
    parser.add_argument("--delay",      type=float, default=0.6,
                        help="每只间隔秒（默认0.6）")
    parser.add_argument("--start-year", type=int,   default=2018,
                        help="起始年份（默认2018）")
    parser.add_argument("--test",       action="store_true",
                        help="测试模式（5只）")
    parser.add_argument("--codes",      nargs="+",  default=None,
                        help="指定股票代码")
    parser.add_argument("--merge-only", action="store_true",
                        help="仅合并已有 CSV，不重新下载")
    parser.add_argument("--probe",      action="store_true",
                        help="仅探测可用接口并退出")
    # [BUG-ADATA-02-FIX] 新增：从 daily_parquet 提取估值
    parser.add_argument("--extract-valuation", action="store_true",
                        help="[BUG-ADATA-02-FIX] 从 daily_parquet 提取 PE/PB/PS → valuation_daily.csv")
    # [BUG-SHARE-FIX] 新增：仅对已有 CSV 补充 totalShare/liqaShare，不重新下载 adata
    parser.add_argument("--supplement-shares-only", action="store_true",
                        help="[BUG-SHARE-FIX] 对已有 fundamental_adata_*.csv 补充 totalShare/liqaShare，"
                             "不重新下载 adata 数据。已下载数据直接用此选项修复即可。")
    args = parser.parse_args()

    # 依赖检查
    try:
        import adata
        print(f"✓ adata 版本: {getattr(adata, '__version__', 'unknown')}")
    except ImportError:
        print("✗ adata 未安装：pip install adata"); sys.exit(1)

    if args.probe or args.test:
        print("\n[探测] adata.stock.finance 可用接口...")
        try:
            available = [x for x in dir(adata.stock.finance) if not x.startswith("_")]
            print(f"  全部接口: {available}")
        except Exception as e:
            print(f"  探测失败: {e}")

    if args.probe:
        fn_name, fn = _get_finance_fn()
        print(f"\n→ 可用基本面接口: {fn_name or '未找到'}")
        sys.exit(0)

    # [BUG-ADATA-02-FIX] --extract-valuation 独立模式
    if args.extract_valuation:
        ok = extract_valuation_from_parquets()
        sys.exit(0 if ok else 1)

    if args.merge_only:
        merge_csvs(); sys.exit(0)

    # [BUG-SHARE-FIX] --supplement-shares-only：对已有 CSV 打补丁，不重新下载
    if args.supplement_shares_only:
        print("\n" + "=" * 68)
        print("  [SPEED-FIX iter37] 多线程补充股本：对已有 CSV 追加 totalShare / liqaShare")
        print("  数据来源：adata.capital → AKShare ths → 东财快照（三级路径）")
        print("  说明：adata get_core_index 不含股本字段，此操作无需重新下载 adata 数据")
        print("=" * 68)

        csv_files = sorted(FUND_DIR.glob("fundamental_adata_*.csv"))
        if not csv_files:
            print(f"✗ {FUND_DIR} 中无 fundamental_adata_*.csv 文件")
            print("  请先完整运行 step1b 下载 adata 数据，再使用 --supplement-shares-only")
            sys.exit(1)

        # 过滤出需要补充的文件（totalShare 全 NaN 或不存在）
        need_patch = []
        for csv_path in csv_files:
            try:
                df_check = pd.read_csv(str(csv_path), dtype=str)
                if df_check.empty:
                    continue
                if "totalShare" in df_check.columns:
                    ts_valid = pd.to_numeric(df_check["totalShare"], errors="coerce").notna().any()
                    ls_valid = pd.to_numeric(df_check.get("liqaShare", pd.Series()), errors="coerce").notna().any()
                    if ts_valid and ls_valid:
                        continue  # 已有完整股本，跳过
                need_patch.append(csv_path)
            except Exception:
                pass

        print(f"\n  需要补充: {len(need_patch)}/{len(csv_files)} 只（其余已有股本数据）")
        if not need_patch:
            print("  全部已有股本数据，无需补充！")
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            t0 = time.time()
            patched = skipped = failed = 0
            total_patch = len(need_patch)
            done_count = 0

            def _patch_one(csv_path):
                code_6 = csv_path.stem.replace("fundamental_adata_", "")
                try:
                    df = pd.read_csv(str(csv_path), dtype=str)
                    if df.empty:
                        return "skip"
                    # HTTP 在线程内执行
                    shares_raw = _fetch_shares_raw(code_6)
                    for col in df.columns:
                        if col not in ("code", "statDate", "pubDate"):
                            df[col] = pd.to_numeric(df[col], errors="coerce")
                    df = _merge_shares_raw(df, shares_raw)
                    ts_after = pd.to_numeric(df.get("totalShare", pd.Series(dtype=float)),
                                             errors="coerce").notna().any()
                    if ts_after:
                        df.to_csv(str(csv_path), index=False, encoding="utf-8-sig")
                        return "patched"
                    return "failed"
                except Exception as e:
                    return f"error:{e}"

            n_sup_workers = min(args.workers, 16)
            print(f"  使用 {n_sup_workers} 线程并行补充...")
            with ThreadPoolExecutor(max_workers=n_sup_workers) as executor:
                fmap = {executor.submit(_patch_one, p): p for p in need_patch}
                for future in as_completed(fmap):
                    done_count += 1
                    r = future.result()
                    if r == "patched":   patched += 1
                    elif r == "skip":    skipped += 1
                    else:                failed  += 1
                    elapsed = time.time() - t0
                    speed = done_count / max(elapsed, 0.1)
                    eta = (total_patch - done_count) / max(speed, 0.01)
                    print(f"\r  [{done_count:4d}/{total_patch}] patched={patched} "
                          f"skip={skipped} fail={failed} | {speed:.1f}只/s  ETA={eta:.0f}s",
                          end="", flush=True)

            print(f"\n\n  ✓ 完成: patched={patched} skip={skipped} fail={failed}  "
                  f"耗时={time.time()-t0:.0f}s")
            if patched > 0:
                print("\n  下一步：重建矩阵")
                print("    python scripts/step3_build_fundamental_npy.py")

        sys.exit(0)

    # 加载股票列表
    if args.test:
        codes = [_to_full_code(c) for c in
                 ["600519", "000001", "300750", "601318", "000858"]]
    elif args.codes:
        codes = [_to_full_code(c) for c in args.codes]
    else:
        codes = _load_codes()

    print(f"✓ 目标: {len(codes)} 只  起始年份: {args.start_year}")

    # 探测接口（首次调用时自动执行，这里提前触发以便用户看到结果）
    fn_name, fn = _get_finance_fn()
    if fn is None:
        print("\n✗ 无可用基本面接口，请升级 adata 或改用 step1a（AKShare）")
        sys.exit(1)

    run(codes=codes, n_workers=args.workers,
        delay=args.delay, start_year=args.start_year)

    print("\n  合并 CSV...")
    merge_csvs()

    # [BUG-ADATA-02-FIX] 自动尝试从 daily_parquet 提取估值
    print("\n  [BUG-ADATA-02-FIX] 尝试从 daily_parquet 提取估值数据...")
    extract_valuation_from_parquets()

    print("\n✓ Step 1b 完成！后续：")
    print("  python scripts/step3_build_fundamental_npy.py")
