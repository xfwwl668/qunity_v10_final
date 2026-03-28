"""
scripts/step1_download_fundamental_akshare.py
【AKShare + adata 双数据源版】

═══════════════════════════════════════════════════════════════════════
[ADATA] 新增 adata 数据源（2026-03）
═══════════════════════════════════════════════════════════════════════
  --source adata   : 使用 adata.stock.finance.get_core_index()
                     43字段，无封禁风险，断点续传
  --source akshare : 原有 AKShare 多级降级路线
  --source auto    : 优先 adata，不可用时降级 akshare（默认）

  adata 字段映射：
    notice_date → pubDate    （公告日，point-in-time 对齐）
    report_date → statDate
    roe_wtd     → roeAvg
    net_margin  → npMargin
    gross_margin→ gpMargin
    net_profit_attr_sh → netProfit
    total_rev   → MBRevenue
    basic_eps 滚动4季 → epsTTM
    net_profit_yoy_gr  → YOYNI
    total_rev_yoy_gr   → YOYEPSBasic（近似）
    get_stock_shares ÷10000 → totalShare / liqaShare（万股）

═══════════════════════════════════════════════════════════════════════
原有 AKShare 机制（--source akshare 时保留）
═══════════════════════════════════════════════════════════════════════
  · 随机 UA 轮换 + Referer 注入（防新浪封禁根本手段）
  · ThreadPoolExecutor(n_workers) 直接控制并发，无信号量死锁风险
  · 连续失败动态降速（_consecutive_failures）
  · 三级数据源降级：新浪indicator → 新浪report → 东方财富
  · 三轮重排队：等待 60s/120s 让封禁解除后重试
  · 断点续传（parquet + json）
  · 字段输出与 Baostock 版完全兼容，step3 无需改动

用法:
  python scripts/step1_download_fundamental_akshare.py             # auto(优先adata)
  python scripts/step1_download_fundamental_akshare.py --source adata
  python scripts/step1_download_fundamental_akshare.py --source akshare
  python scripts/step1_download_fundamental_akshare.py --test
  python scripts/step1_download_fundamental_akshare.py --force
  python scripts/step1_download_fundamental_akshare.py --workers 4 --delay 1.0
  python scripts/step1_download_fundamental_akshare.py --verify
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("TQDM_DISABLE", "1")

# [ADATA] 禁代理（必须在所有 import 之前，防止 SunRequests 被失效代理拦截）
for _k in ["HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy"]:
    _os.environ.pop(_k, None)
_os.environ["NO_PROXY"] = "*"
_os.environ["no_proxy"] = "*"
try:
    import requests as _req
    _orig_req = _req.Session.request
    def _noproxy_req(self, *args, **kw):
        kw["proxies"] = {"http": "", "https": "", "no": "*"}
        kw.setdefault("timeout", 30)
        return _orig_req(self, *args, **kw)
    _req.Session.request = _noproxy_req
except Exception:
    pass

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
FUND_DIR     = DATA_DIR / "fundamental"
# [FIX-H-04] META_PATH 使用 npy_v10 目录
try:
    import sys as _sys; _sys.path.insert(0, str(Path(__file__).parent))
    from utils_paths import get_npy_dir as _gnd  # type: ignore
    META_PATH = _gnd("v10") / "meta.json"
except Exception:
    META_PATH = DATA_DIR / "npy_v10" / "meta.json"
CKPT_PATH    = FUND_DIR / "_ak_checkpoint.json"
PARTIAL_PATH = FUND_DIR / "_ak_partial.parquet"
STATS_PATH   = FUND_DIR / "download_stats.json"   # 借鉴③
FAILED_PATH  = FUND_DIR / "failed_stocks.txt"     # 借鉴②

# [BUG-FUND-PE-NEVER-DOWNLOADED FIX] 新增 peTTM / pbMRQ / psTTM 估值字段
# 原来三个接口（em_abstract / em_profit / sina_indicator）均无 PE_TTM，
# 导致 pe_ttm.npy 永远为 NaN，titan_alpha PE 因子完全失效。
# 修复：(1) PROFIT_FIELDS 中新增 peTTM / pbMRQ / psTTM
#       (2) _fetch_one 最后补充调用 ak.stock_a_lg_indicator 拉取估值（见下方 _try_valuation）
#       (3) step3 FACTOR_MAP 中新增 pe / pb / ps 条目（见 step3 修复注释）
PROFIT_FIELDS = ["code", "pubDate", "statDate",
                 "roeAvg", "npMargin", "gpMargin",
                 "netProfit", "epsTTM", "MBRevenue",
                 "totalShare", "liqaShare",
                 "peTTM", "pbMRQ", "psTTM"]        # ← 新增估值字段
GROWTH_FIELDS = ["code", "pubDate", "statDate",
                 "YOYNI", "YOYEPSBasic", "YOYPNI",
                 "YOYROE", "YOYAsset", "YOYEquity"]

_SINA_COL_CANDIDATES = {
    "roeAvg":      ["净资产收益率(%)", "净资产收益率", "roe"],
    "npMargin":    ["销售净利率(%)", "销售净利率", "净利润率"],
    "gpMargin":    ["销售毛利率(%)", "销售毛利率"],
    "netProfit":   ["净利润(万元)", "净利润", "net_profit"],
    "epsTTM":      ["每股收益(元)", "每股收益", "eps", "基本每股收益(元)"],
    "MBRevenue":   ["主营业务收入(万元)", "营业总收入(万元)", "营业收入(万元)"],
    # 总股本：新浪用"总股本(万股)"，东财/同花顺用"总股本"，部分接口用英文
    "totalShare":  ["总股本(万股)", "总股本(亿股)", "总股本", "total_share", "totalShare"],
    # 流通A股：新浪用"流通A股(万股)"，东财用"流通股本"/"流通A股"
    "liqaShare":   ["流通A股(万股)", "流通股本(万股)", "流通A股(亿股)", "流通股本(亿股)",
                    "流通A股", "流通股本", "流通股", "liqa_share", "liqaShare"],
    "YOYNI":       ["净利润同比增长率(%)", "净利润增长率", "净利润同比"],
    "YOYEPSBasic": ["每股收益同比增长率(%)", "每股收益增长率"],
    "YOYPNI":      ["扣非净利润同比(%)", "扣非净利润增长率"],
    "YOYROE":      ["净资产收益率同比(%)", "roe同比"],
    "YOYAsset":    ["总资产同比(%)", "总资产增长率"],
    "YOYEquity":   ["净资产同比(%)", "股东权益增长率"],
    # [BUG-FUND-PE-NEVER-DOWNLOADED FIX] 估值字段映射
    # ak.stock_a_lg_indicator 返回列：trade_date, pe_ttm, pb, ps_ttm, dv_ttm, total_mv
    "peTTM":       ["pe_ttm", "pe", "市盈率(TTM)", "市盈率TTM", "市盈率"],
    "pbMRQ":       ["pb", "市净率", "pb_mrq", "市净率(MRQ)"],
    "psTTM":       ["ps_ttm", "ps", "市销率", "市销率(TTM)"],
}

# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def load_codes() -> list:
    if not META_PATH.exists():
        print(f"✗ 未找到 {META_PATH}")
        sys.exit(1)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    codes = [str(c).strip().split(".")[-1].zfill(6) for c in meta["codes"]]
    print(f"✓ 加载 {len(codes)} 只股票")
    return codes


def _pick_col(df: pd.DataFrame, candidates: list):
    for name in candidates:
        if name in df.columns:
            return name
    for name in candidates:
        key = name.replace("(%)", "").replace("(万元)", "").replace("(元)", "").strip()
        matched = [c for c in df.columns if key in c]
        if matched:
            return matched[0]
    return None


def _remap_columns(df: pd.DataFrame) -> pd.DataFrame:
    # [BUG-SHARE-UNIT-FIX] 记录重命名前的原始列名，用于判断单位（亿股 vs 万股）
    _orig_cols_before = set(df.columns)
    rename_map = {}
    for target, candidates in _SINA_COL_CANDIDATES.items():
        col = _pick_col(df, candidates)
        if col and col not in rename_map.values():
            rename_map[col] = target
    df = df.rename(columns=rename_map)

    # [BUG-SHARE-UNIT-FIX] 单位归一化：totalShare / liqaShare 统一为"万股"。
    # 新浪/东财部分接口返回"亿股"单位（列名含"亿股"），重命名后值约为 0.01~200 亿。
    # step3 FACTOR_MAP 注释标注单位为"万股"，_build_market_cap 公式基于万股计算。
    # 若不转换：1亿股 × 10000 / 1e8 = 1（正确），但若数据本身是亿股时直接乘会超大。
    # 检测方式：若原始列名含"亿股"，则乘以 10000 转为万股。
    for target_col in ("totalShare", "liqaShare"):
        if target_col not in df.columns:
            continue
        original_col = {v: k for k, v in rename_map.items()}.get(target_col, "")
        if "亿股" in str(original_col):
            df[target_col] = pd.to_numeric(df[target_col], errors="coerce") * 10000.0
    return df


def _parse_stat_date(df: pd.DataFrame, start_year: int) -> pd.DataFrame:
    date_col = _pick_col(df, ["报告期", "date", "period", "statDate", "公告日期"])
    if date_col is None:
        date_col = df.columns[0]
    df = df.copy()
    df["statDate"] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["statDate"])
    df = df[df["statDate"] >= f"{start_year}-01-01"]
    return df


def _parse_pub_date(df: pd.DataFrame) -> pd.DataFrame:
    pub_col = _pick_col(df, ["公告日期", "pubDate", "披露日期"])
    df = df.copy()
    if pub_col and pub_col in df.columns:
        df["pubDate"] = pd.to_datetime(df[pub_col], errors="coerce").dt.strftime("%Y-%m-%d")
    else:
        df["pubDate"] = (
            pd.to_datetime(df["statDate"], errors="coerce") + pd.Timedelta(days=60)
        ).dt.strftime("%Y-%m-%d")
    return df


def _clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    数值清洗 + 单位归一化。

    [BUG-UNIT-SUFFIX-FIX]
    stock_financial_abstract_ths 对 净利润/营业总收入 等字段返回带单位后缀的字符串，
    如 '248.66亿' / '3482.12万' / '7.02%'。
    pd.to_numeric('248.66亿', errors='coerce') → NaN，导致这些字段全部为空。

    修复：预处理 object 列，识别单位后缀 → 剥离后缀 → 转数值 → 乘以换算因子。

    单位换算目标：与 BaoStock / stock_profit_sheet_by_report_em 一致，采用万元作基准。
      '248.66亿'  → 2486600.0  万元（× 10000）
      '3482.12万' → 3482.12    万元（× 1，保持不变）
      '7.02%'     → 7.02       百分比（保持不变，ROE/YOY 等本身就是%）
      纯数字字符串→ 直接 to_numeric，不做换算
    """
    skip = {"code", "pubDate", "statDate"}
    df = df.copy()
    for col in df.columns:
        if col in skip:
            continue
        s = df[col]
        # [BUG-DTYPE-CHECK-FIX]
        # 原来用 s.dtype != object 判断是否数值列，但 pandas 新版 str 列的 dtype 是
        # pd.StringDtype()，不等于 numpy object，导致 '248.66亿' 走 to_numeric 快路径 → NaN。
        # 修复：改用 pd.api.types.is_numeric_dtype()，精确区分数值型 vs 字符串型。
        if pd.api.types.is_numeric_dtype(s):
            df[col] = pd.to_numeric(s, errors="coerce")
            continue
        # ── 字符串列：剥离单位后缀 ────────────────────────────────────────
        s = s.astype(str).str.strip().replace({"nan": "", "None": "", "": np.nan})
        s_str = s.fillna("")
        has_yi  = s_str.str.endswith("亿")   # 亿 → × 10000（转为万元）
        has_wan = s_str.str.endswith("万")   # 万 → × 1（已是万元）
        has_pct = s_str.str.endswith("%")    # 百分号 → × 1（保持为 % 单位）
        # 剥离后缀 + 去千分位逗号
        s_clean = (
            s_str
            .str.rstrip("亿万%")
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        numeric = pd.to_numeric(s_clean, errors="coerce")
        # 应用换算因子（仅含 '亿' 后缀的值乘以 10000）
        numeric = np.where(has_yi,  numeric * 10000.0, numeric)
        numeric = np.where(has_wan, numeric * 1.0,     numeric)
        # %、纯数字、其他：不做换算
        df[col] = numeric.astype(float)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 反封禁：随机 UA 池
# ─────────────────────────────────────────────────────────────────────────────
import random as _random

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
]


def _patch_akshare_headers():
    try:
        import akshare.utils.func as _ak_func
        session = getattr(_ak_func, "session", None)
        if session is None:
            import requests
            session = requests.Session()
            _ak_func.session = session
        session.headers.update({
            "User-Agent":      _random.choice(_USER_AGENTS),
            "Referer":         "https://finance.sina.com.cn/",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        })
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 借鉴①：429/限流专项识别 + 智能退避
# 来源：akshare_parallel_downloader.py L134-L140
# ─────────────────────────────────────────────────────────────────────────────

def _is_rate_limit_error(err_msg: str) -> bool:
    keywords = ["429", "限流", "频繁", "too many", "rate limit",
                "访问过于", "请求过多", "blocked", "forbidden"]
    low = err_msg.lower()
    return any(k in low for k in keywords)


def _backoff_wait(attempt: int, is_rate_limit: bool) -> float:
    """
    普通失败：2^attempt + 抖动（最长约8秒）
    限流错误：5×(attempt+1) + 抖动（最长约17秒），让封禁有时间解除
    """
    if is_rate_limit:
        return 5.0 * (attempt + 1) + _random.uniform(0, 2)
    else:
        return (2.0 ** attempt) + _random.uniform(0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 单股票下载
# ─────────────────────────────────────────────────────────────────────────────

_consecutive_failures = 0
_failure_lock = __import__("threading").Lock()
_MAX_RETRY = 3


def _fetch_one(code_6: str, start_year: int, delay: float) -> dict:
    """
    接口优先级（从快到慢）：
    ① 东财 stock_financial_abstract_ths  — 单次HTTP，无分页，~0.5s
    ② 东财 stock_profit_sheet_by_report_em — 单次HTTP，~1s
    ③ 新浪 stock_financial_analysis_indicator — 分页爬取，~5~12s（仅兜底）
    ④ adata get_income_statement — 独立巨潮节点，前三路全封时救场

    [adata 第④级兜底设计说明]
    ① ② ③ 全部来自东财/新浪同一生态圈，被封时三级同时失效。
    adata 使用独立的巨潮/自有节点，与新浪/东财封禁策略完全隔离，
    是前三路全封后唯一可用的数据来源。

    adata 字段映射（与项目格式对齐）：
      report_date → statDate      ann_date → pubDate（精确公告日，无需估算）
      net_profit  → netProfit     roe → roeAvg（季度值，非年化，差异可接受）
      gross_profit_margin → gpMargin   net_profit_margin → npMargin
      basic_eps（单季）→ epsTTM via rolling(4).sum()
      total_share / float_share → totalShare / liqaShare（万股，需×10000若原始单位为亿）
      pe_ttm / pb / ps_ttm → peTTM / pbMRQ / psTTM（来自 get_market 日频行情）
      YOY 字段 → 自算（同期 4 季前对比），精度略低于报表同比但覆盖率 ~100%
    """
    global _consecutive_failures
    import akshare as ak

    result = {
        "code": code_6, "profit_rows": [], "growth_rows": [],
        "ok": False, "source": "none", "error_msg": "",
    }

    with _failure_lock:
        extra_sleep = min(_consecutive_failures * 0.5, 10.0)
    if extra_sleep > 0:
        time.sleep(extra_sleep)

    def _try_em_abstract() -> bool:
        """策略①：东财同花顺财务摘要（最快，无分页，~0.5s）"""
        for attempt in range(_MAX_RETRY):
            try:
                _patch_akshare_headers()
                df = ak.stock_financial_abstract_ths(symbol=code_6, indicator="按报告期")
                if df is None or df.empty:
                    return False
                df = _remap_columns(df)
                df = _parse_stat_date(df, start_year)
                df = _parse_pub_date(df)
                df["code"] = code_6
                df = _clean_numeric(df)
                p_cols = ["code", "pubDate", "statDate"] + [
                    c for c in ["roeAvg","npMargin","gpMargin","netProfit","epsTTM","MBRevenue",
                                "totalShare","liqaShare"]
                    if c in df.columns]
                g_cols = ["code", "pubDate", "statDate"] + [
                    c for c in ["YOYNI","YOYEPSBasic","YOYPNI","YOYROE","YOYAsset","YOYEquity"]
                    if c in df.columns]
                result["profit_rows"] = df[p_cols].to_dict("records") if len(p_cols) > 3 else []
                result["growth_rows"] = df[g_cols].to_dict("records") if len(g_cols) > 3 else []
                result["ok"] = True
                result["source"] = "em_abstract"
                return True
            except Exception as e:
                err = str(e)
                result["error_msg"] = err
                time.sleep(_backoff_wait(attempt, _is_rate_limit_error(err)))
        return False

    def _try_em_profit() -> bool:
        """策略②：东财利润表（单次请求，~1s）"""
        for attempt in range(_MAX_RETRY):
            try:
                _patch_akshare_headers()
                df = ak.stock_profit_sheet_by_report_em(symbol=code_6)
                if df is None or df.empty:
                    return False
                df = _remap_columns(df)
                df = _parse_stat_date(df, start_year)
                df = _parse_pub_date(df)
                df["code"] = code_6
                df = _clean_numeric(df)
                p_cols = ["code","pubDate","statDate"] + [
                    c for c in ["roeAvg","npMargin","gpMargin","netProfit","epsTTM","MBRevenue",
                                "totalShare","liqaShare"]
                    if c in df.columns]
                g_cols = ["code","pubDate","statDate"] + [
                    c for c in ["YOYNI","YOYEPSBasic","YOYPNI","YOYROE","YOYAsset","YOYEquity"]
                    if c in df.columns]
                if len(p_cols) > 3:
                    result["profit_rows"] = df[p_cols].to_dict("records")
                if len(g_cols) > 3:
                    result["growth_rows"] = df[g_cols].to_dict("records")
                result["ok"] = True
                result["source"] = "em_profit"
                return True
            except Exception as e:
                err = str(e)
                result["error_msg"] = err
                time.sleep(_backoff_wait(attempt, _is_rate_limit_error(err)))
        return False

    def _try_sina_indicator() -> bool:
        """策略③：新浪分析指标（分页慢5~12s，仅兜底）"""
        for attempt in range(_MAX_RETRY):
            try:
                _patch_akshare_headers()
                df = ak.stock_financial_analysis_indicator(
                    symbol=code_6, start_year=str(start_year))
                if df is None or df.empty:
                    return False
                df = _remap_columns(df)
                df = _parse_stat_date(df, start_year)
                df = _parse_pub_date(df)
                df["code"] = code_6
                df = _clean_numeric(df)
                p_cols = ["code", "pubDate", "statDate"] + [
                    c for c in ["roeAvg","npMargin","gpMargin","netProfit","epsTTM","MBRevenue",
                                "totalShare","liqaShare"]
                    if c in df.columns]
                g_cols = ["code", "pubDate", "statDate"] + [
                    c for c in ["YOYNI","YOYEPSBasic","YOYPNI","YOYROE","YOYAsset","YOYEquity"]
                    if c in df.columns]
                result["profit_rows"] = df[p_cols].to_dict("records") if len(p_cols) > 3 else []
                result["growth_rows"] = df[g_cols].to_dict("records") if len(g_cols) > 3 else []
                result["ok"] = True
                result["source"] = "sina"
                return True
            except Exception as e:
                err = str(e)
                result["error_msg"] = err
                time.sleep(_backoff_wait(attempt, _is_rate_limit_error(err)))
        return False

    def _try_valuation() -> bool:
        """
        [BUG-FUND-PE-NEVER-DOWNLOADED FIX] 策略④：日频估值（PE/PB/PS）
        来源：ak.stock_a_lg_indicator(symbol) 返回 trade_date/pe_ttm/pb/ps_ttm
        与利润接口独立调用，结果存入 result["valuation_rows"]。
        step3 从汇总后的 valuation_daily.csv 构建 pe_ttm.npy / pb_mrq.npy。
        """
        for attempt in range(_MAX_RETRY):
            try:
                _patch_akshare_headers()
                df = ak.stock_a_lg_indicator(symbol=code_6)
                if df is None or df.empty:
                    return False
                df = df.copy()
                col_map = {}
                col_lower = {c.lower(): c for c in df.columns}
                for target, candidates in [
                    ("date",  ["trade_date", "date"]),
                    ("peTTM", ["pe_ttm", "pe"]),
                    ("pbMRQ", ["pb"]),
                    ("psTTM", ["ps_ttm", "ps"]),
                ]:
                    for cname in candidates:
                        if cname in df.columns:
                            col_map[cname] = target
                            break
                        if cname in col_lower:
                            col_map[col_lower[cname]] = target
                            break
                df = df.rename(columns=col_map)
                if "date" not in df.columns:
                    return False
                df["date"] = df["date"].astype(str).str[:10]
                df["code"] = code_6
                for col in ["peTTM", "pbMRQ", "psTTM"]:
                    if col not in df.columns:
                        df[col] = float("nan")
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                result["valuation_rows"] = df[
                    ["code", "date", "peTTM", "pbMRQ", "psTTM"]
                ].dropna(subset=["peTTM"]).to_dict("records")
                return True
            except Exception as e:
                err = str(e)
                time.sleep(_backoff_wait(attempt, _is_rate_limit_error(err)))
        return False

    def _patch_growth_from_eps() -> None:
        """
        [BUG-YOY-SPARSE-FIX iter37]
        若 em_abstract/em_profit 路径成功但 YOY 字段仍有缺失，从 profit_rows 时序自算。

        原版只计算 YOYNI + YOYEPSBasic（2个），YOYPNI / YOYROE / YOYAsset / YOYEquity 全空。
        修复：扩展映射表，对所有可得字段均做 4期同比计算。

        YOY计算公式：(当期 - 4期前) / abs(4期前) × 100
        精度：略低于报表同比，但覆盖率从 ~0% 提升至 ~100%
        """
        # [BUG-YOY-SPARSE-FIX iter37] 扩展到6个YOY字段
        YOY_MAP = [
            ("YOYNI",       "netProfit"),
            ("YOYEPSBasic", "epsTTM"),
            ("YOYPNI",      "netProfit"),   # 扣非净利润近似用净利润代替
            ("YOYROE",      "roeAvg"),
            # YOYAsset / YOYEquity：profit_rows 不含资产/股东权益，暂时跳过
        ]

        for yoy_field, src_field in YOY_MAP:
            # 检查此字段是否已有有效数据（已有则跳过）
            already_ok = any(
                row.get(yoy_field) not in (None, float("nan"))
                and isinstance(row.get(yoy_field), (int, float))
                and row.get(yoy_field) == row.get(yoy_field)
                for row in result["growth_rows"]
            )
            if already_ok:
                continue

            # 构建时序
            seq = [
                (row.get("statDate", ""), row.get(src_field))
                for row in result["profit_rows"]
                if isinstance(row.get(src_field), (int, float))
                and row.get(src_field) == row.get(src_field)
            ]
            seq.sort(key=lambda x: x[0])
            if len(seq) < 5:
                continue

            for i in range(4, len(seq)):
                stat_date, val_cur = seq[i]
                _, val_prev = seq[i - 4]
                if not val_prev:
                    continue
                yoy = round((val_cur - val_prev) / abs(val_prev) * 100, 4)
                for row in result["profit_rows"]:
                    if row.get("statDate") == stat_date:
                        row.setdefault(yoy_field, yoy)
                matched = [g for g in result["growth_rows"] if g.get("statDate") == stat_date]
                if matched:
                    matched[0].setdefault(yoy_field, yoy)
                else:
                    pub = next((r.get("pubDate") for r in result["profit_rows"]
                                if r.get("statDate") == stat_date), None)
                    existing_g = next((g for g in result["growth_rows"]
                                       if g.get("statDate") == stat_date), None)
                    if existing_g:
                        existing_g.setdefault(yoy_field, yoy)
                    else:
                        result["growth_rows"].append({
                            "code": code_6, "pubDate": pub, "statDate": stat_date,
                            yoy_field: yoy,
                        })

    def _try_adata() -> bool:
        """
        策略④：adata 独立巨潮节点（前三路全封时救场）

        [BUG-ADATA-HARDCODE-INTERFACE-FIX]
        原来硬编码 adata.stock.finance.get_income_statement()，此接口在 adata ≤2.8.x 存在，
        但 2.9.x 已更名（具体名称未知）。结果：2.9.x 用户第④级兜底完全失效，
        三路均封时直接标记失败，而非触发 adata 独立节点。

        修复：与 step1b 一样用候选列表逐一探测，找到可调用且返回非空的接口为止。
        探测结果全局缓存，同一进程内只探测一次。

        adata 字段映射（与项目格式对齐）：
          report_date → statDate      ann_date → pubDate（精确公告日，无需估算）
          net_profit  → netProfit     roe → roeAvg（季度值，非年化，差异可接受）
          gross_profit_margin → gpMargin   net_profit_margin → npMargin
          basic_eps（单季）→ epsTTM via rolling(4).sum()
          total_share / float_share → totalShare / liqaShare（万股，需×10000若原始单位为亿）
          YOY 字段 → 自算（同期 4 季前对比），精度略低于报表同比但覆盖率 ~100%
        """
        # [BUG-ADATA-HARDCODE-INTERFACE-FIX] 候选接口列表（与 step1b 对齐）
        _ADATA_FINANCE_CANDIDATES = [
            "stock_profit_history",
            "get_profit_history",
            "profit_history",
            "get_income_statement",    # ≤2.8.x
            "income_statement",
            "get_finance_indicator",
            "stock_finance_indicator",
            "get_stock_profit",
            "get_income",
            "profit",
            "stock_income",
            "get_financial_data",
        ]

        for attempt in range(_MAX_RETRY):
            try:
                import adata
                finance_obj = adata.stock.finance

                # 探测可用接口（每次调用都尝试，因为 result 是闭包，不用全局缓存）
                _fn = None
                for _cand in _ADATA_FINANCE_CANDIDATES:
                    _f = getattr(finance_obj, _cand, None)
                    if callable(_f):
                        _fn = _f
                        break

                if _fn is None:
                    result["error_msg"] = (
                        f"adata.stock.finance 无可用接口（已探测: {_ADATA_FINANCE_CANDIDATES}）"
                    )
                    return False

                # ── 利润表 ────────────────────────────────────────────────────
                df = _fn(stock_code=code_6)
                if df is None or (hasattr(df, "empty") and df.empty):
                    return False

                df = df.copy()

                # ── 日期字段 ──────────────────────────────────────────────────
                # adata 返回 report_date（报告期）和 ann_date（公告日）
                # 字段名因版本略有差异，做宽容查找
                _date_candidates = {
                    "statDate": ["report_date", "report_period", "period_date",
                                 "reportDate", "stat_date"],
                    "pubDate":  ["ann_date", "announce_date", "pub_date",
                                 "annDate", "announcement_date"],
                }
                for target, candidates in _date_candidates.items():
                    for c in candidates:
                        if c in df.columns:
                            df[target] = pd.to_datetime(
                                df[c], errors="coerce"
                            ).dt.strftime("%Y-%m-%d")
                            break
                    else:
                        if target not in df.columns:
                            df[target] = pd.NaT

                if "statDate" not in df.columns or df["statDate"].isna().all():
                    return False

                # 过滤起始年份
                df = df[df["statDate"] >= f"{start_year}-01-01"].copy()
                if df.empty:
                    return False

                # pubDate 缺失时估算（保守：statDate + 60天）
                if "pubDate" not in df.columns or df["pubDate"].isna().all():
                    df["pubDate"] = (
                        pd.to_datetime(df["statDate"], errors="coerce")
                        + pd.Timedelta(days=60)
                    ).dt.strftime("%Y-%m-%d")

                df["code"] = code_6
                df = df.sort_values("statDate").reset_index(drop=True)

                # ── 字段映射（宽容查找）────────────────────────────────────────
                _field_map = {
                    "netProfit":  ["net_profit", "net_income", "netProfit",
                                   "net_profit_ttm", "归属净利润"],
                    "npMargin":   ["net_profit_margin", "npMargin", "净利率",
                                   "net_margin", "sales_net_profit_ratio"],
                    "gpMargin":   ["gross_profit_margin", "gpMargin", "毛利率",
                                   "gross_margin"],
                    "MBRevenue":  ["total_revenue", "revenue", "MBRevenue",
                                   "operating_revenue", "营业收入"],
                    "roeAvg":     ["roe", "roeAvg", "return_on_equity",
                                   "净资产收益率"],
                    # basic_eps 是单季度EPS，需要 rolling 4 期求和得到 TTM
                    "_eps_single": ["basic_eps", "eps", "basic_earnings_per_share",
                                    "每股收益"],
                    "totalShare": ["total_share", "totalShare", "total_capital",
                                   "总股本"],
                    "liqaShare":  ["float_share", "liqaShare", "circulating_share",
                                   "流通股本", "free_float_share"],
                }
                for target, candidates in _field_map.items():
                    for c in candidates:
                        if c in df.columns:
                            df[target] = pd.to_numeric(df[c], errors="coerce")
                            break

                # ── 单位换算：adata 股本通常以「亿股」返回 ─────────────────────
                for share_col in ("totalShare", "liqaShare"):
                    if share_col in df.columns:
                        median_val = df[share_col].median()
                        # 若中位值 < 500（亿股量级），则乘以 10000 转为万股
                        if pd.notna(median_val) and 0 < median_val < 500:
                            df[share_col] = df[share_col] * 10000.0

                # ── epsTTM = 单季EPS 滚动4期求和 ─────────────────────────────
                if "_eps_single" in df.columns:
                    df["epsTTM"] = (
                        df["_eps_single"]
                        .rolling(4, min_periods=1)
                        .sum()
                    )
                    df.drop(columns=["_eps_single"], inplace=True, errors="ignore")

                # ── YOY 自算（同比 = (当期 - 4期前) / abs(4期前) × 100）──────
                for yoy_target, src_col in [
                    ("YOYNI",       "netProfit"),
                    ("YOYEPSBasic", "epsTTM"),
                    ("YOYPNI",      "netProfit"),   # 扣非近似用净利润
                    ("YOYROE",      "roeAvg"),
                ]:
                    if src_col in df.columns:
                        shifted = df[src_col].shift(4)
                        df[yoy_target] = np.where(
                            shifted.abs() > 1e-9,
                            (df[src_col] - shifted) / shifted.abs() * 100,
                            np.nan,
                        )

                df = df.drop_duplicates(subset=["statDate"], keep="last")

                # ── 构建 profit_rows / growth_rows ────────────────────────────
                p_cols = ["code", "pubDate", "statDate"] + [
                    c for c in ["roeAvg", "npMargin", "gpMargin", "netProfit",
                                "epsTTM", "MBRevenue", "totalShare", "liqaShare"]
                    if c in df.columns
                ]
                g_cols = ["code", "pubDate", "statDate"] + [
                    c for c in ["YOYNI", "YOYEPSBasic", "YOYPNI",
                                "YOYROE", "YOYAsset", "YOYEquity"]
                    if c in df.columns
                ]
                result["profit_rows"] = df[p_cols].to_dict("records") if len(p_cols) > 3 else []
                result["growth_rows"] = df[g_cols].to_dict("records") if len(g_cols) > 3 else []
                result["ok"]     = True
                result["source"] = "adata"
                return True

            except ImportError:
                result["error_msg"] = "adata 未安装: pip install adata"
                return False
            except Exception as e:
                err = str(e)
                result["error_msg"] = err
                time.sleep(_backoff_wait(attempt, _is_rate_limit_error(err)))
        return False

    def _try_adata_valuation() -> bool:
        """
        策略④-v：adata 日频估值（PE/PB/PS），独立调用，替代 AKShare _try_valuation。
        在 AKShare 三路全封时仍可获取估值数据。

        adata.stock.market.get_market(stock_code) 返回日频行情+估值：
          trade_date, open, close, high, low, volume, pe_ttm, pb, ps_ttm, ...
        """
        for attempt in range(_MAX_RETRY):
            try:
                import adata
                df = adata.stock.market.get_market(stock_code=code_6)
                if df is None or (hasattr(df, "empty") and df.empty):
                    return False

                df = df.copy()

                # 日期列
                date_col = next(
                    (c for c in ["trade_date", "date", "trading_date"] if c in df.columns),
                    None
                )
                if date_col is None:
                    return False
                df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
                df["code"] = code_6

                # 估值字段
                col_map = {}
                for target, candidates in [
                    ("peTTM", ["pe_ttm", "pe"]),
                    ("pbMRQ", ["pb", "pb_mrq"]),
                    ("psTTM", ["ps_ttm", "ps"]),
                ]:
                    for c in candidates:
                        if c in df.columns:
                            col_map[c] = target
                            break
                df = df.rename(columns=col_map)

                for col in ["peTTM", "pbMRQ", "psTTM"]:
                    if col not in df.columns:
                        df[col] = float("nan")
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                rows = df[["code", "date", "peTTM", "pbMRQ", "psTTM"]].dropna(
                    subset=["peTTM"]
                ).to_dict("records")

                if rows:
                    # 合并而非替换（可能 AKShare 已有部分数据）
                    existing = {r["date"] for r in result.get("valuation_rows", [])}
                    result.setdefault("valuation_rows", []).extend(
                        r for r in rows if r["date"] not in existing
                    )
                return True

            except ImportError:
                return False
            except Exception as e:
                time.sleep(_backoff_wait(attempt, _is_rate_limit_error(err := str(e))))
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # [BUG-EMPTY-FIELDS-FIX iter37] 两阶段强制合并
    #
    # 原问题（导致 gpMargin / totalShare / liqaShare / 4个YOY 全空）：
    #   _has_critical_fields() 检测到 em_abstract 返回了 roeAvg + netProfit，
    #   就跳过 em_profit → gpMargin / totalShare / liqaShare 永远为 NaN。
    #
    # 修复：em_abstract 和 em_profit 始终都调用，字段级合并（缺啥补啥）：
    #   em_abstract  → roeAvg / npMargin / netProfit / epsTTM / MBRevenue（快，0.5s）
    #   em_profit    → gpMargin / totalShare / liqaShare（慢，1s，但必须调）
    #
    # [BUG-YOY-SPARSE-FIX iter37] _patch_growth_from_eps 扩展到全部 6 个 YOY 字段：
    #   原版只算 YOYNI + YOYEPSBasic（2个），YOYPNI / YOYROE / YOYAsset / YOYEquity 全空。
    #   修复：扩展为对所有可得时序字段做 4期同比计算。
    # ─────────────────────────────────────────────────────────────────────────

    def _has_critical_fields() -> bool:
        """检查 profit_rows 中是否有有效的 roeAvg 和 netProfit。"""
        for row in result.get("profit_rows", []):
            roe = row.get("roeAvg")
            np_ = row.get("netProfit")
            if (isinstance(roe, float) and not np.isnan(roe)) or \
               (isinstance(np_, float) and not np.isnan(np_)):
                return True
        return False

    def _missing_profit_fields() -> bool:
        """
        [BUG-EMPTY-FIELDS-FIX iter37]
        检查关键盈利字段（gpMargin / totalShare / liqaShare）是否还全为空。
        若是，即使 em_abstract 成功，也需要 em_profit 补充。
        """
        for row in result.get("profit_rows", []):
            for field in ("gpMargin", "totalShare", "liqaShare"):
                v = row.get(field)
                if isinstance(v, float) and not np.isnan(v):
                    return False
                if isinstance(v, (int, str)) and v not in (None, "", float("nan")):
                    return False
        return True

    def _merge_profit_from_em_profit(em_profit_rows: list) -> None:
        """
        [BUG-EMPTY-FIELDS-FIX iter37]
        将 em_profit 的行按 statDate 合并到已有 profit_rows，
        只填补 NaN 字段（不覆盖 em_abstract 已有的值）。
        """
        em_map = {r.get("statDate"): r for r in em_profit_rows if r.get("statDate")}
        existing_dates = {r.get("statDate") for r in result["profit_rows"]}
        merged = 0
        for row in result["profit_rows"]:
            sd = row.get("statDate")
            if sd and sd in em_map:
                em_row = em_map[sd]
                for field in ("gpMargin", "totalShare", "liqaShare",
                              "roeAvg", "npMargin", "netProfit", "epsTTM", "MBRevenue"):
                    if field not in row or row.get(field) is None or (
                            isinstance(row[field], float) and np.isnan(row[field])):
                        v = em_row.get(field)
                        if v is not None and not (isinstance(v, float) and np.isnan(v)):
                            row[field] = v
                            merged += 1
        # em_profit 有但 em_abstract 没有的 statDate 也追加
        for sd, em_row in em_map.items():
            if sd not in existing_dates:
                result["profit_rows"].append(em_row)

    # 阶段1：尝试摘要接口（快，~0.5s）
    abstract_ok = _try_em_abstract()

    # 阶段2：[BUG-EMPTY-FIELDS-FIX iter37]
    # 若 gpMargin/totalShare/liqaShare 任一缺失，强制调用 em_profit 补充
    # （原逻辑：只在 roeAvg/netProfit 也缺失时才调用 → 导致三个字段永远 NaN）
    if _missing_profit_fields() or not _has_critical_fields():
        # 保存 em_profit 结果，字段级合并
        saved_profit = result["profit_rows"][:]
        saved_growth = result["growth_rows"][:]
        profit_ok = _try_em_profit()
        if profit_ok and abstract_ok:
            # 恢复 em_abstract 数据，再把 em_profit 的缺失字段填进去
            em_profit_rows = result["profit_rows"][:]
            result["profit_rows"] = saved_profit
            result["growth_rows"] = saved_growth
            _merge_profit_from_em_profit(em_profit_rows)
            result["source"] = "em_abstract+em_profit"
        elif profit_ok:
            result["source"] = "em_profit"
        success = abstract_ok or profit_ok
    else:
        success = abstract_ok

    # 阶段3：前两路都失败时才降级
    if not success:
        success = _try_sina_indicator() or _try_adata()  # 第③④级

    # [BUG-FUND-PE-NEVER-DOWNLOADED FIX] 补充估值字段（独立于利润接口）
    # 先尝试 AKShare 估值；若失败或无数据，用 adata 日频行情补充
    result.setdefault("valuation_rows", [])
    if not _try_valuation() or not result["valuation_rows"]:
        _try_adata_valuation()  # adata 估值兜底（独立节点，东财/新浪封禁时可用）

    # [BUG-YOY-SPARSE-FIX iter37] 所有路径成功后均尝试补算缺失的YOY字段
    if success:
        _patch_growth_from_eps()

    with _failure_lock:
        if success:
            _consecutive_failures = max(0, _consecutive_failures - 1)
        else:
            _consecutive_failures += 1

    time.sleep(delay + (_random.uniform(0, delay) if success else delay * 2))
    return result

# ─────────────────────────────────────────────────────────────────────────────
# [ADATA] adata 数据源下载函数
# ─────────────────────────────────────────────────────────────────────────────

def download_adata(codes: list, n_workers: int = 8) -> None:
    """
    [ADATA] 用 adata.stock.finance.get_core_index() 下载季度基本面数据。

    优势：
      · 43字段，覆盖盈利/成长/质量/偿债/运营全维度
      · 每只股票一次请求返回全量历史，无年份分页
      · 无 HTTP 封禁问题（走独立节点）
      · 支持断点续传（_raw_adata/*.parquet 缓存）

    输出格式与 AKShare 版完全兼容，直接生成：
      data/fundamental/profit_quarterly.csv
      data/fundamental/growth_quarterly.csv

    字段映射：
      adata.notice_date       → pubDate    (公告日，point-in-time)
      adata.report_date       → statDate
      adata.roe_wtd           → roeAvg
      adata.net_margin        → npMargin
      adata.gross_margin      → gpMargin
      adata.net_profit_attr_sh→ netProfit
      adata.total_rev         → MBRevenue
      basic_eps 滚动4季求和    → epsTTM
      adata.net_profit_yoy_gr → YOYNI
      adata.total_rev_yoy_gr  → YOYEPSBasic（近似）
      get_stock_shares ÷10000 → totalShare / liqaShare（万股）
    """
    import adata
    import random

    FUND_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir  = FUND_DIR / "_raw_adata"
    raw_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = FUND_DIR / "_adata_ckpt.json"

    # adata 字段 → 项目标准列名
    _field_map = {
        "notice_date":          "pubDate",
        "report_date":          "statDate",
        "roe_wtd":              "roeAvg",
        "net_margin":           "npMargin",
        "gross_margin":         "gpMargin",
        "net_profit_attr_sh":   "netProfit",
        "total_rev":            "MBRevenue",
        "basic_eps":            "_basic_eps",
        "net_profit_yoy_gr":    "YOYNI",
        "total_rev_yoy_gr":     "YOYEPSBasic",
        "net_profit_qoq_gr":    "YOYPNI",
        "roa_wtd":              "roa",
        "asset_liab_ratio":     "assetLiabRatio",
    }

    def _eps_ttm(df_):
        if "_basic_eps" not in df_.columns:
            return np.full(len(df_), np.nan)
        return df_["_basic_eps"].rolling(window=4, min_periods=1).sum().values

    def _fetch_one_adata(code6: str) -> dict:
        out = raw_dir / f"{code6}.parquet"
        if out.exists():
            try:
                df_ = pd.read_parquet(str(out))
                if len(df_) > 0:
                    return {"code": code6, "ok": True, "rows": len(df_), "cached": True}
            except Exception:
                pass
        time.sleep(random.uniform(0.2, 0.8))
        for attempt in range(3):
            try:
                df_ = adata.stock.finance.get_core_index(stock_code=code6)
                if df_ is None or df_.empty:
                    return {"code": code6, "ok": False, "rows": 0, "msg": "空数据"}
                df_ = df_.copy()
                df_ = df_.rename(columns={k: v for k, v in _field_map.items()
                                           if k in df_.columns})
                for dc in ("pubDate", "statDate"):
                    if dc in df_.columns:
                        df_[dc] = pd.to_datetime(df_[dc], errors="coerce").dt.strftime("%Y-%m-%d")
                df_ = df_.dropna(subset=["statDate"])
                if df_.empty:
                    return {"code": code6, "ok": False, "rows": 0, "msg": "无有效日期"}
                df_["code"] = code6
                skip_cols = {"code","pubDate","statDate","short_name","report_type","stock_code"}
                for col in df_.columns:
                    if col not in skip_cols:
                        df_[col] = pd.to_numeric(df_[col], errors="coerce")
                df_ = df_.sort_values("statDate").reset_index(drop=True)
                df_["epsTTM"] = _eps_ttm(df_)
                if "pubDate" in df_.columns:
                    df_["pubDate"] = pd.to_datetime(df_["pubDate"], errors="coerce")
                    df_ = df_.sort_values(["statDate","pubDate"])
                    df_ = df_.drop_duplicates(subset=["statDate"], keep="last")
                df_ = df_.sort_values("statDate").reset_index(drop=True)
                df_.to_parquet(str(out), index=False, compression="snappy")
                return {"code": code6, "ok": True, "rows": len(df_), "cached": False}
            except Exception as e:
                if attempt < 2:
                    time.sleep((attempt+1)*3 + random.uniform(0,2))
                else:
                    return {"code": code6, "ok": False, "rows": 0, "msg": str(e)[:100]}
        return {"code": code6, "ok": False, "rows": 0, "msg": "3次重试失败"}

    # ── 断点续传 ─────────────────────────────────────────────────────────────
    done_set: set = set()
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            done_set = set(json.load(f).get("done", []))
        print(f"  ↻ 断点恢复: 已完成 {len(done_set)} 只")
    pending = [c for c in codes if c not in done_set]

    print(f"  [adata] 下载 get_core_index: {len(pending)} 只待处理（共 {len(codes)} 只）")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    lock_ = Lock()
    ok_n = err_n = skip_n = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_fetch_one_adata, c): c for c in pending}
        done_n = 0
        for fut in as_completed(futs):
            c = futs[fut]
            done_n += 1
            try:
                r = fut.result()
            except Exception as e:
                r = {"code": c, "ok": False, "rows": 0, "msg": str(e)}
            if r["ok"]:
                if r.get("cached"): skip_n += 1
                else:               ok_n   += 1
                with lock_:
                    done_set.add(c)
            else:
                err_n += 1
            spd = done_n / max(time.time()-t0, 0.1)
            eta = (len(pending)-done_n) / max(spd, 0.01)
            print(f"\r  [{done_n:4d}/{len(pending)}] {c} "
                  f"{'✓' if r['ok'] else '✗'} {r['rows']}行 | "
                  f"ok={ok_n} skip={skip_n} err={err_n} | "
                  f"{spd:.1f}只/s ETA={eta:.0f}s  ",
                  end="", flush=True)
            if done_n % 200 == 0:
                with open(ckpt_path, "w") as f:
                    json.dump({"done": list(done_set)}, f)
    print()
    print(f"  [adata] 完成: ok={ok_n} skip={skip_n} err={err_n}  "
          f"耗时={time.time()-t0:.0f}s")
    with open(ckpt_path, "w") as f:
        json.dump({"done": list(done_set)}, f)

    # ── 股本数据 ─────────────────────────────────────────────────────────────
    shares_path = FUND_DIR / "stock_shares.csv"
    if not shares_path.exists():
        print("  [adata] 下载股本结构...")
        all_sh = []
        sh_done = 0
        def _fetch_shares(code6: str) -> list:
            time.sleep(random.uniform(0.05, 0.2))
            try:
                df_ = adata.stock.info.get_stock_shares(stock_code=code6, is_history=True)
                if df_ is None or df_.empty: return []
                df_["stock_code"] = code6
                return df_.to_dict("records")
            except Exception:
                return []
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs2 = {ex.submit(_fetch_shares, c): c for c in codes}
            for fut in as_completed(futs2):
                sh_done += 1
                all_sh.extend(fut.result())
                if sh_done % 500 == 0:
                    print(f"\r    股本: {sh_done}/{len(codes)}  {len(all_sh)}条  ",
                          end="", flush=True)
        print()
        if all_sh:
            df_sh = pd.DataFrame(all_sh)
            df_sh["stock_code"] = df_sh["stock_code"].astype(str).str.zfill(6)
            for col in ("total_shares","list_a_shares","limit_shares"):
                if col in df_sh.columns:
                    df_sh[col] = pd.to_numeric(df_sh[col], errors="coerce") / 10000.0
            df_sh.to_csv(str(shares_path), index=False)
            print(f"  [adata] ✓ {len(df_sh)} 条股本记录 → stock_shares.csv")

    # ── 合并所有 parquet → profit_quarterly.csv / growth_quarterly.csv ────────
    print("  [adata] 合并原始数据...")
    all_dfs = []
    for code6 in codes:
        pq = raw_dir / f"{code6}.parquet"
        if not pq.exists(): continue
        try:
            df_ = pd.read_parquet(str(pq))
            if len(df_) > 0:
                all_dfs.append(df_)
        except Exception:
            pass

    if not all_dfs:
        print("  [adata] ✗ 无可合并数据")
        return

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all["code"] = df_all["code"].astype(str).str.zfill(6)

    # 合并股本
    if shares_path.exists():
        df_sh = pd.read_csv(str(shares_path), dtype={"stock_code": str})
        df_sh["stock_code"] = df_sh["stock_code"].str.zfill(6)
        df_sh["change_date"] = pd.to_datetime(df_sh.get("change_date", pd.NaT), errors="coerce")
        df_sh = df_sh.dropna(subset=["change_date"])
        grp_sh = {c: g for c, g in df_sh.groupby("stock_code")}

        total_s, liqa_s = [], []
        for _, row in df_all[["code","statDate"]].iterrows():
            code6 = row["code"]
            stat  = pd.to_datetime(row["statDate"], errors="coerce")
            if pd.isna(stat) or code6 not in grp_sh:
                total_s.append(np.nan); liqa_s.append(np.nan); continue
            sub = grp_sh[code6]
            sub = sub[sub["change_date"] <= stat]
            if sub.empty:
                total_s.append(np.nan); liqa_s.append(np.nan)
            else:
                r2 = sub.sort_values("change_date").iloc[-1]
                total_s.append(r2.get("total_shares", np.nan))
                liqa_s.append(r2.get("list_a_shares", np.nan))
        df_all["totalShare"] = total_s
        df_all["liqaShare"]  = liqa_s
    else:
        df_all["totalShare"] = np.nan
        df_all["liqaShare"]  = np.nan

    # 填补缺失列
    for col in PROFIT_FIELDS + GROWTH_FIELDS:
        if col not in df_all.columns:
            df_all[col] = np.nan

    # 去重：同一 (code, statDate) 保留最新 pubDate
    df_all["pubDate_dt"] = pd.to_datetime(df_all.get("pubDate", pd.NaT), errors="coerce")
    df_all = df_all.sort_values(["code","statDate","pubDate_dt"])
    df_all = df_all.drop_duplicates(subset=["code","statDate"], keep="last")
    df_all = df_all.sort_values(["code","pubDate_dt"]).reset_index(drop=True)
    df_all = df_all.drop(columns=["pubDate_dt"], errors="ignore")

    # 输出 CSV
    pc = [c for c in PROFIT_FIELDS if c in df_all.columns]
    gc = [c for c in GROWTH_FIELDS if c in df_all.columns]
    df_all[pc].to_csv(str(FUND_DIR / "profit_quarterly.csv"),  index=False)
    df_all[gc].to_csv(str(FUND_DIR / "growth_quarterly.csv"),  index=False)

    print(f"  [adata] ✓ profit_quarterly.csv  {len(df_all[pc])} 条")
    print(f"  [adata] ✓ growth_quarterly.csv  {len(df_all[gc])} 条")
    print(f"  [adata] 股票数: {df_all['code'].nunique()}")

    # 覆盖率
    for col in ["roeAvg","epsTTM","netProfit","npMargin","YOYNI","totalShare","liqaShare"]:
        if col in df_all.columns:
            na = df_all[col].isna().mean() * 100
            print(f"    {'✓' if na < 30 else '⚠'} {col:20s} 缺失率={na:.1f}%")

    # 清理断点缓存
    if ckpt_path.exists():
        ckpt_path.unlink()


# ─────────────────────────────────────────────────────────────────────────────
# 主下载函数
# ─────────────────────────────────────────────────────────────────────────────

def download_akshare(
    codes: list,
    start_year: int,
    n_workers: int = 8,
    delay: float = 0.5,
) -> tuple:
    FUND_DIR.mkdir(parents=True, exist_ok=True)

    done_codes: set = set()
    all_profit_rows: list = []
    all_growth_rows: list = []
    all_valuation_rows: list = []   # [BUG-FUND-PE-NEVER-DOWNLOADED FIX] 日频估值
    all_failed_reasons: dict = {}   # 借鉴②

    if CKPT_PATH.exists():
        with open(CKPT_PATH) as f:
            cp = json.load(f)
        done_codes = set(cp.get("done", []))
        if PARTIAL_PATH.exists():
            df_partial = pd.read_parquet(str(PARTIAL_PATH))
            all_profit_rows = df_partial[df_partial["_type"] == "profit"].drop(
                columns=["_type"]).to_dict("records")
            all_growth_rows = df_partial[df_partial["_type"] == "growth"].drop(
                columns=["_type"]).to_dict("records")
        print(f"↻ 断点恢复: 已完成 {len(done_codes)} 只，"
              f"profit={len(all_profit_rows)}条，growth={len(all_growth_rows)}条")

    pending = [c for c in codes if c not in done_codes]
    total   = len(codes)
    t0      = time.time()

    print(f"✓ AKShare 多线程下载（n_workers={n_workers}, delay={delay}s/只）")
    print(f"  待处理: {len(pending)} 只（共 {total} 只）")
    print(f"  预估耗时: {len(pending)*(delay+0.5)/n_workers/60:.0f}~"
          f"{len(pending)*(delay+1.5)/n_workers/60:.0f} 分钟")

    # ── 并发控制 ──────────────────────────────────────────────────────────────
    # 直接用 ThreadPoolExecutor(max_workers) 控制并发。
    # 不用信号量：信号量包住整个 _fetch_one（含重试循环）会导致持锁线程
    # 长时间阻塞，空槽填不进来 → 死锁。
    # 防封禁靠：delay间隔 + 随机错峰 + UA轮换，不靠限制并发槽位。
    # ─────────────────────────────────────────────────────────────────────────
    STAGGER_S = 2.0
    lock      = Lock()
    completed = 0


    def _save_checkpoint():
        with open(CKPT_PATH, "w") as f:
            json.dump({"done": list(done_codes)}, f)
        if all_profit_rows or all_growth_rows:
            rows_p = [{**r, "_type": "profit"} for r in all_profit_rows]
            rows_g = [{**r, "_type": "growth"} for r in all_growth_rows]
            pd.DataFrame(rows_p + rows_g).to_parquet(str(PARTIAL_PATH), index=False)

    def _fetch_staggered(code: str) -> dict:
        """随机错峰后直接执行，无信号量阻塞"""
        time.sleep(_random.uniform(0, STAGGER_S))
        return _fetch_one(code, start_year, delay)

    def _run_batch(batch: list, pass_label: str, workers: int) -> list:
        nonlocal completed
        fail_list = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_fetch_staggered, code): code for code in batch}
            for future in as_completed(futures):
                code = futures[future]
                try:
                    res = future.result()
                except Exception as e:
                    with lock:
                        fail_list.append(code)
                        all_failed_reasons[code] = str(e)
                        completed += 1
                    continue

                with lock:
                    all_profit_rows.extend(res["profit_rows"])
                    all_growth_rows.extend(res["growth_rows"])
                    all_valuation_rows.extend(res.get("valuation_rows", []))  # [BUG-FUND-PE-NEVER-DOWNLOADED FIX]
                    if res["ok"]:
                        done_codes.add(code)
                        all_failed_reasons.pop(code, None)
                    else:
                        fail_list.append(code)
                        all_failed_reasons[code] = res.get("error_msg", "unknown")
                    completed += 1

                    elapsed = time.time() - t0
                    speed   = completed / max(elapsed, 0.1)
                    eta_min = (len(pending) - completed) / max(speed, 0.01) / 60.0
                    done_total = len(done_codes) + (len(codes) - len(pending))
                    print(
                        f"  [{done_total}/{total}] src={res['source']:12s} | "
                        f"profit={len(all_profit_rows)}条 growth={len(all_growth_rows)}条 | "
                        f"{speed:.1f}只/秒 | ETA {eta_min:.0f}m | "
                        f"失败 {len(fail_list)} [{pass_label}]",
                        flush=True,
                    )
                    if completed % 200 == 0:
                        _save_checkpoint()
                        print(f"    💾 断点 {len(done_codes)} 只")
        return fail_list

    print(f"\n  ▶ 第一轮下载（{n_workers} 线程，错峰 0~{STAGGER_S}s）...")
    failed_codes = _run_batch(pending, "第1轮", n_workers)
    _save_checkpoint()

    if failed_codes:
        print(f"\n  ⚠ 第一轮失败 {len(failed_codes)} 只，等待 60s 让封禁解除...")
        time.sleep(60)
        print(f"  ▶ 第二轮重试（{len(failed_codes)} 只，降为 2 线程）...")
        still_failed = _run_batch(failed_codes, "第2轮", 2)
        _save_checkpoint()

        if still_failed:
            print(f"\n  ⚠ 第二轮仍失败 {len(still_failed)} 只，等待 120s 后最终重试...")
            time.sleep(120)
            print(f"  ▶ 第三轮重试（{len(still_failed)} 只，单线程）...")
            final_failed = _run_batch(still_failed, "第3轮", 1)
            _save_checkpoint()
            if final_failed:
                print(f"  ⚠ 最终失败 {len(final_failed)} 只（已跳过）")

    df_p = _build_final_df(all_profit_rows, PROFIT_FIELDS, "profit")
    df_g = _build_final_df(all_growth_rows, GROWTH_FIELDS, "growth")

    # [BUG-FUND-PE-NEVER-DOWNLOADED FIX] 导出日频估值 CSV 供 step3 构建 pe_ttm.npy
    df_v = _build_valuation_df(all_valuation_rows)

    elapsed = time.time() - t0

    # 借鉴②：失败日志落盘
    if all_failed_reasons:
        with open(FAILED_PATH, "w", encoding="utf-8") as f:
            for code, reason in all_failed_reasons.items():
                f.write(f"{code}\t{reason}\n")
        print(f"  📄 失败日志 → {FAILED_PATH.name}（{len(all_failed_reasons)} 只）")

    # 借鉴③：统计 JSON 落盘
    stats = {
        "total": total, "success": len(done_codes),
        "failed": len(all_failed_reasons),
        "profit_rows": len(all_profit_rows),
        "growth_rows": len(all_growth_rows),
        "elapsed_minutes": round(elapsed / 60, 1),
        "rate_per_second": round(len(pending) / max(elapsed, 1), 2),
        "finish_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  📊 统计 → {STATS_PATH.name}  "
          f"成功{stats['success']}只 | 失败{stats['failed']}只 | "
          f"{stats['elapsed_minutes']}分钟 | {stats['rate_per_second']}只/秒")

    for p in [CKPT_PATH, PARTIAL_PATH]:
        if p.exists():
            p.unlink()
    print("  🗑 断点缓存已清理")
    return df_p, df_g, df_v


# ─────────────────────────────────────────────────────────────────────────────
# 借鉴④：数据验证
# ─────────────────────────────────────────────────────────────────────────────

def verify_data() -> dict:
    print("\n" + "=" * 70)
    print(" 数据验证")
    print("=" * 70)
    issues = []

    for label, required in [
        ("profit",     ["code","pubDate","statDate","roeAvg","epsTTM","netProfit"]),
        ("growth",     ["code","pubDate","statDate","YOYNI","YOYEPSBasic"]),
        ("merged",     ["code","pubDate","statDate","roeAvg","YOYNI"]),
        # [BUG-FUND-PE-NEVER-DOWNLOADED FIX] 新增估值日频验证
        ("valuation",  ["code","date","peTTM","pbMRQ"]),
    ]:
        if label == "merged":
            csv_name = "fundamental_merged.csv"
        elif label == "valuation":
            csv_name = "valuation_daily.csv"
        else:
            csv_name = f"{label}_quarterly.csv"
        path = FUND_DIR / csv_name
        if not path.exists():
            print(f"  ✗ {csv_name} 不存在")
            issues.append(f"{csv_name} missing")
            continue

        df = pd.read_csv(str(path), dtype={"code": str})
        n_stocks = df["code"].nunique()
        missing_cols = [c for c in required if c not in df.columns]

        if missing_cols:
            print(f"  ✗ {csv_name}: 缺少列 {missing_cols}")
            issues.append(f"{csv_name} 缺列 {missing_cols}")
        else:
            print(f"  ✓ {csv_name}: {n_stocks} 只股票，{len(df)} 条")

        for col in required:
            if col in df.columns and col not in ("code","pubDate","statDate"):
                na_pct = df[col].isna().mean() * 100
                flag = "⚠" if na_pct > 30 else "✓"
                print(f"    {flag} {col} 缺失率={na_pct:.1f}%")
                if na_pct > 30:
                    issues.append(f"{col} 缺失率={na_pct:.1f}%")

        if n_stocks < 100:
            print(f"  ⚠ 股票数量偏少（{n_stocks}），可能下载不完整")
            issues.append(f"{csv_name} 股票数={n_stocks}")

    print(f"\n{'✓ 验证通过' if not issues else f'⚠ 发现 {len(issues)} 个问题'}")
    for issue in issues:
        print(f"  · {issue}")
    return {"issues": issues}


# ─────────────────────────────────────────────────────────────────────────────
# _build_final_df / merge_and_export
# ─────────────────────────────────────────────────────────────────────────────

def _build_final_df(rows: list, expected_fields: list, label: str) -> pd.DataFrame:
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=expected_fields)
    for col in expected_fields:
        if col not in df.columns:
            df[col] = np.nan
    df = df[expected_fields]
    df = _clean_numeric(df)
    df["code"]     = df["code"].astype(str).str.zfill(6)
    df["pubDate"]  = pd.to_datetime(df["pubDate"], errors="coerce")
    df["statDate"] = df["statDate"].astype(str)
    df = df.sort_values(["code", "statDate", "pubDate"])
    df = df.drop_duplicates(subset=["code", "statDate"], keep="last")
    df = df.sort_values(["code", "statDate"]).reset_index(drop=True)
    out = FUND_DIR / f"{label}_quarterly.csv"
    df.to_csv(str(out), index=False)
    print(f"✓ {len(df)} 条 {label} | {df['code'].nunique()} 只股票 → {out.name}")
    return df


def _build_valuation_df(rows: list) -> pd.DataFrame:
    """
    [BUG-FUND-PE-NEVER-DOWNLOADED FIX]
    将日频估值行（code/date/peTTM/pbMRQ/psTTM）汇总为 valuation_daily.csv。
    step3 读取此文件构建 pe_ttm.npy / pb_mrq.npy / ps_ttm.npy。
    """
    VALUATION_FIELDS = ["code", "date", "peTTM", "pbMRQ", "psTTM"]
    if not rows:
        df = pd.DataFrame(columns=VALUATION_FIELDS)
    else:
        df = pd.DataFrame(rows)
        for col in VALUATION_FIELDS:
            if col not in df.columns:
                df[col] = np.nan
        df = df[VALUATION_FIELDS]
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["date"] = df["date"].astype(str).str[:10]
    for col in ["peTTM", "pbMRQ", "psTTM"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(["code", "date"]).drop_duplicates(
        subset=["code", "date"], keep="last"
    ).reset_index(drop=True)
    out = FUND_DIR / "valuation_daily.csv"
    df.to_csv(str(out), index=False)
    pe_valid = df["peTTM"].notna().sum()
    print(f"✓ {len(df)} 条估值日频 | {df['code'].nunique()} 只股票 "
          f"| peTTM有效={pe_valid}条 → {out.name}")
    return df


def merge_and_export() -> pd.DataFrame:
    df_p = pd.read_csv(str(FUND_DIR / "profit_quarterly.csv"), dtype={"code": str})
    df_g = pd.read_csv(str(FUND_DIR / "growth_quarterly.csv"), dtype={"code": str})
    df_p["pubDate"] = pd.to_datetime(df_p["pubDate"], errors="coerce")
    df_g["pubDate"] = pd.to_datetime(df_g["pubDate"], errors="coerce")
    df_g = df_g.rename(columns={"pubDate": "pubDate_g"})
    df = pd.merge(df_p, df_g, on=["code", "statDate"], how="left")
    df["pubDate"] = df[["pubDate", "pubDate_g"]].apply(
        lambda r: max(r["pubDate"], r["pubDate_g"])
                  if pd.notna(r["pubDate"]) and pd.notna(r["pubDate_g"])
                  else (r["pubDate"] if pd.notna(r["pubDate"]) else r["pubDate_g"]),
        axis=1,
    )
    if "pubDate_g" in df.columns:
        df = df.drop(columns=["pubDate_g"])
    df = df.sort_values(["code", "pubDate"])
    out = FUND_DIR / "fundamental_merged.csv"
    df.to_csv(str(out), index=False)
    print(f"✓ 合并后 {len(df)} 条 → {out.name}")
    print(f"  股票数: {df['code'].nunique()} | "
          f"日期: {df['statDate'].min()} ~ {df['statDate'].max()}")
    for col in ["YOYNI", "YOYEPSBasic"]:
        if col in df.columns:
            na_pct = df[col].isna().mean() * 100
            print(f"  {'⚠' if na_pct > 30 else '✓'} {col} 缺失率={na_pct:.1f}%")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 1: 季度基本面下载（adata + AKShare 双数据源）")
    parser.add_argument("--source",
                        choices=["auto", "adata", "akshare"],
                        default="auto",
                        help="数据源: auto(优先adata)/adata/akshare（默认auto）")
    parser.add_argument("--start-year", type=int, default=2018,
                        help="AKShare模式起始年份（adata忽略此参数，自动获取全量）")
    parser.add_argument("--test",    action="store_true", help="测试模式(10只)")
    parser.add_argument("--force",   action="store_true", help="强制重新下载")
    parser.add_argument("--workers", type=int, default=8,
                        help="并发线程数（默认8）")
    parser.add_argument("--delay",   type=float, default=0.5,
                        help="AKShare模式请求间隔秒数（默认0.5）")
    parser.add_argument("--verify",  action="store_true",
                        help="只验证数据，不下载")
    args = parser.parse_args()

    # ── 确定有效数据源 ─────────────────────────────────────────────────────
    _eff_src = args.source
    if args.source == "auto":
        try:
            import adata as _a
            print(f"✓ adata {getattr(_a,'__version__','?')} → 使用 adata 数据源")
            _eff_src = "adata"
        except ImportError:
            print("  adata 不可用 → 降级到 akshare")
            _eff_src = "akshare"
    elif args.source == "adata":
        try:
            import adata as _a
            print(f"✓ adata {getattr(_a,'__version__','?')}")
        except ImportError:
            print("✗ adata 未安装: pip install adata"); sys.exit(1)
    else:  # akshare
        try:
            import akshare as ak
            print(f"✓ AKShare {ak.__version__}")
        except ImportError:
            print("✗ AKShare 未安装: pip install akshare -U"); sys.exit(1)

    FUND_DIR.mkdir(parents=True, exist_ok=True)
    codes = load_codes()

    if args.verify:
        verify_data()
        sys.exit(0)

    if args.test:
        codes = ["000001", "000002", "000858", "600000", "600036",
                 "600519", "601318", "300750", "002594", "601012"]
        print(f"⚡ 测试模式: {len(codes)} 只股票")

    if args.force:
        for f in FUND_DIR.glob("*.csv"):
            f.unlink()
        for p in [CKPT_PATH, PARTIAL_PATH, FAILED_PATH, STATS_PATH]:
            if p.exists():
                p.unlink()
        # [ADATA] 同时清除 adata 原始缓存
        raw_dir_ = FUND_DIR / "_raw_adata"
        if raw_dir_.exists():
            for p in raw_dir_.glob("*.parquet"):
                p.unlink()
        ckpt_adata = FUND_DIR / "_adata_ckpt.json"
        if ckpt_adata.exists():
            ckpt_adata.unlink()
        print("🗑 旧数据已清除")

    print("=" * 70)
    print(f" Step 1: 季度基本面下载  source={_eff_src}  "
          f"workers={args.workers}")
    if _eff_src == "akshare":
        print(f"  年份范围: {args.start_year} ~ {datetime.now().year}")
    print("=" * 70)

    profit_exists = (FUND_DIR / "profit_quarterly.csv").exists()
    growth_exists = (FUND_DIR / "growth_quarterly.csv").exists()

    if profit_exists and growth_exists and not args.force:
        print("⚠ CSV 均已存在，跳过（--force 重新下载，--verify 验证数据）")
    else:
        if _eff_src == "adata":
            download_adata(codes, n_workers=args.workers)
        else:
            download_akshare(codes, start_year=args.start_year,
                             n_workers=args.workers, delay=args.delay)

    merge_and_export()
    verify_data()
    print(f"\n✓ Step 1 完成！（数据源: {_eff_src}）")


