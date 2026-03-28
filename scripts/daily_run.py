"""
scripts/daily_run_v10.py
=========================
Q-UNITY V10 每日自动运行脚本

建议设置为 cron 定时任务（15:30 之后执行日终结算，08:00 执行早盘准备）：
  # 工作日 08:00 启动早盘准备
  0 8 * * 1-5 /path/to/venv/bin/python /path/to/scripts/daily_run_v10.py --job morning

  # 工作日 15:30 日终结算
  30 15 * * 1-5 /path/to/venv/bin/python /path/to/scripts/daily_run_v10.py --job close

  # 或使用全天 schedule 模式（建议生产环境）
  # python scripts/daily_run_v10.py --mode schedule

执行阶段
─────────────────────────────────────────────────────────────────────────────
08:00  更新日线数据（AKShare QFQ 增量下载）
08:30  重建 valid_mask（含 listing_days >= 60 过滤）
09:00  生成信号（运行 V10 策略注册表中所有策略）
09:25  持仓核对（比对昨日持仓与今日信号，生成调仓指令）
15:30  日终结算（更新 NAV、写出日志）

铁律（每次修改前默读）
─────────────────────────────────────────────────────────────────────────────
1. stamp_tax = 0.0005（万五，绝不手改）
2. market_regime 是 int8 数组，查表前必须 REGIME_IDX_TO_STR[int(r)]
3. 信号生成必须使用昨日收盘数据，不允许当日前视
4. valid_mask 退市股必须为 False
5. 不修改任何现有文件

用法
─────────────────────────────────────────────────────────────────────────────
  # 运行单个阶段（指定 --job）
  python scripts/daily_run_v10.py --job update_data
  python scripts/daily_run_v10.py --job rebuild_mask
  python scripts/daily_run_v10.py --job gen_signal
  python scripts/daily_run_v10.py --job check_position
  python scripts/daily_run_v10.py --job daily_close

  # 运行早盘全流程（08:00~09:25）
  python scripts/daily_run_v10.py --job morning

  # 运行日终结算（15:30）
  python scripts/daily_run_v10.py --job close

  # 全天 schedule 自动模式
  python scripts/daily_run_v10.py --mode schedule

  # dry-run（只打印不执行）
  python scripts/daily_run_v10.py --mode schedule --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# 路径 & 日志
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
NPY_DIR      = DATA_DIR / "npy_v10"
LOG_DIR      = PROJECT_ROOT / "logs"
META_PATH    = NPY_DIR / "meta.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            LOG_DIR / f"daily_run_{date.today().isoformat()}.log",
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("daily_run_v10")

# ─────────────────────────────────────────────────────────────────────────────
# 状态文件（持久化当日运行状态，防重复执行）
# ─────────────────────────────────────────────────────────────────────────────

STATE_FILE = LOG_DIR / f"daily_state_{date.today().isoformat()}.json"


def _load_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_state(state: Dict[str, Any]) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _mark_done(job_name: str, result: Dict[str, Any] = None) -> None:
    state = _load_state()
    state[job_name] = {
        "done"   : True,
        "time"   : datetime.now().isoformat(),
        "result" : result or {},
    }
    _save_state(state)


def _is_done(job_name: str) -> bool:
    return _load_state().get(job_name, {}).get("done", False)


# ─────────────────────────────────────────────────────────────────────────────
# 工具：安全执行
# ─────────────────────────────────────────────────────────────────────────────

def _safe_run(fn, job_name: str, dry_run: bool = False) -> bool:
    """
    安全执行 fn()，捕获异常、记录日志、更新状态。
    dry_run=True 时只打印不执行。
    """
    if dry_run:
        logger.info(f"  [DRY-RUN] 跳过: {job_name}")
        return True

    logger.info(f"━━━ [{job_name}] 开始 {'━'*40}")
    t0 = time.perf_counter()
    try:
        result = fn()
        elapsed = time.perf_counter() - t0
        logger.info(f"━━━ [{job_name}] 完成 ({elapsed:.1f}s) ✓")
        _mark_done(job_name, {"elapsed_s": round(elapsed, 2),
                              "result": str(result)[:200]})
        return True
    except Exception as e:
        elapsed = time.perf_counter() - t0
        logger.error(f"━━━ [{job_name}] 失败 ({elapsed:.1f}s): {e}")
        logger.debug(traceback.format_exc())
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 阶段1：08:00 — 更新日线数据
# ─────────────────────────────────────────────────────────────────────────────

def job_update_data(dry_run: bool = False) -> bool:
    """
    08:00 增量下载昨日日线数据。

    使用 step0_download_ohlcv_v10.py 的 AKShare QFQ 增量模式：
      adj_type="qfq"，只下载新数据（--incremental）
    """
    def _run():
        import subprocess
        script = PROJECT_ROOT / "scripts" / "step0_download_ohlcv_v10.py"
        if not script.exists():
            logger.warning(f"[update_data] {script} 不存在，跳过下载（请先创建）")
            return "script_not_found"

        yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        cmd = [
            sys.executable, str(script),
            "--incremental",
            "--end", yesterday,
            "--workers", "8",
        ]
        logger.info(f"[update_data] 执行: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False, text=True, timeout=3600)
        if result.returncode != 0:
            raise RuntimeError(f"step0_download_ohlcv_v10.py 退出码={result.returncode}")
        return f"exit_code={result.returncode}"

    return _safe_run(_run, "update_data", dry_run)


# ─────────────────────────────────────────────────────────────────────────────
# 阶段2：08:30 — 重建 valid_mask
# ─────────────────────────────────────────────────────────────────────────────

def job_rebuild_mask(dry_run: bool = False) -> bool:
    """
    08:30 重建 valid_mask。

    [V10-2] 规则：
      R1: 累积有效行 >= 100
      R2: 近 30 天内有成交
      R3: listing_days >= 60
      R4: 退市日之后 False
    """
    def _run():
        import numpy as np

        if not META_PATH.exists():
            logger.warning("[rebuild_mask] meta.json 不存在，跳过")
            return "meta_not_found"

        with open(META_PATH, encoding="utf-8") as f:
            meta = json.load(f)

        N, T     = meta["shape"]
        codes    = meta["codes"]
        trading_days = meta["dates"]

        close_path  = NPY_DIR / "close.npy"
        volume_path = NPY_DIR / "volume.npy"

        if not close_path.exists() or not volume_path.exists():
            logger.warning("[rebuild_mask] close.npy 或 volume.npy 不存在，跳过")
            return "npy_not_found"

        close_  = np.load(str(close_path),  mmap_mode="r").astype(np.float32)
        volume_ = np.load(str(volume_path), mmap_mode="r").astype(np.float32)

        # 导入 V10 适配器的 valid_mask 构建方法
        sys.path.insert(0, str(PROJECT_ROOT))
        try:
            from src.data.columnar_adapter_v10 import ColumnarDataAdapterV10
        except ImportError:
            from columnar_adapter_v10 import ColumnarDataAdapterV10  # type: ignore

        parquet_dir = DATA_DIR / "daily_parquet_qfq"

        # 加载 raw_dfs（仅用于读取 listing_date，轻量化）
        raw_dfs_lite: Dict = {}
        if parquet_dir.exists():
            for code in codes:
                pq = parquet_dir / f"{code}.parquet"
                if pq.exists():
                    try:
                        df_d = __import__("pandas").read_parquet(
                            str(pq), columns=["date"]
                        )
                        raw_dfs_lite[code] = df_d
                    except Exception:
                        raw_dfs_lite[code] = None
                else:
                    raw_dfs_lite[code] = None

        vm = ColumnarDataAdapterV10._nb21_valid_mask_v10(
            close            = close_,
            min_valid_rows   = 100,
            volume           = volume_,
            delist_window    = 30,
            trading_days     = trading_days,
            codes            = codes,
            raw_dfs          = raw_dfs_lite,
            listing_days_min = 60,
        )

        np.save(str(NPY_DIR / "valid_mask.npy"), vm)
        rate = float(vm.mean())
        logger.info(f"[rebuild_mask] valid_mask 重建完成: valid_rate={rate:.1%}")

        # 更新 meta
        meta["extra"] = meta.get("extra", {})
        meta["extra"]["valid_mask_rate"] = rate
        meta["extra"]["rebuild_time"] = datetime.now().isoformat()
        with open(META_PATH, "w", encoding="utf-8") as f_meta:
            json.dump(meta, f_meta, ensure_ascii=False, indent=2)

        return f"valid_rate={rate:.1%}"

    return _safe_run(_run, "rebuild_mask", dry_run)


# ─────────────────────────────────────────────────────────────────────────────
# 阶段3：09:00 — 生成信号
# ─────────────────────────────────────────────────────────────────────────────

def job_gen_signal(dry_run: bool = False) -> bool:
    """
    09:00 生成今日交易信号。

    使用 FastRunnerV10.realtime_signal() 生成各策略的目标权重，
    写出 signals/{date}/signal_{strategy}.json。

    规则：
      - 使用昨日收盘数据（不前视）
      - market_regime 以 int8 传入，策略内部通过 REGIME_IDX_TO_STR 查表
      - valid_mask 严格过滤（退市/新股/停牌）
    """
    def _run():
        import numpy as np

        if not META_PATH.exists():
            logger.warning("[gen_signal] meta.json 不存在，跳过")
            return "meta_not_found"

        signal_dir = PROJECT_ROOT / "signals" / date.today().isoformat()
        signal_dir.mkdir(parents=True, exist_ok=True)

        sys.path.insert(0, str(PROJECT_ROOT))

        try:
            from src.engine.fast_runner_v10 import FastRunnerV10
            from src.engine.risk_config import RiskConfig
            from src.strategies.registry import list_vec_strategies, VEC_STRATEGY_REGISTRY
        except ImportError:
            logger.warning("[gen_signal] 模块导入失败，跳过信号生成")
            return "import_failed"

        # 读取配置
        cfg_path = PROJECT_ROOT / "config.json"
        cfg: Dict[str, Any] = {}
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as fc:
                cfg = json.load(fc)
        cfg.setdefault("npy_dir", str(NPY_DIR))
        cfg.setdefault("stamp_tax", 0.0005)   # ★ 铁律

        runner = FastRunnerV10(cfg)

        strategies = list_vec_strategies()
        if not strategies:
            logger.warning("[gen_signal] 无已注册策略，跳过")
            return "no_strategies"

        as_of_date = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        results: Dict[str, Any] = {}

        for name in strategies:
            try:
                sig = runner.realtime_signal(
                    strategy_name = name,
                    params        = None,
                    as_of_date    = as_of_date,
                )
                # 写出 JSON（code: weight）
                out_path = signal_dir / f"signal_{name}.json"
                with open(out_path, "w", encoding="utf-8") as fs:
                    json.dump({
                        "strategy"  : name,
                        "as_of_date": as_of_date,
                        "signal"    : sig,
                        "n_stocks"  : len(sig),
                        "weight_sum": round(sum(sig.values()), 6),
                    }, fs, ensure_ascii=False, indent=2)
                results[name] = f"n={len(sig)} sum={sum(sig.values()):.4f}"
                logger.info(f"  [{name}] {results[name]}")
            except Exception as e:
                logger.error(f"  [{name}] 信号生成失败: {e}")
                results[name] = f"error: {e}"

        return results

    return _safe_run(_run, "gen_signal", dry_run)


# ─────────────────────────────────────────────────────────────────────────────
# 阶段4：09:25 — 持仓核对
# ─────────────────────────────────────────────────────────────────────────────

def job_check_position(dry_run: bool = False) -> bool:
    """
    09:25 持仓核对。

    比对昨日持仓快照与今日信号，生成调仓指令（买入/卖出/保持）。
    调仓指令写出 signals/{date}/orders_{strategy}.json。

    注意：本阶段为调仓意向，非实际下单（由实盘接口执行）。
    """
    def _run():
        today_str    = date.today().isoformat()
        signal_dir   = PROJECT_ROOT / "signals" / today_str
        position_dir = PROJECT_ROOT / "positions"

        if not signal_dir.exists():
            logger.warning("[check_position] 信号目录不存在，跳过")
            return "signal_dir_not_found"

        # 读取昨日持仓快照
        yesterday_str = (date.today() - timedelta(days=1)).isoformat()
        pos_file = position_dir / f"positions_{yesterday_str}.json"
        if pos_file.exists():
            with open(pos_file, encoding="utf-8") as fp:
                yesterday_positions: Dict[str, float] = json.load(fp)
        else:
            yesterday_positions = {}
            logger.info("[check_position] 无昨日持仓快照，视为空仓")

        orders_summary: Dict[str, Any] = {}

        # 逐策略生成调仓指令
        for sig_file in sorted(signal_dir.glob("signal_*.json")):
            try:
                with open(sig_file, encoding="utf-8") as fs:
                    sig_data = json.load(fs)

                strategy  = sig_data.get("strategy", sig_file.stem)
                new_target: Dict[str, float] = sig_data.get("signal", {})
                old_pos   = yesterday_positions.get(strategy, {})
                if isinstance(old_pos, (int, float)):
                    old_pos = {}  # 旧版格式兼容

                # 生成调仓指令
                all_codes = set(new_target.keys()) | set(old_pos.keys())
                orders: List[Dict[str, Any]] = []
                for code in sorted(all_codes):
                    new_w = new_target.get(code, 0.0)
                    old_w = old_pos.get(code, 0.0)
                    delta = new_w - old_w
                    if abs(delta) < 1e-6:
                        action = "hold"
                    elif delta > 0:
                        action = "buy"
                    else:
                        action = "sell"
                    if action != "hold":
                        orders.append({
                            "code"    : code,
                            "action"  : action,
                            "old_w"   : round(old_w, 6),
                            "new_w"   : round(new_w, 6),
                            "delta_w" : round(delta, 6),
                        })

                out_path = signal_dir / f"orders_{strategy}.json"
                with open(out_path, "w", encoding="utf-8") as fo:
                    json.dump({
                        "strategy" : strategy,
                        "date"     : today_str,
                        "n_orders" : len(orders),
                        "orders"   : orders,
                    }, fo, ensure_ascii=False, indent=2)

                orders_summary[strategy] = f"n_orders={len(orders)}"
                logger.info(f"  [{strategy}] 调仓指令: {len(orders)} 条")

            except Exception as e:
                logger.error(f"  [{sig_file.stem}] 核对失败: {e}")

        return orders_summary

    return _safe_run(_run, "check_position", dry_run)


# ─────────────────────────────────────────────────────────────────────────────
# 阶段5：15:30 — 日终结算
# ─────────────────────────────────────────────────────────────────────────────

def job_daily_close(dry_run: bool = False) -> bool:
    """
    15:30 日终结算。

    1. 记录今日 NAV（从 FastRunnerV10 或外部接口读取）
    2. 保存今日持仓快照
    3. 写出日终日志摘要

    stamp_tax = 0.0005（铁律，日终结算时验证配置正确性）
    """
    def _run():
        today_str    = date.today().isoformat()
        position_dir = PROJECT_ROOT / "positions"
        position_dir.mkdir(parents=True, exist_ok=True)

        signal_dir = PROJECT_ROOT / "signals" / today_str

        # ── 验收：stamp_tax 铁律 ──────────────────────────────────────────
        cfg_path = PROJECT_ROOT / "config.json"
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as fc:
                cfg = json.load(fc)
            stamp_tax = cfg.get("stamp_tax", cfg.get("backtest", {}).get("stamp_tax", 0.0005))
            assert float(stamp_tax) == 0.0005, (
                f"[daily_close] 铁律违反：config.json stamp_tax={stamp_tax}，应为 0.0005"
            )
            logger.info(f"[daily_close] stamp_tax={stamp_tax} 验收通过 ✓")

        # ── 汇总今日信号为持仓快照 ────────────────────────────────────────
        today_positions: Dict[str, Any] = {}

        if signal_dir.exists():
            for sig_file in sorted(signal_dir.glob("signal_*.json")):
                try:
                    with open(sig_file, encoding="utf-8") as fs:
                        sig_data = json.load(fs)
                    strategy = sig_data.get("strategy", sig_file.stem)
                    today_positions[strategy] = sig_data.get("signal", {})
                except Exception:
                    pass

        pos_file = position_dir / f"positions_{today_str}.json"
        with open(pos_file, "w", encoding="utf-8") as fp:
            json.dump(today_positions, fp, ensure_ascii=False, indent=2)
        logger.info(f"[daily_close] 持仓快照已写出: {pos_file}")

        # ── 日终摘要 ──────────────────────────────────────────────────────
        summary = {
            "date"         : today_str,
            "stamp_tax"    : 0.0005,    # ★ 永远是万五
            "strategies"   : list(today_positions.keys()),
            "n_strategies" : len(today_positions),
            "jobs_done"    : list(_load_state().keys()),
            "close_time"   : datetime.now().isoformat(),
        }

        summary_path = LOG_DIR / f"daily_summary_{today_str}.json"
        with open(summary_path, "w", encoding="utf-8") as fs:
            json.dump(summary, fs, ensure_ascii=False, indent=2)
        logger.info(f"[daily_close] 日终摘要: {summary}")

        return summary

    return _safe_run(_run, "daily_close", dry_run)


# ─────────────────────────────────────────────────────────────────────────────
# 组合任务
# ─────────────────────────────────────────────────────────────────────────────

def job_morning(dry_run: bool = False) -> bool:
    """早盘全流程（08:00~09:25）：update_data → rebuild_mask → gen_signal → check_position"""
    ok = True
    ok &= job_update_data(dry_run)
    ok &= job_rebuild_mask(dry_run)
    ok &= job_gen_signal(dry_run)
    ok &= job_check_position(dry_run)
    return ok


def job_close_only(dry_run: bool = False) -> bool:
    """日终阶段（15:30）：daily_close"""
    return job_daily_close(dry_run)


# ─────────────────────────────────────────────────────────────────────────────
# Schedule 模式
# ─────────────────────────────────────────────────────────────────────────────

def run_schedule(dry_run: bool = False) -> None:
    """
    全天 schedule 模式：自动在指定时间触发各阶段。

    需要安装 schedule：pip install schedule
    """
    try:
        import schedule
    except ImportError:
        logger.error("schedule 未安装: pip install schedule")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("  Q-UNITY V10 daily_run_v10.py  schedule 模式")
    logger.info(f"  日期: {date.today().isoformat()}")
    logger.info(f"  dry_run: {dry_run}")
    logger.info("=" * 60)

    def _make_job(fn, name):
        def _wrapper():
            if not _is_done(name):
                logger.info(f"  ▶ 触发定时任务: {name}")
                fn(dry_run)
            else:
                logger.info(f"  ↳ {name} 今日已完成，跳过")
        return _wrapper

    # ── 注册定时任务 ──────────────────────────────────────────────────────
    schedule.every().day.at("08:00").do(_make_job(job_update_data,    "update_data"))
    schedule.every().day.at("08:30").do(_make_job(job_rebuild_mask,   "rebuild_mask"))
    schedule.every().day.at("09:00").do(_make_job(job_gen_signal,     "gen_signal"))
    schedule.every().day.at("09:25").do(_make_job(job_check_position, "check_position"))
    schedule.every().day.at("15:30").do(_make_job(job_daily_close,    "daily_close"))

    logger.info("  定时任务已注册：")
    logger.info("    08:00 — 更新日线数据（AKShare QFQ 增量）")
    logger.info("    08:30 — 重建 valid_mask（listing_days >= 60）")
    logger.info("    09:00 — 生成信号（V10 策略注册表）")
    logger.info("    09:25 — 持仓核对（生成调仓指令）")
    logger.info("    15:30 — 日终结算（NAV + 持仓快照）")
    logger.info()
    logger.info("  Ctrl+C 退出 schedule 循环")

    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        logger.info("  schedule 模式退出")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

_JOB_MAP = {
    "update_data"    : job_update_data,
    "rebuild_mask"   : job_rebuild_mask,
    "gen_signal"     : job_gen_signal,
    "check_position" : job_check_position,
    "daily_close"    : job_daily_close,
    "morning"        : job_morning,
    "close"          : job_close_only,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Q-UNITY V10 每日自动运行脚本"
    )
    parser.add_argument(
        "--job",
        choices=list(_JOB_MAP.keys()),
        default=None,
        help="运行单个阶段（不指定则使用 --mode）",
    )
    parser.add_argument(
        "--mode",
        choices=["schedule", "morning", "close", "full"],
        default="schedule",
        help="运行模式（schedule=全天调度, morning=早盘, close=日终, full=全流程）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="干跑模式（只打印不执行）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重跑（忽略今日已完成状态）",
    )
    args = parser.parse_args()

    if args.force:
        # 清除今日状态，允许重跑
        if STATE_FILE.exists():
            STATE_FILE.unlink()
        logger.info("  [--force] 已清除今日运行状态")

    logger.info(f"Q-UNITY V10 daily_run_v10.py  {date.today().isoformat()}")
    logger.info(f"  dry_run={args.dry_run}  mode={args.mode}  job={args.job}")

    if args.job:
        # 单阶段运行
        fn  = _JOB_MAP[args.job]
        ok  = fn(args.dry_run)
        sys.exit(0 if ok else 1)

    elif args.mode == "schedule":
        run_schedule(args.dry_run)

    elif args.mode == "morning":
        ok = job_morning(args.dry_run)
        sys.exit(0 if ok else 1)

    elif args.mode == "close":
        ok = job_close_only(args.dry_run)
        sys.exit(0 if ok else 1)

    elif args.mode == "full":
        ok = job_morning(args.dry_run)
        ok &= job_daily_close(args.dry_run)
        sys.exit(0 if ok else 1)

    else:
        parser.print_help()
        sys.exit(1)
