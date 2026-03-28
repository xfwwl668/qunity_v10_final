"""
build_npy.py — Q-UNITY V8.0 日线 npy 矩阵构建 CLI

使用方法：
  python -m src.data.build_npy --parquet-dir ./data/daily_parquet --npy-dir ./data/npy
  python -m src.data.build_npy --parquet-dir ./data/daily_parquet --force-rebuild

详细参数：
  --parquet-dir   : Parquet 文件所在目录（每只股票一个 .parquet 文件）
  --npy-dir       : 输出 npy 文件目录（默认 ./data/npy）
  --force-rebuild : 强制重建（即使已存在）
  --workers       : 并行读取线程数（默认 8）
  --min-valid-rows: NB-21 保护参数（默认 300）
  --verify-sha256 : 构建后验证文件完整性

构建成功后会输出：
  data/npy/
    ├── close.npy
    ├── open.npy
    ├── high.npy
    ├── low.npy
    ├── volume.npy
    ├── valid_mask.npy
    └── meta.json
"""
from __future__ import annotations
import os, sys
from pathlib import Path

# ── [PATH-FIX-V3] Project Root Multi-Tier Discovery ──────────────────
# 向上无限溯源直到找到包含 src 的根目录，解决子进程启动时的路径断裂
_cur = Path(__file__).resolve()
_root = _cur.parent
for _ in range(5):
    if (_root / "src").exists() and (_root / "config.json").exists():
        break
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
# ───────────────────────────────────────────────────────────────────────

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_npy")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Q-UNITY V8.0 — 构建日线 npy 矩阵（ColumnarDataAdapter.build）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--parquet-dir",
        type=str,
        required=True,
        help="Parquet 文件所在目录（每只股票一个 .parquet 文件）",
    )
    parser.add_argument(
        "--npy-dir",
        type=str,
        default="./data/npy",
        help="输出 npy 文件目录（默认 ./data/npy）",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        default=False,
        help="强制重建（即使 meta.json 已存在）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="并行读取 Parquet 的线程数（默认 8）",
    )
    parser.add_argument(
        "--min-valid-rows",
        type=int,
        default=100,  # [BUG-MIN-VALID-ROWS-MISMATCH FIX] 与 columnar_adapter.MIN_RSRS_VALID_ROWS=100 对齐
        help="NB-21 新股保护：累积有效交易日少于此值的信号全部屏蔽（默认100，约0.4年）",
    )
    parser.add_argument(
        "--verify-sha256",
        action="store_true",
        default=False,
        help="构建后对每个 npy 文件验证 SHA-256 完整性",
    )
    parser.add_argument(
        "--adj-policy",
        type=str,
        choices=["strict", "tag_only", "skip"],
        default="tag_only",
        help="前复权混入处理策略：strict=拒绝写入，tag_only=标记，skip=不检测（默认 tag_only）",
    )
    parser.add_argument(
        "--no-strict-a-stock",
        action="store_true",
        default=False,
        help="[V9] 跳过 A 股标的纯净化过滤（仅调试用，生产环境不要使用）",
    )
    parser.add_argument(
        "--amount-col",
        type=str,
        default=None,
        help="[V9] 指定成交额列名（默认自动检测 amount/turnover/money）",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """CLI 入口，返回退出码（0=成功，1=失败）。"""
    args = parse_args(argv)

    parquet_dir = Path(args.parquet_dir)
    npy_dir     = Path(args.npy_dir)

    if not parquet_dir.exists():
        logger.error(f"Parquet 目录不存在: {parquet_dir}")
        return 1

    # 导入 ColumnarDataAdapter
    try:
        from src.data.columnar_adapter import ColumnarDataAdapter
    except ImportError as e:
        logger.error(f"无法导入 ColumnarDataAdapter: {e}")
        logger.error("请确认项目根目录在 PYTHONPATH 中，或在项目根目录运行此命令。")
        return 1

    # 构建配置（透传 CLI 参数）
    config = {
        "data_adj_policy": {
            "daily": {
                "reject_qfq": args.adj_policy == "strict",
                "tag_suspect": args.adj_policy in ("strict", "tag_only"),
            }
        }
    }

    logger.info(f"=== Q-UNITY V8 build_npy 开始 ===")
    logger.info(f"  parquet_dir  : {parquet_dir}")
    logger.info(f"  npy_dir      : {npy_dir}")
    logger.info(f"  force_rebuild: {args.force_rebuild}")
    logger.info(f"  workers      : {args.workers}")
    logger.info(f"  min_valid_rows: {args.min_valid_rows}")
    logger.info(f"  adj_policy   : {args.adj_policy}")
    logger.info(f"  strict_a_stock: {not args.no_strict_a_stock} (V9 纯净化)")

    t0 = time.perf_counter()

    try:
        adapter = ColumnarDataAdapter(
            parquet_dir=str(parquet_dir),
            npy_dir=str(npy_dir),
            n_workers=args.workers,
            min_valid_rows=args.min_valid_rows,
            config=config,
            strict_a_stock=not args.no_strict_a_stock,  # [V9] 默认开启纯净化
        )

        meta = adapter.build(force_rebuild=args.force_rebuild)
        N, T = meta.shape

        elapsed = time.perf_counter() - t0
        logger.info(f"=== 构建完成 ===")
        logger.info(f"  股票数 N = {N}")
        logger.info(f"  交易日 T = {T}")
        logger.info(f"  耗时   = {elapsed:.1f}s")
        logger.info(f"  输出   = {npy_dir}")

        if args.verify_sha256:
            logger.info("  正在验证 SHA-256 完整性...")
            _, info = adapter.load(verify_sha256=True)
            logger.info("  SHA-256 验证通过 ✓")

        # 打印输出文件列表
        for npy_file in sorted(npy_dir.glob("*.npy")):
            size_mb = npy_file.stat().st_size / 1024 / 1024
            logger.info(f"  {npy_file.name:<20s} {size_mb:.1f} MB")

        logger.info("  meta.json            ←─ FastStrategyRunner 加载入口")

    except FileNotFoundError as e:
        logger.error(f"文件不存在: {e}")
        return 1
    except Exception as e:
        logger.error(f"构建失败: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())



