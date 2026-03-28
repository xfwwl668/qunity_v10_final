"""
scripts/step4_build_concept_npy.py

将概念板块 CSV 映射转为 Q-UNITY 的 npy 哑变量矩阵。

输出:
  data/npy/concept_matrix.npy     shape=(n_stocks, n_concepts)  dtype=uint8
  data/npy/concept_names.json     概念名称列表（与矩阵列一一对应）

用法:
  python scripts/step4_build_concept_npy.py
  python scripts/step4_build_concept_npy.py --min-stocks 10
  python scripts/step4_build_concept_npy.py --max-concepts 400
"""

import json
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONCEPT_DIR = DATA_DIR / "concepts"
# [FIX-C-01]
try:
    import sys as _s; _s.path.insert(0, str(Path(__file__).parent))
    from utils_paths import get_npy_dir  # type: ignore
except Exception:
    def get_npy_dir(v="v10"): return Path(__file__).parent.parent / "data/npy_v10"
NPY_DIR = get_npy_dir("v10")
META_PATH = NPY_DIR / "meta.json"


def _build_concept_ids(
    matrix: np.ndarray,        # (N, C) uint8 概念哑变量
    concept_names: list[str],  # 长度=C
    n_stocks: int,
    n_dates: int | None,       # None→从 meta.json 读取
) -> None:
    """
    [BUG-CONCEPT-IDS-MISSING FIX]
    将 concept_matrix (N, C) 转换为 concept_ids (N, T) uint16。

    规则：
    - 每只股票取「成分股最多」的概念作为主概念（tie-break 取 index 更小的）
    - concept_id = 概念在 concept_names 中的索引 + 1（0 保留给"无概念"）
    - 沿时间轴广播为 (N, T)，填充到 data/npy/concept_ids.npy
    """
    meta_path = NPY_DIR / "meta.json"
    if not meta_path.exists():
        print("  ⚠ meta.json 不存在，跳过 concept_ids.npy 生成（请先运行 build_npy）")
        return

    if n_dates is None:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        T = meta["shape"][1]
    else:
        T = n_dates

    N, C = matrix.shape

    # 每只股票的主概念 ID（0=无概念）
    concept_id_1d = np.zeros(N, dtype=np.uint16)
    concept_sizes = matrix.sum(axis=0)  # 每个概念的成分股数量，shape=(C,)

    for i in range(N):
        row = matrix[i]  # 该股票所属概念的 one-hot，(C,)
        if not row.any():
            concept_id_1d[i] = 0  # 无概念标签
            continue
        # 取「成分股最多的概念」（能让 snma 板块共振更有意义）
        # 候选：该股属于的所有概念
        cands = np.where(row > 0)[0]
        best  = cands[np.argmax(concept_sizes[cands])]
        concept_id_1d[i] = int(best) + 1  # 1-indexed，0 保留给无概念

    # 广播到 (N, T)
    concept_ids = np.broadcast_to(concept_id_1d[:, None], (N, T)).copy()

    out_path = NPY_DIR / "concept_ids.npy"
    np.save(str(out_path), concept_ids.astype(np.uint16))

    has_concept = int((concept_id_1d > 0).sum())
    unique_ids  = int(np.unique(concept_id_1d[concept_id_1d > 0]).size)
    print(f"  ✓ concept_ids.npy  shape={concept_ids.shape}  "
          f"有概念={has_concept}/{N}只  使用{unique_ids}个不同概念ID")
    print(f"    （静态广播：每只股票取成分股最多的主概念，沿时间轴不变）")


def build_concept_npy(min_stocks: int = 5, max_concepts: int = 600) -> None:
    print("=" * 70)
    print(" Step 4: 构建概念板块 npy 矩阵")
    print("=" * 70)

    # ── 1. 加载 meta ──
    if not META_PATH.exists():
        print(f"✗ 未找到 {META_PATH}")
        sys.exit(1)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    codes = [str(c).zfill(6) for c in meta["codes"]]
    code_to_idx = {c: i for i, c in enumerate(codes)}
    n_stocks = len(codes)

    # ── 2. 加载映射 ──
    # [BUG-PATH FIX] 按优先级查找概念文件，兼容不同下载方式产生的不同路径和列名
    CANDIDATE_PATHS = [
        CONCEPT_DIR / "concept_mapping.csv",          # step2 脚本产生
        DATA_DIR / "industry" / "ths_map.csv",        # main.py pywencai/akshare 产生
        CONCEPT_DIR / "concept_for_qunity.csv",       # step2 manual 模式产生
    ]
    mapping_path = next((p for p in CANDIDATE_PATHS if p.exists()), None)
    if mapping_path is None:
        print("✗ 未找到概念映射文件，已查找路径:")
        for p in CANDIDATE_PATHS:
            print(f"    {p}")
        sys.exit(1)
    print(f"  使用概念文件: {mapping_path.relative_to(PROJECT_ROOT)}")

    df = pd.read_csv(str(mapping_path), dtype={"code": str})
    df.columns = df.columns.str.strip()
    df["code"] = df["code"].str.zfill(6)

    # 统一列名：兼容 concept/concept_name/板块名称 等不同格式
    if "concept" not in df.columns:
        alt = next((c for c in df.columns if "concept" in c.lower() or "板块" in c or "名称" in c), None)
        if alt:
            df = df.rename(columns={alt: "concept"})
        else:
            print(f"  ✗ 无法识别概念列，可用列: {df.columns.tolist()}")
            sys.exit(1)

    df = df[df["code"].isin(code_to_idx)]
    print(f"  映射数据: {len(df)} 条 | {df['concept'].nunique()} 个概念 | "
          f"{df['code'].nunique()} 只股票")

    # ── 3. 过滤概念（最小成分股数） ──
    concept_counts = df["concept"].value_counts()
    valid_concepts = concept_counts[concept_counts >= min_stocks].index.tolist()

    if len(valid_concepts) > max_concepts:
        print(f"  概念数 {len(valid_concepts)} > {max_concepts}，"
              f"取成分股最多的 {max_concepts} 个")
        valid_concepts = valid_concepts[:max_concepts]

    df = df[df["concept"].isin(valid_concepts)]
    concept_names = sorted(valid_concepts)
    concept_to_idx = {c: i for i, c in enumerate(concept_names)}
    n_concepts = len(concept_names)

    print(f"  过滤后: {n_concepts} 个概念 (≥ {min_stocks} 只成分股)")

    # ── 4. 构建哑变量矩阵 ──
    matrix = np.zeros((n_stocks, n_concepts), dtype=np.uint8)

    for _, row in df.iterrows():
        si = code_to_idx.get(str(row["code"]).zfill(6))
        ci = concept_to_idx.get(row["concept"])
        if si is not None and ci is not None:
            matrix[si, ci] = 1

    # ── 5. 统计 ──
    stocks_with_any = int(np.sum(matrix.any(axis=1)))
    avg_per_stock = float(matrix.sum(axis=1).mean())
    print(f"\n  矩阵形状: {matrix.shape}")
    print(f"  有概念标签: {stocks_with_any}/{n_stocks} "
          f"({stocks_with_any / n_stocks * 100:.1f}%)")
    print(f"  每股平均概念数: {avg_per_stock:.1f}")

    top10 = np.argsort(matrix.sum(axis=0))[::-1][:10]
    print(f"\n  Top-10 概念（按成分股数）:")
    for idx in top10:
        print(f"    {concept_names[idx]}: {int(matrix[:, idx].sum())} 只")

    # ── 6. 保存 ──
    cm_path = NPY_DIR / "concept_matrix.npy"
    cn_path = NPY_DIR / "concept_names.json"

    np.save(str(cm_path), matrix)
    with open(cn_path, "w", encoding="utf-8") as f:
        json.dump(concept_names, f, ensure_ascii=False, indent=2)

    print(f"\n  ✓ concept_matrix.npy  → {NPY_DIR}")
    print(f"  ✓ concept_names.json  → {NPY_DIR}")

    # ── 7. 生成 concept_ids.npy (N, T) ──────────────────────────────────────
    # [BUG-CONCEPT-IDS-MISSING FIX]
    # fast_runner 和策略（snma_v4, titan_alpha_v1）需要 concept_ids (N, T) 时序矩阵，
    # 其中每个 (i, t) 存储股票 i 在 t 日的「主概念 ID」（uint16）。
    # 原 step4 只生成静态 concept_matrix (N, C)，无法满足需求。
    #
    # 生成方法：取每只股票「成分股最多的概念」作为主概念 ID，沿时间轴静态广播。
    # 说明：
    # - 概念归属在实际中变化极少（偶尔调整），静态广播对回测无实质影响
    # - concept_ids=0 表示「未分类」（无任何概念标签的股票）
    # - 精确动态 concept_ids 需要历史概念调整记录，目前无来源，此方案为最优近似
    _build_concept_ids(matrix, concept_names, n_stocks, n_dates=None)

    print(f"\n✓ Step 4 完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4: 概念板块 CSV → npy")
    parser.add_argument("--min-stocks", type=int, default=5,
                        help="概念最少成分股数量（默认5）")
    parser.add_argument("--max-concepts", type=int, default=600,
                        help="最多保留概念数（默认600）")
    args = parser.parse_args()

    build_concept_npy(min_stocks=args.min_stocks, max_concepts=args.max_concepts)



