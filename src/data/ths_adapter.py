# Q-UNITY V8 ULTIMATE — 同花顺(THS)板块适配层
"""
ths_adapter.py — 同花顺概念板块数据适配器

功能：
  1. load_ths_concepts: 读取 data/industry/ths_map.csv（字段: code, concept_name）
  2. ConceptEncoder: 概念名称 → uint16 整数双向映射
  3. build_concept_id_matrix: 将概念 ID 对齐广播至整个回测时间轴 (N, T) uint16

CSV 格式约定（data/industry/ths_map.csv）：
  code,concept_name
  000001,大数据
  000001,人工智能
  600519,白酒
  ...

每只股票可属于多个概念，本模块取"第一个出现"的概念为主概念。
若一只股票无概念数据，其 concept_ids 行填充为 0（表示"无概念"）。

前视偏差防护：
  概念数据通常是静态元数据（不随时间变化），因此在整个时间轴上广播。
  如需支持动态概念变化，可将 CSV 扩展为 (code, concept_name, effective_date) 格式。

使用示例::

    from src.data.ths_adapter import load_ths_concepts, ConceptEncoder, build_concept_id_matrix

    encoder = ConceptEncoder()
    concept_map = load_ths_concepts("data/industry/ths_map.csv", encoder)
    concept_ids = build_concept_id_matrix(codes, n_days=T, concept_map=concept_map)
    # concept_ids: (N, T) np.ndarray uint16
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 保留 ID=0 为"无概念"占位符，编码从 1 开始
_NO_CONCEPT_ID: int = 0
_MAX_UINT16: int = 65535


class ConceptEncoder:
    """
    同花顺概念名称 ↔ uint16 整数 双向映射。

    特性：
      - 概念 ID 从 1 开始（0 保留为"无概念/未知"）
      - 线程不安全（单线程构建，之后只读）
      - 支持最多 65534 个不同概念（uint16 上限 - 1）

    Attributes
    ----------
    name2id : Dict[str, int]
        概念名称 → uint16 ID
    id2name : Dict[int, str]
        uint16 ID → 概念名称
    """

    def __init__(self) -> None:
        self.name2id: Dict[str, int] = {}
        self.id2name: Dict[int, str] = {_NO_CONCEPT_ID: "__NO_CONCEPT__"}
        self._next_id: int = 1

    def encode(self, name: str) -> int:
        """
        将概念名称编码为 uint16 ID。

        若名称已存在则返回已有 ID；否则分配新 ID。
        超过 uint16 上限（65534）时抛出 ValueError。

        Parameters
        ----------
        name : str
            概念名称（会做 strip() 去除首尾空白）

        Returns
        -------
        int  (uint16 范围: 1 ~ 65534)
        """
        name = name.strip()
        if not name:
            return _NO_CONCEPT_ID
        if name in self.name2id:
            return self.name2id[name]
        if self._next_id > _MAX_UINT16 - 1:
            raise ValueError(
                f"ConceptEncoder: 概念数量超过 uint16 上限 ({_MAX_UINT16 - 1})，"
                f"当前尝试添加: '{name}'"
            )
        cid = self._next_id
        self.name2id[name] = cid
        self.id2name[cid] = name
        self._next_id += 1
        return cid

    def decode(self, cid: int) -> str:
        """
        将 uint16 ID 解码为概念名称。

        Parameters
        ----------
        cid : int

        Returns
        -------
        str  概念名称，若未知返回 "__UNKNOWN__"
        """
        return self.id2name.get(cid, "__UNKNOWN__")

    def __len__(self) -> int:
        """已注册的概念数量（不含 ID=0 占位符）"""
        return self._next_id - 1

    def n_concepts(self) -> int:
        """同 __len__"""
        return len(self)


def load_ths_concepts(
    csv_path: str,
    encoder: Optional[ConceptEncoder] = None,
    primary_only: bool = True,
) -> Tuple[Dict[str, int], ConceptEncoder]:
    """
    加载同花顺概念板块 CSV，返回 {股票代码: 主概念 ID} 映射。

    CSV 格式：
        code,concept_name
        000001,大数据
        000001,人工智能
        600519,白酒

    Parameters
    ----------
    csv_path : str
        CSV 文件路径（data/industry/ths_map.csv）
    encoder : ConceptEncoder, optional
        外部传入的编码器（用于复用已有编码器），若为 None 则新建
    primary_only : bool
        True = 每只股票只取首个出现的概念（主概念）
        False = 所有概念（此时返回值为 {code: List[int]}，暂不支持，保留扩展接口）

    Returns
    -------
    concept_map : Dict[str, int]
        {stock_code: primary_concept_id}
        无概念或不在 CSV 中的股票对应 ID=0
    encoder : ConceptEncoder
        已更新的编码器（可继续用于编码新概念）

    Raises
    ------
    FileNotFoundError
        若 CSV 文件不存在
    ValueError
        若 CSV 文件缺少必要字段（code 或 concept_name）
    """
    if encoder is None:
        encoder = ConceptEncoder()

    csv_path_obj = Path(csv_path)
    if not csv_path_obj.exists():
        raise FileNotFoundError(
            f"THS 概念 CSV 不存在: {csv_path}\n"
            f"请将概念映射文件放置于 data/industry/ths_map.csv\n"
            f"CSV 格式: code,concept_name (每行一个股票-概念对)"
        )

    concept_map: Dict[str, int] = {}
    n_rows = 0
    n_skipped = 0

    try:
        with open(csv_path_obj, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV 文件为空或无法解析表头: {csv_path}")

            # 检查必要字段
            fieldnames_lower = [fn.strip().lower() for fn in reader.fieldnames]
            if "code" not in fieldnames_lower or "concept_name" not in fieldnames_lower:
                raise ValueError(
                    f"CSV 缺少必要字段。需要: code, concept_name\n"
                    f"实际字段: {reader.fieldnames}\n"
                    f"文件: {csv_path}"
                )

            # 重建字段名映射（大小写不敏感）
            fn_map = {fn.strip().lower(): fn.strip() for fn in reader.fieldnames}
            code_col    = fn_map["code"]
            concept_col = fn_map["concept_name"]

            for row in reader:
                n_rows += 1
                code = row.get(code_col, "").strip()
                concept_name = row.get(concept_col, "").strip()

                if not code or not concept_name:
                    n_skipped += 1
                    continue

                # 标准化股票代码（去前缀/后缀，保留纯数字部分）
                code_clean = _normalize_code(code)

                if primary_only:
                    # 只记录首次出现的概念作为主概念
                    if code_clean not in concept_map:
                        cid = encoder.encode(concept_name)
                        concept_map[code_clean] = cid
                else:
                    # 扩展：未来支持多概念
                    cid = encoder.encode(concept_name)
                    if code_clean not in concept_map:
                        concept_map[code_clean] = cid

    except UnicodeDecodeError:
        # 尝试 GBK 编码（部分同花顺导出文件）
        logger.warning(f"UTF-8 解码失败，尝试 GBK 编码: {csv_path}")
        with open(csv_path_obj, newline="", encoding="gbk") as f:
            reader = csv.DictReader(f)
            fn_map = {fn.strip().lower(): fn.strip() for fn in (reader.fieldnames or [])}
            code_col    = fn_map.get("code", "code")
            concept_col = fn_map.get("concept_name", "concept_name")
            for row in reader:
                code = row.get(code_col, "").strip()
                concept_name = row.get(concept_col, "").strip()
                if not code or not concept_name:
                    continue
                code_clean = _normalize_code(code)
                if code_clean not in concept_map:
                    concept_map[code_clean] = encoder.encode(concept_name)

    logger.info(
        f"[THS] 加载完毕: {csv_path} | "
        f"总行数={n_rows} 跳过={n_skipped} 有效股票={len(concept_map)} "
        f"概念数={len(encoder)}"
    )

    return concept_map, encoder


def build_concept_id_matrix(
    codes: List[str],
    n_days: int,
    concept_map: Dict[str, int],
    fill_value: int = _NO_CONCEPT_ID,
    backtest_mode: bool = False,
) -> np.ndarray:
    """
    [BUG-NEW-14 / BUG-NEW-32 NOTE] 当前实现每只股票在整个时间轴上只存储一个主概念 ID
    (uint16)，多概念信息被丢弃。对于同属多个热门概念的股票，板块共振信号可能被低估。
    当前版本限制：取股票首个出现的概念为主概念（单值）。
    路线图：未来版本计划扩展为 (N, T, K) 多概念矩阵（K上限8），支持多概念交叉共振。
    将股票代码 → 概念 ID 映射广播至完整时间轴，生成 (N, T) uint16 矩阵。

    由于同花顺概念通常为静态数据（不随时间变化），本函数将每只股票的
    主概念 ID 在整个时间轴上广播（每一天均填充相同值）。

    缺失数据处理：
      - 不在 concept_map 中的股票填充 fill_value（默认 0 = 无概念）
      - 此设计确保下游策略可安全过滤 concept_id == 0 的股票

    Parameters
    ----------
    codes : List[str]
        股票代码列表（长度 N）
    n_days : int
        时间轴长度 T
    concept_map : Dict[str, int]
        {stock_code: concept_id}，由 load_ths_concepts 生成
    fill_value : int
        缺失数据填充值（默认 0）

    Returns
    -------
    concept_ids : np.ndarray
        (N, T) dtype=uint16，每个元素为该股票在该时间点的主概念 ID
        无概念 = 0，有效概念 ID ≥ 1

    Notes
    -----
    内存占用：N=5000, T=1250 → 5000 × 1250 × 2B ≈ 12.5MB（相比 float32 节省 50%）

    前视偏差防护：
    ──────────────
    本函数产生的概念数据在时间轴上完全静态（所有时间点相同），
    不引入任何前视偏差。若需动态概念，调用者应在传入 concept_map
    之前按时间过滤，或修改本函数接受 (code, date → concept_id) 的映射。
    """
    N = len(codes)

    if n_days <= 0:
        raise ValueError(f"n_days 必须为正整数，得到: {n_days}")

    # ── [BUG-1.1 FIX] 前视偏差防护 ──────────────────────────────────────
    # ths_map.csv 是当前时间点的静态快照。将其广播到全时间轴会引入严重前视偏差：
    # 例如某股票 2023 年才被划入"人工智能"概念，但广播后 2015 年的回测就已
    # 使用该概念进行板块共振加权，导致策略收益虚高。
    # 修复：backtest_mode=True（默认）时将所有概念 ID 强制清零，
    #       使下游 compute_concept_resonance / _apply_concept_resonance
    #       跳过板块共振加权（concept_id=0 表示无概念，不参与共振）。
    # 若您拥有 Point-in-Time 的历史概念数据，请传入 backtest_mode=False。
    if backtest_mode:
        logger.warning(
            "[THS] [BUG-1.1 FIX] backtest_mode=True: 概念板块数据为当前时间点静态快照，"
            "广播至全时间轴将引入前视偏差。已强制禁用板块共振（全矩阵填充 fill_value=%d）。"
            "如需启用，请提供 Point-in-Time 历史数据并设置 backtest_mode=False。",
            fill_value,
        )
        concept_ids = np.full((N, n_days), fill_value, dtype=np.uint16)
        logger.info(
            f"[THS] 概念 ID 矩阵已清零（回测模式）: shape=({N},{n_days})"
        )
        return concept_ids

    # 构建每只股票的主概念 ID 向量 (N,)
    stock_cid = np.full(N, fill_value, dtype=np.uint16)
    n_found = 0
    for i, code in enumerate(codes):
        code_clean = _normalize_code(code)
        cid = concept_map.get(code_clean, fill_value)
        if cid != fill_value:
            n_found += 1
        stock_cid[i] = np.uint16(cid)

    # 广播至 (N, T)：每列相同
    concept_ids = np.broadcast_to(
        stock_cid[:, np.newaxis],
        (N, n_days),
    ).copy()  # copy() 保证 C-contiguous 且可写

    concept_ids = concept_ids.astype(np.uint16)

    coverage_pct = n_found / max(N, 1) * 100
    logger.info(
        f"[THS] 概念 ID 矩阵构建完毕: shape=({N},{n_days}) "
        f"覆盖率={n_found}/{N} ({coverage_pct:.1f}%)"
    )

    if coverage_pct < 30.0:
        logger.warning(
            f"[THS] 概念覆盖率仅 {coverage_pct:.1f}%，建议检查 ths_map.csv 中的股票代码格式"
        )

    return concept_ids


def compute_concept_resonance(
    whale_pulse: np.ndarray,      # (N, T) float64 — Whale Pulse 因子截面
    concept_ids: np.ndarray,      # (N, T) uint16  — 概念 ID 矩阵
    top_pct: float = 0.10,        # 前10%触发共振加权
    boost_factor: float = 1.5,    # 加权倍数
) -> np.ndarray:
    """
    板块共振计算：基于同花顺概念的 Whale_Pulse 均值加权。

    算法（逐日截面操作，严格 t 时刻数据，无前视偏差）：
    ─────────────────────────────────────────────────────────────
    For each day t:
      1. 对每个 concept_id，计算该概念内所有股票的 Whale_Pulse 均值
         → concept_avg_pulse[concept_id] = mean(whale_pulse[stocks_of_concept, t])
      2. 对所有 concept_avg_pulse 做截面排名（降序）
         → 前 top_pct 的概念视为"强势共振概念"
      3. 属于强势共振概念的个股：Alpha 原始分 × boost_factor
    ─────────────────────────────────────────────────────────────

    Parameters
    ----------
    whale_pulse : (N, T) float64
        Whale Pulse 因子矩阵（已完成截面标准化）
    concept_ids : (N, T) uint16
        概念 ID 矩阵
    top_pct : float
        共振概念比例阈值（默认前 10%）
    boost_factor : float
        共振加权倍数（默认 1.5x）

    Returns
    -------
    resonance_weight : (N, T) float64
        每只股票每日的共振加权因子（1.0 = 不加权，boost_factor = 共振加权）

    Notes
    -----
    - concept_id == 0（无概念）的股票不参与共振加权，权重保持 1.0
    - 概念内样本 < 2 只时不计算共振（避免单票噪音主导）
    - 严格基于 t 日闭盘数据，交易信号在 t+1 日执行
    """
    N, T = whale_pulse.shape
    resonance_weight = np.ones((N, T), dtype=np.float64)

    # 获取所有有效的 concept_id（排除 0）
    unique_cids = np.unique(concept_ids)
    unique_cids = unique_cids[unique_cids > 0]

    if len(unique_cids) == 0:
        logger.warning("[THS] 无有效概念 ID，跳过板块共振计算")
        return resonance_weight

    n_concepts = len(unique_cids)
    top_k = max(1, int(np.ceil(n_concepts * top_pct)))

    # [D-05-FIX] 向量化重写：消除双重Python循环 O(T×N×C) → O(T×C)
    # 原实现：外层for t，内层for cid，最内层for i，约5000×T×C次Python调用。
    # 修复：外层for t保留（截面独立），内层改为向量化np操作。
    for t in range(T):
        wp_t  = whale_pulse[:, t]    # (N,)
        cid_t = concept_ids[:, t]    # (N,)

        # 向量化计算每个概念的平均 Whale Pulse
        concept_pulse: Dict[int, float] = {}
        # 对每个cid用布尔掩码（向量化）
        cid_t_int = cid_t.astype(np.int32)
        for cid in unique_cids:
            mask = (cid_t_int == int(cid))
            if not mask.any():
                continue
            valid_vals = wp_t[mask]
            valid_vals = valid_vals[~np.isnan(valid_vals)]
            if len(valid_vals) >= 2:
                concept_pulse[int(cid)] = float(np.mean(valid_vals))

        if not concept_pulse:
            continue

        # 排序：找到前 top_k 共振概念
        sorted_concepts = sorted(concept_pulse.items(), key=lambda x: x[1], reverse=True)
        resonance_cids = np.array([cid for cid, _ in sorted_concepts[:top_k]], dtype=np.int32)

        # [D-05-FIX] np.isin 替代内层 for i 循环（O(N) → O(1) 广播）
        boost_mask = np.isin(cid_t_int, resonance_cids)
        resonance_weight[boost_mask, t] = boost_factor

    return resonance_weight


def _normalize_code(code: str) -> str:
    """
    标准化股票代码为6位纯数字格式。

    处理以下格式：
      - "000001.SZ" → "000001"
      - "sh000001"  → "000001"
      - "SH.000001" → "000001"
      - "000001"    → "000001"（不变）
    """
    code = code.strip()
    # 去除交易所后缀（.SZ, .SH, .BJ 等）
    if "." in code:
        parts = code.split(".")
        # 取纯数字部分
        for part in parts:
            if part.isdigit():
                return part.zfill(6)
    # 去除 sh/sz/bj 前缀
    code_lower = code.lower()
    for prefix in ("sh", "sz", "bj", "sh.", "sz.", "bj."):
        if code_lower.startswith(prefix):
            rest = code[len(prefix):]
            if rest.isdigit():
                return rest.zfill(6)
    # 纯数字，补零到6位
    if code.isdigit():
        return code.zfill(6)
    return code


def generate_sample_csv(output_path: str = "data/industry/ths_map_sample.csv") -> None:
    """
    生成示例 ths_map.csv（含常见A股股票的同花顺概念板块数据）。

    此示例仅用于功能测试，不代表真实市场数据。
    真实数据请从同花顺终端或第三方数据接口获取。

    Parameters
    ----------
    output_path : str
        输出文件路径
    """
    sample_data = [
        ("code", "concept_name"),
        # 人工智能概念
        ("000卷11", "人工智能"),  # placeholder, see below
    ]

    # 真实格式示例
    rows = [
        ("code", "concept_name"),
        ("600519", "白酒"),
        ("000858", "白酒"),
        ("002304", "白酒"),
        ("000568", "白酒"),
        ("600809", "白酒"),
        ("600779", "白酒"),
        ("000596", "白酒"),
        ("002915", "白酒"),
        ("300999", "人工智能"),
        ("688981", "人工智能"),
        ("300750", "新能源"),
        ("002594", "新能源"),
        ("600900", "新能源"),
        ("601012", "新能源"),
        ("300015", "创新药"),
        ("600196", "创新药"),
        ("688180", "创新药"),
        ("002415", "安防"),
        ("601888", "免税"),
        ("600036", "银行"),
        ("601318", "保险"),
        ("000001", "银行"),
        ("600016", "银行"),
        ("601398", "银行"),
        ("000002", "地产"),
        ("600048", "地产"),
        ("600031", "工程机械"),
        ("000333", "家电"),
        ("600690", "家电"),
        ("002081", "芯片"),
        ("688012", "芯片"),
        ("600745", "芯片"),
        ("002049", "芯片"),
        ("688041", "芯片"),
        ("300760", "医疗设备"),
        ("603259", "医疗设备"),
        ("300122", "医疗设备"),
        ("000725", "面板"),
        ("600183", "汽车"),
        ("000100", "汽车"),
        ("601006", "铁路"),
        ("601766", "铁路"),
        ("600628", "零售"),
        ("002711", "大数据"),
        ("300033", "大数据"),
        ("002236", "储能"),
        ("300316", "储能"),
        ("600886", "储能"),
    ]

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"[THS] 示例 CSV 已生成: {out_path}  ({len(rows)-1} 条记录)")


if __name__ == "__main__":
    # 快速功能测试
    import tempfile, os

    print("=" * 60)
    print("  THS Adapter — 功能测试")
    print("=" * 60)

    # 生成临时测试 CSV
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        f.write("code,concept_name\n")
        f.write("000001,银行\n")
        f.write("600519,白酒\n")
        f.write("000858,白酒\n")
        f.write("300750,新能源\n")
        f.write("002594,新能源\n")
        f.write("688981,人工智能\n")
        tmp_csv = f.name

    try:
        encoder = ConceptEncoder()
        concept_map, encoder = load_ths_concepts(tmp_csv, encoder)

        print(f"\n  概念映射: {concept_map}")
        print(f"  编码器大小: {len(encoder)} 个概念")
        print(f"  白酒 ID: {encoder.encode('白酒')}")
        print(f"  ID 2 解码: {encoder.decode(2)}")

        codes = ["000001", "600519", "000858", "300750", "999999"]
        concept_ids = build_concept_id_matrix(codes, n_days=5, concept_map=concept_map)

        print(f"\n  概念 ID 矩阵 shape: {concept_ids.shape}")
        print(f"  dtype: {concept_ids.dtype}")
        print(f"  矩阵内容:\n{concept_ids}")

        # 测试共振计算
        rng = np.random.default_rng(42)
        wp = rng.standard_normal((len(codes), 5))
        cid_int = concept_ids.astype(np.uint16)
        rw = compute_concept_resonance(wp, cid_int, top_pct=0.5, boost_factor=1.5)
        print(f"\n  共振权重矩阵:\n{rw}")

        print("\n  ✅ 功能测试通过！")
    finally:
        os.unlink(tmp_csv)
