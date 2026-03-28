"""
scripts/step2_download_concepts.py

同花顺概念板块数据获取器（四模式设计）

架构说明:
  概念数据变化慢（月度级别），推荐一次手动准备后长期复用。
  四种模式优先级: manual > adata > pywencai > akshare

  [推荐] manual  : 从现有 CSV 文件导入（零依赖，稳定，推荐日常使用）
  [新增] adata   : adata THS 接口（无封禁，断点续传，推荐自动下载）
  [可选] pywencai: 使用 PyWencai（需 Node.js + pip install pywencai）
  [备用] akshare : 爬同花顺/东方财富（存在封IP风险，作为最后备用）

  adata 接口：
    · all_concept_code_ths()           → 391个 THS 概念列表
    · concept_constituent_ths(index_code) → 每个概念成分股
    · 实测可用，断点续传，约10~20分钟完成

输入文件格式 (manual 模式):
  data/concepts/ths_map.csv  或  data/industry/ths_map.csv
  格式: code,concept_name
        000001,大数据
        000001,金融科技
        600519,白酒

输出:
  data/concepts/concept_mapping.csv      概念↔股票映射（长表）
  data/concepts/concept_for_qunity.csv   每只股票的概念标签（聚合）

用法:
  python scripts/step2_download_concepts.py                  # 自动检测模式
  python scripts/step2_download_concepts.py --mode adata     # adata模式（推荐）
  python scripts/step2_download_concepts.py --mode manual    # 强制手动CSV模式
  python scripts/step2_download_concepts.py --mode pywencai  # PyWencai模式
  python scripts/step2_download_concepts.py --mode akshare   # AKShare模式（备用）
  python scripts/step2_download_concepts.py --mode skip      # 跳过（无概念数据）
"""

import os as _os
# [ADATA] 禁代理（必须在所有 import 之前）
for _k in ["HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy"]:
    _os.environ.pop(_k, None)
_os.environ["NO_PROXY"] = "*"
_os.environ["no_proxy"] = "*"
try:
    import requests as _req
    _orig_req2 = _req.Session.request
    def _noproxy_req2(self, *args, **kw):
        kw["proxies"] = {"http": "", "https": "", "no": "*"}
        kw.setdefault("timeout", 30)
        return _orig_req2(self, *args, **kw)
    _req.Session.request = _noproxy_req2
except Exception:
    pass

import sys
import json
import time
import random
import argparse
import traceback
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
CONCEPT_DIR  = DATA_DIR / "concepts"
META_PATH = DATA_DIR / "npy_v10" / "meta.json"  # [FIX-C-01]

# 手动 CSV 的可能路径（按优先级顺序）
MANUAL_CSV_CANDIDATES = [
    DATA_DIR / "concepts" / "ths_map.csv",
    DATA_DIR / "industry" / "ths_map.csv",
    DATA_DIR / "industry" / "ths_map_sample.csv",
    DATA_DIR / "concepts" / "concept_mapping.csv",
]


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def load_meta_codes() -> set:
    if not META_PATH.exists():
        return set()
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return set(str(c).zfill(6) for c in meta["codes"])


def build_qunity_format(mapping_df: pd.DataFrame, meta_codes: set) -> pd.DataFrame:
    """
    将长表 (concept, code) 聚合为 Q-UNITY 格式:
    code, concepts, concept_count
    """
    mapping_df = mapping_df.copy()
    mapping_df["code"] = mapping_df["code"].astype(str).str.zfill(6)
    mapping_df = mapping_df[mapping_df["code"].isin(meta_codes)]

    df_agg = (
        mapping_df.groupby("code")
        .agg(
            concepts     = ("concept", lambda x: "|".join(sorted(x.dropna().unique()))),
            concept_count= ("concept", "nunique"),
        )
        .reset_index()
    )

    # 补齐所有股票（无概念的填空）
    df_all = pd.DataFrame({"code": sorted(meta_codes)})
    df_agg = df_all.merge(df_agg, on="code", how="left")
    df_agg["concepts"]      = df_agg["concepts"].fillna("")
    df_agg["concept_count"] = df_agg["concept_count"].fillna(0).astype(int)

    out = CONCEPT_DIR / "concept_for_qunity.csv"
    df_agg.to_csv(out, index=False, encoding="utf-8-sig")

    has_concept = (df_agg["concept_count"] > 0).sum()
    print(f"✓ Q-UNITY 格式: {has_concept}/{len(df_agg)} 只有概念标签 "
          f"(平均 {df_agg['concept_count'].mean():.1f} 个/只) → {out}")
    return df_agg


# ─────────────────────────────────────────────────────────────────────────────
# 模式1：手动 CSV 导入（推荐）
# ─────────────────────────────────────────────────────────────────────────────

def run_manual_mode(csv_path: Path = None) -> pd.DataFrame:
    """
    从手动准备的 CSV 文件导入概念映射。

    支持两种 CSV 格式:
      格式A (ths_map 格式): code,concept_name
      格式B (mapping 格式): concept,code[,name]
    """
    # 自动寻找 CSV
    if csv_path is None:
        for candidate in MANUAL_CSV_CANDIDATES:
            if candidate.exists():
                csv_path = candidate
                print(f"  找到手动 CSV: {csv_path}")
                break

    if csv_path is None or not csv_path.exists():
        print("\n⚠ 未找到手动 CSV 文件。")
        print("\n  请准备以下格式的 CSV 文件并放置到 data/concepts/ths_map.csv:")
        print("  ─────────────────────────────────")
        print("  code,concept_name")
        print("  000001,大数据")
        print("  000001,金融科技")
        print("  600519,白酒")
        print("  ─────────────────────────────────")
        print("\n  获取概念数据的方法:")
        print("  1. 同花顺: 概念板块 → 右键导出 → 整理为上述格式")
        print("  2. PyWencai: python scripts/step2_download_concepts.py --mode pywencai")
        print("  3. 使用样本数据: cp data/industry/ths_map_sample.csv data/concepts/ths_map.csv")
        print("  4. 跳过: python scripts/step2_download_concepts.py --mode skip")
        return pd.DataFrame()

    df = pd.read_csv(csv_path, dtype=str)
    df.columns = df.columns.str.strip()

    # 自动检测格式
    if "concept_name" in df.columns:
        # 格式A: code, concept_name
        df = df.rename(columns={"concept_name": "concept"})
    elif "concept" in df.columns and "code" in df.columns:
        # 格式B: concept, code
        pass
    else:
        print(f"  ✗ 无法识别 CSV 格式，列: {df.columns.tolist()}")
        print("    期望格式: code,concept_name 或 concept,code")
        return pd.DataFrame()

    df["code"] = df["code"].str.zfill(6)
    df = df[["concept", "code"]].dropna()
    df = df.drop_duplicates()

    print(f"  ✓ 加载 {len(df)} 条映射: "
          f"{df['code'].nunique()} 只股票 × {df['concept'].nunique()} 个概念")

    # 保存标准化 mapping
    out = CONCEPT_DIR / "concept_mapping.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 模式2：PyWencai（比 AKShare 更稳定的自动化方案）
# ─────────────────────────────────────────────────────────────────────────────

def run_pywencai_mode() -> pd.DataFrame:
    """
    使用 PyWencai 获取同花顺概念数据。

    依赖:
      pip install pywencai
      (还需要 Node.js v14+，pywencai 内部使用 js2py)

    注意:
      - 需要同花顺账号（免费）
      - 每次查询有频率限制，内置 0.5s 间隔
    """
    try:
        import pywencai
    except ImportError:
        print("✗ pywencai 未安装")
        print("  安装: pip install pywencai")
        print("  依赖: Node.js v14+")
        return pd.DataFrame()

    print("  使用 PyWencai 获取概念板块数据...")

    # 获取所有概念列表
    try:
        df_all = pywencai.get(query="所有概念板块", query_type="stock_board")
        if df_all is None or df_all.empty:
            print("  ✗ 获取概念列表失败")
            return pd.DataFrame()
    except Exception as e:
        print(f"  ✗ PyWencai 查询失败: {e}")
        return pd.DataFrame()

    # 提取概念名称
    name_col = next((c for c in df_all.columns if "概念" in c or "板块" in c), None)
    if name_col is None:
        print(f"  ⚠ 无法识别概念列，可用列: {df_all.columns.tolist()}")
        return pd.DataFrame()

    concepts = df_all[name_col].dropna().unique().tolist()
    print(f"  共 {len(concepts)} 个概念，开始获取成分股...")

    all_rows = []
    failed  = []
    CONCEPT_DIR.mkdir(parents=True, exist_ok=True)

    for i, concept in enumerate(concepts):
        try:
            df_cons = pywencai.get(
                query=f"{concept}概念股",
                query_type="stock"
            )
            if df_cons is None or df_cons.empty:
                failed.append(concept)
                continue

            code_col = next(
                (c for c in df_cons.columns if c in ("代码", "股票代码", "code")),
                None
            )
            if code_col is None:
                failed.append(concept)
                continue

            for code in df_cons[code_col].dropna():
                all_rows.append({"concept": concept, "code": str(code).zfill(6)})

        except Exception as e:
            failed.append(concept)
            if i % 20 == 0:
                print(f"  [{i+1}/{len(concepts)}] 已收集 {len(all_rows)} 条，"
                      f"失败 {len(failed)} 个")

        time.sleep(0.5)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(concepts)}] 已收集 {len(all_rows)} 条，"
                  f"失败 {len(failed)} 个")

    df = pd.DataFrame(all_rows).drop_duplicates()
    out = CONCEPT_DIR / "concept_mapping.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"  ✓ {len(df)} 条映射 → {out}  (失败 {len(failed)} 个概念)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 模式3：AKShare（备用，存在封IP风险）
# ─────────────────────────────────────────────────────────────────────────────

def run_akshare_mode(source: str = "auto", delay: float = 0.4) -> pd.DataFrame:
    """
    使用 AKShare 获取概念数据（最后备用，存在封IP风险）。

    source: 'auto'(先ths后em), 'ths', 'em'
    delay:  请求间隔（秒），建议 0.4~1.0
    """
    try:
        import akshare as ak
    except ImportError:
        print("✗ akshare 未安装: pip install akshare --upgrade")
        return pd.DataFrame()

    # 数据源配置
    # [BUG-NEW-03 FIX] AKShare ≥1.12 返回 'name'/'code' 列，旧版返回 '概念名称'/'板块名称'。
    # 改为候选列表，按顺序找第一个存在的列名，同时兼容新旧版本。
    SOURCES = {
        "ths": {
            "list_func":   "stock_board_concept_name_ths",
            "cons_func":   "stock_board_concept_cons_ths",
            "name_col_candidates": ["概念名称", "name", "concept_name", "board_name", "名称"],
        },
        "em": {
            "list_func":   "stock_board_concept_name_em",
            "cons_func":   "stock_board_concept_cons_em",
            "name_col_candidates": ["板块名称", "name", "概念名称", "board_name", "名称"],
        },
    }

    # 选择数据源
    sources_to_try = ["ths", "em"] if source == "auto" else [source]
    df_list = None
    active_src = None

    for src in sources_to_try:
        cfg = SOURCES[src]
        print(f"  尝试获取概念列表 ({cfg['list_func']})...")
        try:
            func = getattr(ak, cfg["list_func"])
            df_list = func()
            active_src = src
            print(f"  ✓ {len(df_list)} 个概念 (source={src})")
            break
        except Exception as e:
            print(f"  ✗ {cfg['list_func']} 失败: {e}")
            if src != sources_to_try[-1]:
                print("    → 降级到下一个数据源...")

    if df_list is None:
        print("  ✗ 所有数据源均失败")
        return pd.DataFrame()

    cfg = SOURCES[active_src]
    # [BUG-NEW-03 FIX] 按候选列表顺序找第一个存在的列名
    name_col = next((c for c in cfg["name_col_candidates"] if c in df_list.columns), None)
    if name_col is None:
        print(f"  ✗ 无法识别概念名称列，可用列: {df_list.columns.tolist()}")
        print(f"  提示: AKShare 版本 ≥1.12 返回 'name' 列；旧版返回 '概念名称' 列")
        return pd.DataFrame()
    print(f"  ✓ 概念名称列识别为: '{name_col}'")
    concepts = df_list[name_col].tolist()
    # [BUG-CONCEPT-TIMEOUT FIX] akshare 内部 requests 无 timeout，
    # 空闲后服务端关闭 TCP，客户端挂起 60~120s 才超时报错（表现为"断线"）。
    # 强制设置 30s timeout，断线立即抛出而非挂起。
    try:
        import requests as _req
        _orig_get  = _req.Session.get
        _orig_post = _req.Session.post
        def _patched_get(self_s, url, **kw):
            kw.setdefault("timeout", 30)
            return _orig_get(self_s, url, **kw)
        def _patched_post(self_s, url, **kw):
            kw.setdefault("timeout", 30)
            return _orig_post(self_s, url, **kw)
        _req.Session.get  = _patched_get
        _req.Session.post = _patched_post
    except Exception:
        pass

    # [BUG-AK-CONS-THS FIX] akshare 1.14+ 移除了 stock_board_concept_cons_ths，自动降级到 EM
    _cons_func_name = cfg["cons_func"]
    if not hasattr(ak, _cons_func_name):
        _em_cons = "stock_board_concept_cons_em"
        if hasattr(ak, _em_cons):
            print(f"  ⚠ {_cons_func_name} 在当前 akshare 中不存在，自动降级到 {_em_cons}")
            _cons_func_name = _em_cons
        else:
            print(f"  ✗ {_cons_func_name} 和 {_em_cons} 均不存在，请升级: pip install akshare --upgrade")
            return pd.DataFrame()
    cons_func = getattr(ak, _cons_func_name)

    # 断点续传
    cp_path      = CONCEPT_DIR / "_akshare_checkpoint.json"
    partial_path = CONCEPT_DIR / "_akshare_partial.csv"
    start_idx    = 0
    all_rows: list = []

    if cp_path.exists():
        with open(cp_path) as f:
            cp = json.load(f)
        start_idx = cp.get("next_idx", 0)
        if partial_path.exists():
            all_rows = pd.read_csv(partial_path, dtype=str).to_dict("records")
        print(f"  ↻ 断点恢复: idx={start_idx}, {len(all_rows)} 条")

    failed = []
    t0 = time.time()

    for i in range(start_idx, len(concepts)):
        concept = concepts[i]
        for attempt in range(3):
            try:
                # [BUG-AK-CONS-THS FIX] 兼容不同版本参数名
                try:
                    kwargs = {cfg.get("symbol_key", "symbol"): concept}
                    df_cons = cons_func(**kwargs)
                except TypeError:
                    df_cons = cons_func(concept)

                code_col = next(
                    (c for c in df_cons.columns if c in ("代码", "股票代码", "code")),
                    None
                )
                if code_col:
                    for code in df_cons[code_col].dropna():
                        all_rows.append({
                            "concept": concept,
                            "code"   : str(code).zfill(6),
                        })
                break
            except Exception as e:
                if attempt < 2:
                    # [BUG-CONCEPT-TIMEOUT FIX] 指数退避：5s/15s，给服务端足够恢复时间
                    wait = (attempt + 1) * 5 + attempt * 10
                    time.sleep(wait)
                else:
                    failed.append(concept)

        time.sleep(delay)

        # [BUG-CONCEPT-CP FIX] 每20个保存断点（原50）：断线时最多重下20个
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            speed   = (i + 1 - start_idx) / elapsed if elapsed > 0 else 1
            eta     = (len(concepts) - i - 1) / speed / 60
            print(f"  [{i+1}/{len(concepts)}] {len(all_rows)} 条 | ETA {eta:.0f}m")
            # 存断点
            with open(cp_path, "w") as f:
                json.dump({"next_idx": i + 1}, f)
            pd.DataFrame(all_rows).to_csv(
                partial_path, index=False, encoding="utf-8-sig"
            )

    df = pd.DataFrame(all_rows).drop_duplicates()
    out = CONCEPT_DIR / "concept_mapping.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    # 清理断点
    for p in [cp_path, partial_path]:
        if p.exists():
            p.unlink()

    print(f"  ✓ {len(df)} 条映射 → {out}  (失败 {len(failed)} 个概念)")
    if failed:
        fail_path = CONCEPT_DIR / "failed_concepts.txt"
        fail_path.write_text("\n".join(failed), encoding="utf-8")
        print(f"  ⚠ {len(failed)} 个失败概念 → {fail_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 模式4：跳过（生成空占位文件）
# ─────────────────────────────────────────────────────────────────────────────

def run_skip_mode(meta_codes: set) -> pd.DataFrame:
    """
    跳过概念数据获取，生成全空占位文件。
    strategy（如 SNMA-V4）在 concept_matrix=None 时会自动降级，无概念约束。
    """
    CONCEPT_DIR.mkdir(parents=True, exist_ok=True)

    # 空映射文件
    df_empty = pd.DataFrame(columns=["concept", "code"])
    df_empty.to_csv(CONCEPT_DIR / "concept_mapping.csv", index=False)

    # 空 qunity 格式
    df_all = pd.DataFrame({
        "code"         : sorted(meta_codes),
        "concepts"     : "",
        "concept_count": 0,
    })
    df_all.to_csv(CONCEPT_DIR / "concept_for_qunity.csv",
                  index=False, encoding="utf-8-sig")

    print("  ✓ 已生成空占位文件 (概念数据已跳过)")
    print("    SNMA-V4 等策略将在无概念约束模式下运行")
    return df_empty


# ─────────────────────────────────────────────────────────────────────────────
# 自动检测模式
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# [ADATA] 模式4：adata THS 接口
# ─────────────────────────────────────────────────────────────────────────────

def run_adata_mode() -> pd.DataFrame:
    """
    [ADATA] 用 adata THS 接口下载概念成分股。

    流程：
      1. all_concept_code_ths()              → 391 个 THS 概念列表
      2. concept_constituent_ths(index_code) → 每个概念的成分股
      3. 写出 concept_mapping.csv（格式: code, concept）

    特点：
      · 无 IP 封禁风险
      · 支持 Ctrl+C 中断断点续传（每30个概念保存一次进度）
      · 约 10~20 分钟完成
    """
    import adata

    CONCEPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path    = CONCEPT_DIR / "_adata_concept_ckpt.json"
    partial_path = CONCEPT_DIR / "_adata_concept_partial.csv"
    meta_codes   = load_meta_codes()

    # Step 1: 概念列表
    print("  [adata] 获取 THS 概念列表...")
    try:
        df_c = adata.stock.info.all_concept_code_ths()
        if df_c is None or df_c.empty:
            print("  ✗ 概念列表为空")
            return pd.DataFrame()
    except Exception as e:
        print(f"  ✗ all_concept_code_ths() 失败: {e}")
        return pd.DataFrame()

    ic_col   = "index_code" if "index_code" in df_c.columns else df_c.columns[0]
    name_col = "name"       if "name"       in df_c.columns else df_c.columns[1]
    concepts = df_c[[ic_col, name_col]].dropna().values.tolist()
    print(f"  [adata] ✓ {len(concepts)} 个概念")

    # 断点续传
    start_idx = 0
    all_rows: list = []
    if ckpt_path.exists():
        try:
            with open(ckpt_path) as f:
                cp = json.load(f)
            start_idx = cp.get("next_idx", 0)
            if partial_path.exists():
                all_rows = pd.read_csv(str(partial_path), dtype=str).to_dict("records")
            print(f"  [adata] ↻ 断点恢复: 从第{start_idx}个概念，已有{len(all_rows)}条")
        except Exception:
            start_idx = 0; all_rows = []

    # Step 2: 逐概念下载成分股
    print(f"  [adata] 下载成分股 | Ctrl+C 中断后重启自动续传")
    failed = []
    t0 = time.time()

    try:
        for i in range(start_idx, len(concepts)):
            idx_code, cname = str(concepts[i][0]), str(concepts[i][1])
            done_n = i - start_idx
            spd    = done_n / max(time.time() - t0, 0.1)
            eta    = (len(concepts) - i) / max(spd, 0.01) / 60
            print(f"\r  [{i+1:3d}/{len(concepts)}] "
                  f"{cname[:14]:<14} "
                  f"已收集{len(all_rows)}条 失败{len(failed)}个 "
                  f"ETA {eta:.0f}m    ",
                  end="", flush=True)

            success = False
            for attempt in range(3):
                try:
                    try:
                        df_m = adata.stock.info.concept_constituent_ths(index_code=idx_code)
                    except TypeError:
                        df_m = adata.stock.info.concept_constituent_ths(name=cname)
                    if df_m is None or df_m.empty:
                        df_m = adata.stock.info.concept_constituent_ths(name=cname)
                    if df_m is not None and not df_m.empty:
                        sc_col = next((c for c in ["stock_code","code","证券代码","股票代码",
                                                    "scode","ts_code","symbol"]
                                       if c in df_m.columns), None)
                        # [FIX-CONCEPT-COL] adata 版本升级可能改列名，兜底取第一列
                        if sc_col is None and len(df_m.columns) > 0:
                            sc_col = df_m.columns[0]
                        if sc_col:
                            for sc in df_m[sc_col].dropna():
                                code6 = str(sc).strip().zfill(6)
                                if meta_codes and code6 not in meta_codes:
                                    continue
                                all_rows.append({"code": code6, "concept": cname})
                        success = True
                        break
                except Exception:
                    if attempt < 2:
                        time.sleep((attempt+1)*5 + random.uniform(0,3))

            if not success:
                failed.append(cname)

            if (i+1) % 30 == 0:
                with open(ckpt_path,"w") as f:
                    json.dump({"next_idx": i+1}, f)
                pd.DataFrame(all_rows).to_csv(str(partial_path), index=False)

            time.sleep(random.uniform(0.3, 0.8))

    except KeyboardInterrupt:
        print(f"\n  中断！保存断点（第{i}个概念）...")
        with open(ckpt_path,"w") as f:
            json.dump({"next_idx": i}, f)
        pd.DataFrame(all_rows).to_csv(str(partial_path), index=False)
        print(f"  重启后从 [{concepts[i][1]}] 继续")
        return pd.DataFrame()

    print()

    if not all_rows:
        print("  ✗ 无映射数据")
        return pd.DataFrame()

    df_out = pd.DataFrame(all_rows).drop_duplicates()

    # 保存 concept_mapping.csv（与其他模式格式一致）
    out_path = CONCEPT_DIR / "concept_mapping.csv"
    df_out.to_csv(str(out_path), index=False, encoding="utf-8-sig")

    nc  = df_out["code"].nunique()
    ncp = df_out["concept"].nunique()
    cov = nc / max(len(meta_codes), 1) * 100
    print(f"  [adata] ✓ {len(df_out)} 条  {nc} 只股票  {ncp} 个概念  "
          f"覆盖率={cov:.1f}%")

    if failed:
        fp = CONCEPT_DIR / "failed_concepts.txt"
        fp.write_text("\n".join(failed), encoding="utf-8")
        print(f"  ⚠ {len(failed)} 个概念失败 → {fp.name}")

    # 清理断点
    for p in [ckpt_path, partial_path]:
        if p.exists(): p.unlink()

    return df_out


def auto_detect_mode() -> str:
    """自动选择最优模式（优先级: manual > adata > pywencai > akshare > skip）"""
    # 1. 已有手动 CSV？
    for candidate in MANUAL_CSV_CANDIDATES:
        if candidate.exists():
            print(f"  [auto] 检测到手动 CSV: {candidate.name} → 使用 manual 模式")
            return "manual"

    # 2. [ADATA] adata 可用？（优先于 pywencai/akshare，无封禁风险）
    try:
        import adata  # noqa
        print("  [auto] 检测到 adata → 使用 adata 模式（THS概念，无封禁）")
        return "adata"
    except ImportError:
        pass

    # 3. PyWencai 可用？
    try:
        import pywencai  # noqa
        print("  [auto] 检测到 pywencai → 使用 pywencai 模式")
        return "pywencai"
    except ImportError:
        pass

    # 4. AKShare 可用？
    try:
        import akshare  # noqa
        print("  [auto] 检测到 akshare → 使用 akshare 模式（存在封IP风险）")
        return "akshare"
    except ImportError:
        pass

    # 5. 全部不可用：跳过
    print("  [auto] 无可用数据源 → 使用 skip 模式")
    return "skip"


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="获取同花顺概念板块数据")
    parser.add_argument(
        "--mode",
        choices=["auto", "manual", "adata", "pywencai", "akshare", "skip"],
        default="auto",
        help=(
            "数据获取模式: "
            "auto(自动检测) | "
            "manual(手动CSV，推荐) | "
            "adata(THS接口，推荐自动下载) | "
            "pywencai(需Node.js) | "
            "akshare(备用) | "
            "skip(跳过)"
        ),
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="manual 模式下指定 CSV 路径",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "ths", "em"],
        default="auto",
        help="akshare 模式下的数据源 (默认auto)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.4,
        help="akshare 模式请求间隔(秒，默认0.4)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(" Step 2: 同花顺概念板块数据")
    print("=" * 70)
    print()
    print("数据源优先级: manual(推荐) > adata(推荐自动) > pywencai > akshare(备用) > skip")
    print()

    CONCEPT_DIR.mkdir(parents=True, exist_ok=True)
    meta_codes = load_meta_codes()

    # 确定运行模式
    mode = args.mode if args.mode != "auto" else auto_detect_mode()
    print(f"运行模式: {mode}")
    print()

    # 检查是否已有 mapping
    mapping_path = CONCEPT_DIR / "concept_mapping.csv"
    if mapping_path.exists() and mode not in ("skip",):
        df_existing = pd.read_csv(mapping_path, dtype=str)
        if len(df_existing) > 100:
            print(f"⚠ 已有 concept_mapping.csv ({len(df_existing)} 条)")
            ans = input("  重新下载? (y/N): ").strip().lower()
            if ans != "y":
                print("  使用已有文件，直接生成 Q-UNITY 格式...")
                build_qunity_format(df_existing, meta_codes)
                return

    # 执行对应模式
    if mode == "manual":
        csv_path = Path(args.csv) if args.csv else None
        mapping_df = run_manual_mode(csv_path)
    elif mode == "adata":
        mapping_df = run_adata_mode()
    elif mode == "pywencai":
        mapping_df = run_pywencai_mode()
    elif mode == "akshare":
        mapping_df = run_akshare_mode(args.source, args.delay)
    elif mode == "skip":
        run_skip_mode(meta_codes)
        return
    else:
        print(f"未知模式: {mode}")
        return

    if not mapping_df.empty:
        build_qunity_format(mapping_df, meta_codes)
    else:
        print("\n⚠ 未获取到概念数据，使用 skip 模式生成占位文件")
        run_skip_mode(meta_codes)

    print(f"\n✓ Step 2 完成（模式: {mode}）")
    print("  下一步: python scripts/step4_build_concept_npy.py")


if __name__ == "__main__":
    main()



