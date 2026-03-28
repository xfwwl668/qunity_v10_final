# Q-UNITY V10 量化回测框架 — 使用说明

版本：V10（adata + TdxQuant 双数据源版）
更新：2026-03

---

## 一、项目结构

```
v10_patched/
├── main.py                          # 交互主菜单（唯一入口）
├── config.json                      # 全局配置（路径/参数）
├── requirements.txt                 # Python 依赖
│
├── scripts/                         # 数据下载脚本（subprocess 调用）
│   ├── step0_download_tdxquant.py   # ★ 日线下载（TdxQuant，推荐）
│   ├── step0_download_ohlcv.py      # 日线下载（adata/AKShare 备用）
│   ├── step1_download_fundamental_akshare.py  # 季度基本面
│   ├── step2_download_concepts.py   # 概念板块
│   ├── step3_build_fundamental_npy.py  # 构建基本面 npy
│   ├── step4_build_concept_npy.py   # 构建概念 npy
│   ├── realtime_tdxquant.py         # ★ 实时行情引擎（TdxQuant）
│   ├── live_trade_tools.py          # 实盘辅助（仓位/滑点）
│   └── validate_npy.py              # 数据验证
│
├── src/
│   ├── engine/
│   │   ├── fast_runner_v10.py       # 回测引擎核心
│   │   ├── numba_kernels_v10.py     # Numba 加速内核
│   │   ├── optimizer_v10.py         # Optuna 参数优化
│   │   └── alpha_signal.py          # 信号基类
│   └── strategies/
│       ├── ultra_short_signal.py    # ★ 超短线实盘引擎（含TdxFeedAdapter）
│       ├── registry.py              # 策略注册表
│       └── vectorized/              # 向量化回测策略
│           ├── titan_alpha_v1_alpha.py
│           ├── kunpeng_v10_alpha.py
│           ├── short_term_rsrs_alpha.py
│           └── ...（其他策略）
│
└── data/                            # 数据目录（运行后自动生成）
    ├── daily_parquet_qfq/           # 日线 parquet（每只股票一个文件）
    ├── npy_v10/                     # npy 矩阵（回测引擎直接读取）
    │   ├── meta.json                # 股票列表、日期、shape
    │   ├── close.npy / open.npy ... # OHLCV 矩阵 (N, T)
    │   ├── fundamental_roe.npy      # 基本面矩阵
    │   ├── pe_ttm.npy / pb_mrq.npy  # 估值矩阵
    │   ├── concept_ids.npy          # 概念板块矩阵
    │   └── market_index.npy         # 沪深300 指数
    ├── fundamental/                 # 季度财务 CSV
    └── concepts/                    # 概念映射 CSV
```

---

## 二、快速开始（首次使用完整流程）

### 前提条件

```bash
pip install -r requirements.txt
# 必须：numpy pandas akshare adata numba optuna colorama
# TdxQuant：需要通达信客户端（免费），安装路径见下文
```

### 第一步：启动主菜单

```bash
cd v10_patched
python main.py
```

### 第二步：数据下载（三选一）

**方案A：TdxQuant（强烈推荐）**
- 前提：安装通达信客户端 → 系统 → 盘后数据下载（起始2015年）
- 主菜单 → [1] 数据管理 → [2] TdxQuant 本地下载
- 特点：完全本地，零网络依赖，约 30~60 分钟完成全量

**方案B：adata/AKShare（备用）**
- 主菜单 → [1] 数据管理 → [2a] adata/AKShare auto 模式
- 特点：有封IP风险，adata `get_market` 接口时好时坏

**方案C：命令行直接运行**
```bash
# TdxQuant 全量（需通达信已开）
python scripts/step0_download_tdxquant.py --workers 16 --start 20150101

# 测试模式（10只，验证链路）
python scripts/step0_download_tdxquant.py --test
```

### 第三步：构建 npy 矩阵

主菜单 → [1] 数据管理 → [3] 下载日线 + 构建 npy

或命令行：
```bash
python scripts/step0_download_ohlcv.py --build-npy --incremental \
    --parquet-dir data/daily_parquet_qfq \
    --npy-dir data/npy_v10
```

### 第四步：下载基本面数据（可选，策略有用到时才需要）

主菜单 → [1] 数据管理 → [4] 下载季度基本面

```bash
python scripts/step1_download_fundamental_akshare.py --source auto --workers 8 --test
```

### 第五步：回测

主菜单 → [2] 回测 → [1] 单策略回测

---

## 三、主菜单完整说明

```
[1] 数据管理
    ├── 1.   查看数据状态（文件数、矩阵 shape、字段完整性）
    │
    ├── 日线数据（TdxQuant 最优先）
    ├── 2.   TdxQuant 本地下载 ★推荐（零网络，需通达信客户端）
    ├── 2a.  adata/AKShare auto（自动选源，有封IP风险）
    ├── 2b.  强制 adata（adata get_market 时好时坏，慎用）
    ├── 2c.  强制 AKShare
    ├── 2t.  测试模式（TdxQuant 10只验证）
    │
    ├── npy 矩阵构建
    ├── 3.   下载日线 + 立即构建 npy
    ├── 3a.  仅构建 npy（parquet 已有时）
    ├── 3b.  对齐辅助矩阵（增量下载后需执行）
    │
    ├── 基本面数据
    ├── 4.   季度基本面 auto（优先 adata get_core_index）
    ├── 4a.  强制 adata（43字段，无封禁）
    ├── 4b.  强制 AKShare
    ├── 4c.  BaoStock（已封IP，备用）
    ├── 4d.  BaoStock PE/PB/isST（已封IP，备用）
    ├── 5.   构建基本面 npy（step3）
    │
    ├── 概念板块
    ├── 6.   概念板块数据（adata THS / manual / AKShare）
    │
    ├── 实时行情（需通达信已开）
    ├── 7.   实时行情快照（含 PE/PB/市值/涨跌幅）
    ├── 8.   全市场扫描 + 超短线预警推送到通达信板块
    │
    └── 验证工具
        ├── 9.   验证 npy 数据完整性
        └── 9a.  复权兼容性检查

[2] 回测
    ├── 1.  单策略回测（选策略 + 时间段 + NAV 曲线）
    ├── 2.  多策略组合回测（风险平价/等权/动量倾斜）
    ├── 3.  参数扫描（top_n × rsrs_window 矩阵对比）
    └── 4.  Walk-Forward 优化（Optuna OOS）

[3] 实盘信号
    ├── 1.  生成今日目标持仓权重
    ├── 2.  多策略信号等权合并
    └── 3.  查看历史信号文件

[4] 策略优化工具
    ├── 1.  前视偏差检查（数据可信度，必须先做）
    ├── 2.  单因子归因分析
    ├── 3.  流动性分层测试
    ├── 4.  Regime 切换成本
    ├── 5.  最终 OOS 测试（只用一次）
    ├── 6.  仓位计算器
    └── 7.  滑点分析

[5] 系统工具
    ├── 配置管理、npy 路径、策略参数
    └── 回测参数快速设置
```

---

## 四、数据源说明

### TdxQuant（★ 主力数据源）

| 特点 | 说明 |
|---|---|
| 本质 | 通达信客户端本地接口，进程内通信 |
| 依赖 | 通达信客户端已开+已登录+已下载盘后数据 |
| 稳定性 | ★★★★★ 完全本地，零网络，不存在封IP/超时 |
| 历史数据 | 以通达信本地下载数据为准（建议从2015年起） |
| Volume 单位 | 手（×100=股，fast_runner vol_multiplier=100） |
| Amount 单位 | 万元（脚本内×10000转元存储） |
| 复权 | `dividend_type='front'` = QFQ 前复权 |

**通达信数据下载步骤：**
1. 通达信客户端 → 系统 → 盘后数据下载
2. 勾选：沪深全部股票 + 指数
3. 起始日期：2015-01-01
4. 点击下载，等待完成（约30~60分钟）

### adata（备用数据源）

| 接口 | 状态 | 备注 |
|---|---|---|
| `get_market()` 日线 | ⚠ 时好时坏 | 日期格式必须 `YYYY-MM-DD`（有横线） |
| `get_core_index()` 基本面 | ✅ 稳定 | 43字段，一次请求全量历史 |
| `all_concept_code_ths()` 概念 | ✅ 稳定 | THS 391个概念 |
| `get_industry_sw()` 申万行业 | ✅（需禁代理） | 来源百度 |
| `trade_calendar()` 交易日历 | ✅ 稳定 | |

**关键配置（脚本内置）：**
```python
adjust_type = 1    # 1=QFQ前复权（实测确认）
start_date  = '2024-01-01'  # 必须有横线
```

### AKShare（降级备用）

```bash
pip install akshare -U
```

有东财节点封IP风险，单日频繁调用后可能被限。

---

## 五、实时行情 & 超短线策略

### TdxQuant 实时行情引擎

```python
from scripts.realtime_tdxquant import TdxRealtimeEngine

engine = TdxRealtimeEngine()
engine.initialize()

# 单只快照（含 PE/PB/市值/涨跌幅）
snap = engine.snapshot('600519.SH')
# snap = {
#   'price': 1445.0, 'change_pct': -0.54,
#   'pe_ttm': 20.10, 'pb_mrq': 7.04,
#   'mkt_cap_total': 18095.30,    # 亿元
#   'zt_price': 1598.16,          # 涨停价
#   'con_zaf': -3,                # 连涨/跌天数
# }

# 订阅推送（有更新自动回调）
def on_update(code, snap):
    print(f"{code}: {snap['price']:.2f} {snap['change_pct']:+.2f}%")
engine.subscribe(['600519.SH', '000001.SZ'], callback=on_update)

# 发预警到通达信
engine.send_alert('600519.SH', price=1450.0, reason='RSRS突破信号', bs_flag=0)

# 推送选股结果到通达信板块
engine.push_to_block(['600519.SH', '000001.SZ'], block_code='QUNITY')

engine.close()
```

### 超短线策略引擎（UltraShortSignalEngine）

**三重门控逻辑：**
```
Gate-1：高开 > 1.2%（当日开盘缺口，确认趋势方向）
Gate-2：量比 > 2.0（今日成交量 / 20日均量，确认资金关注）
Gate-3：当日涨幅 > 1.5%（突破确认）
止损：持仓亏损 > 1.5% 立即出场
止盈：持仓盈利 > 2.0% 立即出场
时间止损：持仓超过60个tick（约5分钟）强制出场
```

**与 TdxQuant 对接：**
```python
from src.strategies.ultra_short_signal import (
    UltraShortSignalEngine, TdxFeedAdapter, create_ultra_short_engine
)
from scripts.realtime_tdxquant import TdxRealtimeEngine

tdx = TdxRealtimeEngine()
tdx.initialize()

engine = create_ultra_short_engine(config=cfg)

# 开盘前：从 npy 矩阵加载历史数据
prev_closes, vol_avg20s = TdxFeedAdapter.build_history_from_npy(
    codes, npy_dir='data/npy_v10'
)
engine.update_history(prev_closes, vol_avg20s)

# 盘中循环（每次 tick）
while True:
    snapshots = tdx.batch_snapshot(codes)
    feed = TdxFeedAdapter.convert(snapshots)
    signals = engine.scan(feed)
    for code, direction in signals.items():
        if direction == 'buy':
            tdx.send_alert(code, feed[code]['close'],
                          reason='超短线买入', bs_flag=0)
        elif direction == 'sell':
            tdx.send_alert(code, feed[code]['close'],
                          reason='超短线卖出', bs_flag=1)
```

---

## 六、数据格式规格

### Parquet 文件（`data/daily_parquet_qfq/`）

每只股票一个文件，文件名格式：`sh.600519.parquet`

| 列名 | 类型 | 单位 | 说明 |
|---|---|---|---|
| date | str | - | `YYYY-MM-DD` |
| open/high/low/close | float32 | 元 | QFQ 前复权 |
| volume | float32 | **手** | ×100 = 股 |
| amount | float64 | **元** | TdxQuant ×10000换算 |
| code | str | - | `sh.600519` |
| listing_date | str | - | 上市日期 |
| adj_type | str | - | `"qfq"` |

### npy 矩阵（`data/npy_v10/`）

形状：`(N, T)`，N=股票数（约5500），T=交易日数（约2700）

| 文件 | 来源 | 说明 |
|---|---|---|
| close/open/high/low.npy | parquet | 价格矩阵，单位元 |
| volume.npy | parquet | 成交量，单位**手** |
| amount.npy | parquet | 成交额，单位元 |
| valid_mask.npy | 构建时计算 | 1=可交易，0=无数据/退市 |
| fundamental_roe.npy | step3 | ROE 矩阵，前向填充 |
| pe_ttm.npy / pb_mrq.npy | step3 | 估值，季度前向填充 |
| concept_ids.npy | step4 | 概念板块 ID |
| market_index.npy | step0 | 沪深300 收盘价 (1, T) |

**fast_runner 关键参数：**
```python
vol_multiplier = 100   # volume 单位=手，×100=股
stamp_tax      = 0.0005  # 印花税万五（仅卖出）
adj_type       = 'qfq'
```

---

## 七、常见问题

**Q：TdxQuant 初始化报错 "can't find tqcenter.py"？**
A：确认通达信客户端已打开。`step0_download_tdxquant.py` 会自动检测以下路径：
- `D:\SOFT(DONE)\tdx\ncb\PYPlugins\user`
- `C:\new_tdx\PYPlugins\user`
- `D:\new_tdx\PYPlugins\user`

如果路径不同，运行时加 `--tq-dir 你的路径`。

**Q：adata get_market 全部返回空 DataFrame？**
A：已知问题，adata 的百度股市通接口时好时坏，与代码无关。
解决方案：改用 TdxQuant（主菜单 [2]）。

**Q：get_stock_list 返回0只？**
A：正确调用方式是 `tq.get_stock_list('5')`（字符串参数），不是 `list_type=5`。
`step0_download_tdxquant.py` 内部已使用正确调用方式。

**Q：Volume 单位混乱？**
A：
- TdxQuant 返回：**手**
- parquet 存储：**手**（不转换）
- fast_runner 读取：手 × `vol_multiplier=100` = 股
- amount 单位：TdxQuant 返回万元，脚本内 ×10000 转元存储

**Q：build_npy 之后 npy_v10 里没有 pe_ttm.npy？**
A：PE/PB 来自 step3（基本面 npy 构建），需要先完成 step1（下载季度基本面），
再运行 [5] 构建基本面 npy。

**Q：回测曲线平坦，几乎不交易？**
A：检查 `valid_mask.npy`，如果大量为0则 listing_days_min 过滤太严格，或数据日期范围不匹配。
运行 [9] 验证数据完整性查看详情。

**Q：超短线策略在 scan() 里全部返回 hold？**
A：需要先调用 `engine.update_history(prev_closes, vol_avg20s)`。
如果 `prev_close == 0` 则跳过该股票。使用 `TdxFeedAdapter.build_history_from_npy()` 自动加载。

---

## 八、推荐工作流（日常使用）

### 每日盘前（约5分钟）

```bash
# 1. 增量下载今日数据（通达信已开）
python scripts/step0_download_tdxquant.py --incremental --workers 16

# 2. 重建 npy 矩阵（含今日数据）
python scripts/step0_download_ohlcv.py --build-npy --incremental

# 3. 生成今日信号
python main.py  # -> [3] 实盘信号 -> [1] 生成今日目标持仓
```

### 盘中超短线（实时）

```bash
# 实时全市场扫描 + 预警推送到通达信
python scripts/realtime_tdxquant.py --scan --block QUNITY
```

### 每月（基本面更新）

```bash
python scripts/step1_download_fundamental_akshare.py --source adata --workers 8
python scripts/step3_build_fundamental_npy.py
```

---

## 九、风险提示

- 本程序为量化研究工具，**不构成任何投资建议**
- 策略回测基于历史数据，实盘表现可能与回测存在重大偏差
- 本程序**不具备自动下单功能**，所有交易须手动在券商平台执行
- 止损/持仓上限在实盘中须用户自律执行
- 量化策略可能因市场结构变化而失效，勿重仓依赖单一策略
