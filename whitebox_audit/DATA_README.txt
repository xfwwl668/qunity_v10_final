白盒审计数据说明
=================

数据文件夹位置: whitebox_audit/data/

执行脚本生成CSV数据:
  cd whitebox_audit
  python gen_data.py  # 生成 01_base_data.csv
  python whitebox_audit_real.py  # 完整审计

数据规格:
- 行数: 1500 (交易日, 2020-01-01 至 2024-12-17)
- 列数: 
  * Date: 日期
  * Close_HFQ: 后复权收盘价 (范围: 19.52 - 133.11)
  * Adj_Factor: 复权因子 (累积除权倍数)
  * Split_Day: 除权标记 (1=除权日期)

特殊处理:
- 3周期正弦波: 252天 + 84天 + 30天周期叠加
- 5次除权事件: 日300/600/900/1200/1400, 倍数0.5/0.8/0.7/0.9/0.95
- 完全确定性: 使用固定随机种子(42), 可重现

如何生成CSV:
1. 执行: python gen_data.py
2. 输出: 01_base_data.csv (1500行×4列)
3. 或执行: python whitebox_audit_real.py (生成完整审计报告)
