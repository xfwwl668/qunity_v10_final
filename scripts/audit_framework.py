import numpy as np
import pandas as pd
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

print("[v0] 开始创建白盒审计框架...")
print("[v0] 目标：1500天正弦波 + 除权 + 生成Excel")
print("[v0] 13个策略全覆盖审计")
