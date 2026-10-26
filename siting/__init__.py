"""末端配送节点：模糊选址中心 + 容量配置 —— 集成后端。

分层
----
contract  数据契约（必填/选填/默认值）
profile   数据画像 L0–L3 与方法可用性判定
geo       投影、网格、凸包
methods   M1 连续重心 / M2 离散候选 / M3 核密度 / M4 多准则
capacity  容量引擎（Erlang-C + 档位选择 + 蒙特卡洛）
fusion    μ 场加权融合 → 模糊选址中心
validate  方法间一致性 + 实现间一致性（参考实现 vs 自研实现）
solver    编排：画像 → 选方法 → 求解 → 交叉验证 → 输出
"""
__version__ = "2.0.0"

from .solver import solve                      # noqa: F401
from .contract import DATA_CONTRACT, TIER_CATALOG  # noqa: F401
