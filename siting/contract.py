"""数据契约：字段清单 + 校验 + 默认值 + 档位表。"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Field:
    name: str
    tier: str            # R 条件必填 | O 选填
    desc: str
    default: Any = None
    used_by: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


DATA_CONTRACT: tuple[Field, ...] = (
    # —— 标识与需求面 ——
    Field("area_id", "R", "服务区编号；一次调用 = 一个配送节点", None, ("all",)),
    Field("demand_points", "R*", "需求点 [id, lat, lng, qty_daily]；与 candidates 至少其一",
          None, ("M1", "M2", "M3", "capacity")),
    Field("candidates", "R*", "候选点 [id, lat, lng, (area_sqm, rent_month, 准则字段)]",
          None, ("M2", "M4")),
    Field("qty_by_hour", "O", "单点 24 点件量；用于聚合出到达曲线", None, ("capacity",)),
    Field("hour_profile", "O", "24 点占比（缺失用内置曲线）", None, ("capacity",)),
    Field("peak_factor", "O", "峰值日/日均倍数", 1.6, ("capacity",)),
    # —— 空间 ——
    Field("dist_matrix", "O", "需求点×候选点的行驶时间(分钟)或距离(米)", None, ("M2",)),
    Field("road_detour_factor", "O", "无矩阵时直线距离的绕行系数", 1.35, ("M2",)),
    Field("service_radius_min", "O", "服务圈时间标准（分钟）", 10.0, ("capacity",)),
    Field("access_speed_kmh", "O", "末端接驳平均速度", 12.0, ("capacity",)),
    # —— 评价与行为 ——
    Field("criteria", "O", "多准则矩阵或 {matrix, keys, benefit}；≥3 项则启用 M4", None, ("M4",)),
    Field("self_pick_ratio", "O", "自提比例，决定格口需求", None, ("capacity",)),
    # —— L0 代理 ——
    Field("proxy_daily_demand", "O", "无需求面时的代理日件量", None, ("capacity",)),
    Field("area_households", "O", "片区户数（二级代理）", None, ("capacity",)),
    Field("penetration", "O", "快递渗透率", 0.35, ("capacity",)),
    Field("per_household_daily", "O", "户均日件量", 0.6, ("capacity",)),
    # —— 容量引擎参数 ——
    Field("tier_catalog", "O", "档位表（缺失用内置）", None, ("capacity",)),
    Field("service_time_s", "O", "单件处理工时（秒）", 45.0, ("capacity",)),
    Field("sort_per_staff_day", "O", "分拣理货人均日处理件量", 1200.0, ("capacity",)),
    Field("queue_wait_target_s", "O", "排队等待目标（秒）", 180.0, ("capacity",)),
    Field("queue_fail_prob", "O", "等待超时概率上限", 0.10, ("capacity",)),
    Field("turnover_per_grid", "O", "单格口日周转次数", 1.8, ("capacity",)),
    Field("cv_margin", "O", "波动缓冲系数：安全边际 = 1 + CV × 该系数", 0.75,
          ("capacity",)),
    Field("vehicle_daily_capacity", "O", "单车日配送件量", 160.0, ("capacity",)),
    Field("robot_share", "O", "无人车承担比例", 0.0, ("capacity",)),
    Field("robot_daily_capacity", "O", "单台无人车日能力", 90.0, ("capacity",)),
    Field("charger_minutes", "O", "单次充电时长（分）", 90.0, ("capacity",)),
    Field("charger_available_min", "O", "单桩日可用时长（分）", 900.0, ("capacity",)),
)

# 档位表
#   grids       : 储存单元（柜=格口；站=货架位等价）
#   daily_cap   : 日通过能力（件）
#   peak_hour_cap: 峰值小时通过能力（件/h）
#   area_sqm    : 场地面积门槛（m²）
TIER_CATALOG: tuple[dict, ...] = (
    {"id": "locker-S",   "kind": "locker",  "grids": 100,  "daily_cap": 260,  "peak_hour_cap": 45,  "area_sqm": 15,  "cost_index": 1.0},
    {"id": "locker-M",   "kind": "locker",  "grids": 200,  "daily_cap": 520,  "peak_hour_cap": 90,  "area_sqm": 22,  "cost_index": 1.6},
    {"id": "locker-L",   "kind": "locker",  "grids": 400,  "daily_cap": 1040, "peak_hour_cap": 180, "area_sqm": 35,  "cost_index": 2.6},
    {"id": "locker-XL",  "kind": "locker",  "grids": 800,  "daily_cap": 2080, "peak_hour_cap": 360, "area_sqm": 60,  "cost_index": 4.2},
    {"id": "station-M",  "kind": "station", "grids": 700,  "daily_cap": 1800, "peak_hour_cap": 240, "area_sqm": 120, "cost_index": 3.4},
    {"id": "station-L",  "kind": "station", "grids": 1400, "daily_cap": 3200, "peak_hour_cap": 420, "area_sqm": 240, "cost_index": 5.8},
    {"id": "station-XL", "kind": "station", "grids": 2400, "daily_cap": 5600, "peak_hour_cap": 700, "area_sqm": 400, "cost_index": 9.0},
)

CRIT_KEYS: tuple[str, ...] = ("rent_month", "accessibility", "competition",
                              "site_availability", "expandability", "policy")

DEFAULT_HOUR_PROFILE = np.array(
    [0.004, 0.003, 0.002, 0.002, 0.002, 0.003, 0.008, 0.018, 0.032, 0.045,
     0.052, 0.055, 0.062, 0.068, 0.070, 0.072, 0.078, 0.085, 0.080, 0.058,
     0.038, 0.028, 0.018, 0.008], dtype=float)
DEFAULT_HOUR_PROFILE /= DEFAULT_HOUR_PROFILE.sum()


def get(payload: dict, key: str) -> Any:
    """取字段值，缺失用契约默认值。"""
    v = payload.get(key)
    if v is not None:
        return v
    for f in DATA_CONTRACT:
        if f.name == key:
            return f.default
    return None


def validate(payload: dict) -> dict:
    """返回 errors / warnings —— 后端入口第一件事。"""
    errors, warnings = [], []
    dp = payload.get("demand_points") or []
    cd = payload.get("candidates") or []

    if not payload.get("area_id"):
        warnings.append("缺少 area_id，结果不带节点标识")
    if len(dp) < 3 and not cd:
        errors.append("必须提供 demand_points(≥3) 或 candidates(≥1)")
    for i, p in enumerate(dp):
        for k in ("id", "lat", "lng", "qty_daily"):
            if k not in p:
                errors.append(f"demand_points[{i}] 缺字段 {k}")
        if p.get("qty_daily", 0) <= 0:
            warnings.append(f"demand_points[{i}] 件量为 0，将被忽略权重")
    for i, c in enumerate(cd):
        for k in ("id", "lat", "lng"):
            if k not in c:
                errors.append(f"candidates[{i}] 缺字段 {k}")

    dm = payload.get("dist_matrix")
    if isinstance(dm, (list, tuple)) and dm:
        if len(dm) != len(dp):
            errors.append(f"dist_matrix 行数 {len(dm)} ≠ 需求点数 {len(dp)}")
        elif any(len(r) != len(cd) for r in dm):
            errors.append("dist_matrix 列数 ≠ 候选点数")

    for f in DATA_CONTRACT:
        if f.default is None or f.name in payload:
            continue
        payload[f.name] = f.default
    return {"ok": not errors, "errors": errors, "warnings": warnings}


def summary() -> list[dict]:
    """给 /schema 接口用：契约的可序列化视图。"""
    return [f.to_dict() for f in DATA_CONTRACT]
