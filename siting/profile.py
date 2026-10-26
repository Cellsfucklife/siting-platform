"""数据画像 L0–L3 与方法可用性 / 权重。"""
from __future__ import annotations

from .contract import CRIT_KEYS


def build(payload: dict) -> dict:
    dp = payload.get("demand_points") or []
    cd = payload.get("candidates") or []
    dm = payload.get("dist_matrix")
    crit = payload.get("criteria")

    has_demand = len(dp) >= 3
    has_cand = len(cd) >= 1
    has_dist = isinstance(dm, (list, tuple)) and len(dm) > 0

    if isinstance(crit, dict) and crit.get("matrix"):
        n_crit = len(crit["matrix"][0])
    elif isinstance(crit, (list, tuple)) and crit and isinstance(crit[0], (list, tuple)):
        n_crit = len(crit[0])
    else:                                  # 候选点自带属性也算
        n_crit = len({k for c in cd for k in CRIT_KEYS if k in c})
    has_crit = n_crit >= 3

    has_behavior = payload.get("self_pick_ratio") is not None
    has_hour = bool(payload.get("hour_profile")) or any(p.get("qty_by_hour") for p in dp)
    has_proxy = (payload.get("proxy_daily_demand") is not None
                 or payload.get("area_households") is not None)

    if has_demand and has_cand and has_dist and has_crit:
        level, label = 3, "L3 完整"
    elif has_demand and has_cand:
        level, label = 2, "L2 标准"
    elif has_demand:
        level, label = 1, "L1 精简"
    elif has_cand and has_proxy:
        level, label = 0, "L0 极限（无需求面，代理件量）"
    else:
        level, label = -1, "L-1 不可用"

    quality = (0.35 * has_demand + 0.20 * has_cand + 0.20 * has_dist
               + 0.10 * has_crit + 0.10 * has_hour + 0.05 * has_behavior)

    return {
        "level": level, "label": label, "quality_score": round(float(quality), 3),
        "has_demand": has_demand, "has_candidates": has_cand, "has_dist_matrix": has_dist,
        "has_criteria": has_crit, "n_criteria": n_crit, "has_behavior": has_behavior,
        "has_hour_profile": has_hour, "has_proxy": has_proxy,
        "n_demand_points": len(dp), "n_candidates": len(cd),
    }


def select(profile: dict, methods: list) -> dict:
    """按需求字段筛方法；权重 = 先验 × 数据质量。"""
    usable, rejected = [], []
    for m in methods:
        miss = []
        if "demand_points" in m.needs and not profile["has_demand"]:
            miss.append("demand_points")
        if "candidates" in m.needs and not profile["has_candidates"]:
            miss.append("candidates")
        if "criteria" in m.needs and not profile["has_criteria"]:
            miss.append(f"criteria(≥3项, 当前 {profile['n_criteria']})")
        if miss:
            rejected.append({"method": m.key, "name": m.name,
                             "family": m.family, "missing": miss})
        else:
            usable.append(m)

    raw = {m.key: m.prior * (0.5 + 0.5 * profile["quality_score"]) for m in usable}
    tot = sum(raw.values()) or 1.0
    weights = {k: round(v / tot, 3) for k, v in raw.items()}
    mode = "multi" if len(usable) >= 2 else ("single" if len(usable) == 1 else "none")
    return {"mode": mode, "used": [m.key for m in usable], "weights": weights,
            "rejected": rejected, "_usable": usable}


def recommend(profile: dict) -> list[dict]:
    out = []
    if not profile["has_dist_matrix"]:
        out.append({"field": "dist_matrix", "expected_gain": "方法一致性 +0.10~0.15",
                    "reason": "M2 现在用直线×绕行系数，路网矩阵可消除系统性偏差"})
    if not profile["has_criteria"]:
        out.append({"field": "criteria(≥3项)", "expected_gain": "新增一个独立证据源",
                    "reason": "启用 M4，引入与几何无关的判别信息"})
    if not profile["has_hour_profile"]:
        out.append({"field": "hour_profile", "expected_gain": "容量误差 -5~8pp",
                    "reason": "默认曲线抹平片区差异，直接影响峰值与档位"})
    if not profile["has_behavior"]:
        out.append({"field": "self_pick_ratio", "expected_gain": "格口需求 ±15%",
                    "reason": "自提比例决定柜/站档位；缺失按 100% 计入偏保守"})
    if not profile["has_candidates"]:
        out.append({"field": "candidates", "expected_gain": "解锁 M2/M4",
                    "reason": "当前只能给理论重心，落不到可建地块"})
    return out
