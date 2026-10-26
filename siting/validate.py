"""交叉验证：方法间一致性 + 实现间一致性 + 结构单调性。

重叠度量用 Bhattacharyya 系数（把 μ 归一化成分布后比较形状），
而不是硬阈值掩膜 Jaccard —— 后者对不同的核宽度极不公平。
"""
from __future__ import annotations

import math

import numpy as np

from .geo import Grid


def _bhattacharyya(a: np.ndarray, b: np.ndarray) -> float:
    pa = a.ravel() / max(a.sum(), 1e-12)
    pb = b.ravel() / max(b.sum(), 1e-12)
    return float(np.sqrt(pa * pb).sum())


def method_consistency(grid: Grid, fields: dict[str, np.ndarray], per: dict[str, dict],
                       fuse_mu: np.ndarray, fuse_center: tuple[float, float],
                       r90: float) -> dict:
    keys = list(fields)
    fused_mask = fuse_mu > 0.5

    jac = {}
    for k in keys:
        m = fields[k] > 0.5
        union = (m | fused_mask).sum() or 1
        jac[k] = round(float((m & fused_mask).sum() / union), 3)

    bc_vs_fused = {k: round(_bhattacharyya(fields[k], fuse_mu), 3) for k in keys}

    pairwise, pairwise_bc = {}, {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            ma, mb = fields[a] > 0.5, fields[b] > 0.5
            union = (ma | mb).sum() or 1
            pairwise[f"{a}×{b}"] = round(float((ma & mb).sum() / union), 3)
            pairwise_bc[f"{a}×{b}"] = round(_bhattacharyya(fields[a], fields[b]), 3)

    ref = max(r90, 300.0)
    offsets, in_region = {}, 0
    for k in keys:
        c = per[k]["center_m"]
        d = math.hypot(c[0] - fuse_center[0], c[1] - fuse_center[1])
        offsets[k] = round(d, 1)
        if d <= ref:
            in_region += 1

    n = max(len(keys), 1)
    region_score = in_region / n
    shape_score = float(np.mean(list(bc_vs_fused.values()))) if bc_vs_fused else 0.0
    score = 0.6 * region_score + 0.4 * shape_score if len(keys) >= 2 else 0.35

    divergence = []
    if len(keys) >= 2:
        for k in keys:
            near = offsets[k] <= max(ref * 0.8, 300.0)
            shape = bc_vs_fused[k]
            if not near or shape < 0.35:
                divergence.append({
                    "method": k,
                    "offset_m": offsets[k],
                    "shape_similarity": shape,
                    "assumption": per[k].get("assumption", ""),
                    "hint": ("方法假设与数据条件不匹配" if not near
                             else "位置接近但模糊区形状差异大：核宽度/不确定度模型不一致"),
                })

    return {
        "per_method_bc_vs_fused": bc_vs_fused,
        "per_method_jaccard_vs_fused": jac,
        "pairwise_jaccard": pairwise,
        "pairwise_bhattacharyya": pairwise_bc,
        "center_offset_to_fused_m": offsets,
        "centers_in_fused_region": f"{in_region}/{n}",
        "region_score": round(region_score, 3),
        "shape_score": round(shape_score, 3),
        "score": round(score, 3),
        "verdict": "high" if score >= 0.75 else ("medium" if score >= 0.55 else "low"),
        "divergence": divergence,
        "metric_note": "region_score=中心落在融合90%区内的比例；shape_score=Bhattacharyya 形状相似度",
    }


def implementation_consistency(per: dict[str, dict]) -> dict:
    """参考实现 vs 自研实现 的一致性汇总。"""
    out, ok, total = {}, 0, 0
    for k, v in per.items():
        ag = (v.get("backends") or {}).get("agreement")
        if not ag:
            continue
        out[k] = ag
        flag = ag.get("consistent")
        if flag is None:
            continue
        total += 1
        ok += int(bool(flag))
    out["summary"] = {
        "checked": total,
        "consistent": ok,
        "all_consistent": (ok == total) if total else None,
        "note": "consistent=None 表示该方法无第三方等价实现，不做此项校验",
    }
    return out


def capacity_consistency(per: dict[str, dict]) -> dict:
    caps = [v["capacity_demand_p50"] for v in per.values() if "capacity_demand_p50" in v]
    if len(caps) < 2:
        return {"spread_ratio": None, "consistent": None}
    spread = (max(caps) - min(caps)) / max(max(caps), 1e-6)
    return {"values": caps, "spread_ratio": round(float(spread), 3),
            "consistent": bool(spread <= 0.25),
            "note": "各方法中心不同 → 同一服务圈内件量不同；≤25% 视为一致"}


def monotonic_check(payload: dict, grid: Grid, center: tuple[float, float],
                    radius: float, capacity_fn) -> dict:
    """结构一致性：件量 ×1.2 → 容量单调不减；服务半径 ×1.2 → 需求单调不减。"""
    from copy import deepcopy
    base = capacity_fn(payload, grid, center, radius)

    p2 = deepcopy(payload)
    for p in p2.get("demand_points") or []:
        p["qty_daily"] = float(p.get("qty_daily", 0)) * 1.2
    if p2.get("proxy_daily_demand"):
        p2["proxy_daily_demand"] = float(p2["proxy_daily_demand"]) * 1.2
    up = capacity_fn(p2, grid, center, radius)

    ok_qty = (up.get("daily_demand_p50") or 0) >= (base.get("daily_demand_p50") or 0) - 1e-9
    ok_cap = (up.get("peak_day_demand") or 0) >= (base.get("peak_day_demand") or 0) - 1e-9
    wider = capacity_fn(payload, grid, center, radius * 1.2)
    ok_r = (wider.get("daily_demand_p50") or 0) >= (base.get("daily_demand_p50") or 0) - 1e-9

    return {"demand_monotone": bool(ok_qty and ok_r),
            "capacity_monotone": bool(ok_cap),
            "radius_effect": {"r": round(radius, 1),
                              "demand": base.get("daily_demand_p50"),
                              "r_plus_20pct": wider.get("daily_demand_p50")}}
