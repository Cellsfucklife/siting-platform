"""编排层：画像 → 选方法 → 求解 → 融合 → 交叉验证 → 输出。"""
from __future__ import annotations

import math
import time

import numpy as np

from . import capacity as capacity_mod
from . import fusion, profile, validate
from .contract import get, validate as validate_payload
from .geo import Grid, hull_lnglat
from .methods import ALL as METHODS


def solve(payload: dict, with_checks: bool = True) -> dict:
    t0 = time.perf_counter()
    chk = validate_payload(payload)
    if not chk["ok"]:
        return {"ok": False, "stage": "contract",
                "errors": chk["errors"], "warnings": chk["warnings"]}

    prof = profile.build(payload)
    if prof["level"] < 0:
        return {"ok": False, "stage": "profile", "data_profile": prof,
                "errors": ["数据不足：需 demand_points(≥3)，或 candidates + 代理件量"]}

    sel = profile.select(prof, METHODS)
    if sel["mode"] == "none":
        return {"ok": False, "stage": "select", "data_profile": prof,
                "method_selection": {"rejected": sel["rejected"]},
                "errors": ["没有可运行的方法"]}

    grid = Grid(payload)
    fields, per, warnings = {}, {}, list(chk["warnings"])

    for spec in sel["_usable"]:
        try:
            r = spec.run(payload, grid)
        except Exception as exc:                     # 单方法失败不影响整体
            warnings.append(f"{spec.key} 运行失败：{exc}")
            continue
        fields[spec.key] = r["mu"]
        per[spec.key] = {
            "name": spec.name, "family": spec.family, "assumption": spec.assumption,
            "ref_impl": spec.ref_impl,
            "center_m": [round(r["center"][0], 1), round(r["center"][1], 1)],
            "mu_peak": round(float(r["mu"].max()), 3),
            "sigma_m": [round(v, 1) for v in r["sigma"]],
            "detail": r["detail"], "backends": r["backends"],
        }

    if not fields:
        return {"ok": False, "stage": "run", "errors": ["所有方法均失败"], "warnings": warnings}

    w = {k: sel["weights"][k] for k in fields}
    fmu, f = fusion.combine(grid, fields, w, single_mode=(len(fields) == 1))
    center = f["center"]

    speed_m_min = float(get(payload, "access_speed_kmh")) * 1000.0 / 60.0
    r_service = float(get(payload, "service_radius_min")) * speed_m_min

    # 各方法自己的中心 → 容量需求（用于容量侧交叉验证）
    for k in list(per):
        ds = capacity_mod.demand_series(payload, grid, per[k]["center_m"], r_service)
        per[k]["capacity_demand_p50"] = None if ds["qty"] is None else round(ds["qty"], 1)

    cap = capacity_mod.evaluate(payload, grid, center, r_service)

    lat, lng = grid.to_latlon(*center)
    ys, xs = (fmu >= 0.5).nonzero()
    poly = hull_lnglat(grid, np.column_stack([grid.X[ys, xs], grid.Y[ys, xs]]))

    center_obj = {
        "lat": round(lat, 6), "lng": round(lng, 6),
        "x_m": round(center[0], 1), "y_m": round(center[1], 1),
        "mu_peak": round(f["mu_peak"], 3),
        "radius_68_m": round(f["radius_68_m"], 1),
        "radius_90_m": round(f["radius_90_m"], 1),
        "service_radius_m": round(r_service, 1),
        "bimodal": f["bimodal"],
        "alt_centers": [
            {"lat": round(grid.to_latlon(a["x_m"], a["y_m"])[0], 6),
             "lng": round(grid.to_latlon(a["x_m"], a["y_m"])[1], 6),
             "x_m": round(a["x_m"], 1), "y_m": round(a["y_m"], 1),
             "mu": round(a["mu"], 3)} for a in f["alt_centers"]],
        "region_polygon_lnglat": poly,
        "definition": ("中心=融合μ场峰值邻域质心；半径=μ质量68%/90%分位；"
                       "多边形=μ≥0.5格点凸包（GeoJSON [lng,lat]）"),
    }
    # 相对坐标输入：把结果还原回用户自己的坐标系
    lo = payload.get("_local_origin")
    if lo:
        clat = math.cos(math.radians(lo["lat"]))
        center_obj["local_xy"] = {
            "x": round((lng - lo["lng"]) * 111320.0 * clat, 1),
            "y": round((lat - lo["lat"]) * 110540.0, 1),
            "origin": lo,
            "note": "输入为相对米制坐标；lat/lng 由伪原点换算而来，请以 local_xy 为准",
        }

    result = {
        "ok": True,
        "area_id": payload.get("area_id"),
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        "data_profile": prof,
        "method_selection": {"mode": sel["mode"], "used": list(fields),
                             "weights": w, "rejected": sel["rejected"],
                             "single_mode_penalty": len(fields) == 1},
        "fuzzy_center": center_obj,
        "capacity": cap,
        "cross_validation": {
            "methods": validate.method_consistency(grid, fields, per, fmu, center,
                                                   f["radius_90_m"]),
            "implementations": validate.implementation_consistency(per),
            "capacity": validate.capacity_consistency(per),
        },
        "per_method": per,
        "recommendations": profile.recommend(prof),
        "warnings": warnings,
    }

    if with_checks:
        result["cross_validation"]["monotonicity"] = validate.monotonic_check(
            payload, grid, center, r_service, capacity_mod.evaluate)

    result["verdict"] = _verdict(result)
    return result


def _verdict(res: dict) -> dict:
    cv = res["cross_validation"]
    m, i, c = cv["methods"], cv["implementations"], cv["capacity"]
    parts = [m["score"]]
    if i["summary"]["all_consistent"] is not None:
        parts.append(0.9 if i["summary"]["all_consistent"] else 0.55)
    if c["consistent"] is not None:
        parts.append(1.0 if c["consistent"] else 0.6)

    n_methods = len(res["method_selection"]["used"])
    # 方法多样性上限：1 个方法再自洽也不能算高置信
    cap = 0.45 if n_methods <= 1 else (0.72 if n_methods == 2 else 1.0)
    s = min(sum(parts) / len(parts), cap)

    level = "high" if s >= 0.75 else ("medium" if s >= 0.55 else "low")
    return {"score": round(s, 3), "level": level, "n_methods": n_methods,
            "diversity_cap": cap,
            "advice": {"high": "可直接作为推荐点位与容量方案使用",
                       "medium": "可用，但需在方案中标注区间与假设",
                       "low": "不建议直接采用；先补齐 divergence 指向的数据"}[level]}
