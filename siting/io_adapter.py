"""数据接入层：把「三张表 + 参数文件」装配成求解 payload。

这就是平台的真实入口 —— 前端/上游不需要懂 JSON 契约，
只要按模板交出三张 CSV：

    demand_points.csv   id,lat,lng,qty_daily
    candidates.csv      id,lat,lng,area_sqm,rent_month,<准则1..n>
    dist_matrix.csv     d_id,c_id,minutes          （长表，缺省则由直线×绕行系数替代）
    params.json         area_id / peak_factor / service_radius_min / ...

用法：
    from siting.io_adapter import load_scope
    payload = load_scope("examples")
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

M_PER_DEG_LAT = 110540.0
M_PER_DEG_LNG = 111320.0
# 只有相对坐标时使用的伪原点（不影响任何距离/方位计算）
LOCAL_ORIGIN = (39.9000, 116.4000)

# 这些列不当作"多准则指标"，而是空间/成本属性
NON_CRITERIA = {"id", "lat", "lng", "name", "addr", "area_sqm", "rent_month"}

# 候选点属性里可当准则的字段（与 M4 的 benefit 方向对应）
CRITERIA_BENEFIT = {
    "accessibility": True,
    "competition": False,          # 竞争越强越差
    "site_availability": True,
    "expandability": True,
    "policy": True,
    "rent_month": False,           # 租金越低越好
}


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return [{k.strip(): (v.strip() if isinstance(v, str) else v)
                 for k, v in row.items()} for row in csv.DictReader(fh)]


def _f(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _coords(r: dict) -> tuple[float | None, float | None]:
    """支持两种坐标：经纬度(lat/lng)，或相对米制坐标(x/y)。"""
    lat, lng = _f(r.get("lat")), _f(r.get("lng"))
    if lat is not None and lng is not None:
        return lat, lng
    x, y = _f(r.get("x")), _f(r.get("y"))
    if x is None or y is None:
        return None, None
    clat = math.cos(math.radians(LOCAL_ORIGIN[0]))
    return (LOCAL_ORIGIN[0] + y / M_PER_DEG_LAT,
            LOCAL_ORIGIN[1] + x / (M_PER_DEG_LNG * clat))


def load_scope(root: str | Path) -> dict:
    """读取一个片区的四份文件，返回可直接交给 solve() 的 payload。

    坐标列二选一：lat/lng（经纬度）或 x/y（米制相对坐标，原点任取）。
    用相对坐标时结果里会多返回 fuzzy_center.local_xy，保持原始坐标系。
    """
    root = Path(root)
    params = {}
    pfile = root / "params.json"
    if pfile.exists():
        params = json.loads(pfile.read_text(encoding="utf-8"))
        params = {k: v for k, v in params.items() if not k.startswith("_")}

    used_local = False

    # ---- 表1：需求点 ----
    demand = []
    for r in _read_csv(root / "demand_points.csv"):
        q = _f(r.get("qty_daily"))
        lat, lng = _coords(r)
        if q is None or lat is None:
            continue
        if "lat" not in r:
            used_local = True
        demand.append({"id": r["id"], "lat": lat, "lng": lng, "qty_daily": q})

    # ---- 表2：候选点（自动识别准则列）----
    cand = []
    crit_keys: list[str] = []
    for r in _read_csv(root / "candidates.csv"):
        lat, lng = _coords(r)
        if lat is None:
            continue
        if "lat" not in r:
            used_local = True
        c = {"id": r["id"], "lat": lat, "lng": lng}
        if _f(r.get("area_sqm")) is not None:
            c["area_sqm"] = _f(r["area_sqm"])
        if _f(r.get("rent_month")) is not None:
            c["rent_month"] = _f(r["rent_month"])
        for k, v in r.items():
            if k in NON_CRITERIA or k in ("id", "lat", "lng", "x", "y"):
                continue
            if _f(v) is not None:
                c[k] = _f(v)
                if k not in crit_keys:
                    crit_keys.append(k)
        cand.append(c)

    # ---- 表3：距离矩阵（长表 → 宽矩阵）----
    dist_rows = _read_csv(root / "dist_matrix.csv")
    dist = None
    if dist_rows and demand and cand:
        didx = {d["id"]: i for i, d in enumerate(demand)}
        cidx = {c["id"]: j for j, c in enumerate(cand)}
        M = [[None] * len(cand) for _ in demand]
        for r in dist_rows:
            i, j = didx.get(r.get("d_id")), cidx.get(r.get("c_id"))
            if i is None or j is None:
                continue
            M[i][j] = _f(r.get("minutes")) or _f(r.get("meters"))
        if all(v is not None for row in M for v in row):
            dist = M

    payload = {**params, "demand_points": demand, "candidates": cand}
    if dist is not None:
        payload["dist_matrix"] = dist
    if len(crit_keys) >= 3 and cand:
        payload["criteria"] = {
            "matrix": [[c.get(k, 0.0) for k in crit_keys] for c in cand],
            "keys": crit_keys,
            "benefit": [CRITERIA_BENEFIT.get(k, True) for k in crit_keys],
        }
    payload["_meta"] = {
        "root": str(root),
        "n_demand": len(demand), "n_candidates": len(cand),
        "has_dist_matrix": dist is not None,
        "criteria_keys": crit_keys,
        "coordinate_frame": "local_xy" if used_local else "wgs84",
        "files": {"demand": (root / "demand_points.csv").exists(),
                  "candidates": (root / "candidates.csv").exists(),
                  "dist": (root / "dist_matrix.csv").exists(),
                  "params": pfile.exists()},
    }
    if used_local:
        payload["_local_origin"] = {"lat": LOCAL_ORIGIN[0], "lng": LOCAL_ORIGIN[1]}
    return payload


def describe(payload: dict) -> str:
    """把 payload 打印成人能读的「输入清单」。"""
    m = payload.get("_meta", {})
    lines = [f"片区：{payload.get('area_id')} {payload.get('area_name', '')}",
             f"  需求点 {m.get('n_demand')} 个，合计 {sum(d['qty_daily'] for d in payload.get('demand_points', [])):.0f} 件/日",
             f"  候选点 {m.get('n_candidates')} 个",
             f"  距离矩阵：{'有（分钟）' if m.get('has_dist_matrix') else '无（将用直线×绕行系数）'}",
             f"  准则指标：{m.get('criteria_keys') or '无'}"]
    return "\n".join(lines)


if __name__ == "__main__":                          # pragma: no cover
    import json
    import sys

    from .solver import solve

    root = sys.argv[1] if len(sys.argv) > 1 else "examples"
    payload = load_scope(root)
    print(describe(payload))
    print("-" * 60)
    print(json.dumps(solve(payload), ensure_ascii=False, indent=2))
