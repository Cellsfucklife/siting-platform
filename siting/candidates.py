"""候选点生成器：把「找哪些场地可以租」变成「生成候选 + 事后筛选」。

核心判断：
    选址模型对**候选点数量**不敏感，对**候选点覆盖区域的完整性**敏感。
    只要候选集覆盖了片区里"可能的好位置"，最优点就稳定 ——
    因此不需要枚举全部真实房源，先生成再落地。

三种生成方式：
    A. grid_candidates      规则网格 / H3 六边形（最省力，默认）
    B. anchor_candidates    以现有网点、社区出入口、主要路口为锚，环形布点
    C. existing_candidates  直接抓同城已有的驿站/快递柜位置（已被人验证过）

配套：
    density_test()          候选集加密测试 —— 用数据说明"多少个候选就够了"
"""
from __future__ import annotations

import math

import numpy as np

M_PER_DEG_LAT = 110540.0
M_PER_DEG_LNG = 111320.0


def _origin(payload: dict) -> tuple[float, float, float]:
    pts = list(payload.get("demand_points") or []) + list(payload.get("candidates") or [])
    if not pts:
        raise ValueError("需要至少一个需求点来确定片区中心")
    lng0 = float(np.mean([p["lng"] for p in pts]))
    lat0 = float(np.mean([p["lat"] for p in pts]))
    return lng0, lat0, math.cos(math.radians(lat0))


def _to_lnglat(x: float, y: float, o: tuple[float, float, float]) -> tuple[float, float]:
    lng0, lat0, clat = o
    return lat0 + y / M_PER_DEG_LAT, lng0 + x / (M_PER_DEG_LNG * clat)


def _demand_centroid(payload: dict) -> tuple[float, float, float, float]:
    """件量加权的需求重心与 RMS 半径（米）。"""
    o = _origin(payload)
    lng0, lat0, clat = o
    pts = payload["demand_points"]
    xy = np.array([[(p["lng"] - lng0) * M_PER_DEG_LNG * clat,
                    (p["lat"] - lat0) * M_PER_DEG_LAT] for p in pts])
    h = np.array([float(p.get("qty_daily", 1.0)) for p in pts])
    w = h / h.sum()
    c = (xy * w[:, None]).sum(axis=0)
    rms = float(np.sqrt((((xy - c) ** 2).sum(axis=1) * w).sum()))
    return float(c[0]), float(c[1]), rms, clat


def grid_candidates(payload: dict, spacing_m: float = 180.0,
                    radius_factor: float = 1.8, prefix: str = "G") -> list[dict]:
    """在需求重心周围生成规则网格候选（圆形裁剪）。

    默认半径 = 1.8 × 需求 RMS 半径，保证覆盖"比需求簇稍大"的范围。
    经验：一个 1–2 km² 片区，spacing=150–200 m 时约 8–20 个候选，足够。
    """
    o = _origin(payload)
    cx, cy, rms, _ = _demand_centroid(payload)
    R = max(radius_factor * rms, 400.0)
    n = int(math.ceil(R / spacing_m))
    out = []
    k = 0
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            x, y = cx + i * spacing_m, cy + j * spacing_m
            if math.hypot(x - cx, y - cy) > R:
                continue
            lat, lng = _to_lnglat(x, y, o)
            out.append({"id": f"{prefix}{k:02d}", "lat": round(lat, 6), "lng": round(lng, 6),
                        "synthetic": True, "_x": x, "_y": y})
            k += 1
    return out


def h3_candidates(payload: dict, resolution: int = 9,
                  radius_factor: float = 1.8, prefix: str = "H") -> list[dict]:
    """H3 六边形候选（六边形比方格的距离各向同性更好）。需 pip install h3。"""
    import h3
    o = _origin(payload)
    cx, cy, rms, _ = _demand_centroid(payload)
    R = max(radius_factor * rms, 400.0)
    clat, clng = _to_lnglat(cx, cy, o)
    center_cell = h3.latlng_to_cell(clat, clng, resolution)
    out = []
    for k, cell in enumerate(h3.grid_disk(center_cell, 3)):
        la, ln = h3.cell_to_latlng(cell)
        x = (ln - o[0]) * M_PER_DEG_LNG * o[2]
        y = (la - o[1]) * M_PER_DEG_LAT
        if math.hypot(x - cx, y - cy) > R:
            continue
        out.append({"id": f"{prefix}{k:02d}", "lat": round(la, 6), "lng": round(ln, 6),
                    "synthetic": True, "_x": x, "_y": y})
    return out


def anchor_candidates(payload: dict, anchors: list[dict] | None = None,
                      radii_m: tuple[float, ...] = (300.0, 600.0),
                      n_bearings: int = 8, prefix: str = "A") -> list[dict]:
    """以"锚点"为中心环形布点。

    锚点通常是：现有配送站/驿站、社区主要出入口、临街主干道路口。
    这类位置已经通过了"可达性 + 有铺面"的现实筛选，命中率最高。
    """
    o = _origin(payload)
    if anchors is None:
        anchors = payload.get("candidates") or payload.get("demand_points") or []
    out, k = [], 0
    for a in anchors:
        ax = (a["lng"] - o[0]) * M_PER_DEG_LNG * o[2]
        ay = (a["lat"] - o[1]) * M_PER_DEG_LAT
        for r in radii_m:
            for b in range(n_bearings):
                th = 2 * math.pi * b / n_bearings
                x, y = ax + r * math.cos(th), ay + r * math.sin(th)
                lat, lng = _to_lnglat(x, y, o)
                out.append({"id": f"{prefix}{k:02d}", "lat": round(lat, 6), "lng": round(lng, 6),
                            "synthetic": True, "_x": x, "_y": y})
                k += 1
    return out


def thin(cands: list[dict], keep: int, seed: int = 5) -> list[dict]:
    """从密集候选集里抽样 keep 个（用于加密测试）。"""
    if len(cands) <= keep:
        return cands
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(cands), size=keep, replace=False)
    return [cands[i] for i in sorted(idx)]


def nested_subsets(cands: list[dict], ks: list[int], seed: int = 5) -> dict[int, list[dict]]:
    """打乱一次后取前缀 → 保证 K ⊂ K'（嵌套子集），否则加密测试会被抽样噪声污染。"""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cands))
    shuffled = [cands[i] for i in order]
    return {k: shuffled[:min(k, len(shuffled))] for k in ks}


def density_test(payload: dict, solve_fn, ks: int | list[int] = (6, 10, 20, 40),
                 spacing_m: float = 150.0, tol_m: float = 60.0,
                 tol_ratio: float = 0.10) -> dict:
    """候选集加密测试：候选数量从少到多，最优点会不会一直跑？

    方法论要点：
      · 子集**嵌套**（K ⊂ K'），避免抽样顺序造成的假位移
      · 参考解取候选最多的那一档
      · 容差 = max(tol_m, tol_ratio × r90) —— 用"相对模糊半径"判断才合理
      · sufficient_K = 最小的 K，使得**所有 K' ≥ K 都满足容差且档位一致**

    写进方案的说法：
        "候选数 ≥ K 后最优点游走 <10% r90、容量档不变 → 该规模下 K 个候选足够，
         无需枚举真实房源；房源筛选只需在最终模糊区内进行。"
    """
    if isinstance(ks, int):
        ks = [ks]
    ks = sorted(set(ks))
    base = dict(payload)
    base.pop("candidates", None)
    base.pop("dist_matrix", None)

    pool = grid_candidates(payload, spacing_m=spacing_m)
    subsets = nested_subsets(pool, ks)

    rows = []
    for k in ks:
        p = dict(base)
        p["candidates"] = subsets[k]
        res = solve_fn(p)
        if not res.get("ok"):
            rows.append({"k": len(subsets[k]), "ok": False, "error": res.get("errors")})
            continue
        fc, cp = res["fuzzy_center"], res["capacity"]
        rows.append({"k": len(subsets[k]), "ok": True,
                     "lat": fc["lat"], "lng": fc["lng"],
                     "r90_m": fc["radius_90_m"],
                     "tier": cp.get("tier"), "staff": cp.get("staff"),
                     "methods": len(res["method_selection"]["used"]),
                     "verdict": res["verdict"]["level"]})

    finite = [r for r in rows if r["ok"]]
    out = {"rows": rows, "pool_size": len(pool), "spacing_m": spacing_m}
    if len(finite) < 2:
        out["conclusion"] = "可用结果不足，无法做加密测试"
        return out

    ref = finite[-1]
    coslat = math.cos(math.radians(ref["lat"]))
    tol = max(tol_m, tol_ratio * ref["r90_m"])

    laps = []
    for r in finite:
        dx = (r["lng"] - ref["lng"]) * M_PER_DEG_LNG * coslat
        dy = (r["lat"] - ref["lat"]) * M_PER_DEG_LAT
        d = math.hypot(dx, dy)
        laps.append({"k": r["k"], "shift_m": round(d, 1),
                     "shift_pct_r90": round(d / max(ref["r90_m"], 1e-6) * 100, 1),
                     "same_tier": r["tier"] == ref["tier"]})

    ok_ks = [x["k"] for x in laps if x["shift_m"] <= tol and x["same_tier"]]
    sufficient = next((k for k in ks
                       if all(kk in ok_ks for kk in ks if kk >= k)), None)
    tier_stable = all(x["same_tier"] for x in laps)
    med_shift = float(np.median([x["shift_m"] for x in laps]))
    out.update({
        "reference_k": ref["k"], "tolerance_m": round(tol, 1),
        "ladder": laps, "sufficient_k": sufficient,
        "tier_stable": tier_stable,
        "median_shift_m": round(med_shift, 1),
        "shift_vs_spacing": round(med_shift / max(spacing_m, 1e-6), 2),
        "conclusion": _conclude(sufficient, ref, tol, tol_ratio, spacing_m,
                                med_shift, tier_stable),
    })
    return out


def _conclude(sufficient, ref, tol, tol_ratio, spacing, med_shift, tier_stable) -> str:
    if sufficient:
        head = (f"候选数 ≥ {sufficient} 后，最优点相对 {ref['k']} 个候选的游走 "
                f"≤{tol:.0f} m（<{tol_ratio * 100:.0f}% r90），且容量档不变。")
    else:
        head = ("加密到最大档最优点仍在游走：可能需要扩大候选**范围**（而不只是加密），"
                "或该片区呈多峰，应考虑拆成两个节点。")
    detail = (f"中位游走 {med_shift:.0f} m ≈ {med_shift / spacing:.1f} 倍候选间距 —— "
              f"说明位置分辨率由**候选间距**决定，而不是由候选数量决定。"
              if tier_stable else
              f"中位游走 {med_shift:.0f} m，且容量档发生过变化，需复核边界条件。")
    tail = ("因此不必枚举真实房源：把候选间距控制在 150–250 m、覆盖住需求簇外沿，"
            "最后只在融合模糊区内实地筛 3–5 处房源即可。")
    return head + detail + tail
