"""容量引擎：Erlang-C 排队 + 档位选择 + 蒙特卡洛区间。

设计：任何选址方法只要能给出（中心，服务圈，需求），就必然产出容量配置。
"""
from __future__ import annotations

import math

import numpy as np

from .contract import DEFAULT_HOUR_PROFILE, TIER_CATALOG, get
from .geo import Grid, points_xy


# ------------------------------------------------------------------ 排队论
def erlang_c(c: int, a: float) -> float:
    """M/M/c 排队概率 P(wait>0)，a=λ/μ。"""
    if a >= c:
        return 1.0
    term, s = 1.0, 1.0
    for k in range(1, c):
        term *= a / k
        s += term
    tail = term * (a / c) * (c / (c - a))
    return float(min(1.0, tail / (s + tail)))


def required_staff(lam: float, service_time_s: float, wait_target_s: float,
                   fail_prob: float, c_max: int = 40) -> tuple[int, float, float]:
    """反解最小服务台数 c s.t. P(wait)≤fail_prob 且 Wq≤wait_target_s。"""
    mu = 1.0 / service_time_s
    a = lam / mu
    for c in range(max(1, int(math.ceil(a)) + 1), c_max + 1):
        pw = erlang_c(c, a)
        wq = pw / (c * mu - lam) if c * mu > lam else float("inf")
        if pw <= fail_prob and wq <= wait_target_s:
            return c, pw, wq
    pw = erlang_c(c_max, a)
    wq = pw / (c_max * mu - lam) if c_max * mu > lam else float("inf")
    return c_max, pw, wq


# ------------------------------------------------------------------ 需求来源
def demand_series(payload: dict, grid: Grid, center: tuple[float, float],
                  radius: float) -> dict:
    xy = points_xy(payload.get("demand_points") or [], grid.proj)
    if len(xy):
        h = np.array([float(p.get("qty_daily", 0.0)) for p in payload["demand_points"]], float)
        d = np.linalg.norm(xy - np.array(center), axis=1)
        inside = d <= radius
        if not inside.any():                       # 兜底：取最近 3 个点
            inside = np.zeros(len(d), bool)
            inside[np.argsort(d)[:3]] = True
        return {"qty": float(h[inside].sum()), "n_points": int(inside.sum()),
                "source": "demand_points"}

    q = payload.get("proxy_daily_demand")
    if q is not None:
        return {"qty": float(q), "n_points": 0, "source": "proxy"}
    hh = payload.get("area_households")
    if hh is not None:
        q = (float(hh) * float(get(payload, "penetration"))
             * float(get(payload, "per_household_daily")))
        return {"qty": q, "n_points": 0, "source": "households_proxy"}
    return {"qty": None, "n_points": 0, "source": "none"}


def hour_profile(payload: dict) -> np.ndarray:
    prof = payload.get("hour_profile")
    if prof:
        p = np.asarray(prof, float)
        return p / p.sum()
    rows, ws = [], []
    for p in payload.get("demand_points") or []:
        q = p.get("qty_by_hour")
        if q:
            q = np.asarray(q, float)
            rows.append(q / max(q.sum(), 1e-9))
            ws.append(float(p.get("qty_daily", 1.0)))
    if rows:
        return np.average(np.vstack(rows), axis=0, weights=np.asarray(ws) / np.sum(ws))
    return DEFAULT_HOUR_PROFILE


# ------------------------------------------------------------------ 容量分配
def allocate(payload: dict, qty: float, profile: np.ndarray,
             tiers: list[dict]) -> dict:
    peak_factor = float(get(payload, "peak_factor"))
    self_pick = payload.get("self_pick_ratio")
    self_pick = 1.0 if self_pick is None else min(max(float(self_pick), 0.05), 1.0)
    s_time = float(get(payload, "service_time_s"))
    tau = float(get(payload, "queue_wait_target_s"))
    eps = float(get(payload, "queue_fail_prob"))
    turn = float(get(payload, "turnover_per_grid"))
    veh_cap = float(get(payload, "vehicle_daily_capacity"))
    rob_share = float(get(payload, "robot_share") or 0.0)
    rob_cap = float(get(payload, "robot_daily_capacity"))
    ch_min = float(get(payload, "charger_minutes"))
    ch_avail = float(get(payload, "charger_available_min"))

    peak_share = float(profile.max())
    peak_day = qty * peak_factor
    peak_hour = peak_day * peak_share

    # 波动缓冲：常规日残差越大，配置越要靠上，避免"P50 刚好、实际常超"。
    # 放大比例由 cv_margin 控制（默认 0.75，可在参数区改），CV 本身封顶 0.4。
    cv_raw = float(payload.get("volatility_cv") or 0.0)
    cv = min(max(cv_raw, 0.0), 0.4)
    margin_k = payload.get("cv_margin")
    margin_k = 0.75 if margin_k is None else min(max(float(margin_k), 0.0), 2.0)
    safety = 1.0 + cv * margin_k
    peak_day_design = peak_day * safety
    peak_hour_design = peak_hour * safety

    grids_need = peak_day_design * self_pick / max(turn, 1e-6)

    # 人力 = 分拣/理货（按日件量摊） + 取件窗口（排队反解，只服务自提部分）
    sort_per = float(get(payload, "sort_per_staff_day") or 1200.0)
    sort_staff = max(1, math.ceil(qty / sort_per))
    counter_lam = peak_hour_design * self_pick / 3600.0
    counter_staff, pw, wq = required_staff(counter_lam, s_time, tau, eps)
    staff = sort_staff + counter_staff

    pick = next((t for t in tiers if t["daily_cap"] >= peak_day_design
                 and t["peak_hour_cap"] >= peak_hour_design
                 and t["grids"] >= grids_need), None)
    overflow = pick is None
    pick = pick or max(tiers, key=lambda t: t["daily_cap"])

    veh = math.ceil(qty / max(veh_cap, 1e-6))
    rob = math.ceil(qty * rob_share / max(rob_cap, 1e-6)) if rob_share > 0 else 0
    chg = math.ceil(veh * ch_min / max(ch_avail, 1e-6)) if veh else 0

    return {
        "tier": pick["id"], "kind": pick["kind"], "grids": pick["grids"],
        "area_sqm": pick["area_sqm"], "daily_cap": pick["daily_cap"],
        "grids_need": round(grids_need, 1),
        "peak_day_demand": round(peak_day, 1),
        "peak_day_design": round(peak_day_design, 1),
        "peak_hour_demand": round(peak_hour, 1),
        "volatility_cv": round(cv, 3),
        "safety_margin": round(safety, 3),
        "cv_margin": round(margin_k, 3),
        "staff": staff, "staff_breakdown": {"sorting": sort_staff,
                                            "counter": counter_staff},
        "vehicles": veh, "robots": rob, "chargers": chg,
        "utilization": round(peak_day / max(pick["daily_cap"], 1e-6), 3),
        "queue": {"p_wait": round(pw, 3), "wq_s": round(wq, 1),
                  "target_wq_s": tau, "fail_prob": eps},
        "overflow": bool(overflow),
    }


def evaluate(payload: dict, grid: Grid, center: tuple[float, float],
             radius: float, n_draw: int = 250) -> dict:
    ds = demand_series(payload, grid, center, radius)
    if ds["qty"] is None:
        return {"feasible": False, "status": "blocked",
                "reason": "既无需求面也无代理件量，无法产出容量",
                "needed": ["demand_points", "proxy_daily_demand", "area_households"]}

    profile = hour_profile(payload)
    tiers = list(payload.get("tier_catalog") or TIER_CATALOG)
    base = allocate(payload, ds["qty"], profile, tiers)

    rng = np.random.default_rng(23)
    draws = rng.lognormal(0.0, 0.10, size=n_draw) * ds["qty"]
    hits = [allocate(payload, float(v), profile, tiers)["tier"] for v in draws]
    top = max(set(hits), key=hits.count)
    q10, q90 = np.percentile(draws, [10, 90])

    return {
        "feasible": not base["overflow"],
        "status": "ok" if not base["overflow"] else "capacity_overflow",
        "demand_source": ds["source"],
        "demand_points_in_radius": ds["n_points"],
        "service_radius_m": round(radius, 1),
        "daily_demand_p50": round(float(np.median(draws)), 1),
        "daily_demand_ci90": [round(float(q10), 1), round(float(q90), 1)],
        "tier": base["tier"], "kind": base["kind"],
        "tier_hit_rate": round(hits.count(top) / n_draw, 3),
        "tier_boundary_risk": bool(hits.count(top) / n_draw < 0.75),
        "grids": base["grids"], "grids_need": base["grids_need"],
        "area_sqm": base["area_sqm"], "daily_cap": base["daily_cap"],
        "peak_day_demand": base["peak_day_demand"],
        "peak_day_design": base["peak_day_design"],
        "peak_hour_demand": base["peak_hour_demand"],
        "volatility_cv": base["volatility_cv"],
        "safety_margin": base["safety_margin"],
        "cv_margin": base["cv_margin"],
        "staff": base["staff"], "staff_breakdown": base["staff_breakdown"],
        "staff_note": "峰值在岗人数 = 分拣理货（按日件量）+ 取件窗口（排队反解）",
        "vehicles": base["vehicles"],
        "robots": base["robots"], "chargers": base["chargers"],
        "utilization": base["utilization"], "queue": base["queue"],
        "split_suggestion": base["overflow"],
        "note": "" if not base["overflow"] else "需求超出最大档位，建议拆分为 2 个节点",
    }
