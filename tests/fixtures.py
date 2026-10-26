"""测试用数据：四档数据画像（L3/L2/L1/L0），同一片社区只是字段多少不同。"""
from __future__ import annotations

import math

import numpy as np

from siting.contract import DEFAULT_HOUR_PROFILE

M_PER_DEG_LAT = 110540.0
M_PER_DEG_LNG = 111320.0


def build(level: int, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    lng0, lat0 = 116.400, 39.950

    dp = [{"id": f"D{i:02d}",
           "lat": lat0 + float(rng.normal(0, 0.0035)),
           "lng": lng0 + float(rng.normal(0, 0.0045)),
           "qty_daily": float(rng.integers(140, 320))} for i in range(8)]

    cd = [{"id": f"C{j:02d}",
           "lat": lat0 + float(rng.normal(0, 0.006)),
           "lng": lng0 + float(rng.normal(0, 0.007)),
           "area_sqm": float(rng.integers(40, 260)),
           "rent_month": float(rng.integers(8000, 36000)),
           "accessibility": float(rng.uniform(0.3, 1.0)),
           "competition": float(rng.uniform(0.0, 1.0)),
           "site_availability": float(rng.uniform(0.25, 1.0)),
           "expandability": float(rng.uniform(0.2, 1.0))} for j in range(7)]

    p: dict = {"area_id": f"TEST-L{level}", "demand_points": dp, "candidates": cd,
               "peak_factor": 1.6, "service_radius_min": 10.0,
               "access_speed_kmh": 12.0, "service_time_s": 45.0,
               "self_pick_ratio": 0.7}

    if level >= 2:
        p["hour_profile"] = list(DEFAULT_HOUR_PROFILE)
        D = []
        for a in dp:
            row = []
            for c in cd:
                dx = (a["lng"] - c["lng"]) * M_PER_DEG_LNG * math.cos(math.radians(lat0))
                dy = (a["lat"] - c["lat"]) * M_PER_DEG_LAT
                row.append(round(math.hypot(dx, dy) * 1.35 / (25000 / 60), 3))
            D.append(row)
        p["dist_matrix"] = D
    if level == 1:
        p.pop("candidates", None)
    if level == 0:
        p.pop("demand_points", None)
        p["proxy_daily_demand"] = 2100.0
    return p
