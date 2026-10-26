"""局部米制投影、网格、凸包、几何工具。"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import ConvexHull, QhullError

M_PER_DEG_LAT = 110540.0
M_PER_DEG_LNG = 111320.0


def projection(pts: list[dict]) -> tuple[float, float, float]:
    lng0 = float(np.mean([p["lng"] for p in pts]))
    lat0 = float(np.mean([p["lat"] for p in pts]))
    return lng0, lat0, math.cos(math.radians(lat0))


def to_xy(p: dict, lng0: float, lat0: float, clat: float) -> tuple[float, float]:
    return ((p["lng"] - lng0) * M_PER_DEG_LNG * clat,
            (p["lat"] - lat0) * M_PER_DEG_LAT)


def to_latlon(x: float, y: float, lng0: float, lat0: float, clat: float) -> tuple[float, float]:
    """返回 (lat, lng)。"""
    return lat0 + y / M_PER_DEG_LAT, lng0 + x / (M_PER_DEG_LNG * clat)


def points_xy(items: list[dict], proj: tuple[float, float, float]) -> np.ndarray:
    if not items:
        return np.zeros((0, 2))
    return np.array([to_xy(p, *proj) for p in items], dtype=float)


class Grid:
    """规则米制网格。所有方法都在同一张格上产出 μ(x)。"""

    def __init__(self, payload: dict, cells_hint: int = 140, min_cell: float = 50.0):
        pts = list(payload.get("demand_points") or []) + list(payload.get("candidates") or [])
        if not pts:
            pts = [{"lat": 39.95, "lng": 116.40}]
        self.proj = projection(pts)
        xy = points_xy(pts, self.proj)
        span = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 800.0)
        pad = max(500.0, 0.40 * span)
        x0, x1 = float(xy[:, 0].min()) - pad, float(xy[:, 0].max()) + pad
        y0, y1 = float(xy[:, 1].min()) - pad, float(xy[:, 1].max()) + pad
        self.cell = max(min_cell, max(x1 - x0, y1 - y0) / cells_hint)
        self.nx = int(math.ceil((x1 - x0) / self.cell))
        self.ny = int(math.ceil((y1 - y0) / self.cell))
        self.x0, self.y0 = x0, y0
        self.gx = x0 + (np.arange(self.nx) + 0.5) * self.cell
        self.gy = y0 + (np.arange(self.ny) + 0.5) * self.cell
        self.X, self.Y = np.meshgrid(self.gx, self.gy)
        self.diag = float(math.hypot(x1 - x0, y1 - y0))

    def gaussian(self, cx: float, cy: float, sx: float, sy: float) -> np.ndarray:
        mu = np.exp(-0.5 * (((self.X - cx) / sx) ** 2 + ((self.Y - cy) / sy) ** 2))
        return normalise(mu)

    def distances_to(self, cx: float, cy: float) -> np.ndarray:
        return np.sqrt((self.X - cx) ** 2 + (self.Y - cy) ** 2)

    def peak(self, mu: np.ndarray) -> tuple[float, float]:
        iy, ix = np.unravel_index(int(np.argmax(mu)), mu.shape)
        return float(self.gx[ix]), float(self.gy[iy])

    def to_latlon(self, x: float, y: float) -> tuple[float, float]:
        return to_latlon(x, y, *self.proj)


def normalise(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    m = float(a.max()) if a.size else 0.0
    return a / m if m > 0 else a


def hull_lnglat(grid: Grid, pts_xy: np.ndarray) -> list[list[float]]:
    """μ≥阈值格点的凸包，输出 GeoJSON 顺序 [lng, lat]。"""
    if len(pts_xy) < 3:
        return []
    try:
        v = ConvexHull(pts_xy)
        ring = pts_xy[v.vertices]
    except QhullError:
        return []
    out = []
    for x, y in ring:
        lat, lng = grid.to_latlon(float(x), float(y))
        out.append([round(lng, 6), round(lat, 6)])
    if out:
        out.append(out[0])
    return out


def mass_radius(grid: Grid, mu: np.ndarray, cx: float, cy: float,
                quantile: float) -> float:
    """以 (cx,cy) 为中心，μ 质量累积到 quantile 处的半径。"""
    d = grid.distances_to(cx, cy).ravel()
    m = mu.ravel()
    order = np.argsort(d)
    cum = np.cumsum(m[order]) / max(m.sum(), 1e-12)
    idx = int(np.searchsorted(cum, quantile))
    idx = min(idx, len(d) - 1)
    return float(d[order][idx])


def local_peaks(mu: np.ndarray, grid: Grid, thr: float = 0.35) -> list[dict]:
    """局部极大值（带邻域抑制），用于多峰告警。"""
    cand = []
    ny, nx = mu.shape
    for iy in range(1, ny - 1):
        for ix in range(1, nx - 1):
            v = mu[iy, ix]
            if v >= thr and v >= mu[iy - 1:iy + 2, ix - 1:ix + 2].max() - 1e-12:
                cand.append((v, float(grid.gx[ix]), float(grid.gy[iy])))
    cand.sort(reverse=True)
    kept: list[dict] = []
    for v, x, y in cand:
        if all(math.hypot(x - k["x_m"], y - k["y_m"]) > max(300.0, 3 * grid.cell)
               for k in kept):
            kept.append({"mu": float(v), "x_m": x, "y_m": y})
        if len(kept) >= 4:
            break
    return kept
