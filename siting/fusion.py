"""μ 场融合 → 模糊选址中心。"""
from __future__ import annotations

import math

import numpy as np

from .geo import Grid, local_peaks, mass_radius, normalise


def combine(grid: Grid, fields: dict[str, np.ndarray], weights: dict[str, float],
            single_mode: bool = False) -> tuple[np.ndarray, dict]:
    mu = np.zeros_like(grid.X)
    for key, f in fields.items():
        mu += float(weights.get(key, 0.0)) * f
    mu = normalise(mu / max(sum(weights.values()), 1e-12))

    # 峰值 + 邻域质心细化（避免格点量化误差）
    px, py = grid.peak(mu)
    ix = int(np.argmin(np.abs(grid.gx - px)))
    iy = int(np.argmin(np.abs(grid.gy - py)))
    y0, y1 = max(0, iy - 2), min(grid.ny, iy + 3)
    x0, x1 = max(0, ix - 2), min(grid.nx, ix + 3)
    win = mu[y0:y1, x0:x1]
    ws = float(win.sum()) or 1.0
    cx = float((grid.X[y0:y1, x0:x1] * win).sum() / ws)
    cy = float((grid.Y[y0:y1, x0:x1] * win).sum() / ws)

    if single_mode:                  # 单方法降级：放宽模糊区
        mu = normalise(mu ** 0.65)

    r68 = mass_radius(grid, mu, cx, cy, 0.68)
    r90 = mass_radius(grid, mu, cx, cy, 0.90)

    peaks = local_peaks(mu, grid)
    alts = [p for p in peaks
            if p["mu"] >= 0.55 and math.hypot(p["x_m"] - cx, p["y_m"] - cy) > 300.0]

    return mu, {
        "center": (cx, cy),
        "mu_peak": float(mu.max()),
        "radius_68_m": float(r68),
        "radius_90_m": float(r90),
        "bimodal": bool(alts),
        "alt_centers": alts[:2],
        "entropy": float(_entropy(mu)),
    }


def _entropy(mu: np.ndarray) -> float:
    p = mu.ravel()
    p = p / max(p.sum(), 1e-12)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / math.log(max(len(p), 2)))
