"""M3 核密度聚集：sklearn KernelDensity 参考 vs 手写高斯核自研。"""
from __future__ import annotations

import numpy as np

from ..geo import Grid, normalise, points_xy
from .base import MethodSpec, agreement_field, agreement_point


def _bw(xy: np.ndarray, h: np.ndarray, grid: Grid) -> float:
    """Scott 规则的加权版本。"""
    n = max(len(xy), 1)
    area = (grid.nx * grid.cell) * (grid.ny * grid.cell)
    return max(90.0, float(np.sqrt(area / n)) * 0.85)


def run(payload: dict, grid: Grid, engine: str = "auto") -> dict:
    dp = payload["demand_points"]
    xy = points_xy(dp, grid.proj)
    h = np.array([float(p.get("qty_daily", 1.0)) for p in dp], float)
    h = h / h.sum()
    bw = _bw(xy, h, grid)

    # —— 参考：sklearn ——
    from sklearn.neighbors import KernelDensity
    kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(xy, sample_weight=h)
    logd = kde.score_samples(np.column_stack([grid.X.ravel(), grid.Y.ravel()]))
    mu_ref = normalise(np.exp(logd).reshape(grid.X.shape))

    # —— 自研：手写高斯核 ——
    mu_bi = np.zeros_like(grid.X)
    for (x, y), w in zip(xy, h):
        mu_bi += w * np.exp(-0.5 * (((grid.X - x) / bw) ** 2 + ((grid.Y - y) / bw) ** 2))
    mu_bi = normalise(mu_bi)

    mu = mu_ref if engine == "reference" else mu_bi
    cx, cy = grid.peak(mu)
    cx_r, cy_r = grid.peak(mu_ref)

    return {
        "key": "M3_kde",
        "mu": mu,
        "center": (cx, cy),
        "sigma": (bw, bw),
        "detail": {"bandwidth_m": round(bw, 1), "weighting": "件量归一化"},
        "backends": {
            "builtin": {"impl": "手写高斯核叠加"},
            "reference": {"impl": "sklearn KernelDensity(gaussian)",
                          "center": [round(cx_r, 2), round(cy_r, 2)]},
            "agreement": {**agreement_field(mu_bi, mu_ref),
                          **agreement_point((cx, cy), (cx_r, cy_r))},
        },
    }


SPEC = MethodSpec(
    key="M3_kde", name="核密度聚集（需求热点峰）", family="空间统计型",
    needs=("demand_points",), prior=0.22,
    assumption="需求空间聚集即最优；不含成本与路网信息",
    ref_impl="sklearn.neighbors.KernelDensity",
    run=run,
)
