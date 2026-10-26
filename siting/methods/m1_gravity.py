"""M1 连续重心：Weber（Σh·d 最小）—— Weiszfeld 自研 vs SciPy 参考。"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ..geo import Grid, normalise, points_xy
from .base import MethodSpec, agreement_point


def _demand(payload: dict, grid: Grid):
    dp = payload["demand_points"]
    xy = points_xy(dp, grid.proj)
    h = np.array([float(p.get("qty_daily", 1.0)) for p in dp], float)
    return xy, h


def weiszfeld(xy: np.ndarray, h: np.ndarray, iters: int = 500,
              tol: float = 1e-7) -> np.ndarray:
    """自研实现：Σ h_i·‖x−p_i‖ 的迭代解（Weber 问题）。"""
    w0 = h / h.sum()
    x = (xy * w0[:, None]).sum(axis=0)
    for _ in range(iters):
        d = np.maximum(np.linalg.norm(xy - x, axis=1), 1e-9)
        w = h / d
        x_new = (xy * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    return x


def _objective(x: np.ndarray, xy: np.ndarray, h: np.ndarray) -> float:
    return float((h * np.linalg.norm(xy - x, axis=1)).sum())


def run(payload: dict, grid: Grid, engine: str = "auto") -> dict:
    xy, h = _demand(payload, grid)

    # —— 自研：Weiszfeld ——
    cb = weiszfeld(xy, h)
    obj_b = _objective(cb, xy, h)

    # —— 参考：SciPy Nelder-Mead 直接最小化同一目标 ——
    r = minimize(lambda v: _objective(v, xy, h), x0=cb + 50.0,
                 method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-4,
                                                "maxiter": 4000})
    cr = np.asarray(r.x, float)
    obj_r = _objective(cr, xy, h)

    center = cb if engine != "reference" else cr

    # —— μ 场：ε-最优区域（目标值相对最优的容差带）——
    # 不做"中心 + 固定 σ"的高斯，而是用目标函数地形本身定义可接受区域，
    # 这样模糊区的尺度由成本曲线的曲率决定，而不是拍一个核宽度。
    beta = 0.045                                  # μ 衰减尺度（相对目标比）
    obj_grid = np.zeros_like(grid.X)
    for (px, py), hi in zip(xy, h):
        obj_grid += hi * np.sqrt((grid.X - px) ** 2 + (grid.Y - py) ** 2)
    ratio = obj_grid / max(obj_b, 1e-9)
    mu = np.exp(-(ratio - 1.0) / beta)

    # 等效尺度（报告用）：μ≥0.5 区的等效半径
    mask = mu >= 0.5
    if mask.any():
        dx = grid.X[mask] - center[0]
        dy = grid.Y[mask] - center[1]
        s_eq = float(np.sqrt((dx ** 2 + dy ** 2).mean()))
    else:
        s_eq = 100.0

    rms = float(np.sqrt(np.average(((xy - center) ** 2).sum(axis=1), weights=h)))

    return {
        "key": "M1_gravity",
        "mu": normalise(mu),
        "center": (float(center[0]), float(center[1])),
        "sigma": (max(s_eq, 60.0), max(s_eq, 60.0)),
        "detail": {"objective": round(obj_b, 1),
                   "equivalent_radius_m": round(s_eq, 1),
                   "demand_rms_radius_m": round(rms, 1),
                   "tolerance_beta": beta,
                   "note": "μ=exp(-(目标比-1)/β)，即 Weber 目标的 ε-最优区域，不是高斯核"},
        "backends": {
            "builtin": {"impl": "Weiszfeld 迭代", "center": [round(cb[0], 2), round(cb[1], 2)],
                        "objective": round(obj_b, 2)},
            "reference": {"impl": "SciPy Nelder-Mead", "center": [round(cr[0], 2), round(cr[1], 2)],
                          "objective": round(obj_r, 2), "success": bool(r.success)},
            "agreement": {**agreement_point(tuple(cb), tuple(cr)),
                          "objective_gap_pct": round(abs(obj_b - obj_r) / max(obj_r, 1e-9) * 100, 4)},
        },
    }


SPEC = MethodSpec(
    key="M1_gravity", name="连续重心（Weber 加权最小二乘）", family="连续型",
    needs=("demand_points",), prior=0.22,
    assumption="运费∝需求×直线距离；无候选约束、无容量、无路网",
    ref_impl="Weiszfeld 迭代 vs scipy.optimize.minimize(Nelder-Mead)",
    run=run,
)
