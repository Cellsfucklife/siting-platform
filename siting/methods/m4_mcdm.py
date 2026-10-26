"""M4 多准则：TOPSIS（熵权 / 等权），基于候选点属性或外部准则矩阵。"""
from __future__ import annotations

import numpy as np

from ..contract import CRIT_KEYS, get
from ..geo import Grid, normalise, points_xy
from .base import MethodSpec


def _criteria_matrix(payload: dict) -> tuple[np.ndarray, list[str], list[bool]]:
    cd = payload["candidates"]
    crit = payload.get("criteria")
    if isinstance(crit, dict) and crit.get("matrix"):
        return (np.asarray(crit["matrix"], float),
                list(crit.get("keys", [f"c{i}" for i in range(len(crit["matrix"][0]))])),
                list(crit.get("benefit", [True] * len(crit["matrix"][0]))))
    if isinstance(crit, (list, tuple)) and crit:
        M = np.asarray(crit, float)
        return M, [f"c{i}" for i in range(M.shape[1])], [True] * M.shape[1]
    keys = [k for k in CRIT_KEYS if any(k in c for c in cd)]
    if len(keys) < 3:
        raise ValueError(f"M4 需要 ≥3 个准则，当前 {len(keys)}：{keys}")
    M = np.asarray([[float(c.get(k, 0.0)) for k in keys] for c in cd], float)
    return M, keys, [k not in ("rent_month", "competition") for k in keys]


def entropy_weights(M: np.ndarray) -> np.ndarray:
    P = M / (M.sum(axis=0, keepdims=True) + 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.where(P > 0, P * np.log(P + 1e-12), 0.0)
    e = -L.sum(axis=0) / np.log(max(len(M), 2))
    d = 1.0 - e
    return d / (d.sum() + 1e-12)


def topsis(M: np.ndarray, benefit: list[bool], weights: np.ndarray) -> np.ndarray:
    norm = M / np.sqrt((M ** 2).sum(axis=0, keepdims=True) + 1e-12)
    for j, b in enumerate(benefit):
        if not b:                                   # 成本型取补
            norm[:, j] = norm[:, j].min() + norm[:, j].max() - norm[:, j]
    v = norm * weights
    best, worst = v.max(axis=0), v.min(axis=0)
    d_pos = np.linalg.norm(v - best, axis=1)
    d_neg = np.linalg.norm(v - worst, axis=1)
    return d_neg / (d_pos + d_neg + 1e-12)


def run(payload: dict, grid: Grid, engine: str = "auto") -> dict:
    M, keys, benefit = _criteria_matrix(payload)
    w = entropy_weights(M) if len(M) > 2 else np.ones(M.shape[1]) / M.shape[1]
    cc = topsis(M, benefit, w)

    cd = payload["candidates"]
    cxy = points_xy(cd, grid.proj)
    p = np.exp(8.0 * cc)
    p = p / p.sum()

    width = 140.0
    if len(cxy) > 1:
        d = np.linalg.norm(cxy[:, None, :] - cxy[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        width = max(90.0, float(np.median(d.min(axis=1))))

    mu = np.zeros_like(grid.X)
    for j in range(len(cd)):
        mu += p[j] * np.exp(-0.5 * (((grid.X - cxy[j, 0]) / width) ** 2
                                    + ((grid.Y - cxy[j, 1]) / width) ** 2))
    top = int(np.argmax(cc))
    order = np.argsort(-cc)

    return {
        "key": "M4_mcdm",
        "mu": normalise(mu),
        "center": (float(cxy[top, 0]), float(cxy[top, 1])),
        "sigma": (width, width),
        "detail": {"topsis_top": {cd[j]["id"]: round(float(cc[j]), 3) for j in order[:5]},
                   "criteria": keys,
                   "weights": {k: round(float(v), 3) for k, v in zip(keys, w)},
                   "weighting": "entropy" if len(M) > 2 else "equal"},
        "backends": {
            "builtin": {"impl": "熵权-TOPSIS（自研）"},
            "reference": {"impl": "无第三方等价实现（方法本身即闭式计算）"},
            "agreement": {"consistent": None,
                          "robustness": "建议对权重 ±20% 扰动做排序稳定性检验"},
        },
    }


SPEC = MethodSpec(
    key="M4_mcdm", name="多准则排序（熵权-TOPSIS over 候选）", family="半定性评价型",
    needs=("candidates", "criteria"), prior=0.22,
    assumption="属性偏好可表达；与需求几何弱相关",
    ref_impl="熵权 + TOPSIS（闭式，无外部依赖）",
    run=run,
)
