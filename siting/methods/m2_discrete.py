"""M2 离散候选：p-中值（spopt 参考）+ 容量约束 CFLP（PuLP 参考）+ 扰动自研。

定位策略：
  · 位置由 p-中值（spopt）决定 —— 标准选址模型
  · 档位由 CFLP（PuLP/CBC，含容量与场地约束）给出
  · 若两者位置不同，不作"不一致"，而输出「租金敏感度」诊断
  · 自研扰动给出 ε-最优集合，用于构造 μ 场（不确定度）
"""
from __future__ import annotations

import warnings

import numpy as np

from ..contract import TIER_CATALOG, get
from ..geo import Grid, normalise, points_xy
from .base import MethodSpec

warnings.filterwarnings("ignore")

UNIT_TRANSPORT_YUAN_PER_TKM = 0.45     # 元/件·公里，可外置到配置


def cbc_solver(msg: bool = False):
    """返回一个**确实可用**的 CBC 求解器。

    注意：pulp.COIN_CMD 只在 cbc 位于 PATH 时可用；pip 安装的 pulp 自带
    PULP_CBC_CMD 二进制。必须探测 listSolvers(onlyAvailable=True)，否则会在
    solve() 阶段抛错，而错误容易被上层吞掉 —— 参考实现就变成"静默失败"。
    """
    import pulp

    available = set(pulp.listSolvers(onlyAvailable=True))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if "COIN_CMD" in available:
            return pulp.COIN_CMD(msg=msg)
        if "PULP_CBC_CMD" in available:
            return pulp.PULP_CBC_CMD(msg=msg)
    raise RuntimeError(f"没有可用的 MIP 求解器，pulp.listSolvers={sorted(available)}；"
                       f"请 pip install 'pulp[cbc]'")


def solver_name() -> str:
    import pulp
    available = set(pulp.listSolvers(onlyAvailable=True))
    if "COIN_CMD" in available:
        return "COIN_CMD"
    if "PULP_CBC_CMD" in available:
        return "PULP_CBC_CMD"
    return "none"


# ---------------------------------------------------------------- 共用前处理
def _matrices(payload: dict, grid: Grid):
    dp = payload.get("demand_points") or []
    cd = payload["candidates"]
    dxy = points_xy(dp, grid.proj)
    cxy = points_xy(cd, grid.proj)
    h = np.array([float(p.get("qty_daily", 1.0)) for p in dp], float)

    dm = payload.get("dist_matrix")
    detour = float(get(payload, "road_detour_factor"))
    used = isinstance(dm, (list, tuple)) and len(dm) == len(dp) and len(dp) > 0
    if used:
        D = np.asarray(dm, float)
        if D.max() < 5.0:                       # 分钟 → 等效米
            D = D * (25_000.0 / 60.0)
    else:
        D = np.linalg.norm(dxy[:, None, :] - cxy[None, :, :], axis=2) * detour
    return dxy, cxy, h, D, used


# ---------------------------------------------------------------- 参考实现一：spopt
def reference_pmedian(D: np.ndarray, h: np.ndarray) -> dict:
    from spopt.locate import PMedian

    model = PMedian.from_cost_matrix(D.copy(), weights=h, p_facilities=1)
    model.solve(cbc_solver())
    sel = [j for j, v in enumerate(model.fac_vars) if v.varValue and v.varValue > 0.5]
    return {"impl": "spopt.locate.PMedian (p=1, CBC)",
            "index": int(sel[0]) if sel else None,
            "objective": float(model.problem.objective.value())}


# ---------------------------------------------------------------- 参考实现二：PuLP CFLP
def reference_cflp(D: np.ndarray, h: np.ndarray, cd: list[dict],
                   peak_factor: float = 1.6) -> dict:
    """单节点 CFLP：容量约束按**峰值日**校核，运输成本按日均。

    两阶段：先按真实场地面积约束求解；若不可行（所需档位的面积超过所有候选场地），
    放宽面积约束重解，并标注 `relaxation`，提示"需扩建场地或拆分节点"。
    """
    import pulp

    tiers = list(TIER_CATALOG)
    J, K, I = len(cd), len(tiers), len(h)
    h_pk = h * peak_factor                    # 峰值日件量，用于容量约束
    rent = [float(c.get("rent_month", 15000.0)) for c in cd]
    area = [float(c.get("area_sqm", 120.0)) for c in cd]

    def build(enforce_area: bool):
        prob = pulp.LpProblem("cflp_single_node", pulp.LpMinimize)
        y = {(j, k): pulp.LpVariable(f"y_{j}_{k}", cat="Binary")
             for j in range(J) for k in range(K)}
        x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary")
             for i in range(I) for j in range(J)}

        transport = pulp.lpSum(h[i] * D[i][j] / 1000.0 * UNIT_TRANSPORT_YUAN_PER_TKM * x[(i, j)]
                               for i in range(I) for j in range(J))
        fixed = pulp.lpSum(tiers[k]["cost_index"] * 1e4 * (1.0 + rent[j] / 50000.0) * y[(j, k)]
                           for j in range(J) for k in range(K))
        prob += fixed + transport

        prob += pulp.lpSum(y[(j, k)] for j in range(J) for k in range(K)) == 1
        for i in range(I):
            prob += pulp.lpSum(x[(i, j)] for j in range(J)) == 1
        for j in range(J):
            for i in range(I):
                prob += x[(i, j)] <= pulp.lpSum(y[(j, k)] for k in range(K))
            prob += (pulp.lpSum(h_pk[i] * x[(i, j)] for i in range(I))
                     <= pulp.lpSum(tiers[k]["daily_cap"] * y[(j, k)] for k in range(K)))
        if enforce_area:
            for j in range(J):
                for k in range(K):
                    if area[j] < tiers[k]["area_sqm"]:
                        prob += y[(j, k)] == 0
        prob.solve(cbc_solver())
        return prob, y

    prob, y = build(True)
    status = pulp.LpStatus[prob.status]
    relaxation = None
    if status != "Optimal":
        prob, y = build(False)
        status = pulp.LpStatus[prob.status]
        relaxation = "area_relaxed" if status == "Optimal" else None

    j_sel = k_sel = None
    if status == "Optimal":
        ys = [(j, k) for (j, k), v in y.items() if v.varValue and v.varValue > 0.5]
        if ys:
            j_sel, k_sel = ys[0]
    return {"impl": "PuLP CFLP（单节点，含容量与场地约束）",
            "status": status, "index": j_sel,
            "tier": tiers[k_sel]["id"] if k_sel is not None else None,
            "relaxation": relaxation,
            "note": ("所需档位面积超过所有候选场地，需扩建或拆分节点"
                     if relaxation else ""),
            "objective": float(pulp.value(prob.objective)) if status == "Optimal" else None}


# ---------------------------------------------------------------- 自研：扰动 → ε-最优集合
def builtin_perturbation(D: np.ndarray, h: np.ndarray, cxy: np.ndarray,
                         n: int = 300, seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    qn = rng.lognormal(0.0, 0.10, size=(n, len(h)))
    dn = rng.lognormal(0.0, 0.05, size=(n,) + D.shape)
    hits = np.zeros(len(cxy))
    for t in range(n):
        cost = (qn[t] * h)[:, None] * (D * dn[t])
        hits[int(np.argmin(cost.sum(axis=0)))] += 1
    return {"freq": hits / hits.sum(), "draws": n}


def _kernel_width(cxy: np.ndarray) -> float:
    if len(cxy) < 2:
        return 140.0
    d = np.linalg.norm(cxy[:, None, :] - cxy[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return max(90.0, float(np.median(d.min(axis=1))))


def run(payload: dict, grid: Grid, engine: str = "auto") -> dict:
    _, cxy, h, D, used = _matrices(payload, grid)
    cd = payload["candidates"]

    bt = builtin_perturbation(D, h, cxy)
    freq = bt["freq"]

    ref_pm = ref_cflp = None
    errors = []
    try:
        ref_pm = reference_pmedian(D, h)
    except Exception as exc:                     # pragma: no cover
        errors.append(f"spopt: {exc}")
    try:
        ref_cflp = reference_cflp(D, h, cd, peak_factor=float(get(payload, "peak_factor")))
    except Exception as exc:                     # pragma: no cover
        errors.append(f"pulp: {exc}")

    ref_ok = bool(ref_pm and ref_pm.get("index") is not None)

    if ref_pm and ref_pm.get("index") is not None:
        idx = ref_pm["index"]
    elif ref_cflp and ref_cflp.get("index") is not None:
        idx = ref_cflp["index"]
    else:
        idx = int(np.argmax(freq))

    same = None
    if ref_pm and ref_cflp and ref_pm.get("index") is not None and ref_cflp.get("index") is not None:
        same = bool(ref_pm["index"] == ref_cflp["index"])

    top3 = [int(j) for j in np.argsort(-freq)[:3]]
    # 参考实现没跑起来时不能"声称一致"，只能标为未校验
    consistent = bool(idx in top3) if ref_ok else None

    w = _kernel_width(cxy)
    mu = np.zeros_like(grid.X)
    for j in np.nonzero(freq > 0)[0]:
        mu += freq[j] * np.exp(-0.5 * (((grid.X - cxy[j, 0]) / w) ** 2
                                       + ((grid.Y - cxy[j, 1]) / w) ** 2))

    sensitivity = {
        "pmedian_index": (ref_pm or {}).get("index"),
        "cflp_index": (ref_cflp or {}).get("index"),
        "pmedian_id": (cd[ref_pm["index"]]["id"]
                       if ref_pm and ref_pm.get("index") is not None else None),
        "cflp_id": (cd[ref_cflp["index"]]["id"]
                    if ref_cflp and ref_cflp.get("index") is not None else None),
        "same": same,
        "rent_shift": bool(same is False),
        "reading": ("租金不改变最优位置" if same is not False else
                    "加入租金/固定成本后最优位置移动：该片区租金梯度足以改变选址，"
                    "建议作为敏感度情景单列"),
    }

    back_agreement = {"consistent": consistent, "sensitivity": sensitivity,
                      "reference_ran": ref_ok}
    if errors:
        back_agreement["error"] = " | ".join(errors)

    return {
        "key": "M2_discrete",
        "mu": normalise(mu),
        "center": (float(cxy[idx, 0]), float(cxy[idx, 1])),
        "sigma": (w, w),
        "detail": {"selected_candidate": cd[idx]["id"],
                   "tier_from_cflp": (ref_cflp or {}).get("tier"),
                   "frequency_top": {cd[j]["id"]: round(float(freq[j]), 3)
                                     for j in np.argsort(-freq)[:5] if freq[j] > 0},
                   "dist_source": "matrix" if used else "euclid×绕行",
                   "kernel_width_m": round(w, 1),
                   "rent_shift": sensitivity["rent_shift"]},
        "backends": {
            "builtin": {"impl": "扰动 300 次 argmin（ε-最优集合）",
                        "perturb_top3": top3, "consistent": consistent},
            "reference": {"pmedian": ref_pm, "cflp": ref_cflp},
            "agreement": back_agreement,
        },
    }


SPEC = MethodSpec(
    key="M2_discrete", name="离散候选（p-中值 + 容量约束 CFLP）", family="离散成本型",
    needs=("demand_points", "candidates"), prior=0.34,
    assumption="只能建在候选点；成本按路网阻抗；容量与场地受限",
    ref_impl="spopt.locate.PMedian + PuLP/CBC (CFLP)",
    run=run,
)
