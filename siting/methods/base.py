"""方法基类：统一结果结构 + 双引擎（builtin / reference）约定。

每个方法返回：
    {
      "key", "mu", "center", "sigma", "detail",
      "backends": {"builtin": {...}, "reference": {...}, "agreement": {...}}
    }
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class MethodSpec:
    key: str
    name: str
    family: str
    needs: tuple[str, ...]
    prior: float
    assumption: str
    ref_impl: str          # 参考实现来源（用于报告与论文引用）
    run: Callable = None   # type: ignore[assignment]

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.key}({self.family})"


def agreement_point(a: tuple[float, float], b: tuple[float, float], tol: float = 60.0) -> dict:
    d = float(np.hypot(a[0] - b[0], a[1] - b[1]))
    return {"distance_m": round(d, 1), "consistent": bool(d <= tol), "tolerance_m": tol}


def agreement_field(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        return {"corr": None, "consistent": None}
    af, bf = a.ravel(), b.ravel()
    corr = float(np.corrcoef(af, bf)[0, 1]) if af.std() > 0 and bf.std() > 0 else 1.0
    return {"corr": round(corr, 4), "consistent": bool(corr >= 0.98)}
