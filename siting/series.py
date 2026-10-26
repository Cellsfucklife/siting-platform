"""原始流水推演：点集 → 线性方程 → 抽象指标。

输入是逐条流水（小区, 日期, 件量），可以是逐日、逐周、逐月或逐年。
输出是几组可以直接喂给选址与容量算法的抽象指标：

    水平项     该小区当前的日均件量
    趋势项     年增长率，用来外推到设计年
    周期项     周内系数（日粒度时才可识别）、月系数（跨度满一年才可识别）
    峰值系数   观测到的最大周期 / 均值，替代拍出来的 1.6
    波动项     残差变异系数，供档位安全边际与鲁棒性判断使用
    拟合质量   R²、样本量、时间跨度

模型是最小平方的线性方程：
    y_t = β0 + β1·t + Σ 周期哑变量 + ε
t 以「年」为单位，所以 β1 直接就是年增量（件/日/年）。
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

WEEKDAY_NAMES = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
FREQ_LABEL = {"day": "按日", "week": "按周", "month": "按月", "year": "按年"}

_DATE_PATTERNS = (
    (re.compile(r"^(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})日?$"), "day"),
    (re.compile(r"^(\d{4})(\d{2})(\d{2})$"), "day"),
    (re.compile(r"^(\d{4})[-/.年](\d{1,2})月?$"), "month"),
    (re.compile(r"^(\d{4})(\d{2})$"), "month"),
    (re.compile(r"^(\d{4})年?$"), "year"),
)


def parse_period(text: str) -> tuple[date, str] | None:
    """把各种写法的日期解析成 (起始日, 粒度)。认不出返回 None。"""
    s = str(text).strip()
    if not s:
        return None
    for pat, kind in _DATE_PATTERNS:
        m = pat.match(s)
        if not m:
            continue
        g = m.groups()
        try:
            if kind == "day":
                return date(int(g[0]), int(g[1]), int(g[2])), "day"
            if kind == "month":
                return date(int(g[0]), int(g[1]), 1), "month"
            return date(int(g[0]), 1, 1), "year"
        except ValueError:
            return None
    return None


def _bucket(d: date, freq: str) -> date:
    if freq == "day":
        return d
    if freq == "week":
        return d - timedelta(days=d.weekday())
    if freq == "month":
        return date(d.year, d.month, 1)
    return date(d.year, 1, 1)


def _to_years(d: date, origin: date) -> float:
    return (d - origin).days / 365.25


# ---------------------------------------------------------------- 聚合
def aggregate(records: list[dict], freq: str = "auto") -> dict:
    """records: [{id, date, qty}] → 每个 id 一条按周期聚合后的序列。

    freq 为 auto 时，按流水自身的日期粒度决定：有日粒度数据就按日聚合。
    """
    parsed = []
    granularities = set()
    for r in records:
        p = parse_period(r.get("date", ""))
        q = r.get("qty")
        if p is None or q is None:
            continue
        d, g = p
        granularities.add(g)
        parsed.append({"id": str(r.get("id") or "整体").strip(), "date": d, "qty": float(q)})

    if not parsed:
        return {"ok": False, "error": "流水里没有可解析的日期与件量"}

    if freq == "auto":
        freq = "day" if "day" in granularities else (
            "week" if "week" in granularities else
            ("month" if "month" in granularities else "year"))

    by_id: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    for r in parsed:
        by_id[r["id"]][_bucket(r["date"], freq)] += r["qty"]

    series = {}
    for pid, buckets in by_id.items():
        periods = sorted(buckets)
        series[pid] = {"periods": periods,
                       "values": np.array([buckets[p] for p in periods], float)}
    total = defaultdict(float)
    for buckets in by_id.values():
        for p, v in buckets.items():
            total[p] += v
    periods = sorted(total)
    return {"ok": True, "freq": freq, "series": series,
            "total": {"periods": periods,
                      "values": np.array([total[p] for p in periods], float)},
            "n_records": len(parsed),
            "granularities": sorted(granularities)}


# ---------------------------------------------------------------- 拟合
@dataclass
class Fit:
    level: float                      # 拟合到末期的日均水平（件/周期）
    level_per_day: float              # 折算成「每天」
    growth_per_year: float            # 年增量占水平的比例
    weekday_factor: list[float] = field(default_factory=list)   # 7 个，均值 1
    month_factor: list[float] = field(default_factory=list)     # 12 个，均值 1
    peak_factor: float = 1.0          # 最大周期 / 均值
    cv: float = 0.0                   # 残差变异系数
    r2: float = 0.0
    n_obs: int = 0
    span_days: int = 0
    freq: str = "day"
    n_promo: int = 0
    promo_multiplier: float = 1.0

    def to_dict(self) -> dict:
        return {
            "level": round(self.level, 1),
            "level_per_day": round(self.level_per_day, 1),
            "growth_per_year": round(self.growth_per_year, 4),
            "weekday_factor": [round(v, 3) for v in self.weekday_factor],
            "month_factor": [round(v, 3) for v in self.month_factor],
            "peak_factor": round(self.peak_factor, 3),
            "cv": round(self.cv, 3),
            "r2": round(self.r2, 3),
            "n_obs": self.n_obs,
            "span_days": self.span_days,
            "freq": self.freq,
            "n_promo": self.n_promo,
            "promo_multiplier": round(self.promo_multiplier, 3),
        }


def fit_series(periods: list[date], values: np.ndarray, freq: str) -> Fit:
    """对一条周期序列做最小二乘，得到水平、趋势与周期项。"""
    n = len(values)
    if n < 2:
        lvl = float(values[-1]) if n else 0.0
        return Fit(level=lvl, level_per_day=lvl, growth_per_year=0.0,
                   peak_factor=1.0, cv=0.0, r2=0.0, n_obs=n,
                   span_days=(periods[-1] - periods[0]).days if n else 0, freq=freq)

    origin = periods[0]
    t = np.array([_to_years(p, origin) for p in periods], float)
    cols = [np.ones(n), t]
    names = ["intercept", "trend"]

    # 周内系数：只有日粒度且跨度够 3 周才可识别
    if freq == "day" and (periods[-1] - periods[0]).days >= 20:
        wd = np.array([p.weekday() for p in periods])
        for k in range(1, 7):
            cols.append((wd == k).astype(float))
            names.append(f"wd{k}")
    # 月系数：只有跨度满两年才与趋势项可分辨（一年内趋势会被月度哑变量吸收光）
    if (periods[-1] - periods[0]).days >= 730:
        mo = np.array([p.month for p in periods])
        for k in range(2, 13):
            cols.append((mo == k).astype(float))
            names.append(f"m{k}")

    X = np.column_stack(cols)
    # 参数不能逼近样本数，否则趋势项不可识别；退化为「截距 + 趋势」
    if X.shape[1] >= n - 8:
        X = X[:, :2]
        names = names[:2]

    # 稳健拟合：大促日是少数极端点，会把趋势项拉偏，先按残差剔除再重解
    mask = np.ones(n, bool)
    coef = np.zeros(X.shape[1])
    for _ in range(4):
        coef, *_ = np.linalg.lstsq(X[mask], values[mask], rcond=None)
        resid_all = values - X @ coef
        sigma = float(np.std(resid_all[mask]))
        keep = np.abs(resid_all) <= 2.5 * max(sigma, 1e-9)
        if keep.sum() < max(10, int(0.6 * n)) or keep.sum() == mask.sum():
            break
        mask = keep

    pred = X @ coef
    resid = values - pred
    ss_res = float((resid[mask] ** 2).sum())
    ss_tot = float(((values[mask] - values[mask].mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-9 else 1.0

    # 峰值系数取「实际值 / 当时的趋势值」，避免把多年增长算成峰值
    peak = float(np.clip((values / np.maximum(pred, 1e-9)).max(), 1.0, 5.0))

    level = float(X[-1] @ coef)             # 末期拟合值
    mean_level = float(pred.mean())
    growth = float(coef[1]) / max(mean_level, 1e-9) if len(coef) > 1 else 0.0

    # 周期因子：相对截距的比值，再归一化到均值 1
    wd_factor = np.ones(7)
    mo_factor = np.ones(12)
    idx = {nm: i for i, nm in enumerate(names)}
    base = float(coef[0]) if abs(float(coef[0])) > 1e-9 else float(values.mean())
    if "wd1" in idx:
        for k in range(1, 7):
            wd_factor[k] = (base + coef[idx[f"wd{k}"]]) / base
        wd_factor = wd_factor / max(wd_factor.mean(), 1e-9)
    if "m2" in idx:
        for k in range(2, 13):
            mo_factor[k - 1] = (base + coef[idx[f"m{k}"]]) / base
        mo_factor = mo_factor / max(mo_factor.mean(), 1e-9)

    # 大促日单独识别（被稳健拟合剔除、且明显高于趋势的点），波动只统计常规日
    ratio = values / np.maximum(pred, 1e-9)
    spike = (~mask) & (ratio > 1.5)
    n_promo = int(spike.sum())
    cv = float(np.std(resid[mask]) / max(values[mask].mean(), 1e-9))
    promo_mult = float(ratio[spike].mean()) if n_promo else 1.0
    per_day = {"day": 1.0, "week": 7.0, "month": 30.44, "year": 365.25}[freq]

    return Fit(level=level, level_per_day=level / per_day,
               growth_per_year=growth,
               weekday_factor=[float(v) for v in wd_factor],
               month_factor=[float(v) for v in mo_factor],
               peak_factor=peak, cv=cv, r2=float(r2), n_obs=n,
               span_days=(periods[-1] - periods[0]).days, freq=freq,
               n_promo=n_promo, promo_multiplier=promo_mult)


# ---------------------------------------------------------------- 推演
def derive(records: list[dict], freq: str = "auto", horizon_years: float = 1.0) -> dict:
    """整条推演链：聚合 → 拟合 → 外推到设计年 → 抽象指标。"""
    agg = aggregate(records, freq)
    if not agg["ok"]:
        return agg

    f = agg["freq"]
    per_year = {"day": 365.25, "week": 52.18, "month": 12.0, "year": 1.0}[f]

    fits = {}
    for pid, s in agg["series"].items():
        fit = fit_series(s["periods"], s["values"], f)
        # 设计年日均取该年年中的拟合值：level·(1+g)^(h−0.5)
        proj = fit.level * (1 + fit.growth_per_year) ** (horizon_years - 0.5)
        fits[pid] = {"fit": fit.to_dict(),
                     "qty_projected": round(proj / {"day": 1, "week": 7,
                                                    "month": 30.44, "year": 365.25}[f], 1),
                     "qty_observed": round(float(s["values"].mean()) /
                                           {"day": 1, "week": 7, "month": 30.44,
                                            "year": 365.25}[f], 1)}

    # 整体序列给出全局峰值系数与增长率
    tot = agg["total"]
    gfit = fit_series(tot["periods"], tot["values"], f)
    r2_mean = float(np.mean([v["fit"]["r2"] for v in fits.values()])) if fits else 0.0
    cv_mean = float(np.mean([v["fit"]["cv"] for v in fits.values()])) if fits else 0.0
    growth_mean = float(np.mean([v["fit"]["growth_per_year"] for v in fits.values()])) if fits else 0.0

    return {
        "ok": True,
        "freq": f,
        "freq_label": FREQ_LABEL[f],
        "n_records": agg["n_records"],
        "span_days": gfit.span_days,
        "n_periods": gfit.n_obs,
        "growth_per_year": round(growth_mean, 4),
        "peak_factor": round(gfit.peak_factor, 3),
        "cv": round(cv_mean, 3),
        "r2": round(r2_mean, 3),
        "weekday_factor": gfit.weekday_factor,
        "month_factor": gfit.month_factor,
        "weekday_names": WEEKDAY_NAMES,
        "n_promo": gfit.n_promo,
        "promo_multiplier": round(gfit.promo_multiplier, 3),
        "horizon_years": horizon_years,
        "nodes": fits,
        "note": _note(f, gfit),
    }


def _note(freq: str, fit: Fit) -> str:
    bits = []
    if freq != "day":
        bits.append("输入不是逐日粒度，" + FREQ_LABEL[freq] + "拟合只能给趋势与总体峰值")
    if fit.span_days < 60:
        bits.append("跨度不足 60 天，趋势项不稳")
    if fit.n_obs < 8:
        bits.append("样本少于 8 个周期，仅作参考")
    if not bits:
        bits.append("按日拟合，趋势与周内系数均已识别")
    return "；".join(bits)
