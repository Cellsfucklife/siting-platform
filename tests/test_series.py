"""推演层测试：日期解析、聚合、拟合还原真值、与 Web 的接入。"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient

from siting.series import aggregate, derive, parse_period
from webapp.app import app
from webapp.forms import build_payload

TRUE_WD = [1.15, 1.10, 1.05, 1.00, 1.05, 0.85, 0.80]
DEMAND3 = ("望京西园三区, 39.98862, 116.47139, 286\n"
           "花家地北里, 39.98604, 116.47612, 214\n"
           "南湖东园, 39.99127, 116.46807, 338")
PROMO = {date(2025, 6, 18), date(2025, 11, 11)}


def synth(days: int = 365, growth: float = 0.10, base: float = 280.0,
          noise: float = 0.05, seed: int = 7) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    start = date(2025, 1, 1)
    for i in range(days):
        d = start + timedelta(days=i)
        v = base * (1 + growth) ** (i / 365.25) * TRUE_WD[d.weekday()]
        if d in PROMO:
            v *= 2.6
        out.append({"id": "望京西园三区", "date": d.isoformat(),
                    "qty": round(v * float(rng.lognormal(0, noise)), 1)})
    return out


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------- 解析
@pytest.mark.parametrize("text,kind", [
    ("2025-01-03", "day"), ("2025/1/3", "day"), ("2025年1月3日", "day"),
    ("20250103", "day"), ("2025-01", "month"), ("2025年1月", "month"),
    ("2025", "year"),
])
def test_parse_period(text, kind):
    got = parse_period(text)
    assert got and got[1] == kind


def test_parse_period_rejects_garbage():
    assert parse_period("上周三") is None
    assert parse_period("") is None


# ---------------------------------------------------------------- 聚合
def test_aggregate_by_week_and_month():
    recs = synth(60)
    for freq, expect_max in (("day", 60), ("week", 10), ("month", 3)):
        agg = aggregate(recs, freq)
        assert agg["ok"] and agg["freq"] == freq
        assert len(agg["total"]["values"]) <= expect_max


def test_auto_detects_monthly_input():
    recs = [{"id": "A", "date": f"2025-{m:02d}", "qty": 100 + m} for m in range(1, 13)]
    agg = aggregate(recs, "auto")
    assert agg["freq"] == "month" and len(agg["total"]["values"]) == 12


# ---------------------------------------------------------------- 拟合
def test_recovers_growth_and_weekday_factors():
    d = derive(synth(365), "day", horizon_years=1.0)
    assert d["ok"]
    assert abs(d["growth_per_year"] - 0.10) < 0.04, d["growth_per_year"]
    wd = d["weekday_factor"]
    for got, want in zip(wd, TRUE_WD):
        assert abs(got - want) < 0.06, (wd, TRUE_WD)
    assert d["r2"] > 0.75
    assert d["n_promo"] >= 1                  # 大促日被识别出来
    assert d["peak_factor"] > 2.0             # 峰值含大促


def test_projection_moves_quantity_forward():
    d = derive(synth(365), "day", horizon_years=2.0)
    node = next(iter(d["nodes"].values()))
    assert node["qty_projected"] > node["qty_observed"] > 0


def test_short_series_is_flagged_not_crashing():
    d = derive(synth(20), "day")
    assert d["ok"] and "不足" in d["note"]


# ---------------------------------------------------------------- 接入
def test_payload_uses_derived_quantities():
    lines = "\n".join(f"{r['id']}, {r['date']}, {r['qty']}" for r in synth(365))
    form = {"demand_points": DEMAND3,
            "series_points": lines, "series_horizon": "1"}
    payload, err, warn = build_payload(form)
    assert not err, err
    p = payload["demand_points"][0]
    assert p["qty_daily"] != 286                      # 已被推演值替换
    assert "qty_observed" in p
    assert payload["_series"]["ok"]
    assert any("推演" in w for w in warn)
    assert payload["peak_factor"] > 1.6               # 含大促，取观测值


def test_no_promo_window_keeps_default_peak():
    """窗口内没有大促日时观测峰值偏低，应保留默认 1.6 并提示。"""
    lines = "\n".join(f"{r['id']}, {r['date']}, {r['qty']}" for r in synth(120))
    payload, err, warn = build_payload({"demand_points": DEMAND3, "series_points": lines})
    assert not err
    assert payload["peak_factor"] == 1.6
    assert any("低于默认的 1.6" in w for w in warn)


def test_user_peak_factor_wins():
    lines = "\n".join(f"{r['id']}, {r['date']}, {r['qty']}" for r in synth(120))
    form = {"demand_points": DEMAND3,
            "series_points": lines, "peak_factor": "2.0"}
    payload, err, _ = build_payload(form)
    assert not err
    assert payload["peak_factor"] == 2.0
    assert payload["_series"]["peak_factor_used"] == 2.0


def test_bad_date_is_explained():
    form = {"demand_points": "A, 39.98, 116.47, 200\nB, 39.99, 116.48, 150\n"
                             "C, 39.97, 116.46, 120",
            "series_points": "A, 上周三, 200"}
    _, err, _ = build_payload(form)
    assert err and "认不出" in err[0]


def test_result_page_shows_derivation(client):
    lines = "\n".join(f"{r['id']}, {r['date']}, {r['qty']}" for r in synth(200))
    r = client.post("/result", data={
        "area_name": "推演测试",
        "demand_points": DEMAND3,
        "series_points": lines,
        "series_horizon": "1",
    })
    assert r.status_code == 200
    assert "数据推演" in r.text and "年增长率" in r.text and "周内系数" in r.text
    assert "week-bars" in r.text                     # 周内系数柱状图


def test_volatility_flows_into_capacity():
    """推演出的波动率要变成容量缓冲，而不是只显示。"""
    lines = "\n".join(f"{r['id']}, {r['date']}, {r['qty']}" for r in synth(365, noise=0.02))
    noisy = "\n".join(f"{r['id']}, {r['date']}, {r['qty']}" for r in synth(365, noise=0.12))
    calm_p, _, _ = build_payload({"demand_points": DEMAND3, "series_points": lines})
    wild_p, _, warn = build_payload({"demand_points": DEMAND3, "series_points": noisy})
    from siting.capacity import evaluate
    from siting.geo import Grid

    def cap(p):
        g = Grid(p)
        return evaluate(p, g, (0.0, 0.0), 2000.0)

    assert calm_p["volatility_cv"] < wild_p["volatility_cv"]
    assert cap(wild_p)["safety_margin"] > cap(calm_p)["safety_margin"] > 1.0
    assert any("波动 CV" in w for w in warn)


def test_order_level_rows_are_summed_per_day():
    """原始订单级流水（同一小区同一天多行）应自动按日累加。"""
    recs = [{"id": "A", "date": "2025-01-01", "qty": 10},
            {"id": "A", "date": "2025-01-01", "qty": 15},
            {"id": "A", "date": "2025-01-02", "qty": 20}]
    agg = aggregate(recs, "day")
    vals = agg["series"]["A"]["values"]
    assert list(vals) == [25, 20]


def test_weekday_peak_raises_too_low_user_peak():
    """用户手填的峰值系数低于流水里的周内高点时，应被抬到周内高点。"""
    lines = "\n".join(f"{r['id']}, {r['date']}, {r['qty']}" for r in synth(365))
    payload, err, warn = build_payload({"demand_points": DEMAND3,
                                        "series_points": lines,
                                        "peak_factor": "0.9"})
    assert not err
    assert payload["peak_factor"] > 0.9
    assert any("低于流水" in w or "已按" in w for w in warn)
