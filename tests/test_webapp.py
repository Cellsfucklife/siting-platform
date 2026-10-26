"""Web 层测试：表单解析 + 两页渲染 + 端到端提交。

前端只有两页（填表 / 结果），所以这里断言的就是"这两页能出、能通"。
"""
from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from webapp.app import app
from webapp.forms import build_payload
from webapp.sample import SAMPLE


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------- 表单解析
def test_parse_demand_ok():
    p, e, _ = build_payload({"demand_points":
                             "A, 39.98, 116.47, 200\nB, 39.99, 116.48, 150\n"
                             "C, 39.97, 116.46, 120"})
    assert not e
    assert len(p["demand_points"]) == 3
    assert p["demand_points"][0]["qty_daily"] == 200


def test_parse_demand_errors_are_readable():
    _, e, _ = build_payload({"demand_points": "A, 39.98, 116.47\nB, x, y, 12"})
    assert len(e) == 2
    assert "只有 3 列" in e[0] and "日件量" in e[0]
    assert "非数字" in e[1]


def test_swapped_latlng_is_auto_corrected():
    p, e, w = build_payload({"demand_points":
                             "A, 116.47139, 39.98862, 286\n"
                             "B, 116.47612, 39.98604, 214\n"
                             "C, 116.46807, 39.99127, 338"})
    assert not e
    d = p["demand_points"][0]
    assert abs(d["lat"] - 39.98862) < 1e-6        # 已调换回来
    assert abs(d["lng"] - 116.47139) < 1e-6
    assert any("填反" in x for x in w)


def test_out_of_range_coords_explained():
    _, e, _ = build_payload({"demand_points":
                             "A, 320, 180, 286\nB, 39.99, 116.48, 150\nC, 39.97, 116.46, 120"})
    assert e and "超出经纬度范围" in e[0] and "相对米制" in e[0]


def test_too_few_points_explained():
    _, e, _ = build_payload({"demand_points": "A, 39.98, 116.47, 200\n"
                                              "B, 39.99, 116.48, 150"})
    assert e and "至少要 3 个小区" in e[0]


def test_comment_only_input_is_rejected_clearly():
    _, e, _ = build_payload({"demand_points": "# 只有注释\n\n"})
    assert len(e) == 1
    assert "没有解析出任何一行" in e[0]


def test_comment_only_input_is_rejected_clearly():
    _, e, _ = build_payload({"demand_points": "# 只有注释\n\n"})
    assert len(e) == 1
    assert "没有解析出任何一行" in e[0]


def test_parse_local_frame():
    p, e, _ = build_payload({"frame": "local",
                             "demand_points": "A, 100, 200, 150\nB, -100, 0, 150\n"
                                              "C, 0, -120, 120"})
    assert not e
    assert "_local_origin" in p
    assert abs(p["demand_points"][0]["lat"] - 39.9) < 0.01   # 伪原点附近


def test_parse_matrix_partial_falls_back():
    form = {"demand_points": "A, 39.98, 116.47, 100\nB, 39.99, 116.48, 100",
            "candidates": "C1, 39.985, 116.475, 120, 20000",
            "dist_matrix": "A, C1, 5.0"}
    p, e, w = build_payload(form)
    assert not e
    assert "dist_matrix" not in p            # 不完整 → 退回估算
    assert any("不完整" in x for x in w)


def test_requires_at_least_one_table():
    _, e, _ = build_payload({"demand_points": "", "candidates": ""})
    assert e and "至少" in e[0]


# ---------------------------------------------------------------- 页面
def test_index_has_all_ports_and_labels(client):
    r = client.get("/")
    assert r.status_code == 200
    html = r.text
    for name in ("demand_points", "candidates", "dist_matrix", "series_points",
                 "peak_factor", "service_radius_min", "self_pick_ratio"):
        assert f'name="{name}"' in html
    assert "必填" in html and "可留空" in html
    # 可选数据都在抽屉里
    for drawer in ("d-candidates", "d-matrix", "d-series", "d-params"):
        assert f'id="{drawer}"' in html


def test_sample_prefills(client):
    html = client.get("/?sample=1").text
    assert "望京西园三区" in html and "望京西园底商" in html


def test_result_page_renders_center_and_capacity(client):
    r = client.post("/result", data=SAMPLE)
    assert r.status_code == 200
    html = r.text
    assert "模糊选址中心" in html
    assert "容量配置" in html
    assert "<svg" in html and "模糊区" in html
    assert re.search(r"station-|locker-", html)


def test_result_page_rejects_empty(client):
    r = client.post("/result", data={"area_name": "空"})
    assert r.status_code == 400
    assert "开始测算" in r.text           # 回到表单页且保留输入
    assert "至少要填一个" in r.text


def test_health_and_json_api(client):
    assert client.get("/health").json()["ok"] is True
    r = client.post("/api/solve", data=SAMPLE)
    body = r.json()
    assert r.status_code == 200
    assert body["ok"] and body["fuzzy_center"]["radius_90_m"] > 0
    assert body["capacity"]["tier"]


def test_get_result_redirects(client):
    r = client.get("/result", follow_redirects=False)
    assert r.status_code == 303
