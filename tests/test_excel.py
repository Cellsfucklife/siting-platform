"""Excel / CSV 上传通道的测试。

覆盖：表头自动识别、列名别名重排、无表头、多工作表按名字挑、
文件优先于粘贴文本、坏文件类型、模板下载。
"""
from __future__ import annotations

import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from webapp import tables
from webapp.app import app
from webapp.forms import build_payload


def xlsx(rows: list[list], sheet: str = "Sheet1") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        pd.DataFrame(rows).to_excel(w, sheet_name=sheet, index=False, header=False)
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


DEMAND_HDR = [["小区名", "纬度", "经度", "日件量"],
              ["望京西园三区", 39.98862, 116.47139, 286],
              ["花家地北里", 39.98604, 116.47612, 214],
              ["南湖东园", 39.99127, 116.46807, 338]]


# ---------------------------------------------------------------- 读表层
def test_reads_xlsx_with_header():
    rows = tables.read_rows(xlsx(DEMAND_HDR), "a.xlsx")
    assert len(rows) == 4
    assert rows[1][0] == "望京西园三区"


def test_header_is_dropped_and_columns_reordered():
    # 表头顺序故意打乱，并用英文别名
    rows = [["qty_daily", "lng", "lat", "name"],
            [286, 116.47139, 39.98862, "望京西园三区"]]
    table = tables.to_table(xlsx(rows), "a.xlsx", "demand")
    p, e = __import__("webapp.forms", fromlist=["x"]).parse_demand_rows(table, "geo")
    assert not e
    d = p["demand_points"][0] if isinstance(p, dict) else p[0]
    assert d["id"] == "望京西园三区"
    assert abs(d["lat"] - 39.98862) < 1e-6
    assert d["qty_daily"] == 286


def test_no_header_still_works():
    table = tables.to_table(xlsx([["A", 39.98, 116.47, 200]]), "a.xlsx", "demand")
    assert table[0][0] == "A"


def test_multiple_sheets_picked_by_name():
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        pd.DataFrame([["说明"], ["随便写点"]], columns=None).to_excel(
            w, sheet_name="说明", index=False, header=False)
        pd.DataFrame(DEMAND_HDR).to_excel(w, sheet_name="②小区", index=False, header=False)
    rows = tables.read_rows(buf.getvalue(), "book.xlsx", hint="小区")
    assert rows[1][0] == "望京西园三区"


def test_csv_upload():
    data = ("小区名,纬度,经度,日件量\nA,39.98,116.47,200\n"
            "B,39.99,116.48,150\nC,39.97,116.46,120\n").encode("utf-8")
    payload, err, _ = build_payload({}, {"demand": {"filename": "a.csv", "data": data}})
    assert not err
    assert len(payload["demand_points"]) == 3


def test_bad_extension_is_explained():
    payload, err, _ = build_payload({}, {"demand": {"filename": "a.txt", "data": b"x"}})
    assert err and "不支持的文件类型" in err[0]


def test_empty_workbook_is_explained():
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        pd.DataFrame([[None, None]]).to_excel(w, sheet_name="S", index=False, header=False)
    payload, err, _ = build_payload({}, {"demand": {"filename": "e.xlsx",
                                                    "data": buf.getvalue()}})
    assert err and ("空的" in err[0] or "没有读到数据行" in err[0])


def test_file_wins_over_pasted_text():
    data = xlsx(DEMAND_HDR)
    payload, err, warn = build_payload(
        {"demand_points": "粘贴的A, 39.9, 116.4, 100"},
        {"demand": {"filename": "a.xlsx", "data": data}})
    assert not err
    assert len(payload["demand_points"]) == 3      # 用的是文件里的 3 行
    assert any("以文件为准" in w or "忽略了同时粘贴的文本" in w for w in warn)


def test_too_few_points_explained():
    rows = [["A", 39.98, 116.47, 200], ["B", 39.99, 116.48, 150]]
    _, err, _ = build_payload({}, {"demand": {"filename": "a.xlsx", "data": xlsx(rows)}})
    assert err and "至少要 3 个小区" in err[0]


# ---------------------------------------------------------------- 页面
def test_result_page_accepts_excel_upload(client):
    r = client.post("/result", data={"area_name": "Excel 片区"},
                    files={"file_demand": ("d.xlsx", xlsx(DEMAND_HDR),
                                           tables.XLSX_MIME if hasattr(tables, "XLSX_MIME")
                                           else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
    assert r.status_code == 200
    assert "模糊选址中心" in r.text and "Excel 片区" in r.text


def test_full_workbook_upload_three_sheets(client):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        pd.DataFrame(DEMAND_HDR[1:], columns=DEMAND_HDR[0]).to_excel(
            w, sheet_name="①小区", index=False)
        pd.DataFrame([["望京西园底商", 39.98791, 116.47358, 85, 26000],
                      ["花家地社区用房", 39.99033, 116.47724, 320, 18500]],
                     columns=["场地名", "纬度", "经度", "面积m²", "月租金"]).to_excel(
            w, sheet_name="②场地", index=False)
        mat = [["望京西园三区", "望京西园底商", 4.6],
               ["望京西园三区", "花家地社区用房", 3.1],
               ["花家地北里", "望京西园底商", 5.9],
               ["花家地北里", "花家地社区用房", 3.8],
               ["南湖东园", "望京西园底商", 4.1],
               ["南湖东园", "花家地社区用房", 5.3]]
        pd.DataFrame(mat, columns=["小区名", "场地名", "分钟"]).to_excel(
            w, sheet_name="③距离", index=False)
    blob = buf.getvalue()
    r = client.post("/result", data={"area_name": "整本上传"},
                    files={"file_demand": ("book.xlsx", blob),
                           "file_candidates": ("book.xlsx", blob),
                           "file_matrix": ("book.xlsx", blob)})
    assert r.status_code == 200
    assert "容量配置" in r.text
    # 距离表齐了 → 不应出现“退回直线估算”
    assert "不完整" not in r.text


def test_template_download(client):
    r = client.get("/template.xlsx")
    assert r.status_code == 200
    assert "spreadsheetml" in r.headers["content-type"]
    assert r.content[:2] == b"PK" and len(r.content) > 2000
    book = pd.read_excel(io.BytesIO(r.content), sheet_name=None)
    assert any("小区" in s for s in book)
    assert any("说明" in s for s in book)
    # 模板必须覆盖平台支持的第四张表（流水）
    assert any("流水" in s for s in book), list(book)


# ---------------------------------------------------------------- 输入方式
def test_mode_excel_without_file_is_explained():
    payload, err, _ = build_payload({"input_mode": "excel"})
    assert err and "还没有选择文件" in err[0]


def test_mode_excel_ignores_pasted_text():
    payload, err, warn = build_payload(
        {"input_mode": "excel", "demand_points": "粘贴的, 39.9, 116.4, 100"},
        {"demand": {"filename": "d.xlsx", "data": xlsx(DEMAND_HDR)}})
    assert not err
    assert len(payload["demand_points"]) == 3
    assert any("忽略粘贴的文本" in w for w in warn)


def test_mode_text_ignores_uploaded_file():
    payload, err, warn = build_payload(
        {"input_mode": "text",
         "demand_points": "A, 39.98, 116.47, 200\nB, 39.99, 116.48, 150\n"
                          "C, 39.97, 116.46, 120"},
        {"demand": {"filename": "d.xlsx", "data": xlsx(DEMAND_HDR)}})
    assert not err
    assert len(payload["demand_points"]) == 3
    assert payload["demand_points"][0]["id"] == "A"      # 用的是文本
    assert any("忽略上传" in w for w in warn)


def test_index_has_mode_switcher(client):
    html = client.get("/").text
    assert 'name="input_mode"' in html and 'value="excel"' in html
    assert "mode-text" in html                          # 默认粘贴文本
    assert 'enctype="multipart/form-data"' in html


def test_result_keeps_chosen_mode(client):
    r = client.post("/result", data={"input_mode": "excel", "area_name": "x"},
                    files={"file_demand": ("d.xlsx", xlsx(DEMAND_HDR))})
    assert r.status_code == 200


def test_error_page_keeps_excel_mode(client):
    r = client.post("/result", data={"input_mode": "excel"})
    assert r.status_code == 400
    assert "mode-file" in r.text                        # 回到页面仍是上传方式
    assert "还没有选择文件" in r.text


def test_api_solve_accepts_excel(client):
    r = client.post("/api/solve", files={"file_demand": ("d.xlsx", xlsx(DEMAND_HDR))})
    body = r.json()
    assert r.status_code == 200 and body["ok"]
    assert body["capacity"]["tier"]
