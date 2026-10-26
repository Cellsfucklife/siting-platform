"""一页填表 → 一页出结果。平台的最小 Web 外壳。

    python -m webapp.app          # http://127.0.0.1:8000
    run_web.bat                   # 双击同一个

路由：
    GET  /                填表页
    GET  /template.xlsx   下载 Excel 填写模板
    POST /result          计算并跳转到结果页
    GET  /health          健康检查
    POST /api/solve       纯 JSON 接口（脚本用，同样支持 Excel 上传）
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import Request
from fastapi.responses import (HTMLResponse, JSONResponse, RedirectResponse,
                               Response)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from siting.solver import solve

from . import svgmap, tables
from .forms import build_payload
from .sample import SAMPLE

try:
    from fastapi import FastAPI
except Exception:                                        # pragma: no cover
    raise

HERE = Path(__file__).resolve().parent
app = FastAPI(title="配送节点选址与容量测算", version="1.1.0")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=str(HERE / "templates"))

XLSX_MIME = ("application/vnd.openxmlformats-officedocument"
             ".spreadsheetml.sheet")

BLANK = {"area_name": "", "frame": "geo", "input_mode": "text",
         "demand_points": "", "candidates": "",
         "dist_matrix": "", "series_points": "", "series_freq": "auto",
         "series_horizon": "1",
         "proxy_daily_demand": "", "peak_factor": "", "cv_margin": "",
         "service_radius_min": "", "self_pick_ratio": "", "service_time_s": "",
         "queue_wait_target_s": "", "vehicle_daily_capacity": "", "access_speed_kmh": ""}

# 表单字段名 → 内部分类
UPLOAD_FIELDS = {"file_demand": "demand", "file_candidates": "candidates",
                 "file_matrix": "matrix", "file_series": "series"}


async def _collect(request: Request) -> tuple[dict, dict, dict]:
    """读出表单文本、以及三个可选的文件上传。"""
    raw = await request.form()
    form = {k: str(v) for k, v in raw.items()
            if k not in UPLOAD_FIELDS and not hasattr(v, "filename")}
    uploads: dict = {}
    for field, key in UPLOAD_FIELDS.items():
        up = raw.get(field)
        if up is None or not getattr(up, "filename", ""):
            continue
        data = await up.read()
        if data:
            uploads[key] = {"filename": up.filename, "data": data}
    return form, uploads, dict(raw)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, sample: int = 0, mode: str = "text",
                dir: str = "wayfinding", clean: int = 0):
    form = dict(BLANK)
    form["input_mode"] = "excel" if mode == "excel" else "text"
    if sample:
        form.update(SAMPLE)
        form["input_mode"] = "excel" if mode == "excel" else "text"
    return templates.TemplateResponse(
        request, "index.html",
        {"form": form, "errors": [], "warnings": [], "has_sample": bool(sample),
         "dir": dir, "clean": bool(clean)})


@app.get("/template.xlsx")
async def template():
    return Response(
        tables.build_template(), media_type=XLSX_MIME,
        headers={"Content-Disposition":
                 'attachment; filename="siting_input_template.xlsx"'})


@app.post("/result", response_class=HTMLResponse)
async def result(request: Request):
    form, uploads, _ = await _collect(request)
    payload, errors, warnings = build_payload(form, uploads)
    dir_name = str(form.get("dir") or "wayfinding")

    def back(msg_errors, status=400):
        merged = dict(BLANK)
        merged.update(form)
        return templates.TemplateResponse(
            request, "index.html",
            {"form": merged, "errors": msg_errors, "warnings": warnings,
             "has_sample": False, "dir": dir_name, "clean": False}, status_code=status)

    if errors:
        return back(errors)

    res = solve(payload)
    if not res.get("ok"):
        return back(res.get("errors") or ["数据不足，无法求解"])

    return templates.TemplateResponse(request, "result.html", {
        "form": form,
        "payload": payload,
        "r": res,
        "series": payload.get("_series"),
        "dir": dir_name,
        "clean": bool(str(form.get("clean") or 0)),
        "map_svg": svgmap.render(payload, res),
        "warnings": warnings + (res.get("warnings") or []),
        "raw_json": json.dumps(res, ensure_ascii=False, indent=2),
    })


@app.get("/result")
async def result_get():
    return RedirectResponse("/", status_code=303)


@app.get("/health")
async def health():
    return {"ok": True, "service": "siting-webapp", "version": app.version}


@app.post("/api/solve")
async def api_solve(request: Request):
    form, uploads, _ = await _collect(request)
    payload, errors, warnings = build_payload(form, uploads)
    if errors:
        return JSONResponse({"ok": False, "errors": errors, "warnings": warnings},
                            status_code=400)
    return JSONResponse(solve(payload))


if __name__ == "__main__":                               # pragma: no cover
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
