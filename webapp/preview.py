"""开发用：把两页渲染成可离线打开的静态快照（CSS 内联，无需起服务）。

    python -m webapp.preview

产物在 webapp/preview/ 下：index.html（填表页）、result.html（结果页）。
用于在不起服务的情况下快速看一眼样式，或做截图。
"""
from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient

from webapp.app import app
from webapp.sample import SAMPLE

HERE = Path(__file__).resolve().parent
OUT = HERE / "preview"
STATIC = HERE / "static"


def series_sample(days: int = 400) -> str:
    """给快照用的流水样例：含年增长、周内波动与一次大促。"""
    import random
    from datetime import date, timedelta

    rng = random.Random(3)
    wd = [1.15, 1.10, 1.05, 1.00, 1.05, 0.85, 0.80]
    promo = {date(2025, 11, 11)}
    nodes = [("望京西园三区", 286.0), ("花家地北里", 214.0), ("南湖东园", 338.0)]
    start = date(2025, 1, 1)
    lines = []
    for i in range(days):
        d = start + timedelta(days=i)
        for name, base in nodes:
            v = base * (1.09 ** (i / 365.25)) * wd[d.weekday()]
            if d in promo:
                v *= 2.6
            v *= rng.lognormvariate(0, 0.05)
            lines.append(f"{name}, {d.isoformat()}, {v:.1f}")
    return "\n".join(lines)


def _inline(html: str) -> str:
    css = (STATIC / "pico.min.css").read_text(encoding="utf-8")
    app_css = (STATIC / "app.css").read_text(encoding="utf-8")
    html = re.sub(r'<link[^>]*pico\.min\.css[^>]*>', f"<style>{css}</style>", html)
    html = re.sub(r'<link[^>]*app\.css[^>]*>', f"<style>{app_css}</style>", html)
    return html


DIRECTIONS = ("wayfinding",)


def build() -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    out = []
    with TestClient(app) as c:
        data = dict(SAMPLE)
        data["series_points"] = series_sample()
        data["series_horizon"] = "1"
        data["peak_factor"] = ""          # 让快照走"由流水推演峰值"的分支
        for d in DIRECTIONS:
            idx = c.get(f"/?sample=1&mode=text&dir={d}&clean=1").text
            data_d = dict(data)
            data_d["dir"] = d
            data_d["clean"] = "1"
            res = c.post("/result", data=data_d).text
            for name, html in ((f"index-{d}.html", idx), (f"result-{d}.html", res)):
                p = OUT / name
                p.write_text(_inline(html), encoding="utf-8")
                out.append(p)
        # 保留默认文件名，方便直接打开
        for src, dst in (("index-drawing.html", "index.html"),
                         ("result-drawing.html", "result.html")):
            p = OUT / dst
            p.write_text((OUT / src).read_text(encoding="utf-8"), encoding="utf-8")
            out.append(p)
    return out


if __name__ == "__main__":                      # pragma: no cover
    for p in build():
        print(f"已生成 {p}（{p.stat().st_size / 1024:.0f} KB）")
