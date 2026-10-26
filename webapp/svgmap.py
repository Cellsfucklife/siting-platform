"""结果页示意图：手写内联 SVG，无第三方绘图库。

画：米制网格与刻度、同心半径环、68%/90% 模糊区、需求点（面积∝件量）、
候选场地（三角+短编号）、模糊中心与次峰、指北针、带刻度的比例尺。
"""
from __future__ import annotations

import html
import math

C_DEMAND = "#3b6ea5"
C_CAND = "#c9820c"
C_CENTER = "#c0392b"
C_ALT = "#8e44ad"
C_SELECTED = "#1f5fa8"
C_GRID = "#eef1f5"
C_INK = "#5b6b7f"


def _n(v, nd=0) -> str:
    try:
        return f"{v:,.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def render(payload: dict, result: dict, w: int = 640, h: int = 560) -> str:
    fc = result["fuzzy_center"]
    dp = payload.get("demand_points") or []
    cd = payload.get("candidates") or []
    allp = dp + cd
    if not allp:
        return ""

    lat0 = sum(p["lat"] for p in allp) / len(allp)
    lng0 = sum(p["lng"] for p in allp) / len(allp)
    clat = math.cos(math.radians(lat0))

    def xy(p):
        return ((p["lng"] - lng0) * 111320 * clat, (p["lat"] - lat0) * 110540)

    cx = (fc["lng"] - lng0) * 111320 * clat
    cy = (fc["lat"] - lat0) * 110540
    r68, r90 = fc["radius_68_m"], fc["radius_90_m"]

    xs = [xy(p)[0] for p in allp] + [cx - r90, cx + r90]
    ys = [xy(p)[1] for p in allp] + [cy - r90, cy + r90]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    spanx, spany = max(xmax - xmin, 1.0), max(ymax - ymin, 1.0)
    pad_l, pad_r, pad_t, pad_b = 34, 24, 30, 46
    scale = min((w - pad_l - pad_r) / spanx, (h - pad_t - pad_b) / spany)
    ox = pad_l + ((w - pad_l - pad_r) - spanx * scale) / 2
    oy = pad_t + ((h - pad_t - pad_b) - spany * scale) / 2

    def sx(x):
        return ox + (x - xmin) * scale

    def sy(y):
        return oy + (ymax - y) * scale

    o = [f'<svg viewBox="0 0 {w} {h}" class="map" role="img" '
         f'aria-label="选址示意图"><rect width="{w}" height="{h}" fill="#fbfcfd"/>']

    # 网格（每 250 m 一条线）
    step = 250.0
    left, right = pad_l, w - pad_r
    top, bottom = pad_t, h - pad_b
    g = math.floor(xmin / step) * step
    while g <= xmax:
        o.append(f'<line x1="{sx(g):.1f}" y1="{top}" x2="{sx(g):.1f}" y2="{bottom}" '
                 f'stroke="{C_GRID}"/>')
        g += step
    g = math.floor(ymin / step) * step
    while g <= ymax:
        o.append(f'<line x1="{left}" y1="{sy(g):.1f}" x2="{right}" y2="{sy(g):.1f}" '
                 f'stroke="{C_GRID}"/>')
        g += step

    # 刻度数字：以模糊中心为原点，等间距标注（与网格线解耦，保证一定落在画布内）
    label_step = 250.0 if max(spanx, spany) < 1600 else 500.0
    k = math.ceil((xmin - cx) / label_step)
    while cx + k * label_step <= xmax:
        if k != 0:
            X = sx(cx + k * label_step)
            o.append(f'<text x="{X:.1f}" y="{bottom + 12}" font-size="9" '
                     f'text-anchor="middle" fill="{C_INK}">{k * label_step:+.0f}</text>')
        k += 1
    k = math.ceil((ymin - cy) / label_step)
    while cy + k * label_step <= ymax:
        if k != 0:
            Y = sy(cy + k * label_step)
            o.append(f'<text x="{left - 6}" y="{Y + 3:.1f}" font-size="9" '
                     f'text-anchor="end" fill="{C_INK}">{k * label_step:+.0f}</text>')
        k += 1
    o.append(f'<text x="{right}" y="{bottom + 24}" font-size="9" text-anchor="end" '
             f'fill="{C_INK}">相对模糊中心 / 米</text>')

    # 同心环（可读的确定性刻度）
    for frac in (0.33, 0.66):
        o.append(f'<circle cx="{sx(cx):.1f}" cy="{sy(cy):.1f}" '
                 f'r="{r90 * frac * scale:.1f}" fill="none" stroke="#f0f3f7"/>')

    # 模糊区
    o.append(f'<circle cx="{sx(cx):.1f}" cy="{sy(cy):.1f}" r="{r90 * scale:.1f}" '
             f'fill="none" stroke="{C_CENTER}" stroke-width="1.3" '
             f'stroke-dasharray="5 4" opacity=".78"/>')
    o.append(f'<circle cx="{sx(cx):.1f}" cy="{sy(cy):.1f}" r="{r68 * scale:.1f}" '
             f'fill="{C_CENTER}" fill-opacity=".05" stroke="{C_CENTER}" '
             f'stroke-width="1.5"/>')

    # 需求点
    qmax = max([p.get("qty_daily", 1) for p in dp] + [1])
    for p in dp:
        x, y = xy(p)
        r = 5 + 11 * math.sqrt(max(p.get("qty_daily", 1), 0) / qmax)
        o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="{r:.1f}" '
                 f'fill="{C_DEMAND}" fill-opacity=".32" stroke="{C_DEMAND}" '
                 f'stroke-width="1"><title>{html.escape(p["id"])}：'
                 f'{_n(p.get("qty_daily"))} 件/日</title></circle>')

    # 候选场地（M2 选中的那个加环）
    sel = ((result.get("per_method") or {}).get("M2_discrete", {})
           .get("detail") or {}).get("selected_candidate")
    placed: list[tuple[float, float]] = []
    for idx, p in enumerate(cd, start=1):
        x, y = xy(p)
        X, Y = sx(x), sy(y)
        if p["id"] == sel:
            o.append(f'<circle cx="{X:.1f}" cy="{Y - 1:.1f}" r="11" fill="none" '
                     f'stroke="{C_SELECTED}" stroke-width="1.3" opacity=".55"/>')
        o.append(f'<polygon points="{X:.1f},{Y - 6:.1f} {X + 6:.1f},{Y + 4:.1f} '
                 f'{X - 6:.1f},{Y + 4:.1f}" fill="{C_CAND}" stroke="#7a5200" '
                 f'stroke-width=".6"><title>{html.escape(p["id"])}'
                 f'（{_n(p.get("area_sqm"))} m² / {_n(p.get("rent_month"))} 元月）'
                 f'</title></polygon>')
        lx, ly = X + 9, Y + 3
        if all(abs(lx - a) > 26 or abs(ly - b) > 11 for a, b in placed):
            placed.append((lx, ly))
            o.append(f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="9.5" '
                     f'fill="#8a6412" stroke="#fbfcfd" stroke-width="2.4" '
                     f'paint-order="stroke">C{idx}</text>')

    # 次峰
    for a in fc.get("alt_centers") or []:
        X = sx((a["lng"] - lng0) * 111320 * clat)
        Y = sy((a["lat"] - lat0) * 110540)
        o.append(f'<polygon points="{X:.1f},{Y - 8:.1f} {X + 2.5:.1f},{Y - 2:.1f} '
                 f'{X + 7:.1f},{Y - 1.5:.1f} {X + 3.5:.1f},{Y + 3:.1f} '
                 f'{X + 4.5:.1f},{Y + 8:.1f} {X:.1f},{Y + 5:.1f} '
                 f'{X - 4.5:.1f},{Y + 8:.1f} {X - 3.5:.1f},{Y + 3:.1f} '
                 f'{X - 7:.1f},{Y - 1.5:.1f} {X - 2.5:.1f},{Y - 2:.1f}" '
                 f'fill="none" stroke="{C_ALT}" stroke-width="1.4"><title>次峰</title>'
                 f'</polygon>')

    # 模糊中心
    X, Y = sx(cx), sy(cy)
    star = " ".join(f"{X + 8.5 * math.cos(math.pi / 2 + k * math.pi / 5):.1f},"
                    f"{Y - 8.5 * math.sin(math.pi / 2 + k * math.pi / 5):.1f}"
                    for k in range(10))
    o.append(f'<polygon points="{star}" fill="{C_CENTER}" stroke="#fff" '
             f'stroke-width="1.1"><title>模糊选址中心 μ={fc["mu_peak"]}</title></polygon>')

    # 指北针
    nx, ny = w - 30, 24
    o.append(f'<polygon points="{nx},{ny - 12} {nx + 5},{ny} {nx},{ny - 4} '
             f'{nx - 5},{ny}" fill="{C_INK}"/>'
             f'<text x="{nx}" y="{ny + 11}" font-size="9" text-anchor="middle" '
             f'fill="{C_INK}">N</text>')

    # 图例（两行，底部整条浅底）
    o.append(f'<rect x="0" y="{h - 44}" width="{w}" height="44" fill="#fbfcfd" '
             f'opacity=".96"/>')
    y1, y2 = h - 40, h - 16
    o.append(f'<circle cx="14" cy="{y1 - 3}" r="4.5" fill="{C_DEMAND}" '
             f'fill-opacity=".4" stroke="{C_DEMAND}"/>'
             f'<text x="23" y="{y1}" font-size="11" fill="{C_INK}">需求点（面积∝件量）</text>')
    o.append(f'<polygon points="166,{y1 - 7} 172,{y1 + 1} 160,{y1 + 1}" fill="{C_CAND}"/>'
             f'<text x="177" y="{y1}" font-size="11" fill="{C_INK}">候选场地</text>')
    o.append(f'<circle cx="256" cy="{y1 - 3}" r="4.5" fill="none" stroke="{C_CENTER}"/>'
             f'<text x="265" y="{y1}" font-size="11" fill="{C_INK}">68% / 90% 模糊区</text>')
    o.append(f'<text x="380" y="{y1}" font-size="12" fill="{C_CENTER}">★</text>'
             f'<text x="392" y="{y1}" font-size="11" fill="{C_INK}">模糊中心</text>'
             f'<text x="462" y="{y1}" font-size="12" fill="{C_ALT}">✳</text>'
             f'<text x="474" y="{y1}" font-size="11" fill="{C_INK}">次峰</text>')

    # 比例尺（带刻度，整条落在画布内）
    bar = 250 * scale
    bx = 14
    o.append(f'<line x1="{bx}" y1="{y2}" x2="{bx + bar:.1f}" y2="{y2}" '
             f'stroke="{C_INK}" stroke-width="1.4"/>')
    for t in (0, 0.5, 1):
        X = bx + bar * t
        o.append(f'<line x1="{X:.1f}" y1="{y2 - 3}" x2="{X:.1f}" y2="{y2 + 2}" '
                 f'stroke="{C_INK}" stroke-width="1.2"/>')
    o.append(f'<text x="{bx + bar + 6:.1f}" y="{y2 + 4}" font-size="10" '
             f'fill="{C_INK}">250 m</text>')

    o.append("</svg>")
    return "".join(o)
