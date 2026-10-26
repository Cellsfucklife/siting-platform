"""把「粘贴的文本 / 上传的 Excel」统一解析成求解 payload。

两种输入走同一套行解析器，所以文案、报错、默认值完全一致；
若同一张表既粘了文本又传了文件，以上传文件为准并给出提示。
"""
from __future__ import annotations

import math

from . import tables

M_PER_DEG_LAT = 110540.0
M_PER_DEG_LNG = 111320.0
LOCAL_ORIGIN = (39.9000, 116.4000)      # 相对坐标的伪原点，不影响距离与方位
CRIT_KEYS_EXTRA = ("accessibility", "competition", "site_availability",
                   "expandability", "policy")

# 四张表的展示名与 Excel 工作表提示词
TABLE_LABEL = {"demand": "小区表", "candidates": "场地表",
               "matrix": "距离表", "series": "流水表"}
TABLE_HINT = {"demand": "小区", "candidates": "场地",
              "matrix": "距离", "series": "流水"}


# ---------------------------------------------------------------- 行解析
def split_rows(text: str | None) -> list[list[str]]:
    """文本 → 行列表。逗号 / 制表符 / 空格分隔，空行与 # 注释忽略。"""
    if not text:
        return []
    out = []
    for line in text.replace("\t", ",").splitlines():
        line = line.strip().strip(",")
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")] if "," in line else line.split()
        parts = [p for p in parts if p != ""]
        if parts:
            out.append(parts)
    return out


def _num(v: str) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_latlon(a: float, b: float, frame: str) -> tuple[float, float]:
    if frame == "local":
        clat = math.cos(math.radians(LOCAL_ORIGIN[0]))
        return (LOCAL_ORIGIN[0] + b / M_PER_DEG_LAT,
                LOCAL_ORIGIN[1] + a / (M_PER_DEG_LNG * clat))
    return a, b


def parse_demand_rows(rows: list[list[str]], frame: str) -> tuple[list[dict], list[str]]:
    pts, errs = [], []
    for i, r in enumerate(rows, start=1):
        if len(r) < 4:
            errs.append(f"小区第 {i} 行「{r[0]}」只有 {len(r)} 列，还缺"
                        f"「{'件量' if len(r) == 3 else '纬度/经度'}」。"
                        f"正确写法：小区名, 纬度, 经度, 日件量")
            continue
        a, b, q = _num(r[1]), _num(r[2]), _num(r[3])
        if a is None or b is None or q is None:
            errs.append(f"小区第 {i} 行「{r[0]}」里有非数字：纬度、经度、日件量都要是数字"
                        f"（例：望京西园三区, 39.98862, 116.47139, 286）")
            continue
        lat, lng = _to_latlon(a, b, frame)
        pts.append({"id": r[0], "lat": lat, "lng": lng, "qty_daily": q})
    return pts, errs


def parse_candidate_rows(rows: list[list[str]], frame: str) -> tuple[list[dict], list[str]]:
    pts, errs = [], []
    for i, r in enumerate(rows, start=1):
        if len(r) < 3:
            errs.append(f"场地第 {i} 行「{r[0]}」只有 {len(r)} 列，至少要"
                        f"「场地名, 纬度, 经度」。例：C-01, 39.98791, 116.47358, 85, 26000")
            continue
        a, b = _num(r[1]), _num(r[2])
        if a is None or b is None:
            errs.append(f"场地第 {i} 行「{r[0]}」的纬度/经度不是数字")
            continue
        lat, lng = _to_latlon(a, b, frame)
        c = {"id": r[0], "lat": lat, "lng": lng}
        if len(r) >= 4 and _num(r[3]) is not None:
            c["area_sqm"] = _num(r[3])
        if len(r) >= 5 and _num(r[4]) is not None:
            c["rent_month"] = _num(r[4])
        for idx, key in enumerate(CRIT_KEYS_EXTRA, start=5):
            if len(r) > idx and _num(r[idx]) is not None:
                c[key] = _num(r[idx])
        pts.append(c)
    return pts, errs


def parse_matrix_rows(rows: list[list[str]]) -> tuple[list[tuple[str, str, float]], list[str]]:
    out, errs = [], []
    for i, r in enumerate(rows, start=1):
        if len(r) < 3:
            errs.append(f"距离表第 {i} 行只有 {len(r)} 列，需要「小区名, 场地名, 分钟」")
            continue
        m = _num(r[2])
        if m is None:
            errs.append(f"距离表第 {i} 行第三列「{r[2]}」不是数字（应为分钟数）")
            continue
        out.append((r[0], r[1], m))
    return out, errs


def parse_series_rows(rows: list[list[str]]) -> tuple[list[dict], list[str]]:
    """流水：小区, 日期, 件量。日期支持 2025-01-03 / 2025/1/3 / 2025年1月 / 2025。"""
    from siting.series import parse_period

    out, errs = [], []
    for i, r in enumerate(rows, start=1):
        if len(r) < 3:
            errs.append(f"流水表第 {i} 行只有 {len(r)} 列，需要「小区名, 日期, 件量」")
            continue
        q = _num(r[2])
        if q is None:
            errs.append(f"流水表第 {i} 行「{r[0]}」的件量不是数字")
            continue
        if parse_period(r[1]) is None:
            errs.append(f"流水表第 {i} 行日期「{r[1]}」认不出，"
                        f"支持 2025-01-03、2025/1/3、2025年1月、2025 这几种写法")
            continue
        out.append({"id": r[0], "date": r[1], "qty": q})
    return out, errs


# 兼容旧接口（文本版）
def parse_demand(text: str, frame: str):
    return parse_demand_rows(split_rows(text), frame)


def parse_candidates(text: str, frame: str):
    return parse_candidate_rows(split_rows(text), frame)


def parse_matrix(text: str):
    return parse_matrix_rows(split_rows(text))


# ---------------------------------------------------------------- 取表
def _rows_of(form: dict, uploads: dict | None, key: str, text_key: str,
             warnings: list[str], mode: str) -> tuple[list[list[str]], str | None]:
    """按当前输入方式取表：excel 只认文件，text 只认文本，未指定时文件优先。"""
    up = (uploads or {}).get(key) or {}
    text = (form.get(text_key) or "").strip()

    if mode == "text" and up.get("data"):
        warnings.append(f"{TABLE_LABEL[key]}：当前是「粘贴文本」方式，"
                        f"已忽略上传的「{up.get('filename') or '文件'}」。")
        up = {}
    elif mode == "excel" and text:
        warnings.append(f"{TABLE_LABEL[key]}：当前是「上传 Excel」方式，已忽略粘贴的文本。")
        text = ""
    elif mode == "auto" and up.get("data") and text:
        warnings.append(f"{TABLE_LABEL[key]}：已使用上传的「{up.get('filename') or '文件'}」，"
                        f"忽略了同时粘贴的文本。")

    if up.get("data"):
        name = up.get("filename") or "上传文件"
        try:
            rows = tables.to_table(up["data"], name, key, TABLE_HINT[key])
        except tables.TableError as exc:
            return [], f"{TABLE_LABEL[key]}：{exc}"
        if rows:
            return rows, None
        return [], f"{TABLE_LABEL[key]}：「{name}」里没有读到数据行"

    return split_rows(text), None


def _float(form: dict, key: str, default: float) -> float:
    v = _num(str(form.get(key, "")).strip())
    return default if v is None else v


# ---------------------------------------------------------------- 组装
def build_payload(form: dict, uploads: dict | None = None
                  ) -> tuple[dict, list[str], list[str]]:
    frame = "local" if str(form.get("frame")) == "local" else "geo"
    mode = str(form.get("input_mode") or "").strip().lower()
    if mode not in ("text", "excel"):
        # 未指定方式时保持旧行为：文件优先于文本
        mode = "auto" if (uploads or {}).get("demand") or (uploads or {}).get("candidates") else ""
    errors: list[str] = []
    warnings: list[str] = []

    d_rows, e_d = _rows_of(form, uploads, "demand", "demand_points", warnings, mode)
    c_rows, e_c = _rows_of(form, uploads, "candidates", "candidates", warnings, mode)
    m_rows, e_m = _rows_of(form, uploads, "matrix", "dist_matrix", warnings, mode)
    s_rows, e_s = _rows_of(form, uploads, "series", "series_points", warnings, mode)
    errors += [e for e in (e_d, e_c, e_m, e_s) if e]

    demand, e1 = parse_demand_rows(d_rows, frame)
    cands, e2 = parse_candidate_rows(c_rows, frame)
    mat, e3 = parse_matrix_rows(m_rows)
    records, e4 = parse_series_rows(s_rows)
    parse_errors = e1 + e2 + e3 + e4
    errors += parse_errors

    # ① 空表：区分「没填」「填了但没解析出来」「选了 Excel 却没传文件」
    if not d_rows and not c_rows and not parse_errors:
        if mode == "excel":
            has_any = bool((uploads or {}).get("demand") or (uploads or {}).get("candidates")
                           or (uploads or {}).get("matrix") or (uploads or {}).get("series"))
            errors.append(
                "还没有选择「小区」表文件（这张是必填）。也可把上面⓪切回「粘贴文本」。"
                if has_any else
                "你选择了「上传 Excel / CSV」，但还没有选择文件。"
                "请上传文件，或把上面⓪切回「粘贴文本」。")
        else:
            typed = any((form.get(k) or "").strip()
                        for k in ("demand_points", "candidates", "dist_matrix", "series_points"))
            if typed or uploads:
                errors.append("表格里没有解析出任何一行 —— 请检查分隔符与列顺序，"
                              "或换用 Excel 上传。")
            else:
                errors.append("「小区」与「能租的场地」至少要填一个（可粘贴，也可上传 Excel）。")

    # ② 坐标体检（只在没有解析错误时做，避免掩盖更具体的问题）
    if frame == "geo" and not parse_errors:
        swapped, bad = 0, []
        for p in demand + cands:
            la, ln = p["lat"], p["lng"]
            if abs(la) > 90 and abs(ln) <= 90:
                p["lat"], p["lng"] = ln, la
                swapped += 1
            elif abs(la) > 90 or abs(ln) > 180:
                bad.append(p["id"])
        if swapped:
            warnings.append(f"有 {swapped} 行看起来「纬度/经度」填反了，已自动调换"
                            f"（纬度应在 −90~90，北京约 39.9；经度 −180~180，约 116.4）。")
        if bad:
            errors.append("这些点的坐标超出经纬度范围：" + "、".join(bad[:5])
                          + (" 等" if len(bad) > 5 else "")
                          + "。若你填的是米制相对坐标，请把 ① 的「位置怎么填」改成「相对米制」。")

    # ③ 样本量：少于 3 个小区且没有场地时，模型算不出稳定中心
    if not errors and 0 < len(demand) < 3 and not cands:
        errors.append(f"至少要 3 个小区才能算出稳定的选址中心（现在只有 {len(demand)} 个）。"
                      f"如果这个片区就只有这么几个小区，请把「能租的场地」也一起提供。")

    # ④ 原始流水推演：点集 → 线性方程 → 抽象指标（件量、增长、峰值、波动）
    series_info = None
    peak_default = 1.6
    payload_cv = 0.0
    if records and not parse_errors:
        from siting.series import derive
        s_freq = str(form.get("series_freq") or "auto")
        s_horizon = _num(str(form.get("series_horizon") or "").strip() or "") or 1.0
        series_info = derive(records, s_freq, s_horizon)
        if not series_info.get("ok"):
            warnings.append(f"流水推演失败：{series_info.get('error')}")
            series_info = None
        else:
            matched = 0
            for p in demand:
                node = series_info["nodes"].get(p["id"])
                if node:
                    p["qty_observed"] = node["qty_observed"]
                    p["qty_daily"] = node["qty_projected"]
                    matched += 1
            if matched:
                warnings.append(
                    f"件量已按流水推演替换：{matched} 个小区采用设计年日均"
                    f"（外推 {s_horizon:g} 年，年增长 {series_info['growth_per_year'] * 100:.1f}%）。")
            else:
                warnings.append("流水里的小区名与②「小区」表对不上，件量未替换；"
                                "请检查两边名称写法是否一致。")
            peak_default = series_info["peak_factor"]
            given_peak = _num(str(form.get("peak_factor", "")).strip() or "")
            week_peak = max(series_info.get("weekday_factor") or [1.0])
            if given_peak is not None:
                peak_used = given_peak
                if week_peak > peak_used + 1e-6:
                    peak_used = week_peak
                    warnings.append(f"你填的峰值系数 {given_peak} 低于流水里最忙的那天"
                                    f"（{week_peak:.2f} 倍），已按 {week_peak:.2f} 计。")
            elif peak_default < 1.6:
                # 观测峰值低于默认值时，说明窗口没覆盖大促，直接采用会低估峰值
                peak_used = 1.6
                warnings.append(f"流水观测峰值只有 {peak_default}，低于默认的 1.6，"
                                f"说明这段窗口可能没有覆盖大促；峰值系数仍取 1.6。"
                                f"建议补入大促期流水，或在⑥里手动指定。")
            else:
                peak_used = peak_default
                warnings.append(f"峰值系数取流水观测值 {peak_default}（未填时原默认 1.6）。")
            series_info["peak_factor_used"] = peak_used
            peak_default = peak_used
            payload_cv = float(series_info.get("cv") or 0.0)
            if payload_cv > 0.02:
                warnings.append(f"常规日波动 CV = {payload_cv:.3f}，"
                                f"容量按 {(1 + min(payload_cv, 0.4) * 0.75 - 1) * 100:+.0f}% "
                                f"的缓冲配置。")
            if demand and not cands:
                warnings.append("流水只用于标定，选址仍需要小区位置或候选场地。")

    # 有推演时 peak_default 已含"用户填的值 / 波动下限 / 观测值"的取舍；无推演时仍读表单
    _given_pf = _num(str(form.get("peak_factor", "")).strip() or "")
    peak_payload = peak_default if series_info else (
        _given_pf if _given_pf is not None else 1.6)

    payload: dict = {
        "area_id": (str(form.get("area_name") or "").strip() or "未命名片区"),
        "peak_factor": peak_payload,
        "volatility_cv": payload_cv,
        "cv_margin": _float(form, "cv_margin", 0.75),
        "service_radius_min": _float(form, "service_radius_min", 10.0),
        "self_pick_ratio": _float(form, "self_pick_ratio", 0.7),
        "service_time_s": _float(form, "service_time_s", 45.0),
        "queue_wait_target_s": _float(form, "queue_wait_target_s", 180.0),
        "vehicle_daily_capacity": _float(form, "vehicle_daily_capacity", 160.0),
        "access_speed_kmh": _float(form, "access_speed_kmh", 12.0),
    }
    if demand:
        payload["demand_points"] = demand
    if cands:
        payload["candidates"] = cands

    proxy = _num(str(form.get("proxy_daily_demand", "")).strip() or "")
    if proxy:
        payload["proxy_daily_demand"] = proxy
    elif cands and not demand:
        warnings.append("只有场地、没有件量：容量算不出来，请在参数里填「代理日件量」。")

    if mat and demand and cands:
        didx = {d["id"]: i for i, d in enumerate(demand)}
        cidx = {c["id"]: j for j, c in enumerate(cands)}
        M: list[list[float | None]] = [[None] * len(cands) for _ in demand]
        miss = 0
        for d_id, c_id, m in mat:
            i, j = didx.get(d_id), cidx.get(c_id)
            if i is None or j is None:
                miss += 1
                continue
            M[i][j] = m
        if all(v is not None for row in M for v in row):
            payload["dist_matrix"] = M
        else:
            fill = sum(1 for row in M for v in row if v is None)
            warnings.append(f"距离表不完整（缺 {fill} 个组合），已退回直线×绕行系数计算。"
                            f"未匹配到名称的有 {miss} 行，请检查是否与上面两张表写法一致。")
    elif mat:
        warnings.append("距离表需要同时存在小区与场地才生效，已忽略。")

    if frame == "local":
        payload["_local_origin"] = {"lat": LOCAL_ORIGIN[0], "lng": LOCAL_ORIGIN[1]}
    if series_info:
        payload["_series"] = series_info

    return payload, errors, warnings
