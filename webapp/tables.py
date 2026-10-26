"""Excel / CSV 表格读取：把上传的文件变成「行 × 列」的文本矩阵。

设计约定
--------
· 支持 .xlsx / .xlsm / .csv（.xls 老格式需要 xlrd，不提供）。
· 表头可有可无：自动识别，并支持中英文列名（小区名 / 纬度 / 经度 / 件量 ……）。
· 多工作表时优先挑名字里带关键字的表（小区、场地、距离），否则取第一个非空表。
· 输出统一为 list[list[str]]，交给 forms.py 的同一套解析器，保证与粘贴文本行为一致。
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

MAX_BYTES = 5 * 1024 * 1024          # 单文件 5 MB 上限

# 列名别名：用于表头识别与列重排
ALIASES: dict[str, tuple[str, ...]] = {
    "name": ("小区", "小区名", "小区名称", "名称", "场地", "场地名", "站点", "名字",
             "id", "编号", "name", "site", "site_id", "d_id", "c_id"),
    "lat": ("纬度", "lat", "latitude", "北纬"),
    "lng": ("经度", "lng", "lon", "long", "longitude", "东经"),
    "x": ("x", "横坐标", "东向", "东西向偏移"),
    "y": ("y", "纵坐标", "北向", "南北向偏移"),
    "qty": ("件量", "日件量", "日均件量", "日均件数", "件数", "单量", "qty",
            "qty_daily", "daily"),
    "area": ("面积", "面积m2", "面积m²", "面积(平米)", "area", "area_sqm"),
    "rent": ("租金", "月租金", "租金元", "rent", "rent_month"),
    "minutes": ("分钟", "时间", "时长", "行驶时间", "minutes", "min", "time"),
    "date": ("日期", "date", "day", "年月", "月份", "统计日期", "业务日期"),
}

CRIT_ALIASES: dict[str, tuple[str, ...]] = {
    "accessibility": ("可达性", "可达", "accessibility"),
    "competition": ("竞争", "竞争度", "competition"),
    "site_availability": ("可租性", "可租", "场地可得", "site_availability"),
    "expandability": ("可扩展", "可扩展性", "expandability"),
    "policy": ("政策", "policy"),
}

# 距离表的两列都叫“名称”，必须分方向识别
MATRIX_ALIASES: dict[str, tuple[str, ...]] = {
    "d": ("小区", "小区名", "小区名称", "需求点", "起点", "from", "d_id", "origin"),
    "c": ("场地", "场地名", "场地名称", "候选点", "终点", "to", "c_id", "dest"),
    "m": ("分钟", "时间", "时长", "行驶时间", "minutes", "min", "time"),
}


class TableError(ValueError):
    """读表失败（格式不支持、文件过大、空表）。"""


def _norm(s: object) -> str:
    return str(s).strip().lower().replace(" ", "").replace("_", "")


def _alias_of(col: str) -> str | None:
    key = _norm(col)
    for canonical, names in ALIASES.items():
        if key in {_norm(n) for n in names}:
            return canonical
    for canonical, names in CRIT_ALIASES.items():
        if key in {_norm(n) for n in names}:
            return canonical
    return None


def _is_number(v: object) -> bool:
    try:
        float(str(v).strip())
        return True
    except (TypeError, ValueError):
        return False


def _clean_cell(v: object) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):                       # type: ignore[arg-type]
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


def _frame_to_rows(df: pd.DataFrame) -> list[list[str]]:
    rows = [[_clean_cell(v) for v in rec] for rec in df.itertuples(index=False, name=None)]
    rows = [r for r in rows if any(c != "" for c in r)]
    return rows


def read_rows(data: bytes, filename: str, hint: str | None = None) -> list[list[str]]:
    """把文件读成行列表；hint 用于多工作表时挑表。"""
    if len(data) > MAX_BYTES:
        raise TableError(f"文件超过 {MAX_BYTES // 1024 // 1024} MB，请先精简后再传")
    if not data:
        raise TableError("文件是空的")

    suffix = Path(filename or "").suffix.lower()
    if suffix == ".csv":
        for enc in ("utf-8-sig", "utf-8", "gbk"):
            try:
                df = pd.read_csv(io.BytesIO(data), header=None, dtype=str, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise TableError("CSV 编码无法识别，请另存为 UTF-8 或上传 .xlsx")
        return _frame_to_rows(df)

    if suffix not in (".xlsx", ".xlsm"):
        raise TableError(f"不支持的文件类型 {suffix or '(无扩展名)'}，"
                         f"请上传 .xlsx 或 .csv（老式 .xls 请另存为 .xlsx）")
    try:
        book = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None, dtype=str)
    except Exception as exc:                 # pragma: no cover - 依赖外部库报错
        raise TableError(f"读取 Excel 失败：{exc}") from exc

    if not book:
        raise TableError("Excel 里没有工作表")

    sheet = None
    if hint:
        sheet = next((name for name in book if hint in str(name)), None)
    if sheet is None:
        sheet = next((name for name, df in book.items()
                      if df.dropna(how="all").shape[0] > 0), next(iter(book)))

    rows = _frame_to_rows(book[sheet])
    if not rows:
        raise TableError(f"工作表「{sheet}」是空的")
    return rows


def _looks_like_header(row: list[str]) -> bool:
    """第一行只要后几列不是数字，且首列不是数字，就当表头。"""
    if not row or _is_number(row[0]):
        return False
    body = [c for c in row[1:5] if c != ""]
    return bool(body) and not any(_is_number(c) for c in body)


def _reorder(header: list[str], rows: list[list[str]],
             canonical: tuple[str, ...]) -> list[list[str]]:
    """按表头别名把列重排成 canonical 顺序；认不出就原样返回。"""
    mapping = {i: _alias_of(h) for i, h in enumerate(header)}
    known = {v: i for i, v in mapping.items() if v}
    if len(known) < 2:
        return rows
    order = [known[k] for k in canonical if k in known]
    rest = [i for i in range(len(header)) if i not in order]
    width = max((len(r) for r in rows), default=0)
    out = []
    for r in rows:
        r = list(r) + [""] * (width - len(r))
        out.append([r[i] for i in order] + [r[i] for i in rest if i < len(r)])
    return out


def _reorder_matrix(header: list[str], rows: list[list[str]]) -> list[list[str]]:
    """距离表：按 (小区名, 场地名, 分钟) 重排；认不全就原样返回。"""
    found: dict[str, int] = {}
    for i, h in enumerate(header):
        key = _norm(h)
        for k, names in MATRIX_ALIASES.items():
            if k not in found and key in {_norm(n) for n in names}:
                found[k] = i
                break
    if len(found) < 3:
        return rows
    order = [found["d"], found["c"], found["m"]]
    width = max((len(r) for r in rows), default=0)
    return [[(list(r) + [""] * width)[i] for i in order] for r in rows]


def to_table(data: bytes, filename: str, kind: str, hint: str | None = None) -> list[list[str]]:
    """kind ∈ {demand, candidates, matrix}：读表 + 去表头 + 列重排。"""
    rows = read_rows(data, filename, hint)
    if rows and _looks_like_header(rows[0]):
        header, body = rows[0], rows[1:]
        if kind == "matrix":
            return _reorder_matrix(header, body)
        canonical = {"demand": ("name", "lat", "lng", "qty"),
                     "candidates": ("name", "lat", "lng", "area", "rent"),
                     "series": ("name", "date", "qty")}[kind]
        return _reorder(header, body, canonical)
    return rows


# ------------------------------------------------------------------ 模板
TEMPLATE_SHEETS: dict[str, tuple[list[str], list[list[object]]]] = {
    "①小区": (
        ["小区名", "纬度", "经度", "日件量"],
        [["望京西园三区", 39.98862, 116.47139, 286],
         ["花家地北里", 39.98604, 116.47612, 214],
         ["南湖东园", 39.99127, 116.46807, 338]],
    ),
    "②场地": (
        ["场地名", "纬度", "经度", "面积m²", "月租金", "可达性", "竞争", "可租性", "可扩展"],
        [["望京西园底商", 39.98791, 116.47358, 85, 26000, 0.78, 0.62, 0.55, 0.40],
         ["花家地社区用房", 39.99033, 116.47724, 320, 18500, 0.66, 0.35, 0.80, 0.72]],
    ),
    "③距离": (
        ["小区名", "场地名", "分钟"],
        [["望京西园三区", "望京西园底商", 4.6],
         ["望京西园三区", "花家地社区用房", 3.1]],
    ),
    "④流水": (
        ["小区名", "日期", "件量"],
        [["望京西园三区", "2025-01-03", 291],
         ["望京西园三区", "2025-01-04", 305],
         ["望京西园三区", "2025-01-05", 268],
         ["花家地北里", "2025-01-03", 208],
         ["花家地北里", "2025-01-04", 221]],
    ),
}

TEMPLATE_NOTES = [
    ["填写说明"],
    ["1. 只有「①小区」这张表是必填：小区名 + 纬度 + 经度 + 日件量，一行一个小区。"],
    ["2. 「②场地」「③距离」「④流水」都可以留空，留空就用默认规则。"],
    ["3. 纬度是南北位置（北京约 39.9），经度是东西位置（北京约 116.4）。"],
    ["4. 没有经纬度时，可以把两张表的两列改成相对米制（x=往东米数，y=往北米数），"],
    ["   并在网页上把「位置怎么填」选成「相对米制」。"],
    ["5. 表头行可以保留，也可以删掉；列名支持中英文（小区名/id、纬度/lat、件量/qty）。"],
    ["6. ③距离表要把「每个小区 × 每个场地」的组合都列全，否则会退回直线估算。"],
    ["7. ④流水是逐日/逐周/逐月的原始记录，日期支持 2025-01-03、2025/1/3、2025年1月、2025；"],
    ["   平台会先拟合出年增长与周内系数，再把件量换成设计年日均。"],
    ["8. 单个文件不超过 5 MB。"],
]


def build_template() -> bytes:
    """生成填写模板（3 张数据表 + 1 张说明）。"""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, (head, rows) in TEMPLATE_SHEETS.items():
            pd.DataFrame(rows, columns=head).to_excel(writer, sheet_name=name, index=False)
        pd.DataFrame(TEMPLATE_NOTES).to_excel(writer, sheet_name="说明", index=False,
                                             header=False)
    return buf.getvalue()
