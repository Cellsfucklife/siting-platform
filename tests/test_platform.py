"""冒烟与一致性测试：pytest -q

覆盖：
1. 契约校验
2. 四档数据画像 → 方法自动选择
3. 每个方法「自研实现 vs 参考实现」一致性
4. 方法间一致性、容量产出完整性
5. 单调性
"""
from __future__ import annotations

import pytest

from fixtures import build
from siting.contract import validate
from siting.solver import solve


@pytest.mark.parametrize("level,min_methods", [(3, 4), (2, 3), (1, 2), (0, 1)])
def test_profile_and_selection(level, min_methods):
    res = solve(build(level))
    assert res["ok"], res
    assert len(res["method_selection"]["used"]) >= min_methods


def test_contract_rejects_empty():
    r = validate({"area_id": "X"})
    assert not r["ok"]


def test_center_and_capacity_present():
    for lvl in (3, 2, 1, 0):
        res = solve(build(lvl))
        fc, cap = res["fuzzy_center"], res["capacity"]
        assert -180 <= fc["lng"] <= 180 and -90 <= fc["lat"] <= 90
        assert fc["radius_68_m"] <= fc["radius_90_m"]
        assert cap["tier"] and cap["staff"] >= 1
        assert cap["daily_demand_ci90"][0] <= cap["daily_demand_ci90"][1]


def test_implementation_agreement():
    res = solve(build(3))
    impl = res["cross_validation"]["implementations"]
    # M1 Weiszfeld vs SciPy
    assert impl["M1_gravity"]["consistent"] is True
    # M2 参考实现应落在自研 ε-最优集合的前三名
    assert impl["M2_discrete"]["consistent"] is True
    # M3 手写核 vs sklearn
    assert impl["M3_kde"]["consistent"] is True


def test_reference_implementations_actually_run():
    """防止参考实现静默失败（例如 CBC 路径不可用）却被判成一致。"""
    res = solve(build(3))
    m2 = res["per_method"]["M2_discrete"]["backends"]
    assert m2["reference"]["pmedian"] is not None
    assert m2["reference"]["pmedian"]["index"] is not None
    assert m2["reference"]["cflp"]["status"] == "Optimal"
    assert "error" not in m2["agreement"]
    assert m2["agreement"]["reference_ran"] is True


def test_io_adapter_examples(tmp_path_factory):
    from pathlib import Path
    from siting.io_adapter import load_scope

    root = Path(__file__).resolve().parents[1] / "examples"
    payload = load_scope(root)
    assert payload["_meta"]["n_demand"] == 8
    assert payload["_meta"]["has_dist_matrix"] is True
    assert len(payload["criteria"]["keys"]) >= 3
    res = solve(payload)
    assert res["ok"]
    assert res["verdict"]["level"] in ("high", "medium")
    assert res["capacity"]["tier"]


def test_capacity_monotone():
    res = solve(build(3))
    mono = res["cross_validation"]["monotonicity"]
    assert mono["demand_monotone"] is True
    assert mono["capacity_monotone"] is True


def test_low_data_degrades():
    res = solve(build(1))
    assert res["method_selection"]["mode"] in ("multi", "single")
    assert res["cross_validation"]["methods"]["verdict"] in ("high", "medium", "low")
    assert any(r["field"] for r in res["recommendations"])
