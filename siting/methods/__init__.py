"""方法注册表。"""
from __future__ import annotations

from . import m1_gravity, m2_discrete, m3_kde, m4_mcdm

ALL = [m1_gravity.SPEC, m2_discrete.SPEC, m3_kde.SPEC, m4_mcdm.SPEC]
BY_KEY = {s.key: s for s in ALL}
RUNNERS = {s.key: s.run for s in ALL}

__all__ = ["ALL", "BY_KEY", "RUNNERS"]
