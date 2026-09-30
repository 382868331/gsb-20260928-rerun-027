"""两个分支在相同时刻的位置、碰撞与触发原因对比，以及首处差异定位。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .engine import CollisionRecord, ReplayEngine, Vec2


@dataclass(frozen=True)
class CollisionSnapshot:
    pairs: Tuple[Tuple[str, str], ...]
    reasons: Dict[Tuple[str, str], str]


@dataclass(frozen=True)
class BranchComparison:
    tick_ms: int
    positions_match: bool
    collisions_match: bool
    trigger_reasons_match: bool
    positions_a: Dict[str, Vec2]
    positions_b: Dict[str, Vec2]
    collisions_a: CollisionSnapshot
    collisions_b: CollisionSnapshot
    reasons_a: Dict[str, Optional[str]]
    reasons_b: Dict[str, Optional[str]]

    @property
    def equal(self) -> bool:
        return (
            self.positions_match
            and self.collisions_match
            and self.trigger_reasons_match
        )


def compare_at(engine_a: ReplayEngine, engine_b: ReplayEngine, tick_ms: int) -> BranchComparison:
    engine_a.seek(tick_ms)
    engine_b.seek(tick_ms)
    positions_a = engine_a.positions()
    positions_b = engine_b.positions()
    collisions_a = _collision_snapshot(engine_a, tick_ms)
    collisions_b = _collision_snapshot(engine_b, tick_ms)
    reasons_a = _reasons_until(engine_a, tick_ms)
    reasons_b = _reasons_until(engine_b, tick_ms)
    return BranchComparison(
        tick_ms=tick_ms,
        positions_match=positions_a == positions_b,
        collisions_match=(collisions_a.pairs, collisions_a.reasons)
        == (collisions_b.pairs, collisions_b.reasons),
        trigger_reasons_match=reasons_a == reasons_b,
        positions_a=positions_a,
        positions_b=positions_b,
        collisions_a=collisions_a,
        collisions_b=collisions_b,
        reasons_a=reasons_a,
        reasons_b=reasons_b,
    )


@dataclass(frozen=True)
class FirstDivergence:
    """两个分支第一处行为差异：发生的时刻与差异的方面。"""

    tick_ms: int
    aspects: Tuple[str, ...]  # positions / signals / collisions / events 的子集


def first_divergence(
    engine_a: ReplayEngine,
    engine_b: ReplayEngine,
    until_ms: Optional[int] = None,
    from_ms: int = 0,
) -> Optional[FirstDivergence]:
    """逐 tick 对比两个引擎，返回第一处行为差异；完全一致时返回 None。

    从 from_ms 到 until_ms（缺省为两引擎当前时刻的较大者）按 tick 步进，
    每个时刻对齐比较位置、信号相位、碰撞集合与事件触发原因。
    """
    step = engine_a.scenario.tick_ms
    if until_ms is None:
        until_ms = max(engine_a.now_ms, engine_b.now_ms)
    tick = from_ms
    while tick <= until_ms:
        engine_a.seek(tick)
        engine_b.seek(tick)
        aspects = []
        if engine_a.positions() != engine_b.positions():
            aspects.append("positions")
        if engine_a.signal_phases() != engine_b.signal_phases():
            aspects.append("signals")
        if _collision_snapshot(engine_a, tick) != _collision_snapshot(engine_b, tick):
            aspects.append("collisions")
        if _reasons_until(engine_a, tick) != _reasons_until(engine_b, tick):
            aspects.append("events")
        if aspects:
            return FirstDivergence(tick_ms=tick, aspects=tuple(aspects))
        tick += step
    return None


def _collision_snapshot(engine: ReplayEngine, tick_ms: int) -> CollisionSnapshot:
    records: Tuple[CollisionRecord, ...] = tuple(
        record for record in engine.collisions if record.tick_ms <= tick_ms
    )
    return CollisionSnapshot(
        pairs=tuple(record.pair for record in records),
        reasons={record.pair: record.reason for record in records},
    )


def _reasons_until(engine: ReplayEngine, tick_ms: int) -> Dict[str, Optional[str]]:
    reasons: Dict[str, Optional[str]] = {}
    for record in engine.journal:
        if record.tick_ms <= tick_ms:
            reasons[record.event_id] = record.reason
    return reasons
