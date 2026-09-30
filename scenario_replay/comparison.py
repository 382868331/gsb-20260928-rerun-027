"""两个分支在相同时刻的位置、碰撞与触发原因对比。"""
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
